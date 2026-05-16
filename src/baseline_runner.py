"""Phase 1 baseline model runner: ridge + random forest + PROPHET-Ab ridge.

Design
------
Given a HarmonizedBatch (from src.ab_benchmark_loader), this module runs
each baseline model under leakage-resistant GroupKFold, collects per-fold
predictions, and reports:

    - Spearman rho of predictions vs targets (bootstrap 95% CI, seed fixed)
    - RMSE (bootstrap 95% CI)
    - R²   (bootstrap 95% CI)

The predictions concatenated across folds form an out-of-fold ("OOF")
prediction vector, which is the right quantity to compare against the
measured endpoint under grouped CV. Models are NOT retrained on the full
dataset — this module is strictly for evaluation.

"No ΔΔG claims" and "no state-of-the-art claims unless supported by
grouped CV + CIs" are enforced by construction: every reported metric
has a CI attached.

Public API
----------
    evaluate_ridge(batch, ...)
    evaluate_random_forest(batch, ...)
    evaluate_prophet_ab(batch, ...)     # same as ridge, named differently for clarity
    run_all_baselines(batch, ...)       # convenience, returns list of results
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Bootstrap CIs — self-contained to avoid importing from ab_benchmark.
# ---------------------------------------------------------------------------


@dataclass
class MetricCI:
    point: float
    low: float
    high: float
    n: int
    method: str = "bootstrap"


def _bootstrap_ci(
    stat_fn,
    n: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = stat_fn(idx)
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def _spearman_ci(y_true: np.ndarray, y_pred: np.ndarray, seed: int = 0) -> MetricCI:
    from scipy.stats import spearmanr

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 4:
        return MetricCI(float("nan"), float("nan"), float("nan"), n=n)
    rho_hat = float(spearmanr(y_true, y_pred).statistic)
    low, high = _bootstrap_ci(
        lambda idx: float(spearmanr(y_true[idx], y_pred[idx]).statistic),
        n=n,
        random_state=seed,
    )
    return MetricCI(point=rho_hat, low=low, high=high, n=n)


def _rmse_ci(y_true: np.ndarray, y_pred: np.ndarray, seed: int = 0) -> MetricCI:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 4:
        return MetricCI(float("nan"), float("nan"), float("nan"), n=n)
    rmse_hat = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    low, high = _bootstrap_ci(
        lambda idx: float(np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))),
        n=n,
        random_state=seed,
    )
    return MetricCI(point=rmse_hat, low=low, high=high, n=n)


def _r2_ci(y_true: np.ndarray, y_pred: np.ndarray, seed: int = 0) -> MetricCI:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 4:
        return MetricCI(float("nan"), float("nan"), float("nan"), n=n)

    def r2(ys: np.ndarray, yp: np.ndarray) -> float:
        ss_res = float(((ys - yp) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        if ss_tot == 0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    r2_hat = r2(y_true, y_pred)
    low, high = _bootstrap_ci(
        lambda idx: r2(y_true[idx], y_pred[idx]),
        n=n,
        random_state=seed,
    )
    return MetricCI(point=r2_hat, low=low, high=high, n=n)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BaselineModelResult:
    model: str
    endpoint: str
    n: int
    n_clusters: int
    n_folds: int
    spearman: MetricCI
    rmse: MetricCI
    r2: MetricCI
    # Out-of-fold predictions and targets, in original row order.
    oof_predictions: np.ndarray
    oof_targets: np.ndarray
    # Per-fold records for downstream aggregation / plots.
    fold_records: list[dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "n": self.n,
            "n_clusters": self.n_clusters,
            "n_folds": self.n_folds,
            "spearman": asdict(self.spearman),
            "rmse": asdict(self.rmse),
            "r2": asdict(self.r2),
            "fold_records": self.fold_records,
        }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _grouped_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_factory,
    standardize: bool,
    n_splits: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n = len(y)
    oof = np.full(n, np.nan, dtype=float)
    fold_records: list[dict[str, Any]] = []

    n_unique_groups = len(np.unique(groups))
    effective_splits = min(n_splits, n_unique_groups)
    if effective_splits < 2:
        raise ValueError(
            f"Need ≥2 unique groups for GroupKFold, got {n_unique_groups}"
        )

    gkf = GroupKFold(n_splits=effective_splits)
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        oof[test_idx] = preds

        from scipy.stats import spearmanr

        fold_records.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "spearman": float(spearmanr(y_test, preds).statistic),
                "rmse": float(np.sqrt(np.mean((y_test - preds) ** 2))),
            }
        )

    return oof, fold_records


def _evaluate(
    model_name: str,
    batch,
    X: np.ndarray,
    model_factory,
    standardize: bool,
    n_splits: int,
    bootstrap_seed: int,
) -> BaselineModelResult:
    # Duck-type HarmonizedBatch to tolerate sys.path aliasing
    # (e.g. "src.ab_benchmark_loader" vs "ab_benchmark_loader").
    required = ("y", "vh", "vl", "groups", "endpoint")
    missing = [a for a in required if not hasattr(batch, a)]
    if missing:
        raise TypeError(
            f"batch missing required attributes {missing} — expected HarmonizedBatch-like."
        )
    if len(X) != len(batch.y):
        raise ValueError(f"X has {len(X)} rows but batch.y has {len(batch.y)}")

    oof, fold_records = _grouped_oof_predictions(
        X=X,
        y=batch.y,
        groups=batch.groups,
        model_factory=model_factory,
        standardize=standardize,
        n_splits=n_splits,
    )

    return BaselineModelResult(
        model=model_name,
        endpoint=batch.endpoint,
        n=len(batch),
        n_clusters=int(len(np.unique(batch.groups))),
        n_folds=len(fold_records),
        spearman=_spearman_ci(batch.y, oof, seed=bootstrap_seed),
        rmse=_rmse_ci(batch.y, oof, seed=bootstrap_seed),
        r2=_r2_ci(batch.y, oof, seed=bootstrap_seed),
        oof_predictions=oof,
        oof_targets=batch.y.copy(),
        fold_records=fold_records,
    )


# ---------------------------------------------------------------------------
# Public evaluators
# ---------------------------------------------------------------------------


def evaluate_ridge(
    batch,
    X: np.ndarray,
    alpha: float = 1.0,
    n_splits: int = 5,
    bootstrap_seed: int = 0,
) -> BaselineModelResult:
    """Ridge regression on standardized features with GroupKFold."""
    return _evaluate(
        model_name="ridge",
        batch=batch,
        X=X,
        model_factory=lambda: Ridge(alpha=alpha, random_state=0),
        standardize=True,
        n_splits=n_splits,
        bootstrap_seed=bootstrap_seed,
    )


def evaluate_random_forest(
    batch,
    X: np.ndarray,
    n_estimators: int = 300,
    max_depth: int | None = None,
    n_splits: int = 5,
    bootstrap_seed: int = 0,
) -> BaselineModelResult:
    """Random forest regression with GroupKFold. No feature standardization."""
    return _evaluate(
        model_name="random_forest",
        batch=batch,
        X=X,
        model_factory=lambda: RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=0,
            n_jobs=-1,
        ),
        standardize=False,
        n_splits=n_splits,
        bootstrap_seed=bootstrap_seed,
    )


def evaluate_prophet_ab(
    batch,
    X: np.ndarray,
    alpha: float = 1.0,
    n_splits: int = 5,
    bootstrap_seed: int = 0,
) -> BaselineModelResult:
    """PROPHET-Ab style: ridge on ESM-2 embeddings.

    Named as a distinct evaluator so the v1.5 report can cite PROPHET-Ab
    as a labeled baseline without changing the ridge math. When the feature
    matrix X is the 960-dim ESM-2 embedding, this IS the PROPHET-Ab ridge.
    """
    result = evaluate_ridge(
        batch=batch,
        X=X,
        alpha=alpha,
        n_splits=n_splits,
        bootstrap_seed=bootstrap_seed,
    )
    result.model = "prophet_ab_ridge"
    return result


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def run_all_baselines(
    batch,
    X: np.ndarray,
    n_splits: int = 5,
    bootstrap_seed: int = 0,
) -> list[BaselineModelResult]:
    """Run ridge + random forest + PROPHET-Ab ridge on the same batch."""
    return [
        evaluate_ridge(batch, X, n_splits=n_splits, bootstrap_seed=bootstrap_seed),
        evaluate_random_forest(batch, X, n_splits=n_splits, bootstrap_seed=bootstrap_seed),
        evaluate_prophet_ab(batch, X, n_splits=n_splits, bootstrap_seed=bootstrap_seed),
    ]
