"""Phase 1 v1.5 report plots.

Four diagnostic plots for the v1.5 report:

    1. Parity scatter (OOF predicted vs measured), one panel per endpoint.
    2. Per-fold Spearman ρ distribution (box/strip), one panel per endpoint.
    3. Model×endpoint Spearman ρ bar chart with bootstrap 95% CI whiskers.
    4. Per-endpoint prediction–residual plot (OOF residual vs measured).

Every plot is saved as PNG under a caller-specified directory. `plt.close()`
is called on each figure so notebook/CI use is clean.

Imports matplotlib lazily so tests that don't render don't pay the cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class _ReportInputs:
    """Minimal duck-typed inputs so tests can use synthetic data."""

    results_by_endpoint: dict[str, list]  # {endpoint: [BaselineModelResult, ...]}


def _get_mpl():
    """Lazy matplotlib import to keep test module load cheap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------


def plot_parity(results_by_endpoint: dict[str, list], out_dir: Path | str, model: str = "random_forest") -> list[Path]:
    """One parity panel per endpoint for the named model.

    Saves {out_dir}/parity_{endpoint}_{model}.png. Returns list of paths.
    """
    plt = _get_mpl()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for endpoint, results in results_by_endpoint.items():
        r = _find_model(results, model)
        if r is None:
            continue
        y_true = r.oof_targets
        y_pred = r.oof_predictions

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(y_true, y_pred, alpha=0.6, s=22, edgecolor="white", linewidth=0.5)
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="0.6", linewidth=1, label="y=x")
        ax.set_xlabel(f"measured {endpoint}")
        ax.set_ylabel(f"OOF predicted {endpoint}")
        ax.set_title(
            f"{endpoint}  ·  {r.model}\n"
            f"ρ={r.spearman.point:+.2f} [{r.spearman.low:+.2f}, {r.spearman.high:+.2f}]  ·  n={r.n}"
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = out_dir / f"parity_{endpoint}_{model}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_fold_distribution(results_by_endpoint: dict[str, list], out_dir: Path | str, model: str = "random_forest") -> Path:
    """Per-endpoint strip plot of per-fold Spearman ρ with the overall point.

    Saves {out_dir}/fold_distribution_{model}.png.
    """
    plt = _get_mpl()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoints = []
    fold_rhos: list[list[float]] = []
    overall_rho: list[float] = []
    ci_lo: list[float] = []
    ci_hi: list[float] = []
    for ep, results in results_by_endpoint.items():
        r = _find_model(results, model)
        if r is None:
            continue
        endpoints.append(ep)
        fold_rhos.append([float(f["spearman"]) for f in r.fold_records])
        overall_rho.append(r.spearman.point)
        ci_lo.append(r.spearman.low)
        ci_hi.append(r.spearman.high)

    n = len(endpoints)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * n + 3), 4.2))
    xs = np.arange(n)

    for i, rhos in enumerate(fold_rhos):
        ax.scatter([i] * len(rhos), rhos, alpha=0.6, s=30, color="0.3", zorder=2)

    ax.errorbar(
        xs, overall_rho,
        yerr=[np.array(overall_rho) - np.array(ci_lo), np.array(ci_hi) - np.array(overall_rho)],
        fmt="o", color="C0", ecolor="C0", capsize=4, markersize=7, linewidth=1.5,
        label="overall ρ (bootstrap 95% CI)", zorder=3,
    )

    ax.axhline(0, color="0.7", linestyle=":", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(endpoints, rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Per-fold ρ and overall OOF CI · {model}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = out_dir / f"fold_distribution_{model}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_model_comparison(results_by_endpoint: dict[str, list], out_dir: Path | str) -> Path:
    """Grouped bar chart of Spearman ρ by (endpoint, model) with 95% CI whiskers.

    Saves {out_dir}/model_comparison.png.
    """
    plt = _get_mpl()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoints = list(results_by_endpoint.keys())
    models_seen: list[str] = []
    for results in results_by_endpoint.values():
        for r in results:
            if r.model not in models_seen:
                models_seen.append(r.model)

    n_ep = len(endpoints)
    n_m = len(models_seen)
    if n_ep == 0 or n_m == 0:
        raise ValueError("no endpoints or models to plot")

    width = 0.78 / n_m
    fig, ax = plt.subplots(figsize=(max(7.5, 1.1 * n_ep + 2), 4.8))
    x = np.arange(n_ep)

    for mi, model in enumerate(models_seen):
        rhos = []
        lows = []
        highs = []
        for ep in endpoints:
            r = _find_model(results_by_endpoint[ep], model)
            if r is None:
                rhos.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                continue
            rhos.append(r.spearman.point)
            lows.append(r.spearman.low)
            highs.append(r.spearman.high)
        rhos_a = np.asarray(rhos)
        yerr = np.array([rhos_a - np.asarray(lows), np.asarray(highs) - rhos_a])
        offset = (mi - (n_m - 1) / 2) * width
        ax.bar(x + offset, rhos_a, width=width, label=model, alpha=0.9,
               yerr=yerr, error_kw={"capsize": 3, "linewidth": 1})

    ax.axhline(0, color="0.7", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(endpoints, rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ (OOF, grouped-CV)")
    ax.set_title("ProtePilot v1.5 baselines on Jain 137 (95% bootstrap CI)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = out_dir / "model_comparison.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_residuals(results_by_endpoint: dict[str, list], out_dir: Path | str, model: str = "random_forest") -> list[Path]:
    """OOF residual vs measured endpoint for diagnostic inspection.

    Saves {out_dir}/residuals_{endpoint}_{model}.png.
    """
    plt = _get_mpl()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for endpoint, results in results_by_endpoint.items():
        r = _find_model(results, model)
        if r is None:
            continue
        y_true = r.oof_targets
        y_pred = r.oof_predictions
        resid = y_pred - y_true

        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        ax.scatter(y_true, resid, alpha=0.6, s=22, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
        ax.set_xlabel(f"measured {endpoint}")
        ax.set_ylabel("residual (OOF pred − measured)")
        ax.set_title(f"residuals · {endpoint} · {model}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = out_dir / f"residuals_{endpoint}_{model}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_model(results: Iterable, model_name: str):
    for r in results:
        if r.model == model_name:
            return r
    return None


def make_all_plots(
    results_by_endpoint: dict[str, list],
    out_dir: Path | str,
    parity_model: str = "random_forest",
) -> dict[str, list[Path]]:
    """One-shot: make every plot type and return a map of plot-kind → paths."""
    return {
        "parity": plot_parity(results_by_endpoint, out_dir, model=parity_model),
        "fold_distribution": [plot_fold_distribution(results_by_endpoint, out_dir, model=parity_model)],
        "model_comparison": [plot_model_comparison(results_by_endpoint, out_dir)],
        "residuals": plot_residuals(results_by_endpoint, out_dir, model=parity_model),
    }
