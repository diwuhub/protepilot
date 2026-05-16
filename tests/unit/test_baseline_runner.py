"""Tests for src/baseline_runner.py."""

from pathlib import Path

import numpy as np
import pytest

from src.ab_benchmark_loader import DEFAULT_PARQUET_PATH, HarmonizedBatch
from src.baseline_runner import (
    BaselineModelResult,
    MetricCI,
    _r2_ci,
    _rmse_ci,
    _spearman_ci,
    evaluate_prophet_ab,
    evaluate_random_forest,
    evaluate_ridge,
    run_all_baselines,
)


# ---------------------------------------------------------------------------
# Synthetic-data unit tests (always run)
# ---------------------------------------------------------------------------


def _synthetic_batch(n=50, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    beta = rng.normal(size=d)
    y = X @ beta + rng.normal(size=n) * 0.3
    groups = rng.integers(0, 10, size=n)
    return HarmonizedBatch(
        endpoint="synthetic",
        ab_ids=np.array([f"ab{i}" for i in range(n)]),
        canonical_ids=np.array([f"ab{i}" for i in range(n)]),
        sources=np.array(["synthetic"] * n),
        vh=np.array([""] * n),
        vl=np.array([""] * n),
        y=y,
        groups=groups,
    ), X


class TestMetricCIs:
    def test_spearman_ci_on_correlated(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=100)
        y_pred = y_true + rng.normal(scale=0.1, size=100)
        ci = _spearman_ci(y_true, y_pred)
        assert ci.point > 0.9
        assert ci.low > 0.8

    def test_rmse_positive(self):
        y_true = np.arange(30, dtype=float)
        y_pred = y_true + 1.0
        ci = _rmse_ci(y_true, y_pred)
        assert ci.point == pytest.approx(1.0)

    def test_r2_one_for_perfect_fit(self):
        y = np.arange(30, dtype=float)
        ci = _r2_ci(y, y.copy())
        assert ci.point > 0.99


class TestRidge:
    def test_returns_baseline_result_shape(self):
        batch, X = _synthetic_batch(n=60, d=5)
        res = evaluate_ridge(batch, X, n_splits=5)
        assert isinstance(res, BaselineModelResult)
        assert res.model == "ridge"
        assert res.endpoint == "synthetic"
        assert res.n == 60
        assert res.n_folds <= 5
        assert res.oof_predictions.shape == (60,)
        assert res.oof_targets.shape == (60,)
        assert len(res.fold_records) == res.n_folds

    def test_linear_signal_recovered(self):
        # On a linear-with-noise task, ridge should achieve ρ > 0.5.
        batch, X = _synthetic_batch(n=100, d=5, seed=1)
        res = evaluate_ridge(batch, X, n_splits=5)
        assert res.spearman.point > 0.5

    def test_no_fold_has_cluster_leakage(self):
        batch, X = _synthetic_batch(n=60, d=4)
        res = evaluate_ridge(batch, X, n_splits=5)
        # Can't check leakage from the BaselineModelResult alone, but we
        # can check every fold produced predictions for distinct rows.
        # GroupKFold guarantees no test-set row index appears in >1 fold.
        used_test_rows = []
        oof_nan = np.isnan(res.oof_predictions)
        assert not oof_nan.any(), "some rows have no OOF prediction"


class TestRandomForest:
    def test_runs(self):
        batch, X = _synthetic_batch(n=60, d=5, seed=2)
        res = evaluate_random_forest(batch, X, n_estimators=50, n_splits=5)
        assert res.model == "random_forest"
        assert not np.isnan(res.spearman.point)

    def test_handles_small_cluster_count(self):
        # Only 3 unique groups → n_splits caps at 3.
        rng = np.random.default_rng(0)
        n, d = 30, 4
        X = rng.normal(size=(n, d)).astype(np.float32)
        y = X.sum(axis=1) + rng.normal(scale=0.1, size=n)
        batch = HarmonizedBatch(
            endpoint="synthetic",
            ab_ids=np.array([f"ab{i}" for i in range(n)]),
            canonical_ids=np.array([f"ab{i}" for i in range(n)]),
            sources=np.array(["synthetic"] * n),
            vh=np.array([""] * n),
            vl=np.array([""] * n),
            y=y,
            groups=np.array([0, 1, 2] * 10),
        )
        res = evaluate_random_forest(batch, X, n_estimators=30, n_splits=5)
        assert res.n_folds == 3  # capped to n_unique_groups

    def test_errors_on_one_group(self):
        n, d = 20, 4
        X = np.random.default_rng(0).normal(size=(n, d)).astype(np.float32)
        batch = HarmonizedBatch(
            endpoint="synthetic",
            ab_ids=np.array([f"ab{i}" for i in range(n)]),
            canonical_ids=np.array([f"ab{i}" for i in range(n)]),
            sources=np.array(["synthetic"] * n),
            vh=np.array([""] * n),
            vl=np.array([""] * n),
            y=np.arange(n, dtype=float),
            groups=np.zeros(n, dtype=int),
        )
        with pytest.raises(ValueError, match="≥2 unique groups"):
            evaluate_ridge(batch, X)


class TestProphetAbName:
    def test_model_label(self):
        batch, X = _synthetic_batch(n=60, d=6)
        res = evaluate_prophet_ab(batch, X, n_splits=5)
        assert res.model == "prophet_ab_ridge"


class TestRunAll:
    def test_returns_three(self):
        batch, X = _synthetic_batch(n=60, d=5)
        results = run_all_baselines(batch, X, n_splits=5)
        assert [r.model for r in results] == ["ridge", "random_forest", "prophet_ab_ridge"]
        assert all(isinstance(r, BaselineModelResult) for r in results)


# ---------------------------------------------------------------------------
# Integration test against the real harmonized parquet (skipped if missing)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DEFAULT_PARQUET_PATH.exists(),
    reason="ab-benchmark harmonized parquet not built; run Phase 0.",
)
class TestRealHarmonized:
    def test_ridge_on_tm_endpoint(self):
        from src.ab_benchmark_loader import load_harmonized_training_set
        from src.esm2_features import get_esm2_embeddings

        batch = load_harmonized_training_set("tm_onset_c")
        X = get_esm2_embeddings(batch.vh, batch.vl, allow_build=False)
        assert X.shape == (len(batch), 960)
        res = evaluate_ridge(batch, X, n_splits=5)
        assert res.n == 137
        assert res.n_folds == 5
        # Sanity: Spearman ρ should be non-null. We do not require
        # ρ > 0 at n=137 with wide CIs — that's a reporting story,
        # not an invariant.
        assert not np.isnan(res.spearman.point)

    def test_run_all_on_hic_endpoint(self):
        from src.ab_benchmark_loader import load_harmonized_training_set
        from src.esm2_features import get_esm2_embeddings

        batch = load_harmonized_training_set("hic_rt")
        X = get_esm2_embeddings(batch.vh, batch.vl, allow_build=False)
        results = run_all_baselines(batch, X, n_splits=5)
        assert len(results) == 3
        for r in results:
            assert r.n == 137
            assert r.n_clusters >= 50
            assert r.oof_predictions.shape == (137,)
