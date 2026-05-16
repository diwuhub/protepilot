"""Tests for src/v15_plots.py — render every plot type without crashing,
confirm PNGs are written, verify basic image invariants.
"""

from pathlib import Path

import numpy as np
import pytest

from src.baseline_runner import BaselineModelResult, MetricCI
from src.v15_plots import (
    make_all_plots,
    plot_fold_distribution,
    plot_model_comparison,
    plot_parity,
    plot_residuals,
)


def _fake_result(model: str, endpoint: str, rho: float = 0.3, n: int = 80) -> BaselineModelResult:
    rng = np.random.default_rng(abs(hash((model, endpoint))) % (2 ** 32))
    y_true = rng.normal(size=n).astype(np.float32)
    # Produce predictions with the target rank correlation approximately.
    y_pred = rho * y_true + np.sqrt(max(0.0, 1 - rho ** 2)) * rng.normal(size=n).astype(np.float32)
    return BaselineModelResult(
        model=model,
        endpoint=endpoint,
        n=n,
        n_clusters=40,
        n_folds=5,
        spearman=MetricCI(point=float(rho), low=float(rho - 0.15), high=float(rho + 0.15), n=n),
        rmse=MetricCI(point=1.0, low=0.85, high=1.15, n=n),
        r2=MetricCI(point=0.1, low=-0.05, high=0.3, n=n),
        oof_predictions=y_pred,
        oof_targets=y_true,
        fold_records=[
            {"fold": i, "n_train": 60, "n_test": 20,
             "spearman": float(rho + 0.02 * (i - 2)), "rmse": 1.0}
            for i in range(5)
        ],
    )


@pytest.fixture
def fake_results():
    endpoints = ["tm_onset_c", "hic_rt", "ac_sins"]
    return {
        ep: [
            _fake_result("ridge", ep, rho=0.2 + 0.05 * i),
            _fake_result("random_forest", ep, rho=0.25 + 0.05 * i),
            _fake_result("prophet_ab_ridge", ep, rho=0.2 + 0.05 * i),
        ]
        for i, ep in enumerate(endpoints)
    }


class TestParity:
    def test_writes_one_png_per_endpoint(self, fake_results, tmp_path):
        paths = plot_parity(fake_results, tmp_path, model="random_forest")
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"
            assert p.stat().st_size > 2000

    def test_skips_missing_model(self, fake_results, tmp_path):
        # Drop random_forest from one endpoint.
        fake_results["hic_rt"] = [r for r in fake_results["hic_rt"] if r.model != "random_forest"]
        paths = plot_parity(fake_results, tmp_path, model="random_forest")
        assert len(paths) == 2


class TestFoldDistribution:
    def test_writes_single_png(self, fake_results, tmp_path):
        path = plot_fold_distribution(fake_results, tmp_path, model="random_forest")
        assert path.exists()
        assert path.stat().st_size > 3000


class TestModelComparison:
    def test_writes_png(self, fake_results, tmp_path):
        path = plot_model_comparison(fake_results, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 3000

    def test_raises_on_empty(self, tmp_path):
        with pytest.raises(ValueError, match="no endpoints"):
            plot_model_comparison({}, tmp_path)


class TestResiduals:
    def test_writes_one_per_endpoint(self, fake_results, tmp_path):
        paths = plot_residuals(fake_results, tmp_path, model="ridge")
        assert len(paths) == 3
        for p in paths:
            assert p.exists()


class TestMakeAll:
    def test_make_all_returns_all_kinds(self, fake_results, tmp_path):
        out = make_all_plots(fake_results, tmp_path)
        assert set(out.keys()) == {"parity", "fold_distribution", "model_comparison", "residuals"}
        assert len(out["parity"]) == 3
        assert len(out["residuals"]) == 3
        assert len(out["fold_distribution"]) == 1
        assert len(out["model_comparison"]) == 1
