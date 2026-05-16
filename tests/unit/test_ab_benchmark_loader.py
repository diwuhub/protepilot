"""Tests for the ab-benchmark bridge loader (Phase 1 Step 2).

The real parquet lives at /Users/di/Projects/ab-benchmark/data/processed/.
These tests skip automatically if the parquet is absent (Phase 0 not
yet run) so the existing 524-test baseline gate remains portable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ab_benchmark_loader import (
    DEFAULT_PARQUET_PATH,
    KNOWN_ENDPOINTS,
    HarmonizedBatch,
    available_endpoints,
    load_harmonized_training_set,
    load_harmonized_wide,
)


PARQUET_EXISTS = DEFAULT_PARQUET_PATH.exists()
pytestmark = pytest.mark.skipif(
    not PARQUET_EXISTS,
    reason=f"ab-benchmark parquet not built at {DEFAULT_PARQUET_PATH}",
)


class TestLoadWide:
    def test_loads_without_filter(self):
        df = load_harmonized_wide()
        assert len(df) > 1000
        assert {"ab_id", "ab_id_canonical", "source", "vh", "vl"}.issubset(df.columns)

    def test_missing_parquet_error_includes_build_hint(self, tmp_path):
        missing = tmp_path / "does_not_exist.parquet"
        with pytest.raises(FileNotFoundError, match="build_harmonized"):
            load_harmonized_wide(missing)

    def test_env_var_override(self, tmp_path, monkeypatch):
        # Point env var at a fake path → should be used.
        fake = tmp_path / "fake.parquet"
        monkeypatch.setenv("PROTEPILOT_AB_BENCHMARK_PARQUET", str(fake))
        with pytest.raises(FileNotFoundError, match=str(fake)):
            load_harmonized_wide()


class TestLoadTrainingSet:
    def test_tm_onset_returns_expected_count(self):
        batch = load_harmonized_training_set("tm_onset_c")
        # Jain 137 contributes all 137 Tm measurements; other sources
        # contribute 0 in Phase 0 (SAbDab is sequence-only).
        assert len(batch) == 137
        assert batch.endpoint == "tm_onset_c"

    def test_hic_rt_returns_expected_count(self):
        batch = load_harmonized_training_set("hic_rt")
        assert len(batch) == 137

    def test_batch_shapes_consistent(self):
        batch = load_harmonized_training_set("tm_onset_c")
        n = len(batch)
        assert batch.ab_ids.shape == (n,)
        assert batch.canonical_ids.shape == (n,)
        assert batch.sources.shape == (n,)
        assert batch.vh.shape == (n,)
        assert batch.vl.shape == (n,)
        assert batch.y.shape == (n,)
        assert batch.groups.shape == (n,)

    def test_vh_vl_are_aa_sequences(self):
        batch = load_harmonized_training_set("tm_onset_c")
        allowed = set("ACDEFGHIKLMNPQRSTVWY")
        for seq in batch.vh[:5]:
            assert set(seq).issubset(allowed), f"Non-standard AA in {seq[:30]}..."
        for seq in batch.vl[:5]:
            assert set(seq).issubset(allowed)

    def test_y_values_physical_for_tm(self):
        batch = load_harmonized_training_set("tm_onset_c")
        # Therapeutic mAb Tm onset is 55-95°C.
        assert np.all(batch.y > 50)
        assert np.all(batch.y < 100)

    def test_groups_integer_and_no_singleton_domination(self):
        batch = load_harmonized_training_set("tm_onset_c")
        n_clusters = len(np.unique(batch.groups))
        # Expect many clusters but fewer than n (some antibodies cluster).
        assert 50 <= n_clusters <= len(batch), (
            f"n={len(batch)}, n_clusters={n_clusters}: unexpected clustering depth"
        )

    def test_groupkfold_no_leakage(self):
        from sklearn.model_selection import GroupKFold

        batch = load_harmonized_training_set("tm_onset_c")
        gkf = GroupKFold(n_splits=5)
        for fold, (train_idx, test_idx) in enumerate(
            gkf.split(X=np.zeros(len(batch)), y=batch.y, groups=batch.groups)
        ):
            train_clusters = set(batch.groups[train_idx])
            test_clusters = set(batch.groups[test_idx])
            assert not (train_clusters & test_clusters), (
                f"fold {fold}: cluster leakage between train/test"
            )

    def test_rejects_unknown_endpoint(self):
        with pytest.raises(ValueError, match="Unknown endpoint"):
            load_harmonized_training_set("fake_endpoint_that_does_not_exist")

    def test_summary_has_expected_keys(self):
        batch = load_harmonized_training_set("hic_rt")
        s = batch.summary()
        assert set(s) == {"endpoint", "n", "n_clusters", "sources", "y_min", "y_max", "y_mean"}
        assert s["n"] == len(batch)


class TestAvailableEndpoints:
    def test_all_six_jain_endpoints_present(self):
        ep = available_endpoints()
        # The six Jain 2017 endpoints should each have n≈137.
        for name in ("tm_onset_c", "hic_rt", "ac_sins", "bvp_score", "psr_score", "expression_mgl"):
            assert ep.get(name, 0) >= 130, f"{name}: {ep.get(name)} rows"

    def test_returned_endpoints_subset_of_known(self):
        ep = available_endpoints()
        assert set(ep).issubset(KNOWN_ENDPOINTS)
