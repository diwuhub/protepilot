"""Tests for src/mutation/proteingym.py."""

import numpy as np
import pandas as pd
import pytest

from src.mutation.proteingym import (
    _bootstrap_spearman,
    _parse_mutation,
    evaluate_proteingym_csv,
)
from src.mutation.schema import AntibodyChain


TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)


class TestParseMutation:
    def test_simple_label(self):
        # S27A — WT S at pos 27 (1-indexed), mutant A.
        parsed = _parse_mutation("S27A")
        assert parsed is not None
        chain, pos, wt, mt = parsed
        assert chain is AntibodyChain.VH  # default
        assert pos == 26  # 1-indexed → 0-indexed
        assert wt == "S"
        assert mt == "A"

    def test_heavy_prefix(self):
        parsed = _parse_mutation("H:S27A")
        assert parsed is not None
        assert parsed[0] is AntibodyChain.VH

    def test_vh_prefix(self):
        parsed = _parse_mutation("VH:S27A")
        assert parsed[0] is AntibodyChain.VH

    def test_light_prefix(self):
        parsed = _parse_mutation("L:S27A")
        assert parsed[0] is AntibodyChain.VL

    def test_junk_returns_none(self):
        assert _parse_mutation("not a mutation") is None
        assert _parse_mutation("XYZ123") is None

    def test_lowercase_invalid(self):
        # Mutations must be uppercase AAs.
        assert _parse_mutation("s27a") is None


class TestBootstrapSpearman:
    def test_perfect_correlation(self):
        x = np.arange(30.0)
        y = x * 2
        rho, lo, hi = _bootstrap_spearman(x, y, n_boot=300, random_state=0)
        assert rho == pytest.approx(1.0)
        assert lo > 0.9

    def test_tiny_n_returns_nan(self):
        rho, lo, hi = _bootstrap_spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0]), n_boot=100)
        assert np.isnan(rho)


class TestEvaluateProteinGym:
    @pytest.fixture
    def fake_scorer(self, monkeypatch):
        """Fake MaskedMarginalScorer: LLR = -|pos - 27| / 10 so mutations
        near pos 27 score highest. Deterministic."""
        pytest.importorskip("torch", reason="torch not installed (Layer 4 training)")
        from src.mutation.masked_marginal import MaskedMarginalScorer
        from src.mutation.schema import ScoredMutation

        class _FakeScorer:
            def score(self, vh, vl, candidates):
                return [
                    ScoredMutation(
                        candidate=c,
                        llr=-abs(c.position - 26) / 10.0,
                        wildtype_prob=0.1,
                        mutant_prob=0.3,
                    )
                    for c in candidates
                ]

        return _FakeScorer()

    def test_evaluates_simple_csv(self, tmp_path, fake_scorer):
        csv = tmp_path / "dms.csv"
        # Build a CSV with 5 valid single mutations and known DMS scores.
        rows = []
        for i, mutant_aa in enumerate("ACDEF"):
            wt = TRASTUZUMAB_VH[20 + i]
            if wt == mutant_aa:
                continue
            rows.append({
                "mutant": f"{wt}{21 + i}{mutant_aa}",  # 1-indexed
                "DMS_score": 10.0 - i,
            })
        pd.DataFrame(rows).to_csv(csv, index=False)

        result = evaluate_proteingym_csv(
            csv, parent_vh=TRASTUZUMAB_VH, parent_vl="", scorer=fake_scorer,
        )
        s = result.summary()
        assert s["n_rows"] == len(rows)
        assert s["n_scored"] == len(rows)
        assert not np.isnan(result.spearman_rho)

    def test_skips_mismatched_wildtype(self, tmp_path, fake_scorer):
        csv = tmp_path / "dms.csv"
        # Force a WT that doesn't match the parent at that position.
        actual_wt_at_1 = TRASTUZUMAB_VH[0]
        fake_wt = "W" if actual_wt_at_1 != "W" else "Y"
        pd.DataFrame({
            "mutant": [f"{fake_wt}1A"],
            "DMS_score": [5.0],
        }).to_csv(csv, index=False)

        result = evaluate_proteingym_csv(
            csv, parent_vh=TRASTUZUMAB_VH, parent_vl="", scorer=fake_scorer,
        )
        assert result.n_rows == 1
        assert result.n_scored == 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            evaluate_proteingym_csv(
                tmp_path / "nope.csv",
                parent_vh=TRASTUZUMAB_VH,
            )

    def test_missing_columns_raise(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="mutation column"):
            evaluate_proteingym_csv(csv, parent_vh=TRASTUZUMAB_VH)
