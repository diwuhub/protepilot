"""
Test suite for ood_baseline module (src/ood_baseline.py)
========================================================

Tests class-specific baselines, OOD baseline calculator, and
composite OOD score computation.

Markers:
  @pytest.mark.core — No torch/sklearn/streamlit dependencies
"""

import pytest
from src.ood_baseline import (
    DEFAULT_IGG_REFERENCE,
    CLASS_SPECIFIC_BASELINES,
    OODBaselineCalculator,
    compute_composite_ood_score,
)


@pytest.mark.core
class TestClassSpecificBaselines:
    """Test that class-specific reference baselines are well-formed."""

    def test_all_molecule_classes_covered(self):
        """Every expected molecule class has a baseline."""
        expected = {
            "canonical_mab", "bispecific", "fc_fusion", "adc",
            "single_domain", "peptide", "fusion_protein",
            "engineered_scaffold", "unknown",
        }
        assert set(CLASS_SPECIFIC_BASELINES.keys()) == expected

    def test_all_baselines_have_required_metrics(self):
        """Each baseline must have all 5 metrics."""
        required = {"gravy", "cys_count_per_100", "length", "pI", "MW_kDa"}
        for cls_name, baseline in CLASS_SPECIFIC_BASELINES.items():
            assert required.issubset(baseline.keys()), (
                f"Missing metrics for {cls_name}: {required - baseline.keys()}"
            )

    def test_all_metrics_have_mean_and_std(self):
        """Each metric must have at least mean and std."""
        for cls_name, baseline in CLASS_SPECIFIC_BASELINES.items():
            for metric_name, stats in baseline.items():
                assert "mean" in stats, f"{cls_name}.{metric_name} missing 'mean'"
                assert "std" in stats, f"{cls_name}.{metric_name} missing 'std'"
                assert stats["std"] > 0, f"{cls_name}.{metric_name} has non-positive std"

    def test_peptide_much_shorter_than_mab(self):
        """Peptide baseline length should be << mAb length."""
        pep_len = CLASS_SPECIFIC_BASELINES["peptide"]["length"]["mean"]
        mab_len = CLASS_SPECIFIC_BASELINES["canonical_mab"]["length"]["mean"]
        assert pep_len < mab_len / 5, (
            f"Peptide length {pep_len} should be << mAb length {mab_len}"
        )

    def test_single_domain_shorter_than_mab(self):
        """Single-domain baseline length should be < mAb length."""
        sd_len = CLASS_SPECIFIC_BASELINES["single_domain"]["length"]["mean"]
        mab_len = CLASS_SPECIFIC_BASELINES["canonical_mab"]["length"]["mean"]
        assert sd_len < mab_len / 2

    def test_adc_matches_mab(self):
        """ADC baseline should match canonical mAb (same scaffold)."""
        adc = CLASS_SPECIFIC_BASELINES["adc"]
        mab = CLASS_SPECIFIC_BASELINES["canonical_mab"]
        assert adc["length"]["mean"] == mab["length"]["mean"]

    def test_default_igg_reference_unchanged(self):
        """DEFAULT_IGG_REFERENCE should still exist for backward compat."""
        assert "length" in DEFAULT_IGG_REFERENCE
        assert DEFAULT_IGG_REFERENCE["length"]["mean"] == 450


@pytest.mark.core
class TestOODBaselineCalculator:
    """Test the OODBaselineCalculator class."""

    def test_default_baseline(self):
        """Without computed data, should return default IgG reference."""
        calc = OODBaselineCalculator()
        baseline = calc.get_active_baseline()
        assert baseline["length"]["mean"] == DEFAULT_IGG_REFERENCE["length"]["mean"]

    def test_source_default(self):
        """New calculator should report source='default'."""
        calc = OODBaselineCalculator()
        assert calc.source == "default"
        assert calc.get_baseline_info()["is_default"] is True

    def test_get_baseline_for_class_peptide(self):
        """get_baseline_for_class('peptide') should return peptide stats."""
        calc = OODBaselineCalculator()
        baseline = calc.get_baseline_for_class("peptide")
        assert baseline["length"]["mean"] == CLASS_SPECIFIC_BASELINES["peptide"]["length"]["mean"]

    def test_get_baseline_for_class_unknown_falls_back(self):
        """Unknown class should fall back to default IgG reference."""
        calc = OODBaselineCalculator()
        baseline = calc.get_baseline_for_class("nonexistent_class")
        assert baseline["length"]["mean"] == DEFAULT_IGG_REFERENCE["length"]["mean"]

    def test_serialization_roundtrip(self):
        """to_dict() → from_dict() should preserve state."""
        calc = OODBaselineCalculator()
        d = calc.to_dict()
        restored = OODBaselineCalculator.from_dict(d)
        assert restored.source == calc.source
        assert restored.n_samples == calc.n_samples


@pytest.mark.core
class TestCompositeOODScore:
    """Test the composite OOD score function."""

    def test_empty_z_scores(self):
        """Empty z-scores should return 0.0."""
        assert compute_composite_ood_score([]) == 0.0

    def test_all_zero_z_scores(self):
        """All-zero z-scores should return 0.0."""
        assert compute_composite_ood_score([0.0, 0.0, 0.0]) == 0.0

    def test_moderate_z_scores(self):
        """Moderate z-scores should give borderline composite."""
        score = compute_composite_ood_score([1.0, 1.5, 0.5])
        assert 0.5 < score < 2.5

    def test_high_z_scores(self):
        """High z-scores should give clear OOD composite."""
        score = compute_composite_ood_score([4.0, 3.5, 5.0])
        assert score > 2.5

    def test_single_outlier(self):
        """One large z-score with others normal should still flag."""
        score = compute_composite_ood_score([0.5, 0.3, 5.0])
        assert score > 1.5


@pytest.mark.core
class TestOODIntegrationWithPredictor:
    """Test that compute_ood_flags uses class-specific baselines."""

    def test_peptide_not_ood_with_class(self):
        """A 30-aa peptide should NOT be OOD when using peptide baseline."""
        from src.developability_predictor import compute_ood_flags
        peptide_seq = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"
        result = compute_ood_flags(peptide_seq, molecule_class="peptide")
        assert result["is_ood"] is False
        assert result["confidence"] == "High"

    def test_peptide_ood_without_class(self):
        """A 30-aa peptide IS OOD against default IgG baseline."""
        from src.developability_predictor import compute_ood_flags
        peptide_seq = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"
        result = compute_ood_flags(peptide_seq)
        assert result["is_ood"] is True

    def test_mab_sequence_not_ood(self):
        """A ~450-aa mAb-like sequence should not be OOD for canonical_mab."""
        from src.developability_predictor import compute_ood_flags
        mab_seq = "EVQLVESGGGLVQPGG" * 28  # ~448 aa
        result = compute_ood_flags(mab_seq, molecule_class="canonical_mab")
        # Length should be in-distribution for mAb baseline
        length_flag = next(
            (f for f in result["flags"] if f["metric"] == "Sequence length"), None
        )
        assert length_flag is not None
        assert length_flag["z_score"] < 2.0
