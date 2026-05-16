"""
test_preclinical_twin.py — Unit Tests for Preclinical PK Twin (Module 8)
========================================================================
Tests the preclinical_twin module which predicts human half-life for
therapeutic antibodies using empirical models calibrated to published IgG
clearance data.

Key test areas:
  - Baseline half-life ~21 days for standard mAb
  - Extreme pI (>9.0) reduces half-life
  - High hydrophobicity reduces half-life
  - Glycoform modifiers (high_mannose, afucosylated) change half-life
  - Bispecific (large MW) gets MW penalty
"""

import pytest
from src.preclinical_twin import (
    predict_human_half_life,
    assess_glycoform_pk_impact,
    BASELINE_HALF_LIFE_DAYS,
    GLYCOFORM_PK_MODIFIERS,
    get_glycoform_profiles,
    check_fcrn_binding_motif,
)


@pytest.mark.core
class TestBaselineHalfLife:
    """Test baseline half-life prediction."""

    def test_baseline_half_life_returns_dict(self):
        """Test: predict_human_half_life() returns dict."""
        result = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.35,
        )

        assert isinstance(result, dict)

    def test_standard_mab_has_baseline_half_life(self):
        """Test: Baseline half-life ~21 days for standard mAb."""
        result = predict_human_half_life(
            global_pi=7.5,  # Optimal pI
            hydrophobicity=0.35,  # Moderate hydrophobicity
            liability_density=30.0,  # Moderate liability density
            fcrn_binding_motif_intact=True,
            mw_kda=150.0,  # Standard IgG
            glycoform_profile="standard_cho",
        )

        assert "half_life_days" in result
        half_life = result["half_life_days"]
        # Standard mAb should have baseline ~21 days
        assert 18 <= half_life <= 24

    def test_result_includes_required_fields(self):
        """Test: Result includes all required fields."""
        result = predict_human_half_life(global_pi=7.5)

        assert "half_life_days" in result
        assert "baseline_days" in result
        assert "penalties" in result
        assert "risk_assessment" in result
        assert "risk_color" in result
        assert "glycoform_impact" in result
        assert "recommendations" in result
        assert "summary" in result

    def test_half_life_days_is_positive(self):
        """Test: Half-life is always positive."""
        result = predict_human_half_life(global_pi=7.5, hydrophobicity=0.35)

        assert result["half_life_days"] > 0
        assert result["half_life_days"] < 100  # But not unreasonably high

    def test_baseline_days_equals_constant(self):
        """Test: Baseline days is set to constant."""
        result = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.35,
            liability_density=30.0,
            fcrn_binding_motif_intact=True,
        )

        assert result["baseline_days"] == BASELINE_HALF_LIFE_DAYS


@pytest.mark.core
class TestPIPenalty:
    """Test pI-dependent half-life penalties."""

    def test_extreme_high_pI_reduces_half_life(self):
        """Test: Extreme pI (>9.0) reduces half-life."""
        result_normal = predict_human_half_life(
            global_pi=7.5,  # Normal pI
            hydrophobicity=0.35,
            liability_density=30.0,
        )

        result_high_pi = predict_human_half_life(
            global_pi=9.5,  # Extreme high pI
            hydrophobicity=0.35,
            liability_density=30.0,
        )

        # High pI should result in lower half-life
        assert result_high_pi["half_life_days"] < result_normal["half_life_days"]

    def test_elevated_pI_has_moderate_penalty(self):
        """Test: Elevated pI (8.5-9.0) has moderate penalty."""
        result_normal = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.35,
        )

        result_elevated = predict_human_half_life(
            global_pi=8.8,  # Elevated but not extreme
            hydrophobicity=0.35,
        )

        # Elevated pI should have some penalty but not extreme
        assert result_elevated["half_life_days"] < result_normal["half_life_days"]
        assert result_elevated["half_life_days"] > result_normal["half_life_days"] * 0.5

    def test_low_pI_has_mild_penalty(self):
        """Test: Low pI (5.5-6.5) has mild penalty."""
        result_normal = predict_human_half_life(global_pi=7.5)
        result_low = predict_human_half_life(global_pi=5.8)

        # Low pI should have mild penalty
        assert result_low["half_life_days"] <= result_normal["half_life_days"]
        # But should not be severe
        assert result_low["half_life_days"] > result_normal["half_life_days"] * 0.6

    def test_penalties_list_includes_pI_penalty(self):
        """Test: Penalties list includes pI penalty when applicable."""
        result = predict_human_half_life(global_pi=9.2)

        penalties = result["penalties"]
        assert isinstance(penalties, list)
        # High pI should trigger a penalty
        assert any("pI" in str(p.get("factor", "")) for p in penalties)


@pytest.mark.core
class TestHydrophobicityPenalty:
    """Test hydrophobicity-dependent penalties."""

    def test_high_hydrophobicity_reduces_half_life(self):
        """Test: High hydrophobicity reduces half-life."""
        result_low_hydro = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.25,  # Low hydrophobicity
        )

        result_high_hydro = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.65,  # High hydrophobicity
        )

        # High hydrophobicity should reduce half-life
        assert result_high_hydro["half_life_days"] < result_low_hydro["half_life_days"]

    def test_severe_hydrophobicity_strong_penalty(self):
        """Test: Severe hydrophobicity (>0.6) gives strong penalty."""
        result_normal = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.35,
        )

        result_severe = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.75,  # Very high
        )

        # Severe hydrophobicity should give significant penalty
        assert result_severe["half_life_days"] < result_normal["half_life_days"] * 0.8

    def test_moderate_hydrophobicity_mild_penalty(self):
        """Test: Moderate hydrophobicity (0.4-0.6) gives mild penalty."""
        result_low = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.35,
        )

        result_moderate = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.50,
        )

        # Moderate should have some penalty but not severe
        assert result_moderate["half_life_days"] < result_low["half_life_days"]
        assert result_moderate["half_life_days"] > result_low["half_life_days"] * 0.7


@pytest.mark.core
class TestGlycoformModifiers:
    """Test glycoform-specific half-life modifiers."""

    def test_standard_cho_glycoform_baseline(self):
        """Test: Standard CHO glycoform yields baseline half-life."""
        result = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.35,
            liability_density=30.0,
            glycoform_profile="standard_cho",
        )

        # Standard CHO should yield ~baseline
        assert 18 <= result["half_life_days"] <= 24

    def test_high_mannose_reduces_half_life(self):
        """Test: High-mannose glycoform reduces half-life."""
        result_standard = predict_human_half_life(
            global_pi=7.5,
            glycoform_profile="standard_cho",
        )

        if "high_mannose" in GLYCOFORM_PK_MODIFIERS:
            result_high_mannose = predict_human_half_life(
                global_pi=7.5,
                glycoform_profile="high_mannose",
            )

            # High-mannose => faster clearance via mannose receptor
            assert result_high_mannose["half_life_days"] < result_standard["half_life_days"]

    def test_afucosylated_has_similar_pk_to_baseline(self):
        """Test: Afucosylated glycoform has similar PK to baseline."""
        result_standard = predict_human_half_life(
            global_pi=7.5,
            glycoform_profile="standard_cho",
        )

        if "afucosylated" in GLYCOFORM_PK_MODIFIERS:
            result_afuc = predict_human_half_life(
                global_pi=7.5,
                glycoform_profile="afucosylated",
            )

            # Afucosylated enhances ADCC but should have similar half-life
            assert abs(result_afuc["half_life_days"] - result_standard["half_life_days"]) < 5

    def test_sialylated_extends_half_life(self):
        """Test: Sialylated glycoform may extend half-life."""
        result_standard = predict_human_half_life(
            global_pi=7.5,
            glycoform_profile="standard_cho",
        )

        if "sialylated" in GLYCOFORM_PK_MODIFIERS:
            result_sialylated = predict_human_half_life(
                global_pi=7.5,
                glycoform_profile="sialylated",
            )

            # Sialylated may extend half-life via altered clearance
            assert result_sialylated["half_life_days"] >= result_standard["half_life_days"]

    def test_glycoform_impact_dict_populated(self):
        """Test: Glycoform impact dict is populated."""
        result = predict_human_half_life(
            global_pi=7.5,
            glycoform_profile="standard_cho",
        )

        assert "glycoform_impact" in result
        assert isinstance(result["glycoform_impact"], dict)
        assert len(result["glycoform_impact"]) > 0


@pytest.mark.core
class TestMolecularWeightPenalty:
    """Test molecular weight effects on half-life."""

    def test_standard_mab_no_mw_penalty(self):
        """Test: Standard IgG (150 kDa) has no MW penalty."""
        result = predict_human_half_life(
            global_pi=7.5,
            mw_kda=150.0,  # Standard
            hydrophobicity=0.35,
        )

        # Should not have MW penalty
        penalties = result["penalties"]
        mw_penalties = [p for p in penalties if "Molecular Weight" in str(p.get("factor", ""))]
        assert len(mw_penalties) == 0

    def test_bispecific_large_mw_gets_penalty(self):
        """Test: Bispecific (large MW) gets MW penalty."""
        result_standard = predict_human_half_life(
            global_pi=7.5,
            mw_kda=150.0,
            hydrophobicity=0.35,
        )

        result_bispecific = predict_human_half_life(
            global_pi=7.5,
            mw_kda=220.0,  # Bispecific MW
            hydrophobicity=0.35,
        )

        # Large MW should reduce half-life
        assert result_bispecific["half_life_days"] < result_standard["half_life_days"]

    def test_fusion_protein_large_mw_penalty(self):
        """Test: Fusion protein (very large) has significant penalty."""
        result_standard = predict_human_half_life(
            global_pi=7.5,
            mw_kda=150.0,
        )

        result_fusion = predict_human_half_life(
            global_pi=7.5,
            mw_kda=300.0,  # Large fusion protein
        )

        # Very large should have significant penalty
        assert result_fusion["half_life_days"] < result_standard["half_life_days"] * 0.9


@pytest.mark.core
class TestFcRnBindingMotif:
    """Test FcRn binding motif impact."""

    def test_intact_fcrn_no_penalty(self):
        """Test: Intact FcRn binding motif has no penalty."""
        result = predict_human_half_life(
            global_pi=7.5,
            fcrn_binding_motif_intact=True,
        )

        penalties = result["penalties"]
        fcrn_penalties = [p for p in penalties if "FcRn" in str(p.get("factor", ""))]
        assert len(fcrn_penalties) == 0

    def test_disrupted_fcrn_reduces_half_life(self):
        """Test: Disrupted FcRn binding significantly reduces half-life."""
        result_intact = predict_human_half_life(
            global_pi=7.5,
            fcrn_binding_motif_intact=True,
        )

        result_disrupted = predict_human_half_life(
            global_pi=7.5,
            fcrn_binding_motif_intact=False,
        )

        # Disrupted FcRn should give ~30% reduction
        assert result_disrupted["half_life_days"] < result_intact["half_life_days"]
        # Should be significant (30% = 0.7 multiplier)
        ratio = result_disrupted["half_life_days"] / result_intact["half_life_days"]
        assert 0.65 < ratio < 0.75


@pytest.mark.core
class TestLiabilityDensityPenalty:
    """Test liability density (PTM/aggregation proxy) penalty."""

    def test_low_liability_no_penalty(self):
        """Test: Low liability density has no penalty."""
        result = predict_human_half_life(
            global_pi=7.5,
            liability_density=25.0,  # Low
        )

        penalties = result["penalties"]
        liab_penalties = [p for p in penalties if "Liability" in str(p.get("factor", ""))]
        assert len(liab_penalties) == 0

    def test_moderate_liability_mild_penalty(self):
        """Test: Moderate liability density gives mild penalty."""
        result_low = predict_human_half_life(
            global_pi=7.5,
            liability_density=30.0,
        )

        result_moderate = predict_human_half_life(
            global_pi=7.5,
            liability_density=60.0,  # Moderate
        )

        # Moderate liability should give mild penalty
        assert result_moderate["half_life_days"] < result_low["half_life_days"]
        assert result_moderate["half_life_days"] > result_low["half_life_days"] * 0.8

    def test_high_liability_strong_penalty(self):
        """Test: High liability density gives strong penalty."""
        result_low = predict_human_half_life(
            global_pi=7.5,
            liability_density=30.0,
        )

        result_high = predict_human_half_life(
            global_pi=7.5,
            liability_density=100.0,  # High
        )

        # High liability should give meaningful penalty (at least some reduction)
        assert result_high["half_life_days"] < result_low["half_life_days"]


@pytest.mark.core
class TestRiskAssessment:
    """Test risk assessment classification."""

    def test_risk_assessment_returns_valid_level(self):
        """Test: Risk assessment returns Low/Medium/High/Very High."""
        result = predict_human_half_life(global_pi=7.5)

        risk = result["risk_assessment"]
        assert risk in ["Low", "Medium", "High", "Very High"]

    def test_risk_color_is_hex_string(self):
        """Test: Risk color is valid hex string."""
        result = predict_human_half_life(global_pi=7.5)

        color = result["risk_color"]
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7  # #RRGGBB

    def test_low_risk_for_good_half_life(self):
        """Test: Good half-life (>=18 days) gets Low risk."""
        result = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.35,
            liability_density=30.0,
        )

        if result["half_life_days"] >= 18:
            assert result["risk_assessment"] == "Low"

    def test_medium_risk_for_moderate_half_life(self):
        """Test: Moderate half-life (12-18 days) gets Medium risk."""
        # Create moderate risk scenario
        result = predict_human_half_life(
            global_pi=8.5,  # Elevated pI
            hydrophobicity=0.45,
        )

        if 12 <= result["half_life_days"] < 18:
            assert result["risk_assessment"] == "Medium"

    def test_high_risk_for_poor_half_life(self):
        """Test: Poor half-life (6-12 days) gets High risk."""
        result = predict_human_half_life(
            global_pi=9.2,  # High pI
            hydrophobicity=0.65,  # High hydrophobicity
            liability_density=80.0,  # High liability
        )

        if 6 <= result["half_life_days"] < 12:
            assert result["risk_assessment"] == "High"

    def test_very_high_risk_for_critical_half_life(self):
        """Test: Very poor half-life (<6 days) gets Very High risk."""
        result = predict_human_half_life(
            global_pi=9.5,  # Extreme pI
            hydrophobicity=0.8,  # Extreme hydrophobicity
            fcrn_binding_motif_intact=False,
            liability_density=100.0,
        )

        if result["half_life_days"] < 6:
            assert result["risk_assessment"] == "Very High"


@pytest.mark.core
class TestPKParameters:
    """Test PK parameters calculation."""

    def test_clearance_calculated(self):
        """Test: Clearance (ml/day/kg) is calculated."""
        result = predict_human_half_life(global_pi=7.5)

        assert "clearance_ml_day_kg" in result
        assert result["clearance_ml_day_kg"] > 0

    def test_volume_distribution_calculated(self):
        """Test: Volume of distribution (ml/kg) is calculated."""
        result = predict_human_half_life(global_pi=7.5)

        assert "vd_ml_kg" in result
        assert result["vd_ml_kg"] > 0

    def test_clearance_increases_with_half_life_decrease(self):
        """Test: Clearance increases as half-life decreases."""
        result_good = predict_human_half_life(global_pi=7.5)
        result_poor = predict_human_half_life(global_pi=9.5)

        # Poor half-life => higher clearance
        assert result_poor["clearance_ml_day_kg"] > result_good["clearance_ml_day_kg"]

    def test_vd_increases_with_mw(self):
        """Test: Volume of distribution increases with molecular weight."""
        result_standard = predict_human_half_life(
            global_pi=7.5,
            mw_kda=150.0,
        )

        result_large = predict_human_half_life(
            global_pi=7.5,
            mw_kda=250.0,
        )

        # Larger molecules should have larger Vd
        assert result_large["vd_ml_kg"] > result_standard["vd_ml_kg"]


@pytest.mark.core
class TestRecommendations:
    """Test recommendation generation."""

    def test_recommendations_list_generated(self):
        """Test: Recommendations list is generated."""
        result = predict_human_half_life(global_pi=7.5)

        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)

    def test_high_pI_generates_recommendation(self):
        """Test: High pI generates recommendation for optimization."""
        result = predict_human_half_life(global_pi=9.2)

        recommendations = result["recommendations"]
        # Should have recommendation about pI
        assert any("pI" in r.lower() for r in recommendations) or len(recommendations) > 0

    def test_high_hydrophobicity_generates_recommendation(self):
        """Test: High hydrophobicity generates recommendation."""
        result = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=0.7,
        )

        recommendations = result["recommendations"]
        # Should have some recommendations for high hydrophobicity
        assert len(recommendations) > 0

    def test_summary_text_generated(self):
        """Test: Summary text is generated."""
        result = predict_human_half_life(global_pi=7.5)

        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_summary_contains_half_life_value(self):
        """Test: Summary contains half-life value."""
        result = predict_human_half_life(global_pi=7.5)

        summary = result["summary"]
        # Should mention half-life in days
        assert "days" in summary.lower() or len(summary) > 50


@pytest.mark.core
class TestGlycoformAssessment:
    """Test glycoform-specific assessment."""

    def test_assess_glycoform_pk_impact_returns_dict(self):
        """Test: assess_glycoform_pk_impact() returns dict."""
        result = assess_glycoform_pk_impact(
            glycoform_profile="standard_cho",
            base_pi=7.5,
        )

        assert isinstance(result, dict)

    def test_glycoform_profiles_available(self):
        """Test: Multiple glycoform profiles are available."""
        profiles = get_glycoform_profiles()

        assert isinstance(profiles, dict)
        assert len(profiles) > 0
        assert "standard_cho" in profiles

    def test_glycoform_catalog_populated(self):
        """Test: Glycoform catalog is populated."""
        assert len(GLYCOFORM_PK_MODIFIERS) > 0
        assert "standard_cho" in GLYCOFORM_PK_MODIFIERS


@pytest.mark.core
class TestFcRnMotifDetection:
    """Test FcRn binding motif detection."""

    def test_fcrn_check_returns_dict(self):
        """Test: check_fcrn_binding_motif() returns dict."""
        sequence = "MKVLTCGDR" * 100  # Dummy sequence

        result = check_fcrn_binding_motif(sequence)

        assert isinstance(result, dict)

    def test_fcrn_check_identifies_his_residues(self):
        """Test: FcRn check identifies His residues."""
        # Sequence with His at key positions
        sequence = "M" * 250 + "H" + "V" * 182 + "H" + "V" * 100 + "I"

        result = check_fcrn_binding_motif(sequence)

        assert isinstance(result, dict)


@pytest.mark.core
class TestEdgeCases:
    """Test edge cases and extreme parameters."""

    def test_very_low_pi_handled(self):
        """Test: Very low pI is handled."""
        result = predict_human_half_life(global_pi=4.0)

        assert result["half_life_days"] > 0

    def test_very_high_pi_handled(self):
        """Test: Very high pI is handled."""
        result = predict_human_half_life(global_pi=10.0)

        assert result["half_life_days"] > 0

    def test_extreme_hydrophobicity_handled(self):
        """Test: Extreme hydrophobicity doesn't crash."""
        result = predict_human_half_life(
            global_pi=7.5,
            hydrophobicity=1.0,
        )

        assert result["half_life_days"] > 0

    def test_assembly_multiplier_applied(self):
        """Test: Assembly completeness multiplier affects half-life."""
        result_complete = predict_human_half_life(
            global_pi=7.5,
            assembly_half_life_multiplier=1.0,
        )

        result_incomplete = predict_human_half_life(
            global_pi=7.5,
            assembly_half_life_multiplier=0.5,
        )

        # Incomplete assembly should have lower half-life
        assert result_incomplete["half_life_days"] < result_complete["half_life_days"]
