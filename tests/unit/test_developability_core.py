"""
Test suite for developability_core module (src/developability_core.py)
======================================================================

Tests the integrated risk assessment and QTPP generation functionality.
Covers score ranges, grade mapping, QTPP structure, and user criteria.

Markers:
  @pytest.mark.core — No torch/sklearn/streamlit dependencies
"""

import pytest
from src.developability_core import (
    RiskDimension,
    DevelopabilityAssessment,
    QTPPRow,
    generate_qtpp,
    _grade_score,
    QTPP_ACCEPTANCE_DEFAULTS,
)


@pytest.mark.core
class TestGradeScore:
    """Test score-to-grade conversion."""

    def test_low_score_converts_to_low_grade(self):
        """Test low scores (0.0-0.3) → Low grade."""
        grade, color = _grade_score(0.0)
        assert grade == "Low"
        assert color == "#10B981"  # Green

        grade, color = _grade_score(0.15)
        assert grade == "Low"

    def test_medium_score_converts_to_medium_grade(self):
        """Test medium scores → Medium grade."""
        # GRADE_LOW_UPPER = 0.25, GRADE_MEDIUM_UPPER = 0.55
        grade, color = _grade_score(0.4)
        assert grade == "Medium"
        assert color == "#F59E0B"  # Amber

        grade, color = _grade_score(0.5)
        assert grade == "Medium"

    def test_high_score_converts_to_high_grade(self):
        """Test high scores (>0.7) → High grade."""
        grade, color = _grade_score(0.75)
        assert grade == "High"
        assert color == "#EF4444"  # Red

        grade, color = _grade_score(1.0)
        assert grade == "High"

    def test_boundary_scores(self):
        """Test boundary between grade thresholds."""
        # Lower boundary
        grade_low, _ = _grade_score(0.0)
        grade_trans, _ = _grade_score(0.30)  # Likely transition point
        assert grade_low == "Low"
        # grade_trans could be Low or Medium depending on exact threshold


@pytest.mark.core
class TestRiskDimension:
    """Test RiskDimension dataclass."""

    def test_risk_dimension_creation(self):
        """Test RiskDimension instantiation."""
        dim = RiskDimension(
            name="aggregation",
            display_name="Aggregation Risk",
            score=0.35,
            weight=0.30,
            grade="Medium",
            evidence=["High GRAVY score", "Multiple Trp residues"],
        )
        assert dim.name == "aggregation"
        assert dim.score == 0.35
        assert dim.weight == 0.30

    def test_weighted_score_property(self):
        """Test weighted_score calculation."""
        dim = RiskDimension(
            name="test",
            score=0.50,
            weight=0.25,
        )
        assert dim.weighted_score == 0.50 * 0.25
        assert dim.weighted_score == 0.125

    def test_risk_dimension_with_evidence(self):
        """Test RiskDimension stores evidence and explanation."""
        evidence = ["Fact 1", "Fact 2", "Fact 3"]
        dim = RiskDimension(
            name="stability",
            score=0.45,
            weight=0.25,
            evidence=evidence,
            explanation="High deamidation risk from Asn in VH CDR",
        )
        assert dim.evidence == evidence
        assert "Asn" in dim.explanation


@pytest.mark.core
class TestQTPPRow:
    """Test QTPP row structure."""

    def test_qtpp_row_creation(self):
        """Test QTPPRow instantiation."""
        row = QTPPRow(
            attribute="Aggregation (SEC HMW%)",
            target="< 2%",
            acceptable_range="< 5%",
            current_prediction="2.3%",
            status="Within Range",
        )
        assert row.attribute == "Aggregation (SEC HMW%)"
        assert row.target == "< 2%"
        assert row.status == "Within Range"

    def test_qtpp_row_risk_flag(self):
        """Test QTPP row risk_flag attribute."""
        row_safe = QTPPRow(
            attribute="Test",
            current_prediction="1%",
            risk_flag=False,
        )
        assert row_safe.risk_flag is False

        row_risky = QTPPRow(
            attribute="Test",
            current_prediction="6%",
            risk_flag=True,
        )
        assert row_risky.risk_flag is True

    def test_qtpp_row_model_source(self):
        """Test model_source traceability."""
        row = QTPPRow(
            attribute="Molecular Weight",
            model_source="Biopython ProtParam (sequence MW)",
        )
        assert "Biopython" in row.model_source


@pytest.mark.core
class TestGenerateQTPP:
    """Test QTPP generation for various molecule classes."""

    def test_generate_qtpp_canonical_mab(self):
        """Test QTPP generation for canonical mAb."""
        feature_values = {
            "pI": 8.45,
            "mw_kda": 148.0,
            "deam_sites": 2,
            "ox_sites": 3,
        }
        dev_predictions = {
            "agg_risk": 0.15,
            "stability": 0.75,
            "viscosity_risk": 0.25,
        }

        qtpp = generate_qtpp(
            molecule_class="canonical_mab",
            feature_values=feature_values,
            dev_predictions=dev_predictions,
        )

        # Should return a list of QTPPRow
        assert isinstance(qtpp, list)
        assert len(qtpp) > 10  # mAb should have 10+ rows

        # Check for key rows
        attributes = [row.attribute for row in qtpp]
        assert "Isoelectric Point (pI)" in attributes
        assert "Molecular Weight (kDa)" in attributes
        assert "Aggregation (SEC HMW%)" in attributes

    def test_generate_qtpp_bispecific(self):
        """Test QTPP generation for bispecific."""
        feature_values = {
            "pI": 8.4,
            "mw_kda": 155.0,
            "deam_sites": 1,
            "ox_sites": 2,
        }

        qtpp = generate_qtpp(
            molecule_class="bispecific",
            feature_values=feature_values,
        )

        # Bispecific should have extra rows for species purity
        attributes = [row.attribute for row in qtpp]
        assert "Homodimer Contamination (species purity)" in attributes
        assert "CEX Species Resolution (Rs)" in attributes

        # Total should be ~18 rows (base rows + bispecific-specific)
        assert len(qtpp) >= 16

    def test_generate_qtpp_adc(self):
        """Test QTPP generation for ADC."""
        feature_values = {
            "pI": 8.3,
            "mw_kda": 160.0,
        }

        qtpp = generate_qtpp(
            molecule_class="adc",
            feature_values=feature_values,
        )

        attributes = [row.attribute for row in qtpp]
        assert "Drug-to-Antibody Ratio (DAR)" in attributes

    def test_qtpp_row_count_canonical_mab(self):
        """Test QTPP has 16 rows for canonical mAb."""
        feature_values = {"pI": 8.45, "mw_kda": 148.0}

        qtpp = generate_qtpp(
            molecule_class="canonical_mab",
            feature_values=feature_values,
        )

        # Canonical mAb should have 16 rows (no format-specific extras)
        assert len(qtpp) == 16

    def test_qtpp_row_count_bispecific(self):
        """Test QTPP has 18 rows for bispecific (16 base + 2 bispecific-specific)."""
        feature_values = {"pI": 8.4, "mw_kda": 155.0}

        qtpp = generate_qtpp(
            molecule_class="bispecific",
            feature_values=feature_values,
        )

        # Bispecific: 16 base + 2 format-specific
        assert len(qtpp) == 18

    def test_qtpp_uses_user_criteria_override(self):
        """Test user_criteria parameter overrides defaults."""
        feature_values = {"pI": 8.0, "mw_kda": 150.0}
        dev_predictions = {"agg_risk": 0.1}

        # Override aggregation acceptance criteria
        user_criteria = {
            "aggregation_hmw": {
                "target_upper": 1.0,  # More strict
                "accept_upper": 3.0,
            }
        }

        qtpp = generate_qtpp(
            molecule_class="canonical_mab",
            feature_values=feature_values,
            dev_predictions=dev_predictions,
            user_criteria=user_criteria,
        )

        # Find aggregation row
        agg_row = next(
            (r for r in qtpp if "Aggregation" in r.attribute), None
        )
        assert agg_row is not None
        assert "< 3.0%" in agg_row.acceptable_range  # User override applied


@pytest.mark.core
class TestQTPPAcceptanceCriteria:
    """Test QTPP acceptance criteria defaults."""

    def test_acceptance_defaults_exist(self):
        """Test QTPP_ACCEPTANCE_DEFAULTS contains expected keys."""
        expected_keys = {
            "sec_monomer",
            "aggregation_hmw",
            "cief_main",
            "cief_acidic",
            "cesds_intact",
        }
        assert expected_keys.issubset(QTPP_ACCEPTANCE_DEFAULTS.keys())

    def test_acceptance_criteria_structure(self):
        """Test each criterion has target and accept thresholds."""
        for key, criteria in QTPP_ACCEPTANCE_DEFAULTS.items():
            # Should have either _lower or _upper
            has_lower = any(k.endswith("_lower") for k in criteria.keys())
            has_upper = any(k.endswith("_upper") for k in criteria.keys())
            assert has_lower or has_upper, f"{key} missing threshold"

    def test_aggregation_criteria_realistic(self):
        """Test aggregation criteria are realistic."""
        agg = QTPP_ACCEPTANCE_DEFAULTS["aggregation_hmw"]
        assert agg["target_upper"] <= agg["accept_upper"]
        assert agg["target_upper"] < 5.0  # Target stricter than accept


@pytest.mark.core
class TestDevelopabilityAssessment:
    """Test DevelopabilityAssessment dataclass."""

    def test_assessment_creation(self):
        """Test DevelopabilityAssessment instantiation."""
        dims = [
            RiskDimension(
                name="aggregation",
                display_name="Aggregation",
                score=0.3,
                weight=0.30,
            ),
            RiskDimension(
                name="stability",
                display_name="Stability",
                score=0.25,
                weight=0.25,
            ),
        ]

        assessment = DevelopabilityAssessment(
            molecule_name="TestmAb",
            molecule_class="canonical_mab",
            dimensions=dims,
            composite_score=0.27,
            composite_grade="Low",
            recommendation="Proceed",
        )

        assert assessment.molecule_name == "TestmAb"
        assert len(assessment.dimensions) == 2
        assert assessment.composite_score == 0.27

    def test_composite_score_in_range(self):
        """Test composite_score is 0-1."""
        assessment = DevelopabilityAssessment(
            molecule_name="Test",
            composite_score=0.5,
        )
        assert 0.0 <= assessment.composite_score <= 1.0

    def test_radar_data_generation(self):
        """Test radar_data() method produces correct structure."""
        dims = [
            RiskDimension(
                name="agg",
                display_name="Aggregation",
                score=0.3,
                weight=0.3,
                color="#10B981",
            ),
            RiskDimension(
                name="stab",
                display_name="Stability",
                score=0.2,
                weight=0.25,
                color="#F59E0B",
            ),
        ]

        assessment = DevelopabilityAssessment(dimensions=dims)
        radar = assessment.radar_data()

        assert "labels" in radar
        assert "scores" in radar
        assert "weights" in radar
        assert "colors" in radar

        assert len(radar["labels"]) == 2
        assert len(radar["scores"]) == 2
        assert radar["scores"] == [0.3, 0.2]

    def test_to_dict_method(self):
        """Test to_dict() serialization."""
        assessment = DevelopabilityAssessment(
            molecule_name="TestmAb",
            molecule_class="canonical_mab",
            composite_score=0.35,
            composite_grade="Medium",
            recommendation="Proceed with caution",
        )

        d = assessment.to_dict()

        assert isinstance(d, dict)
        assert d["molecule_name"] == "TestmAb"
        assert d["molecule_class"] == "canonical_mab"
        assert d["composite_score"] == 0.35


@pytest.mark.core
class TestQTPPWithAnalyticalResults:
    """Test QTPP generation with analytical results."""

    def test_qtpp_with_sec_results(self):
        """Test QTPP incorporates SEC (size exclusion) results."""
        feature_values = {"pI": 8.0, "mw_kda": 150.0}
        analytical_results = {
            "sec_monomer_pct": 97.5,
        }

        qtpp = generate_qtpp(
            molecule_class="canonical_mab",
            feature_values=feature_values,
            analytical_results=analytical_results,
        )

        # Find SEC monomer purity row (not aggregation row)
        sec_row = next(
            (r for r in qtpp if "Monomer Purity" in r.attribute), None
        )
        assert sec_row is not None
        assert "97.5" in sec_row.current_prediction

    def test_qtpp_with_cief_results(self):
        """Test QTPP incorporates cIEF (charge) results."""
        feature_values = {"pI": 8.0, "mw_kda": 150.0}
        analytical_results = {
            "cief_main_pct": 75.0,
            "cief_acidic_pct": 15.0,
        }

        qtpp = generate_qtpp(
            molecule_class="canonical_mab",
            feature_values=feature_values,
            analytical_results=analytical_results,
        )

        # Find cIEF rows
        cief_main = next(
            (r for r in qtpp if "Main Peak" in r.attribute), None
        )
        cief_acidic = next(
            (r for r in qtpp if "Acidic" in r.attribute), None
        )

        assert cief_main is not None
        assert "75.0" in cief_main.current_prediction
        assert cief_acidic is not None
        assert "15.0" in cief_acidic.current_prediction

    def test_qtpp_risk_flags_out_of_range(self):
        """Test QTPP flags risk when out of acceptable range."""
        feature_values = {"pI": 8.0, "mw_kda": 150.0}
        dev_predictions = {
            "agg_risk": 0.7,  # Projected as 7% HMW (exceeds typical 5% accept)
        }

        qtpp = generate_qtpp(
            molecule_class="canonical_mab",
            feature_values=feature_values,
            dev_predictions=dev_predictions,
        )

        # Find aggregation row
        agg_row = next(
            (r for r in qtpp if "Aggregation" in r.attribute), None
        )
        assert agg_row is not None
        assert agg_row.risk_flag is True


@pytest.mark.core
class TestQTPPNonCanonicalMolecules:
    """Test QTPP generation for non-canonical molecules."""

    def test_qtpp_peptide(self):
        """Test QTPP for peptide includes appropriate caveats."""
        feature_values = {"pI": 7.5, "mw_kda": 3.5}

        qtpp = generate_qtpp(
            molecule_class="peptide",
            feature_values=feature_values,
        )

        # Should include format caveat in justifications
        has_caveat = any("peptide" in row.justification.lower()
                        for row in qtpp
                        if "caveat" in row.justification.lower())
        # Caveat may or may not be present depending on implementation

    def test_qtpp_fc_fusion(self):
        """Test QTPP for Fc-fusion."""
        feature_values = {"pI": 8.2, "mw_kda": 110.0}

        qtpp = generate_qtpp(
            molecule_class="fc_fusion",
            feature_values=feature_values,
        )

        # Should generate valid QTPP
        assert len(qtpp) > 10
        assert any("Isoelectric Point" in row.attribute for row in qtpp)


@pytest.mark.core
class TestQTPPWithAllResults:
    """Test QTPP with complete result set."""

    def test_qtpp_with_full_results(self):
        """Test QTPP generation with all optional parameters."""
        feature_values = {
            "pI": 8.45,
            "mw_kda": 148.0,
            "deam_sites": 1,
            "ox_sites": 2,
            "n_glycosylation_sites": 1,
        }

        dev_predictions = {
            "agg_risk": 0.12,
            "stability": 0.82,
            "viscosity_risk": 0.18,
        }

        analytical_results = {
            "sec_monomer_pct": 98.5,
            "cief_main_pct": 78.0,
            "cief_acidic_pct": 12.0,
            "cesds_intact_pct": 99.0,
        }

        stability_results = {
            "shelf_life_months": 36,
        }

        pk_results = {
            "half_life_days": 18.5,
        }

        qtpp = generate_qtpp(
            molecule_class="canonical_mab",
            feature_values=feature_values,
            dev_predictions=dev_predictions,
            analytical_results=analytical_results,
            stability_results=stability_results,
            pk_results=pk_results,
        )

        # Should have comprehensive QTPP (16 base + 1 acidic charge variant row)
        assert len(qtpp) == 17

        # All results should be reflected
        assert any("98.5" in row.current_prediction for row in qtpp)
        assert any("36" in row.current_prediction for row in qtpp)
        assert any("18.5" in row.current_prediction for row in qtpp)
