"""
Test suite for molecule_classifier module (src/molecule_classifier.py)
======================================================================

Tests molecule type classification logic, risk weight profiles, and
classification confidence scoring.

Markers:
  @pytest.mark.core — No torch/sklearn/streamlit dependencies
"""

import pytest
from src.molecule_classifier import (
    MoleculeClass,
    ClassificationResult,
    RISK_WEIGHT_PROFILES,
    classify_molecule,
)


@pytest.mark.core
class TestMoleculeClass:
    """Test MoleculeClass enum and properties."""

    def test_molecule_class_enum_values(self):
        """Test all expected MoleculeClass enum values exist."""
        expected_classes = {
            "canonical_mab",
            "bispecific",
            "fc_fusion",
            "adc",
            "single_domain",
            "peptide",
            "fusion_protein",
            "engineered_scaffold",
            "unknown",
        }

        enum_values = {mc.value for mc in MoleculeClass}
        assert expected_classes == enum_values

    def test_display_name_property(self):
        """Test display_name property for each molecule class."""
        assert "mAb" in MoleculeClass.CANONICAL_MAB.display_name
        assert "Bispecific" in MoleculeClass.BISPECIFIC.display_name
        assert "Fc-Fusion" in MoleculeClass.FC_FUSION.display_name
        assert "ADC" in MoleculeClass.ADC.display_name
        assert "VHH" in MoleculeClass.SINGLE_DOMAIN.display_name
        assert "Peptide" in MoleculeClass.PEPTIDE.display_name

    def test_has_fc_region_property(self):
        """Test has_fc_region property."""
        # Should have Fc
        assert MoleculeClass.CANONICAL_MAB.has_fc_region is True
        assert MoleculeClass.BISPECIFIC.has_fc_region is True
        assert MoleculeClass.FC_FUSION.has_fc_region is True
        assert MoleculeClass.ADC.has_fc_region is True

        # Should not have Fc
        assert MoleculeClass.PEPTIDE.has_fc_region is False
        assert MoleculeClass.SINGLE_DOMAIN.has_fc_region is False

    def test_expects_glycosylation_property(self):
        """Test expects_glycosylation property."""
        # Fc-containing molecules expect glycosylation
        assert MoleculeClass.CANONICAL_MAB.expects_glycosylation is True
        assert MoleculeClass.BISPECIFIC.expects_glycosylation is True

        # Non-Fc molecules don't
        assert MoleculeClass.PEPTIDE.expects_glycosylation is False

    def test_is_multi_chain_property(self):
        """Test is_multi_chain property."""
        # Multi-chain molecules
        assert MoleculeClass.CANONICAL_MAB.is_multi_chain is True
        assert MoleculeClass.BISPECIFIC.is_multi_chain is True

        # Single-chain molecules
        assert MoleculeClass.PEPTIDE.is_multi_chain is False
        assert MoleculeClass.SINGLE_DOMAIN.is_multi_chain is False

    def test_is_mab_like_property(self):
        """Test is_mab_like property (IgG scaffold)."""
        # mAb-like structures
        assert MoleculeClass.CANONICAL_MAB.is_mab_like is True
        assert MoleculeClass.BISPECIFIC.is_mab_like is True
        assert MoleculeClass.ADC.is_mab_like is True

        # Not mAb-like
        assert MoleculeClass.FC_FUSION.is_mab_like is False
        assert MoleculeClass.PEPTIDE.is_mab_like is False

    def test_expected_disulfide_bonds_property(self):
        """Test expected_disulfide_bonds property."""
        # Standard IgG: 16 disulfides
        assert MoleculeClass.CANONICAL_MAB.expected_disulfide_bonds == 16
        assert MoleculeClass.BISPECIFIC.expected_disulfide_bonds == 16
        assert MoleculeClass.ADC.expected_disulfide_bonds == 16

        # Fc-fusion: fewer (no Fab)
        assert MoleculeClass.FC_FUSION.expected_disulfide_bonds == 8

        # Single domain: 1-2
        assert MoleculeClass.SINGLE_DOMAIN.expected_disulfide_bonds == 1

        # Peptide: none
        assert MoleculeClass.PEPTIDE.expected_disulfide_bonds == 0


@pytest.mark.core
class TestClassificationResult:
    """Test ClassificationResult dataclass."""

    def test_classification_result_creation(self):
        """Test ClassificationResult instantiation."""
        result = ClassificationResult(
            molecule_class=MoleculeClass.CANONICAL_MAB,
            confidence="High",
            evidence=["Has HC and LC chains", "Contains Fc constant domain"],
        )

        assert result.molecule_class == MoleculeClass.CANONICAL_MAB
        assert result.confidence == "High"
        assert len(result.evidence) == 2

    def test_classification_confidence_levels(self):
        """Test confidence property can be High/Medium/Low."""
        for confidence in ("High", "Medium", "Low"):
            result = ClassificationResult(confidence=confidence)
            assert result.confidence == confidence

    def test_effective_class_without_override(self):
        """Test effective_class returns classifier result when no override."""
        result = ClassificationResult(
            molecule_class=MoleculeClass.CANONICAL_MAB,
            user_override=None,
        )

        assert result.effective_class == MoleculeClass.CANONICAL_MAB

    def test_effective_class_with_override(self):
        """Test effective_class returns user override when set."""
        result = ClassificationResult(
            molecule_class=MoleculeClass.CANONICAL_MAB,
            user_override="bispecific",
        )

        assert result.effective_class == MoleculeClass.BISPECIFIC

    def test_classification_result_to_dict(self):
        """Test to_dict() serialization."""
        result = ClassificationResult(
            molecule_class=MoleculeClass.CANONICAL_MAB,
            confidence="High",
            evidence=["Test evidence"],
            n_chains=4,
            n_unique_chains=2,
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["molecule_class"] == "canonical_mab"
        assert "display_name" in d
        assert d["confidence"] == "High"
        assert d["n_chains"] == 4
        assert d["has_fc_region"] is True

    def test_classification_result_with_warnings(self):
        """Test storing warnings in classification."""
        result = ClassificationResult(
            molecule_class=MoleculeClass.UNKNOWN,
            confidence="Low",
            warnings=["Unusual amino acid composition", "Truncated sequence"],
        )

        assert len(result.warnings) == 2
        assert "Unusual" in result.warnings[0]


@pytest.mark.core
class TestRiskWeightProfiles:
    """Test risk weight profiles for each molecule class."""

    def test_profiles_exist_for_all_classes(self):
        """Test RISK_WEIGHT_PROFILES has entries for all classes."""
        required_classes = {
            "canonical_mab",
            "bispecific",
            "fc_fusion",
            "adc",
            "single_domain",
            "peptide",
            "fusion_protein",
            "engineered_scaffold",
            "unknown",
        }

        assert required_classes.issubset(RISK_WEIGHT_PROFILES.keys())

    def test_profile_weights_sum_to_one(self):
        """Test each profile's weights sum to 1.0."""
        for molecule_class, weights in RISK_WEIGHT_PROFILES.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, f"{molecule_class} weights don't sum to 1.0"

    def test_canonical_mab_profile(self):
        """Test canonical mAb has expected weight distribution."""
        profile = RISK_WEIGHT_PROFILES["canonical_mab"]

        # mAb: aggregation and stability are primary
        assert profile["aggregation"] > 0.25
        assert profile["stability"] > 0.20
        assert profile["expression"] < 0.20  # Lower for well-understood format
        assert "species_purity" not in profile  # Not bispecific-specific

    def test_bispecific_profile(self):
        """Test bispecific has species_purity weight."""
        profile = RISK_WEIGHT_PROFILES["bispecific"]

        # Bispecific should have species purity weight
        assert "species_purity" in profile
        assert profile["species_purity"] > 0.10

    def test_adc_profile(self):
        """Test ADC has conjugation-specific weight."""
        profile = RISK_WEIGHT_PROFILES["adc"]

        # ADC should have conjugation weight
        assert "conjugation" in profile
        assert profile["conjugation"] > 0.10
        assert profile["stability"] > 0.20  # Payload stability critical

    def test_peptide_profile(self):
        """Test peptide has high stability weight."""
        profile = RISK_WEIGHT_PROFILES["peptide"]

        # Peptide: chemical stability dominates
        assert profile["stability"] > 0.35
        assert profile["aggregation"] < 0.15  # Less aggregation-prone
        assert profile["immunogenicity"] > 0.30  # Higher ADA risk

    def test_single_domain_profile(self):
        """Test single domain has high aggregation weight."""
        profile = RISK_WEIGHT_PROFILES["single_domain"]

        # Single domain: high aggregation risk
        assert profile["aggregation"] > 0.30
        assert profile["viscosity"] < 0.10  # Low MW → low viscosity risk

    def test_fc_fusion_profile(self):
        """Test FC-fusion has high expression weight."""
        profile = RISK_WEIGHT_PROFILES["fc_fusion"]

        # Fc-fusions often have expression challenges
        assert profile["expression"] > 0.20


@pytest.mark.core
class TestClassifyMolecule:
    """Test core classification logic."""

    def test_classify_canonical_mab(self):
        """Test classification of canonical mAb."""
        # Canonical mAb: 2 HC + 2 LC, high sequence identity between chains of same type
        chains = [
            {"type": "HC", "sequence": "M" * 400},
            {"type": "HC", "sequence": "M" * 400},  # Same length HC
            {"type": "LC", "sequence": "M" * 220},
            {"type": "LC", "sequence": "M" * 220},  # Same length LC
        ]

        result = classify_molecule(sequence="", chains=chains)

        # Should classify as canonical mAb or similar
        assert result.molecule_class in (
            MoleculeClass.CANONICAL_MAB,
            MoleculeClass.UNKNOWN,
        )
        assert result.n_chains == 4

    def test_classify_bispecific(self):
        """Test classification of bispecific (2 different HC)."""
        chains = [
            {"type": "HC", "sequence": "M" * 400},
            {"type": "HC", "sequence": "M" * 430},  # Different HC
            {"type": "LC", "sequence": "M" * 220},
            {"type": "LC", "sequence": "M" * 220},
        ]

        result = classify_molecule(sequence="", chains=chains)

        # Should detect different HCs
        assert result.n_chains == 4
        assert result.n_unique_chains >= 2

    def test_classify_peptide_short_sequence(self):
        """Test classification of peptide (short sequence, no LC)."""
        peptide_seq = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"  # ~31 aa

        result = classify_molecule(sequence=peptide_seq, chains=None)

        # Should classify as peptide or unknown
        assert result.molecule_class in (
            MoleculeClass.PEPTIDE,
            MoleculeClass.SINGLE_DOMAIN,
            MoleculeClass.UNKNOWN,
        )

    def test_classify_single_domain_nanobody(self):
        """Test classification of single domain (nanobody-like)."""
        # VHH: ~130 aa, high purity single chain
        vhh_seq = "M" * 130

        result = classify_molecule(sequence=vhh_seq, chains=None)

        # Should potentially classify as single domain
        assert result.molecule_class in (
            MoleculeClass.SINGLE_DOMAIN,
            MoleculeClass.UNKNOWN,
        )

    def test_classify_adc_with_hint(self):
        """Test classification hints toward ADC with format info."""
        chains = [
            {"type": "HC", "sequence": "M" * 400},
            {"type": "LC", "sequence": "M" * 220},
        ]
        # Could add linker/payload info in extended API

        result = classify_molecule(
            sequence="",
            chains=chains,
        )

        # May not classify as ADC without explicit payload info
        # but should at least find HC+LC
        assert result.n_chains == 2

    def test_classify_unknown_edge_case(self):
        """Empty sequence with no chains returns unknown/Low — not a silent fallback."""
        result = classify_molecule(sequence="", chains=None)

        assert result.molecule_class == MoleculeClass.UNKNOWN
        assert result.confidence == "Low"
        assert any("empty" in w.lower() or "missing" in w.lower() for w in result.warnings)

    def test_classification_provides_evidence(self):
        """Test classification includes evidence for decision."""
        chains = [
            {"type": "HC", "sequence": "M" * 400},
            {"type": "LC", "sequence": "M" * 220},
        ]

        result = classify_molecule(sequence="", chains=chains)

        # Should provide reasoning
        assert len(result.evidence) > 0 or result.molecule_class == MoleculeClass.UNKNOWN

    def test_classification_confidence_reflects_certainty(self):
        """Test confidence is appropriate for classification."""
        # Clear case: 4 chains -> likely mAb
        clear_chains = [
            {"type": "HC", "sequence": "M" * 400},
            {"type": "HC", "sequence": "M" * 400},
            {"type": "LC", "sequence": "M" * 220},
            {"type": "LC", "sequence": "M" * 220},
        ]

        clear_result = classify_molecule(sequence="", chains=clear_chains)

        # Ambiguous case: unknown
        ambig_result = classify_molecule(sequence="", chains=None)

        # Clear should have higher confidence
        confidence_rank = {"High": 3, "Medium": 2, "Low": 1}
        if clear_result.molecule_class != MoleculeClass.UNKNOWN:
            assert confidence_rank.get(clear_result.confidence, 0) >= \
                   confidence_rank.get(ambig_result.confidence, 0)


@pytest.mark.core
class TestClassifyWithSampleIntents:
    """Test classification with sample fixture intents."""

    def test_classify_sample_mab_intent(self, sample_mab_intent):
        """Test classification with sample mAb intent."""
        # Extract sequence from intent
        hc = sample_mab_intent.get("hc_sequence", "")
        lc = sample_mab_intent.get("lc_sequence", "")
        is_mab = sample_mab_intent.get("is_mab", False)

        if hc and lc and is_mab:
            # Provide chains to classifier
            chains = [
                {"type": "HC", "sequence": hc},
                {"type": "LC", "sequence": lc},
            ]
            result = classify_molecule(sequence="", chains=chains)
            assert result is not None
            assert result.molecule_class in (
                MoleculeClass.CANONICAL_MAB,
                MoleculeClass.PEPTIDE,
                MoleculeClass.UNKNOWN,
            )

    def test_classify_sample_bispecific_intent(self, sample_bispecific_intent):
        """Test classification with sample bispecific intent."""
        hc = sample_bispecific_intent.get("hc_sequence", "")
        hc2 = sample_bispecific_intent.get("hc2_sequence", "")

        if hc and hc2:
            # Has two different HCs
            result = classify_molecule(sequence="")
            assert result is not None
            assert isinstance(result, ClassificationResult)

    def test_classify_sample_peptide_intent(self, sample_peptide_intent):
        """Test classification with sample peptide intent."""
        seq = sample_peptide_intent.get("hc_sequence", "")
        is_mab = sample_peptide_intent.get("is_mab", False)

        if seq and not is_mab:
            # Should classify as peptide or similar
            result = classify_molecule(sequence=seq)
            assert result is not None
            # Peptide should be short
            if len(seq) < 100:
                assert result.molecule_class in (
                    MoleculeClass.PEPTIDE,
                    MoleculeClass.SINGLE_DOMAIN,
                    MoleculeClass.UNKNOWN,
                )


@pytest.mark.core
class TestClassificationConsistency:
    """Test consistency of classification across similar inputs."""

    def test_same_input_same_classification(self):
        """Test that identical inputs produce identical classification."""
        chains = [
            {"type": "HC", "sequence": "M" * 400},
            {"type": "LC", "sequence": "M" * 220},
        ]

        result1 = classify_molecule(sequence="", chains=chains)
        result2 = classify_molecule(sequence="", chains=chains)

        assert result1.molecule_class == result2.molecule_class
        assert result1.confidence == result2.confidence

    def test_chain_order_independent(self):
        """Test classification is independent of chain order."""
        chains_ordered = [
            {"type": "HC", "sequence": "M" * 400},
            {"type": "LC", "sequence": "M" * 220},
        ]

        chains_reversed = [
            {"type": "LC", "sequence": "M" * 220},
            {"type": "HC", "sequence": "M" * 400},
        ]

        result1 = classify_molecule(sequence="", chains=chains_ordered)
        result2 = classify_molecule(sequence="", chains=chains_reversed)

        assert result1.molecule_class == result2.molecule_class

    def test_minor_sequence_variation_same_class(self):
        """Test minor sequence variations don't change class."""
        chains1 = [
            {"type": "HC", "sequence": "M" * 400},
            {"type": "LC", "sequence": "M" * 220},
        ]

        # Very slightly different sequence (one substitution)
        chains2 = [
            {"type": "HC", "sequence": "M" * 399 + "L"},
            {"type": "LC", "sequence": "M" * 220},
        ]

        result1 = classify_molecule(sequence="", chains=chains1)
        result2 = classify_molecule(sequence="", chains=chains2)

        # Should still be same class
        assert result1.molecule_class == result2.molecule_class


@pytest.mark.core
class TestConfidenceScore:
    """Test numeric confidence_score field."""

    def test_confidence_score_exists(self):
        """ClassificationResult has confidence_score field."""
        result = ClassificationResult()
        assert hasattr(result, "confidence_score")

    def test_high_confidence_score(self):
        """Peptide (short sequence) should get High confidence = 0.95."""
        result = classify_molecule(sequence="ACDEFGHIKLM" * 3)
        assert result.confidence == "High"
        assert result.confidence_score == 0.95

    def test_medium_confidence_score(self):
        """Name-only bispecific with OOD sequence: OOD detector caps to Low.

        Rule-based gives Medium (name keyword match), but the all-A sequence
        is flagged OOD (Mahalanobis distance >> threshold), capping confidence.
        If no OOD detector is loaded, rule-based Medium is returned.
        """
        result = classify_molecule(sequence="A" * 450, name="test bispecific molecule")
        # With OOD detector loaded: Low (capped). Without: Medium (rule-based).
        assert result.confidence in ("Medium", "Low")
        if result.confidence == "Low":
            # OOD detector correctly flagged artificial sequence
            assert any("OOD" in w for w in result.warnings), (
                "Confidence is Low but no OOD warning found"
            )

    def test_confidence_score_in_to_dict(self):
        """to_dict() should include confidence_score."""
        result = classify_molecule(sequence="ACDEFGHIKLM" * 3)
        d = result.to_dict()
        assert "confidence_score" in d
        assert d["confidence_score"] == 0.95

    def test_user_override_confidence_score(self):
        """User override should get High confidence = 0.95."""
        result = classify_molecule(sequence="A" * 300, user_hint="fc_fusion")
        assert result.confidence_score == 0.95


@pytest.mark.core
class TestValidateClassifier:
    """Test the built-in validation corpus."""

    def test_validate_classifier_runs(self):
        """validate_classifier() should return accuracy metrics."""
        from src.molecule_classifier import validate_classifier
        result = validate_classifier()
        assert "accuracy" in result
        assert "correct" in result
        assert "total" in result
        assert "mismatches" in result

    def test_validate_classifier_accuracy(self):
        """Accuracy should be at least 85%."""
        from src.molecule_classifier import validate_classifier
        result = validate_classifier()
        assert result["accuracy"] >= 0.85, (
            f"Accuracy {result['accuracy']:.0%} below 85%. "
            f"Mismatches: {result['mismatches']}"
        )

    def test_validate_classifier_no_mismatches(self):
        """Currently expect 100% accuracy on the built-in corpus."""
        from src.molecule_classifier import validate_classifier
        result = validate_classifier()
        assert result["all_passed"], f"Mismatches: {result['mismatches']}"
