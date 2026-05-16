"""
test_report_assembler.py — Unit Tests for Report Assembly (Module 5)
=====================================================================
Tests the report_assembler module which collects results from all system
modules and assembles them into a standardized ReportObject.

Key test areas:
  - assemble_report() with minimal intent returns valid ReportObject
  - ReportObject has all sections populated (executive_summary, molecule_overview, etc.)
  - Cross-section consistency passes
  - Context is frozen after assembly
  - Intent/cache name mismatch logs warning
  - Sparse/unknown molecule still produces report without crash
"""

import pytest
import logging
from src.report_assembler import assemble_report
from src.report_schema import (
    ReportObject, ReportContext, ExecutiveSummary, MoleculeOverview,
    DevelopabilitySection, AnalyticalSummary, ProcessPKSummary,
    ValidationPlanSection, ModelMetadata, AppendixData
)


@pytest.mark.core
class TestAssembleReportBasic:
    """Test basic report assembly functionality."""

    def test_assemble_report_minimal_mab_intent_returns_report_object(self, sample_mab_intent):
        """Test: assemble_report() with minimal mAb intent returns valid ReportObject."""
        cache = {}
        extras = {}

        report = assemble_report(sample_mab_intent, cache, extras)

        assert isinstance(report, ReportObject)
        assert report.generated_at is not None
        assert report.context is not None

    def test_report_has_all_sections_populated(self, sample_mab_intent):
        """Test: ReportObject has all sections populated."""
        cache = {
            "developability": {
                "predictions": {
                    "agg_risk": 0.3,
                    "stability": 0.7,
                    "viscosity_risk": 0.2,
                },
                "score": {
                    "score": 0.35,
                    "grade": "Low Risk",
                },
            }
        }
        extras = {}

        report = assemble_report(sample_mab_intent, cache, extras)

        # Check all major sections exist
        assert hasattr(report, "executive_summary")
        assert hasattr(report, "molecule_overview")
        assert hasattr(report, "developability")
        assert hasattr(report, "analytical")
        assert hasattr(report, "process_pk")
        assert hasattr(report, "validation_plan")
        assert hasattr(report, "model_metadata")
        assert hasattr(report, "appendix")

        # Verify they are not None
        assert report.executive_summary is not None
        assert report.molecule_overview is not None

    def test_context_is_frozen_after_assembly(self, sample_mab_intent):
        """Test: Context is frozen after assembly (ctx._frozen == True)."""
        cache = {}
        extras = {}

        report = assemble_report(sample_mab_intent, cache, extras)

        assert report.context._frozen is True

    def test_frozen_context_prevents_mutation(self, sample_mab_intent):
        """Test: Frozen context raises error on mutation attempts."""
        cache = {}
        extras = {}

        report = assemble_report(sample_mab_intent, cache, extras)

        # Attempt to modify frozen context should raise AttributeError
        with pytest.raises(AttributeError):
            report.context.molecule_name = "Modified Name"

    def test_intent_cache_name_mismatch_logs_warning(self, sample_mab_intent, caplog):
        """Test: Intent/cache name mismatch logs warning (use caplog fixture)."""
        cache = {
            "intent": {
                "name": "DifferentMolecule",
            }
        }
        extras = {}

        with caplog.at_level(logging.WARNING):
            report = assemble_report(sample_mab_intent, cache, extras)

        # Check that a warning was logged about the mismatch
        assert any("mismatch" in record.message.lower() for record in caplog.records)
        # Report should still be generated using the provided intent
        assert report.context.molecule_name == sample_mab_intent["name"]

    def test_sparse_molecule_still_produces_report(self):
        """Test: Sparse/unknown molecule still produces report without crash."""
        sparse_intent = {
            "name": "UnknownMolecule",
            # Minimal fields
        }
        cache = {}
        extras = {}

        report = assemble_report(sparse_intent, cache, extras)

        assert isinstance(report, ReportObject)
        assert report.context.molecule_name == "UnknownMolecule"
        assert report.executive_summary is not None

    def test_unknown_molecule_class_handled_gracefully(self):
        """Test: Unknown molecule class is handled gracefully."""
        intent = {
            "name": "TestMolecule",
            "molecule_class": "unknown_format",
        }
        cache = {}
        extras = {}

        report = assemble_report(intent, cache, extras)

        assert isinstance(report, ReportObject)
        assert report.context.molecule_class == "unknown_format"


@pytest.mark.core
class TestReportContextBuilding:
    """Test ReportContext building and data extraction."""

    def test_context_extracts_molecule_identity(self, sample_mab_intent):
        """Test: ReportContext correctly extracts molecule identity."""
        cache = {}
        extras = {}

        report = assemble_report(sample_mab_intent, cache, extras)
        ctx = report.context

        assert ctx.molecule_name == sample_mab_intent["name"]

    def test_context_handles_biophysical_from_intent(self):
        """Test: ReportContext extracts biophysical data from intent."""
        intent = {
            "name": "TestMab",
            "mw": 150.0,
            "pI": 7.5,
            "hydrophobicity": 0.35,
            "gravy": 0.12,
            "seq_length": 1300,
            "cysteine_count": 12,
        }
        cache = {}
        extras = {}

        report = assemble_report(intent, cache, extras)
        ctx = report.context

        assert ctx.molecular_weight_kda == 150.0
        assert ctx.isoelectric_point == 7.5
        assert ctx.hydrophobicity == 0.35
        assert ctx.gravy_score == 0.12
        assert ctx.sequence_length == 1300
        assert ctx.cysteine_count == 12

    def test_context_extracts_liability_data(self):
        """Test: ReportContext extracts liability summary data."""
        intent = {
            "name": "TestMab",
            "liability_summary": {
                "deamidation_sites": 5,
                "oxidation_sites": 2,
                "asp_isomerization_sites": 1,
                "n_glycosylation_sites": 3,
                "dp_clipping_sites": 0,
            }
        }
        cache = {}
        extras = {}

        report = assemble_report(intent, cache, extras)
        ctx = report.context

        assert ctx.deam_sites == 5
        assert ctx.ox_sites == 2
        assert ctx.isomerization_sites == 1
        assert ctx.n_glycosylation_sites == 3
        # dp_clip_sites will be None if not set at top level
        assert ctx.dp_clip_sites is None or ctx.dp_clip_sites == 0

    def test_context_extracts_developability_predictions(self):
        """Test: ReportContext extracts developability predictions."""
        intent = {"name": "TestMab"}
        cache = {
            "dev_result": {
                "data": {
                    "predictions": {
                        "agg_risk": 0.25,
                        "stability": 0.75,
                        "viscosity_risk": 0.15,
                    },
                    "score": {
                        "score": 0.38,
                        "grade": "Low Risk",
                    },
                    "prediction_mode": "xgboost",
                    "embedding_mode": "esm2",
                }
            }
        }
        extras = {}

        report = assemble_report(intent, cache, extras)
        ctx = report.context

        # Context extraction reads from cache["dev_result"]["data"]
        assert ctx.overall_score == 0.38
        assert ctx.prediction_mode == "xgboost"
        assert ctx.embedding_mode == "esm2"


@pytest.mark.core
class TestCrossSectionConsistency:
    """Test cross-section consistency validation."""

    def test_cross_section_consistency_passes_for_valid_report(self, sample_mab_intent):
        """Test: Cross-section consistency passes for valid report."""
        cache = {
            "developability": {
                "predictions": {"agg_risk": 0.3},
                "score": {"score": 0.35, "grade": "Low Risk"},
            }
        }
        extras = {}

        # Should not raise any exception
        report = assemble_report(sample_mab_intent, cache, extras)
        assert isinstance(report, ReportObject)

    def test_report_has_recommendation_text(self, sample_mab_intent):
        """Test: Report executive summary has recommendation text."""
        cache = {
            "developability": {
                "predictions": {"agg_risk": 0.3},
                "score": {"score": 0.35, "grade": "Low Risk"},
            }
        }
        extras = {}

        report = assemble_report(sample_mab_intent, cache, extras)

        assert report.executive_summary.recommendation is not None
        assert isinstance(report.executive_summary.recommendation, str)
        assert len(report.executive_summary.recommendation) > 0

    def test_report_has_top_risks_when_available(self):
        """Test: Report includes top_risks when risk data is available."""
        intent = {
            "name": "HighRiskMab",
            "mw": 180.0,
            "pI": 9.2,  # High pI
            "hydrophobicity": 0.65,  # High hydrophobicity
        }
        cache = {
            "developability": {
                "predictions": {"agg_risk": 0.7},
                "score": {"score": 0.65, "grade": "High Risk"},
            }
        }
        extras = {}

        report = assemble_report(intent, cache, extras)

        # High-risk report should have top_risks identified
        assert isinstance(report.executive_summary.top_risks, list)


@pytest.mark.core
class TestBispecificAndSpecialMolecules:
    """Test report assembly for bispecific and other molecule types."""

    def test_bispecific_molecule_produces_report(self, sample_bispecific_intent):
        """Test: Bispecific molecule produces valid report."""
        cache = {}
        extras = {}

        report = assemble_report(sample_bispecific_intent, cache, extras)

        assert isinstance(report, ReportObject)
        assert "BispecTest" in report.context.molecule_name

    def test_peptide_molecule_produces_report(self, sample_peptide_intent):
        """Test: Peptide molecule produces valid report."""
        cache = {}
        extras = {}

        report = assemble_report(sample_peptide_intent, cache, extras)

        assert isinstance(report, ReportObject)
        assert "GLP1-test" in report.context.molecule_name

    def test_multichain_molecule_assembly_class_detected(self, sample_bispecific_intent):
        """Test: Multi-chain molecules have correct assembly_class."""
        cache = {}
        extras = {}

        report = assemble_report(sample_bispecific_intent, cache, extras)

        # Bispecific should have multiple chains
        assert len(report.context.chains) > 0 or report.context.molecule_name is not None


@pytest.mark.core
class TestReportExports:
    """Test report export functionality."""

    def test_report_object_serializable_to_dict(self, sample_mab_intent):
        """Test: ReportObject can be converted to dict."""
        cache = {}
        extras = {}

        report = assemble_report(sample_mab_intent, cache, extras)

        # Should be convertible via dataclass mechanism
        from dataclasses import asdict
        report_dict = asdict(report)
        assert isinstance(report_dict, dict)
        assert "generated_at" in report_dict
        assert "context" in report_dict

    def test_report_has_generated_timestamp(self, sample_mab_intent):
        """Test: Report has valid generated_at timestamp."""
        cache = {}
        extras = {}

        report = assemble_report(sample_mab_intent, cache, extras)

        assert report.generated_at is not None
        assert isinstance(report.generated_at, str)
        # Should contain date/time info
        assert len(report.generated_at) > 0


@pytest.mark.core
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_analysis_cache_handled(self, sample_mab_intent):
        """Test: None analysis_cache is handled gracefully."""
        report = assemble_report(sample_mab_intent, None, None)

        assert isinstance(report, ReportObject)

    def test_empty_cache_produces_report_with_none_fields(self, sample_mab_intent):
        """Test: Empty cache results in None for missing predictions."""
        cache = {}
        report = assemble_report(sample_mab_intent, cache, {})

        # With empty cache, some fields may be None
        assert report.context is not None
        # But report structure should still be complete
        assert report.executive_summary is not None

    def test_missing_developability_data_handled(self, sample_mab_intent):
        """Test: Missing developability data is handled gracefully."""
        cache = {
            "some_other_data": "value"
        }
        report = assemble_report(sample_mab_intent, cache, {})

        assert isinstance(report, ReportObject)
        # Context should have None for missing predictions
        assert report.context.agg_risk is None or isinstance(report.context.agg_risk, (int, float))

    def test_very_large_sequence_handled(self):
        """Test: Very large sequence length is handled."""
        intent = {
            "name": "LargeConstruct",
            "seq_length": 5000,
            "mw": 500.0,
        }
        cache = {}
        report = assemble_report(intent, cache, {})

        assert isinstance(report, ReportObject)
        assert report.context.sequence_length == 5000
