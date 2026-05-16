"""
test_regulatory_context.py — Unit Tests for Regulatory Signal Bridge
====================================================================
Tests the lightweight cross-repo bridge to reg-intel-biopharma's
Policy-Signal Classifier.

Key test areas:
  - assess_regulatory_context() returns a valid dict with required keys
  - Handles missing reg-intel-biopharma gracefully (fallback dict)
  - Empty input returns fallback
  - Valid classification returns one of the four expected classes
"""

import pytest
import sys
from unittest.mock import patch


VALID_CLASSES = {"new_requirement", "relaxation", "maintenance", "ambiguous", "unknown"}


class TestAssessRegulatoryContext:
    """Tests for the assess_regulatory_context() function."""

    def test_returns_dict_with_required_keys(self):
        """assess_regulatory_context always returns dict with signal_class and source."""
        from src.regulatory_context import assess_regulatory_context

        result = assess_regulatory_context("FDA now requires comprehensive data.")

        assert isinstance(result, dict)
        assert "signal_class" in result
        assert "source" in result
        assert result["signal_class"] in VALID_CLASSES

    def test_empty_input_returns_fallback(self):
        """Empty string input should return unknown/fallback."""
        from src.regulatory_context import assess_regulatory_context

        result = assess_regulatory_context("")

        assert result["signal_class"] == "unknown"
        assert result["source"] == "fallback"
        assert "empty" in result.get("note", "").lower()

    def test_whitespace_only_returns_fallback(self):
        """Whitespace-only input should return unknown/fallback."""
        from src.regulatory_context import assess_regulatory_context

        result = assess_regulatory_context("   \n  ")

        assert result["signal_class"] == "unknown"
        assert result["source"] == "fallback"

    def test_handles_missing_reg_intel_gracefully(self):
        """When reg-intel-biopharma is not importable, returns graceful fallback."""
        import src.regulatory_context as rc
        orig_available = rc._classifier_available
        orig_model = rc._model
        orig_vectorizer = rc._vectorizer

        try:
            # Force the module to think the classifier is unavailable
            rc._classifier_available = False
            rc._model = None
            rc._vectorizer = None

            result = rc.assess_regulatory_context(
                "FDA now requires comprehensive data."
            )

            assert result["signal_class"] == "unknown"
            assert "not available" in result.get("note", "")
            assert result["source"] == "fallback"
        finally:
            rc._classifier_available = orig_available
            rc._model = orig_model
            rc._vectorizer = orig_vectorizer

    def test_successful_classification_has_probabilities(self):
        """When the classifier is available, result includes probabilities."""
        from src.regulatory_context import assess_regulatory_context

        result = assess_regulatory_context(
            "FDA now requires comprehensive analytical similarity data "
            "using orthogonal methods."
        )

        if result["source"] == "policy_signal_classifier":
            assert "probabilities" in result
            assert isinstance(result["probabilities"], dict)
            assert len(result["probabilities"]) == 4
            # All probabilities should sum to ~1.0
            total = sum(result["probabilities"].values())
            assert 0.99 <= total <= 1.01, f"Probabilities sum to {total}"
        else:
            # Classifier not available in this environment — still valid
            assert result["signal_class"] == "unknown"


class TestRegulatoryContextInReportSchema:
    """Tests that the report schema includes the regulatory_context field."""

    def test_report_object_has_regulatory_context_field(self):
        """ReportObject should have a regulatory_context section."""
        from src.report_schema import ReportObject, RegulatoryContextSection

        report = ReportObject()
        assert hasattr(report, "regulatory_context")
        assert isinstance(report.regulatory_context, RegulatoryContextSection)

    def test_regulatory_context_section_defaults(self):
        """RegulatoryContextSection defaults should be safe/unassessed."""
        from src.report_schema import RegulatoryContextSection

        section = RegulatoryContextSection()
        assert section.signal_class == "unknown"
        assert section.assessed is False
        assert section.probabilities == {}

    def test_report_serialization_includes_regulatory_context(self):
        """ReportObject.to_dict() should include regulatory_context."""
        from src.report_schema import ReportObject

        report = ReportObject()
        d = report.to_dict()
        assert "regulatory_context" in d
        assert d["regulatory_context"]["signal_class"] == "unknown"
        assert d["regulatory_context"]["assessed"] is False
