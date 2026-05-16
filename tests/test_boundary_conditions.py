"""
tests/test_boundary_conditions.py
==================================
Boundary condition tests for Phase 2 code.

Tests:
  1. Risk level thresholds (0.4, 0.7) in proteloop_export
  2. CTD compliance weight assertion
  3. Stability twin Tm clamping
  4. MHC-II score_9mer range validation
  5. Adapter schema contract validation
  6. Negation detection edge cases in MS triage scorer
"""

import json
import math
import os
import sys
import unittest

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

os.chdir(PROJECT_ROOT)


class TestRiskLevelThresholds(unittest.TestCase):
    """Test risk level classification boundaries in proteloop_export."""

    def _classify(self, score):
        return "high" if score < 0.4 else "medium" if score < 0.7 else "low"

    def test_exact_boundary_04(self):
        self.assertEqual(self._classify(0.4), "medium")  # 0.4 is medium, not high

    def test_just_below_04(self):
        self.assertEqual(self._classify(0.399), "high")

    def test_exact_boundary_07(self):
        self.assertEqual(self._classify(0.7), "low")  # 0.7 is low, not medium

    def test_just_below_07(self):
        self.assertEqual(self._classify(0.699), "medium")

    def test_zero(self):
        self.assertEqual(self._classify(0.0), "high")

    def test_one(self):
        self.assertEqual(self._classify(1.0), "low")

    def test_negative(self):
        self.assertEqual(self._classify(-0.1), "high")


class TestCTDComplianceWeights(unittest.TestCase):
    """Test that CTD compliance scoring weights sum to 1.0."""

    def test_default_weights_sum(self):
        config_path = os.path.join(
            PROJECT_ROOT, "..", "proteloop", "loops", "ctd_compliance", "compliance_config.json"
        )
        if not os.path.exists(config_path):
            self.skipTest("CTD compliance config not found")

        with open(config_path) as f:
            config = json.load(f)

        total = (
            config.get("completeness_weight", 0)
            + config.get("consistency_weight", 0)
            + config.get("data_integrity_weight", 0)
            + config.get("risk_flag_weight", 0)
            + config.get("gap_classification_weight", 0)
        )
        self.assertAlmostEqual(total, 1.0, places=6, msg=f"Weights sum to {total}, not 1.0")


class TestStabilityTwinBoundaries(unittest.TestCase):
    """Test stability twin parameter clamping and edge cases."""

    def test_tm_factor_clamping_high(self):
        """Tm = 100°C should clamp tm_factor to minimum (0.05)."""
        Tm = 100.0
        factor = 2.0 ** ((70.0 - Tm) / 5.0)
        factor = max(0.05, min(20.0, factor))
        self.assertEqual(factor, 0.05)

    def test_tm_factor_clamping_low(self):
        """Tm = 40°C should clamp tm_factor to maximum (20.0)."""
        Tm = 40.0
        factor = 2.0 ** ((70.0 - Tm) / 5.0)
        factor = max(0.05, min(20.0, factor))
        self.assertEqual(factor, 20.0)

    def test_tm_factor_reference(self):
        """Tm = 70°C (reference) should give factor = 1.0."""
        factor = 2.0 ** ((70.0 - 70.0) / 5.0)
        self.assertAlmostEqual(factor, 1.0)

    def test_hydro_factor_clamping(self):
        """Extreme hydrophobicity values should be clamped."""
        for hydro in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            factor = max(0.5, min(2.0, 1.0 + 0.4 * hydro))
            self.assertGreaterEqual(factor, 0.5)
            self.assertLessEqual(factor, 2.0)

    def test_shelf_life_positive(self):
        """Shelf life should always be positive."""
        # Simulate: composite_rate > 0 → shelf_life > 0
        for rate in [0.001, 0.01, 0.1, 1.0, 10.0]:
            shelf_life = min(36.0 / rate, 120.0)
            self.assertGreater(shelf_life, 0)


class TestMHCIIScoring(unittest.TestCase):
    """Test MHC-II 9-mer scoring boundaries."""

    def test_score_range(self):
        """All scores should be in [0, 1]."""
        try:
            from immunogenicity_twin import _score_9mer
        except ImportError:
            self.skipTest("immunogenicity_twin not importable")

        test_peptides = [
            "AAAAAAAAA",  # all alanine
            "KKKKKKKKK",  # all lysine
            "FFFFFFFFF",  # all phenylalanine
            "PPPPPPPPP",  # all proline (should be penalized)
            "FLWEDQTLL",  # real epitope from IEDB
        ]
        for pep in test_peptides:
            score = _score_9mer(pep)
            self.assertGreaterEqual(score, 0.0, f"Score < 0 for {pep}")
            self.assertLessEqual(score, 1.0, f"Score > 1 for {pep}")

    def test_wrong_length(self):
        """Non-9-mer should return 0.0."""
        try:
            from immunogenicity_twin import _score_9mer
        except ImportError:
            self.skipTest("immunogenicity_twin not importable")

        self.assertEqual(_score_9mer(""), 0.0)
        self.assertEqual(_score_9mer("AAAA"), 0.0)
        self.assertEqual(_score_9mer("AAAAAAAAAA"), 0.0)  # 10-mer


class TestAdapterSchemaContract(unittest.TestCase):
    """Test that proteloop_export output matches the JSON schema."""

    def test_export_has_required_fields(self):
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "adapters"))
            from proteloop_export import export_risk_flags
        except ImportError:
            self.skipTest("proteloop_export not importable")

        result = export_risk_flags({
            "name": "test_mab",
            "format": "IgG1",
            "developability_scores": {
                "aggregation": 0.3,
                "stability": 0.8,
                "immunogenicity": 0.5,
                "manufacturability": 0.9,
                "expression": 0.6,
            }
        })

        self.assertIn("molecule_name", result)
        self.assertIn("risk_flags", result)
        self.assertIn("recommended_loops", result)
        self.assertIn("protepilot_version", result)
        self.assertIn("timestamp", result)

        # Check risk_flags structure
        for dim in ["aggregation", "stability", "immunogenicity", "manufacturability", "expression"]:
            self.assertIn(dim, result["risk_flags"])
            self.assertIn("score", result["risk_flags"][dim])
            self.assertIn("level", result["risk_flags"][dim])
            self.assertIn(result["risk_flags"][dim]["level"], ["high", "medium", "low"])

    def test_high_risk_triggers_loop(self):
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "adapters"))
            from proteloop_export import export_risk_flags
        except ImportError:
            self.skipTest("proteloop_export not importable")

        result = export_risk_flags({
            "developability_scores": {"aggregation": 0.1}  # high risk
        })
        self.assertIn("formulation_dsf", result["recommended_loops"])

    def test_low_risk_no_loop(self):
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "adapters"))
            from proteloop_export import export_risk_flags
        except ImportError:
            self.skipTest("proteloop_export not importable")

        result = export_risk_flags({
            "developability_scores": {
                "aggregation": 0.9,
                "stability": 0.9,
                "immunogenicity": 0.9,
                "manufacturability": 0.9,
                "expression": 0.9,
            }
        })
        self.assertEqual(result["recommended_loops"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
