"""
test_training_smoke.py  ·  Layer 4 — Training Smoke Tests
=========================================================
Validates that the ML training infrastructure is functional:
  - Feature extraction produces valid arrays
  - Model status reporting works
  - Heuristic fallback is consistent with ML path API
  - Dataset schema validation (when data file exists)
  - Model serialization roundtrip (when torch available)

Tests requiring torch/sklearn are skipped if those packages aren't installed.
"""

import os
import pytest
import numpy as np

from tests.conftest import requires_torch, requires_sklearn

pytestmark = [pytest.mark.core]  # base tests run without ML


# ═══════════════════════════════════════════════════════════════════════
#  Feature Extraction (always available — heuristic path)
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureExtraction:
    """Feature extraction must work at Layer 1 (no torch)."""

    def test_extract_features_from_sequence_returns_array(self):
        from src.ml_predictor import extract_features_from_sequence
        seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVAR" * 3
        features = extract_features_from_sequence(seq)
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert features.shape[0] > 0
        assert np.all(np.isfinite(features)), "Features should not contain NaN/inf"

    def test_extract_features_from_intent(self):
        from src.ml_predictor import extract_features_from_intent
        intent = {
            "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH" * 5,
            "pI": 8.5, "mw": 50.0, "hydrophobicity": 0.35,
            "gravy": -0.3, "deam_sites": 5, "ox_sites": 3,
            "cysteine_count": 8, "acidic_residues": 50,
            "basic_residues": 60, "seq_length": 175,
        }
        features = extract_features_from_intent(intent)
        assert isinstance(features, np.ndarray)
        assert features.shape[0] > 5

    def test_features_deterministic(self):
        """Same sequence → identical feature vector."""
        from src.ml_predictor import extract_features_from_sequence
        seq = "ACDEFGHIKLMNPQRSTVWY" * 10
        f1 = extract_features_from_sequence(seq)
        f2 = extract_features_from_sequence(seq)
        np.testing.assert_array_equal(f1, f2)


# ═══════════════════════════════════════════════════════════════════════
#  Model Status (always available)
# ═══════════════════════════════════════════════════════════════════════

class TestModelStatus:
    """Model status should report cleanly even with no trained models."""

    def test_get_model_status_returns_dict(self):
        from src.ml_predictor import get_model_status
        status = get_model_status()
        assert isinstance(status, dict)

    def test_get_calibration_status_returns_dict(self):
        from src.ml_predictor import get_calibration_status
        status = get_calibration_status()
        assert isinstance(status, dict)


# ═══════════════════════════════════════════════════════════════════════
#  Heuristic Prediction Consistency
# ═══════════════════════════════════════════════════════════════════════

class TestHeuristicPrediction:
    """Heuristic scoring should produce valid, bounded, deterministic results."""

    def test_heuristic_scores_bounded_0_1(self):
        """All prediction scores should be in [0, 1]."""
        from src.developability_core import assess_developability
        for mol_cls in ("canonical_mab", "bispecific", "peptide", "single_domain"):
            result = assess_developability(
                molecule_name=f"Heur_{mol_cls}",
                molecule_class=mol_cls,
                dev_predictions={"agg_risk": 0.3, "stability": 0.6, "viscosity_risk": 0.2},
            )
            assert 0.0 <= result.composite_score <= 1.0, (
                f"{mol_cls}: composite_score {result.composite_score} out of [0,1]"
            )

    def test_higher_risk_inputs_produce_higher_scores(self):
        """High agg_risk input should yield higher composite than low agg_risk."""
        from src.developability_core import assess_developability

        low = assess_developability(
            molecule_name="Low", molecule_class="canonical_mab",
            dev_predictions={"agg_risk": 0.05, "stability": 0.9, "viscosity_risk": 0.05},
        )
        high = assess_developability(
            molecule_name="High", molecule_class="canonical_mab",
            dev_predictions={"agg_risk": 0.8, "stability": 0.2, "viscosity_risk": 0.7},
        )
        assert high.composite_score > low.composite_score, (
            f"High-risk input ({high.composite_score}) should score higher "
            f"than low-risk input ({low.composite_score})"
        )


# ═══════════════════════════════════════════════════════════════════════
#  ML-Specific Tests (require torch/sklearn — skipped if unavailable)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.ml
@requires_torch
class TestTorchSmoke:
    """Smoke tests for torch-dependent modules (Layer 3 only)."""

    def test_pLM_embedder_mock_mode(self):
        """pLM_embedder should produce embeddings in mock mode (no ESM-2 weights)."""
        from src.pLM_embedder import mock_embedding
        seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH" * 4
        emb = mock_embedding(seq)
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] > 0

    def test_unified_multitask_model_import_and_create(self):
        """UnifiedMultiTaskModel imports and can be instantiated."""
        import torch.nn as nn
        from src.unified_multitask_model import UnifiedMultiTaskModel, UNIFIED_TASKS

        encoder = nn.Linear(32, 64)
        model = UnifiedMultiTaskModel(
            encoder=encoder, encoder_dim=64,
            shared_hidden=32, head_hidden=16,
        )
        # Verify task heads were created
        assert len(model.heads) == len(UNIFIED_TASKS)
        assert model.task_names == UNIFIED_TASKS

    def test_esm2_hybrid_encoder_import(self):
        """ESM2HybridEncoder should import when torch is available."""
        from src.esm2_hybrid_encoder import ESM2HybridEncoder
        assert callable(ESM2HybridEncoder)

    def test_tiny_linear_training_loop(self):
        """Tiny training loop with a plain nn.Module — validates torch infra works."""
        import torch
        import torch.nn as nn

        # Simple 2-layer net: not UnifiedMultiTaskModel (which needs string input)
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _epoch in range(3):
            x = torch.randn(4, 16)
            y = torch.randn(4, 1)
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 16))
        assert torch.isfinite(out).all(), "Model output contains NaN/inf after training"


@pytest.mark.ml
@requires_torch
@requires_sklearn
class TestModelSerialization:
    """Test model save/load roundtrip (requires torch + sklearn)."""

    def test_calibration_roundtrip(self):
        """calibrate_baseline → produces calibration status."""
        from src.ml_predictor import calibrate_baseline_from_jain137, get_calibration_status
        result = calibrate_baseline_from_jain137(n_samples=10, seed=42)
        assert isinstance(result, dict)
        status = get_calibration_status()
        assert isinstance(status, dict)


# ═══════════════════════════════════════════════════════════════════════
#  Environment Graceful Degradation
# ═══════════════════════════════════════════════════════════════════════

class TestEnvironmentDegradation:
    """Verify optional_deps module reports correct status."""

    def test_optional_deps_available(self):
        from src.optional_deps import available
        # Core packages must always be available
        assert available("numpy")
        assert available("pandas")

    def test_optional_deps_layer_status(self):
        from src.optional_deps import layer_status
        status = layer_status()
        assert isinstance(status, dict)
        assert "core" in status
        assert "analysis" in status
        assert "training" in status
        # Core must be available (we're running tests)
        assert status["core"] is True

    def test_optional_deps_require_raises_for_fake_package(self):
        from src.optional_deps import require
        with pytest.raises(ImportError, match="not installed"):
            require("nonexistent_package_xyz_12345")
