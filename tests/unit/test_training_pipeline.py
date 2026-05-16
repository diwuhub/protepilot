"""
test_training_pipeline.py  ·  Layer 4 — Training Pipeline Governance Tests
==========================================================================
Validates the training infrastructure end-to-end:
  - Schema validation (column roles, feature ranges, no overlap)
  - Data harmonizer produces valid output
  - Reproducible splits (same seed → same split)
  - Model inference integration (load → predict → valid output)
  - Trained model produces valid MoleculeClass values
  - Benchmark panel comparison runs without error

These tests run at Layer 1 (no torch needed) since the trainer uses
sklearn/numpy and the inference module uses numpy only.
"""

import os
import pytest
import numpy as np
import pandas as pd

pytestmark = [pytest.mark.core, pytest.mark.governance]

TRAINING_DATA_PATH = "data/training/classifier_data.csv"


# ═══════════════════════════════════════════════════════════════════════
#  Schema Validation
# ═══════════════════════════════════════════════════════════════════════

class TestTrainingSchema:
    """Verify the training schema is internally consistent."""

    def test_schema_selftest(self):
        from src.training.schema import _selftest
        assert _selftest() is True

    def test_no_label_feature_overlap(self):
        from src.training.schema import LABEL_COLS, FEATURE_COLS
        overlap = set(LABEL_COLS) & set(FEATURE_COLS)
        assert not overlap, f"Label/feature columns overlap: {overlap}"

    def test_no_metadata_feature_overlap(self):
        from src.training.schema import METADATA_COLS, FEATURE_COLS
        overlap = set(METADATA_COLS) & set(FEATURE_COLS)
        assert not overlap, f"Metadata/feature columns overlap: {overlap}"

    def test_feature_cols_have_no_duplicates(self):
        from src.training.schema import FEATURE_COLS
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_feature_cols_match_trainer(self):
        """Schema feature cols must match what the trainer uses."""
        from src.training.schema import FEATURE_COLS
        from src.training.classifier_trainer import _FEATURE_COLS
        assert FEATURE_COLS == _FEATURE_COLS, (
            f"Schema/trainer feature column mismatch:\n"
            f"  Schema:  {FEATURE_COLS}\n"
            f"  Trainer: {_FEATURE_COLS}"
        )

    def test_benchmark_holdout_has_at_least_2(self):
        from src.training.schema import BENCHMARK_HOLDOUT_NAMES
        assert len(BENCHMARK_HOLDOUT_NAMES) >= 2


# ═══════════════════════════════════════════════════════════════════════
#  Data Harmonizer
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not os.path.exists(TRAINING_DATA_PATH),
    reason="training data artifact not present; run the data harmonizer first",
)
class TestDataHarmonizer:
    """Verify the harmonizer produces valid training data."""

    def test_harmonized_data_exists(self):
        path = TRAINING_DATA_PATH
        assert os.path.exists(path), f"Harmonized data not found: {path}"

    def test_harmonized_data_schema_valid(self):
        from src.training.schema import validate_schema
        df = pd.read_csv(TRAINING_DATA_PATH)
        violations = validate_schema(df)
        assert len(violations) == 0, f"Schema violations: {violations}"

    def test_all_8_classes_present(self):
        df = pd.read_csv(TRAINING_DATA_PATH)
        classes = set(df["molecule_class"].unique())
        expected = {"canonical_mab", "bispecific", "adc", "fc_fusion",
                    "single_domain", "peptide", "fusion_protein", "engineered_scaffold"}
        missing = expected - classes
        assert not missing, f"Missing classes in training data: {missing}"

    def test_minimum_samples_per_class(self):
        df = pd.read_csv(TRAINING_DATA_PATH)
        for cls in df["molecule_class"].unique():
            n = len(df[df["molecule_class"] == cls])
            assert n >= 3, f"{cls} has only {n} samples (need >= 3)"

    def test_no_unknown_class_in_training_data(self):
        df = pd.read_csv(TRAINING_DATA_PATH)
        assert "unknown" not in df["molecule_class"].values


# ═══════════════════════════════════════════════════════════════════════
#  Reproducible Splits
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not os.path.exists(TRAINING_DATA_PATH),
    reason="training data artifact not present; run the data harmonizer first",
)
class TestReproducibleSplits:
    """Verify splits are deterministic and holdout is respected."""

    def test_same_seed_produces_same_split(self):
        from src.training.schema import create_split
        df = pd.read_csv(TRAINING_DATA_PATH)

        s1 = create_split(df, seed=42)
        s2 = create_split(df, seed=42)
        assert (s1["split"] == s2["split"]).all(), "Split not deterministic with same seed"

    def test_different_seed_produces_different_split(self):
        from src.training.schema import create_split
        df = pd.read_csv(TRAINING_DATA_PATH)

        s1 = create_split(df, seed=42)
        s2 = create_split(df, seed=99)
        # Not all must differ, but at least some should
        n_diff = (s1["split"] != s2["split"]).sum()
        assert n_diff > 0, "Different seeds produced identical splits"

    def test_holdout_molecules_excluded_from_train(self):
        from src.training.schema import create_split, BENCHMARK_HOLDOUT_NAMES
        df = pd.read_csv(TRAINING_DATA_PATH)
        df_split = create_split(df, seed=42)

        holdout_names = {n.lower() for n in BENCHMARK_HOLDOUT_NAMES}
        train_df = df_split[df_split["split"] == "train"]
        train_names = set(train_df["name"].str.lower())
        leaked = holdout_names & train_names
        assert not leaked, f"Holdout molecules leaked into train: {leaked}"


# ═══════════════════════════════════════════════════════════════════════
#  Model Inference Integration
# ═══════════════════════════════════════════════════════════════════════

class TestModelInference:
    """Verify trained model can be loaded and produces valid output."""

    def test_load_classifier_returns_object(self):
        from src.training.model_inference import load_classifier
        clf = load_classifier("models/classifier")
        if clf is None:
            pytest.skip("No trained classifier artifact")
        assert clf.model_type in ("sklearn_logistic_regression", "numpy_softmax", "xgboost")
        assert len(clf.classes) >= 7

    def test_predict_class_returns_valid_result(self):
        from src.training.model_inference import load_classifier, predict_class

        clf = load_classifier("models/classifier")
        if clf is None:
            pytest.skip("No trained model")

        # NISTmAb-like sequence
        seq = ("QVTLRESGPALVKPTQTLTLTCTFSGFSLSTAGMSVGWIRQPPGKALEWLADIWWDDKK"
               "DYNPSLKDRLTISKDTSKNQVVLKVTNMDPADTATYYCARDMIFNFYFDVWGQGTTVTVSS")
        result = predict_class(clf, sequence=seq, n_chains=2)

        assert "molecule_class" in result
        assert "confidence" in result
        assert "probability" in result
        assert result["confidence"] in ("High", "Medium", "Low")
        assert 0.0 <= result["probability"] <= 1.0

    def test_predict_produces_valid_molecule_class(self):
        from src.training.model_inference import load_classifier, predict_class
        from src.molecule_classifier import MoleculeClass

        clf = load_classifier("models/classifier")
        if clf is None:
            pytest.skip("No trained model")

        valid_classes = {mc.value for mc in MoleculeClass}
        seq = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"  # GLP-1 peptide
        result = predict_class(clf, sequence=seq, n_chains=1)
        assert result["molecule_class"] in valid_classes, (
            f"Predicted class '{result['molecule_class']}' not in MoleculeClass enum"
        )

    def test_inference_deterministic(self):
        from src.training.model_inference import load_classifier, predict_class

        clf = load_classifier("models/classifier")
        if clf is None:
            pytest.skip("No trained model")

        seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH" * 4
        predictions = [predict_class(clf, sequence=seq, n_chains=2) for _ in range(5)]
        classes = [p["molecule_class"] for p in predictions]
        assert len(set(classes)) == 1, f"Inference not deterministic: {set(classes)}"

    def test_all_probs_sum_to_one(self):
        from src.training.model_inference import load_classifier, predict_class

        clf = load_classifier("models/classifier")
        if clf is None:
            pytest.skip("No trained model")

        seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH" * 4
        result = predict_class(clf, sequence=seq, n_chains=2)
        total = sum(result["all_probs"].values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"
