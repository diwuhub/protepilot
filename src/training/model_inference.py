"""
model_inference.py — Load trained model and run inference at platform level
============================================================================
This module bridges the trained classifier model back into the platform.
It loads the saved artifact and provides a predict() function that returns
ClassificationResult-compatible output.

The platform's classify_molecule() can optionally call this module to get
a trained-model prediction alongside the rule-based one, enabling:
  - Side-by-side comparison during evaluation
  - Ensemble / confidence-boosted classification
  - Gradual migration from rule-based to trained

Usage:
    from src.training.model_inference import load_classifier, predict_class

    model = load_classifier("models/classifier")
    result = predict_class(model, sequence="EVQLVE...", n_chains=2)
    # result = {"molecule_class": "canonical_mab", "confidence": "High", "probability": 0.92}
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("ProtePilot.Training.Inference")


@dataclass
class TrainedClassifier:
    """Loaded model artifact ready for inference."""
    model_type: str
    coef: Optional[np.ndarray] = None       # sklearn logistic regression
    intercept: Optional[np.ndarray] = None
    W: Optional[np.ndarray] = None           # numpy softmax fallback
    b: Optional[np.ndarray] = None
    xgb_model: Optional[Any] = None         # xgboost model
    mean: Optional[np.ndarray] = None        # feature scaler
    std: Optional[np.ndarray] = None
    label_to_idx: Dict[str, int] = None
    idx_to_label: Dict[int, str] = None
    feature_cols: List[str] = None
    classes: List[str] = None
    metadata: Dict[str, Any] = None


def load_classifier(artifact_dir: str = "models/classifier") -> Optional[TrainedClassifier]:
    """
    Load a trained classifier from disk.

    Returns None if no artifact exists (graceful degradation).
    """
    model_path = os.path.join(artifact_dir, "classifier_model.npz")
    meta_path = os.path.join(artifact_dir, "classifier_metadata.json")

    if not os.path.exists(model_path):
        log.debug("No trained classifier model at %s", artifact_dir)
        return None

    if not os.path.exists(meta_path):
        log.error("Classifier metadata missing: %s. Refusing to load unversioned artifact.", meta_path)
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        # ── Strong format version validation (reject, not warn) ──
        fmt_ver = meta.get("format_version", "")
        if fmt_ver not in ("1.0", "2.0"):
            log.error("Classifier artifact format_version='%s' not supported (expected 1.0 or 2.0). "
                      "Retrain with: python -m src.training.pipeline --step train", fmt_ver)
            return None

        # ── NPZ key validation ──
        data = np.load(model_path, allow_pickle=False)
        required_keys = {"mean", "std"}
        missing_keys = required_keys - set(data.files)
        if missing_keys:
            log.error("Classifier NPZ missing required keys: %s. Retrain required.", missing_keys)
            return None

        # Warn on library version mismatch (major.minor level)
        artifact_xgb = meta.get("xgboost_version", "")
        if artifact_xgb:
            try:
                import xgboost
                current_xgb = xgboost.__version__
                if current_xgb != artifact_xgb:
                    art_mm = artifact_xgb.split(".")[:2]
                    cur_mm = current_xgb.split(".")[:2]
                    if art_mm != cur_mm:
                        log.warning("XGBoost MAJOR.MINOR mismatch: artifact=%s, current=%s. Retrain recommended.",
                                    artifact_xgb, current_xgb)
                    else:
                        log.debug("XGBoost patch version differs (artifact=%s, current=%s) — OK",
                                  artifact_xgb, current_xgb)
            except ImportError:
                pass

        clf = TrainedClassifier(
            model_type=meta["model_type"],
            mean=data["mean"],
            std=data["std"],
            label_to_idx=meta["label_to_idx"],
            idx_to_label={int(k): v for k, v in meta["idx_to_label"].items()},
            feature_cols=meta["feature_cols"],
            classes=meta["classes"],
            metadata=meta,
        )

        if clf.model_type == "xgboost":
            xgb_path = os.path.join(artifact_dir, "classifier_xgboost.json")
            try:
                import xgboost as xgb
                clf.xgb_model = xgb.XGBClassifier()
                clf.xgb_model.load_model(xgb_path)
            except (ImportError, Exception) as e:
                log.warning("XGBoost model found but cannot load: %s", e)
                return None
        elif clf.model_type == "sklearn_logistic_regression":
            clf.coef = data["coef"]
            clf.intercept = data["intercept"]
        else:
            clf.W = data["W"]
            clf.b = data["b"]

        # Version safety check: warn if feature schema drifted
        artifact_schema = meta.get("feature_schema_version", "")
        if artifact_schema:
            from src.training.pipeline import _schema_hash
            code_schema = _schema_hash()
            if artifact_schema != code_schema:
                log.warning(
                    "FEATURE SCHEMA MISMATCH: artifact=%s, code=%s. "
                    "Retrain recommended (python -m src.training.pipeline --step train).",
                    artifact_schema, code_schema,
                )

        # Dataset version traceability (informational)
        artifact_dataset = meta.get("dataset_version", "")
        if artifact_dataset:
            log.info("Classifier trained on dataset_version=%s", artifact_dataset)
        else:
            log.warning("Classifier artifact missing dataset_version — provenance unknown. "
                        "Retrain with latest pipeline for full traceability.")

        log.info("Loaded trained classifier: %s, %d classes, acc=%.3f",
                 clf.model_type, len(clf.classes), meta.get("test_accuracy", 0))
        return clf

    except Exception as e:
        log.warning("Failed to load trained classifier: %s", e)
        return None


def predict_class(
    clf: TrainedClassifier,
    sequence: str = "",
    n_chains: int = 1,
    n_unique_chains: Optional[int] = None,
    features: Optional[Dict[str, float]] = None,
    hc_sequence: str = "",
    lc_sequence: str = "",
) -> Dict[str, Any]:
    """
    Run inference with the trained classifier.

    Parameters
    ----------
    clf : TrainedClassifier
        Loaded model from load_classifier().
    sequence : str
        Amino acid sequence (used to compute features if not provided).
    n_chains : int
        Number of chains.
    features : dict, optional
        Pre-computed features. If not provided, computed from sequence.

    Returns
    -------
    dict with keys:
        molecule_class : str  — predicted class
        confidence : str      — "High" / "Medium" / "Low"
        probability : float   — softmax probability of predicted class
        all_probs : dict      — {class: probability} for all classes
    """
    if features is None:
        from src.training.features import compute_all_features
        features = compute_all_features(
            sequence, n_chains=n_chains,
            n_unique_chains=n_unique_chains if n_unique_chains is not None else n_chains,
            hc_sequence=hc_sequence, lc_sequence=lc_sequence,
        )

    # Build feature vector in the EXACT order used during training
    x = np.array([features.get(c, 0.0) for c in clf.feature_cols], dtype=np.float64)

    # Predict based on model type
    if clf.model_type == "xgboost" and clf.xgb_model is not None:
        # XGBoost: no scaling needed, direct predict
        x_2d = x.reshape(1, -1)
        pred_idx = int(clf.xgb_model.predict(x_2d)[0])
        probs = clf.xgb_model.predict_proba(x_2d)[0]
    else:
        # Scale for LR / softmax
        x_scaled = (x - clf.mean) / (clf.std + 1e-8)
        if clf.model_type == "sklearn_logistic_regression":
            logits = x_scaled @ clf.coef.T + clf.intercept
        else:
            logits = x_scaled @ clf.W + clf.b
        logits = logits.flatten()
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        pred_idx = int(np.argmax(probs))

    pred_class = clf.idx_to_label.get(pred_idx, "unknown")
    pred_prob = float(probs[pred_idx])

    # Confidence from probability
    # Thresholds from platform_config (cross-module constants)
    try:
        from src.platform_config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM
    except ImportError:
        CONFIDENCE_HIGH, CONFIDENCE_MEDIUM = 0.80, 0.50
    if pred_prob >= CONFIDENCE_HIGH:
        confidence = "High"
    elif pred_prob >= CONFIDENCE_MEDIUM:
        confidence = "Medium"
    else:
        confidence = "Low"

    all_probs = {clf.idx_to_label.get(i, f"class_{i}"): round(float(p), 4)
                 for i, p in enumerate(probs)}

    return {
        "molecule_class": pred_class,
        "confidence": confidence,
        "probability": round(pred_prob, 4),
        "all_probs": all_probs,
        "model_type": clf.model_type,
    }


def _compute_features_from_seq(
    sequence: str,
    n_chains: int = 1,
    hc_sequence: str = "",
    lc_sequence: str = "",
) -> Dict[str, float]:
    """Backward-compatible wrapper — delegates to features.compute_all_features()."""
    from src.training.features import compute_all_features
    return compute_all_features(
        sequence, n_chains=n_chains, n_unique_chains=n_chains,
        hc_sequence=hc_sequence, lc_sequence=lc_sequence,
    )
