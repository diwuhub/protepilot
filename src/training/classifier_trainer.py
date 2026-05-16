"""
classifier_trainer.py — Train molecule classifier from harmonized data
======================================================================
Minimal training loop for molecule classification.

Pipeline:
    1. Load harmonized CSV (from data_harmonizer)
    2. Split into train (70%) / val (15%) / test (15%) with stratification
    3. Build feature matrix from biophysical features
    4. Train baseline model (logistic regression — no torch needed)
    5. Evaluate on val + test, produce confusion matrix
    6. Save model artifact + metadata
    7. Compare against rule-based classifier

Usage:
    from src.training.classifier_trainer import train_classifier
    result = train_classifier("data/training/classifier_data.csv")

    # Or from command line:
    python -m src.training.classifier_trainer
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.training.schema import FEATURE_COLS as _FEATURE_COLS

log = logging.getLogger("ProtePilot.Training.ClassifierTrainer")


# ═══════════════════════════════════════════════════════════════════════
#  Helper functions for versioning and hashing
# ═══════════════════════════════════════════════════════════════════════

def _dataset_hash(path: str) -> str:
    """SHA-256 of training data file, first 16 hex chars."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except FileNotFoundError:
        return "unknown"


def _get_xgboost_version() -> str:
    try:
        import xgboost
        return xgboost.__version__
    except ImportError:
        return "unknown"

# Classes we train on (must match MoleculeClass enum values)
_TARGET_CLASSES = [
    "canonical_mab", "bispecific", "adc", "fc_fusion",
    "single_domain", "peptide", "fusion_protein", "engineered_scaffold",
]

# Minimum samples per class to include in training
_MIN_CLASS_SAMPLES = 3


# ═══════════════════════════════════════════════════════════════════════
#  Result dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainingResult:
    """Captures everything about a training run."""
    model_type: str = "logistic_regression"
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_classes: int = 0
    classes: List[str] = field(default_factory=list)
    feature_cols: List[str] = field(default_factory=list)

    # Metrics
    val_accuracy: float = 0.0
    test_accuracy: float = 0.0
    val_f1_macro: float = 0.0
    test_f1_macro: float = 0.0
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[List[List[int]]] = None

    # Comparison with rule-based
    rule_based_accuracy: float = 0.0
    improvement_vs_rule: float = 0.0

    # Artifact
    artifact_path: Optional[str] = None
    metadata_path: Optional[str] = None
    training_time_s: float = 0.0
    timestamp: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "TRAINING RESULT SUMMARY",
            "=" * 60,
            f"Model:          {self.model_type}",
            f"Train/Val/Test: {self.n_train}/{self.n_val}/{self.n_test}",
            f"Classes:        {self.n_classes} ({', '.join(self.classes)})",
            f"",
            f"Val  Accuracy:  {self.val_accuracy:.3f}",
            f"Test Accuracy:  {self.test_accuracy:.3f}",
            f"Val  F1 Macro:  {self.val_f1_macro:.3f}",
            f"Test F1 Macro:  {self.test_f1_macro:.3f}",
            f"",
            f"Rule-Based Acc: {self.rule_based_accuracy:.3f}",
            f"Improvement:    {self.improvement_vs_rule:+.3f}",
            f"Training Time:  {self.training_time_s:.2f}s",
            f"Artifact:       {self.artifact_path or 'not saved'}",
            "=" * 60,
        ]
        if self.per_class_f1:
            lines.insert(-1, "Per-class F1:")
            for cls, f1 in sorted(self.per_class_f1.items()):
                lines.insert(-1, f"  {cls:25s} {f1:.3f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Core training function
# ═══════════════════════════════════════════════════════════════════════

def train_classifier(
    data_path: str = "data/training/classifier_data.csv",
    artifact_dir: str = "models/classifier",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> TrainingResult:
    """
    End-to-end molecule classifier training.

    Parameters
    ----------
    data_path : str
        Path to harmonized CSV from data_harmonizer.
    artifact_dir : str
        Where to save the model artifact.
    test_size : float
        Fraction for test split.
    val_size : float
        Fraction for validation split.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    TrainingResult
        Complete training metrics and artifact info.
    """
    result = TrainingResult(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    t0 = time.time()

    # ── 1. Load data ──
    log.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Filter to trainable classes with enough samples
    class_counts = df["molecule_class"].value_counts()
    valid_classes = [c for c in class_counts.index
                     if c in _TARGET_CLASSES and class_counts[c] >= _MIN_CLASS_SAMPLES]
    df = df[df["molecule_class"].isin(valid_classes)].copy()
    log.info("After filtering: %d rows, %d classes", len(df), len(valid_classes))

    if len(df) < 20:
        raise ValueError(f"Too few training samples ({len(df)}). Need at least 20.")

    # ── 2. Build feature matrix ──
    X = df[_FEATURE_COLS].fillna(0).values.astype(np.float64)
    y_labels = df["molecule_class"].values

    # Encode labels
    label_to_idx = {c: i for i, c in enumerate(sorted(valid_classes))}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    y = np.array([label_to_idx[c] for c in y_labels])

    result.classes = sorted(valid_classes)
    result.n_classes = len(valid_classes)
    result.feature_cols = list(_FEATURE_COLS)

    # ── 3. Stratified split: train / val / test ──
    np.random.seed(seed)

    n_test = max(int(len(df) * test_size), 1)
    n_val = max(int(len(df) * val_size), 1)
    n_train = len(df) - n_test - n_val

    # Simple stratified split
    train_idx, val_idx, test_idx = _stratified_split(y, n_train, n_val, n_test, seed)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    result.n_train = len(train_idx)
    result.n_val = len(val_idx)
    result.n_test = len(test_idx)
    log.info("Split: train=%d, val=%d, test=%d", result.n_train, result.n_val, result.n_test)

    # ── 4. Feature scaling ──
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std

    # ── 5. Train model ──
    # Try XGBoost first (uses unscaled), then LR/softmax (uses scaled)
    model, model_type = _train_model(
        X_train, y_train, result.n_classes, seed,
        X_val=X_val, y_val=y_val,
    )
    result.model_type = model_type

    # XGBoost uses unscaled features; LR/softmax need scaling
    if model_type == "xgboost":
        val_X, test_X = X_val, X_test
    else:
        # Retrain with scaled data if we fell back to LR/softmax
        model, model_type = _train_model(
            X_train_scaled, y_train, result.n_classes, seed,
        )
        val_X, test_X = X_val_scaled, X_test_scaled

    # ── 6. Evaluate ──
    val_pred = _predict(model, val_X, model_type)
    test_pred = _predict(model, test_X, model_type)

    result.val_accuracy = float(np.mean(val_pred == y_val))
    result.test_accuracy = float(np.mean(test_pred == y_test))
    result.val_f1_macro = _f1_macro(y_val, val_pred, result.n_classes)
    result.test_f1_macro = _f1_macro(y_test, test_pred, result.n_classes)

    # Per-class F1 on test set
    for cls_idx, cls_name in idx_to_label.items():
        tp = int(np.sum((test_pred == cls_idx) & (y_test == cls_idx)))
        fp = int(np.sum((test_pred == cls_idx) & (y_test != cls_idx)))
        fn = int(np.sum((test_pred != cls_idx) & (y_test == cls_idx)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        result.per_class_f1[cls_name] = round(f1, 3)

    # Confusion matrix
    cm = np.zeros((result.n_classes, result.n_classes), dtype=int)
    for true, pred in zip(y_test, test_pred):
        cm[true, pred] += 1
    result.confusion_matrix = cm.tolist()

    log.info("Val accuracy=%.3f, F1=%.3f", result.val_accuracy, result.val_f1_macro)
    log.info("Test accuracy=%.3f, F1=%.3f", result.test_accuracy, result.test_f1_macro)

    # ── 7. Compare with rule-based classifier ──
    result.rule_based_accuracy = _evaluate_rule_based(df, test_idx, label_to_idx)
    result.improvement_vs_rule = result.test_accuracy - result.rule_based_accuracy
    log.info("Rule-based accuracy=%.3f, improvement=%+.3f",
             result.rule_based_accuracy, result.improvement_vs_rule)

    # ── 8. Save artifact ──
    result.training_time_s = time.time() - t0
    try:
        _save_artifact(model, model_type, mean, std, label_to_idx, idx_to_label,
                       _FEATURE_COLS, result, artifact_dir)
        result.artifact_path = os.path.join(artifact_dir, "classifier_model.npz")
        result.metadata_path = os.path.join(artifact_dir, "classifier_metadata.json")
    except Exception as e:
        log.warning("Could not save artifact: %s", e)

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _stratified_split(y, n_train, n_val, n_test, seed):
    """Simple stratified split maintaining class proportions."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []

    for c in classes:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        n_c = len(c_idx)
        n_c_test = max(1, int(n_c * n_test / len(y)))
        n_c_val = max(1, int(n_c * n_val / len(y)))
        n_c_train = n_c - n_c_test - n_c_val

        test_idx.extend(c_idx[:n_c_test])
        val_idx.extend(c_idx[n_c_test:n_c_test + n_c_val])
        train_idx.extend(c_idx[n_c_test + n_c_val:])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def _train_model(X, y, n_classes, seed, X_val=None, y_val=None):
    """Train a model. Try XGBoost → sklearn → numpy fallback."""

    # ── Attempt 1: XGBoost (best for imbalanced multi-class) ──
    try:
        import xgboost as xgb
        # Class-balanced sample weights
        class_counts = np.bincount(y, minlength=n_classes)
        class_weights = len(y) / (n_classes * class_counts + 1e-8)
        sample_weights = np.array([class_weights[yi] for yi in y])

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            random_state=seed,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        eval_set = [(X_val, y_val)] if X_val is not None else None
        model.fit(X, y, sample_weight=sample_weights,
                  eval_set=eval_set, verbose=False)
        log.info("Trained XGBoost classifier (300 trees, depth=6)")
        return model, "xgboost"
    except ImportError:
        log.info("XGBoost not available, trying sklearn LogisticRegression")

    # ── Attempt 2: sklearn LogisticRegression ──
    try:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            max_iter=2000, random_state=seed,
            solver="lbfgs", C=1.0,
        )
        model.fit(X, y)
        return model, "sklearn_logistic_regression"
    except ImportError:
        log.info("sklearn not available, using numpy softmax regression")

    # ── Attempt 3: Numpy fallback ──
    n_features = X.shape[1]
    rng = np.random.RandomState(seed)
    W = rng.randn(n_features, n_classes) * 0.01
    b = np.zeros(n_classes)
    lr = 0.1

    for epoch in range(500):
        logits = X @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        y_oh = np.zeros((len(y), n_classes))
        y_oh[np.arange(len(y)), y] = 1.0

        grad_logits = (probs - y_oh) / len(y)
        dW = X.T @ grad_logits
        db = grad_logits.sum(axis=0)

        W -= lr * dW
        b -= lr * db

    return {"W": W, "b": b}, "numpy_softmax"


def _predict(model, X, model_type):
    """Predict class indices."""
    if model_type in ("sklearn_logistic_regression", "xgboost"):
        return model.predict(X)
    else:
        logits = X @ model["W"] + model["b"]
        return np.argmax(logits, axis=1)


def _f1_macro(y_true, y_pred, n_classes):
    """Compute macro F1 without sklearn."""
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(f1)
    return float(np.mean(f1s))


def _evaluate_rule_based(df, test_idx, label_to_idx):
    """Run the platform's rule-based classifier on test set and compare."""
    try:
        from src.molecule_classifier import classify_molecule
    except ImportError:
        log.warning("Cannot import rule-based classifier for comparison")
        return 0.0

    correct = 0
    total = 0
    for idx in test_idx:
        row = df.iloc[idx]
        hc = str(row.get("hc_sequence", ""))
        lc = str(row.get("lc_sequence", ""))
        name = str(row.get("name", ""))
        true_cls = str(row["molecule_class"])

        combined_seq = hc + lc if lc else hc
        chains = []
        if hc and len(hc) >= 20:
            chains.append({"sequence": hc, "chain_type": "HC", "copy_number": 1})
        if lc and len(lc) >= 20:
            chains.append({"sequence": lc, "chain_type": "LC", "copy_number": 1})

        result = classify_molecule(
            sequence=combined_seq,
            chains=chains if chains else None,
            name=name,
        )
        pred_cls = result.molecule_class.value if hasattr(result.molecule_class, "value") else str(result.molecule_class)

        if pred_cls == true_cls:
            correct += 1
        total += 1

    return correct / max(total, 1)


def _save_artifact(model, model_type, mean, std, label_to_idx, idx_to_label,
                   feature_cols, result, artifact_dir):
    """Save model weights and metadata."""
    os.makedirs(artifact_dir, exist_ok=True)

    # Save weights
    npz_path = os.path.join(artifact_dir, "classifier_model.npz")
    if model_type == "xgboost":
        # XGBoost: save native model + mean/std (for potential fallback)
        xgb_path = os.path.join(artifact_dir, "classifier_xgboost.json")
        model.save_model(xgb_path)
        np.savez(npz_path, mean=mean, std=std)
        log.info("Saved XGBoost model to %s", xgb_path)
    elif model_type == "sklearn_logistic_regression":
        np.savez(npz_path,
                 coef=model.coef_,
                 intercept=model.intercept_,
                 mean=mean, std=std)
    else:
        np.savez(npz_path,
                 W=model["W"],
                 b=model["b"],
                 mean=mean, std=std)

    # Save metadata (with versioning for artifact safety)
    from src.training.pipeline import PIPELINE_VERSION, _schema_hash
    meta = {
        "model_type": model_type,
        "feature_cols": feature_cols,
        "feature_schema_version": _schema_hash(),
        "pipeline_version": PIPELINE_VERSION,
        "format_version": "2.0",
        "dataset_version": _dataset_hash(data_path),
        "xgboost_version": _get_xgboost_version() if model_type == "xgboost" else None,
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        "n_classes": result.n_classes,
        "classes": result.classes,
        "test_accuracy": result.test_accuracy,
        "test_f1_macro": result.test_f1_macro,
        "val_accuracy": result.val_accuracy,
        "rule_based_accuracy": result.rule_based_accuracy,
        "improvement": result.improvement_vs_rule,
        "timestamp": result.timestamp,
        "training_time_s": result.training_time_s,
    }
    meta_path = os.path.join(artifact_dir, "classifier_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Saved model to %s, metadata to %s", npz_path, meta_path)


# ═══════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Train molecule classifier")
    parser.add_argument("--data", default="data/training/classifier_data.csv")
    parser.add_argument("--output", default="models/classifier")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Step 1: Harmonize data if not exists
    if not os.path.exists(args.data):
        log.info("Training data not found, running harmonizer...")
        from src.training.data_harmonizer import harmonize
        harmonize(data_dir="data", output_path=args.data)

    # Step 2: Train
    result = train_classifier(data_path=args.data, artifact_dir=args.output, seed=args.seed)
    print(result.summary())
