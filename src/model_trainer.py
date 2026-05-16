"""
model_trainer.py  ·  ProtePilot — STEP 2
===========================================================
Deterministic Model Training Pipeline

Version   : 1.0
Depends   : numpy, xgboost, scikit-learn, joblib

Architecture
------------------------------------------------------------
  1. Ingest feature matrix (N, 327) + target matrix (N, T)
  2. Deterministic train/validation split (seed-controlled)
  3. Feature scaling (min-max normalization)
  4. Per-target XGBRegressor training
  5. Validation metrics: R², RMSE, MAE, feature importance
  6. Model serialization to models/baseline_model.pkl

Entry Point
------------------------------------------------------------
  trigger_model_training(dataset_path) -> dict
    Orchestrates DataCurator + ModelTrainer end-to-end.
    Returns training metrics and model path.

All operations are 100% deterministic. No LLM inference.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ProtePilot.ModelTrainer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_XGB_PARAMS = {
    "max_depth": 4,               # Shallower trees to reduce overfitting on small N
    "learning_rate": 0.08,
    "n_estimators": 150,
    "subsample": 0.8,
    "colsample_bytree": 0.3,      # Low colsample for high-dim features (327)
    "reg_alpha": 0.5,             # L1 regularization
    "reg_lambda": 2.0,            # L2 regularization
    "min_child_weight": 5,        # Prevent overfitting on small samples
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
}

MODEL_FILENAME = "baseline_model.pkl"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R², RMSE, MAE for a single target."""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    return {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}


# ---------------------------------------------------------------------------
# FeatureScaler (lightweight, self-contained)
# ---------------------------------------------------------------------------

class FeatureScaler:
    """Min-max feature scaler. Deterministic, no external deps."""

    def __init__(self):
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.min_vals = X.min(axis=0)
        self.max_vals = X.max(axis=0)
        # Avoid division by zero
        self.max_vals = np.where(
            self.max_vals == self.min_vals,
            self.min_vals + 1.0,
            self.max_vals,
        )
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return (X - self.min_vals) / (self.max_vals - self.min_vals)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    XGBoost-based model trainer for deterministic antibody property prediction.

    Trains one XGBRegressor per target on a standardized feature matrix.
    Fully reproducible via fixed seed.

    Parameters
    ----------
    feature_matrix : np.ndarray (n_samples, n_features)
    targets        : np.ndarray (n_samples, n_targets)
    target_names   : List[str], target column names
    feature_names  : Optional[List[str]], feature column names
    seed           : int, random seed (default 42)
    """

    def __init__(
        self,
        feature_matrix: np.ndarray,
        targets: np.ndarray,
        target_names: List[str],
        feature_names: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.X = np.asarray(feature_matrix, dtype=np.float32)
        self.y = np.asarray(targets, dtype=np.float32)
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)
        self.target_names = target_names
        self.feature_names = feature_names
        self.n_targets = self.y.shape[1]
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.scaler = FeatureScaler()
        self.models: List[Any] = []
        self.metrics: Dict[str, Any] = {}
        self.trained = False

        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(self, val_fraction: float = 0.2) -> Tuple:
        """
        Deterministic train/val split.

        Returns (X_train, X_val, y_train, y_val).
        """
        n = len(self.X)
        n_val = max(1, int(n * val_fraction))
        indices = self.rng.permutation(n)
        train_idx = indices[:-n_val]
        val_idx = indices[-n_val:]

        self.X_train = self.X[train_idx]
        self.X_val = self.X[val_idx]
        self.y_train = self.y[train_idx]
        self.y_val = self.y[val_idx]

        log.info("Split: %d train, %d val (%.0f%%)",
                 len(train_idx), len(val_idx), 100 * val_fraction)
        return self.X_train, self.X_val, self.y_train, self.y_val

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_xgboost(self, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train XGBoost regressors (one per target).

        Parameters
        ----------
        params : Optional XGBoost hyperparameters override

        Returns
        -------
        dict with: status, n_models, metrics (per-target), training_time
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return {"status": "error",
                    "message": "xgboost not installed. pip install xgboost"}

        t0 = time.time()

        # Split if not already done
        if self.X_train is None:
            self.split()

        # Scale features
        X_train_s = self.scaler.fit_transform(self.X_train)
        X_val_s = self.scaler.transform(self.X_val)

        xgb_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
        xgb_params["random_state"] = self.seed

        self.models = []
        self.metrics = {}

        for i, tname in enumerate(self.target_names):
            y_tr = self.y_train[:, i]
            y_va = self.y_val[:, i]

            model = XGBRegressor(**xgb_params)
            # Early stopping to prevent overfitting on small datasets
            fit_params = {
                "eval_set": [(X_val_s, y_va)],
                "verbose": False,
            }
            try:
                # XGBoost >= 1.6 supports early_stopping_rounds in fit()
                model.set_params(early_stopping_rounds=20)
            except Exception:
                pass
            model.fit(X_train_s, y_tr, **fit_params)

            y_pred_val = model.predict(X_val_s)
            y_pred_train = model.predict(X_train_s)

            val_metrics = _compute_metrics(y_va, y_pred_val)
            train_metrics = _compute_metrics(y_tr, y_pred_train)

            # Feature importance (top 20)
            importance = dict(zip(
                self.feature_names or [f"f{j}" for j in range(self.X.shape[1])],
                model.feature_importances_.tolist(),
            ))
            top_features = dict(sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )[:20])

            self.metrics[tname] = {
                "val": val_metrics,
                "train": train_metrics,
                "top_features": top_features,
            }
            self.models.append(model)

            log.info(
                "  [%s] Val R²=%.3f RMSE=%.3f | Train R²=%.3f",
                tname, val_metrics["R2"], val_metrics["RMSE"],
                train_metrics["R2"],
            )

        elapsed = time.time() - t0
        self.trained = True

        return {
            "status": "success",
            "n_models": len(self.models),
            "metrics": self.metrics,
            "training_time_s": round(elapsed, 2),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str = None) -> str:
        """
        Serialize trained models to disk via joblib.

        Parameters
        ----------
        path : Output path. Defaults to models/baseline_model.pkl

        Returns absolute path to saved file.
        """
        if not self.trained:
            raise RuntimeError("No trained models. Call train_xgboost() first.")

        import joblib

        if path is None:
            # Use ml_predictor's models dir convention
            try:
                import sys
                src_dir = os.path.dirname(os.path.abspath(__file__))
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)
                from ml_predictor import _get_models_dir
                path = os.path.join(_get_models_dir(), MODEL_FILENAME)
            except ImportError:
                models_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models",
                )
                os.makedirs(models_dir, exist_ok=True)
                path = os.path.join(models_dir, MODEL_FILENAME)

        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        artifact = {
            "models": self.models,
            "scaler": self.scaler,
            "target_names": self.target_names,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "seed": self.seed,
            "n_features": self.X.shape[1],
            "n_targets": self.n_targets,
            "n_train": len(self.X_train) if self.X_train is not None else 0,
            "n_val": len(self.X_val) if self.X_val is not None else 0,
        }

        joblib.dump(artifact, abs_path)
        size_kb = os.path.getsize(abs_path) / 1024
        log.info("Model saved: %s (%.1f KB)", abs_path, size_kb)
        return abs_path

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for new feature matrix."""
        if not self.trained:
            raise RuntimeError("No trained models. Call train_xgboost() first.")

        X_s = self.scaler.transform(np.asarray(X, dtype=np.float32))
        preds = np.column_stack([m.predict(X_s) for m in self.models])
        return preds

    def get_metrics(self) -> Dict[str, Any]:
        """Return full metrics dict."""
        return self.metrics


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def trigger_model_training(
    dataset_path: str,
    model_type: str = "xgboost",
    seed: int = 42,
    val_fraction: float = 0.2,
) -> Dict[str, Any]:
    """
    Single entry point for deterministic model training pipeline.

    Orchestrates: DataCurator → ModelTrainer → Model Serialization.
    This function is the backend 'Skill' for Agent orchestration.

    Parameters
    ----------
    dataset_path  : Path to CSV (Jain-137, TheraSAbDab, or generic)
    model_type    : "xgboost" (primary)
    seed          : Random seed for full reproducibility (default 42)
    val_fraction  : Fraction of data for validation (default 0.2)

    Returns
    -------
    dict with:
      - status: "success" or "error"
      - message: str
      - model_path: absolute path to saved model
      - n_samples, n_features, n_targets: int
      - target_names: list
      - metrics: per-target {R2, RMSE, MAE}
      - curator_time_s, training_time_s: float
    """
    import sys
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from data_curator import DataCurator

    log.info("=" * 50)
    log.info("trigger_model_training(dataset=%s, seed=%d)", dataset_path, seed)
    log.info("=" * 50)

    # 1. Data Curation
    t0 = time.time()
    curator = DataCurator(dataset_path, seed=seed)
    features = curator.curate()
    curator_time = time.time() - t0

    if features["status"] != "success":
        return {
            "status": "error",
            "message": f"Data curation failed: {features.get('message', 'unknown')}",
        }

    X = features["X"]
    y = features["y"]
    target_names = features["target_names"]
    feature_names = features["feature_names"]

    log.info("Curation: %d samples × %d features → %d targets in %.1fs",
             X.shape[0], X.shape[1], len(target_names), curator_time)

    # 2. Model Training
    if model_type != "xgboost":
        return {"status": "error",
                "message": f"Unsupported model_type: {model_type}. Use 'xgboost'."}

    trainer = ModelTrainer(
        X, y, target_names,
        feature_names=feature_names,
        seed=seed,
    )
    trainer.split(val_fraction=val_fraction)
    train_result = trainer.train_xgboost()

    if train_result["status"] != "success":
        return train_result

    # 3. Save Model
    try:
        model_path = trainer.save_model()
    except Exception as e:
        log.warning("Model save failed: %s", e)
        model_path = None

    # 4. Build response
    total_time = time.time() - t0
    return {
        "status": "success",
        "message": (
            f"Trained {len(target_names)} XGBoost models on "
            f"{X.shape[0]} samples ({X.shape[1]} features) in {total_time:.1f}s"
        ),
        "model_path": model_path,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_targets": len(target_names),
        "target_names": target_names,
        "metrics": {
            tname: train_result["metrics"][tname]["val"]
            for tname in target_names
        },
        "full_metrics": train_result["metrics"],
        "curator_time_s": round(curator_time, 2),
        "training_time_s": train_result["training_time_s"],
        "total_time_s": round(total_time, 2),
        "metadata": features["metadata"],
    }


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  ModelTrainer v1.0 — Self-Test")
    print("=" * 60)

    # Find Jain-137 CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "Jain137_Cleaned_Training_Data.csv")

    if not os.path.exists(csv_path):
        print(f"  SKIP: Jain-137 CSV not found at {csv_path}")
        import sys
        sys.exit(0)

    # Test 1: End-to-end trigger
    print("\n[1/4] End-to-end trigger_model_training...")
    result = trigger_model_training(csv_path, seed=42)
    assert result["status"] == "success", f"Training failed: {result}"
    print(f"  ✓ Status: {result['status']}")
    print(f"    Samples: {result['n_samples']}, Features: {result['n_features']}")
    print(f"    Targets: {result['target_names']}")
    print(f"    Model path: {result['model_path']}")

    # Test 2: Metrics
    print("\n[2/4] Checking metrics...")
    for tname in result["target_names"]:
        m = result["metrics"][tname]
        print(f"    {tname:20s}: R²={m['R2']:.3f}  RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}")
    print("  ✓ All targets have computed metrics")

    # Test 3: Model file exists
    print("\n[3/4] Verifying saved model...")
    assert result["model_path"] is not None, "Model path is None"
    assert os.path.exists(result["model_path"]), f"Model file not found: {result['model_path']}"
    size_kb = os.path.getsize(result["model_path"]) / 1024
    print(f"  ✓ Model saved: {result['model_path']} ({size_kb:.0f} KB)")

    # Test 4: Determinism
    print("\n[4/4] Testing determinism...")
    result2 = trigger_model_training(csv_path, seed=42)
    for tname in result["target_names"]:
        assert result["metrics"][tname]["R2"] == result2["metrics"][tname]["R2"], \
            f"Non-deterministic R² for {tname}"
    print("  ✓ Deterministic: identical metrics across runs")

    print(f"\nSelf-test: 4/4 passed")
    print(f"Total time: {result['total_time_s']}s")
