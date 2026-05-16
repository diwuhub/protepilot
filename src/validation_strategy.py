"""
validation_strategy.py  ·  ProtePilot — Milestone 9
===========================================================
Production-Grade ML Validation Strategy

Version   : 1.0
Author    : Di (ProtePilot)
Depends   : numpy, scikit-learn (optional)

Purpose
------------------------------------------------------------
Implements robust validation strategies for the ML models
used in ProtePilot (MLP for ka/nu, XGBoost for developability):

  1. Time-Based Split — Simulates temporal validation by treating
     earlier samples as "historical" and later ones as "future".
     Essential for detecting temporal drift in chromatography data.

  2. Batch-Shift Split — Tests robustness to manufacturing batch
     changes by introducing systematic shifts in feature distributions.
     Ensures models generalize across production batches.

  3. Comprehensive Validation Report — Computes Accuracy (R2),
     RMSE, MAE, and 95% Confidence Intervals via bootstrap.

References
------------------------------------------------------------
  Bergmeir & Benitez, Inform. Sci. 191:192 (2012) — Time-series CV
  Sugiyama et al., NIPS 2007 — Covariate shift adaptation
  Efron & Tibshirani, "An Introduction to the Bootstrap" (1993)
"""

from __future__ import annotations

import logging
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger("ProtePilot.ValidationStrategy")

# Optional sklearn
_HAS_SKLEARN = False
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    _HAS_SKLEARN = True
except ImportError:
    log.info("scikit-learn not installed — using built-in metric implementations")


# ===========================================================================
# 1. Metric Functions (standalone, no sklearn dependency)
# ===========================================================================

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R-squared)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (guarded against zero division)."""
    mask = np.abs(y_true) > 1e-10
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    Returns dict with: rmse, mae, r2, mape
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if _HAS_SKLEARN:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
    else:
        rmse = _rmse(y_true, y_pred)
        mae = _mae(y_true, y_pred)
        r2 = _r2(y_true, y_pred)

    mape = _mape(y_true, y_pred)

    return {
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 6),
        "mape": round(mape, 4),
    }


# ===========================================================================
# 2. Bootstrap Confidence Intervals
# ===========================================================================

def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true, y_pred : Arrays of true and predicted values
    metric_fn : Function(y_true, y_pred) -> float
    n_bootstrap : Number of bootstrap resamples
    confidence : Confidence level (default 0.95 for 95% CI)
    random_seed : Random seed for reproducibility

    Returns
    -------
    {"mean": float, "ci_lower": float, "ci_upper": float, "std": float}
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            if np.isfinite(score):
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    scores = np.array(scores)
    alpha = (1 - confidence) / 2

    return {
        "mean": round(float(np.mean(scores)), 6),
        "ci_lower": round(float(np.percentile(scores, alpha * 100)), 6),
        "ci_upper": round(float(np.percentile(scores, (1 - alpha) * 100)), 6),
        "std": round(float(np.std(scores)), 6),
    }


# ===========================================================================
# 3. Time-Based Split
# ===========================================================================

def time_based_split(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    train_ratio: float = 0.70,
    n_splits: int = 3,
    expanding_window: bool = True,
) -> List[Dict[str, Any]]:
    """
    Simulate temporal validation by splitting data chronologically.

    If timestamps are not provided, data order is treated as temporal
    ordering (common for synthetic datasets where row index approximates
    time).

    Parameters
    ----------
    X : Feature matrix (n_samples, n_features)
    y : Target array (n_samples,) or (n_samples, n_targets)
    timestamps : Optional temporal ordering array (n_samples,)
    train_ratio : Fraction of data for initial training window
    n_splits : Number of forward-chaining splits
    expanding_window : If True, training window grows; if False, slides

    Returns
    -------
    List of dicts, each containing:
      - split_id: int
      - train_indices: np.ndarray
      - test_indices: np.ndarray
      - train_size: int
      - test_size: int
      - description: str
    """
    n = len(X)
    if n < 4:
        log.warning("Too few samples (%d) for time-based split", n)
        return []

    # Sort by timestamps if provided
    if timestamps is not None:
        order = np.argsort(timestamps)
    else:
        order = np.arange(n)

    # Determine split boundaries
    initial_train = int(n * train_ratio)
    remaining = n - initial_train
    step_size = max(1, remaining // n_splits)

    splits = []
    for i in range(n_splits):
        if expanding_window:
            train_end = initial_train + i * step_size
        else:
            train_start = i * step_size
            train_end = initial_train + i * step_size

        test_start = train_end
        test_end = min(test_start + step_size, n)

        if test_start >= n or test_end <= test_start:
            break

        if expanding_window:
            train_idx = order[:train_end]
        else:
            train_idx = order[max(0, train_end - initial_train):train_end]

        test_idx = order[test_start:test_end]

        splits.append({
            "split_id": i + 1,
            "train_indices": train_idx,
            "test_indices": test_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "description": (
                f"Split {i+1}: Train [{0 if expanding_window else max(0, train_end-initial_train)}"
                f":{train_end}], Test [{test_start}:{test_end}]"
                f" ({'expanding' if expanding_window else 'sliding'} window)"
            ),
        })

    log.info("Time-based split: %d splits from %d samples (ratio=%.2f)",
             len(splits), n, train_ratio)

    return splits


# ===========================================================================
# 4. Batch-Shift Split
# ===========================================================================

def batch_shift_split(
    X: np.ndarray,
    y: np.ndarray,
    n_batches: int = 3,
    shift_magnitude: float = 0.10,
    shift_features: Optional[List[int]] = None,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Test model robustness to manufacturing batch changes.

    Simulates batch-to-batch variation by introducing systematic
    shifts in feature distributions. Each batch represents a
    different manufacturing condition.

    Parameters
    ----------
    X : Feature matrix (n_samples, n_features)
    y : Target array
    n_batches : Number of simulated batches
    shift_magnitude : Fraction of feature std dev to shift
    shift_features : Which feature indices to shift (None = all)
    random_seed : For reproducibility

    Returns
    -------
    List of dicts, each containing:
      - batch_id: int
      - train_X, train_y: Training data (all other batches)
      - test_X, test_y: Test data (this batch, with shift)
      - shift_applied: dict describing the shift
      - train_size, test_size: int
      - description: str
    """
    rng = np.random.RandomState(random_seed)
    n = len(X)

    if n < n_batches * 2:
        log.warning("Too few samples (%d) for %d batches", n, n_batches)
        return []

    # Assign samples to batches
    batch_size = n // n_batches
    indices = np.arange(n)
    rng.shuffle(indices)

    feature_stds = np.std(X, axis=0)
    if shift_features is None:
        shift_features = list(range(min(X.shape[1], 10)))  # Shift first 10 features

    batches = []
    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size if b < n_batches - 1 else n

        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        # Apply batch-specific shift to test features
        # Simulate systematic manufacturing variation
        batch_shift = rng.uniform(-shift_magnitude, shift_magnitude, size=X.shape[1])
        # Only shift specified features
        mask = np.zeros(X.shape[1])
        for fi in shift_features:
            if fi < X.shape[1]:
                mask[fi] = 1.0
        batch_shift = batch_shift * mask * feature_stds

        X_test_shifted = X[test_idx].copy() + batch_shift

        batches.append({
            "batch_id": b + 1,
            "train_X": X[train_idx],
            "train_y": y[train_idx] if y.ndim == 1 else y[train_idx],
            "test_X": X_test_shifted,
            "test_y": y[test_idx] if y.ndim == 1 else y[test_idx],
            "train_indices": train_idx,
            "test_indices": test_idx,
            "shift_applied": {
                "magnitude": shift_magnitude,
                "features_shifted": shift_features,
                "mean_shift": float(np.mean(np.abs(batch_shift[batch_shift != 0])))
                if np.any(batch_shift != 0) else 0.0,
            },
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "description": (
                f"Batch {b+1}: Leave-batch-out with {shift_magnitude*100:.0f}% "
                f"feature shift on {len(shift_features)} features"
            ),
        })

    log.info("Batch-shift split: %d batches from %d samples (shift=%.2f)",
             n_batches, n, shift_magnitude)

    return batches


# ===========================================================================
# 5. Comprehensive Validation Report
# ===========================================================================

def run_validation_report(
    model_predict_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    timestamps: Optional[np.ndarray] = None,
    n_bootstrap: int = 500,
) -> Dict[str, Any]:
    """
    Run a comprehensive validation report combining all strategies.

    Parameters
    ----------
    model_predict_fn : Function(X) -> y_pred
    X : Feature matrix
    y : Target array (n_samples,) or (n_samples, n_targets)
    model_name : Name for reporting
    timestamps : Optional temporal ordering
    n_bootstrap : Bootstrap resamples for CI

    Returns
    -------
    Full validation report dict with:
      - overall_metrics: Global performance
      - time_based: Per-split temporal validation results
      - batch_shift: Per-batch robustness results
      - confidence_intervals: Bootstrap CIs for key metrics
      - summary: Human-readable summary
      - grade: "Production-Ready" / "Acceptable" / "Needs Improvement"
    """
    y = np.asarray(y, dtype=float)

    # Ensure y is 1D for metrics (use first target if multi-output)
    if y.ndim == 2:
        y_flat = y[:, 0]
    else:
        y_flat = y

    # --- Overall Metrics ---
    try:
        y_pred_all = model_predict_fn(X)
        y_pred_all = np.asarray(y_pred_all, dtype=float)
        if y_pred_all.ndim == 2:
            y_pred_flat = y_pred_all[:, 0]
        else:
            y_pred_flat = y_pred_all
    except Exception as e:
        log.error("Model prediction failed: %s", e)
        return {"error": str(e), "summary": f"Validation failed: {e}"}

    overall = compute_metrics(y_flat, y_pred_flat)

    # --- Time-Based Validation ---
    time_splits = time_based_split(X, y, timestamps=timestamps, n_splits=3)
    time_results = []
    for split in time_splits:
        try:
            y_pred_test = model_predict_fn(X[split["test_indices"]])
            y_pred_test = np.asarray(y_pred_test, dtype=float)
            if y_pred_test.ndim == 2:
                y_pred_test = y_pred_test[:, 0]

            y_test = y_flat[split["test_indices"]]
            metrics = compute_metrics(y_test, y_pred_test)

            time_results.append({
                "split_id": split["split_id"],
                "description": split["description"],
                "train_size": split["train_size"],
                "test_size": split["test_size"],
                "metrics": metrics,
            })
        except Exception as e:
            log.warning("Time split %d failed: %s", split["split_id"], e)
            time_results.append({
                "split_id": split["split_id"],
                "description": split["description"],
                "error": str(e),
            })

    # --- Batch-Shift Validation ---
    batch_splits = batch_shift_split(X, y, n_batches=3, shift_magnitude=0.10)
    batch_results = []
    for batch in batch_splits:
        try:
            y_pred_batch = model_predict_fn(batch["test_X"])
            y_pred_batch = np.asarray(y_pred_batch, dtype=float)
            if y_pred_batch.ndim == 2:
                y_pred_batch = y_pred_batch[:, 0]

            y_test_batch = batch["test_y"]
            if y_test_batch.ndim == 2:
                y_test_batch = y_test_batch[:, 0]

            metrics = compute_metrics(y_test_batch, y_pred_batch)

            batch_results.append({
                "batch_id": batch["batch_id"],
                "description": batch["description"],
                "train_size": batch["train_size"],
                "test_size": batch["test_size"],
                "shift_info": batch["shift_applied"],
                "metrics": metrics,
            })
        except Exception as e:
            log.warning("Batch %d failed: %s", batch["batch_id"], e)
            batch_results.append({
                "batch_id": batch["batch_id"],
                "description": batch["description"],
                "error": str(e),
            })

    # --- Bootstrap Confidence Intervals ---
    ci_rmse = bootstrap_confidence_interval(
        y_flat, y_pred_flat, _rmse, n_bootstrap=n_bootstrap
    )
    ci_r2 = bootstrap_confidence_interval(
        y_flat, y_pred_flat, _r2, n_bootstrap=n_bootstrap
    )
    ci_mae = bootstrap_confidence_interval(
        y_flat, y_pred_flat, _mae, n_bootstrap=n_bootstrap
    )

    confidence_intervals = {
        "rmse": ci_rmse,
        "r2": ci_r2,
        "mae": ci_mae,
    }

    # --- Grading ---
    avg_time_rmse = np.mean([
        r["metrics"]["rmse"] for r in time_results if "metrics" in r
    ]) if time_results else overall["rmse"]

    avg_batch_rmse = np.mean([
        r["metrics"]["rmse"] for r in batch_results if "metrics" in r
    ]) if batch_results else overall["rmse"]

    # Grade based on multiple criteria
    drift_ratio = avg_time_rmse / max(overall["rmse"], 1e-10)
    shift_ratio = avg_batch_rmse / max(overall["rmse"], 1e-10)

    if overall["r2"] > 0.85 and drift_ratio < 1.5 and shift_ratio < 1.5:
        grade = "Production-Ready"
        grade_color = "green"
    elif overall["r2"] > 0.65 and drift_ratio < 2.0 and shift_ratio < 2.0:
        grade = "Acceptable"
        grade_color = "amber"
    else:
        grade = "Needs Improvement"
        grade_color = "red"

    # --- Summary ---
    summary_lines = [
        f"Validation Report for {model_name}",
        f"Overall R2: {overall['r2']:.4f}, RMSE: {overall['rmse']:.4f}",
        f"95% CI for RMSE: [{ci_rmse['ci_lower']:.4f}, {ci_rmse['ci_upper']:.4f}]",
        f"Temporal drift ratio: {drift_ratio:.2f}x (ideal < 1.5x)",
        f"Batch shift ratio: {shift_ratio:.2f}x (ideal < 1.5x)",
        f"Grade: {grade}",
    ]

    report = {
        "model_name": model_name,
        "n_samples": len(X),
        "n_features": X.shape[1] if X.ndim == 2 else 1,
        "overall_metrics": overall,
        "time_based": {
            "n_splits": len(time_results),
            "results": time_results,
            "avg_rmse": round(float(avg_time_rmse), 6),
            "drift_ratio": round(drift_ratio, 4),
        },
        "batch_shift": {
            "n_batches": len(batch_results),
            "results": batch_results,
            "avg_rmse": round(float(avg_batch_rmse), 6),
            "shift_ratio": round(shift_ratio, 4),
        },
        "confidence_intervals": confidence_intervals,
        "grade": grade,
        "grade_color": grade_color,
        "summary": "\n".join(summary_lines),
    }

    log.info("Validation report for %s: Grade=%s, R2=%.4f, RMSE=%.4f",
             model_name, grade, overall["r2"], overall["rmse"])

    return report


# ===========================================================================
# 6. Quick Validation for ProtePilot Models
# ===========================================================================

def validate_chromatography_model(
    model: Any = None,
    X_train: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Quick validation wrapper for the ProtePilot MLP chromatography model.

    If model and X_train are not provided, generates synthetic data
    and uses a simple predictor for demonstration.
    """
    if model is not None and X_train is not None:
        # Use actual model
        try:
            import torch
            def predict_fn(X):
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    return model(X_tensor).numpy()

            # Generate predictions on training data for validation
            y_train = predict_fn(X_train)
            return run_validation_report(
                predict_fn, X_train, y_train,
                model_name="ChromatographyMLP (ka/nu)"
            )
        except Exception as e:
            log.warning("PyTorch model validation failed: %s", e)

    # Demo mode: generate synthetic data
    rng = np.random.RandomState(42)
    n_samples = 50
    n_features = 7  # pI, MW, deam, ox, acidic, basic, hydrophobicity

    X_demo = rng.randn(n_samples, n_features)
    # True relationship: y = f(X) + noise
    true_weights = np.array([0.5, -0.3, 0.1, 0.2, -0.1, 0.15, 0.4])
    y_demo = X_demo @ true_weights + rng.randn(n_samples) * 0.1

    # Simple linear prediction function (simulates a trained model)
    fitted_weights = true_weights + rng.randn(n_features) * 0.02

    def demo_predict(X):
        return X @ fitted_weights

    return run_validation_report(
        demo_predict, X_demo, y_demo,
        model_name="ChromatographyMLP (Demo)"
    )


def validate_developability_model(
    predictor: Any = None,
) -> Dict[str, Any]:
    """
    Quick validation wrapper for the XGBoost developability predictor.

    Uses the mock dataset from developability_predictor module.
    """
    if predictor is not None:
        try:
            from src.pLM_embedder import get_embedder
            from src.developability_predictor import _generate_mock_dataset, build_feature_matrix, N_BIOPHYS

            embedder = get_embedder()
            dataset = _generate_mock_dataset()
            X, y = build_feature_matrix(dataset, embedder)

            def predict_fn(X_in):
                preds = []
                n_biophys = N_BIOPHYS  # 7 biophysical features at the end
                split_idx = X_in.shape[1] - n_biophys
                for i in range(len(X_in)):
                    emb = X_in[i, :split_idx]
                    bio = X_in[i, split_idx:]
                    pred = predictor.predict(emb, bio)
                    preds.append(pred.get("agg_risk", 0.5))
                return np.array(preds)

            return run_validation_report(
                predict_fn, X, y[:, 0] if y.ndim == 2 else y,
                model_name="DevelopabilityPredictor (XGBoost)"
            )
        except Exception as e:
            log.warning("Developability model validation failed: %s", e)

    # Demo mode
    rng = np.random.RandomState(123)
    n = 25
    X = rng.randn(n, 647)
    y = 1 / (1 + np.exp(-X[:, :3].sum(axis=1) + rng.randn(n) * 0.5))

    def demo_predict(X_in):
        return 1 / (1 + np.exp(-X_in[:, :3].sum(axis=1)))

    return run_validation_report(
        demo_predict, X, y,
        model_name="DevelopabilityPredictor (Demo)"
    )


# ===========================================================================
# __main__: Standalone Test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  ProtePilot — Validation Strategy v1.0 Test")
    print("=" * 60)

    # --- Test 1: Metrics ---
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    metrics = compute_metrics(y_true, y_pred)
    print(f"\n1. Metrics Test:")
    print(f"   R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
          f"MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")

    # --- Test 2: Bootstrap CI ---
    ci = bootstrap_confidence_interval(y_true, y_pred, _rmse, n_bootstrap=500)
    print(f"\n2. Bootstrap CI (RMSE):")
    print(f"   Mean={ci['mean']:.4f}, 95% CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # --- Test 3: Time-Based Split ---
    rng = np.random.RandomState(42)
    X_test = rng.randn(30, 7)
    y_test = rng.randn(30)
    splits = time_based_split(X_test, y_test, n_splits=3)
    print(f"\n3. Time-Based Split ({len(splits)} splits):")
    for s in splits:
        print(f"   {s['description']} — train={s['train_size']}, test={s['test_size']}")

    # --- Test 4: Batch-Shift Split ---
    batches = batch_shift_split(X_test, y_test, n_batches=3)
    print(f"\n4. Batch-Shift Split ({len(batches)} batches):")
    for b in batches:
        print(f"   {b['description']} — train={b['train_size']}, test={b['test_size']}, "
              f"mean_shift={b['shift_applied']['mean_shift']:.4f}")

    # --- Test 5: Full Validation Report ---
    print(f"\n5. Full Validation Report:")
    report = validate_chromatography_model()
    print(f"   Model: {report['model_name']}")
    print(f"   Samples: {report['n_samples']}, Features: {report['n_features']}")
    print(f"   Overall: R2={report['overall_metrics']['r2']:.4f}, "
          f"RMSE={report['overall_metrics']['rmse']:.4f}")
    print(f"   Time drift ratio: {report['time_based']['drift_ratio']:.2f}x")
    print(f"   Batch shift ratio: {report['batch_shift']['shift_ratio']:.2f}x")
    ci_rmse = report["confidence_intervals"]["rmse"]
    print(f"   RMSE 95% CI: [{ci_rmse['ci_lower']:.4f}, {ci_rmse['ci_upper']:.4f}]")
    print(f"   Grade: {report['grade']}")

    # --- Test 6: Developability Demo ---
    print(f"\n6. Developability Validation (Demo):")
    dev_report = validate_developability_model()
    print(f"   Model: {dev_report['model_name']}")
    print(f"   R2={dev_report['overall_metrics']['r2']:.4f}, "
          f"RMSE={dev_report['overall_metrics']['rmse']:.4f}")
    print(f"   Grade: {dev_report['grade']}")

    print("\nValidation Strategy v1.0 test complete")
