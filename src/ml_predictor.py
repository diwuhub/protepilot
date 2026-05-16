"""
ml_predictor.py  ·  ProtePilot — Milestone 19
===========================================================
Hybrid AI Core: PyTorch Deep Learning + XGBoost Wet-Lab Predictor + Potency Predictor + SHAP

Version   : 6.0 (Model Persistence + GoSilico DoE)
Author    : Di (ProtePilot)
Depends   : torch, numpy, shap, matplotlib, xgboost (optional), joblib

Architecture (v5.0)
------------------------------------------------------------
Three ML subsystems:

  A) ChromatographyMLP (PyTorch):  IEX ka/nu prediction (v2.0+)
  B) WetLabPredictor (XGBoost):    Supervised on actual wet-lab CSV data
     - Predicts Exp_Aggregation_Percent and Exp_Tm_MeltingTemp
     - Trained directly on ingested Jain-137-format datasets
     - Returns R², RMSE, per-target SHAP importance
  C) PotencyPredictor (XGBoost):   Potency/Affinity prediction (NEW M16)
     - Maps ESM-2 embeddings (CDR focus) → Predicted_Potency_Score
     - Trained on historical ELISA/Kd data from Discovery CSV
     - Returns R², RMSE for potency target

v6.0 Changes (M19)
------------------------------------------------------------
  - Model Persistence: joblib for XGBoost, torch.save for PyTorch MLP
  - Auto-save on training, auto-load on startup
  - get_model_status(): returns trained/heuristic state for all predictors
  - Models saved to models/ directory alongside app.py

v5.0 Changes (M16)
------------------------------------------------------------
  - PotencyPredictor: XGBoost trained on ELISA OD / Kd data → potency score
  - train_potency_model(): Train from biophysical features + binding data
  - predict_potency(): Predict potency/affinity score for any candidate
  - get_potency_model() / get_potency_metrics(): Cached model access
  - generate_mock_potency_dataset(): Synthetic training data for testing
  - All v4.0 APIs preserved (backward-compatible)

v4.0 Changes
------------------------------------------------------------
  - WetLabPredictor: XGBoost multi-output regressor trained on CSV wet-lab data
  - train_wetlab_model(): Train from (X, y) arrays with aggregation & Tm targets
  - predict_wetlab(): Predict aggregation% and Tm for any biophysical feature set
  - get_wetlab_model_metrics(): Return R², RMSE, feature importance
  - evaluate_variant_wetlab(): Score a mutated sequence through the trained model
  - All v3.0 APIs preserved (backward-compatible)

Multi-Layer Perceptron (MLP) that predicts IEX chromatographic
parameters from sequence-derived biophysical features:

    Input (7 features):
        1. Theoretical pI
        2. Molecular Weight (kDa)
        3. Deamidation site count
        4. Oxidation site count
        5. Acidic residue count (D + E)
        6. Basic residue count (K + R + H)
        7. Hydrophobicity (GRAVY score, normalized to [0,1])

    Output (2 targets):
        1. ka  -- SMA adsorption rate constant
        2. nu  -- SMA characteristic charge number

v3.0 Changes
------------------------------------------------------------
  - retrain_with_labels(): Merge expert-corrected data with synthetic
  - Training run counter and snapshot tracking for Continuous Learning
  - get_training_run_count() for dashboard integration
  - Backward-compatible: all v2.0 APIs unchanged

v2.0 Changes
------------------------------------------------------------
  - Feature expansion: Hydrophobicity (7th input) via Biopython GRAVY
  - Target-driven learning: Synthetic targets are calibrated so that
    predicted ka/nu values produce main peak RT in the 15-20 min window
    under standard gradient conditions (50->500 mM, 15 mM/min, 0.25m col)
  - RT estimation function for analytical validation
  - MLP architecture: 7->64->32->16->2

References
------------------------------------------------------------
  SHAP:    https://shap.readthedocs.io/
  PyTorch: https://pytorch.org/docs/stable/
  Kyte-Doolittle: J. Mol. Biol. 157(1):105-132 (1982)
"""

from __future__ import annotations

import logging
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ProtePilot.MLPredictor")


# ===========================================================================
# 0. Model Persistence Infrastructure (M19)
# ===========================================================================

# Default models directory (relative to project root)
_MODELS_DIR: Optional[str] = None


def _get_models_dir() -> str:
    """Return (and create if needed) the models directory."""
    import os
    global _MODELS_DIR
    if _MODELS_DIR is None:
        # Try to find the project root (where app.py lives)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _MODELS_DIR = os.path.join(root, "models")
    os.makedirs(_MODELS_DIR, exist_ok=True)
    return _MODELS_DIR


def _save_model_joblib(model_obj: Any, filename: str) -> Optional[str]:
    """Save a model object using joblib. Returns path or None."""
    try:
        import joblib
        import os
        path = os.path.join(_get_models_dir(), filename)
        joblib.dump(model_obj, path)
        log.info("Model saved: %s (%d bytes)", path, os.path.getsize(path))
        return path
    except Exception as e:
        log.warning("Model save failed (%s): %s", filename, e)
        return None


def _load_model_joblib(filename: str) -> Optional[Any]:
    """Load a model object using joblib. Returns model or None."""
    try:
        import joblib
        import os
        path = os.path.join(_get_models_dir(), filename)
        if os.path.exists(path):
            model = joblib.load(path)
            log.info("Model loaded: %s", path)
            return model
        return None
    except Exception as e:
        log.warning("Model load failed (%s): %s", filename, e)
        return None


# Model version tag: bumped whenever PropertyMapper ka/nu range changes.
# Cached models with a different version are auto-invalidated on load.
_MODEL_VERSION_TAG = "v7.3.1-std-ka"


def _save_pytorch_model(model_obj: Any, filename: str) -> Optional[str]:
    """Save a PyTorch model's state_dict with version tag using torch.save."""
    try:
        import torch
        import os
        path = os.path.join(_get_models_dir(), filename)
        torch.save({
            "state_dict": model_obj.net.state_dict() if hasattr(model_obj, "net") else {},
            "scaler_mean": model_obj.scaler.mean.tolist() if hasattr(model_obj.scaler, "mean") else None,
            "scaler_std": model_obj.scaler.std.tolist() if hasattr(model_obj.scaler, "std") else None,
            "n_inputs": model_obj.n_inputs if hasattr(model_obj, "n_inputs") else 7,
            "n_outputs": model_obj.n_outputs if hasattr(model_obj, "n_outputs") else 2,
            "model_version": _MODEL_VERSION_TAG,
        }, path)
        log.info("PyTorch model saved: %s (version=%s)", path, _MODEL_VERSION_TAG)
        return path
    except Exception as e:
        log.warning("PyTorch model save failed (%s): %s", filename, e)
        return None


def _load_pytorch_model(filename: str) -> Optional[Dict[str, Any]]:
    """Load a PyTorch model checkpoint. Returns dict or None if version mismatch."""
    try:
        import torch
        import os
        path = os.path.join(_get_models_dir(), filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            saved_version = checkpoint.get("model_version", "unknown")
            if saved_version != _MODEL_VERSION_TAG:
                log.warning(
                    "PyTorch model version mismatch: saved=%s, current=%s. "
                    "Discarding stale cache and will retrain.",
                    saved_version, _MODEL_VERSION_TAG,
                )
                # Remove the stale file
                try:
                    os.remove(path)
                    log.info("Removed stale model file: %s", path)
                except OSError:
                    pass
                return None
            log.info("PyTorch model loaded: %s (version=%s)", path, saved_version)
            return checkpoint
        return None
    except Exception as e:
        log.warning("PyTorch model load failed (%s): %s", filename, e)
        return None


# Persistence filenames
MODEL_FILE_WETLAB = "xgboost_wetlab_latest.pkl"
MODEL_FILE_WETLAB_PLM = "xgboost_wetlab_plm_latest.pkl"
MODEL_FILE_POTENCY = "xgboost_potency_latest.pkl"
MODEL_FILE_MLP = "pytorch_mlp_latest.pt"
MODEL_FILE_DEVELOPABILITY = "xgboost_developability_latest.pkl"

# Forward-declared PLM model caches (populated by train_plm_wetlab_model)
_CACHED_PLM_MODEL: Optional[Any] = None
_PLM_TRAINING_METRICS: Optional[Dict[str, Any]] = None


def get_model_status() -> Dict[str, Dict[str, Any]]:
    """
    Check persistence status of all model types.

    Returns dict with model_type -> {loaded, source, path}
    Source values: "custom_trained", "literature_calibrated", "persisted_on_disk", "baseline_heuristic"
    """
    import os
    models_dir = _get_models_dir()

    status = {}
    for label, filename, model_cache in [
        ("wetlab", MODEL_FILE_WETLAB, _CACHED_WETLAB_MODEL),
        ("potency", MODEL_FILE_POTENCY, _CACHED_POTENCY_MODEL),
        ("chromatography_mlp", MODEL_FILE_MLP, None),
    ]:
        path = os.path.join(models_dir, filename)
        file_exists = os.path.exists(path)
        in_memory = model_cache is not None and (
            hasattr(model_cache, "trained") and model_cache.trained
        )

        if in_memory:
            # Distinguish between custom_trained and literature_calibrated
            if label == "wetlab" and _JAIN137_CALIBRATED:
                source = "literature_calibrated"
            else:
                source = "custom_trained"
        elif file_exists:
            source = "persisted_on_disk"
        else:
            source = "baseline_heuristic"

        status[label] = {
            "loaded": in_memory,
            "persisted": file_exists,
            "source": source,
            "path": path if file_exists else None,
        }

    # Add calibration info
    status["calibration"] = get_calibration_status()

    return status


def load_persisted_models() -> Dict[str, bool]:
    """
    Attempt to load all persisted models from disk on startup.

    Returns dict with model_type -> loaded (bool)
    """
    global _CACHED_WETLAB_MODEL, _WETLAB_TRAINING_METRICS
    global _CACHED_POTENCY_MODEL, _POTENCY_TRAINING_METRICS
    global _CACHED_PLM_MODEL, _PLM_TRAINING_METRICS

    results = {}

    # Load WetLab model
    wetlab = _load_model_joblib(MODEL_FILE_WETLAB)
    if wetlab is not None and hasattr(wetlab, "trained") and wetlab.trained:
        _CACHED_WETLAB_MODEL = wetlab
        _WETLAB_TRAINING_METRICS = wetlab.get_metrics() if hasattr(wetlab, "get_metrics") else None
        results["wetlab"] = True
        log.info("Restored persisted WetLab model")
    else:
        results["wetlab"] = False

    # Load Potency model
    potency = _load_model_joblib(MODEL_FILE_POTENCY)
    if potency is not None and hasattr(potency, "trained") and potency.trained:
        _CACHED_POTENCY_MODEL = potency
        _POTENCY_TRAINING_METRICS = potency.get_metrics() if hasattr(potency, "get_metrics") else None
        results["potency"] = True
        log.info("Restored persisted Potency model")
    else:
        results["potency"] = False

    # Load PLM-based WetLab model
    plm = _load_model_joblib(MODEL_FILE_WETLAB_PLM)
    if plm is not None and hasattr(plm, "trained") and plm.trained:
        _CACHED_PLM_MODEL = plm
        _PLM_TRAINING_METRICS = plm.get_metrics() if hasattr(plm, "get_metrics") else None
        results["plm_wetlab"] = True
        log.info("Restored persisted PLM WetLab model")
    else:
        results["plm_wetlab"] = False

    return results


def factory_reset() -> Dict[str, Any]:
    """
    Delete all persisted model files and clear in-memory caches.

    This forces the system to fall back to baseline mechanistic heuristics.
    Returns a dict with deleted files and reset status.
    """
    import os
    global _CACHED_WETLAB_MODEL, _WETLAB_TRAINING_METRICS
    global _CACHED_POTENCY_MODEL, _POTENCY_TRAINING_METRICS
    global _CACHED_PLM_MODEL, _PLM_TRAINING_METRICS
    global _JAIN137_CALIBRATED, _JAIN137_CALIBRATION_METRICS

    deleted = []
    errors = []
    models_dir = _get_models_dir()

    # Delete persisted model files
    for filename in [MODEL_FILE_WETLAB, MODEL_FILE_POTENCY, MODEL_FILE_MLP,
                     MODEL_FILE_DEVELOPABILITY, MODEL_FILE_WETLAB_PLM]:
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            try:
                os.remove(path)
                deleted.append(filename)
                log.info("Factory reset: deleted %s", path)
            except Exception as e:
                errors.append(f"{filename}: {e}")

    # Clear in-memory caches
    _CACHED_WETLAB_MODEL = None
    _WETLAB_TRAINING_METRICS = None
    _CACHED_POTENCY_MODEL = None
    _POTENCY_TRAINING_METRICS = None
    _CACHED_PLM_MODEL = None
    _PLM_TRAINING_METRICS = None
    _JAIN137_CALIBRATED = False
    _JAIN137_CALIBRATION_METRICS = None

    log.warning("Factory reset complete: %d files deleted, in-memory caches cleared", len(deleted))

    return {
        "status": "success" if not errors else "partial",
        "deleted_files": deleted,
        "errors": errors,
        "message": f"Reset complete. Deleted {len(deleted)} model files. System is now using baseline heuristics.",
    }


# ===========================================================================
# 0B. Jain-137 Baseline Calibration (v2.0)
# ===========================================================================

# Global calibration state
_JAIN137_CALIBRATED = False
_JAIN137_CALIBRATION_METRICS: Optional[Dict[str, Any]] = None


def calibrate_baseline_from_jain137(n_samples: int = 50, seed: int = 137) -> Dict[str, Any]:
    """
    Calibrate the heuristic baseline by auto-training XGBoost on Jain-137 mock data.

    This provides a literature-calibrated foundation model that is better than
    the uncalibrated heuristic but lower confidence than user-trained custom models.

    Steps:
      1. Generate synthetic Jain-137 dataset (realistic mAb sequences + assay scores)
      2. Extract biophysical features from each sequence
      3. Train XGBoost WetLab predictor (Agg%, Tm) on the synthetic data
      4. Mark model source as "literature_calibrated" (not "custom_trained")

    Returns dict with calibration metrics (R², MAE per target).
    """
    global _JAIN137_CALIBRATED, _JAIN137_CALIBRATION_METRICS

    try:
        from src.data_pipeline import generate_mock_jain137
    except ImportError:
        return {"status": "error", "message": "data_pipeline not available"}

    log.info("Calibrating baseline from Jain-137 (%d samples, seed=%d)", n_samples, seed)

    # Step 1: Generate dataset
    mock = generate_mock_jain137(n_samples=n_samples, seed=seed)
    if mock.get("status") != "success":
        return {"status": "error", "message": "Failed to generate Jain-137 data"}

    data = mock["data"]

    # Step 2: Extract features and targets
    features_list = []
    targets_list = []
    for row in data:
        hc = row.get("Sequence_HC", "")
        lc = row.get("Sequence_LC", "")
        combined = hc + lc if lc else hc
        if len(combined) < 50:
            continue

        feats = extract_features_from_sequence(combined)
        agg = row.get("Exp_Aggregation_Percent")
        tm = row.get("Exp_Tm_MeltingTemp")
        if feats is not None and agg is not None and tm is not None:
            features_list.append(feats)
            targets_list.append([agg, tm])

    if len(features_list) < 10:
        return {"status": "error", "message": f"Too few valid samples ({len(features_list)})"}

    X = np.array(features_list, dtype=np.float32)
    y = np.array(targets_list, dtype=np.float32)

    log.info("Calibration data: %d samples, %d features, %d targets", X.shape[0], X.shape[1], y.shape[1])

    # Step 3: Train via existing pipeline
    try:
        model, metrics = train_wetlab_model(
            X=X,
            y=y,
            target_names=["Exp_Aggregation_Percent", "Exp_Tm_MeltingTemp"],
        )
        if model is not None and hasattr(model, "trained") and model.trained:
            _JAIN137_CALIBRATED = True
            _JAIN137_CALIBRATION_METRICS = metrics
            log.info("Jain-137 calibration complete: %s", metrics)
            return {
                "status": "success",
                "source": "literature_calibrated",
                "n_samples": X.shape[0],
                "metrics": metrics,
            }
        else:
            return {"status": "error", "message": "Model training returned no valid model"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def extract_features_from_sequence(sequence: str) -> Optional[np.ndarray]:
    """Extract 7-dim biophysical feature vector from amino acid sequence."""
    if not sequence or len(sequence) < 20:
        return None

    import re
    seq = sequence.upper()
    seq_clean = "".join(c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY")
    if len(seq_clean) < 20:
        return None

    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        analysis = ProteinAnalysis(seq_clean)
        pI = analysis.isoelectric_point()
        mw_kda = analysis.molecular_weight() / 1000.0
        gravy = analysis.gravy()
        hydrophobicity = max(0.0, min(1.0, (gravy + 2.0) / 4.0))
    except Exception:
        pI = 7.5
        mw_kda = len(seq_clean) * 0.11
        hydrophobicity = 0.35

    deam_sites = len(re.findall(r"N[GS]", seq_clean))
    ox_sites = seq_clean.count("M") + seq_clean.count("W")
    acidic = seq_clean.count("D") + seq_clean.count("E")
    basic = seq_clean.count("K") + seq_clean.count("R") + seq_clean.count("H")

    return np.array([pI, mw_kda, deam_sites, ox_sites, acidic, basic, hydrophobicity],
                    dtype=np.float32)


def get_calibration_status() -> Dict[str, Any]:
    """Return current calibration state (literature-calibrated vs baseline)."""
    return {
        "calibrated": _JAIN137_CALIBRATED,
        "source": "literature_calibrated" if _JAIN137_CALIBRATED else "baseline_heuristic",
        "metrics": _JAIN137_CALIBRATION_METRICS,
    }


# ===========================================================================
# 1. Feature Engineering (v2.0: +Hydrophobicity)
# ===========================================================================

FEATURE_NAMES = [
    "pI",
    "MW_kDa",
    "deam_sites",
    "ox_sites",
    "acidic_residues",
    "basic_residues",
    "hydrophobicity",
]

TARGET_NAMES_DUAL = ["ka", "nu"]
TARGET_NAMES_RT   = ["RT_main_min"]


def compute_hydrophobicity_from_sequence(sequence: str) -> float:
    """
    Compute normalized hydrophobicity from amino acid sequence using Biopython GRAVY.

    GRAVY (Grand Average of Hydropathicity) typically ranges from -2.0 to +2.0
    for proteins. We normalize to [0, 1] via: h = (GRAVY + 2.0) / 4.0, clamped.

    Returns 0.35 (default mAb) if Biopython unavailable or sequence invalid.
    """
    if not sequence or len(sequence) < 5:
        return 0.35

    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        # Clean sequence: remove non-standard residues
        clean_seq = "".join(c for c in sequence.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
        if len(clean_seq) < 5:
            return 0.35
        analysis = ProteinAnalysis(clean_seq)
        gravy = analysis.gravy()
        # Normalize GRAVY [-2, +2] -> [0, 1]
        h = (gravy + 2.0) / 4.0
        return float(max(0.0, min(1.0, h)))
    except Exception:
        return 0.35


def extract_features(
    pI: float,
    mw: float,
    deam_sites: int = 1,
    ox_sites: int = 1,
    acidic_residues: int = 40,
    basic_residues: int = 50,
    hydrophobicity: float = 0.35,
) -> np.ndarray:
    """
    Build a feature vector from protein properties.

    Returns shape (7,) numpy array.
    """
    return np.array([
        pI,
        mw,
        float(deam_sites),
        float(ox_sites),
        float(acidic_residues),
        float(basic_residues),
        hydrophobicity,
    ], dtype=np.float32)


def extract_features_from_intent(intent: Dict[str, Any]) -> np.ndarray:
    """
    Extract ML feature vector from a parsed intent dictionary.

    Handles both FASTA-derived and text-derived inputs by
    pulling liability counts from chain_analyses when available.
    Computes hydrophobicity from sequence via Biopython GRAVY when possible.
    """
    pI = intent.get("pI", 8.0)
    mw = intent.get("mw", 150.0)
    deam = intent.get("deam_sites", 1)
    ox = intent.get("ox_sites", 1)
    acidic = 40  # default
    basic = 50   # default
    hydro = intent.get("hydrophobicity", 0.35)

    # If chain_analyses available (FASTA input), sum liability counts
    chain_analyses = intent.get("chain_analyses", [])
    if chain_analyses:
        acidic = sum(ca["liabilities"].get("acidic_count", 0) for ca in chain_analyses)
        basic = sum(ca["liabilities"].get("basic_count", 0) for ca in chain_analyses)
        # Also use detected deamidation/oxidation counts from liabilities
        total_deam = sum(ca["liabilities"].get("deamidation_count", 0) for ca in chain_analyses)
        total_ox_met = sum(ca["liabilities"].get("met_count", 0) for ca in chain_analyses)
        total_ox_trp = sum(ca["liabilities"].get("trp_count", 0) for ca in chain_analyses)
        if total_deam > 0:
            deam = total_deam
        if total_ox_met + total_ox_trp > 0:
            ox = total_ox_met + total_ox_trp

    # Compute hydrophobicity from sequence if available
    sequence = intent.get("sequence", None)
    if sequence and len(sequence) > 20:
        hydro = compute_hydrophobicity_from_sequence(sequence)

    return extract_features(pI, mw, deam, ox, acidic, basic, hydro)


# ===========================================================================
# 2. RT Estimation (analytical, for target calibration)
# ===========================================================================

def estimate_rt_from_sma(
    ka: float,
    nu: float,
    kd: float = 1000.0,
    lambda_: float = 1200.0,
    sigma: float = 0.0,
    c_salt_start: float = 50.0,
    c_salt_end: float = 500.0,
    gradient_time: float = 30.0,
    column_length: float = 0.25,
    epsilon: float = 0.37,
    flow_velocity: float = 5.75e-4,
    dynamic_factor: float = 1.0,
) -> float:
    """
    Estimate main peak retention time from SMA parameters using
    gradient elution theory (Yamamoto approximation).

    Standard SMA: elution when k'(c) ~ 1, where:
        k'(c) = F * ((Lambda - sigma)/c)^nu * Keq
    Solving: c_elution = (Lambda - sigma) * (F * Keq)^(1/nu)

    v32.2: Added sigma (steric shielding factor) to reduce effective ionic
    capacity.  Default sigma=0.0 preserves backward compatibility for all
    existing callers that do not pass sigma.

    With standard-range ka (1-5), kd=1000, Keq~0.001-0.005:
        c_elution = 1200 * (1.7 * 0.003)^(1/3) ≈ 206 mM → RT ≈ 17.6 min
    No dynamic_factor correction needed (default 1.0).

    Parameters
    ----------
    ka, nu          : SMA adsorption parameters (standard range, ka 1-5)
    kd              : Desorption rate constant (default 1000)
    lambda_         : Ionic capacity (default 1200 mol/m3)
    sigma           : Steric shielding factor (default 0.0 for backward compat)
    c_salt_start, c_salt_end : Gradient endpoints (mM)
    gradient_time   : Gradient duration (min)
    column_length   : Column length (m)
    epsilon         : Column void fraction
    flow_velocity   : Superficial velocity (m/s)
    dynamic_factor  : Correction for gradient dynamics (default 1.0, v7.3)

    Returns
    -------
    float : Estimated retention time (min)
    """
    # v7.3.1: Clamp nu to physically reasonable mAb CEX range.
    # mAb nu rarely exceeds 3.5 for standard CEX; higher values
    # indicate training artefact and cause RT saturation at gradient end.
    nu = float(max(1.5, min(nu, 3.5)))

    Keq = ka / kd
    F = (1.0 - epsilon) / epsilon  # ~1.70
    lambda_eff = max(lambda_ - sigma, 100.0)  # Steric-corrected ionic capacity

    # Equilibrium elution salt concentration
    try:
        c_elution_eq = lambda_eff * (F * Keq) ** (1.0 / max(nu, 0.5))
    except (OverflowError, ZeroDivisionError):
        c_elution_eq = c_salt_end

    # Apply dynamic correction for CADET PDE effects
    c_elution = c_elution_eq * dynamic_factor

    # Clamp to gradient range
    c_elution = max(c_salt_start, min(c_salt_end, c_elution))

    # Convert salt concentration to time via linear gradient
    gradient_slope_mM_per_min = (c_salt_end - c_salt_start) / gradient_time
    t_gradient = (c_elution - c_salt_start) / gradient_slope_mM_per_min

    # Add column dead time
    t_dead = (column_length / flow_velocity) * epsilon / 60.0  # convert s to min

    rt = t_dead + t_gradient
    return float(max(0.5, min(gradient_time, rt)))


# ===========================================================================
# 3. Synthetic Training Data Generator (v2.0: RT-Targeted)
# ===========================================================================

def generate_synthetic_dataset(
    n_samples: int = 500,
    seed: int = 42,
    target_rt_center: float = 17.5,
    target_rt_window: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data with RT-targeted calibration.

    v2.0 Strategy: The neural network's training targets are calibrated
    so that the predicted ka/nu values produce elution in the 15-20 min
    window. This is achieved by:
      1. Starting from PropertyMapper v5.0 physics as the base
      2. Applying a correction factor that nudges ka/nu toward values
         that yield RT in [target_rt_center - window, target_rt_center + window]
      3. Adding realistic noise for generalization

    Parameters
    ----------
    n_samples         : Number of training samples
    seed              : Random seed for reproducibility
    target_rt_center  : Target retention time center (min), default 17.5
    target_rt_window  : Half-width of target RT window (min), default 2.5

    Returns
    -------
    X : (n_samples, 7) feature matrix [pI, MW, deam, ox, acidic, basic, hydro]
    y : (n_samples, 2) target matrix [ka, nu]
    """
    rng = np.random.RandomState(seed)

    try:
        from src.PropertyMapper import ProteinProperties, PropertyMapper
        mapper = PropertyMapper()
        use_mapper = True
    except ImportError:
        use_mapper = False
        log.warning("PropertyMapper not available; using analytical fallback for synthetic data")

    X_list = []
    y_list = []

    for _ in range(n_samples):
        # Sample from realistic mAb property distributions
        pI = rng.uniform(6.5, 9.8)
        mw = rng.uniform(25.0, 200.0)
        deam = rng.randint(0, 6)
        ox = rng.randint(0, 6)
        acidic = rng.randint(15, 80)
        basic = rng.randint(20, 90)
        hydro = rng.uniform(0.10, 0.65)

        features = extract_features(pI, mw, deam, ox, acidic, basic, hydro)

        if use_mapper:
            protein = ProteinProperties(
                name="synth",
                pI=pI,
                MW_kDa=mw,
                hydrophobicity=hydro,
                pH_working=7.0,
                ptm_profile={"deamidation_sites": deam, "oxidation_sites": ox},
            )
            params = mapper.map(protein)
            ka_base = params["ka"]
            nu_base = params["nu"]
        else:
            # Analytical fallback matching PropertyMapper v7.3 standard-range logic
            charge_dist = max(0.0, pI - 7.0)
            nu_base = max(2.0, min(3.5, 2.5 + min(0.15 * charge_dist, 0.8)))
            ka_base = max(0.3, min(8.0, 3.0 * (1 + 0.8 * hydro)))

        # --- RT-Targeted Correction (v2.0) ---
        # Estimate RT from base parameters
        rt_est = estimate_rt_from_sma(ka_base, nu_base)

        # Compute correction factor to nudge RT toward target window
        rt_target = target_rt_center + rng.uniform(-target_rt_window, target_rt_window)
        rt_ratio = rt_target / max(rt_est, 1.0)

        # Apply correction: if RT is too high, reduce nu slightly;
        # if RT is too low, increase nu. ka gets a proportional tweak.
        # The correction is gentle (damped) to preserve physical relationships.
        damping = 0.6  # how aggressively to correct (0=no correction, 1=full)

        if rt_ratio < 1.0:
            # RT too high -> reduce binding strength
            nu_corrected = nu_base * (1.0 + damping * (rt_ratio - 1.0) * 0.3)
            ka_corrected = ka_base * (1.0 + damping * (rt_ratio - 1.0) * 0.2)
        else:
            # RT too low -> increase binding strength
            nu_corrected = nu_base * (1.0 + damping * (rt_ratio - 1.0) * 0.2)
            ka_corrected = ka_base * (1.0 + damping * (rt_ratio - 1.0) * 0.15)

        # Clamp to physical bounds (standard-range, v7.3.1)
        # nu ≤ 3.5: mAb CEX nu rarely exceeds 3.5; higher causes RT saturation
        ka = float(max(0.3, min(8.0, ka_corrected)))
        nu = float(max(2.0, min(3.5, nu_corrected)))

        # Add small noise for generalization (standard-range ka)
        ka += rng.normal(0, 0.015)
        nu += rng.normal(0, 0.008)

        ka = float(max(0.3, ka))
        nu = float(max(2.0, min(3.5, nu)))

        X_list.append(features)
        y_list.append([ka, nu])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Verify RT distribution of targets
    rt_estimates = [estimate_rt_from_sma(y[i, 0], y[i, 1]) for i in range(len(y))]
    rt_arr = np.array(rt_estimates)
    in_window = np.sum((rt_arr >= target_rt_center - target_rt_window) &
                       (rt_arr <= target_rt_center + target_rt_window))

    log.info("Generated synthetic dataset: %d samples, %d features, %d targets",
             n_samples, X.shape[1], y.shape[1])
    log.info("RT distribution: mean=%.1f min, std=%.1f min, in [%.0f-%.0f] window: %d/%d (%.0f%%)",
             rt_arr.mean(), rt_arr.std(),
             target_rt_center - target_rt_window,
             target_rt_center + target_rt_window,
             in_window, n_samples, 100 * in_window / n_samples)

    return X, y


# ===========================================================================
# 4. Feature Normalization
# ===========================================================================

@dataclass
class FeatureScaler:
    """Simple min-max feature scaler for the ML pipeline."""
    min_vals: Optional[np.ndarray] = None
    max_vals: Optional[np.ndarray] = None
    fitted: bool = False

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


# ===========================================================================
# 5. PyTorch MLP Model (v2.0: 7 inputs)
# ===========================================================================

def _get_torch():
    """Lazy import of torch to avoid startup cost if not needed."""
    import torch
    import torch.nn as nn
    return torch, nn


class ChromatographyMLP:
    """
    Multi-Layer Perceptron for predicting IEX chromatographic parameters.

    v2.0 Architecture:
        Input(7) -> Linear(64) -> ReLU -> Dropout(0.1)
                  -> Linear(32) -> ReLU -> Dropout(0.1)
                  -> Linear(16) -> ReLU
                  -> Linear(n_outputs)

    The network learns RT-targeted ka/nu predictions from sequence-derived
    biophysical features including hydrophobicity.
    """

    def __init__(self, n_inputs: int = 7, n_outputs: int = 2):
        torch, nn = _get_torch()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.device = torch.device("cpu")

        self.model = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_outputs),
        ).to(self.device)

        self.scaler = FeatureScaler()
        self.trained = False
        self.training_history: List[Dict[str, float]] = []

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Train the MLP on the provided dataset.

        Parameters
        ----------
        X          : (n_samples, n_features) feature matrix
        y          : (n_samples, n_outputs) target matrix
        epochs     : Number of training epochs
        lr         : Learning rate (Adam optimizer)
        batch_size : Mini-batch size
        val_split  : Fraction of data for validation
        verbose    : Print training progress

        Returns
        -------
        history : List of {epoch, train_loss, val_loss} dicts
        """
        torch, nn = _get_torch()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train/val split
        n = len(X_scaled)
        n_val = max(1, int(n * val_split))
        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train = torch.tensor(X_scaled[train_idx], dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32, device=self.device)
        X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32, device=self.device)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        history = []
        n_train = len(X_train)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle training data
            perm = torch.randperm(n_train, device=self.device)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            for i in range(0, n_train, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            record = {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(val_loss, 6),
            }
            history.append(record)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                log.info("Epoch %3d/%d  train_MSE=%.6f  val_MSE=%.6f",
                         epoch + 1, epochs, avg_train_loss, val_loss)

        self.trained = True
        self.training_history = history

        # v7.3.2: Compute validation R² per output for quality gating
        self.model.eval()
        with torch.no_grad():
            val_pred = self.model(X_val).cpu().numpy()
            y_val_np = y_val.cpu().numpy()
        self.val_r2 = {}
        for t_idx in range(val_pred.shape[1]):
            ss_res = float(np.sum((y_val_np[:, t_idx] - val_pred[:, t_idx]) ** 2))
            ss_tot = float(np.sum((y_val_np[:, t_idx] - np.mean(y_val_np[:, t_idx])) ** 2))
            r2 = 1.0 - (ss_res / max(ss_tot, 1e-10))
            target_name = ["ka", "nu"][t_idx] if t_idx < 2 else f"output_{t_idx}"
            self.val_r2[target_name] = round(r2, 4)
        _avg_r2 = sum(self.val_r2.values()) / max(len(self.val_r2), 1)
        log.info("Training complete: %d epochs, final val_MSE=%.6f, val_R²=%s (avg=%.4f)",
                 epochs, history[-1]["val_loss"], self.val_r2, _avg_r2)
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict chromatographic parameters from features.

        Parameters
        ----------
        X : (n_samples, n_features) or (n_features,) feature array

        Returns
        -------
        predictions : (n_samples, n_outputs) numpy array
        """
        torch, _ = _get_torch()

        if not self.trained:
            raise RuntimeError("Model not trained. Call train_model() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_tensor).cpu().numpy()

        return pred

    def predict_single(self, features: np.ndarray) -> Dict[str, float]:
        """
        Predict for a single sample, returning a labeled dict.

        Parameters
        ----------
        features : (7,) feature vector

        Returns
        -------
        dict with 'ka', 'nu' and 'estimated_rt_min'
        """
        pred = self.predict(features)
        ka = float(pred[0, 0])
        nu = float(pred[0, 1])

        # Also estimate RT from predicted parameters
        rt_est = estimate_rt_from_sma(ka, nu)

        if self.n_outputs == 2:
            result = {"ka": ka, "nu": nu, "estimated_rt_min": rt_est}
            # v7.3.2: Attach validation R² for downstream quality gating
            if hasattr(self, "val_r2") and self.val_r2:
                _avg_r2 = sum(self.val_r2.values()) / max(len(self.val_r2), 1)
                result["val_r2"] = round(_avg_r2, 4)
            return result
        else:
            return {"RT_main_min": float(pred[0, 0])}


# ===========================================================================
# 6. SHAP Interpretability
# ===========================================================================

def compute_shap_values(
    model: ChromatographyMLP,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    max_background: int = 100,
) -> Dict[str, Any]:
    """
    Compute SHAP values for model predictions using KernelExplainer.

    Parameters
    ----------
    model          : Trained ChromatographyMLP instance
    X_background   : Background dataset for SHAP (unscaled features)
    X_explain      : Samples to explain (unscaled features)
    max_background : Max background samples (for performance)

    Returns
    -------
    dict with:
        - shap_values: SHAP value arrays (one per output)
        - base_values: Expected model output (base prediction)
        - feature_names: Feature name list
        - X_explain: The explained samples (for plotting)
    """
    import shap

    if not model.trained:
        raise RuntimeError("Model must be trained before computing SHAP values.")

    # Subsample background for performance
    if len(X_background) > max_background:
        idx = np.random.choice(len(X_background), max_background, replace=False)
        X_bg = X_background[idx]
    else:
        X_bg = X_background

    # Wrapper function that takes raw features and returns predictions
    def model_fn(X_raw):
        return model.predict(X_raw)

    explainer = shap.KernelExplainer(model_fn, X_bg)

    if X_explain.ndim == 1:
        X_explain = X_explain.reshape(1, -1)

    shap_values_raw = explainer.shap_values(X_explain, nsamples=200)

    # Normalize shap_values to a list of (n_explain, n_features) arrays,
    # one per output. SHAP may return:
    #   - list of arrays (older SHAP versions, multi-output)
    #   - 3D array (n_explain, n_features, n_outputs) (newer SHAP)
    #   - 2D array (n_explain, n_features) (single output)
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw
    elif isinstance(shap_values_raw, np.ndarray):
        if shap_values_raw.ndim == 3:
            # (n_explain, n_features, n_outputs) -> split on last axis
            shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
        else:
            # 2D: single output
            shap_values = [shap_values_raw]
    else:
        shap_values = [np.array(shap_values_raw)]

    result = {
        "shap_values": shap_values,
        "base_values": explainer.expected_value,
        "feature_names": FEATURE_NAMES,
        "X_explain": X_explain,
    }

    log.info("SHAP values computed for %d samples, %d outputs",
             len(X_explain), len(shap_values))
    return result


def plot_shap_waterfall(
    shap_result: Dict[str, Any],
    sample_idx: int = 0,
    output_idx: int = 0,
    output_name: str = "ka",
    max_display: int = 7,
) -> "matplotlib.figure.Figure":
    """
    Generate a SHAP waterfall plot for a single prediction.

    Parameters
    ----------
    shap_result  : Output from compute_shap_values()
    sample_idx   : Which sample to explain
    output_idx   : Which output to explain (0=ka, 1=nu)
    output_name  : Label for the output axis
    max_display  : Max features to display

    Returns
    -------
    matplotlib Figure object (for st.pyplot())
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    sv = shap_result["shap_values"][output_idx]
    base = shap_result["base_values"]
    if isinstance(base, (list, np.ndarray)):
        base_val = float(base[output_idx])
    else:
        base_val = float(base)

    feature_names = shap_result["feature_names"]
    sample_shap = sv[sample_idx]
    sample_features = shap_result["X_explain"][sample_idx]

    # Build a shap.Explanation object for the waterfall plot
    explanation = shap.Explanation(
        values=sample_shap,
        base_values=base_val,
        data=sample_features,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.sca(ax)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    ax.set_title(f"SHAP Explanation: {output_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_shap_summary(
    shap_result: Dict[str, Any],
    output_idx: int = 0,
    output_name: str = "ka",
) -> "matplotlib.figure.Figure":
    """
    Generate a SHAP beeswarm/summary plot for feature importance.

    Parameters
    ----------
    shap_result : Output from compute_shap_values()
    output_idx  : Which output to show
    output_name : Label for the plot title

    Returns
    -------
    matplotlib Figure object
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    sv = shap_result["shap_values"][output_idx]
    X_explain = shap_result["X_explain"]
    feature_names = shap_result["feature_names"]

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.sca(ax)
    shap.summary_plot(
        sv,
        X_explain,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
    )
    ax.set_title(f"SHAP Feature Importance: {output_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ===========================================================================
# 7. Convenience: Train + Predict + Explain Pipeline
# ===========================================================================

_CACHED_MODEL: Optional[ChromatographyMLP] = None
_CACHED_X_TRAIN: Optional[np.ndarray] = None


def get_trained_model(
    force_retrain: bool = False,
    n_samples: int = 500,
    epochs: int = 80,
) -> Tuple[ChromatographyMLP, np.ndarray, List[Dict[str, float]]]:
    """
    Get (or train) the cached ML model.

    Returns
    -------
    model     : Trained ChromatographyMLP
    X_train   : Training feature matrix (for SHAP background)
    history   : Training history list
    """
    global _CACHED_MODEL, _CACHED_X_TRAIN

    if _CACHED_MODEL is not None and _CACHED_MODEL.trained and not force_retrain:
        return _CACHED_MODEL, _CACHED_X_TRAIN, _CACHED_MODEL.training_history

    log.info("Training ML model (n=%d, epochs=%d)...", n_samples, epochs)

    X, y = generate_synthetic_dataset(n_samples=n_samples)
    model = ChromatographyMLP(n_inputs=7, n_outputs=2)
    history = model.train_model(X, y, epochs=epochs, verbose=True)

    _CACHED_MODEL = model
    _CACHED_X_TRAIN = X

    return model, X, history


def predict_and_explain(
    intent: Dict[str, Any],
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Full ML pipeline: extract features, predict, compute SHAP.

    Parameters
    ----------
    intent         : Parsed intent dict from app.py
    force_retrain  : Force model retraining

    Returns
    -------
    dict with:
        - prediction: {ka, nu, estimated_rt_min}
        - features: feature vector
        - feature_names: feature labels
        - shap_result: full SHAP output
        - training_history: loss over epochs
        - model_info: model metadata
    """
    model, X_train, history = get_trained_model(force_retrain=force_retrain)

    features = extract_features_from_intent(intent)
    prediction = model.predict_single(features)

    # Compute SHAP values
    shap_result = compute_shap_values(
        model=model,
        X_background=X_train,
        X_explain=features,
    )

    return {
        "prediction": prediction,
        "features": features.tolist(),
        "feature_names": FEATURE_NAMES,
        "shap_result": shap_result,
        "training_history": history,
        "model_info": {
            "architecture": "MLP (7->64->32->16->2)",
            "n_training_samples": len(X_train),
            "n_epochs": len(history),
            "final_train_mse": history[-1]["train_loss"],
            "final_val_mse": history[-1]["val_loss"],
            "target_rt_window": "15-20 min",
        },
    }


# ===========================================================================
# 8. Continuous Learning: Retraining with Expert Labels (v3.0)
# ===========================================================================

_TRAINING_RUN_COUNT: int = 0


def get_training_run_count() -> int:
    """Return the number of training runs (including initial + retrains)."""
    return _TRAINING_RUN_COUNT


def retrain_with_labels(
    labeled_X: np.ndarray,
    labeled_y: np.ndarray,
    n_synthetic: int = 500,
    epochs: int = 80,
    label_upsample: int = 3,
) -> Tuple[ChromatographyMLP, np.ndarray, List[Dict[str, float]]]:
    """
    Retrain the MLP by merging synthetic data with expert-labeled data.

    Expert labels are upsampled (repeated) to give them proportionally
    more influence during training, since they represent ground truth.

    Parameters
    ----------
    labeled_X : (n_labeled, 7) expert-labeled feature vectors
    labeled_y : (n_labeled, 2) expert-corrected ka/nu targets
    n_synthetic : Number of synthetic samples to generate
    epochs : Training epochs
    label_upsample : How many times to repeat labeled data

    Returns
    -------
    model     : Retrained ChromatographyMLP
    X_train   : Combined training feature matrix
    history   : Training history
    """
    global _CACHED_MODEL, _CACHED_X_TRAIN, _TRAINING_RUN_COUNT

    _TRAINING_RUN_COUNT += 1

    # Generate base synthetic data
    X_syn, y_syn = generate_synthetic_dataset(n_samples=n_synthetic)

    # Merge with labeled data (upsampled for emphasis)
    X_parts = [X_syn]
    y_parts = [y_syn]
    for _ in range(label_upsample):
        X_parts.append(labeled_X)
        y_parts.append(labeled_y)

    X_combined = np.concatenate(X_parts, axis=0)
    y_combined = np.concatenate(y_parts, axis=0)

    log.info("Retrain run #%d: %d synthetic + %d labeled (x%d) = %d total",
             _TRAINING_RUN_COUNT, n_synthetic, len(labeled_X),
             label_upsample, len(X_combined))

    # Train fresh model
    model = ChromatographyMLP(n_inputs=7, n_outputs=2)
    history = model.train_model(X_combined, y_combined, epochs=epochs, verbose=False)

    # Update global cache
    _CACHED_MODEL = model
    _CACHED_X_TRAIN = X_combined

    log.info("Retrain complete: val_MSE=%.6f (run #%d)",
             history[-1]["val_loss"], _TRAINING_RUN_COUNT)

    return model, X_combined, history


# ===========================================================================
# 9. Wet-Lab Supervised Predictor (M15: Data-Driven AI)
# ===========================================================================

# Default target names for wet-lab prediction
WETLAB_TARGET_NAMES = ["Exp_Aggregation_Percent", "Exp_Tm_MeltingTemp"]

# Cached wet-lab model
_CACHED_WETLAB_MODEL: Optional[Any] = None
_WETLAB_TRAINING_METRICS: Optional[Dict[str, Any]] = None


class WetLabPredictor:
    """
    XGBoost-based predictor trained on actual wet-lab experimental data.

    Replaces heuristic developability scoring with supervised learning
    from ingested CSV datasets (e.g., Jain-137 mAbs).

    Supports two feature modes:
        A) Classic 7-dim biophysical features (pI, MW, liabilities, etc.)
        B) N-dim PLM embeddings (e.g., 480-dim ESM-2 mean-pooled vectors)

    Output targets (configurable, default 2):
        1. Exp_Aggregation_Percent (% HMW by SEC)
        2. Exp_Tm_MeltingTemp (°C from DSF)
    """

    def __init__(self, n_targets: int = 2, target_names: Optional[List[str]] = None,
                 feature_names: Optional[List[str]] = None):
        self.n_targets = n_targets
        self.target_names = target_names or WETLAB_TARGET_NAMES[:n_targets]
        self.feature_names_list = feature_names or list(FEATURE_NAMES)
        self.models = []  # One XGBoost regressor per target
        self.scaler = FeatureScaler()
        self.trained = False
        self.training_metrics: Dict[str, Any] = {}
        self.feature_importances: List[Dict[str, float]] = []
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._feature_mode: str = "biophysical"  # "biophysical" or "plm"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2,
        xgb_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train XGBoost regressors on wet-lab data.

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix — 7-dim biophysical or
            N-dim PLM embeddings.
        y : (n_samples, n_targets) target matrix
        val_split : Fraction for validation
        xgb_params : Optional XGBoost hyperparameters

        Returns
        -------
        dict with per-target R², RMSE, and overall summary
        """
        try:
            import xgboost as xgb
            use_xgb = True
        except ImportError:
            use_xgb = False
            log.warning("XGBoost not available — using sklearn GradientBoosting fallback")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self._X_train = X
        self._y_train = y

        # Train/val split
        n = len(X_scaled)
        n_val = max(1, int(n * val_split))
        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        default_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
        }
        if xgb_params:
            default_params.update(xgb_params)

        self.models = []
        self.feature_importances = []
        per_target_metrics = {}

        for t in range(self.n_targets):
            target_name = self.target_names[t] if t < len(self.target_names) else f"target_{t}"

            if use_xgb:
                model = xgb.XGBRegressor(**default_params)
            else:
                # Fallback to sklearn
                try:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(
                        n_estimators=default_params["n_estimators"],
                        max_depth=default_params["max_depth"],
                        learning_rate=default_params["learning_rate"],
                        subsample=default_params["subsample"],
                        random_state=42,
                    )
                except ImportError:
                    log.error("Neither XGBoost nor sklearn available")
                    return {"status": "error", "message": "No ML backend available"}

            model.fit(X_train, y_train[:, t])
            self.models.append(model)

            # Validation metrics
            y_pred_val = model.predict(X_val)
            y_pred_train = model.predict(X_train)

            # R² score
            ss_res = np.sum((y_val[:, t] - y_pred_val) ** 2)
            ss_tot = np.sum((y_val[:, t] - np.mean(y_val[:, t])) ** 2)
            r2 = 1.0 - (ss_res / max(ss_tot, 1e-10))

            # RMSE
            rmse = float(np.sqrt(np.mean((y_val[:, t] - y_pred_val) ** 2)))
            train_rmse = float(np.sqrt(np.mean((y_train[:, t] - y_pred_train) ** 2)))

            # MAE
            mae = float(np.mean(np.abs(y_val[:, t] - y_pred_val)))

            per_target_metrics[target_name] = {
                "r2": round(float(r2), 4),
                "rmse": round(rmse, 4),
                "train_rmse": round(train_rmse, 4),
                "mae": round(mae, 4),
                "n_train": len(X_train),
                "n_val": len(X_val),
            }

            # Feature importance
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                fnames = self.feature_names_list
                importance_dict = {
                    fnames[i]: round(float(fi[i]), 4)
                    for i in range(min(len(fi), len(fnames)))
                }
                # For PLM embeddings (large dim), keep only top-20 by importance
                if len(importance_dict) > 30:
                    top_items = sorted(importance_dict.items(),
                                       key=lambda kv: kv[1], reverse=True)[:20]
                    importance_dict = dict(top_items)
                self.feature_importances.append(importance_dict)
            else:
                self.feature_importances.append({})

            log.info("WetLab %s: R²=%.4f, RMSE=%.4f, MAE=%.4f (n_train=%d, n_val=%d)",
                     target_name, r2, rmse, mae, len(X_train), len(X_val))

        self.trained = True
        self.training_metrics = {
            "status": "success",
            "per_target": per_target_metrics,
            "n_samples": n,
            "n_features": X.shape[1],
            "n_targets": self.n_targets,
            "target_names": self.target_names,
            "feature_names": self.feature_names_list,
            "feature_mode": self._feature_mode,
            "feature_importances": self.feature_importances,
            "model_type": "XGBoost" if use_xgb else "GradientBoosting",
        }

        return self.training_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict wet-lab targets from features.

        Parameters
        ----------
        X : (n_samples, 7) or (7,) feature array (unscaled)

        Returns
        -------
        predictions : (n_samples, n_targets) numpy array
        """
        if not self.trained:
            raise RuntimeError("WetLabPredictor not trained. Call train() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)

        preds = np.zeros((len(X_scaled), self.n_targets), dtype=np.float32)
        for t, model in enumerate(self.models):
            preds[:, t] = model.predict(X_scaled)

        return preds

    def predict_single(self, features: np.ndarray) -> Dict[str, float]:
        """
        Predict for a single sample, returning a labeled dict.

        Parameters
        ----------
        features : (7,) feature vector

        Returns
        -------
        dict with target_name -> predicted_value mappings
        """
        pred = self.predict(features)
        result = {}
        for t in range(self.n_targets):
            name = self.target_names[t] if t < len(self.target_names) else f"target_{t}"
            result[name] = round(float(pred[0, t]), 2)
        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Return training metrics summary."""
        return self.training_metrics


def train_wetlab_model(
    X: np.ndarray,
    y: np.ndarray,
    target_names: Optional[List[str]] = None,
    val_split: float = 0.2,
) -> Tuple[WetLabPredictor, Dict[str, Any]]:
    """
    Convenience function: train a wet-lab predictor and cache globally.

    Parameters
    ----------
    X : (n_samples, 7) biophysical features
    y : (n_samples, n_targets) wet-lab targets
    target_names : Names for targets
    val_split : Validation fraction

    Returns
    -------
    model   : Trained WetLabPredictor
    metrics : Training metrics dict
    """
    global _CACHED_WETLAB_MODEL, _WETLAB_TRAINING_METRICS

    n_targets = y.shape[1] if y.ndim > 1 else 1
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    predictor = WetLabPredictor(n_targets=n_targets, target_names=target_names)
    metrics = predictor.train(X, y, val_split=val_split)

    _CACHED_WETLAB_MODEL = predictor
    _WETLAB_TRAINING_METRICS = metrics

    # M19: Auto-persist to disk
    _save_model_joblib(predictor, MODEL_FILE_WETLAB)

    log.info("Wet-lab model trained, cached, and persisted: %d samples, %d targets",
             len(X), n_targets)

    return predictor, metrics


def train_plm_wetlab_model(
    X: np.ndarray,
    y: np.ndarray,
    target_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    val_split: float = 0.2,
) -> Tuple["WetLabPredictor", Dict[str, Any]]:
    """
    Train a PLM-embedding-based wet-lab predictor and cache globally.

    This is the high-dimensional counterpart to ``train_wetlab_model()``.
    It trains on N-dimensional PLM embedding features (e.g., 480-dim ESM-2)
    instead of the classic 7-dim biophysical features.

    Parameters
    ----------
    X : (n_samples, embed_dim) PLM embedding features
    y : (n_samples, n_targets) wet-lab targets
    target_names : Names for targets
    feature_names : Names for embedding dimensions (e.g., ["plm_0", ...])
    val_split : Validation fraction

    Returns
    -------
    model   : Trained WetLabPredictor
    metrics : Training metrics dict
    """
    global _CACHED_PLM_MODEL, _PLM_TRAINING_METRICS

    n_targets = y.shape[1] if y.ndim > 1 else 1
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    embed_dim = X.shape[1]
    if feature_names is None:
        feature_names = [f"plm_{i}" for i in range(embed_dim)]

    predictor = WetLabPredictor(
        n_targets=n_targets,
        target_names=target_names,
        feature_names=feature_names,
    )
    predictor._feature_mode = "plm"

    # For high-dimensional PLM embeddings, use more regularization
    plm_xgb_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.3,   # Sample fewer features per tree
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": 42,
    }

    metrics = predictor.train(X, y, val_split=val_split, xgb_params=plm_xgb_params)

    _CACHED_PLM_MODEL = predictor
    _PLM_TRAINING_METRICS = metrics

    # Persist to disk
    _save_model_joblib(predictor, MODEL_FILE_WETLAB_PLM)

    log.info("PLM wet-lab model trained: %d samples × %d-dim embeddings, %d targets",
             len(X), embed_dim, n_targets)

    return predictor, metrics


def get_plm_model() -> Optional["WetLabPredictor"]:
    """Return the cached PLM-based wet-lab model, or None if not trained."""
    return _CACHED_PLM_MODEL


def get_plm_metrics() -> Optional[Dict[str, Any]]:
    """Return cached PLM-based wet-lab training metrics."""
    return _PLM_TRAINING_METRICS


def get_wetlab_model() -> Optional[WetLabPredictor]:
    """Return the cached wet-lab model, or None if not trained."""
    return _CACHED_WETLAB_MODEL


def get_wetlab_metrics() -> Optional[Dict[str, Any]]:
    """Return cached wet-lab training metrics."""
    return _WETLAB_TRAINING_METRICS


def predict_wetlab(features: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Predict wet-lab metrics using the cached trained model.

    Returns dict with Exp_Aggregation_Percent, Exp_Tm_MeltingTemp, etc.
    Returns None if no model is trained.
    """
    model = get_wetlab_model()
    if model is None or not model.trained:
        return None
    return model.predict_single(features)


def evaluate_variant_wetlab(
    sequence: str,
    feature_override: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a variant sequence through the wet-lab model.

    Extracts biophysical features from the sequence, runs prediction,
    and returns the predicted wet-lab metrics.

    Parameters
    ----------
    sequence : Amino acid sequence
    feature_override : Optional dict to override computed features

    Returns
    -------
    dict with predicted metrics, or None if model not trained
    """
    model = get_wetlab_model()
    if model is None or not model.trained:
        return None

    # Extract features from sequence
    import re as _re
    seq = _re.sub(r'[^A-Z]', '', sequence.upper())
    if len(seq) < 50:
        return None

    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        pa = ProteinAnalysis(seq)
        pI = pa.isoelectric_point()
        mw_kda = pa.molecular_weight() / 1000.0
        gravy = pa.gravy()
    except Exception:
        pI = 8.0
        mw_kda = 150.0
        gravy = -0.3

    hydrophobicity = max(0.0, min(1.0, (gravy + 2.0) / 4.0))
    deam_sites = len(_re.findall(r"N[GS]", seq))
    ox_sites = seq.count("M") + seq.count("W")
    acidic = seq.count("D") + seq.count("E")
    basic = seq.count("K") + seq.count("R") + seq.count("H")

    features = extract_features(
        pI=feature_override.get("pI", pI) if feature_override else pI,
        mw=feature_override.get("mw_kda", mw_kda) if feature_override else mw_kda,
        deam_sites=int(feature_override.get("deam_sites", deam_sites)) if feature_override else deam_sites,
        ox_sites=int(feature_override.get("ox_sites", ox_sites)) if feature_override else ox_sites,
        acidic_residues=int(feature_override.get("acidic", acidic)) if feature_override else acidic,
        basic_residues=int(feature_override.get("basic", basic)) if feature_override else basic,
        hydrophobicity=feature_override.get("hydrophobicity", hydrophobicity) if feature_override else hydrophobicity,
    )

    pred = model.predict_single(features)

    return {
        "predictions": pred,
        "features": {
            "pI": round(pI, 2),
            "mw_kda": round(mw_kda, 1),
            "hydrophobicity": round(hydrophobicity, 3),
            "deam_sites": deam_sites,
            "ox_sites": ox_sites,
        },
        "model_type": model.training_metrics.get("model_type", "XGBoost"),
    }


# ===========================================================================
# Potency / Affinity Predictor (Milestone 16)
# ===========================================================================

POTENCY_TARGET_NAMES = ["Predicted_Potency_Score"]

_CACHED_POTENCY_MODEL: Optional[Any] = None
_POTENCY_TRAINING_METRICS: Optional[Dict[str, Any]] = None


class PotencyPredictor:
    """
    XGBoost-based predictor for antibody potency / binding affinity.

    Trained on ELISA OD and/or Kd binding data from Early Discovery screens.
    Maps biophysical features (7-dim) to a normalized Potency Score (0-1).

    Input features (7-dim, same as ChromatographyMLP):
        pI, MW_kDa, deam_sites, ox_sites, acidic_residues,
        basic_residues, hydrophobicity

    Output target:
        Predicted_Potency_Score (0-1, higher = more potent)
    """

    def __init__(self, target_names: Optional[List[str]] = None):
        self.target_names = target_names or POTENCY_TARGET_NAMES
        self.n_targets = len(self.target_names)
        self.models = []
        self.scaler = FeatureScaler()
        self.trained = False
        self.training_metrics: Dict[str, Any] = {}
        self.feature_importances: List[Dict[str, float]] = []

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2,
        xgb_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train XGBoost regressors on potency/affinity data.

        Parameters
        ----------
        X : (n_samples, 7) feature matrix
        y : (n_samples, n_targets) target matrix  (potency scores in [0,1])
        val_split : Fraction for validation
        xgb_params : Optional XGBoost hyperparameters

        Returns
        -------
        dict with per-target R², RMSE, and overall summary
        """
        try:
            import xgboost as xgb
            use_xgb = True
        except ImportError:
            use_xgb = False
            log.warning("XGBoost not available for potency — using sklearn fallback")

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train/val split
        n = len(X_scaled)
        n_val = max(1, int(n * val_split))
        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        default_params = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "random_state": 42,
        }
        if xgb_params:
            default_params.update(xgb_params)

        self.models = []
        self.feature_importances = []
        per_target_metrics = {}

        for t in range(self.n_targets):
            target_name = self.target_names[t] if t < len(self.target_names) else f"potency_{t}"

            if use_xgb:
                import xgboost as xgb
                model = xgb.XGBRegressor(**default_params)
            else:
                try:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(
                        n_estimators=default_params["n_estimators"],
                        max_depth=default_params["max_depth"],
                        learning_rate=default_params["learning_rate"],
                        subsample=default_params["subsample"],
                        random_state=42,
                    )
                except ImportError:
                    log.error("Neither XGBoost nor sklearn available for potency")
                    return {"status": "error", "message": "No ML backend available"}

            model.fit(X_train, y_train[:, t])
            self.models.append(model)

            # Validation metrics
            y_pred_val = model.predict(X_val)
            y_pred_train = model.predict(X_train)

            ss_res = np.sum((y_val[:, t] - y_pred_val) ** 2)
            ss_tot = np.sum((y_val[:, t] - np.mean(y_val[:, t])) ** 2)
            r2 = 1.0 - (ss_res / max(ss_tot, 1e-10))
            rmse = float(np.sqrt(np.mean((y_val[:, t] - y_pred_val) ** 2)))
            train_rmse = float(np.sqrt(np.mean((y_train[:, t] - y_pred_train) ** 2)))
            mae = float(np.mean(np.abs(y_val[:, t] - y_pred_val)))

            per_target_metrics[target_name] = {
                "r2": round(float(r2), 4),
                "rmse": round(rmse, 4),
                "train_rmse": round(train_rmse, 4),
                "mae": round(mae, 4),
                "n_train": len(X_train),
                "n_val": len(X_val),
            }

            # Feature importance
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                importance_dict = {
                    FEATURE_NAMES[i]: round(float(fi[i]), 4)
                    for i in range(min(len(fi), len(FEATURE_NAMES)))
                }
                self.feature_importances.append(importance_dict)
            else:
                self.feature_importances.append({})

            log.info("Potency %s: R²=%.4f, RMSE=%.4f, MAE=%.4f",
                     target_name, r2, rmse, mae)

        self.trained = True
        self.training_metrics = {
            "status": "success",
            "per_target": per_target_metrics,
            "n_samples": n,
            "n_targets": self.n_targets,
            "target_names": self.target_names,
            "feature_names": FEATURE_NAMES,
            "feature_importances": self.feature_importances,
            "model_type": "XGBoost" if use_xgb else "GradientBoosting",
        }

        return self.training_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict potency scores from features."""
        if not self.trained:
            raise RuntimeError("PotencyPredictor not trained. Call train() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)

        preds = np.zeros((len(X_scaled), self.n_targets), dtype=np.float32)
        for t, model in enumerate(self.models):
            preds[:, t] = np.clip(model.predict(X_scaled), 0.0, 1.0)

        return preds

    def predict_single(self, features: np.ndarray) -> Dict[str, float]:
        """Predict for a single sample, returning labeled dict."""
        preds = self.predict(features)
        return {
            self.target_names[t]: round(float(preds[0, t]), 4)
            for t in range(self.n_targets)
        }


def generate_mock_potency_dataset(
    n_samples: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mock potency training data.

    Creates synthetic biophysical features with correlated potency scores
    that mimic CDR-driven binding patterns.

    Returns
    -------
    X : (n_samples, 7) biophysical features
    y : (n_samples, 1) potency scores in [0, 1]
    """
    rng = np.random.RandomState(seed)

    X = np.column_stack([
        rng.uniform(6.5, 9.5, n_samples),     # pI
        rng.uniform(140, 160, n_samples),       # MW_kDa
        rng.randint(0, 8, n_samples).astype(float),  # deam_sites
        rng.randint(0, 6, n_samples).astype(float),  # ox_sites
        rng.randint(15, 65, n_samples).astype(float), # acidic
        rng.randint(20, 75, n_samples).astype(float), # basic
        rng.uniform(0.15, 0.65, n_samples),    # hydrophobicity
    ]).astype(np.float32)

    # Potency score: moderate hydrophobicity, balanced charge, few liabilities → high
    hydro = X[:, 6]
    charge_balance = np.minimum(X[:, 4], X[:, 5]) / np.maximum(np.maximum(X[:, 4], X[:, 5]), 1)
    liability_penalty = (X[:, 2] + X[:, 3]) / 15.0

    potency = (
        0.3
        + 0.35 * (1.0 - np.abs(hydro - 0.42) / 0.4)
        + 0.20 * charge_balance
        - 0.15 * liability_penalty
        + rng.normal(0, 0.08, n_samples)
    )
    potency = np.clip(potency, 0.05, 0.95).reshape(-1, 1).astype(np.float32)

    return X, potency


def train_potency_model(
    X: np.ndarray,
    y: np.ndarray,
    target_names: Optional[List[str]] = None,
    val_split: float = 0.2,
) -> Tuple[PotencyPredictor, Dict[str, Any]]:
    """
    Train a potency predictor and cache globally.

    Parameters
    ----------
    X : (n_samples, 7) biophysical features
    y : (n_samples, n_targets) potency targets
    target_names : Names for targets
    val_split : Validation fraction

    Returns
    -------
    model   : Trained PotencyPredictor
    metrics : Training metrics dict
    """
    global _CACHED_POTENCY_MODEL, _POTENCY_TRAINING_METRICS

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    predictor = PotencyPredictor(target_names=target_names)
    metrics = predictor.train(X, y, val_split=val_split)

    _CACHED_POTENCY_MODEL = predictor
    _POTENCY_TRAINING_METRICS = metrics

    # M19: Auto-persist to disk
    _save_model_joblib(predictor, MODEL_FILE_POTENCY)

    log.info("Potency model trained, cached, and persisted: %d samples", len(X))

    return predictor, metrics


def get_potency_model() -> Optional[PotencyPredictor]:
    """Return the cached potency model, or None if not trained."""
    return _CACHED_POTENCY_MODEL


def get_potency_metrics() -> Optional[Dict[str, Any]]:
    """Return cached potency training metrics."""
    return _POTENCY_TRAINING_METRICS


def predict_potency(features: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Predict potency using the cached trained model.

    Returns dict with Predicted_Potency_Score, or None if no model is trained.
    """
    model = get_potency_model()
    if model is None or not model.trained:
        return None
    return model.predict_single(features)


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
    print("  ProtePilot — ML Predictor v4.0 Test")
    print("  (Data-Driven AI — Supervised Wet-Lab Training)")
    print("=" * 60)

    # Generate data
    X, y = generate_synthetic_dataset(n_samples=300)
    print(f"\nSynthetic data: X={X.shape}, y={y.shape}")
    print(f"  pI range:  [{X[:,0].min():.1f}, {X[:,0].max():.1f}]")
    print(f"  hydro range: [{X[:,6].min():.2f}, {X[:,6].max():.2f}]")
    print(f"  ka range:  [{y[:,0].min():.3f}, {y[:,0].max():.3f}]")
    print(f"  nu range:  [{y[:,1].min():.3f}, {y[:,1].max():.3f}]")

    # Verify RT distribution
    rt_estimates = [estimate_rt_from_sma(y[i, 0], y[i, 1]) for i in range(len(y))]
    rt_arr = np.array(rt_estimates)
    in_15_20 = np.sum((rt_arr >= 15) & (rt_arr <= 20))
    print(f"  RT distribution: mean={rt_arr.mean():.1f} min, "
          f"std={rt_arr.std():.1f} min, "
          f"in [15-20]: {in_15_20}/{len(y)} ({100*in_15_20/len(y):.0f}%)")

    # Train model
    model = ChromatographyMLP(n_inputs=7, n_outputs=2)
    history = model.train_model(X, y, epochs=50, verbose=True)
    print(f"\nFinal: train_MSE={history[-1]['train_loss']:.6f}  "
          f"val_MSE={history[-1]['val_loss']:.6f}")

    # Predict for a standard mAb
    features = extract_features(pI=8.5, mw=150.0, deam_sites=1, ox_sites=1,
                                acidic_residues=40, basic_residues=50,
                                hydrophobicity=0.35)
    pred = model.predict_single(features)
    print(f"\nPrediction (pI=8.5, MW=150, h=0.35):")
    print(f"  ka={pred['ka']:.4f}, nu={pred['nu']:.3f}, est_RT={pred['estimated_rt_min']:.1f} min")

    # SHAP
    print("\nComputing SHAP values...")
    shap_result = compute_shap_values(model, X, features)
    for i, name in enumerate(TARGET_NAMES_DUAL):
        sv = shap_result["shap_values"][i][0]
        print(f"\n  SHAP for {name}:")
        for fname, sval in zip(FEATURE_NAMES, sv):
            print(f"    {fname:20s}: {sval:+.4f}")

    # Test WetLab Predictor (v4.0)
    print("\n" + "=" * 60)
    print("  WetLabPredictor v4.0 Test")
    print("=" * 60)

    # Generate mock training data for wet-lab model
    rng_wl = np.random.RandomState(42)
    n_wl = 50
    X_wl = np.column_stack([
        rng_wl.uniform(6.5, 9.5, n_wl),   # pI
        rng_wl.uniform(140, 155, n_wl),     # MW
        rng_wl.randint(1, 6, n_wl).astype(float),  # deam
        rng_wl.randint(1, 6, n_wl).astype(float),  # ox
        rng_wl.randint(20, 60, n_wl).astype(float),  # acidic
        rng_wl.randint(25, 70, n_wl).astype(float),  # basic
        rng_wl.uniform(0.2, 0.55, n_wl),   # hydro
    ]).astype(np.float32)

    # Correlated targets
    agg_target = 3.0 + (X_wl[:, 6] - 0.35) * 30 + X_wl[:, 2] * 0.8 + rng_wl.normal(0, 1.5, n_wl)
    tm_target = 70.0 - (X_wl[:, 6] - 0.35) * 15 - X_wl[:, 2] * 0.4 + rng_wl.normal(0, 1.0, n_wl)
    y_wl = np.column_stack([agg_target, tm_target]).astype(np.float32)

    wl_model, wl_metrics = train_wetlab_model(
        X_wl, y_wl,
        target_names=["Exp_Aggregation_Percent", "Exp_Tm_MeltingTemp"],
    )
    print(f"\nWetLab model type: {wl_metrics.get('model_type')}")
    for tname, tm in wl_metrics.get("per_target", {}).items():
        print(f"  {tname}: R²={tm['r2']:.4f}, RMSE={tm['rmse']:.4f}, MAE={tm['mae']:.4f}")

    # Test prediction
    test_feat = extract_features(pI=8.5, mw=150.0, deam_sites=2, ox_sites=3,
                                 acidic_residues=40, basic_residues=50,
                                 hydrophobicity=0.35)
    wl_pred = wl_model.predict_single(test_feat)
    print(f"\nWetLab Prediction (pI=8.5, MW=150, h=0.35):")
    for k, v in wl_pred.items():
        print(f"  {k}: {v}")

    # Test PotencyPredictor (v5.0)
    print("\n" + "=" * 60)
    print("  PotencyPredictor v5.0 Test")
    print("=" * 60)

    X_pot, y_pot = generate_mock_potency_dataset(n_samples=200)
    pot_model, pot_metrics = train_potency_model(X_pot, y_pot)
    print(f"\nPotency model type: {pot_metrics.get('model_type')}")
    for tname, tm in pot_metrics.get("per_target", {}).items():
        print(f"  {tname}: R²={tm['r2']:.4f}, RMSE={tm['rmse']:.4f}, MAE={tm['mae']:.4f}")

    pot_pred = pot_model.predict_single(test_feat)
    print(f"\nPotency Prediction (pI=8.5, MW=150, h=0.35):")
    for k, v in pot_pred.items():
        print(f"  {k}: {v}")

    print("\nML Predictor v5.0 test complete")
