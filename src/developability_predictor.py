"""
developability_predictor.py  ·  ProtePilot — Milestone 8
===========================================================
XGBoost Multi-Output Developability Predictor with SHAP

Version   : 1.0
Author    : Di (ProtePilot)
Depends   : xgboost (optional), shap (optional), numpy, sklearn (optional)

Architecture
------------------------------------------------------------
Three independent XGBRegressor models predict:
    1. Aggregation Risk      [0, 1]
    2. Thermal Stability     [0, 1]  (higher = more stable)
    3. Viscosity Risk        [0, 1]

Input features (967-dim):
    - pLM embedding (960-dim) from ESM-2 or mock
    - Biophysical features (7-dim): pI, MW, deam, ox, acidic, basic, hydro

Composite Developability Score:
    Score = 0.4 * agg_risk + 0.3 * (1 - stability) + 0.3 * viscosity_risk
    Grade: Low (<0.3), Medium (0.3-0.6), High (>0.6)

Mock Training Dataset
------------------------------------------------------------
Built-in mock dataset based on published properties of:
  - NISTmAb (NIST RM 8671)
  - Adalimumab (Humira)
  - Rituximab (Rituxan)
  + ~22 synthetic variants with realistic noise

SHAP Explainability
------------------------------------------------------------
TreeExplainer provides per-feature SHAP values for each target.
Actionable advice maps top contributing features to domain-specific
recommendations (e.g., reduce hydrophobicity in CDR3).

References
------------------------------------------------------------
  XGBoost: Chen & Guestrin, KDD 2016
  SHAP TreeExplainer: Lundberg et al., NeurIPS 2017
  NISTmAb: Schiel et al., Anal. Bioanal. Chem. 410 (2018)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ProtePilot.Developability")

# ===========================================================================
# Model Persistence — mirrors ml_predictor.py pattern (joblib save/load)
# ===========================================================================
MODEL_FILE_DEVELOPABILITY = "developability_xgb_models.pkl"

_MODELS_DIR: Optional[str] = None


def _get_models_dir() -> str:
    """Locate or create the models/ directory under project root."""
    global _MODELS_DIR
    if _MODELS_DIR is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _MODELS_DIR = os.path.join(root, "models")
    os.makedirs(_MODELS_DIR, exist_ok=True)
    return _MODELS_DIR


def _save_developability_models(models_dict: Dict[str, Any], X_train: Optional[np.ndarray] = None) -> Optional[str]:
    """Save trained XGBoost developability models to disk with joblib."""
    try:
        import joblib
        path = os.path.join(_get_models_dir(), MODEL_FILE_DEVELOPABILITY)
        payload = {
            "models": models_dict,
            "X_train": X_train,
            "version": "1.0",
        }
        joblib.dump(payload, path)
        log.info("Developability models saved: %s (%d bytes)", path, os.path.getsize(path))
        return path
    except Exception as e:
        log.warning("Developability model save failed: %s", e)
        return None


def _load_developability_models() -> Optional[Dict[str, Any]]:
    """Load trained XGBoost developability models from disk."""
    try:
        import joblib
        path = os.path.join(_get_models_dir(), MODEL_FILE_DEVELOPABILITY)
        if os.path.exists(path):
            payload = joblib.load(path)
            log.info("Developability models loaded: %s", path)
            return payload
        return None
    except Exception as e:
        log.warning("Developability model load failed: %s", e)
        return None


# ===========================================================================
# 1. Mock Dataset — Public mAb Reference Data
# ===========================================================================

# Feature indices for biophysical features (appended after 960-dim embedding)
BIOPHYS_NAMES = ["pI", "MW_kDa", "deam_sites", "ox_sites",
                 "acidic_residues", "basic_residues", "hydrophobicity"]
N_EMBED_DIM = 960
N_BIOPHYS = 7
N_TOTAL_FEATURES = N_EMBED_DIM + N_BIOPHYS  # 967

# Developability target names
TARGET_NAMES = ["agg_risk", "stability", "viscosity_risk"]

# Risk score weights for composite developability score
RISK_WEIGHTS = {"agg": 0.40, "stability": 0.30, "viscosity": 0.30}

# Published/estimated properties for reference mAbs
REFERENCE_MABS = [
    {
        "name": "NISTmAb",
        # pI 9.15: assembled IgG (2×HC + 2×LC), not single chain.
        # Ref: Schiel et al., Anal. Bioanal. Chem. 410 (2018); NIST RM 8671 CoA.
        "pI": 9.15, "MW_kDa": 148.0, "hydrophobicity": 0.34,
        "deam_sites": 1, "ox_sites": 1,
        "acidic_residues": 38, "basic_residues": 48,
        "agg_risk": 0.12, "stability": 0.88, "viscosity_risk": 0.08,
        "vh_seq": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
        "vl_seq": "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCLQHNSYPWTFGQGTKVEIK",
    },
    {
        "name": "Adalimumab",
        # pI 8.72: assembled IgG. Ref: Vlasak & Ionescu, mAbs 3 (2011)
        "pI": 8.72, "MW_kDa": 148.0, "hydrophobicity": 0.38,
        "deam_sites": 2, "ox_sites": 2,
        "acidic_residues": 42, "basic_residues": 46,
        "agg_risk": 0.25, "stability": 0.78, "viscosity_risk": 0.22,
        "vh_seq": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
        "vl_seq": "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK",
    },
    {
        "name": "Rituximab",
        # pI 9.40: assembled IgG. Ref: GE Healthcare IEF data; clinical isoform range 8.8-9.5
        "pI": 9.40, "MW_kDa": 145.0, "hydrophobicity": 0.32,
        "deam_sites": 1, "ox_sites": 1,
        "acidic_residues": 36, "basic_residues": 50,
        "agg_risk": 0.18, "stability": 0.83, "viscosity_risk": 0.12,
        "vh_seq": "QVQLQQPGAELVKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARSTYYGGDWYFNVWGAGTTVTVSA",
        "vl_seq": "QIVLSQSPAILSASPGEKVTMTCRASSSVSYIHWFQQKPGSSPKPWIYATSNLASGVPVRFSGSGSGTSYSLTISRVEAEDAATYYCQQWTSNPPTFGGGTKLEIK",
    },
]


# IgG reference statistics for OOD detection
# Default baseline — overridden by dynamic OOD baseline when training data available.
# See src/ood_baseline.py for dynamic computation.
IGG_REFERENCE_STATS = {
    "gravy": {"mean": -0.40, "std": 0.15, "min": -0.8, "max": 0.1},
    "cys_count_per_100": {"mean": 3.5, "std": 0.8, "min": 1.5, "max": 6.0},
    "length": {"mean": 450, "std": 100, "min": 100, "max": 1400},
    "pI": {"mean": 8.2, "std": 0.8, "min": 5.5, "max": 10.0},
    "MW_kDa": {"mean": 148, "std": 30, "min": 12, "max": 200},
}


def _generate_mock_dataset(
    n_synthetic: int = 22,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate mock training dataset with real mAb references + synthetic variants.

    Returns list of dicts with biophysical features and target labels.
    """
    rng = np.random.RandomState(seed)
    dataset = list(REFERENCE_MABS)  # copy references

    for i in range(n_synthetic):
        # Base on a random reference mAb
        base = REFERENCE_MABS[i % len(REFERENCE_MABS)]

        # Add realistic perturbations
        variant = {
            "name": f"Synthetic_{i+1}",
            "pI": base["pI"] + rng.normal(0, 0.3),
            "MW_kDa": base["MW_kDa"] + rng.normal(0, 5),
            "hydrophobicity": np.clip(base["hydrophobicity"] + rng.normal(0, 0.08), 0.1, 0.7),
            "deam_sites": max(0, base["deam_sites"] + rng.randint(-1, 3)),
            "ox_sites": max(0, base["ox_sites"] + rng.randint(-1, 3)),
            "acidic_residues": max(15, base["acidic_residues"] + rng.randint(-10, 10)),
            "basic_residues": max(20, base["basic_residues"] + rng.randint(-10, 10)),
            # Targets with correlated perturbations
            "agg_risk": np.clip(base["agg_risk"] + rng.normal(0, 0.12), 0.02, 0.95),
            "stability": np.clip(base["stability"] + rng.normal(0, 0.10), 0.30, 0.98),
            "viscosity_risk": np.clip(base["viscosity_risk"] + rng.normal(0, 0.10), 0.02, 0.90),
            "vh_seq": base["vh_seq"],  # reuse reference sequences
            "vl_seq": base["vl_seq"],
        }

        # Correlate targets with features (physical reality)
        # Higher hydrophobicity -> higher aggregation risk
        variant["agg_risk"] = np.clip(
            variant["agg_risk"] + 0.3 * (variant["hydrophobicity"] - 0.35),
            0.02, 0.95,
        )
        # More PTM sites -> lower stability
        ptm_penalty = 0.02 * (variant["deam_sites"] + variant["ox_sites"])
        variant["stability"] = np.clip(variant["stability"] - ptm_penalty, 0.30, 0.98)
        # Higher MW -> higher viscosity risk
        variant["viscosity_risk"] = np.clip(
            variant["viscosity_risk"] + 0.001 * (variant["MW_kDa"] - 148.0),
            0.02, 0.90,
        )

        dataset.append(variant)

    return dataset


def build_feature_matrix(
    dataset: List[Dict[str, Any]],
    embedder=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix and target matrix from mock dataset.

    Parameters
    ----------
    dataset  : List of dicts with biophysical features and targets
    embedder : ESM2Embedder instance (optional, uses mock if None)

    Returns
    -------
    X : (n_samples, 647) feature matrix
    y : (n_samples, 3) target matrix [agg_risk, stability, viscosity_risk]
    """
    from src.pLM_embedder import mock_antibody_embedding, mock_embedding

    X_list = []
    y_list = []

    for sample in dataset:
        # Get embedding
        vh = sample.get("vh_seq", "")
        vl = sample.get("vl_seq", "")
        if embedder is not None and vh and vl:
            embedding = embedder.embed_antibody(vh, vl)
        elif vh and vl:
            embedding = mock_antibody_embedding(vh, vl)
        else:
            # Use combined or zero embedding
            combined = vh + vl
            if combined:
                single = mock_embedding(combined)
                embedding = np.concatenate([single, single])
            else:
                embedding = np.zeros(N_EMBED_DIM, dtype=np.float32)

        # Biophysical features
        biophys = np.array([
            sample.get("pI", 8.0),
            sample.get("MW_kDa", 150.0),
            float(sample.get("deam_sites", 1)),
            float(sample.get("ox_sites", 1)),
            float(sample.get("acidic_residues", 40)),
            float(sample.get("basic_residues", 50)),
            sample.get("hydrophobicity", 0.35),
        ], dtype=np.float32)

        features = np.concatenate([embedding, biophys])
        X_list.append(features)

        targets = [
            sample.get("agg_risk", 0.2),
            sample.get("stability", 0.8),
            sample.get("viscosity_risk", 0.15),
        ]
        y_list.append(targets)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ===========================================================================
# 2. Kyte-Doolittle Hydrophobicity Scale for GRAVY Calculation
# ===========================================================================

KYTE_DOOLITTLE_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


def compute_gravy(sequence: str) -> float:
    """
    Compute GRAVY (Grand Average of Hydropathy) using Kyte-Doolittle scale.

    GRAVY = sum of KD values / sequence length
    """
    if not sequence:
        return 0.0
    gravy_sum = sum(KYTE_DOOLITTLE_SCALE.get(aa.upper(), 0.0) for aa in sequence)
    return gravy_sum / len(sequence)


# ===========================================================================
# 2. Rule-Based Fallback Predictor (when XGBoost unavailable)
# ===========================================================================

def _rule_based_predict(biophysical: np.ndarray) -> Dict[str, float]:
    """
    Simple rule-based developability prediction as XGBoost fallback.

    Uses hand-tuned heuristics based on published structure-property
    relationships for therapeutic antibodies.
    """
    pI = biophysical[0]
    mw = biophysical[1]
    deam = biophysical[2]
    ox = biophysical[3]
    acidic = biophysical[4]
    basic = biophysical[5]
    hydro = biophysical[6]

    # Aggregation risk: driven by hydrophobicity and charge asymmetry
    charge_asym = abs(acidic - basic) / max(acidic + basic, 1)
    agg_risk = np.clip(0.15 + 0.5 * (hydro - 0.3) + 0.2 * charge_asym, 0.02, 0.95)

    # Stability: driven by PTM sites and pI distance from physiological pH
    ptm_penalty = 0.03 * (deam + ox)
    pi_penalty = 0.02 * abs(pI - 7.4)
    stability = np.clip(0.85 - ptm_penalty - pi_penalty, 0.30, 0.98)

    # Viscosity risk: driven by MW and charge patches
    viscosity_risk = np.clip(
        0.05 + 0.002 * (mw - 148.0) + 0.1 * charge_asym + 0.15 * max(0, hydro - 0.4),
        0.02, 0.90,
    )

    return {
        "agg_risk": float(round(agg_risk, 4)),
        "stability": float(round(stability, 4)),
        "viscosity_risk": float(round(viscosity_risk, 4)),
    }


def _apply_sequence_composition_adjustment(
    predictions: Dict[str, float],
    sequence: Optional[str] = None,
) -> Dict[str, float]:
    """
    Adjust developability predictions based on sequence composition analysis.

    Addresses the limitation that rule-based (and even XGBoost) models may
    not sufficiently penalize non-antibody or pathological sequences that
    have normal biophysical properties (e.g., normal GRAVY) but abnormal
    amino acid composition (e.g., aromatic overload, repeat content).
    """
    if not sequence or len(sequence) < 20:
        return predictions

    seq = sequence.upper()
    n = len(seq)
    adjusted = dict(predictions)

    # 1. Aromatic overload (W+Y+F): typical IgG ~ 8-12%, >20% is pathological
    aromatic_frac = (seq.count("W") + seq.count("Y") + seq.count("F")) / n
    if aromatic_frac > 0.15:
        excess = aromatic_frac - 0.10  # penalty ramps from 10%
        agg_boost = min(0.6, excess * 4.0)  # up to +0.6
        stab_penalty = min(0.5, excess * 3.0)  # up to -0.5
        adjusted["agg_risk"] = float(np.clip(adjusted["agg_risk"] + agg_boost, 0.02, 0.95))
        adjusted["stability"] = float(np.clip(adjusted["stability"] - stab_penalty, 0.10, 0.98))
        adjusted["viscosity_risk"] = float(np.clip(
            adjusted["viscosity_risk"] + min(0.3, excess * 2.0), 0.02, 0.90))

    # 2. Repeat / low-complexity detection
    repeat_score = 0.0
    for i in range(min(n - 2, 500)):
        tri = seq[i:i + 3]
        count = seq.count(tri)
        if count >= 4:
            repeat_score = max(repeat_score, count / max(n / 3.0, 1.0))
    if repeat_score > 0.25:
        rp = min(0.5, (repeat_score - 0.20) * 2.5)
        adjusted["agg_risk"] = float(np.clip(adjusted["agg_risk"] + rp, 0.02, 0.95))
        adjusted["stability"] = float(np.clip(adjusted["stability"] - rp * 0.8, 0.10, 0.98))

    # 3. Cysteine anomaly (typical IgG: 2-4% Cys for disulfides)
    cys_frac = seq.count("C") / n
    if cys_frac > 0.08:
        cp = min(0.3, (cys_frac - 0.06) * 5.0)
        adjusted["agg_risk"] = float(np.clip(adjusted["agg_risk"] + cp, 0.02, 0.95))
    elif n > 200 and cys_frac < 0.005:
        # No Cys = no disulfides = poor folding
        adjusted["stability"] = float(np.clip(adjusted["stability"] - 0.2, 0.10, 0.98))

    # 4. Unusual MW for antibody (very small fragments only)
    mw_approx = n * 0.11  # rough kDa estimate
    if mw_approx < 5.0:
        adjusted["stability"] = float(np.clip(adjusted["stability"] - 0.15, 0.10, 0.98))
    elif mw_approx > 250.0:
        adjusted["viscosity_risk"] = float(np.clip(adjusted["viscosity_risk"] + 0.15, 0.02, 0.90))

    # Round adjusted values
    for k in ("agg_risk", "stability", "viscosity_risk"):
        if k in adjusted:
            adjusted[k] = round(adjusted[k], 4)

    return adjusted


# ===========================================================================
# 2b. Out-of-Distribution (OOD) Anomaly Detection
# ===========================================================================

def compute_ood_flags(
    sequence: str,
    pI: Optional[float] = None,
    mw_kda: Optional[float] = None,
    baseline_stats: Optional[Dict[str, Any]] = None,
    molecule_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detect Out-of-Distribution anomalies in antibody sequences.

    Compares sequence and biophysical properties against IgG reference
    statistics using z-score analysis.

    Parameters
    ----------
    sequence : str
        Protein sequence (amino acid string)
    pI : float, optional
        Isoelectric point (if available)
    mw_kda : float, optional
        Molecular weight in kDa (if available)
    baseline_stats : dict, optional
        Dynamic baseline statistics (overrides IGG_REFERENCE_STATS if provided)

    Returns
    -------
    dict with keys:
        - is_ood : bool (True if any metric has z > 3.0)
        - confidence : str ("High" / "Medium" / "Low") based on max z-score
        - flags : list of dicts with {"metric", "value", "z_score", "reference_range"}
        - max_z_score : float
        - warning_message : str or None
    """
    # Use dynamic baseline if provided, otherwise class-specific or hardcoded defaults
    if baseline_stats:
        ref = baseline_stats
    elif molecule_class:
        try:
            from src.ood_baseline import CLASS_SPECIFIC_BASELINES
            ref = CLASS_SPECIFIC_BASELINES.get(molecule_class, IGG_REFERENCE_STATS)
        except ImportError:
            ref = IGG_REFERENCE_STATS
    else:
        ref = IGG_REFERENCE_STATS

    flags = []
    z_scores = []

    # --- Metric 1: GRAVY (Hydrophobicity) ---
    gravy = compute_gravy(sequence)
    gravy_ref = ref["gravy"]
    gravy_z = abs(gravy - gravy_ref["mean"]) / gravy_ref["std"]
    z_scores.append(gravy_z)
    ref_range = f"{gravy_ref['mean'] - 2*gravy_ref['std']:.3f} to {gravy_ref['mean'] + 2*gravy_ref['std']:.3f}"
    flags.append({
        "metric": "GRAVY (Hydrophobicity)",
        "value": round(gravy, 4),
        "z_score": round(gravy_z, 3),
        "reference_range": ref_range,
        "is_outlier": gravy_z > 3.0,
    })

    # --- Metric 2: Cysteine count per 100 aa ---
    cys_count = sequence.upper().count('C')
    cys_per_100 = (cys_count / len(sequence) * 100) if sequence else 0.0
    cys_ref = ref["cys_count_per_100"]
    cys_z = abs(cys_per_100 - cys_ref["mean"]) / cys_ref["std"]
    z_scores.append(cys_z)
    ref_range = f"{cys_ref['mean'] - 2*cys_ref['std']:.3f} to {cys_ref['mean'] + 2*cys_ref['std']:.3f}"
    flags.append({
        "metric": "Cys per 100 aa",
        "value": round(cys_per_100, 3),
        "z_score": round(cys_z, 3),
        "reference_range": ref_range,
        "is_outlier": cys_z > 3.0,
    })

    # --- Metric 3: Sequence length ---
    # Normalize assembly-wide lengths to per-chain equivalent for OOD comparison.
    # Bulk mode passes the full concatenated assembly sequence, but OOD baselines
    # are calibrated to per-chain scale (~450 aa for HC, ~214 aa for LC).
    length = len(sequence)
    if molecule_class in ("canonical_mab", "adc") and length > 800:
        length = length // 4        # 2HC + 2LC → average per chain
    elif molecule_class == "bispecific_4chain" and length > 800:
        length = length // 4        # HC1 + LC1 + HC2 + LC2
    elif molecule_class in ("bispecific_3chain", "bispecific") and length > 600:
        length = length // 3        # HC + LC + scFv arm
    elif molecule_class == "fc_fusion" and length > 600:
        length = length // 2        # 2× (Fc + partner)
    len_ref = ref["length"]
    len_z = abs(length - len_ref["mean"]) / len_ref["std"]
    z_scores.append(len_z)
    ref_range = f"{len_ref['mean'] - 2*len_ref['std']:.0f} to {len_ref['mean'] + 2*len_ref['std']:.0f}"
    flags.append({
        "metric": "Sequence length",
        "value": length,
        "z_score": round(len_z, 3),
        "reference_range": ref_range,
        "is_outlier": len_z > 3.0,
    })

    # --- Metric 4: pI (if provided) ---
    if pI is not None:
        pi_ref = ref["pI"]
        pi_z = abs(pI - pi_ref["mean"]) / pi_ref["std"]
        z_scores.append(pi_z)
        ref_range = f"{pi_ref['mean'] - 2*pi_ref['std']:.2f} to {pi_ref['mean'] + 2*pi_ref['std']:.2f}"
        flags.append({
            "metric": "pI (Isoelectric point)",
            "value": round(pI, 2),
            "z_score": round(pi_z, 3),
            "reference_range": ref_range,
            "is_outlier": pi_z > 3.0,
        })

    # --- Metric 5: MW (if provided) ---
    if mw_kda is not None:
        mw_ref = ref["MW_kDa"]
        mw_z = abs(mw_kda - mw_ref["mean"]) / mw_ref["std"]
        z_scores.append(mw_z)
        ref_range = f"{mw_ref['mean'] - 2*mw_ref['std']:.0f} to {mw_ref['mean'] + 2*mw_ref['std']:.0f}"
        flags.append({
            "metric": "MW (Molecular Weight)",
            "value": round(mw_kda, 1),
            "z_score": round(mw_z, 3),
            "reference_range": ref_range,
            "is_outlier": mw_z > 3.0,
        })

    # --- Aggregate OOD Status ---
    max_z = max(z_scores) if z_scores else 0.0
    is_ood = any(flag["is_outlier"] for flag in flags)

    # Confidence level based on max z-score
    if max_z < 2.0:
        confidence = "High"
    elif max_z < 3.0:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Generate warning message
    warning_message = None
    if is_ood:
        outlier_metrics = [f["metric"] for f in flags if f["is_outlier"]]
        warning_message = (
            f"OOD detected (max z-score: {max_z:.2f}). "
            f"Outlier metrics: {', '.join(outlier_metrics)}. "
            "Predictions may be unreliable for out-of-distribution sequences."
        )

    return {
        "is_ood": is_ood,
        "confidence": confidence,
        "flags": flags,
        "max_z_score": round(max_z, 3),
        "warning_message": warning_message,
    }


# ===========================================================================
# 3. DevelopabilityPredictor Class
# ===========================================================================

class DevelopabilityPredictor:
    """
    XGBoost-based developability risk predictor with SHAP explainability.

    Predicts aggregation risk, thermal stability, and viscosity risk from
    pLM embeddings + biophysical features. Falls back to rule-based
    heuristics if XGBoost is not installed.

    Usage
    -----
    >>> predictor = DevelopabilityPredictor()
    >>> predictor.train()  # trains on mock dataset
    >>> result = predictor.predict(embedding, biophysical)
    >>> score = predictor.compute_developability_score(result)
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}  # target_name -> XGBRegressor
        self._trained = False
        self._xgb_available = False
        self._shap_available = False
        self._X_train: Optional[np.ndarray] = None

        try:
            import xgboost
            self._xgb_available = True
        except ImportError:
            log.warning("XGBoost not installed — using rule-based fallback. "
                       "Install: pip install xgboost")

        try:
            import shap
            self._shap_available = True
        except ImportError:
            log.info("SHAP not installed — explainability disabled. "
                    "Install: pip install shap")

        # --- Auto-load persisted models from disk ---
        if self._xgb_available:
            saved = _load_developability_models()
            if saved is not None and "models" in saved:
                self._models = saved["models"]
                self._X_train = saved.get("X_train")
                self._trained = True
                log.info("Developability predictor restored from disk (%d models)",
                         len(self._models))

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def uses_xgboost(self) -> bool:
        return self._xgb_available and self._trained

    def train(self, embedder=None) -> Dict[str, Any]:
        """
        Train XGBoost models on mock dataset.

        Parameters
        ----------
        embedder : Optional ESM2Embedder for generating training embeddings

        Returns
        -------
        dict with training info (n_samples, targets, mode)
        """
        if not self._xgb_available:
            self._trained = True  # mark as "trained" for rule-based mode
            return {
                "mode": "rule_based",
                "message": "XGBoost unavailable — using rule-based heuristics",
            }

        import xgboost as xgb

        # Build training data
        dataset = _generate_mock_dataset()
        X, y = build_feature_matrix(dataset, embedder=embedder)
        self._X_train = X

        # Train one XGBRegressor per target
        for i, target_name in enumerate(TARGET_NAMES):
            model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                verbosity=0,
            )
            model.fit(X, y[:, i])
            self._models[target_name] = model
            log.info("Trained XGBRegressor for '%s' on %d samples", target_name, len(X))

        self._trained = True

        # --- Persist models to disk so they survive Streamlit reruns ---
        save_path = _save_developability_models(self._models, self._X_train)

        return {
            "mode": "xgboost",
            "n_samples": len(X),
            "n_features": X.shape[1],
            "targets": TARGET_NAMES,
            "persisted": save_path is not None,
            "model_path": save_path,
        }

    def predict(
        self,
        embedding: np.ndarray,
        biophysical: np.ndarray,
        sequence: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predict developability metrics with optional OOD detection.

        Parameters
        ----------
        embedding   : (960,) pLM embedding vector
        biophysical : (7,) biophysical feature vector
                     [pI, MW_kDa, deam_sites, ox_sites, acidic_residues, basic_residues, hydrophobicity]
        sequence : str, optional
                   Protein sequence for OOD detection

        Returns
        -------
        dict with agg_risk, stability, viscosity_risk (all in [0, 1])
        and optional "ood_info" key if sequence is provided
        """
        # -- Input validation --
        if embedding is None or biophysical is None:
            raise ValueError("embedding and biophysical arrays must not be None")
        if not isinstance(embedding, np.ndarray):
            try:
                embedding = np.asarray(embedding, dtype=np.float64)
            except (TypeError, ValueError) as e:
                raise TypeError(f"embedding must be array-like of floats, got {type(embedding).__name__}: {e}")
        if not isinstance(biophysical, np.ndarray):
            try:
                biophysical = np.asarray(biophysical, dtype=np.float64)
            except (TypeError, ValueError) as e:
                raise TypeError(f"biophysical must be array-like of floats, got {type(biophysical).__name__}: {e}")
        if embedding.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {embedding.shape}")
        if biophysical.ndim != 1:
            raise ValueError(f"biophysical must be 1-D, got shape {biophysical.shape}")
        if np.isnan(embedding).any() or np.isnan(biophysical).any():
            raise ValueError("embedding and biophysical must not contain NaN values")

        if not self._trained:
            self.train()

        if not self._xgb_available:
            predictions = _rule_based_predict(biophysical)
        else:
            # Concatenate features
            features = np.concatenate([embedding, biophysical]).reshape(1, -1)

            predictions = {}
            for target_name in TARGET_NAMES:
                model = self._models[target_name]
                pred = float(model.predict(features)[0])
                # Clamp to valid range
                if target_name == "stability":
                    pred = np.clip(pred, 0.0, 1.0)
                else:
                    pred = np.clip(pred, 0.0, 1.0)
                predictions[target_name] = round(pred, 4)

        # Apply sequence composition adjustment (catches pathological sequences
        # that may have normal biophysical properties but abnormal composition)
        if sequence is not None:
            predictions = _apply_sequence_composition_adjustment(predictions, sequence)

        # Add OOD detection if sequence is provided
        if sequence is not None:
            pI = biophysical[0] if len(biophysical) > 0 else None
            mw_kda = biophysical[1] if len(biophysical) > 1 else None
            predictions["ood_info"] = compute_ood_flags(sequence, pI=pI, mw_kda=mw_kda, baseline_stats=None)

        return predictions

    def compute_developability_score(
        self,
        predictions: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Compute composite developability risk score.

        Score = w_agg * agg_risk + w_stab * (1 - stability) + w_visc * viscosity_risk

        Parameters
        ----------
        predictions : dict from self.predict()

        Returns
        -------
        dict with score (0-1), grade (Low/Medium/High), color, and individual predictions
        """
        agg = predictions.get("agg_risk", 0.2)
        stab = predictions.get("stability", 0.8)
        visc = predictions.get("viscosity_risk", 0.15)

        score = (
            RISK_WEIGHTS["agg"] * agg
            + RISK_WEIGHTS["stability"] * (1.0 - stab)
            + RISK_WEIGHTS["viscosity"] * visc
        )
        score = float(np.clip(score, 0.0, 1.0))

        # Use canonical grade boundaries from report_schema (SINGLE SOURCE OF TRUTH)
        from src.report_schema import GRADE_LOW_UPPER, GRADE_MEDIUM_UPPER
        if score < GRADE_LOW_UPPER:
            grade = "Low Risk"
            color = "#10B981"  # green
        elif score < GRADE_MEDIUM_UPPER:
            grade = "Medium Risk"
            color = "#F59E0B"  # amber
        else:
            grade = "High Risk"
            color = "#EF4444"  # red

        return {
            "score": round(score, 4),
            "grade": grade,
            "color": color,
            "predictions": predictions,
            "weights": RISK_WEIGHTS,
        }

    def explain_with_shap(
        self,
        embedding: np.ndarray,
        biophysical: np.ndarray,
        max_features: int = 15,
    ) -> Dict[str, Any]:
        """
        Compute SHAP TreeExplainer values for each target.

        Parameters
        ----------
        embedding    : (960,) pLM embedding
        biophysical  : (7,) biophysical features
        max_features : Max features to include in explanation

        Returns
        -------
        dict with per-target SHAP values and top feature contributions
        """
        if not self._shap_available or not self._xgb_available:
            return {"available": False, "message": "SHAP/XGBoost not available"}

        import shap

        features = np.concatenate([embedding, biophysical]).reshape(1, -1)

        # Build feature names
        embed_names = [f"emb_{i}" for i in range(N_EMBED_DIM)]
        all_feature_names = embed_names + BIOPHYS_NAMES

        result = {"available": True, "targets": {}}

        for target_name in TARGET_NAMES:
            model = self._models.get(target_name)
            if model is None:
                continue

            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(features)

                if isinstance(sv, np.ndarray) and sv.ndim == 2:
                    shap_vals = sv[0]  # first sample
                else:
                    shap_vals = np.array(sv).flatten()[:N_TOTAL_FEATURES]

                base_value = float(explainer.expected_value)
                if hasattr(explainer.expected_value, "__len__"):
                    base_value = float(explainer.expected_value[0])

                # Get top contributing features (by absolute SHAP value)
                abs_vals = np.abs(shap_vals)
                top_idx = np.argsort(abs_vals)[::-1][:max_features]

                top_features = []
                for idx in top_idx:
                    fname = all_feature_names[idx] if idx < len(all_feature_names) else f"feature_{idx}"
                    top_features.append({
                        "feature": fname,
                        "shap_value": round(float(shap_vals[idx]), 6),
                        "feature_value": round(float(features[0, idx]), 4),
                        "direction": "increases" if shap_vals[idx] > 0 else "decreases",
                    })

                # Aggregate embedding SHAP contribution
                embed_shap_total = float(np.sum(np.abs(shap_vals[:N_EMBED_DIM])))
                biophys_shap_total = float(np.sum(np.abs(shap_vals[N_EMBED_DIM:])))

                result["targets"][target_name] = {
                    "base_value": base_value,
                    "top_features": top_features,
                    "embed_contribution": round(embed_shap_total, 4),
                    "biophys_contribution": round(biophys_shap_total, 4),
                    "biophys_shap": {
                        name: round(float(shap_vals[N_EMBED_DIM + i]), 6)
                        for i, name in enumerate(BIOPHYS_NAMES)
                    },
                }

            except Exception as e:
                log.warning("SHAP failed for %s: %s", target_name, e)
                result["targets"][target_name] = {
                    "error": str(e),
                }

        return result

    def generate_actionable_advice(
        self,
        predictions: Dict[str, float],
        shap_result: Optional[Dict[str, Any]] = None,
        intent: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate human-readable actionable advice from risk predictions.

        Maps risk scores and SHAP attributions to domain-specific
        engineering recommendations.

        Parameters
        ----------
        predictions : Risk prediction dict
        shap_result : SHAP explanation dict (optional)
        intent      : Parsed intent dict (optional, for sequence context)

        Returns
        -------
        List of advice dicts: [{"category": str, "risk_level": str, "message": str, "priority": str}]
        """
        advice = []

        agg = predictions.get("agg_risk", 0.2)
        stab = predictions.get("stability", 0.8)
        visc = predictions.get("viscosity_risk", 0.15)

        # --- Aggregation Risk Advice ---
        if agg > 0.5:
            msg = (
                f"High aggregation risk ({agg:.2f}). "
                "Consider: (1) Reduce surface hydrophobic patches, especially in CDR regions. "
                "(2) Introduce charged residues (Asp, Glu) to improve solubility. "
                "(3) Screen formulation buffers with polysorbate-80 to suppress aggregation."
            )
            priority = "high"
        elif agg > 0.3:
            msg = (
                f"Moderate aggregation risk ({agg:.2f}). "
                "Recommend SEC-HPLC monitoring during stability studies. "
                "Consider DSF screening to identify stabilizing formulation conditions."
            )
            priority = "medium"
        else:
            msg = f"Low aggregation risk ({agg:.2f}). Standard monitoring recommended."
            priority = "low"

        # Enhance with SHAP if available
        if shap_result and shap_result.get("available"):
            agg_shap = shap_result.get("targets", {}).get("agg_risk", {})
            biophys = agg_shap.get("biophys_shap", {})
            if biophys.get("hydrophobicity", 0) > 0.01:
                msg += " SHAP analysis confirms hydrophobicity as primary driver."

        advice.append({
            "category": "Aggregation",
            "risk_level": "High" if agg > 0.5 else ("Medium" if agg > 0.3 else "Low"),
            "message": msg,
            "priority": priority,
        })

        # --- Thermal Stability Advice ---
        stability_risk = 1.0 - stab
        if stability_risk > 0.4:
            msg = (
                f"Low thermal stability (score: {stab:.2f}). "
                "Consider: (1) Optimize disulfide bonding by reviewing Cys positions. "
                "(2) Reduce deamidation hotspots (NG, NS motifs) via Asn->Gln substitutions. "
                "(3) Screen stabilizing excipients (sucrose, trehalose)."
            )
            priority = "high"
        elif stability_risk > 0.2:
            msg = (
                f"Moderate thermal stability (score: {stab:.2f}). "
                "Recommend DSF/DSC characterization to determine Tm values. "
                "Accelerated stability studies at 25C/40C/50C recommended."
            )
            priority = "medium"
        else:
            msg = f"Good thermal stability (score: {stab:.2f}). Standard ICH stability program."
            priority = "low"

        if shap_result and shap_result.get("available"):
            stab_shap = shap_result.get("targets", {}).get("stability", {})
            biophys = stab_shap.get("biophys_shap", {})
            if biophys.get("deam_sites", 0) < -0.01:
                msg += " SHAP analysis identifies deamidation sites as key instability driver."

        advice.append({
            "category": "Thermal Stability",
            "risk_level": "High" if stability_risk > 0.4 else ("Medium" if stability_risk > 0.2 else "Low"),
            "message": msg,
            "priority": priority,
        })

        # --- Viscosity Risk Advice ---
        if visc > 0.4:
            msg = (
                f"High viscosity risk ({visc:.2f}). "
                "Consider: (1) Reduce positive charge patches that drive self-association. "
                "(2) Evaluate alternative formulation strategies (lower concentration, "
                "arginine/glutamate co-solutes). (3) Target concentration <100 mg/mL for SC delivery."
            )
            priority = "high"
        elif visc > 0.2:
            msg = (
                f"Moderate viscosity risk ({visc:.2f}). "
                "Recommend rheology measurements at target concentration. "
                "Consider DLS to assess hydrodynamic diameter and self-association."
            )
            priority = "medium"
        else:
            msg = f"Low viscosity risk ({visc:.2f}). Suitable for high-concentration formulation."
            priority = "low"

        advice.append({
            "category": "Viscosity",
            "risk_level": "High" if visc > 0.4 else ("Medium" if visc > 0.2 else "Low"),
            "message": msg,
            "priority": priority,
        })

        # --- pLM Embedding Insight (if SHAP available) ---
        if shap_result and shap_result.get("available"):
            for target_name in TARGET_NAMES:
                target_data = shap_result.get("targets", {}).get(target_name, {})
                embed_contrib = target_data.get("embed_contribution", 0)
                biophys_contrib = target_data.get("biophys_contribution", 0)
                total = embed_contrib + biophys_contrib
                if total > 0 and embed_contrib / total > 0.5:
                    advice.append({
                        "category": "Sequence Features",
                        "risk_level": "Info",
                        "message": (
                            f"For {target_name}, pLM (ESM-2) embeddings contribute "
                            f"{100*embed_contrib/total:.0f}% of the prediction signal, "
                            "suggesting sequence-level patterns (beyond simple composition) "
                            "are significant. Deep sequence optimization may be beneficial."
                        ),
                        "priority": "info",
                    })
                    break  # Only one embedding insight

        return advice


# ===========================================================================
# 4. Module-level Cache & Convenience
# ===========================================================================

_CACHED_PREDICTOR: Optional[DevelopabilityPredictor] = None


def get_predictor() -> DevelopabilityPredictor:
    """Get (or create and train) the cached predictor.

    On first call: creates DevelopabilityPredictor which auto-loads from disk.
    If no persisted models exist, trains on mock data and saves to disk.
    Subsequent calls return the cached instance.
    """
    global _CACHED_PREDICTOR
    if _CACHED_PREDICTOR is None or not _CACHED_PREDICTOR.is_trained:
        _CACHED_PREDICTOR = DevelopabilityPredictor()  # auto-loads from disk in __init__
        if not _CACHED_PREDICTOR.is_trained:
            _CACHED_PREDICTOR.train()  # trains + saves to disk
    return _CACHED_PREDICTOR


def predict_developability(
    embedding: np.ndarray,
    biophysical: np.ndarray,
    intent: Optional[Dict[str, Any]] = None,
    sequence: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full developability prediction pipeline: predict + score + explain + advise.

    Parameters
    ----------
    embedding   : (960,) pLM embedding
    biophysical : (7,) biophysical features
    intent      : Parsed intent dict (optional)
    sequence    : Protein sequence for composition-based adjustments and OOD

    Returns
    -------
    dict with predictions, score, shap, advice
    """
    predictor = get_predictor()

    predictions = predictor.predict(embedding, biophysical, sequence=sequence)
    score_result = predictor.compute_developability_score(predictions)
    shap_result = predictor.explain_with_shap(embedding, biophysical)
    advice = predictor.generate_actionable_advice(predictions, shap_result, intent)

    result = {
        "predictions": predictions,
        "score": score_result,
        "shap": shap_result,
        "advice": advice,
        "mode": "xgboost" if predictor.uses_xgboost else "rule_based",
    }

    try:
        from src.label_emitter import emit_prediction_label
        emit_prediction_label("developability", result, {"input_length": len(sequence) if sequence else 0})
    except Exception:
        pass  # Label emission should never break predictions

    return result


# ===========================================================================
# 5. Jain-137 Property Retraining with ESM-2 Embeddings
# ===========================================================================

# Mapping from Jain-137 CSV columns to short target names
JAIN137_TARGET_MAP = {
    "Fab Tm by DSF (°C)":                                   "fab_tm",
    "HIC Retention Time (Min)a":                             "hic_rt",
    "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average": "acsins",
    "Poly-Specificity Reagent (PSR) SMP Score (0-1)":        "psr",
    "Slope for Accelerated Stability":                       "stability_slope",
    "HEK Titer (mg/L)":                                     "titer",
    "SMAC Retention Time (Min)a":                            "smac_rt",
}

# Quality gate: minimum Pearson |r| to consider a model useful
QUALITY_GATE_R = 0.25


def retrain_all_targets(
    data_path: str = "data/Jain137_Cleaned_Training_Data.csv",
    output_dir: str = "models/developability",
    use_esm2: bool = True,
    val_split: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Retrain all Jain-137 property predictors with ESM-2 hybrid features.

    For each target column in the Jain-137 dataset, trains an XGBoost
    regressor on ESM-2 embeddings (960-dim) + biophysical features (7-dim)
    = 967-dim input. Falls back to mock embeddings if ESM-2 is unavailable.

    Parameters
    ----------
    data_path   : Path to Jain-137 CSV with VH/VL sequences and targets
    output_dir  : Directory to save trained model artifacts
    use_esm2    : Whether to attempt real ESM-2 (falls back to mock if unavailable)
    val_split   : Fraction of data for validation
    seed        : Random seed

    Returns
    -------
    dict mapping target_name -> {r, rmse, n, gate, model_path}
    """
    import os
    import json
    import pickle
    import pandas as pd
    from scipy import stats as sp_stats

    try:
        import xgboost as xgb
    except ImportError:
        log.error("XGBoost required for retrain_all_targets. Install: pip install xgboost")
        return {"error": "xgboost not installed"}

    # -- Load data --
    df = pd.read_csv(data_path)
    log.info("Loaded %d molecules from %s", len(df), data_path)

    # -- Generate embeddings --
    hc_seqs = df["VH"].astype(str).tolist()
    lc_seqs = df["VL"].astype(str).tolist()

    embedder = None
    if use_esm2:
        try:
            from src.esm2_hybrid_encoder import _get_esm2_embedder
            embedder = _get_esm2_embedder()
            if embedder is not None:
                log.info("Using real ESM-2 embedder")
            else:
                log.info("ESM-2 unavailable — using mock embeddings")
        except Exception as e:
            log.warning("ESM-2 import failed (%s) — using mock embeddings", e)

    from src.esm2_hybrid_encoder import _embed_antibody_batch
    embeddings = _embed_antibody_batch(hc_seqs, lc_seqs, embedder=embedder)
    log.info("Embeddings shape: %s (mode: %s)",
             embeddings.shape, "esm2" if embedder else "mock")

    # -- Compute biophysical features --
    biophys_list = []
    for _, row in df.iterrows():
        vh, vl = str(row["VH"]), str(row["VL"])
        seq = vh + vl
        n = len(seq)
        gravy = sum(KYTE_DOOLITTLE_SCALE.get(aa, 0) for aa in seq.upper()) / max(n, 1)

        # Count liabilities
        seq_upper = seq.upper()
        deam = sum(1 for i in range(n - 1) if seq_upper[i] == "N" and seq_upper[i + 1] in "GSTD")
        ox = seq_upper.count("M") + seq_upper.count("W")
        acidic = seq_upper.count("D") + seq_upper.count("E")
        basic = seq_upper.count("K") + seq_upper.count("R") + seq_upper.count("H")

        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            pa = ProteinAnalysis(seq_upper)
            pI = pa.isoelectric_point()
            mw = pa.molecular_weight() / 1000.0
        except Exception:
            pI = 8.0
            mw = n * 0.11

        hydro = max(0.0, min(1.0, (gravy + 1.0) / 2.0))
        biophys_list.append([pI, mw, float(deam), float(ox),
                             float(acidic), float(basic), hydro])

    biophys = np.array(biophys_list, dtype=np.float32)

    # -- Concatenate: embeddings (960) + biophysical (7) = 967 features --
    X_all = np.concatenate([embeddings, biophys], axis=1)
    log.info("Feature matrix: %s", X_all.shape)

    # -- Train per-target models --
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    results = {}
    for csv_col, short_name in JAIN137_TARGET_MAP.items():
        if csv_col not in df.columns:
            log.warning("Target column '%s' not found — skipping", csv_col)
            continue

        y_raw = df[csv_col].values
        valid_mask = ~np.isnan(y_raw.astype(float))
        n_valid = valid_mask.sum()
        if n_valid < 10:
            log.warning("Target '%s' has only %d valid samples — skipping", short_name, n_valid)
            results[short_name] = {"n": int(n_valid), "gate": "SKIP", "reason": "too few samples"}
            continue

        X = X_all[valid_mask]
        y = y_raw[valid_mask].astype(float)

        # Train/val split
        indices = rng.permutation(n_valid)
        n_val = max(2, int(n_valid * val_split))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # XGBoost with regularization appropriate for high-dim + small n
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.3,   # Low: prevents overfitting on 647-dim with ~100 samples
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=seed,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred_val = model.predict(X_val)
        r_val, _ = sp_stats.pearsonr(y_val, y_pred_val) if len(y_val) > 2 else (0.0, 1.0)
        rmse_val = float(np.sqrt(np.mean((y_val - y_pred_val) ** 2)))
        rho_val, _ = sp_stats.spearmanr(y_val, y_pred_val) if len(y_val) > 2 else (0.0, 1.0)

        gate = "PASS" if abs(r_val) >= QUALITY_GATE_R else "FAIL"

        # Save model
        model_path = os.path.join(output_dir, f"xgb_{short_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        results[short_name] = {
            "n": int(n_valid),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "pearson_r": round(float(r_val), 4),
            "spearman_rho": round(float(rho_val), 4),
            "rmse": round(rmse_val, 4),
            "gate": gate,
            "model_path": model_path,
            "feature_dim": X.shape[1],
            "embedding_mode": "esm2" if embedder else "mock",
        }
        log.info("  [%s] %s: r=%.3f, rho=%.3f, RMSE=%.3f, n=%d",
                 gate, short_name, r_val, rho_val, rmse_val, n_valid)

    # -- Save metadata --
    metadata = {
        "format_version": "3.0",
        "model_type": "xgboost",
        "feature_dim": int(X_all.shape[1]),
        "embedding_dim": int(embeddings.shape[1]),
        "biophys_dim": int(biophys.shape[1]),
        "embedding_mode": "esm2" if embedder else "mock",
        "n_molecules": len(df),
        "data_source": data_path,
        "seed": seed,
        "targets": results,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved to %s", meta_path)

    return results


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
    print("  ProtePilot — Developability Predictor v1.0 Test")
    print("=" * 60)

    # Test mock dataset generation
    dataset = _generate_mock_dataset()
    print(f"\nMock dataset: {len(dataset)} samples")
    for d in dataset[:3]:
        print(f"  {d['name']}: agg={d['agg_risk']:.2f}, "
              f"stab={d['stability']:.2f}, visc={d['viscosity_risk']:.2f}")

    # Test predictor
    predictor = DevelopabilityPredictor()
    train_info = predictor.train()
    print(f"\nTraining: {train_info}")

    # Test prediction with mock embedding
    from src.pLM_embedder import mock_antibody_embedding
    embedding = mock_antibody_embedding(
        REFERENCE_MABS[0]["vh_seq"],
        REFERENCE_MABS[0]["vl_seq"],
    )
    biophys = np.array([8.44, 148.0, 1, 1, 38, 48, 0.34], dtype=np.float32)

    # Test normal IgG sequence
    nist_seq = REFERENCE_MABS[0]["vh_seq"] + REFERENCE_MABS[0]["vl_seq"]
    predictions = predictor.predict(embedding, biophys, sequence=nist_seq)
    print(f"\nPredictions (NISTmAb - Normal IgG):")
    for k, v in predictions.items():
        if k != "ood_info":
            print(f"  {k}: {v:.4f}")

    if "ood_info" in predictions:
        ood = predictions["ood_info"]
        print(f"  OOD Status: {ood['is_ood']} (confidence: {ood['confidence']}, max z: {ood['max_z_score']})")
        if ood["warning_message"]:
            print(f"  Warning: {ood['warning_message']}")

    score = predictor.compute_developability_score(predictions)
    print(f"\nDevelopability Score: {score['score']:.4f} ({score['grade']})")

    # Test abnormal sequence (all Trp - highly hydrophobic)
    abnormal_seq = "W" * 300
    abnormal_biophys = np.array([9.5, 50.0, 0, 0, 10, 10, 0.75], dtype=np.float32)
    abnormal_embedding = np.zeros(N_EMBED_DIM, dtype=np.float32)  # Mock embedding

    predictions_abnormal = predictor.predict(abnormal_embedding, abnormal_biophys, sequence=abnormal_seq)
    print(f"\n\nPredictions (Abnormal - All Trp):")
    for k, v in predictions_abnormal.items():
        if k != "ood_info":
            print(f"  {k}: {v:.4f}")

    if "ood_info" in predictions_abnormal:
        ood = predictions_abnormal["ood_info"]
        print(f"  OOD Status: {ood['is_ood']} (confidence: {ood['confidence']}, max z: {ood['max_z_score']})")
        if ood["warning_message"]:
            print(f"  Warning: {ood['warning_message']}")
        print(f"  Outlier flags:")
        for flag in ood["flags"]:
            if flag["is_outlier"]:
                print(f"    - {flag['metric']}: {flag['value']} (z={flag['z_score']})")

    # Test full pipeline
    result = predict_developability(embedding, biophys)
    print(f"\n\nFull pipeline result:")
    print(f"  Mode: {result['mode']}")
    print(f"  Score: {result['score']['score']:.4f}")
    print(f"  Advice items: {len(result['advice'])}")
    for a in result["advice"]:
        print(f"    [{a['priority'].upper()}] {a['category']}: {a['risk_level']}")

    print("\nDevelopability Predictor v1.0 test complete")
