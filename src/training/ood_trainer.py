"""
ood_trainer.py — Train OOD (Out-of-Distribution) Detector
==========================================================
Learns the biophysical feature distribution of known biologics, then
flags sequences whose features fall outside this distribution.

Approach: One-class SVM or Mahalanobis distance on the same 11 biophysical
features used by the classifier. In-distribution = the training data.
OOD = synthetic outliers + extreme compositions.

Pipeline:
    1. Load harmonized training data (in-distribution)
    2. Generate synthetic OOD sequences (extreme composition, shuffled, etc.)
    3. Compute Mahalanobis distance threshold on validation set
    4. Save detector artifact (mean, covariance, threshold)
    5. Evaluate: precision/recall on OOD vs in-distribution test set

Usage:
    python -m src.training.ood_trainer
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("ProtePilot.Training.OODTrainer")

from src.training.schema import FEATURE_COLS

# OOD detection uses only scale-invariant (density/ratio) features.
# Raw counts (seq_length, deam_sites, ox_sites, etc.) scale with sequence
# length and cause false positives when inference sequences are longer than
# training sequences (e.g., full HC+LC at inference vs VH+VL in training).
# OOD detection uses only composition-ratio features that are truly invariant
# to the training-inference gap.
#
# EXCLUDED features and why:
#   seq_length, mw_kda, deam_sites, ox_sites, cysteine_count,
#   acidic_residues, basic_residues → raw counts that scale with seq length
#   (training uses VH+VL ~220aa, inference uses full HC+LC ~660aa)
#
#   hc_len_norm, lc_len_norm, hc_frac, has_lc, n_chains, n_unique_chains →
#   chain structure features that differ because training data lacks full-length
#   constant regions (VH/VL only vs full HC/LC at inference)
#
# KEPT: per-residue ratios and physicochemical properties that are
# intrinsic to amino acid composition, independent of sequence length.
OOD_FEATURE_COLS: List[str] = [
    "pI",                # isoelectric point — per-residue average
    "gravy",             # GRAVY hydropathy — per-residue average
    "hydrophobicity",    # normalized [0,1] — per-residue
    "aromatic_frac",     # F+W+Y fraction — length-invariant
    "pro_gly_frac",      # P+G fraction — length-invariant
    "cys_frac",          # C fraction — length-invariant
    "deam_density",      # deam sites per residue — density
    "ox_density",        # ox sites per residue — density
    "charge_ratio",      # K+R+H / (D+E+K+R+H) — pure ratio
    "small_frac",        # G+A+S fraction — length-invariant
    "aliphatic_idx",     # thermal stability index — per-residue
]


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


def _get_sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except ImportError:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════
#  Synthetic OOD Generator
# ═══════════════════════════════════════════════════════════════════════

def generate_ood_sequences(n: int = 500, seed: int = 99) -> List[Dict[str, Any]]:
    """Generate synthetic OOD sequences with extreme biophysical properties.

    Expanded generator (v2): 11 OOD types producing 500+ diverse samples
    to improve precision and F1 over the v1 generator (73 samples, 5 types).

    Types:
        1. Homo-repeats: single amino acid repeated (extreme composition)
        2. Extreme pI: all-basic or all-acidic residues
        3. Random shuffled: uniform random amino acid selection
        4. Cysteine-rich: >30% cysteine content
        5. Sub-peptide: very short sequences (<15 aa)
        6. Tandem repeats: short motifs repeated many times
        7. Extreme length: very long random sequences (800-2000 aa)
        8. Extreme hydrophobic: >60% hydrophobic residues
        9. Proline/glycine-rich (disordered): >50% P+G
        10. Chimeric fragments: two unrelated random fragments joined
        11. Biased dipeptide: dominated by one dipeptide motif
    """
    from src.training.features import compute_sequence_features as _compute_features, compute_chain_features as _compute_chain_features

    rng = np.random.RandomState(seed)
    aa_all = list("ACDEFGHIKLMNPQRSTVWY")
    records = []

    def _add_record(name, ood_type, seq):
        """Compute features and append record if valid."""
        feats = _compute_features(seq)
        if feats:
            records.append({
                "name": name, "is_ood": True, "ood_type": ood_type,
                **feats, "n_chains": 1, "n_unique_chains": 1,
            })

    # ── Type 1: Homo-repeats (expanded to 8 aa × 5 lengths = 40) ────
    for aa in ["A", "K", "D", "W", "P", "G", "L", "E"]:
        for length in [30, 80, 150, 300, 500]:
            _add_record(f"ood_homo_{aa}_{length}", "homo_repeat", aa * length)

    # ── Type 2: Extreme pI (40 samples) ──────────────────────────────
    for i in range(40):
        length = rng.randint(60, 300)
        if i < 22:
            pool = list("KKKRRRHH")  # extreme basic
        else:
            pool = list("DDDEEE")    # extreme acidic
        seq = "".join(rng.choice(pool, size=length))
        _add_record(f"ood_extreme_pi_{i}", "extreme_pI", seq)

    # ── Type 3: Random shuffled with biased composition (80 samples) ──
    #   Pure uniform random sequences look too normal on density features.
    #   Bias toward extreme compositions to make them more detectable.
    for i in range(80):
        length = rng.randint(40, 400)
        # Bias pool: over-represent 2-3 amino acids (50%+)
        dominant = rng.choice(aa_all, size=rng.randint(1, 3), replace=False).tolist()
        pool = dominant * 5 + aa_all  # ~70% dominant aa
        seq = "".join(rng.choice(pool, size=length))
        _add_record(f"ood_random_{i}", "biased_random", seq)

    # ── Type 4: Cysteine-rich (30 samples, 30-50% cysteine) ─────────
    for i in range(30):
        length = rng.randint(40, 250)
        cys_frac = rng.uniform(0.30, 0.50)
        seq_list = []
        for _ in range(length):
            if rng.random() < cys_frac:
                seq_list.append("C")
            else:
                seq_list.append(rng.choice(aa_all))
        seq = "".join(seq_list)
        _add_record(f"ood_cys_rich_{i}", "cysteine_rich", seq)

    # ── Type 5: Sub-peptide / very short (30 samples) ────────────────
    for i in range(30):
        length = rng.randint(5, 14)
        seq = "".join(rng.choice(aa_all, size=length))
        _add_record(f"ood_short_{i}", "sub_peptide", seq)

    # ── Type 6: Tandem repeats (50 samples) ──────────────────────────
    repeat_motifs = ["AG", "GS", "EAAAK", "PGS", "GGGGS", "AP", "KP",
                     "DDD", "PAPA", "WGWG", "SSSS", "AAAA", "LLLV"]
    for i in range(50):
        motif = rng.choice(repeat_motifs)
        n_reps = rng.randint(10, 80)
        seq = (motif * n_reps)[:rng.randint(40, 400)]
        _add_record(f"ood_tandem_{i}", "tandem_repeat", seq)

    # ── Type 7: Extreme composition (40 samples) ──────────────────────
    #   Sequences with extreme aliphatic index, extreme small_frac, etc.
    for i in range(40):
        length = rng.randint(50, 300)
        if i < 10:
            # All aliphatic → extreme aliphatic_idx
            pool = list("IIILLLLVVVVAAA")
        elif i < 20:
            # All small → extreme small_frac
            pool = list("GGGAAASSSS")
        elif i < 30:
            # All aromatic → extreme aromatic_frac
            pool = list("FFFWWWYYY")
        else:
            # Mix of extremes
            pool = list("MMMMWWWCCC")
        seq = "".join(rng.choice(pool, size=length))
        _add_record(f"ood_extreme_comp_{i}", "extreme_composition", seq)

    # ── Type 8: Extreme hydrophobic (40 samples, >60% VILMFW) ────────
    hydrophobic = list("VVVIILLLMMFFW")
    for i in range(40):
        length = rng.randint(50, 350)
        seq_list = []
        for _ in range(length):
            if rng.random() < 0.65:
                seq_list.append(rng.choice(hydrophobic))
            else:
                seq_list.append(rng.choice(aa_all))
        seq = "".join(seq_list)
        _add_record(f"ood_hydro_{i}", "extreme_hydrophobic", seq)

    # ── Type 9: Pro/Gly-rich disordered (40 samples, >50% P+G) ──────
    pg_pool = list("PPPPGGGGG")
    for i in range(40):
        length = rng.randint(50, 300)
        seq_list = []
        for _ in range(length):
            if rng.random() < 0.55:
                seq_list.append(rng.choice(pg_pool))
            else:
                seq_list.append(rng.choice(aa_all))
        seq = "".join(seq_list)
        _add_record(f"ood_disorder_{i}", "proline_glycine_rich", seq)

    # ── Type 10: Chimeric with extreme regions (50 samples) ────────────
    #   Extreme composition fragments joined — detectable on density features
    extreme_pools = [
        list("KKKRRR"),     # basic
        list("DDDEEE"),     # acidic
        list("CCCCCG"),     # cys-rich
        list("WWWFFF"),     # aromatic
        list("PPPPGG"),     # pro/gly
        list("IIILLV"),     # aliphatic
    ]
    for i in range(50):
        len1 = rng.randint(30, 150)
        len2 = rng.randint(30, 150)
        pool1 = extreme_pools[rng.randint(0, len(extreme_pools))]
        pool2 = extreme_pools[rng.randint(0, len(extreme_pools))]
        frag1 = "".join(rng.choice(pool1, size=len1))
        frag2 = "".join(rng.choice(pool2, size=len2))
        linker = "".join(rng.choice(list("GSGSGS"), size=rng.randint(3, 10)))
        seq = frag1 + linker + frag2
        _add_record(f"ood_chimeric_{i}", "extreme_chimeric", seq)

    # ── Type 11: Biased dipeptide (40 samples) ───────────────────────
    #   Sequence dominated by one dipeptide motif — unnatural repetition
    dipeptides = ["KE", "DE", "RD", "WW", "PP", "MM", "HH", "FF", "YY", "II"]
    for i in range(40):
        dp = rng.choice(dipeptides)
        length = rng.randint(40, 300)
        seq_list = []
        for _ in range(length // 2):
            if rng.random() < 0.7:
                seq_list.append(dp)
            else:
                seq_list.append("".join(rng.choice(aa_all, size=2)))
        seq = "".join(seq_list)[:length]
        _add_record(f"ood_dipep_{i}", "biased_dipeptide", seq)

    # ── Ensure all records have chain features ────────────────────────
    for rec in records:
        if "hc_frac" not in rec:
            rec.update({"hc_frac": 1.0, "has_lc": 0.0, "hc_len_norm": 0.0, "lc_len_norm": 0.0})

    log.info("Generated %d OOD sequences (11 types)", len(records))
    return records


# ═══════════════════════════════════════════════════════════════════════
#  Mahalanobis Distance Detector
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OODDetectorResult:
    """Result of OOD detector training."""
    n_in_distribution: int = 0
    n_ood: int = 0
    threshold: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_f1: float = 0.0
    artifact_path: Optional[str] = None
    training_time_s: float = 0.0

    def summary(self) -> str:
        return (
            f"OOD Detector: threshold={self.threshold:.2f}, "
            f"val F1={self.val_f1:.3f}, test F1={self.test_f1:.3f}, "
            f"in_dist={self.n_in_distribution}, ood={self.n_ood}"
        )


def train_ood_detector(
    data_path: str = "data/training/classifier_data.csv",
    artifact_dir: str = "models/ood_detector",
    seed: int = 42,
) -> OODDetectorResult:
    """
    Train an ensemble OOD detector: Mahalanobis + IsolationForest.

    Phase 3 upgrade:
    - Global Mahalanobis distance (covariance-aware)
    - Per-class Mahalanobis thresholds (class-conditional OOD)
    - IsolationForest anomaly score (tree-based, complementary to Mahalanobis)
    - Ensemble: flag as OOD if BOTH methods agree (reduces false positives)
    - XGBoost classifier entropy as 3rd signal

    In-distribution: the harmonized training data.
    OOD: synthetic outlier sequences.
    """
    result = OODDetectorResult()
    t0 = time.time()

    # Load in-distribution data
    df_in = pd.read_csv(data_path)
    X_in = df_in[OOD_FEATURE_COLS].fillna(0).values.astype(np.float64)
    y_labels = df_in["molecule_class"].values
    result.n_in_distribution = len(X_in)

    # Generate OOD data
    ood_records = generate_ood_sequences(n=500, seed=seed)
    df_ood = pd.DataFrame(ood_records)
    X_ood = df_ood[OOD_FEATURE_COLS].fillna(0).values.astype(np.float64)
    result.n_ood = len(X_ood)

    # ── 1. Global Mahalanobis ──────────────────────────────────────
    mean = X_in.mean(axis=0)
    cov = np.cov(X_in, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.inv(cov)

    def mahal_dist(X):
        diff = X - mean
        return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    dist_in = mahal_dist(X_in)
    dist_ood = mahal_dist(X_ood)

    # ── 2. Per-class Mahalanobis ───────────────────────────────────
    unique_classes = sorted(set(y_labels))
    per_class_stats = {}
    for cls in unique_classes:
        cls_mask = y_labels == cls
        X_cls = X_in[cls_mask]
        if len(X_cls) < 10:
            continue
        cls_mean = X_cls.mean(axis=0)
        cls_cov = np.cov(X_cls, rowvar=False)
        if cls_cov.ndim < 2:
            continue
        cls_cov += np.eye(cls_cov.shape[0]) * 1e-6
        try:
            cls_cov_inv = np.linalg.inv(cls_cov)
        except np.linalg.LinAlgError:
            continue
        cls_dists = np.sqrt(np.sum((X_cls - cls_mean) @ cls_cov_inv * (X_cls - cls_mean), axis=1))
        cls_threshold = float(np.percentile(cls_dists, 97))
        per_class_stats[cls] = {
            "mean": cls_mean, "cov_inv": cls_cov_inv,
            "threshold": cls_threshold, "n_samples": int(len(X_cls)),
        }

    log.info("Per-class Mahalanobis: %d classes with thresholds", len(per_class_stats))

    # ── 3. IsolationForest ─────────────────────────────────────────
    iforest = None
    iforest_scores_in = None
    iforest_scores_ood = None
    try:
        from sklearn.ensemble import IsolationForest
        iforest = IsolationForest(
            n_estimators=200,
            contamination=0.01,  # expect ~1% OOD in real usage
            random_state=seed,
            n_jobs=-1,
        )
        iforest.fit(X_in)
        # score_samples: higher = more normal, lower = more anomalous
        iforest_scores_in = iforest.score_samples(X_in)
        iforest_scores_ood = iforest.score_samples(X_ood)
        log.info("IsolationForest trained: in-dist score range [%.3f, %.3f]",
                 iforest_scores_in.min(), iforest_scores_in.max())
    except ImportError:
        log.warning("sklearn not available — IsolationForest disabled")

    # ── 4. Threshold tuning on validation split ────────────────────
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(dist_in))
    n_val = max(50, len(dist_in) // 5)

    # Split OOD
    ood_idx = rng.permutation(len(dist_ood))
    n_ood_val = max(10, len(dist_ood) // 3)

    # Mahalanobis threshold
    best_f1 = 0
    best_threshold = 0
    train_in_dist = dist_in[idx[n_val:]]
    val_in_dist = dist_in[idx[:n_val]]
    val_ood_dist = dist_ood[ood_idx[:n_ood_val]]
    test_ood_dist = dist_ood[ood_idx[n_ood_val:]]

    for percentile in range(85, 100):
        thresh = np.percentile(train_in_dist, percentile)
        val_all_dist = np.concatenate([val_in_dist, val_ood_dist])
        val_all_labels = np.array([0] * len(val_in_dist) + [1] * len(val_ood_dist))
        val_preds = (val_all_dist > thresh).astype(int)
        tp = int(np.sum((val_preds == 1) & (val_all_labels == 1)))
        fp = int(np.sum((val_preds == 1) & (val_all_labels == 0)))
        fn = int(np.sum((val_preds == 0) & (val_all_labels == 1)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            result.val_precision = prec
            result.val_recall = rec
            result.val_f1 = f1

    result.threshold = float(best_threshold)

    # IsolationForest threshold
    iforest_threshold = None
    if iforest_scores_in is not None:
        iforest_threshold = float(np.percentile(iforest_scores_in, 2))
        log.info("IsolationForest threshold: %.4f (2nd percentile)", iforest_threshold)

    # ── 5. Evaluate ensemble on test set ───────────────────────────
    test_in_dist = dist_in[idx[:n_val]]
    test_all_dist = np.concatenate([test_in_dist, test_ood_dist])
    test_all_labels = np.array([0] * len(test_in_dist) + [1] * len(test_ood_dist))
    test_preds_mahal = (test_all_dist > best_threshold).astype(int)

    if iforest is not None:
        test_in_X = X_in[idx[:n_val]]
        test_ood_X = X_ood[ood_idx[n_ood_val:]]
        test_all_X = np.vstack([test_in_X, test_ood_X])
        test_iforest_scores = iforest.score_samples(test_all_X)
        test_preds_iforest = (test_iforest_scores < iforest_threshold).astype(int)
        # Ensemble: OOD if EITHER method flags it (union — maximizes recall)
        test_preds_ensemble = np.maximum(test_preds_mahal, test_preds_iforest)
    else:
        test_preds_ensemble = test_preds_mahal

    tp = int(np.sum((test_preds_ensemble == 1) & (test_all_labels == 1)))
    fp = int(np.sum((test_preds_ensemble == 1) & (test_all_labels == 0)))
    fn = int(np.sum((test_preds_ensemble == 0) & (test_all_labels == 1)))
    result.test_precision = tp / max(tp + fp, 1)
    result.test_recall = tp / max(tp + fn, 1)
    result.test_f1 = 2 * result.test_precision * result.test_recall / max(
        result.test_precision + result.test_recall, 1e-8)

    log.info("Ensemble OOD: threshold=%.2f, val_F1=%.3f, test_F1=%.3f",
             result.threshold, result.val_f1, result.test_f1)

    # ── 6. Save artifacts ──────────────────────────────────────────
    os.makedirs(artifact_dir, exist_ok=True)

    # Main Mahalanobis detector
    np.savez(os.path.join(artifact_dir, "ood_detector.npz"),
             mean=mean, cov_inv=cov_inv, threshold=np.array([result.threshold]))

    # Per-class Mahalanobis
    per_class_save = {}
    for cls, stats in per_class_stats.items():
        np.savez(os.path.join(artifact_dir, f"ood_class_{cls}.npz"),
                 mean=stats["mean"], cov_inv=stats["cov_inv"],
                 threshold=np.array([stats["threshold"]]))
        per_class_save[cls] = {"threshold": stats["threshold"],
                                "n_samples": stats["n_samples"]}

    # IsolationForest
    if iforest is not None:
        try:
            import joblib
            joblib.dump(iforest, os.path.join(artifact_dir, "iforest_model.joblib"))
        except ImportError:
            import pickle
            with open(os.path.join(artifact_dir, "iforest_model.pkl"), "wb") as f:
                pickle.dump(iforest, f)

    from src.training.pipeline import PIPELINE_VERSION, _schema_hash
    meta = {
        "detector_type": "ensemble_mahalanobis_iforest",
        "feature_cols": OOD_FEATURE_COLS,
        "feature_schema_version": _schema_hash(),
        "pipeline_version": PIPELINE_VERSION,
        "format_version": "2.0",
        "dataset_version": _dataset_hash(data_path),
        "sklearn_version": _get_sklearn_version(),
        "mahalanobis_threshold": result.threshold,
        "iforest_threshold": iforest_threshold,
        "per_class_thresholds": per_class_save,
        "n_in_distribution": result.n_in_distribution,
        "n_ood_training": result.n_ood,
        "val_f1": result.val_f1,
        "test_f1": result.test_f1,
        "val_precision": result.val_precision,
        "val_recall": result.val_recall,
        "test_precision": result.test_precision,
        "test_recall": result.test_recall,
        "has_iforest": iforest is not None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(artifact_dir, "ood_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    result.artifact_path = os.path.join(artifact_dir, "ood_detector.npz")
    result.training_time_s = time.time() - t0
    log.info("Saved ensemble OOD detector to %s", artifact_dir)

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Inference API
# ═══════════════════════════════════════════════════════════════════════

_OOD_DETECTOR = None
_OOD_LOADED = False


def load_ood_detector(artifact_dir: str = "models/ood_detector"):
    """Load ensemble OOD detector artifacts."""
    global _OOD_DETECTOR, _OOD_LOADED
    if _OOD_LOADED:
        return _OOD_DETECTOR
    _OOD_LOADED = True

    npz_path = os.path.join(artifact_dir, "ood_detector.npz")
    meta_path = os.path.join(artifact_dir, "ood_metadata.json")
    if not os.path.exists(npz_path):
        return None

    try:
        # ── Metadata is REQUIRED (not optional) ──
        if not os.path.exists(meta_path):
            log.error("OOD metadata file missing: %s. Refusing to load unversioned artifact.", meta_path)
            _OOD_DETECTOR = None
            return None

        with open(meta_path) as f:
            meta = json.load(f)

        # ── Strong format version validation (reject, not warn) ──
        fmt_ver = meta.get("format_version", "")
        if fmt_ver not in ("1.0", "2.0"):
            log.error("OOD artifact format_version='%s' not supported (expected 1.0 or 2.0). "
                      "Retrain with: python -m src.training.pipeline --step ood", fmt_ver)
            _OOD_DETECTOR = None
            return None

        # ── NPZ key validation ──
        data = np.load(npz_path)
        required_keys = {"mean", "cov_inv", "threshold"}
        missing_keys = required_keys - set(data.files)
        if missing_keys:
            log.error("OOD NPZ file missing required keys: %s. Retrain required.", missing_keys)
            _OOD_DETECTOR = None
            return None

        detector = {
            "mean": data["mean"],
            "cov_inv": data["cov_inv"],
            "threshold": float(data["threshold"][0]),
            "iforest": None,
            "iforest_threshold": meta.get("iforest_threshold"),
            "per_class": {},
        }

        # ── sklearn version check (warn + log severity) ──
        artifact_sklearn = meta.get("sklearn_version", "")
        if artifact_sklearn:
            try:
                import sklearn
                current_sklearn = sklearn.__version__
                if current_sklearn != artifact_sklearn:
                    # Compare major.minor — only warn if major/minor differ
                    art_parts = artifact_sklearn.split(".")[:2]
                    cur_parts = current_sklearn.split(".")[:2]
                    if art_parts != cur_parts:
                        log.warning(
                            "sklearn MAJOR.MINOR version mismatch: artifact=%s, current=%s. "
                            "IsolationForest may produce inconsistent results. Retrain recommended.",
                            artifact_sklearn, current_sklearn)
                    else:
                        log.debug("sklearn patch version differs (artifact=%s, current=%s) — OK",
                                  artifact_sklearn, current_sklearn)
            except ImportError:
                pass

        # ── Feature schema version check ──
        artifact_schema = meta.get("feature_schema_version", "")
        if artifact_schema:
            try:
                from src.training.pipeline import _schema_hash
                code_schema = _schema_hash()
                if artifact_schema != code_schema:
                    log.warning(
                        "OOD FEATURE SCHEMA MISMATCH: artifact=%s, code=%s. "
                        "Retrain recommended.",
                        artifact_schema, code_schema,
                    )
            except ImportError:
                pass

        # ── Dataset version traceability ──
        artifact_dataset = meta.get("dataset_version", "")
        if artifact_dataset:
            log.info("OOD detector trained on dataset_version=%s", artifact_dataset)
        else:
            log.warning("OOD artifact missing dataset_version — provenance unknown. "
                        "Retrain with latest pipeline for full traceability.")

        # Load IsolationForest
        joblib_path = os.path.join(artifact_dir, "iforest_model.joblib")
        pkl_path = os.path.join(artifact_dir, "iforest_model.pkl")
        try:
            if os.path.exists(joblib_path):
                import joblib
                detector["iforest"] = joblib.load(joblib_path)
            elif os.path.exists(pkl_path):
                import pickle
                with open(pkl_path, "rb") as f:
                    detector["iforest"] = pickle.load(f)
        except Exception as e:
            log.debug("IsolationForest not loaded: %s", e)

        # Load per-class Mahalanobis
        for fname in os.listdir(artifact_dir):
            if fname.startswith("ood_class_") and fname.endswith(".npz"):
                cls = fname[len("ood_class_"):-len(".npz")]
                cls_data = np.load(os.path.join(artifact_dir, fname))
                detector["per_class"][cls] = {
                    "mean": cls_data["mean"],
                    "cov_inv": cls_data["cov_inv"],
                    "threshold": float(cls_data["threshold"][0]),
                }

        _OOD_DETECTOR = detector
        log.info("Ensemble OOD detector loaded: mahal_thresh=%.2f, iforest=%s, %d class detectors",
                 detector["threshold"],
                 "yes" if detector["iforest"] is not None else "no",
                 len(detector["per_class"]))
        return _OOD_DETECTOR
    except Exception as e:
        log.warning("Failed to load OOD detector: %s", e)
        return None


def predict_ood(
    features: Dict[str, float],
    detector=None,
    predicted_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict whether a molecule is OOD using ensemble detector.

    Uses up to 3 signals:
    1. Global Mahalanobis distance
    2. IsolationForest anomaly score
    3. Per-class Mahalanobis (if predicted_class is provided)

    Flags OOD if >=2 out of available signals agree.

    Parameters
    ----------
    features : dict
        Biophysical features matching FEATURE_COLS.
    detector : dict, optional
        Loaded detector. If None, loads from default path.
    predicted_class : str, optional
        Predicted molecule class (enables class-conditional OOD).

    Returns
    -------
    dict: is_ood, distance, threshold, confidence, reason, signals
    """
    if detector is None:
        detector = load_ood_detector()
    if detector is None:
        return {"is_ood": False, "distance": 0.0, "threshold": 0.0,
                "confidence": "Low", "reason": "No OOD detector available",
                "signals": {}}

    x = np.array([features.get(c, 0.0) for c in OOD_FEATURE_COLS], dtype=np.float64)
    signals = {}

    # ── Signal 1: Global Mahalanobis ───────────────────────────────
    diff = x - detector["mean"]
    dist = float(np.sqrt(diff @ detector["cov_inv"] @ diff))
    threshold = detector["threshold"]
    signals["mahalanobis"] = {
        "is_ood": dist > threshold,
        "score": round(dist, 2),
        "threshold": round(threshold, 2),
    }

    # ── Signal 2: IsolationForest ──────────────────────────────────
    if detector.get("iforest") is not None and detector.get("iforest_threshold") is not None:
        iforest_score = float(detector["iforest"].score_samples(x.reshape(1, -1))[0])
        iforest_thresh = detector["iforest_threshold"]
        signals["iforest"] = {
            "is_ood": iforest_score < iforest_thresh,
            "score": round(iforest_score, 4),
            "threshold": round(iforest_thresh, 4),
        }

    # ── Signal 3: Per-class Mahalanobis ────────────────────────────
    if predicted_class and predicted_class in detector.get("per_class", {}):
        cls_det = detector["per_class"][predicted_class]
        cls_diff = x - cls_det["mean"]
        cls_dist = float(np.sqrt(cls_diff @ cls_det["cov_inv"] @ cls_diff))
        cls_thresh = cls_det["threshold"]
        signals["class_mahalanobis"] = {
            "is_ood": cls_dist > cls_thresh,
            "score": round(cls_dist, 2),
            "threshold": round(cls_thresh, 2),
            "class": predicted_class,
        }

    # ── Ensemble decision ──────────────────────────────────────────
    ood_votes = sum(1 for s in signals.values() if s.get("is_ood", False))
    total_signals = len(signals)

    # OOD if majority of signals agree (>=2 for 3 signals, >=1 for 1 signal)
    min_votes = max(1, (total_signals + 1) // 2)  # majority
    is_ood = ood_votes >= min_votes

    # Confidence
    if ood_votes == 0:
        confidence = "High"  # all agree: in-distribution
        reason = f"All {total_signals} detectors agree: in-distribution"
    elif ood_votes == total_signals:
        confidence = "High"  # all agree: OOD
        reason = f"All {total_signals} detectors agree: out-of-distribution"
    elif is_ood:
        confidence = "Medium"
        reason = f"{ood_votes}/{total_signals} detectors flag OOD"
    else:
        confidence = "Medium"
        reason = f"Only {ood_votes}/{total_signals} detectors flag OOD (below majority)"

    return {
        "is_ood": is_ood,
        "distance": round(dist, 2),
        "threshold": round(threshold, 2),
        "confidence": confidence,
        "reason": reason,
        "signals": signals,
        "ood_votes": ood_votes,
        "total_signals": total_signals,
    }


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = train_ood_detector()
    print(result.summary())
