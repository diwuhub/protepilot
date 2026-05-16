"""
schema.py — Training Data Schema & Split Management
=====================================================
Single source of truth for training data column roles:

  LABEL columns:        What the model predicts (molecule_class, is_ood, etc.)
  FEATURE columns:      Input features for the model (biophysical properties)
  METADATA columns:     Provenance info (name, source, sequences) — not used in training
  TRAIN_ONLY columns:   Columns produced during training (split_role, fold, etc.)

Also manages reproducible train/val/test/holdout splits with fixed seeds.

Usage:
    from src.training.schema import (
        LABEL_COLS, FEATURE_COLS, METADATA_COLS,
        create_split, BENCHMARK_HOLDOUT_NAMES,
    )
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
#  Column Role Definitions
# ═══════════════════════════════════════════════════════════════════════

# What the model predicts — ground truth labels
LABEL_COLS: List[str] = [
    "molecule_class",           # Primary label: MoleculeClass enum value
]

# Biophysical features used as model input — order matters (must match inference)
FEATURE_COLS: List[str] = [
    "seq_length",               # Total assembled sequence length (residues)
    "n_chains",                 # Number of polypeptide chains (from assembly)
    "n_unique_chains",          # Number of unique chains (key for bispecific detection)
    "pI",                       # Isoelectric point (pH units)
    "mw_kda",                   # Molecular weight (kDa)
    "gravy",                    # GRAVY hydropathy index
    "hydrophobicity",           # Normalized hydrophobicity [0, 1]
    "deam_sites",               # Deamidation liability site count
    "ox_sites",                 # Oxidation liability site count (Met + Trp)
    "cysteine_count",           # Total cysteine count
    "acidic_residues",          # D + E count
    "basic_residues",           # K + R + H count
    # Phase 2a features
    "aromatic_frac",            # Aromatic residue fraction (F + W + Y)
    "pro_gly_frac",             # Proline + Glycine fraction (flexibility)
    "cys_frac",                 # Cysteine fraction (disulfide pattern)
    "deam_density",             # Deamidation sites per residue
    "ox_density",               # Oxidation sites per residue
    "charge_ratio",             # Basic / (acidic + basic) ratio
    "small_frac",               # Small residue fraction (G + A + S)
    "aliphatic_idx",            # Aliphatic index (thermal stability)
    # HC/LC chain features
    "hc_frac",                  # HC length / total length
    "has_lc",                   # Binary: has light chain (>=20 aa)
    "hc_len_norm",              # HC length / 450 (normalized)
    "lc_len_norm",              # LC length / 220 (normalized)
]

# Provenance / identification — used for traceability, not training
METADATA_COLS: List[str] = [
    "name",                     # Molecule identifier
    "hc_sequence",              # Heavy chain / primary sequence (raw input)
    "lc_sequence",              # Light chain sequence (may be empty)
    "source",                   # Data source: jain137 / therasabdab / synthetic
]

# Added during split creation — not in raw harmonized data
SPLIT_COLS: List[str] = [
    "split",                    # "train" / "val" / "test" / "holdout"
]

# Full expected column set in the harmonized CSV
ALL_HARMONIZED_COLS: List[str] = METADATA_COLS[:1] + LABEL_COLS + METADATA_COLS[1:] + FEATURE_COLS[:-1]
# Note: the actual CSV may have columns in different order; use sets for validation


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Holdout Panel
# ═══════════════════════════════════════════════════════════════════════

# These molecules are ALWAYS held out from training and used for
# before/after platform-level comparison. They must never appear in
# train or val splits.
BENCHMARK_HOLDOUT_NAMES: Set[str] = {
    "trastuzumab",              # Canonical mAb reference
    "bevacizumab",              # Canonical mAb, known high agg
    "rituximab",                # Canonical mAb, known low agg
    "NISTmAb",                  # Gold standard RM 8671
}


# ═══════════════════════════════════════════════════════════════════════
#  Reproducible Split Creation
# ═══════════════════════════════════════════════════════════════════════

def create_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    holdout_names: Set[str] = None,
) -> pd.DataFrame:
    """
    Create a reproducible stratified train/val/test/holdout split.

    Parameters
    ----------
    df : DataFrame
        Harmonized training data with 'molecule_class' and 'name' columns.
    train_frac, val_frac, test_frac : float
        Proportions (must sum to 1.0).
    seed : int
        Random seed for reproducibility.
    holdout_names : set of str, optional
        Molecule names to exclude from training (benchmark holdout).
        Defaults to BENCHMARK_HOLDOUT_NAMES.

    Returns
    -------
    DataFrame with added 'split' column: "train" / "val" / "test" / "holdout".
    """
    if holdout_names is None:
        holdout_names = BENCHMARK_HOLDOUT_NAMES

    df = df.copy()
    df["split"] = ""

    # Mark holdout entries
    name_lower = df["name"].str.lower()
    holdout_mask = name_lower.isin({n.lower() for n in holdout_names})
    df.loc[holdout_mask, "split"] = "holdout"

    # Stratified split on remaining data
    remaining = df[df["split"] == ""].index
    rng = np.random.RandomState(seed)

    classes = df.loc[remaining, "molecule_class"].unique()
    for cls in classes:
        cls_idx = df.loc[remaining][df.loc[remaining, "molecule_class"] == cls].index.values
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))

        df.loc[cls_idx[:n_test], "split"] = "test"
        df.loc[cls_idx[n_test:n_test + n_val], "split"] = "val"
        df.loc[cls_idx[n_test + n_val:], "split"] = "train"

    return df


def split_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Summarize split distribution."""
    counts = df["split"].value_counts().to_dict()
    counts["total"] = len(df)
    return counts


def validate_schema(df: pd.DataFrame) -> List[str]:
    """
    Validate that a DataFrame conforms to the training schema.

    Returns list of violations (empty = valid).
    """
    violations = []

    # Required columns
    required = set(LABEL_COLS + FEATURE_COLS + ["name", "source"])
    missing = required - set(df.columns)
    if missing:
        violations.append(f"Missing required columns: {missing}")

    # Label values
    if "molecule_class" in df.columns:
        from src.type_defs import MoleculeClass
        valid_classes = {mc.value for mc in MoleculeClass}
        invalid = set(df["molecule_class"].unique()) - valid_classes
        if invalid:
            violations.append(f"Invalid molecule_class values: {invalid}")

    # Feature ranges
    if "hydrophobicity" in df.columns:
        h = df["hydrophobicity"]
        if h.min() < -0.1 or h.max() > 1.1:
            violations.append(f"hydrophobicity out of [0,1]: min={h.min():.2f}, max={h.max():.2f}")

    if "pI" in df.columns:
        pi = df["pI"].dropna()
        if len(pi) > 0 and (pi.min() < 1.0 or pi.max() > 14.0):
            violations.append(f"pI out of [1,14]: min={pi.min():.2f}, max={pi.max():.2f}")

    # No empty features
    for col in FEATURE_COLS:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                violations.append(f"{col}: {n_null} null values")

    return violations


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    """Validate schema definitions are internally consistent."""
    # No overlap between label and feature cols
    overlap = set(LABEL_COLS) & set(FEATURE_COLS)
    assert not overlap, f"Label/feature overlap: {overlap}"

    # No overlap between metadata and feature cols
    overlap2 = set(METADATA_COLS) & set(FEATURE_COLS)
    assert not overlap2, f"Metadata/feature overlap: {overlap2}"

    # Feature columns must be a fixed list (order matters for inference)
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), "Duplicate feature columns"

    # Holdout names are non-empty
    assert len(BENCHMARK_HOLDOUT_NAMES) >= 2, "Need at least 2 holdout molecules"

    print(f"schema selftest PASS ({len(FEATURE_COLS)} features, {len(LABEL_COLS)} labels, "
          f"{len(METADATA_COLS)} metadata, {len(BENCHMARK_HOLDOUT_NAMES)} holdout)")
    return True


if __name__ == "__main__":
    _selftest()
