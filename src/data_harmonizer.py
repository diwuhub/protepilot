"""
data_harmonizer.py -- ProtePilot
===========================================================
Dual-Chain CSV Data Harmonizer for Training Data Ingestion

Intelligently ingests CSV files (e.g., Jain et al. 137-antibody dataset),
detects VH/VL chain columns, fuses them with a standard linker, validates
sequences, and returns a standardized DataFrame for ML training.

Author: ProtePilot
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, List, Tuple, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Standard (G4S)x3 linker for VH-VL concatenation
STANDARD_LINKER = "GGGGSGGGGSGGGGS"

# Column alias groups (case-insensitive matching)
ID_ALIASES = ["name", "antibody", "id", "clone", "molecule", "sample"]
VH_ALIASES = ["vh", "heavy chain", "heavy_chain", "hc", "vh_sequence",
              "heavy_seq", "hc_sequence", "variable_heavy"]
VL_ALIASES = ["vl", "light chain", "light_chain", "lc", "vl_sequence",
              "light_seq", "lc_sequence", "variable_light"]
# Fallback: single combined sequence column
SINGLE_SEQ_ALIASES = ["sequence", "seq", "amino_acid", "protein_sequence",
                      "aa_sequence", "full_sequence"]
TARGET_ALIASES = ["ac-sins", "ac_sins", "bvp", "titer", "aggregation",
                  "agg", "stability", "viscosity", "visc", "hic", "sec",
                  "tm", "hmw", "monomer", "kd", "expression"]

# Kyte-Doolittle hydrophobicity scale
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    """Case-insensitive column matcher.  Returns first hit or None."""
    col_map = {c.lower().strip(): c for c in columns}
    # Exact match first
    for alias in aliases:
        if alias.lower() in col_map:
            return col_map[alias.lower()]
    # Substring match
    for alias in aliases:
        for c_lower, c_orig in col_map.items():
            if alias.lower() in c_lower:
                return c_orig
    return None


def _clean_sequence(raw: object) -> str:
    """Uppercase, strip whitespace/dashes/digits, keep only standard AAs."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    seq = raw.strip().upper()
    seq = re.sub(r"[^A-Z]", "", seq)  # remove non-alpha
    # Keep only valid amino acids
    seq = "".join(ch for ch in seq if ch in VALID_AA)
    return seq


# ---------------------------------------------------------------------------
# DataHarmonizer
# ---------------------------------------------------------------------------

class DataHarmonizer:
    """
    Dual-chain CSV harmonizer for biopharmaceutical ML training.

    Detects VH / VL columns, fuses with (G4S)x3 linker, detects target
    assay columns, and returns a standardized DataFrame:
        [Molecule_ID, Combined_Sequence, <target columns ...>]
    """

    def __init__(self, min_sequence_length: int = 20):
        self.min_sequence_length = min_sequence_length

    # -- Column detection ---------------------------------------------------

    def _detect_columns(
        self, df: pd.DataFrame,
    ) -> Dict[str, Optional[str]]:
        """Return detected column mapping."""
        cols = list(df.columns)
        detected = {
            "id": _match_column(cols, ID_ALIASES),
            "vh": _match_column(cols, VH_ALIASES),
            "vl": _match_column(cols, VL_ALIASES),
            "single_seq": _match_column(cols, SINGLE_SEQ_ALIASES),
            "targets": [],
        }
        # Detect all target columns
        col_lower_map = {c.lower().strip(): c for c in cols}
        for alias in TARGET_ALIASES:
            for c_lower, c_orig in col_lower_map.items():
                if alias.lower() in c_lower and c_orig not in detected["targets"]:
                    detected["targets"].append(c_orig)
        return detected

    # -- Harmonize ----------------------------------------------------------

    def harmonize(self, df: pd.DataFrame) -> Dict:
        """
        Main entry point.  Returns dict with:
            status, data (DataFrame), n_valid, n_dropped,
            target_columns, warnings, detected_columns
        """
        warnings: List[str] = []
        det = self._detect_columns(df)

        has_vh = det["vh"] is not None
        has_vl = det["vl"] is not None
        has_single = det["single_seq"] is not None

        if not has_vh and not has_vl and not has_single:
            return {
                "status": "error",
                "message": "No sequence column detected (looked for VH, VL, Sequence).",
                "data": pd.DataFrame(),
                "n_valid": 0,
                "n_dropped": len(df),
                "target_columns": [],
                "warnings": ["No sequence column found."],
                "detected_columns": det,
            }

        mode = "dual" if (has_vh or has_vl) else "single"
        if mode == "dual":
            if has_vh and has_vl:
                warnings.append(f"Dual-chain mode: VH='{det['vh']}', VL='{det['vl']}'")
            elif has_vh:
                warnings.append(f"VH-only mode: VH='{det['vh']}' (no VL detected)")
            else:
                warnings.append(f"VL-only mode: VL='{det['vl']}' (no VH detected)")
        else:
            warnings.append(f"Single-sequence mode: column='{det['single_seq']}'")

        # Build rows
        rows = []
        n_dropped = 0

        for idx in range(len(df)):
            # ID
            mol_id = str(df.iloc[idx][det["id"]]).strip() if det["id"] else f"Seq_{idx + 1}"

            # Sequence fusion
            combined = ""
            if mode == "dual":
                vh_raw = df.iloc[idx][det["vh"]] if has_vh else ""
                vl_raw = df.iloc[idx][det["vl"]] if has_vl else ""
                vh_clean = _clean_sequence(vh_raw)
                vl_clean = _clean_sequence(vl_raw)

                if vh_clean and vl_clean:
                    combined = vh_clean + STANDARD_LINKER + vl_clean
                elif vh_clean:
                    combined = vh_clean
                elif vl_clean:
                    combined = vl_clean
            else:
                raw = df.iloc[idx][det["single_seq"]]
                combined = _clean_sequence(raw)

            # Validate
            if len(combined) < self.min_sequence_length:
                n_dropped += 1
                continue
            if not all(ch in VALID_AA for ch in combined):
                n_dropped += 1
                continue

            row = {"Molecule_ID": mol_id, "Combined_Sequence": combined}

            # Attach target values
            for tcol in det["targets"]:
                val = df.iloc[idx].get(tcol, np.nan)
                try:
                    row[tcol] = float(val)
                except (ValueError, TypeError):
                    row[tcol] = np.nan

            rows.append(row)

        if not rows:
            return {
                "status": "error",
                "message": f"All {len(df)} rows failed validation (min length={self.min_sequence_length}).",
                "data": pd.DataFrame(),
                "n_valid": 0,
                "n_dropped": n_dropped,
                "target_columns": det["targets"],
                "warnings": warnings,
                "detected_columns": det,
            }

        result_df = pd.DataFrame(rows)
        n_valid = len(result_df)
        log.info("Harmonization complete: %d valid, %d dropped (mode=%s)", n_valid, n_dropped, mode)

        # Compute sequence statistics for OOD baseline
        _combined_seqs = result_df["Combined_Sequence"].dropna().tolist()
        _seq_lengths = [len(s) for s in _combined_seqs if isinstance(s, str) and len(s) > 0]
        _seq_stats = {
            "lengths": _seq_lengths,
            "mean_length": round(sum(_seq_lengths) / max(len(_seq_lengths), 1), 1),
            "std_length": round(
                (sum((x - sum(_seq_lengths)/max(len(_seq_lengths),1))**2 for x in _seq_lengths) / max(len(_seq_lengths)-1, 1)) ** 0.5, 1
            ) if len(_seq_lengths) > 1 else 0.0,
            "n_sequences": len(_seq_lengths),
            "assembly_mode": "dual_vh_vl" if has_vh and has_vl else "single",
        }

        return {
            "status": "success",
            "data": result_df,
            "n_valid": n_valid,
            "n_dropped": n_dropped,
            "target_columns": det["targets"],
            "warnings": warnings,
            "detected_columns": det,
            "sequence_stats": _seq_stats,
        }


# ---------------------------------------------------------------------------
# Biophysical Feature Extraction
# ---------------------------------------------------------------------------

def compute_biophysical_features(sequence: str) -> np.ndarray:
    """
    Compute 7-dim biophysical feature vector from amino acid sequence.

    Features: [MW_kDa, pI_est, GRAVY, Cys_count, Length, Deamidation_NG, Oxidation_MW]
    """
    n = len(sequence)
    if n == 0:
        return np.zeros(7, dtype=np.float32)

    mw_kda = n * 0.11  # rough kDa estimate

    basic = sequence.count("K") + sequence.count("R") + sequence.count("H")
    acidic = sequence.count("D") + sequence.count("E")
    pi_est = 7.0 + 0.5 * (basic - acidic) / max(n, 1)

    gravy = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in sequence) / n

    cys = sequence.count("C")
    deam = len(re.findall(r"N[GS]", sequence))
    ox = sequence.count("M") + sequence.count("W")

    return np.array([mw_kda, pi_est, gravy, cys, n, deam, ox], dtype=np.float32)


BIOPHYS_FEATURE_NAMES = ["MW_kDa", "pI_est", "GRAVY", "Cys", "Length", "Deamidation", "Oxidation"]


# ---------------------------------------------------------------------------
# Unified Multi-Target Detection for UnifiedMultiTaskModel
# ---------------------------------------------------------------------------

# Mapping: unified task name → list of CSV column name patterns (case-insensitive)
UNIFIED_TARGET_PATTERNS = {
    "tm":               ["tm", "melting", "dsf", "melt_temp", "thermal_melt"],
    "aggregation_risk": ["agg", "sec", "hmw", "aggregation", "monomer"],
    "stability":        ["stab", "thermal_stab"],
    "viscosity_risk":   ["visc", "viscosity"],
    "hydrophobicity":   ["hydro", "gravy", "hic"],
    "potency":          ["potency", "binding", "ec50", "kd", "affinity"],
    "ka":               ["ka", "k_a", "adsorption"],
    "nu":               ["nu", "characteristic_charge"],
}


def detect_unified_targets(columns: List[str]) -> Dict[str, str]:
    """
    Detect which unified model tasks can be mapped from CSV columns.

    Parameters
    ----------
    columns : list of str
        Column names from the uploaded CSV.

    Returns
    -------
    dict
        Mapping from unified task name → matched CSV column name.
        Only includes tasks where a matching column was found.
    """
    detected = {}
    col_lower_map = {c.lower().strip(): c for c in columns}

    for task, patterns in UNIFIED_TARGET_PATTERNS.items():
        # Exact match first
        for pat in patterns:
            if pat.lower() in col_lower_map:
                detected[task] = col_lower_map[pat.lower()]
                break
        if task in detected:
            continue
        # Substring match
        for pat in patterns:
            for c_lower, c_orig in col_lower_map.items():
                if pat.lower() in c_lower and task not in detected:
                    detected[task] = c_orig
                    break
            if task in detected:
                break

    return detected


def prepare_training_matrix(
    result_df: pd.DataFrame,
    target_column: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build (X, y) from a harmonized DataFrame.

    Parameters
    ----------
    result_df : DataFrame with 'Combined_Sequence' and target_column
    target_column : name of target assay column

    Returns
    -------
    X : (N, 7) biophysical feature matrix
    y : (N,) target values
    ids : list of Molecule_ID strings
    """
    if target_column not in result_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Drop rows with NaN target
    valid = result_df.dropna(subset=[target_column]).copy()
    if len(valid) == 0:
        raise ValueError(f"No valid (non-NaN) rows for target '{target_column}'.")

    seqs = valid["Combined_Sequence"].tolist()
    ids = valid["Molecule_ID"].tolist()
    y = valid[target_column].values.astype(np.float32)

    X = np.vstack([compute_biophysical_features(s) for s in seqs]).astype(np.float32)

    log.info("Training matrix: X=%s, y=%s, target=%s", X.shape, y.shape, target_column)
    return X, y, ids


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  DataHarmonizer v2.0 Self-Test")
    print("=" * 60)

    # Mock dual-chain CSV (Jain-style)
    mock = pd.DataFrame({
        "Antibody": ["Ab001", "Ab002", "Ab003", "Ab004", "Ab005"],
        "VH": [
            "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTR",
            "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQGLEWMGATYPGNSD",
            "SHORT",  # too short
            "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGST",
            "EVQLVESGGGLVQPGG---SLRLSCAASGFNIKDTYIH",  # dashes
        ],
        "VL": [
            "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKLLIYAASSLQS",
            "DIVMTQSPDSLAVSLGERATINCKSSQSVLYSSNNKNYLAWYQQKPGQPPKLLIY",
            "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKLLIYAASSLQS",
            "",  # missing VL
            "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKLLIYAASSLQS",
        ],
        "AC-SINS": [0.12, 0.45, 0.33, 0.22, 0.55],
        "BVP": [1.2, 3.4, 2.1, 0.8, 2.9],
        "Irrelevant": ["x", "y", "z", "w", "v"],
    })

    harmonizer = DataHarmonizer(min_sequence_length=30)
    result = harmonizer.harmonize(mock)

    print(f"\nStatus: {result['status']}")
    print(f"Valid: {result['n_valid']}, Dropped: {result['n_dropped']}")
    print(f"Targets: {result['target_columns']}")
    print(f"Warnings: {result['warnings']}")

    if result["status"] == "success":
        df = result["data"]
        print(f"\nResult DataFrame ({len(df)} rows):")
        print(df[["Molecule_ID", "Combined_Sequence"]].to_string(
            formatters={"Combined_Sequence": lambda s: s[:40] + "..."}))

        # Test feature extraction
        X, y, ids = prepare_training_matrix(df, "AC-SINS")
        print(f"\nFeature matrix: X={X.shape}, y={y.shape}")
        print(f"Feature names: {BIOPHYS_FEATURE_NAMES}")
        print(f"First sample: {X[0]}")
        print(f"First target: {y[0]}")

        # Verify dual-chain fusion
        first_seq = df.iloc[0]["Combined_Sequence"]
        assert STANDARD_LINKER in first_seq, "Linker not found in fused sequence"
        print(f"\nLinker '{STANDARD_LINKER}' found in fused sequences: OK")

    # Test single-seq fallback
    mock_single = pd.DataFrame({
        "Name": ["X1", "X2"],
        "Sequence": ["EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH", "ABCDE"],
        "Aggregation": [0.5, 0.3],
    })
    r2 = harmonizer.harmonize(mock_single)
    assert r2["n_valid"] == 1
    print(f"\nSingle-seq fallback: {r2['n_valid']} valid -- OK")

    print("\n" + "=" * 60)
    print("  Self-Test: ALL PASSED")
    print("=" * 60)
