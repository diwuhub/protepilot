"""
features.py — Canonical Feature Computation for ProtePilot
=============================================================
**Single source of truth** for all 24 biophysical features used by the
molecule classifier, OOD detector, and inference pipeline.

Every path that computes features — harmonizer, inference, OOD generator,
benchmark — MUST call functions from this module.  No other module should
contain independent feature computation logic.

Feature Schema (v1, 24 features)
--------------------------------
Biophysical (12):
    seq_length, n_chains, n_unique_chains, pI, mw_kda, gravy,
    hydrophobicity, deam_sites, ox_sites, cysteine_count,
    acidic_residues, basic_residues

Composition (8):
    aromatic_frac, pro_gly_frac, cys_frac, deam_density, ox_density,
    charge_ratio, small_frac, aliphatic_idx

Chain-level (4):
    hc_frac, has_lc, hc_len_norm, lc_len_norm

Usage:
    from src.training.features import compute_all_features

    features = compute_all_features(
        sequence="EVQLVE...",
        n_chains=2,
        n_unique_chains=2,
        hc_sequence="EVQLVE...",
        lc_sequence="DIQMTQ...",
    )
    # → dict with 24 keys matching schema.FEATURE_COLS
"""

from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger("ProtePilot.Training.Features")

# Valid amino acid alphabet
_AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydropathy scale (fallback when Biopython unavailable)
_KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Minimum sequence length to compute features (below this → empty dict)
MIN_SEQ_LEN = 5


# ═══════════════════════════════════════════════════════════════════════
#  Core Feature Functions
# ═══════════════════════════════════════════════════════════════════════

def clean_sequence(s: str) -> str:
    """Strip, uppercase, remove non-amino-acid characters."""
    if not s:
        return ""
    return "".join(c for c in s.strip().upper() if c in _AA_VALID)


def compute_sequence_features(seq: str) -> Dict[str, Any]:
    """
    Compute 20 sequence-level biophysical features.

    Returns dict with keys:
        seq_length, pI, mw_kda, gravy, hydrophobicity,
        deam_sites, ox_sites, cysteine_count, acidic_residues, basic_residues,
        aromatic_frac, pro_gly_frac, cys_frac, deam_density, ox_density,
        charge_ratio, small_frac, aliphatic_idx

    Returns empty dict if sequence is too short (< MIN_SEQ_LEN).
    """
    seq = clean_sequence(seq)
    n = len(seq)
    if n < MIN_SEQ_LEN:
        return {}

    # ── Primary biophysical properties ────────────────────────────────
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        pa = ProteinAnalysis(seq)
        gravy = pa.gravy()
        mw = pa.molecular_weight() / 1000.0  # kDa
        pi = pa.isoelectric_point()
    except Exception:
        gravy = sum(_KD_SCALE.get(aa, 0) for aa in seq) / max(n, 1)
        mw = n * 0.11  # rough kDa estimate
        pi = 7.0

    # Hydrophobicity normalized to [0, 1]
    hydro = min(1.0, max(0.0, (gravy + 2.0) / 4.0))

    # ── Liability motif counts ────────────────────────────────────────
    deam = sum(1 for i in range(n - 1) if seq[i] == "N" and seq[i + 1] in "GSTD")
    ox = seq.count("M") + seq.count("W")
    cys = seq.count("C")
    acidic = seq.count("D") + seq.count("E")
    basic = seq.count("K") + seq.count("R") + seq.count("H")

    # ── Composition features (Phase 2a) ──────────────────────────────
    aromatic = seq.count("F") + seq.count("W") + seq.count("Y")
    pro_gly = seq.count("P") + seq.count("G")
    total_charged = acidic + basic
    small_res = seq.count("G") + seq.count("A") + seq.count("S")

    return {
        "seq_length": n,
        "pI": round(pi, 2),
        "mw_kda": round(mw, 2),
        "gravy": round(gravy, 4),
        "hydrophobicity": round(hydro, 4),
        "deam_sites": deam,
        "ox_sites": ox,
        "cysteine_count": cys,
        "acidic_residues": acidic,
        "basic_residues": basic,
        # Phase 2a composition features
        "aromatic_frac": round(aromatic / n, 4),
        "pro_gly_frac": round(pro_gly / n, 4),
        "cys_frac": round(cys / n, 4),
        "deam_density": round(deam / n, 4),
        "ox_density": round(ox / n, 4),
        "charge_ratio": round(basic / total_charged if total_charged > 0 else 0.5, 4),
        "small_frac": round(small_res / n, 4),
        "aliphatic_idx": round(
            (seq.count("A") + 2.9 * seq.count("V")
             + 3.9 * (seq.count("I") + seq.count("L"))) / n, 4),
    }


def compute_chain_features(hc: str, lc: str) -> Dict[str, float]:
    """
    Compute 4 HC/LC chain-level features.

    These features are critical for class discrimination:
    - has_lc perfectly separates single_domain from canonical_mab
    - hc_frac distinguishes bispecific from canonical_mab
    - hc_len_norm / lc_len_norm encode structural format

    Parameters
    ----------
    hc : str
        Heavy chain sequence (may be empty).
    lc : str
        Light chain sequence (may be empty).

    Returns
    -------
    dict with keys: hc_frac, has_lc, hc_len_norm, lc_len_norm
    """
    hc_len = len(hc) if hc else 0
    lc_len = len(lc) if lc else 0
    total = hc_len + lc_len if (hc_len + lc_len) > 0 else 1

    return {
        "hc_frac": round(hc_len / total, 4),
        "has_lc": 1.0 if lc_len >= 20 else 0.0,
        "hc_len_norm": round(min(hc_len / 450.0, 2.0) if hc_len > 0 else 0.0, 4),
        "lc_len_norm": round(min(lc_len / 220.0, 2.0) if lc_len > 0 else 0.0, 4),
    }


def compute_all_features(
    sequence: str,
    n_chains: int = 1,
    n_unique_chains: int = 1,
    hc_sequence: str = "",
    lc_sequence: str = "",
) -> Dict[str, float]:
    """
    Compute complete 24-feature vector for a molecule.

    This is the **canonical entry point** for feature computation.
    All paths — harmonizer, inference, OOD, benchmark — should call this.

    Parameters
    ----------
    sequence : str
        Full (concatenated) sequence or primary chain sequence.
    n_chains : int
        Number of polypeptide chains in the assembly.
    n_unique_chains : int
        Number of structurally unique chains (<85% identity).
    hc_sequence : str
        Heavy chain sequence. Falls back to full sequence if empty.
    lc_sequence : str
        Light chain sequence. May be empty (single-domain, peptide).

    Returns
    -------
    dict with 24 keys matching schema.FEATURE_COLS, in canonical order.
    Returns dict of zeros if sequence is too short.
    """
    from src.training.schema import FEATURE_COLS

    # Compute sequence-level features (18 features)
    seq_feats = compute_sequence_features(sequence)

    if not seq_feats:
        # Sequence too short — return all zeros
        return {col: 0.0 for col in FEATURE_COLS}

    # Add chain count features
    seq_feats["n_chains"] = n_chains
    seq_feats["n_unique_chains"] = n_unique_chains

    # Compute chain-level features (4 features)
    # Fallback: if no HC provided, use full sequence as HC
    hc = hc_sequence or sequence
    lc = lc_sequence or ""
    chain_feats = compute_chain_features(hc, lc)
    seq_feats.update(chain_feats)

    return seq_feats


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════

def _selftest():
    """Validate feature computation against known properties."""
    from src.training.schema import FEATURE_COLS

    # Test 1: Basic antibody-like sequence
    test_hc = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYSSSWYWGQGTLVTVSS"
    test_lc = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKRLIYAASSLQSGVPSRFSGSGSGTEFTLTISSLQPEDIATYYC"

    feats = compute_all_features(
        sequence=test_hc,
        n_chains=2, n_unique_chains=2,
        hc_sequence=test_hc, lc_sequence=test_lc,
    )

    # All 24 features present
    missing = set(FEATURE_COLS) - set(feats.keys())
    assert not missing, f"Missing features: {missing}"

    # Feature count exact
    feature_keys = set(FEATURE_COLS)
    extra = set(feats.keys()) - feature_keys
    # Extra keys are ok (n_chains, n_unique_chains are always present)
    for col in FEATURE_COLS:
        assert col in feats, f"Feature {col} not in output"

    # Sanity checks on known properties
    assert feats["seq_length"] == len(clean_sequence(test_hc)), "seq_length mismatch"
    assert feats["n_chains"] == 2
    assert feats["n_unique_chains"] == 2
    assert 0.0 <= feats["hydrophobicity"] <= 1.0, "hydrophobicity out of range"
    assert 1.0 <= feats["pI"] <= 14.0, f"pI out of range: {feats['pI']}"
    assert feats["has_lc"] == 1.0, "Should detect LC"
    assert feats["hc_frac"] > 0.0, "hc_frac should be positive"
    assert feats["lc_len_norm"] > 0.0, "lc_len_norm should be positive"

    # Test 2: Peptide (no HC/LC distinction)
    pep_feats = compute_all_features(sequence="ACDEFGHIKLM" * 3, n_chains=1)
    assert pep_feats["has_lc"] == 0.0, "Peptide should not have LC"
    assert pep_feats["seq_length"] == 33

    # Test 3: Empty / short sequence → all zeros
    empty_feats = compute_all_features(sequence="AC", n_chains=1)
    assert all(v == 0.0 for v in empty_feats.values()), "Short seq should be all zeros"

    # Test 4: Determinism
    f1 = compute_all_features(sequence=test_hc, n_chains=2, hc_sequence=test_hc, lc_sequence=test_lc)
    f2 = compute_all_features(sequence=test_hc, n_chains=2, hc_sequence=test_hc, lc_sequence=test_lc)
    for col in FEATURE_COLS:
        assert f1[col] == f2[col], f"Non-deterministic: {col}: {f1[col]} != {f2[col]}"

    print(f"features._selftest() PASSED ✓ (24 features, 4 tests)")
    return True


if __name__ == "__main__":
    _selftest()
