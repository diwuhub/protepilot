"""
data_harmonizer.py — Build unified training CSV from public datasets
====================================================================
Reads Jain137 and TheraSAbDab, maps to our MoleculeClass labels,
computes biophysical features, and outputs a harmonized CSV.

Output schema:
    name, molecule_class, hc_sequence, lc_sequence, seq_length,
    n_chains, pI, mw_kda, gravy, hydrophobicity,
    deam_sites, ox_sites, cysteine_count, acidic_residues, basic_residues,
    source

Usage:
    python -m src.training.data_harmonizer
    python -m src.training.data_harmonizer --output data/training/classifier_data.csv
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("ProtePilot.Training.Harmonizer")

# Chain detection thresholds (cross-module constants)
try:
    from src.platform_config import MIN_SEQUENCE_LENGTH
except ImportError:
    MIN_SEQUENCE_LENGTH = 10

# ═══════════════════════════════════════════════════════════════════════
#  TheraSAbDab format → MoleculeClass mapping
# ═══════════════════════════════════════════════════════════════════════

_THERASABDAB_FORMAT_MAP = {
    "Whole mAb": "canonical_mab",
    "Whole mAb ADC": "adc",
    "Bispecific mAb": "bispecific",
    "Bispecific Mixed mAb and scFv": "bispecific",
    "Bispecific scFv": "bispecific",
    "Bispecific mAb with Domain Crossover": "bispecific",
    "Bispecific Whole mAb": "bispecific",
    "Bispecific Single Domains (VH-VH')": "bispecific",
    "Whole mAb Fusion": "fc_fusion",
    "Fusion Protein": "fusion_protein",
    "Fusion Protein (whole mAb with protein)": "fc_fusion",
    "Fab": "single_domain",  # Fab fragments map to single_domain
    "scFv": "single_domain",
    "Whole mAb Radiolabelled": "canonical_mab",
    "Nanobody": "single_domain",
    "VHH": "single_domain",
    "Single Domain": "single_domain",
    "Canine Whole mAb": "canonical_mab",
    "Camelid VHH": "single_domain",
    # Everything else → unknown
}


def _map_format(fmt: str) -> str:
    """Map TheraSAbDab format string to our MoleculeClass value."""
    if not fmt or pd.isna(fmt):
        return "unknown"
    fmt = fmt.strip()
    # Exact match first
    if fmt in _THERASABDAB_FORMAT_MAP:
        return _THERASABDAB_FORMAT_MAP[fmt]
    # Fuzzy match — ORDER MATTERS: more specific patterns first!
    fmt_lower = fmt.lower()

    # ── Priority 1: ADC (explicit drug conjugate) ──
    if "adc" in fmt_lower or "conjugat" in fmt_lower:
        return "adc"

    # ── Priority 2: Fc-fusion (has Fc domain fused to non-Ab protein) ──
    if ("+fc" in fmt_lower or "- fc" in fmt_lower or "+ fc" in fmt_lower
            or "scfc" in fmt_lower
            or ("fc" in fmt_lower and "fusion" in fmt_lower)
            or ("(vl) + fc" in fmt_lower or "(vh) + fc" in fmt_lower)
            or ("domain" in fmt_lower and "fc" in fmt_lower)):
        return "fc_fusion"
    if "whole mab fusion" in fmt_lower or "mab with protein" in fmt_lower:
        return "fc_fusion"

    # ── Priority 3: Fusion protein (any non-Fc fusion) ──
    # Must come BEFORE nanobody/fab checks so "Nanobody Fusion Protein" → fusion_protein
    if "fusion" in fmt_lower:
        return "fusion_protein"

    # ── Priority 4: Multi-specific ──
    if "bispecific" in fmt_lower or "bispec" in fmt_lower:
        return "bispecific"
    if "pentavalent" in fmt_lower or "trispecific" in fmt_lower or "tetraspecific" in fmt_lower:
        return "bispecific"

    # ── Priority 5: Single-domain / fragment ──
    if "nanobody" in fmt_lower or "vhh" in fmt_lower:
        return "single_domain"
    if "single domain" in fmt_lower or "vh-" in fmt_lower:
        return "single_domain"

    # ── Priority 6: Fragment (Fab/scFv without fusion) ──
    if "fab" in fmt_lower or "scfv" in fmt_lower or "fv " in fmt_lower:
        return "single_domain"

    # ── Priority 7: Other ──
    if "peptide" in fmt_lower:
        return "peptide"
    if "scaffold" in fmt_lower or "darpin" in fmt_lower:
        return "engineered_scaffold"
    if "dimer" in fmt_lower:
        return "engineered_scaffold"
    if "igm" in fmt_lower or "mab" in fmt_lower or "antibod" in fmt_lower or "igg" in fmt_lower:
        return "canonical_mab"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════
#  Feature extraction (lightweight — no ML imports)
# ═══════════════════════════════════════════════════════════════════════

_AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")

def _clean_seq(s: str) -> str:
    """Clean and validate a sequence string."""
    if not s or pd.isna(s) or s.strip().lower() == "na":
        return ""
    return "".join(c for c in s.strip().upper() if c in _AA_VALID)


# ── Feature computation: delegated to canonical features.py ──────────
# All feature logic lives in src.training.features (single source of truth).
# These thin wrappers preserve the internal API used throughout this file.
from src.training.features import (
    compute_sequence_features as _compute_features,
    compute_chain_features as _compute_chain_features,
)


# ═══════════════════════════════════════════════════════════════════════
#  Dataset builders
# ═══════════════════════════════════════════════════════════════════════

def load_jain137(data_dir: str) -> List[Dict[str, Any]]:
    """Load Jain-137 dataset as canonical_mab entries."""
    path = os.path.join(data_dir, "Jain137_Cleaned_Training_Data.csv")
    if not os.path.exists(path):
        log.warning("Jain137 not found at %s", path)
        return []

    df = pd.read_csv(path, encoding="utf-8-sig")
    records = []
    for _, row in df.iterrows():
        vh = _clean_seq(str(row.get("VH", "")))
        vl = _clean_seq(str(row.get("VL", "")))
        if len(vh) < 50 or len(vl) < 50:
            continue
        combined = vh + vl
        feats = _compute_features(combined)
        if not feats:
            continue
        records.append({
            "name": str(row.get("Name", "unknown")),
            "molecule_class": "canonical_mab",
            "hc_sequence": vh,
            "lc_sequence": vl,
            "n_chains": 2,
            "n_unique_chains": 2,  # HC + LC always unique for canonical mAb
            "source": "jain137",
            **feats,
        })
    log.info("Jain137: loaded %d canonical_mab entries", len(records))
    return records


def load_therasabdab(data_dir: str) -> List[Dict[str, Any]]:
    """Load TheraSAbDab dataset with format-based class labels."""
    path = os.path.join(data_dir, "TheraSAbDab_SeqStruc_OnlineDownload.csv")
    if not os.path.exists(path):
        log.warning("TheraSAbDab not found at %s", path)
        return []

    df = pd.read_csv(path, encoding="utf-8-sig")
    records = []
    for _, row in df.iterrows():
        fmt = str(row.get("Format", ""))
        mol_cls = _map_format(fmt)

        # TheraSAbDab column names vary by version: "HC"/"LC" or "HeavySequence"/"LightSequence"
        hc = _clean_seq(str(row.get("HC", row.get("HeavySequence", ""))))
        lc = _clean_seq(str(row.get("LC", row.get("LightSequence", ""))))

        # Must have at least one chain
        if len(hc) < 20 and len(lc) < 20:
            continue

        combined = hc + lc if lc else hc
        feats = _compute_features(combined)
        if not feats:
            continue

        n_chains = (1 if len(hc) >= 20 else 0) + (1 if len(lc) >= 20 else 0)

        # Compute n_unique_chains — bispecifics have extra chains (HC2/LC2)
        hc2 = _clean_seq(str(row.get("HeavySequence(ifbispec)", "")))
        lc2 = _clean_seq(str(row.get("LightSequence(ifbispec)", "")))
        n_unique = n_chains
        if len(hc2) >= 20:
            n_unique += 1
        if len(lc2) >= 20:
            n_unique += 1

        records.append({
            "name": str(row.get("Therapeutic", "unknown")),
            "molecule_class": mol_cls,
            "hc_sequence": hc,
            "lc_sequence": lc,
            "n_chains": n_chains,
            "n_unique_chains": n_unique,
            "source": "therasabdab",
            **feats,
        })

        # ── Bispecific arm extraction: create extra training rows ──
        # Bispecific entries have HC2/LC2 (the second arm) which are distinct
        # sequences.  Storing them as additional bispecific rows almost doubles
        # our bispecific training signal at zero cost.
        if mol_cls == "bispecific" and len(hc2) >= 20:
            arm2_combined = hc2 + lc2 if lc2 else hc2
            arm2_feats = _compute_features(arm2_combined)
            if arm2_feats:
                arm2_n_chains = (1 if len(hc2) >= 20 else 0) + (1 if len(lc2) >= 20 else 0)
                records.append({
                    "name": str(row.get("Therapeutic", "unknown")) + "_arm2",
                    "molecule_class": "bispecific",
                    "hc_sequence": hc2,
                    "lc_sequence": lc2,
                    "n_chains": arm2_n_chains,
                    "n_unique_chains": n_unique,
                    "source": "therasabdab",
                    **arm2_feats,
                })

    n_bispec = sum(1 for r in records if r["molecule_class"] == "bispecific")
    log.info("TheraSAbDab: loaded %d entries (%d bispecific incl. arm2 rows)", len(records), n_bispec)
    return records


def generate_synthetic_peptides(n: int = 30, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate synthetic peptide sequences for class balance."""
    rng = np.random.RandomState(seed)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    records = []
    for i in range(n):
        length = rng.randint(15, 60)
        seq = "".join(rng.choice(aa, size=length))
        feats = _compute_features(seq)
        if not feats:
            continue
        records.append({
            "name": f"synth_peptide_{i}",
            "molecule_class": "peptide",
            "hc_sequence": seq,
            "lc_sequence": "",
            "n_chains": 1,
            "n_unique_chains": 1,
            "source": "synthetic",
            **feats,
        })
    log.info("Synthetic: generated %d peptide entries", len(records))
    return records


def generate_synthetic_nanobodies(n: int = 20, seed: int = 43) -> List[Dict[str, Any]]:
    """Generate synthetic VHH-like sequences for class balance."""
    rng = np.random.RandomState(seed)
    # VHH template: starts with QVQL, ~120 aa, single chain
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    records = []
    for i in range(n):
        length = rng.randint(100, 140)
        prefix = "QVQLQESGGGLVQAGGSLRLSCAAS"
        body = "".join(rng.choice(aa, size=length - len(prefix)))
        seq = prefix + body
        feats = _compute_features(seq)
        if not feats:
            continue
        records.append({
            "name": f"synth_vhh_{i}",
            "molecule_class": "single_domain",
            "hc_sequence": seq,
            "lc_sequence": "",
            "n_chains": 1,
            "n_unique_chains": 1,
            "source": "synthetic",
            **feats,
        })
    log.info("Synthetic: generated %d nanobody entries", len(records))
    return records


def generate_synthetic_scaffolds(n: int = 15, seed: int = 44) -> List[Dict[str, Any]]:
    """Generate synthetic engineered scaffold (DARPin/affibody-like) sequences."""
    rng = np.random.RandomState(seed)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    # DARPin-like: ~130-170 aa, ankyrin repeat, single chain, no LC
    records = []
    for i in range(n):
        length = rng.randint(130, 180)
        # DARPin motif: enriched in L, A, N, G (ankyrin repeat)
        weighted_aa = list("LLLAANNGGHDEIKRSTVY")
        seq = "".join(rng.choice(weighted_aa, size=length))
        feats = _compute_features(seq)
        if not feats:
            continue
        records.append({
            "name": f"synth_scaffold_{i}",
            "molecule_class": "engineered_scaffold",
            "hc_sequence": seq,
            "lc_sequence": "",
            "n_chains": 1,
            "n_unique_chains": 1,
            "source": "synthetic",
            **feats,
        })
    log.info("Synthetic: generated %d engineered scaffold entries", len(records))
    return records


# ═══════════════════════════════════════════════════════════════════════
#  External Database Loaders (user must download files first)
# ═══════════════════════════════════════════════════════════════════════

def load_covabdab(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load CoV-AbDab (Coronavirus Antibody Database) — ~12,900 antibodies/nanobodies.

    Download from: https://opig.stats.ox.ac.uk/webapps/covabdab/
    Click "Download CSV" → save as data/CoV-AbDab.csv

    Key columns: Name, Ab or Nb, VH or VHH, VL, CDRH3, Binds to, Origin
    """
    # Try multiple possible filenames
    for fname in ["CoV-AbDab.csv", "CoV-AbDab_latest.csv", "covabdab.csv"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            break
    else:
        log.debug("CoV-AbDab not found in %s (download from opig.stats.ox.ac.uk/webapps/covabdab/)", data_dir)
        return []

    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception as e:
        log.warning("Failed to read CoV-AbDab: %s", e)
        return []

    records = []
    for _, row in df.iterrows():
        name = str(row.get("Name", row.get("name", "unknown")))
        ab_type = str(row.get("Ab or Nb", "")).strip().lower()

        # Determine molecule class from Ab/Nb type
        if "nb" in ab_type or "nanobody" in ab_type or "vhh" in ab_type:
            mol_cls = "single_domain"
        else:
            mol_cls = "canonical_mab"

        # Get sequences — CoV-AbDab uses "VHorVHH" (no space) and "VL" columns
        # Sequences marked "ND" = not determined → skip
        vh_col = "VHorVHH" if "VHorVHH" in df.columns else ("VH or VHH" if "VH or VHH" in df.columns else "VH")
        vh_raw = str(row.get(vh_col, ""))
        vl_raw = str(row.get("VL", ""))
        if vh_raw.strip().upper() in ("ND", "NA", "NAN", ""):
            continue
        vh = _clean_seq(vh_raw)
        vl = _clean_seq(vl_raw) if vl_raw.strip().upper() not in ("ND", "NA", "NAN", "") else ""

        if len(vh) < 30:
            continue

        combined = vh + vl if vl else vh
        feats = _compute_features(combined)
        if not feats:
            continue

        n_chains = (1 if len(vh) >= 20 else 0) + (1 if len(vl) >= 20 else 0)
        records.append({
            "name": name,
            "molecule_class": mol_cls,
            "hc_sequence": vh,
            "lc_sequence": vl,
            "n_chains": n_chains,
            "n_unique_chains": n_chains,
            "source": "covabdab",
            **feats,
        })

    log.info("CoV-AbDab: loaded %d entries (%d canonical_mab, %d single_domain)",
             len(records),
             sum(1 for r in records if r["molecule_class"] == "canonical_mab"),
             sum(1 for r in records if r["molecule_class"] == "single_domain"))
    return records


def load_sabdab(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load SAbDab (Structural Antibody Database) — ~18,700 Fv structures.

    Download from: https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/
    Click "Download" → save summary TSV as data/SAbDab_summary.tsv

    Key columns: pdb, Hchain, Lchain, model, antigen_chain, antigen_type,
                 antigen_name, short_header, date, compound, organism,
                 heavy_subgroup, light_subgroup, light_ctype
    """
    for fname in ["SAbDab_summary.tsv", "sabdab_summary_all.tsv", "sabdab.tsv"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            break
    else:
        log.debug("SAbDab not found in %s (download from opig.stats.ox.ac.uk/webapps/sabdab)", data_dir)
        return []

    try:
        df = pd.read_csv(path, sep="\t", encoding="utf-8-sig", on_bad_lines="skip")
    except Exception as e:
        log.warning("Failed to read SAbDab: %s", e)
        return []

    # SAbDab summary TSV contains structural metadata but NOT sequences.
    # We use heavy_subclass/light_subclass/scfv to get authoritative class
    # labels, then check for optional sequence columns (some versions include them).
    records = []
    seen_pdbs = set()
    for _, row in df.iterrows():
        pdb_id = str(row.get("pdb", "")).strip()

        # Deduplicate by PDB (one entry per antibody structure)
        if pdb_id in seen_pdbs or not pdb_id or pdb_id == "nan":
            continue
        seen_pdbs.add(pdb_id)

        name = str(row.get("compound", row.get("short_header", pdb_id))).strip()

        # Classify from structural metadata
        heavy_sub = str(row.get("heavy_subclass", "")).strip()
        light_sub = str(row.get("light_subclass", "")).strip()
        is_scfv = str(row.get("scfv", "")).strip().lower() == "true"

        has_heavy = heavy_sub and heavy_sub not in ("nan", "", "unknown")
        has_light = light_sub and light_sub not in ("nan", "", "unknown")

        if is_scfv:
            mol_cls = "single_domain"
            n_chains = 1
        elif has_heavy and has_light:
            mol_cls = "canonical_mab"
            n_chains = 2
        elif has_heavy and not has_light:
            mol_cls = "single_domain"
            n_chains = 1
        else:
            continue

        # Check for optional sequence columns (depends on SAbDab version)
        vh_seq = ""
        vl_seq = ""
        for vh_col_name in ("VH_sequence", "Hchain_sequence", "heavy_sequence", "VH"):
            if vh_col_name in df.columns:
                candidate = _clean_seq(str(row.get(vh_col_name, "")))
                if len(candidate) >= 30:
                    vh_seq = candidate
                    break
        for vl_col_name in ("VL_sequence", "Lchain_sequence", "light_sequence", "VL"):
            if vl_col_name in df.columns:
                candidate = _clean_seq(str(row.get(vl_col_name, "")))
                if len(candidate) >= 30:
                    vl_seq = candidate
                    break

        if not vh_seq:
            continue  # Can't compute features without sequence

        combined = vh_seq + vl_seq if vl_seq else vh_seq
        feats = _compute_features(combined)
        if not feats:
            continue

        records.append({
            "name": name[:50] if name else pdb_id,
            "molecule_class": mol_cls,
            "hc_sequence": vh_seq,
            "lc_sequence": vl_seq,
            "n_chains": n_chains,
            "n_unique_chains": n_chains,
            "source": "sabdab",
            **feats,
        })

    if not records:
        log.info("SAbDab: 0 entries (summary TSV has no sequence columns — this is normal for the standard download)")
    else:
        log.info("SAbDab: loaded %d entries", len(records))
    return records


def load_sabdab_sequences(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load pre-extracted SAbDab sequences (from PDB API extraction script).

    The extraction script (scripts/extract_sabdab_sequences.py) fetches FASTA
    sequences from RCSB PDB for all SAbDab entries and saves them as a CSV.
    This loader reads that output.

    Expected file: data/external/raw/sabdab_sequences.csv
    Columns: name, molecule_class, hc_sequence, lc_sequence, source, species
    """
    path = os.path.join(data_dir, "external", "raw", "sabdab_sequences.csv")
    if not os.path.exists(path):
        log.debug("SAbDab sequences not found at %s — run scripts/extract_sabdab_sequences.py first", path)
        return []

    df = pd.read_csv(path, encoding="utf-8")
    records = []
    for _, row in df.iterrows():
        hc = _clean_seq(str(row.get("hc_sequence", "")))
        lc = _clean_seq(str(row.get("lc_sequence", "")))
        mol_cls = str(row.get("molecule_class", "unknown"))

        if len(hc) < 20:
            continue

        combined = hc + lc if lc else hc
        feats = _compute_features(combined)
        if not feats:
            continue

        n_chains = (1 if len(hc) >= 20 else 0) + (1 if len(lc) >= 20 else 0)
        records.append({
            "name": str(row.get("name", "unknown")),
            "molecule_class": mol_cls,
            "hc_sequence": hc,
            "lc_sequence": lc,
            "n_chains": n_chains,
            "n_unique_chains": n_chains,
            "source": "sabdab_pdb",
            **feats,
        })

    log.info("SAbDab sequences (PDB-extracted): loaded %d entries", len(records))
    return records


def load_dramp(data_dir: str, max_samples: int = 5000, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load DRAMP (Data Repository of Antimicrobial Peptides) — ~11,600 peptides.

    Download from: http://dramp.cpu-bioinfor.org/  → Download → xlsx
    Save as data/DRAMP_general.xlsx

    Key columns: DRAMP_ID, Name, Sequence, Sequence_Length, Activity, Family
    """
    for fname in ["DRAMP_general.xlsx", "DRAMP.xlsx", "dramp_general.xlsx"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            break
    else:
        log.debug("DRAMP not found in %s (download from dramp.cpu-bioinfor.org)", data_dir)
        return []

    try:
        df = pd.read_excel(path)
    except Exception as e:
        log.warning("Failed to read DRAMP: %s", e)
        return []

    records = []
    for _, row in df.iterrows():
        seq_raw = str(row.get("Sequence", ""))
        seq = _clean_seq(seq_raw)

        # Filter: must be peptide-length (10–100 aa) with valid residues
        if len(seq) < MIN_SEQUENCE_LENGTH or len(seq) > 100:
            continue

        feats = _compute_features(seq)
        if not feats:
            continue

        name = str(row.get("Name", row.get("DRAMP_ID", "unknown")))[:60]
        records.append({
            "name": name,
            "molecule_class": "peptide",
            "hc_sequence": seq,
            "lc_sequence": "",
            "n_chains": 1,
            "n_unique_chains": 1,
            "source": "dramp",
            **feats,
        })

    # Subsample if too many — keep class balanced with the rest of the dataset
    if len(records) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(records), size=max_samples, replace=False)
        records = [records[i] for i in sorted(indices)]

    log.info("DRAMP: loaded %d peptide entries (from %d total)", len(records), len(df))
    return records


def load_thpd(data_dir: str, max_samples: int = 5000, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load ThPD (Therapeutic Peptide Database) — ~58,500 therapeutic peptides.

    This is the Comprehensive Therapeutic Peptide Dataset (2025).
    Download from: https://doi.org/10.6084/m9.figshare.28691885
    Save as data/ThPD.xlsx

    Key columns: ID, Function, Sequence, Source, Is_natural_peptide
    """
    for fname in ["ThPD.xlsx", "thpd.xlsx", "therapeutic_peptides.xlsx"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            break
    else:
        log.debug("ThPD not found in %s", data_dir)
        return []

    try:
        df = pd.read_excel(path)
    except Exception as e:
        log.warning("Failed to read ThPD: %s", e)
        return []

    records = []
    for _, row in df.iterrows():
        seq_raw = str(row.get("Sequence", ""))
        seq = _clean_seq(seq_raw)

        # Filter: must be peptide-length (10–100 aa) with valid residues
        if len(seq) < MIN_SEQUENCE_LENGTH or len(seq) > 100:
            continue

        feats = _compute_features(seq)
        if not feats:
            continue

        name = str(row.get("ID", "unknown"))
        function_str = str(row.get("Function", ""))[:60]
        records.append({
            "name": f"ThPD_{name}_{function_str}",
            "molecule_class": "peptide",
            "hc_sequence": seq,
            "lc_sequence": "",
            "n_chains": 1,
            "n_unique_chains": 1,
            "source": "thpd",
            **feats,
        })

    # Subsample if too many — keep class balanced
    if len(records) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(records), size=max_samples, replace=False)
        records = [records[i] for i in sorted(indices)]

    log.info("ThPD: loaded %d peptide entries (from %d total)", len(records), len(df))
    return records


def load_imgt_mabdb(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load IMGT/mAb-DB export — ~1,855 therapeutic antibodies.

    Download from: https://www.imgt.org/mAb-DB/index
    Use the query interface → export results → save as data/IMGT_mAbDB.csv

    Also accepts manual CSV with columns: Name, INN, Format, Target, Status, VH, VL
    """
    for fname in ["IMGT_mAbDB.csv", "imgt_mabdb.csv", "IMGT_mAb-DB.csv"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            break
    else:
        log.debug("IMGT/mAb-DB not found in %s (export from imgt.org/mAb-DB/index)", data_dir)
        return []

    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception as e:
        log.warning("Failed to read IMGT/mAb-DB: %s", e)
        return []

    # IMGT format mapping
    _IMGT_FORMAT_MAP = {
        "IgG1": "canonical_mab", "IgG2": "canonical_mab", "IgG4": "canonical_mab",
        "IgG1/IgG4": "canonical_mab", "IgG": "canonical_mab",
        "bispecific": "bispecific", "Bispecific": "bispecific",
        "Fab": "single_domain", "scFv": "single_domain",
        "Nanobody": "single_domain", "VHH": "single_domain",
        "ADC": "adc",
        "Fc fusion": "fc_fusion", "Fc-fusion": "fc_fusion",
        "Fusion protein": "fusion_protein",
    }

    records = []
    for _, row in df.iterrows():
        name = str(row.get("INN", row.get("Name", row.get("name", "unknown"))))
        fmt = str(row.get("Format", row.get("format", "")))

        # Map format
        mol_cls = _IMGT_FORMAT_MAP.get(fmt.strip(), None)
        if mol_cls is None:
            fmt_lower = fmt.lower()
            if "bispecific" in fmt_lower:
                mol_cls = "bispecific"
            elif "adc" in fmt_lower or "conjugat" in fmt_lower:
                mol_cls = "adc"
            elif "nanobody" in fmt_lower or "vhh" in fmt_lower:
                mol_cls = "single_domain"
            elif "fusion" in fmt_lower:
                mol_cls = "fusion_protein"
            elif "fab" in fmt_lower or "scfv" in fmt_lower:
                mol_cls = "single_domain"
            elif "ig" in fmt_lower or "mab" in fmt_lower:
                mol_cls = "canonical_mab"
            else:
                mol_cls = "canonical_mab"  # default for therapeutic mAbs

        vh = _clean_seq(str(row.get("VH", row.get("Heavy", ""))))
        vl = _clean_seq(str(row.get("VL", row.get("Light", ""))))

        if len(vh) < 30 and len(vl) < 30:
            continue

        combined = vh + vl if vl else vh
        feats = _compute_features(combined)
        if not feats:
            continue

        n_chains = (1 if len(vh) >= 20 else 0) + (1 if len(vl) >= 20 else 0)
        records.append({
            "name": name,
            "molecule_class": mol_cls,
            "hc_sequence": vh,
            "lc_sequence": vl,
            "n_chains": n_chains,
            "n_unique_chains": n_chains,
            "source": "imgt_mabdb",
            **feats,
        })

    log.info("IMGT/mAb-DB: loaded %d entries", len(records))
    return records


# ═══════════════════════════════════════════════════════════════════════
#  Main harmonizer
# ═══════════════════════════════════════════════════════════════════════

_OUTPUT_COLUMNS = [
    "name", "molecule_class", "hc_sequence", "lc_sequence",
    "seq_length", "n_chains", "n_unique_chains",
    "pI", "mw_kda", "gravy", "hydrophobicity",
    "deam_sites", "ox_sites", "cysteine_count", "acidic_residues", "basic_residues",
    # Phase 2a new features
    "aromatic_frac", "pro_gly_frac", "cys_frac",
    "deam_density", "ox_density", "charge_ratio", "small_frac", "aliphatic_idx",
    # HC/LC chain-level features
    "hc_frac", "has_lc", "hc_len_norm", "lc_len_norm",
    "source",
]


def load_curated_csv(data_dir: str, filename: str) -> List[Dict[str, Any]]:
    """Load a curated CSV from data/external/curated/ directory.

    These are hand-curated or script-generated training entries for
    rare classes that lack sufficient real data.
    """
    path = os.path.join(data_dir, "external", "curated", filename)
    if not os.path.exists(path):
        log.debug("Curated file not found: %s", path)
        return []

    df = pd.read_csv(path, encoding="utf-8")
    records = df.to_dict("records")
    log.info("Curated data (%s): loaded %d entries", filename, len(records))
    return records


def harmonize(data_dir: str = "data", output_path: str = None) -> pd.DataFrame:
    """
    Build unified training dataset from all sources.

    Returns DataFrame and optionally writes to CSV.
    """
    all_records = []
    # Core sources (always available)
    all_records.extend(load_jain137(data_dir))
    all_records.extend(load_therasabdab(data_dir))
    # External databases (optional — loaded if files exist)
    all_records.extend(load_covabdab(data_dir))
    all_records.extend(load_sabdab(data_dir))
    all_records.extend(load_sabdab_sequences(data_dir))
    all_records.extend(load_imgt_mabdb(data_dir))
    # Peptide databases (optional — dramatically improve peptide classification)
    all_records.extend(load_dramp(data_dir))
    all_records.extend(load_thpd(data_dir))
    # Curated rare class data
    all_records.extend(load_curated_csv(data_dir, "fc_fusion_curated.csv"))
    all_records.extend(load_curated_csv(data_dir, "adc_curated.csv"))
    all_records.extend(load_curated_csv(data_dir, "fusion_protein_curated.csv"))
    # Synthetic class-balancing data
    all_records.extend(generate_synthetic_peptides())
    all_records.extend(generate_synthetic_nanobodies())
    all_records.extend(generate_synthetic_scaffolds())

    df = pd.DataFrame(all_records)

    # Drop unknowns for classification training
    df = df[df["molecule_class"] != "unknown"].copy()

    # ── Compute HC/LC chain-level features for every record ──────────
    chain_feats = df.apply(
        lambda r: _compute_chain_features(
            str(r.get("hc_sequence", "")) if pd.notna(r.get("hc_sequence")) else "",
            str(r.get("lc_sequence", "")) if pd.notna(r.get("lc_sequence")) else "",
        ),
        axis=1, result_type="expand",
    )
    for col in chain_feats.columns:
        df[col] = chain_feats[col]

    # Ensure all columns present
    for col in _OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if col in (
                "aromatic_frac", "pro_gly_frac", "cys_frac", "deam_density",
                "ox_density", "charge_ratio", "small_frac", "aliphatic_idx",
                "hc_frac", "has_lc", "hc_len_norm", "lc_len_norm",
            ) else ""

    df = df[_OUTPUT_COLUMNS].copy()
    df = df.reset_index(drop=True)

    log.info("Harmonized dataset: %d rows, %d classes",
             len(df), df["molecule_class"].nunique())
    log.info("Class distribution:\n%s", df["molecule_class"].value_counts().to_string())

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        log.info("Written to %s", output_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="Harmonize training data")
    parser.add_argument("--output", default="data/training/classifier_data.csv")
    args = parser.parse_args()
    harmonize(data_dir="data", output_path=args.output)
