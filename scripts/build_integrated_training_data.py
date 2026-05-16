"""
build_integrated_training_data.py
=================================
Merge all labeled data sources into unified_training_data.csv with REAL
experimental labels mapped to the 8 unified task columns.

Sources:
  1. Jain-137 (137 rows, VH+VL, 13 experimental targets)
  2. merged_wetlab_training.csv (267 rows, VH+VL, 7 targets)
  3. prophet_ab.csv (246 rows, VH+VL+full HC/LC, 14 targets)
  4. garbinski2023_tm1.csv (86 rows, VH+VL, Tm1 only)
  5. tresanco2023nbthermo_tm.csv (672 rows, nanobody VH only, Tm)
  6. jain2024assessment_SEC.csv (43 rows, VH+VL, SEC % Monomer → aggregation_risk)
  7. shanehsazzadeh2023unlocking_SEC.csv (13 rows, VH+VL, SEC % Monomer → aggregation_risk)
  8. benchmark_sequences.json (12 molecules, full HC+LC or fusion chains)

Deduplicates by VH sequence. Maps experimental values to 8 task columns.
Adds full-length rows for benchmark molecules (seq_type="full_length").
"""

import json
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from training.features import compute_sequence_features


# ── Normalization ranges (from published data distributions) ──
# tm: keep raw °C (model learns raw scale)
# aggregation_risk: AC-SINS Δλmax nm → [0, 1], cap at 20 nm
# stability: slope for accel stability → invert (lower slope = more stable)
# hydrophobicity: HIC RT → [0, 1], range 8.5-13.4 min (Jain-137 range)
# viscosity_risk: PSR 0-1 direct
# potency: titer normalized by max observed

def _norm_acsins(val):
    """AC-SINS Δλmax (nm) → aggregation_risk [0, 1]. Higher = riskier."""
    if pd.isna(val):
        return np.nan
    return min(1.0, max(0.0, float(val) / 20.0))

def _norm_stability(val):
    """Slope for accelerated stability → stability [0, 1]. Higher = more stable."""
    if pd.isna(val):
        return np.nan
    # Jain-137 range: 0.00-0.26. Invert: 0 slope = 1.0 stability
    return max(0.0, min(1.0, 1.0 - float(val) / 0.30))

def _norm_hydro(val):
    """HIC retention time (min) → hydrophobicity [0, 1]."""
    if pd.isna(val):
        return np.nan
    # Jain-137 range: ~8.5-13.4 min
    return max(0.0, min(1.0, (float(val) - 6.0) / 10.0))

def _norm_psr(val):
    """PSR SMP Score already 0-1."""
    if pd.isna(val):
        return np.nan
    return max(0.0, min(1.0, float(val)))

def _norm_titer(val, max_titer=300.0):
    """Titer (mg/L) → potency proxy [0, 1]."""
    if pd.isna(val):
        return np.nan
    return max(0.0, min(1.0, float(val) / max_titer))


def _compute_biophys(seq):
    """Compute 7 biophysical features from sequence."""
    seq = seq.upper()
    n = max(len(seq), 1)
    acidic = seq.count("D") + seq.count("E")
    basic = seq.count("K") + seq.count("R") + seq.count("H")
    net = basic - acidic
    pI = max(4.0, min(12.0, 7.0 + net * 0.05))
    mw_kda = n * 0.110
    deam = sum(1 for i in range(n - 1) if seq[i] == "N" and seq[i + 1] in "GSTD")
    ox = seq.count("M") + seq.count("W")
    kd = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
          "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
          "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
    gravy = sum(kd.get(aa, 0) for aa in seq) / n
    hydro = max(0.0, min(1.0, (gravy + 2.0) / 4.0))
    return {
        "pI": round(pI, 2),
        "MW_kDa": round(mw_kda, 1),
        "deam_sites": deam,
        "ox_sites": ox,
        "acidic_residues": acidic,
        "basic_residues": basic,
        "hydrophobicity_gravy": round(hydro, 4),
    }


def _clean(s):
    if not s or pd.isna(s) or str(s).strip().lower() in ("na", "nan", "nd", ""):
        return ""
    return "".join(c for c in str(s).strip().upper() if c in "ACDEFGHIKLMNPQRSTVWY")


def load_jain137():
    """Load Jain-137 with experimental labels."""
    path = os.path.join(PROJECT_ROOT, "data", "Jain137_Cleaned_Training_Data.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    records = []
    for _, row in df.iterrows():
        vh = _clean(row.get("VH"))
        vl = _clean(row.get("VL"))
        if len(vh) < 50:
            continue
        combined = vh + vl
        bp = _compute_biophys(combined)
        records.append({
            "hc_sequence": vh,
            "lc_sequence": vl,
            **bp,
            "ka": np.nan,  # No chromatographic data in Jain-137
            "nu": np.nan,
            "tm": row.get("Fab Tm by DSF (°C)", np.nan),
            "aggregation_risk": _norm_acsins(row.get(
                "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) Δλmax (nm) Average",
                row.get("Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average", np.nan))),
            "stability": _norm_stability(row.get("Slope for Accelerated Stability", np.nan)),
            "viscosity_risk": _norm_psr(row.get("Poly-Specificity Reagent (PSR) SMP Score (0-1)",
                                                  row.get("PSR SMP Score (0-1)", np.nan))),
            "hydrophobicity": _norm_hydro(row.get("HIC Retention Time (Min)a", np.nan)),
            "potency": _norm_titer(row.get("HEK Titer (mg/L)", np.nan)),
            "_source": "jain137",
            "_name": str(row.get("Name", "unknown")),
        })
    print(f"  Jain-137: {len(records)} rows loaded")
    return records


def load_merged_wetlab():
    """Load merged wetlab training data."""
    path = os.path.join(PROJECT_ROOT, "data", "merged_wetlab_training.csv")
    if not os.path.exists(path):
        print("  merged_wetlab: NOT FOUND")
        return []
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        vh = _clean(row.get("vh"))
        vl = _clean(row.get("vl"))
        if len(vh) < 50:
            continue
        combined = vh + vl
        bp = _compute_biophys(combined)
        records.append({
            "hc_sequence": vh,
            "lc_sequence": vl,
            **bp,
            "ka": np.nan,
            "nu": np.nan,
            "tm": row.get("fab_tm", np.nan),
            "aggregation_risk": _norm_acsins(row.get("acsins", np.nan)),
            "stability": _norm_stability(row.get("stability_slope", np.nan)),
            "viscosity_risk": _norm_psr(row.get("psr", np.nan)),
            "hydrophobicity": _norm_hydro(row.get("hic_rt", np.nan)),
            "potency": _norm_titer(row.get("titer", np.nan)),
            "_source": "merged_wetlab",
            "_name": str(row.get("sources", "unknown")),
        })
    print(f"  merged_wetlab: {len(records)} rows loaded")
    return records


def load_prophet_ab():
    """Load PROPHET-Ab with experimental labels."""
    path = os.path.join(PROJECT_ROOT, "data", "external", "prophet_ab.csv")
    if not os.path.exists(path):
        print("  prophet_ab: NOT FOUND")
        return []
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        # Use VH protein sequence (variable region)
        vh = _clean(row.get("vh_protein_sequence"))
        vl = _clean(row.get("vl_protein_sequence"))
        if len(vh) < 50:
            continue
        combined = vh + vl
        bp = _compute_biophys(combined)
        # AC-SINS at pH 7.4 is the standard measurement
        acsins = row.get("AC-SINS_pH7.4", np.nan)
        records.append({
            "hc_sequence": vh,
            "lc_sequence": vl,
            **bp,
            "ka": np.nan,
            "nu": np.nan,
            "tm": row.get("Tm1", np.nan),
            "aggregation_risk": _norm_acsins(acsins),
            "stability": np.nan,  # No accelerated stability slope in PROPHET-Ab
            "viscosity_risk": np.nan,  # No PSR in PROPHET-Ab
            "hydrophobicity": _norm_hydro(row.get("HIC", np.nan)),
            "potency": _norm_titer(row.get("Titer", np.nan)),
            "_source": "prophet_ab",
            "_name": str(row.get("antibody_name", "unknown")),
        })
    print(f"  prophet_ab: {len(records)} rows loaded")
    return records


def load_garbinski():
    """Load Garbinski 2023 Tm data."""
    path = os.path.join(PROJECT_ROOT, "data", "flab", "garbinski2023_tm1.csv")
    if not os.path.exists(path):
        print("  garbinski: NOT FOUND")
        return []
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        vh = _clean(row.get("heavy"))
        vl = _clean(row.get("light"))
        if len(vh) < 50:
            continue
        combined = vh + vl
        bp = _compute_biophys(combined)
        records.append({
            "hc_sequence": vh,
            "lc_sequence": vl,
            **bp,
            "ka": np.nan,
            "nu": np.nan,
            "tm": row.get("Tm1 (nanoDSF)", np.nan),
            "aggregation_risk": np.nan,
            "stability": np.nan,
            "viscosity_risk": np.nan,
            "hydrophobicity": np.nan,
            "potency": np.nan,
            "_source": "garbinski2023",
            "_name": f"garbinski_{_}",
        })
    print(f"  garbinski: {len(records)} rows loaded")
    return records


def load_tresanco_nanobody_tm():
    """Load Tresanco 2023 nanobody Tm dataset (672 entries)."""
    path = os.path.join(PROJECT_ROOT, "data", "flab", "tresanco2023nbthermo_tm.csv")
    if not os.path.exists(path):
        print("  tresanco_nanobody: NOT FOUND")
        return []
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        vh = _clean(row.get("heavy"))
        if len(vh) < 50:
            continue
        bp = _compute_biophys(vh)  # nanobodies are single chain
        records.append({
            "hc_sequence": vh,
            "lc_sequence": "",
            **bp,
            "ka": np.nan,
            "nu": np.nan,
            "tm": row.get("Tm measurement", np.nan),
            "aggregation_risk": np.nan,
            "stability": np.nan,
            "viscosity_risk": np.nan,
            "hydrophobicity": np.nan,
            "potency": np.nan,
            "_source": "tresanco2023",
            "_name": f"tresanco_{_}",
        })
    print(f"  tresanco_nanobody: {len(records)} rows loaded")
    return records


def _sec_monomer_to_agg_risk(val):
    """SEC % Monomer → aggregation_risk [0, 1]. Lower monomer = higher risk."""
    if pd.isna(val):
        return np.nan
    risk = (100.0 - float(val)) / 10.0
    return min(1.0, max(0.0, risk))


def load_jain2024_sec():
    """Load Jain 2024 SEC monomer data (43 rows)."""
    path = os.path.join(PROJECT_ROOT, "data", "flab", "jain2024assessment_SEC.csv")
    if not os.path.exists(path):
        print("  jain2024_sec: NOT FOUND")
        return []
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        vh = _clean(row.get("heavy"))
        vl = _clean(row.get("light"))
        if len(vh) < 50:
            continue
        combined = vh + vl if vl else vh
        bp = _compute_biophys(combined)
        records.append({
            "hc_sequence": vh,
            "lc_sequence": vl,
            **bp,
            "ka": np.nan,
            "nu": np.nan,
            "tm": np.nan,
            "aggregation_risk": _sec_monomer_to_agg_risk(row.get("SEC % Monomer", np.nan)),
            "stability": np.nan,
            "viscosity_risk": np.nan,
            "hydrophobicity": np.nan,
            "potency": np.nan,
            "_source": "jain2024_sec",
            "_name": f"jain2024_{_}",
        })
    print(f"  jain2024_sec: {len(records)} rows loaded")
    return records


def load_shanehsazzadeh2023_sec():
    """Load Shanehsazzadeh 2023 SEC monomer data (13 rows)."""
    path = os.path.join(PROJECT_ROOT, "data", "flab", "shanehsazzadeh2023unlocking_SEC.csv")
    if not os.path.exists(path):
        print("  shanehsazzadeh2023_sec: NOT FOUND")
        return []
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        vh = _clean(row.get("heavy"))
        vl = _clean(row.get("light"))
        if len(vh) < 50:
            continue
        combined = vh + vl if vl else vh
        bp = _compute_biophys(combined)
        sec_col = "SEC - Main (%)" if "SEC - Main (%)" in df.columns else "SEC % Monomer"
        records.append({
            "hc_sequence": vh,
            "lc_sequence": vl,
            **bp,
            "ka": np.nan,
            "nu": np.nan,
            "tm": np.nan,
            "aggregation_risk": _sec_monomer_to_agg_risk(row.get(sec_col, np.nan)),
            "stability": np.nan,
            "viscosity_risk": np.nan,
            "hydrophobicity": np.nan,
            "potency": np.nan,
            "_source": "shanehsazzadeh2023_sec",
            "_name": f"shanehsazzadeh_{_}",
        })
    print(f"  shanehsazzadeh2023_sec: {len(records)} rows loaded")
    return records


def load_benchmark_full_length():
    """Load full-length HC+LC sequences from benchmark_sequences.json.

    Returns a dict mapping molecule name (lowercase) to:
      {"hc": full_heavy_chain, "lc": full_light_chain, "molecule_type": str}
    For Fc-fusions and nanobodies, uses fusion_chain/vhh_chain instead.
    For bispecifics (emicizumab), uses heavy_chain_A as primary HC.
    """
    path = os.path.join(PROJECT_ROOT, "data", "reference", "benchmark_sequences.json")
    if not os.path.exists(path):
        print("  benchmark_sequences.json: NOT FOUND")
        return {}

    with open(path) as f:
        data = json.load(f)

    molecules = {}
    for name, info in data.items():
        if name.startswith("_"):
            continue
        mol_type = info.get("molecule_type", "mAb")

        if mol_type in ("mAb", "adc"):
            hc = _clean(info.get("heavy_chain", ""))
            lc = _clean(info.get("light_chain", ""))
            if len(hc) > 200 and len(lc) > 100:
                molecules[name.lower()] = {"hc": hc, "lc": lc, "molecule_type": mol_type, "name": name}
        elif mol_type == "bispecific":
            # Use heavy_chain_A as primary HC
            hc = _clean(info.get("heavy_chain_A", ""))
            lc = _clean(info.get("light_chain", ""))
            if len(hc) > 200 and len(lc) > 100:
                molecules[name.lower()] = {"hc": hc, "lc": lc, "molecule_type": mol_type, "name": name}
        elif mol_type == "fc_fusion":
            fc = _clean(info.get("fusion_chain", ""))
            if len(fc) > 200:
                molecules[name.lower()] = {"hc": fc, "lc": "", "molecule_type": mol_type, "name": name}
        elif mol_type == "nanobody":
            vhh = _clean(info.get("vhh_chain", ""))
            if len(vhh) > 50:
                molecules[name.lower()] = {"hc": vhh, "lc": "", "molecule_type": mol_type, "name": name}

    print(f"  benchmark_sequences: {len(molecules)} full-length molecules loaded")
    return molecules


def _match_benchmark_to_csv(df, benchmark_mols):
    """Match benchmark molecules to existing CSV rows by first 50 aa of VH.

    Returns list of (molecule_name, matched_row_index_or_None).
    """
    # Build a lookup: first 50 aa of hc_sequence -> row index
    vh_prefix_to_idx = {}
    for idx, row in df.iterrows():
        hc = str(row["hc_sequence"]).upper()
        if len(hc) >= 50:
            vh_prefix_to_idx[hc[:50]] = idx

    matches = []
    for mol_key, mol_info in benchmark_mols.items():
        full_hc = mol_info["hc"]
        prefix = full_hc[:50]
        matched_idx = vh_prefix_to_idx.get(prefix, None)
        matches.append((mol_key, mol_info, matched_idx))

    return matches


def main():
    print("=" * 60)
    print("Building integrated unified_training_data.csv")
    print("=" * 60)

    # Load all sources
    all_records = []
    all_records.extend(load_jain137())
    all_records.extend(load_merged_wetlab())
    all_records.extend(load_prophet_ab())
    all_records.extend(load_garbinski())
    all_records.extend(load_tresanco_nanobody_tm())
    all_records.extend(load_jain2024_sec())
    all_records.extend(load_shanehsazzadeh2023_sec())

    df = pd.DataFrame(all_records)
    print(f"\n  Total before dedup: {len(df)} rows")

    # Deduplicate by VH sequence (keep first occurrence = prioritize Jain-137)
    df = df.drop_duplicates(subset=["hc_sequence"], keep="first")
    print(f"  After VH dedup: {len(df)} rows")

    # Report source distribution
    print(f"\n  Source distribution:")
    for src, count in df["_source"].value_counts().items():
        print(f"    {src}: {count}")

    # Report label coverage
    task_cols = ["ka", "nu", "tm", "aggregation_risk", "stability",
                 "viscosity_risk", "hydrophobicity", "potency"]
    print(f"\n  Label coverage:")
    for col in task_cols:
        n_valid = df[col].notna().sum()
        print(f"    {col}: {n_valid}/{len(df)} ({n_valid/len(df)*100:.0f}%)")

    # ── Add seq_type column: all existing rows are variable_only ──
    df["seq_type"] = "variable_only"

    # ── Integrate full-length benchmark sequences ──
    benchmark_mols = load_benchmark_full_length()
    matches = _match_benchmark_to_csv(df, benchmark_mols)

    new_rows = []
    added_names = []
    updated_names = []
    unmatched_names = []

    for mol_key, mol_info, matched_idx in matches:
        full_hc = mol_info["hc"]
        full_lc = mol_info["lc"]
        display_name = mol_info["name"]

        # Compute biophysical features for full-length sequence
        combined = full_hc + full_lc if full_lc else full_hc
        bp = _compute_biophys(combined)

        if matched_idx is not None:
            # Copy task labels from the matched variable-only row
            matched_row = df.loc[matched_idx]
            new_row = {
                "hc_sequence": full_hc,
                "lc_sequence": full_lc,
                **bp,
                "ka": matched_row.get("ka", np.nan),
                "nu": matched_row.get("nu", np.nan),
                "tm": matched_row.get("tm", np.nan),
                "aggregation_risk": matched_row.get("aggregation_risk", np.nan),
                "stability": matched_row.get("stability", np.nan),
                "viscosity_risk": matched_row.get("viscosity_risk", np.nan),
                "hydrophobicity": matched_row.get("hydrophobicity", np.nan),
                "potency": matched_row.get("potency", np.nan),
                "_source": "benchmark_full",
                "_name": display_name,
                "seq_type": "full_length",
            }
            new_rows.append(new_row)
            updated_names.append(display_name)
        else:
            # No match found — add with NaN labels
            new_row = {
                "hc_sequence": full_hc,
                "lc_sequence": full_lc,
                **bp,
                "ka": np.nan,
                "nu": np.nan,
                "tm": np.nan,
                "aggregation_risk": np.nan,
                "stability": np.nan,
                "viscosity_risk": np.nan,
                "hydrophobicity": np.nan,
                "potency": np.nan,
                "_source": "benchmark_full",
                "_name": display_name,
                "seq_type": "full_length",
            }
            new_rows.append(new_row)
            unmatched_names.append(display_name)

        added_names.append(display_name)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)

    print(f"\n  Benchmark full-length integration:")
    print(f"    Total added: {len(added_names)} full-length rows")
    if updated_names:
        print(f"    Matched (labels copied): {', '.join(updated_names)}")
    if unmatched_names:
        print(f"    Unmatched (NaN labels):  {', '.join(unmatched_names)}")

    # Final seq_type distribution
    print(f"\n  Sequence type distribution:")
    for st, count in df["seq_type"].value_counts().items():
        print(f"    {st}: {count}")

    # Drop internal columns
    df = df.drop(columns=["_source", "_name"])

    # Reorder columns to match expected schema
    output_cols = [
        "hc_sequence", "lc_sequence",
        "pI", "MW_kDa", "deam_sites", "ox_sites",
        "acidic_residues", "basic_residues", "hydrophobicity_gravy",
        "ka", "nu", "tm", "aggregation_risk",
        "stability", "viscosity_risk", "hydrophobicity", "potency",
        "seq_type",
    ]
    df = df[output_cols].reset_index(drop=True)

    # Save
    output_path = os.path.join(PROJECT_ROOT, "data", "unified_training_data.csv")
    # Backup old file
    if os.path.exists(output_path):
        backup = output_path + ".bak"
        os.rename(output_path, backup)
        print(f"\n  Backed up old file to {backup}")

    df.to_csv(output_path, index=False)
    print(f"  Written: {output_path} ({len(df)} rows)")
    print(f"  ({df['seq_type'].value_counts().to_dict()})")
    print("=" * 60)


if __name__ == "__main__":
    main()
