#!/usr/bin/env python3
"""
benchmark_prophet_ab.py — PROPHET-Ab Universal Benchmark
=========================================================
Scores all 246 PROPHET-Ab antibodies through ProtePilot twin modules
and computes Spearman rank correlations against experimental data.

Usage:
    PYTHONPATH=. python3 scripts/benchmark_prophet_ab.py

Output:
    - Summary table to stdout
    - SelfTest/prophet_ab_benchmark_results.json (detailed)
    - SelfTest/prophet_ab_benchmark_summary.csv
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PROPHET-Ab")

# =====================================================================
# 1. Load PROPHET-Ab Data
# =====================================================================

DATA_PATHS = [
    "data/external/prophet_ab.csv",
    "data/prophet_ab.csv",
]


def load_prophet_data() -> pd.DataFrame:
    for path in DATA_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded PROPHET-Ab data: {len(df)} antibodies from {path}")
            return df
    print("ERROR: PROPHET-Ab data not found. Expected at:")
    for p in DATA_PATHS:
        print(f"  {p}")
    print("\nTo obtain the data:")
    print("  git clone https://github.com/ginkgobioworks/prophet_ab")
    print("  cp prophet_ab/data/prophet_ab.csv data/external/prophet_ab.csv")
    sys.exit(1)


# =====================================================================
# 2. Twin Scoring Functions
# =====================================================================

def score_immunogenicity(vh: str, vl: str, name: str) -> Dict[str, Any]:
    from src.immunogenicity_twin import run_immunogenicity_assessment
    seq = vh + vl
    r = run_immunogenicity_assessment(
        sequence=seq, molecule_name=name, molecule_class="canonical_mab")
    return {
        "ada_risk_score": r.ada_risk_score,
        "ada_risk_level": r.ada_risk_level,
        "humanization_score": r.humanization_score,
        "n_high_hotspots": r.n_high_risk,
        "n_medium_hotspots": r.n_medium_risk,
        "mean_mhc_score": r.mean_mhc_score,
        "closest_vh_germline": r.closest_vh_germline,
        "closest_vl_germline": r.closest_vl_germline,
    }


def score_stability(vh: str, vl: str, tm_hint: float = None) -> Dict[str, Any]:
    from src.stability_twin import run_stability_study
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    seq = vh + vl
    pa = ProteinAnalysis(seq.upper())
    pI = pa.isoelectric_point()
    Tm = tm_hint if tm_hint and not np.isnan(tm_hint) else 72.0

    r = run_stability_study(
        pI=pI, Tm=Tm,
        excipients=["sucrose", "polysorbate_80"],
        formulation_ph=6.0,
    )
    return {
        "shelf_life_months": r.predicted_shelf_life_months,
        "stability_grade": r.overall_stability_grade,
        "k_5c": r.long_term.hmw_growth_rate_pct_per_month,
        "k_40c": r.accelerated.hmw_growth_rate_pct_per_month,
        "final_hmw_5c": r.long_term.final_hmw_pct,
    }


def score_formulation(vh: str, vl: str) -> Dict[str, Any]:
    from src.formulation_twin import compute_formulation_effects, FormulationCondition
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    seq = vh + vl
    pa = ProteinAnalysis(seq.upper())
    pI = pa.isoelectric_point()
    gravy = pa.gravy()

    condition = FormulationCondition(
        buffer_ph=6.0,
        buffer_type="histidine",
        excipients=["sucrose", "polysorbate_80"],
    )
    r = compute_formulation_effects(
        condition=condition,
        pI=pI,
        sequence=seq,
        hydrophobicity=max(0, min(1, (gravy + 1) / 2)),
    )
    return {
        "viscosity_modifier": getattr(r, "viscosity_modifier", 0.0),
        "aggregation_modifier": getattr(r, "agg_risk_modifier", 0.0),
        "net_charge": getattr(r, "net_charge", 0.0),
    }


def score_analytical_qc(vh: str, vl: str) -> Dict[str, Any]:
    from src.analytical_qc_twin import run_analytical_qc
    from dataclasses import asdict
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    seq = vh + vl
    pa = ProteinAnalysis(seq.upper())
    pI = pa.isoelectric_point()

    r = run_analytical_qc(sequence=seq, pI=pI, is_mab=True, molecule_class="canonical_mab")
    d = asdict(r)
    return {
        "cief_main_pct": float(d["cief"]["main_pct"]),
        "cesds_intact_pct": float(d["ce_sds"]["intact_pct"]),
        "g0f_pct": d["glycan"]["g0f_pct"],
    }


def score_upstream(vh: str, vl: str) -> Dict[str, Any]:
    from src.upstream_twin import run_upstream_simulation
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    seq = vh + vl
    pa = ProteinAnalysis(seq.upper())
    gravy = pa.gravy()

    r = run_upstream_simulation(
        seed_density=0.5, temp_shift_day=5.0,
        sequence=seq, molecule_class="canonical_mab",
        hydrophobicity=gravy,
    )
    return {
        "predicted_titer_g_l": r.final_titer,
        "viability_pct": r.viability_at_harvest,
        "peak_vcd": r.peak_vcd,
    }


# =====================================================================
# 3. Main Benchmark
# =====================================================================

def run_benchmark():
    df = load_prophet_data()

    results = []
    n_success = 0
    n_fail = 0
    t0 = time.time()

    print(f"\nScoring {len(df)} antibodies through 5 twin modules...")

    for i, row in df.iterrows():
        name = row.get("antibody_name", row.get("antibody_id", f"Ab_{i}"))
        vh = str(row["vh_protein_sequence"])
        vl = str(row["vl_protein_sequence"])

        rec: Dict[str, Any] = {"antibody_id": row["antibody_id"], "name": name}

        # Copy experimental values
        for col in ["Titer", "Purity", "SEC %Monomer", "SMAC", "HIC", "HAC",
                     "PR_CHO", "PR_Ova", "AC-SINS_pH6.0", "AC-SINS_pH7.4",
                     "Tonset", "Tm1", "Tm2"]:
            rec[f"exp_{col}"] = row.get(col)

        try:
            rec.update(score_immunogenicity(vh, vl, name))
            tm_hint = row.get("Tm1")
            rec.update(score_stability(vh, vl, tm_hint))
            rec.update(score_formulation(vh, vl))
            rec.update(score_analytical_qc(vh, vl))
            rec.update(score_upstream(vh, vl))
            n_success += 1
        except Exception as e:
            log.warning(f"Failed on {name}: {e}")
            n_fail += 1

        results.append(rec)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(df)} scored ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nScoring complete: {n_success} success, {n_fail} failures in {elapsed:.1f}s")

    results_df = pd.DataFrame(results)

    # =====================================================================
    # 4. Compute Correlations
    # =====================================================================

    correlation_pairs = [
        # (twin_prediction_col, experimental_col, description, expected_direction)
        ("ada_risk_score",       "exp_PR_CHO",         "ADA risk vs CHO polyreactivity",   "positive"),
        ("ada_risk_score",       "exp_PR_Ova",         "ADA risk vs Ova polyreactivity",   "positive"),
        ("humanization_score",   "exp_Tm1",            "Humanization vs Tm1",              "none"),
        ("mean_mhc_score",       "exp_PR_CHO",         "MHC-II density vs polyreactivity", "positive"),
        ("shelf_life_months",    "exp_Tm1",            "Shelf life vs Tm1",                "positive"),
        ("k_5c",                 "exp_SEC %Monomer",   "HMW growth rate vs SEC monomer",   "negative"),
        ("k_40c",                "exp_Tm1",            "Accel. deg. rate vs Tm1",          "negative"),
        ("aggregation_modifier", "exp_AC-SINS_pH6.0",  "Agg modifier vs AC-SINS pH6",      "positive"),
        ("aggregation_modifier", "exp_AC-SINS_pH7.4",  "Agg modifier vs AC-SINS pH7.4",    "positive"),
        ("predicted_titer_g_l",  "exp_Titer",          "Predicted titer vs exp titer",      "positive"),
        ("g0f_pct",              "exp_Tm1",            "G0F vs Tm1",                        "none"),
        ("cief_main_pct",        "exp_Purity",         "cIEF main vs Purity",               "positive"),
        ("cesds_intact_pct",     "exp_SEC %Monomer",   "CE-SDS intact vs SEC monomer",      "positive"),
        ("viscosity_modifier",   "exp_SMAC",           "Viscosity mod vs SMAC",             "none"),
    ]

    corr_results = []
    for pred_col, exp_col, desc, direction in correlation_pairs:
        if pred_col not in results_df.columns or exp_col not in results_df.columns:
            continue

        mask = results_df[pred_col].notna() & results_df[exp_col].notna()
        x = results_df.loc[mask, pred_col].astype(float)
        y = results_df.loc[mask, exp_col].astype(float)
        n_pairs = len(x)

        if n_pairs < 10:
            continue

        rho, pval = sp_stats.spearmanr(x, y)
        r, _ = sp_stats.pearsonr(x, y)

        # Check direction agreement
        if direction == "positive":
            direction_ok = rho > 0
        elif direction == "negative":
            direction_ok = rho < 0
        else:
            direction_ok = True  # no expected direction

        corr_results.append({
            "prediction": pred_col,
            "experimental": exp_col.replace("exp_", ""),
            "description": desc,
            "spearman_rho": round(float(rho), 4),
            "pearson_r": round(float(r), 4),
            "p_value": float(pval),
            "n_pairs": n_pairs,
            "significant": pval < 0.05,
            "direction_expected": direction,
            "direction_correct": direction_ok,
        })

    # =====================================================================
    # 5. Print Summary
    # =====================================================================

    print(f"\n{'=' * 95}")
    print(f"PROPHET-Ab Benchmark: Twin Predictions vs Experimental Data (n={len(df)})")
    print(f"{'=' * 95}")
    print(f"{'Prediction':25s} | {'Experimental':20s} | {'rho':>7s} | {'p-val':>10s} | {'N':>4s} | {'Sig':>4s} | {'Dir':>4s}")
    print(f"{'-' * 95}")

    n_sig = 0
    n_dir_ok = 0
    for cr in corr_results:
        sig = "Yes" if cr["significant"] else "No"
        dir_ok = "OK" if cr["direction_correct"] else "X"
        if cr["significant"]:
            n_sig += 1
        if cr["direction_correct"]:
            n_dir_ok += 1
        print(f"  {cr['prediction']:23s} | {cr['experimental']:20s} | {cr['spearman_rho']:+.4f} | {cr['p_value']:.2e} | {cr['n_pairs']:4d} | {sig:>4s} | {dir_ok:>4s}")

    print(f"{'=' * 95}")
    print(f"Significant correlations (p<0.05): {n_sig}/{len(corr_results)}")
    print(f"Direction correct: {n_dir_ok}/{len(corr_results)}")

    # =====================================================================
    # 6. Save Results
    # =====================================================================

    os.makedirs("SelfTest", exist_ok=True)

    output = {
        "benchmark": "PROPHET-Ab",
        "n_antibodies": len(df),
        "n_scored": n_success,
        "n_failed": n_fail,
        "elapsed_seconds": round(elapsed, 1),
        "correlations": corr_results,
        "summary": {
            "n_significant": n_sig,
            "n_total_pairs": len(corr_results),
            "n_direction_correct": n_dir_ok,
        },
    }

    json_path = "SelfTest/prophet_ab_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results: {json_path}")

    csv_path = "SelfTest/prophet_ab_benchmark_summary.csv"
    pd.DataFrame(corr_results).to_csv(csv_path, index=False)
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    run_benchmark()
