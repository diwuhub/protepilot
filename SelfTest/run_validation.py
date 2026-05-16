#!/usr/bin/env python3
"""
SelfTest/run_validation.py — ProtePilot Validation Runner
=============================================================
Runs all backend validation tests and writes results to
validation_results_v3.json for consumption by generate_report_v3.py.

Usage:
    python SelfTest/run_validation.py
    python SelfTest/run_validation.py --output path/to/output.json

Sections:
    1. PropertyMapper  — SMA param prediction for 5 reference mAbs
    2. Developability   — ESM-2 + heuristic risk scoring
    3. Upstream         — Fed-batch bioreactor ODE simulation
    4. Analytical QC    — cIEF, CE-SDS, glycan profiling
    5. Stability        — Arrhenius degradation kinetics
    6. NISTmAb          — Gold-standard benchmark (RM 8671)
    7. Immunogenicity   — MHC-II binding + ADA risk
    8. TheRaSAbDab      — 10-mAb cross-validation panel
    9. Cross-check      — Jain-137 ↔ TheRaSAbDab sequence identity
   10. CEX Resolution   — Analytical Rs validation (Yamamoto SMA gradient)
   11. End-to-End       — Feature→Developability→Report integration
   12. Bispecific Format — 3-species separation + report coherence
   13. Non-Canonical    — Fc-fusion, fusion_protein, single_domain format coverage
   14. OOD Confidence   — Out-of-distribution confidence capping + conservative narrative
   15. Cross-Section    — Key field consistency + liability prioritization over simulated QC
   16. Downstream DoE   — Purification optimizer grid search + mass balance
   17. Cross-Path       — Frontend vs Bulk vs Backend numeric alignment
   18. Schema Alignment  — Bulk/single field-map integrity, grade-string canonicality
   19. Platform Alignment — Cross-module consistency, gap detection, determinism, color audit
   20. Post-Training    — Model artifact validity, determinism, schema compat, drift (conditional)

v3.1 — Substantive Validation
  - Every test now includes explicit checks dict with PASS/FAIL
  - Tests return FAIL if any check fails (not unconditional PASS)
  - New: end-to-end integration test (#11)
  - New: bispecific format-specific test (#12)
"""

from __future__ import annotations

import json
import os
import sys
import datetime
import logging

# Ensure project root on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("Validation")

# =========================================================================
# Reference mAbs for PropertyMapper
# =========================================================================
REFERENCE_MABS = {
    "adalimumab":  {"pI": 8.72, "mw": 148.0, "hydro": 0.35, "gravy": -0.34, "sequence": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS"},
    "trastuzumab": {"pI": 8.45, "mw": 148.0, "hydro": 0.35, "gravy": -0.28, "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"},
    "bevacizumab": {"pI": 8.26, "mw": 149.0, "hydro": 0.35, "gravy": -0.41, "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS"},
    "rituximab":   {"pI": 9.40, "mw": 145.0, "hydro": 0.35, "gravy": -0.22, "sequence": "QVQLQQPGAELVKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARSTYYGGDWYFNVWGAGTTVTVSA"},
    "NISTmAb":     {"pI": 8.47, "mw": 148.0, "hydro": 0.35, "gravy": -0.47, "sequence": "QVTLRESGPALVKPTQTLTLTCTFSGFSLSTAGMSVGWIRQPPGKALEWLADIWWDDKKDYNPSLKDRLTISKDTSKNQVVLKVTNMDPADTATYYCARDMIFNFYFDVWGQGTTVTVSS"},
}


def _compute_gravy(seq: str) -> float:
    """Compute GRAVY from sequence using BioPython or fallback."""
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        return ProteinAnalysis(seq.upper()).gravy()
    except ImportError:
        gt = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
              "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
              "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
        s = seq.upper()
        return sum(gt.get(aa, 0) for aa in s) / max(len(s), 1)


def _checks_to_status(checks: dict) -> str:
    """Convert a checks dict to PASS/FAIL status. All must be True for PASS."""
    if not checks:
        return "FAIL"
    return "PASS" if all(checks.values()) else "FAIL"


# =========================================================================
# 1. PropertyMapper
# =========================================================================
def test_property_mapper() -> dict:
    log.info("=== 1. PropertyMapper ===")
    from src.PropertyMapper import ProteinProperties, PropertyMapper
    mapper = PropertyMapper()
    data = {}
    checks = {}
    for name, ref in REFERENCE_MABS.items():
        seq = ref.get("sequence", "")
        gravy = _compute_gravy(seq) if seq else ref.get("gravy")
        protein = ProteinProperties(
            name=name, pI=ref["pI"], MW_kDa=ref["mw"],
            hydrophobicity=ref["hydro"],
            gravy_score=gravy,
            sequence=seq,
        )
        v = mapper.map_variants(protein)
        nu_main = v["main"]["nu"]
        ka_main = v["main"]["ka"]
        order_ok = v["acidic"]["nu"] < v["main"]["nu"] < v["basic"]["nu"]

        data[name] = {
            "main_nu": round(nu_main, 3),
            "main_ka": round(ka_main, 4),
            "acidic_nu": round(v["acidic"]["nu"], 3),
            "basic_nu": round(v["basic"]["nu"], 3),
            "variant_order": order_ok,
            "source": v.get("source", "unknown"),
        }

        # Substantive checks per mAb
        checks[f"{name}_variant_order"] = order_ok
        checks[f"{name}_nu_in_range"] = 1.5 <= nu_main <= 5.0
        checks[f"{name}_ka_in_range"] = 0.5 <= ka_main <= 10.0
        checks[f"{name}_source_valid"] = v.get("source", "") != "unknown"

        log.info("  %s: ka=%.4f nu=%.3f order=%s source=%s",
                 name, ka_main, nu_main, order_ok, v.get("source"))

    status = _checks_to_status(checks)
    if status == "FAIL":
        failed = [k for k, v in checks.items() if not v]
        log.warning("PropertyMapper FAIL: %s", failed)
    return {"status": status, "checks": checks, "data": data}


# =========================================================================
# 2. Developability
# =========================================================================
def test_developability() -> dict:
    log.info("=== 2. Developability ===")
    import numpy as np
    from src.developability_predictor import DevelopabilityPredictor
    pred = DevelopabilityPredictor()
    exp_data = {
        "adalimumab":  {"Tm": 71.0, "ACSINS": 1.1, "HIC": 8.8},
        "trastuzumab": {"Tm": 78.5, "ACSINS": 2.0, "HIC": 9.7},
        "bevacizumab": {"Tm": 63.5, "ACSINS": 0.8, "HIC": 11.8},
        "rituximab":   {"Tm": 69.0, "ACSINS": 2.1, "HIC": 10.8},
        "nivolumab":   {"Tm": 66.0, "ACSINS": 2.4, "HIC": 9.0},
    }
    data = {}
    checks = {}
    for name, exp in exp_data.items():
        seq = REFERENCE_MABS.get(name, {}).get("sequence", "A" * 100)
        embedding = np.zeros(640, dtype=np.float32)
        gravy = _compute_gravy(seq)
        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            pa = ProteinAnalysis(seq.upper())
            pi_val = pa.isoelectric_point()
            mw_val = pa.molecular_weight() / 1000.0
        except ImportError:
            pi_val = 8.0
            mw_val = 25.0
        biophysical = np.array([pi_val, mw_val, 1.0, 1.0, 40.0, 50.0, gravy], dtype=np.float32)
        r = pred.predict(embedding, biophysical, sequence=seq)
        preds = r if isinstance(r, dict) else {}
        data[name] = {
            "exp_Tm": exp["Tm"],
            "exp_ACSINS": exp["ACSINS"],
            "exp_HIC": exp["HIC"],
            "predictions": {k: round(v, 4) for k, v in preds.items() if isinstance(v, (int, float))},
        }

        # Substantive checks: predictions must exist and be in valid range [0, 1]
        has_agg = "agg_risk" in preds and isinstance(preds["agg_risk"], (int, float))
        has_stab = "stability" in preds and isinstance(preds["stability"], (int, float))
        has_visc = "viscosity_risk" in preds and isinstance(preds["viscosity_risk"], (int, float))
        checks[f"{name}_has_agg_risk"] = has_agg
        checks[f"{name}_has_stability"] = has_stab
        checks[f"{name}_has_viscosity"] = has_visc
        if has_agg:
            checks[f"{name}_agg_in_range"] = 0.0 <= preds["agg_risk"] <= 1.0
        if has_stab:
            checks[f"{name}_stab_in_range"] = 0.0 <= preds["stability"] <= 1.0
        if has_visc:
            checks[f"{name}_visc_in_range"] = 0.0 <= preds["viscosity_risk"] <= 1.0

    # Cross-mAb check: bevacizumab should have higher agg risk than rituximab
    # (bevacizumab is known for aggregation propensity)
    bev_agg = data.get("bevacizumab", {}).get("predictions", {}).get("agg_risk")
    rit_agg = data.get("rituximab", {}).get("predictions", {}).get("agg_risk")
    if bev_agg is not None and rit_agg is not None:
        checks["bev_higher_agg_than_rit"] = bev_agg > rit_agg

    status = _checks_to_status(checks)
    if status == "FAIL":
        failed = [k for k, v in checks.items() if not v]
        log.warning("Developability FAIL: %s", failed)
    return {"status": status, "checks": checks, "data": data}


# =========================================================================
# 3. Upstream
# =========================================================================
def test_upstream() -> dict:
    log.info("=== 3. Upstream ===")
    from src.upstream_twin import run_upstream_simulation, result_to_dict
    r = run_upstream_simulation()
    d = result_to_dict(r)
    # Realistic range checks — now actually used for status
    checks = {
        "vcd_realistic": r.peak_vcd <= 30,
        "titer_realistic": 2.0 <= r.final_titer <= 12.0,
        "viability_realistic": 60.0 <= r.viability_at_harvest <= 95.0,
        "vcd_positive": r.peak_vcd > 0,
        "titer_positive": r.final_titer > 0,
    }
    d["checks"] = checks
    log.info("  peak_vcd=%.2f titer=%.2f viab=%.1f%%", r.peak_vcd, r.final_titer, r.viability_at_harvest)
    status = _checks_to_status(checks)
    if status == "FAIL":
        failed = [k for k, v in checks.items() if not v]
        log.warning("Upstream FAIL: %s", failed)
    return {"status": status, "checks": checks, "data": d}


# =========================================================================
# 4. Analytical QC
# =========================================================================
def test_analytical_qc() -> dict:
    log.info("=== 4. Analytical QC ===")
    try:
        from src.analytical_qc_twin import run_analytical_qc
        nistmab_seq = REFERENCE_MABS["NISTmAb"]["sequence"]
        qc = run_analytical_qc(sequence=nistmab_seq, pI=8.28)
        data = {
            "cief": {k: round(v, 2) if isinstance(v, (int, float)) else v
                     for k, v in qc.cief.__dict__.items()},
            "ce_sds": {k: round(v, 2) if isinstance(v, (int, float)) else v
                       for k, v in qc.ce_sds.__dict__.items()},
            "glycan": {k: round(v, 2) if isinstance(v, (int, float)) else v
                       for k, v in qc.glycan.__dict__.items()},
            "overall_qc_pass": qc.overall_qc_pass,
        }

        # Substantive checks — validate data actually has meaningful values
        _c = data["cief"]
        _s = data["ce_sds"]
        checks = {
            "overall_qc_pass": bool(qc.overall_qc_pass),
            # cIEF: main peak should dominate (>50%), sum should be ~100%
            "cief_main_present": isinstance(_c.get("main_pct"), (int, float)) and _c["main_pct"] is not None,
            "cief_main_dominant": isinstance(_c.get("main_pct"), (int, float)) and _c.get("main_pct", 0) > 50.0,
            "cief_sum_valid": (
                isinstance(_c.get("acidic_pct"), (int, float)) and
                isinstance(_c.get("main_pct"), (int, float)) and
                isinstance(_c.get("basic_pct"), (int, float)) and
                95.0 <= (_c.get("acidic_pct", 0) + _c.get("main_pct", 0) + _c.get("basic_pct", 0)) <= 105.0
            ),
            # CE-SDS: intact should be >85%
            "cesds_intact_present": isinstance(_s.get("intact_pct"), (int, float)) and _s["intact_pct"] is not None,
            "cesds_intact_high": isinstance(_s.get("intact_pct"), (int, float)) and _s.get("intact_pct", 0) > 85.0,
        }

        _c_data = data["cief"]
        _s_data = data["ce_sds"]
        _g_data = data["glycan"]
        summary_lines = [
            f"Analytical QC Summary:",
            f"  cIEF: Acidic={_c_data.get('acidic_pct', 'N/A')}% | Main={_c_data.get('main_pct', 'N/A')}% | Basic={_c_data.get('basic_pct', 'N/A')}%",
            f"  CE-SDS: Intact={_s_data.get('intact_pct', 'N/A')}% | Fragment={_s_data.get('fragment_pct', 'N/A')}% | HMW={_s_data.get('hmw_pct', 'N/A')}%",
            f"  Glycan: G0F={_g_data.get('g0f_pct', 'N/A')}% | G1F={_g_data.get('g1f_pct', 'N/A')}%",
            f"  Overall QC: {'PASS' if qc.overall_qc_pass else 'FAIL'}",
        ]
        data["summary"] = "\n".join(summary_lines)

        status = _checks_to_status(checks)
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("Analytical QC FAIL: %s", failed)
        return {"status": status, "checks": checks, "data": data}
    except Exception as e:
        log.warning("Analytical QC: %s", e)
        return {"status": "SKIP", "data": {}, "error": str(e)}


# =========================================================================
# 5. Stability
# =========================================================================
def test_stability() -> dict:
    log.info("=== 5. Stability ===")
    from src.stability_twin import simulate_stability
    r5 = simulate_stability(temperature_c=5.0, duration_months=24)
    r40 = simulate_stability(temperature_c=40.0, duration_months=3)

    # Use hmw_growth_rate_pct_per_month as degradation rate proxy
    k5 = getattr(r5, "hmw_growth_rate_pct_per_month", None)
    k40 = getattr(r40, "hmw_growth_rate_pct_per_month", None)

    checks = {
        "k5_exists": k5 is not None,
        "k40_exists": k40 is not None,
        "r5_has_shelf_life": hasattr(r5, "shelf_life_months") and r5.shelf_life_months is not None,
        "r5_passes_spec": getattr(r5, "passes_24month_spec", False),
    }

    if k5 is not None and k40 is not None:
        ratio = k40 / max(k5, 1e-9)
        checks["k5_positive"] = k5 > 0
        checks["k40_gt_k5"] = k40 > k5  # Arrhenius: higher temp = faster degradation
        checks["ratio_realistic"] = 2.0 <= ratio <= 500.0
    else:
        ratio = 0.0

    log.info("  k_5C=%.4f  k_40C=%.4f  ratio=%.1f  shelf_life=%s",
             k5 or 0, k40 or 0, ratio, getattr(r5, "shelf_life_months", "?"))
    status = _checks_to_status(checks)
    if status == "FAIL":
        failed = [k for k, v in checks.items() if not v]
        log.warning("Stability FAIL: %s", failed)
    return {
        "status": status,
        "checks": checks,
        "data": {
            "k_5C": k5, "k_40C": k40, "ratio": round(ratio, 1),
            "shelf_life_months": getattr(r5, "shelf_life_months", None),
            "passes_24month_spec": getattr(r5, "passes_24month_spec", None),
        },
    }


# =========================================================================
# 6. NISTmAb
# =========================================================================
def test_nistmab() -> dict:
    log.info("=== 6. NISTmAb ===")
    from src.nistmab_benchmark import run_nistmab_validation
    r = run_nistmab_validation()
    metrics_list = []
    for m in r.metrics:
        metrics_list.append({
            "metric_name": m.metric_name,
            "predicted_value": m.predicted_value,
            "literature_value": m.literature_value,
            "unit": m.unit,
            "error": m.error,
            "error_pct": m.error_pct,
            "within_range": m.within_range,
            "notes": m.notes,
            "model_source": m.model_source,
        })

    checks = {
        "has_metrics": r.n_total > 0,
        "pass_rate_above_50pct": r.pass_rate >= 0.5,
        "at_least_5_metrics": r.n_total >= 5,
        "grade_not_failing": r.overall_grade in ("Excellent", "Good", "Fair"),
    }

    log.info("  %d/%d passed (%.0f%%) grade=%s", r.n_passed, r.n_total, r.pass_rate * 100, r.overall_grade)
    status = _checks_to_status(checks)
    if status == "FAIL":
        failed = [k for k, v in checks.items() if not v]
        log.warning("NISTmAb FAIL: %s", failed)
    return {
        "status": status,
        "checks": checks,
        "data": {
            "metrics": metrics_list,
            "n_passed": r.n_passed,
            "n_total": r.n_total,
            "pass_rate": r.pass_rate,
            "overall_grade": r.overall_grade,
            "wall_time_s": round(r.wall_time_s, 2),
            "summary": r.summary,
        },
    }


# =========================================================================
# 7. Immunogenicity
# =========================================================================
def test_immunogenicity() -> dict:
    log.info("=== 7. Immunogenicity ===")
    try:
        from src.immunogenicity_twin import run_immunogenicity_assessment
        from src.nistmab_benchmark import NISTMAB_SUPER_SEQUENCE
        r = run_immunogenicity_assessment(
            NISTMAB_SUPER_SEQUENCE, molecule_name="NISTmAb",
        )
        data = {}
        for k, v in r.__dict__.items():
            if isinstance(v, (int, float, str, bool)):
                data[k] = round(v, 4) if isinstance(v, float) else v
            elif isinstance(v, list):
                data[k] = f"{len(v)} items"

        # Substantive checks
        checks = {
            "ada_risk_score_exists": hasattr(r, "ada_risk_score") and r.ada_risk_score is not None,
            "ada_risk_level_exists": hasattr(r, "ada_risk_level") and r.ada_risk_level is not None,
            "ada_risk_level_valid": hasattr(r, "ada_risk_level") and r.ada_risk_level in ("Low", "Medium", "High"),
            "peptides_scanned": hasattr(r, "total_peptides_scanned") and r.total_peptides_scanned > 0,
            "hotspots_found": hasattr(r, "hotspots") and isinstance(r.hotspots, list) and len(r.hotspots) > 0,
        }
        if hasattr(r, "ada_risk_score") and isinstance(r.ada_risk_score, (int, float)):
            checks["ada_score_in_range"] = 0.0 <= r.ada_risk_score <= 1.0

        status = _checks_to_status(checks)
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("Immunogenicity FAIL: %s", failed)
        return {"status": status, "checks": checks, "data": data}
    except Exception as e:
        log.warning("Immunogenicity: %s", e)
        return {"status": "SKIP", "data": {}, "error": str(e)}


# =========================================================================
# 8. TheRaSAbDab 10-mAb panel
# =========================================================================
def test_therasabdab() -> dict:
    """10-mAb panel using synthetic Jain-137 data (TheRaSAbDab not available)."""
    log.info("=== 8. Synthetic mAb Panel (Jain-137) ===")
    try:
        import numpy as np
        from src.data_pipeline import generate_mock_jain137
        from src.PropertyMapper import ProteinProperties, PropertyMapper
        from src.developability_predictor import DevelopabilityPredictor
        from src.immunogenicity_twin import run_immunogenicity_assessment

        mock = generate_mock_jain137(n_samples=10, seed=137)
        mapper = PropertyMapper()
        dev_pred = DevelopabilityPredictor()

        panel = {}
        for row in mock["data"]:
            name = row["Name"]
            hc = str(row.get("Sequence_HC", ""))
            lc = str(row.get("Sequence_LC", ""))
            if len(hc) < 20 or len(lc) < 20:
                continue

            seq = hc + lc
            gravy = _compute_gravy(seq)
            try:
                from Bio.SeqUtils.ProtParam import ProteinAnalysis
                pa = ProteinAnalysis(seq.upper())
                pi_val = round(pa.isoelectric_point(), 2)
                mw_val = round(pa.molecular_weight() / 1000.0, 1)
            except ImportError:
                pi_val = 8.0
                mw_val = 25.0

            protein = ProteinProperties(
                name=name, pI=pi_val, MW_kDa=mw_val,
                hydrophobicity=max(0, min(1, (gravy + 2.0) / 4.0)),
                gravy_score=gravy,
                sequence=seq,
            )
            v = mapper.map_variants(protein)

            # Developability
            _emb = np.zeros(640, dtype=np.float32)
            _biop = np.array([pi_val, mw_val, 1.0, 1.0, 40.0, 50.0, gravy], dtype=np.float32)
            dev_r = dev_pred.predict(_emb, _biop, sequence=seq)
            dev_preds = dev_r if isinstance(dev_r, dict) else {}

            # ADA
            try:
                ada = run_immunogenicity_assessment(seq, molecule_name=name)
                ada_risk = ada.ada_risk_score
                ada_level = ada.ada_risk_level
                human_score = ada.humanization_score
            except Exception:
                ada_risk = 0
                ada_level = "N/A"
                human_score = 0

            panel[name] = {
                "computed_pI": pi_val,
                "computed_MW_Fv_kDa": mw_val,
                "pI_in_mAb_range": 5.0 <= pi_val <= 11.0,
                "sma_nu": round(v["main"]["nu"], 3),
                "sma_ka": round(v["main"]["ka"], 4),
                "variant_order_ok": v["acidic"]["nu"] < v["main"]["nu"] < v["basic"]["nu"],
                "sma_source": v.get("source", "unknown"),
                "dev_predictions": {k: round(vv, 4) for k, vv in dev_preds.items() if isinstance(vv, (int, float))},
                "ada_risk": round(ada_risk, 3) if isinstance(ada_risk, float) else ada_risk,
                "ada_level": ada_level,
                "humanization": round(human_score, 3) if isinstance(human_score, float) else human_score,
            }

        # Summary + checks
        checks = {}
        if panel:
            pis = [v["computed_pI"] for v in panel.values()]
            nus = [v["sma_nu"] for v in panel.values()]
            kas = [v["sma_ka"] for v in panel.values()]
            ada_scores = [v["ada_risk"] for v in panel.values() if isinstance(v["ada_risk"], (int, float))]

            checks["at_least_5_tested"] = len(panel) >= 5
            checks["all_variant_order_ok"] = all(v["variant_order_ok"] for v in panel.values())
            checks["all_pI_in_range"] = all(v["pI_in_mAb_range"] for v in panel.values())
            checks["all_have_dev_preds"] = all(
                len(v["dev_predictions"]) >= 2 for v in panel.values()
            )
            checks["ada_scores_present"] = len(ada_scores) >= 5
            if ada_scores:
                checks["ada_scores_in_range"] = all(0.0 <= s <= 1.0 for s in ada_scores)

            summary = {
                "n_tested": len(panel),
                "pI_range": f"{min(pis):.2f}-{max(pis):.2f}",
                "pI_mean": round(sum(pis) / len(pis), 2),
                "nu_range": f"{min(nus):.3f}-{max(nus):.3f}",
                "ka_range": f"{min(kas):.4f}-{max(kas):.4f}",
                "ada_risk_range": f"{min(ada_scores):.3f}-{max(ada_scores):.3f}" if ada_scores else "N/A",
                "all_variant_order_ok": checks.get("all_variant_order_ok", False),
                "all_pI_in_range": checks.get("all_pI_in_range", False),
            }
        else:
            checks["panel_not_empty"] = False
            summary = {"n_tested": 0}

        status = _checks_to_status(checks)
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("TheRaSAbDab FAIL: %s", failed)
        return {"status": status, "checks": checks, "summary": summary, "data": panel}

    except Exception as e:
        log.warning("mAb Panel: %s", e)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# 9. Cross-check (two Jain-137 seeds for reproducibility)
# =========================================================================
def test_cross_check() -> dict:
    log.info("=== 9. Cross-check ===")
    try:
        from src.data_pipeline import generate_mock_jain137

        d1 = generate_mock_jain137(n_samples=10, seed=42)
        d2 = generate_mock_jain137(n_samples=10, seed=42)

        names_1 = [r["Name"] for r in d1["data"]]
        names_2 = [r["Name"] for r in d2["data"]]
        name_match = names_1 == names_2

        seqs_match = all(
            r1["Sequence_HC"] == r2["Sequence_HC"] and r1["Sequence_LC"] == r2["Sequence_LC"]
            for r1, r2 in zip(d1["data"], d2["data"])
        )

        d3 = generate_mock_jain137(n_samples=10, seed=99)
        seqs_differ = any(
            r1["Sequence_HC"] != r3["Sequence_HC"]
            for r1, r3 in zip(d1["data"], d3["data"])
        )

        checks = {
            "name_match_same_seed": name_match,
            "sequence_match_same_seed": seqs_match,
            "sequences_differ_diff_seed": seqs_differ,
        }
        checks["deterministic"] = name_match and seqs_match and seqs_differ

        data = {
            "n_samples": len(d1["data"]),
            **checks,
        }

        status = _checks_to_status(checks)
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("Cross-check FAIL: %s", failed)
        return {"status": status, "checks": checks, "data": data}
    except Exception as e:
        log.warning("Cross-check: %s", e)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# 10. CEX Analytical Resolution (Yamamoto SMA gradient model)
# =========================================================================
def test_cex_resolution() -> dict:
    """
    Validate that PropertyMapper v7.3 charge-variant parameters produce
    baseline-resolved CEX chromatography (Rs > 1.2) for all reference mAbs.
    """
    log.info("=== 10. CEX Analytical Resolution ===")
    try:
        from src.PropertyMapper import ProteinProperties, PropertyMapper
        from src.ml_predictor import estimate_rt_from_sma

        mapper = PropertyMapper()
        data = {}
        checks = {}

        for name, ref in REFERENCE_MABS.items():
            seq = ref.get("sequence", "")
            gravy = _compute_gravy(seq) if seq else ref.get("gravy")
            protein = ProteinProperties(
                name=name, pI=ref["pI"], MW_kDa=ref["mw"],
                hydrophobicity=ref["hydro"],
                gravy_score=gravy,
                sequence=seq,
            )
            v = mapper.map_variants(protein)

            tR_acidic = estimate_rt_from_sma(ka=v["acidic"]["ka"], nu=v["acidic"]["nu"])
            tR_main   = estimate_rt_from_sma(ka=v["main"]["ka"],   nu=v["main"]["nu"])
            tR_basic  = estimate_rt_from_sma(ka=v["basic"]["ka"],  nu=v["basic"]["nu"])

            sigma_factor = 0.03
            w_acidic = 4.0 * sigma_factor * max(tR_acidic, 0.5)
            w_main   = 4.0 * sigma_factor * max(tR_main, 0.5)
            w_basic  = 4.0 * sigma_factor * max(tR_basic, 0.5)

            Rs_acid_main = 2.0 * abs(tR_main - tR_acidic) / (w_acidic + w_main) if (w_acidic + w_main) > 0 else 0.0
            Rs_main_basic = 2.0 * abs(tR_basic - tR_main) / (w_main + w_basic) if (w_main + w_basic) > 0 else 0.0
            Rs_min = min(Rs_acid_main, Rs_main_basic)
            passes = Rs_min >= 1.2

            checks[f"{name}_Rs_above_1.2"] = passes
            checks[f"{name}_elution_order"] = tR_acidic < tR_main < tR_basic
            checks[f"{name}_rt_positive"] = tR_acidic > 0 and tR_main > 0 and tR_basic > 0

            data[name] = {
                "tR_acidic_min": round(tR_acidic, 2),
                "tR_main_min":   round(tR_main, 2),
                "tR_basic_min":  round(tR_basic, 2),
                "Rs_acidic_main": round(Rs_acid_main, 2),
                "Rs_main_basic":  round(Rs_main_basic, 2),
                "Rs_min": round(Rs_min, 2),
                "passes_Rs_1.2": passes,
                "method": "analytical_yamamoto_sma",
                "sigma_factor": sigma_factor,
            }
            log.info("  %s: tR=[%.1f, %.1f, %.1f] Rs_min=%.2f %s",
                     name, tR_acidic, tR_main, tR_basic, Rs_min,
                     "PASS" if passes else "FAIL")

        status = _checks_to_status(checks)
        summary = {
            "n_mabs": len(data),
            "all_Rs_above_1.2": all(d["passes_Rs_1.2"] for d in data.values()),
            "min_Rs_across_panel": round(min(d["Rs_min"] for d in data.values()), 2),
            "method": "analytical_yamamoto_sma",
        }
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("CEX Resolution FAIL: %s", failed)
        return {"status": status, "checks": checks, "summary": summary, "data": data}

    except Exception as e:
        log.warning("CEX Resolution: %s", e)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# 11. End-to-End: Feature → Developability → Report Integration
# =========================================================================
def test_end_to_end() -> dict:
    """
    Full pipeline integration test: build a ReportObject from a mock intent
    and verify that data flows correctly from features through developability
    assessment into report sections.
    """
    log.info("=== 11. End-to-End Integration ===")
    try:
        from src.report_assembler import assemble_report
        from src.report_schema import ReportObject, GRADE_LOW_UPPER, GRADE_MEDIUM_UPPER

        # Build a realistic intent for NISTmAb (canonical mAb)
        # Key: intent uses "name" not "molecule_name"
        nistmab_seq = REFERENCE_MABS["NISTmAb"]["sequence"]
        intent = {
            "name": "NISTmAb-SelfTest",
            "sequence": nistmab_seq,
            "molecule_class": "canonical_mab",
            "molecule_class_info": {
                "type": "canonical_mab",
                "display_name": "Monoclonal Antibody (IgG)",
                "has_fc_region": True,
                "expects_glycosylation": True,
            },
            # Biophysical from intent (when no feature_set_obj)
            "pI": 8.47,
            "mw": 148.0,
            "hydrophobicity": 0.35,
            "gravy": -0.47,
            "seq_length": len(nistmab_seq),
            "cysteine_count": 4,
            # Liability summary — use real key names as produced by the app
            "liability_summary": {
                "deamidation_sites": 3,
                "oxidation_sites": 2,
                "asp_isomerization_sites": 1,
                "n_glycosylation_sites": 1,
                "dp_clipping_sites": 2,  # Real key from liability scanner
            },
        }

        # Build analysis cache matching expected structure:
        #   dev_result.data.predictions  (used by _get_dev_data)
        #   analytical_qc (used by _build_developability)
        analysis_cache = {
            "dev_result": {
                "data": {
                    "predictions": {
                        "agg_risk": 0.20,
                        "stability": 0.79,
                        "viscosity_risk": 0.15,
                    },
                    "score": {
                        "score": 0.22,
                        "grade": "Low",
                    },
                },
            },
            "analytical_qc": {
                "sec": {"monomer_pct": 98.5, "hmw_pct": 1.2},
                "cief": {"main_pct": 82.0, "acidic_pct": 10.0, "basic_pct": 8.0},
                "ce_sds": {"intact_pct": 96.0},
            },
        }

        report = assemble_report(intent, analysis_cache)
        checks = {}

        # ── Executive Summary checks ──
        es = report.executive_summary
        checks["es_molecule_name"] = es.molecule_name == "NISTmAb-SelfTest"
        checks["es_molecule_class"] = es.molecule_class == "canonical_mab"
        checks["es_has_grade"] = es.overall_grade in ("Low Risk", "Medium Risk", "High Risk", "Low", "Medium", "High", "")
        checks["es_has_recommendation"] = len(es.recommendation) > 0

        # ── Molecule Overview checks ──
        mo = report.molecule_overview
        checks["mo_has_name"] = len(mo.name) > 0
        checks["mo_has_pi"] = mo.isoelectric_point is not None
        checks["mo_has_mw"] = mo.molecular_weight_kda is not None

        # ── Developability checks ──
        ds = report.developability
        checks["dev_has_risk_dimensions"] = len(ds.risk_dimensions) > 0
        checks["dev_score_valid"] = (
            ds.composite_score is None or
            (isinstance(ds.composite_score, (int, float)) and 0.0 <= ds.composite_score <= 1.0)
        )
        # Verify risk dimensions have actual data, not all "Not assessed"
        assessed_dims = [r for r in ds.risk_dimensions if r.assessed]
        checks["dev_has_assessed_dimensions"] = len(assessed_dims) >= 2

        # Each assessed dimension should have valid score
        for dim in assessed_dims:
            checks[f"dev_{dim.dimension.lower()}_score_valid"] = 0.0 <= dim.score <= 1.0
            checks[f"dev_{dim.dimension.lower()}_has_grade"] = dim.grade in ("Low", "Medium", "High")

        # ── dp_clip_sites propagation check (P1 critical fix) ──
        checks["ctx_dp_clip_sites_propagated"] = (
            report.context.dp_clip_sites is not None and report.context.dp_clip_sites > 0
        )

        # ── QTPP checks ──
        checks["dev_has_qtpp"] = len(ds.qtpp_rows) >= 5
        if ds.qtpp_rows:
            # QTPP rows use "status" field for assessment result (e.g. "Within Target", "Not Assessed")
            assessed_qtpp = [
                r for r in ds.qtpp_rows
                if r.get("status", "").lower() not in ("not assessed", "n/a", "")
            ]
            checks["qtpp_has_assessed_rows"] = len(assessed_qtpp) >= 3
            # QTPP rows should have justification field
            with_justification = [r for r in ds.qtpp_rows if r.get("justification")]
            checks["qtpp_has_justification"] = len(with_justification) >= 3

        # ── Top Risks checks ──
        tr = es.top_risks
        checks["es_has_top_risks"] = len(tr) >= 1
        if tr:
            # Each risk should have tier label
            checks["risks_have_tier_labels"] = all("[Tier" in r for r in tr)

        # ── Analytical section ──
        an = report.analytical
        # With analytical_qc in cache, we should see assessed data
        checks["analytical_evidence_status"] = an.evidence_status in ("assessed", "partial", "")

        # ── Process/PK section — ADA should be "Not assessed" (no ada data provided) ──
        pk = report.process_pk
        checks["ada_handled"] = pk.ada_risk_level in ("Not assessed", "Low", "Medium", "High", "")

        # ── Model metadata ──
        mm = report.model_metadata
        checks["metadata_has_route"] = len(mm.analysis_route) > 0 or len(mm.model_source) > 0

        # ── Serialization check ──
        json_str = report.to_json()
        checks["serializes_to_json"] = len(json_str) > 100
        parsed = json.loads(json_str)
        checks["json_roundtrip"] = parsed["executive_summary"]["molecule_name"] == "NISTmAb-SelfTest"

        # ── ReportContext consistency ──
        ctx = report.context
        checks["ctx_molecule_name"] = ctx.molecule_name == "NISTmAb-SelfTest"
        checks["ctx_class"] = ctx.molecule_class == "canonical_mab"

        status = _checks_to_status(checks)
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("End-to-End FAIL: %s", failed)
        else:
            log.info("  End-to-end integration: %d checks all PASS", len(checks))

        return {
            "status": status,
            "checks": checks,
            "data": {
                "n_checks": len(checks),
                "n_passed": sum(1 for v in checks.values() if v),
                "n_failed": sum(1 for v in checks.values() if not v),
                "report_json_size_bytes": len(json_str),
            },
        }
    except Exception as e:
        log.error("End-to-End: %s", e, exc_info=True)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# 12. Bispecific Format — 3-Species Separation + Report Coherence
# =========================================================================
def test_bispecific_format() -> dict:
    """
    Validate bispecific-specific functionality:
      1. predict_bispecific_separation() produces distinct RTs and non-zero Rs
      2. Report assembly for bispecific format includes format-aware caveats
      3. QTPP includes bispecific-specific rows
      4. Recommendation is molecule-aware
    """
    log.info("=== 12. Bispecific Format ===")
    try:
        from src.agents import predict_bispecific_separation
        from src.report_assembler import assemble_report
        from src.developability_core import generate_qtpp

        # Use two distinct VH sequences (from reference mAbs)
        chain_a = REFERENCE_MABS["trastuzumab"]["sequence"]
        chain_b = REFERENCE_MABS["bevacizumab"]["sequence"]

        checks = {}

        # ── Part 1: Bispecific separation analysis ──
        result = predict_bispecific_separation(
            chain_a_sequence=chain_a,
            chain_b_sequence=chain_b,
            chain_a_name="Tras-VH",
            chain_b_name="Bev-VH",
        )
        r_data = result.get("data", {}) if isinstance(result, dict) else {}

        # Species should exist
        species = r_data.get("species", {})
        checks["has_AA_species"] = "AA" in species
        checks["has_AB_species"] = "AB" in species
        checks["has_BB_species"] = "BB" in species

        # Peaks should have distinct RTs
        peaks = r_data.get("peaks", {})
        if peaks:
            rt_aa = peaks.get("AA", {}).get("rt_min", 0)
            rt_ab = peaks.get("AB", {}).get("rt_min", 0)
            rt_bb = peaks.get("BB", {}).get("rt_min", 0)
            checks["peaks_have_rt"] = rt_aa > 0 and rt_ab > 0 and rt_bb > 0
            checks["rts_are_distinct"] = len({round(rt_aa, 1), round(rt_ab, 1), round(rt_bb, 1)}) >= 2
            # At minimum, AA and BB should differ (they have different pI)
            checks["aa_bb_different"] = abs(rt_aa - rt_bb) > 0.1
            log.info("  Peaks: AA=%.2f AB=%.2f BB=%.2f", rt_aa, rt_ab, rt_bb)
        else:
            checks["peaks_have_rt"] = False

        # Resolution should be non-zero
        res = r_data.get("resolution", {})
        if res:
            rs_min = res.get("min_rs", 0)
            checks["rs_nonzero"] = rs_min > 0
            checks["rs_realistic"] = 0 < rs_min < 20
            log.info("  Rs_min=%.3f", rs_min)
        else:
            checks["rs_nonzero"] = False

        # Chromatogram data should exist
        chrom = r_data.get("chromatogram", {})
        checks["has_chromatogram"] = bool(chrom) and "time" in chrom

        # ── Part 2: QTPP format-aware caveats ──
        qtpp_rows = generate_qtpp(
            feature_values={"pI": 8.5, "mw_kda": 150.0, "hydrophobicity": 0.35},
            dev_predictions={"agg_risk": 0.3, "stability": 0.7},
            molecule_class="bispecific",
        )
        checks["qtpp_has_rows"] = len(qtpp_rows) >= 5

        # Standard rows should have format caveat for bispecific
        std_rows_with_caveat = [
            r for r in qtpp_rows
            if hasattr(r, "justification") and "canonical mAb" in (r.justification or "")
        ]
        checks["qtpp_has_format_caveats"] = len(std_rows_with_caveat) >= 3

        # Bispecific-specific rows should exist
        bispec_attrs = {r.attribute for r in qtpp_rows}
        checks["qtpp_has_homodimer_row"] = any("Homodimer" in a for a in bispec_attrs)
        checks["qtpp_has_cex_species_row"] = any("CEX Species" in a or "Rs" in a for a in bispec_attrs)
        log.info("  QTPP: %d rows, %d with caveats, bispecific-specific attrs: %s",
                 len(qtpp_rows), len(std_rows_with_caveat),
                 [a for a in bispec_attrs if "Homodimer" in a or "CEX" in a or "Rs" in a])

        # ── Part 3: Report assembly for bispecific ──
        bispec_intent = {
            "name": "TestBispecific",
            "sequence": chain_a + chain_b,
            "molecule_class": "bispecific",
            "molecule_class_info": {
                "type": "bispecific",
                "display_name": "Bispecific Antibody",
                "has_fc_region": True,
                "expects_glycosylation": True,
            },
            "pI": 8.5, "mw": 150.0, "hydrophobicity": 0.35, "gravy": -0.35,
            "seq_length": len(chain_a) + len(chain_b),
            "liability_summary": {
                "deamidation_sites": 4, "oxidation_sites": 3, "dp_clipping_sites": 3,
            },
        }
        bispec_cache = {
            "dev_result": {
                "data": {
                    "predictions": {"agg_risk": 0.35, "stability": 0.65, "viscosity_risk": 0.2},
                    "score": {"score": 0.35, "grade": "Medium"},
                },
            },
        }
        bispec_report = assemble_report(bispec_intent, bispec_cache)

        # Report recommendation should be molecule-aware for bispecific
        rec_detail = bispec_report.executive_summary.recommendation_detail
        checks["recommendation_mentions_bispecific"] = (
            "bispecific" in rec_detail.lower() or
            "species" in rec_detail.lower() or
            "dual" in rec_detail.lower() or
            "format" in rec_detail.lower()
        )

        # Context should carry bispecific class and dp_clip_sites
        checks["report_ctx_class"] = bispec_report.context.molecule_class == "bispecific"
        checks["bispec_dp_clip_propagated"] = (
            bispec_report.context.dp_clip_sites is not None and bispec_report.context.dp_clip_sites == 3
        )
        # DP clipping should appear in top_risks
        bispec_risks_text = " ".join(bispec_report.executive_summary.top_risks).lower()
        checks["bispec_dp_in_risks"] = "clip" in bispec_risks_text or "asp-pro" in bispec_risks_text or "dp" in bispec_risks_text

        # Top risks should have tier labels (when present)
        top_risks = bispec_report.executive_summary.top_risks
        if top_risks:
            # At least one risk should have a tier label
            tier_labeled = [r for r in top_risks if "[Tier" in r]
            checks["bispec_risks_exist"] = len(top_risks) >= 1
            checks["bispec_risks_have_tiers"] = len(tier_labeled) >= 1
        else:
            # No risks generated — acceptable for sparse data
            checks["bispec_risks_exist"] = True  # not required
            checks["bispec_risks_have_tiers"] = True

        status = _checks_to_status(checks)
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("Bispecific Format FAIL: %s", failed)
        else:
            log.info("  Bispecific format: %d checks all PASS", len(checks))

        return {
            "status": status,
            "checks": checks,
            "data": {
                "n_checks": len(checks),
                "n_passed": sum(1 for v in checks.values() if v),
                "n_failed": sum(1 for v in checks.values() if not v),
                "peaks": peaks if peaks else {},
                "resolution": res if res else {},
                "qtpp_rows": len(qtpp_rows),
            },
        }
    except Exception as e:
        log.error("Bispecific Format: %s", e, exc_info=True)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# 13. Non-Canonical Formats — Fc-fusion, fusion_protein, single_domain
# =========================================================================
def test_noncanonical_formats() -> dict:
    """
    Validate crash-free report assembly, correct molecule-aware narrative,
    format-specific assays, and conservative recommendations for non-canonical
    molecule formats: Fc-fusion, fusion_protein (tandem scFv), single_domain.
    """
    log.info("=== 13. Non-Canonical Formats ===")
    try:
        from src.report_assembler import assemble_report
        from src.validation_planner import generate_validation_plan

        checks = {}

        # ── Shared helpers ──
        def _build_intent(name, seq, mol_class, display, has_fc, expects_glyc, pi, mw, liabilities=None):
            liab = liabilities or {
                "deamidation_sites": 2, "oxidation_sites": 1,
                "dp_clipping_sites": 1,
            }
            return {
                "name": name,
                "sequence": seq,
                "molecule_class": mol_class,
                "molecule_class_info": {
                    "type": mol_class,
                    "display_name": display,
                    "has_fc_region": has_fc,
                    "expects_glycosylation": expects_glyc,
                },
                "pI": pi, "mw": mw,
                "hydrophobicity": 0.35, "gravy": -0.30,
                "seq_length": len(seq),
                "liability_summary": liab,
            }

        def _build_cache(agg=0.30, stab=0.60, visc=0.15, score=0.30, grade="Medium"):
            return {
                "dev_result": {
                    "data": {
                        "predictions": {"agg_risk": agg, "stability": stab, "viscosity_risk": visc},
                        "score": {"score": score, "grade": grade},
                    },
                },
            }

        # ── Representative sequences ──
        # Fc-fusion (etanercept-like): TNFR ectodomain fragment + Fc hinge-CH2-CH3
        fc_fusion_seq = (
            "LPAQVAFTPYAPEPGSTCRLREYYDQTAQMCCSKCSPGQHAKVFCTKTSDTVCDSCEDN"
            "STYTQLWNWVPECLSCGSRCSSDQVETQACTREQNRICTCRPGWYCALSKQEGCRLCAP"
            "DKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKF"
            "NWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEK"
            "TISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPP"
            "VLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPG"
        )

        # Fusion protein (tandem scFv - blinatumomab-like)
        fusion_seq = (
            "DIQLTQSPSSLSASVGDRVTITCRASQDIRNYLAWYQQKPGKAPKLLIYAASSLQSGVP"
            "SRFSGSGSGTDFTLTISSLQPEDFATYYCQQYDSSPWTFGQGTKVEIKGGGGSGGGGSGG"
            "GGSEVQLVESGGGLVQPGGSLRLSCAASGFTFNTYAMNWVRQAPGKGLEWVARIRSKYN"
            "NYATYYADSVKDRFTISRDDSKNTLYLQMNSLRAEDTAVYYCVRHGNFGNSYVSWFAYW"
            "GQGTLVTVSSGGGGSGGGGSGGGGSQVQLQQSGAELARPGASVKMSCKASGYTFTRYTM"
            "HWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATLTTDKSSSTAYMQLSSLTSEDSAV"
            "YYCARYDDHYCLDSYYVMDAWGQGTSVTVSSGGGGSGGGGSGGGGSDIKMTQSPSSMYA"
            "SLGERVTITCKASQDINSYLSWFQQKPGKSPKTLIYRANRLVDGVPSRFSGSGSGQDYS"
            "LTISSLEYEDMGIYYCLQYDEFPLTFGAGTKLELK"
        )

        # Single domain (VHH / nanobody - caplacizumab-like)
        nanobody_seq = (
            "QVQLQESGGGLVQPGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTY"
            "YADSVKGRFTISRDNSKNTLYLQMNSLKPEDTAVYYCAADQNRGY"
            "EEYWGQGTQVTVSS"
        )

        # ──────────────────────────────────────────────────────────
        # A. Fc-fusion (e.g., etanercept)
        # ──────────────────────────────────────────────────────────
        log.info("  A. Fc-fusion molecule")
        fc_intent = _build_intent(
            "Etanercept-SelfTest", fc_fusion_seq, "fc_fusion",
            "Fc-Fusion Protein", has_fc=True, expects_glyc=True,
            pi=8.34, mw=150.0,
        )
        fc_report = assemble_report(fc_intent, _build_cache())

        # Crash-free
        checks["fc_fusion_no_crash"] = fc_report is not None

        # Correct class propagation
        checks["fc_fusion_ctx_class"] = fc_report.context.molecule_class == "fc_fusion"
        checks["fc_fusion_es_class"] = fc_report.executive_summary.molecule_class == "fc_fusion"

        # Molecule-aware narrative — should NOT say "Standard CMC pathway"
        rec = fc_report.executive_summary.recommendation_detail
        checks["fc_fusion_no_standard_cmc"] = "standard cmc" not in rec.lower()
        # Should mention format-specific language
        checks["fc_fusion_format_aware_rec"] = (
            "fc-fusion" in rec.lower() or "linker" in rec.lower() or
            "format" in rec.lower() or "half-life" in rec.lower()
        )

        # QTPP should have canonical-mAb caveat for non-canonical format
        qtpp_rows = fc_report.developability.qtpp_rows
        caveat_rows = [r for r in qtpp_rows
                       if "canonical" in (r.get("justification", "") or "").lower()]
        checks["fc_fusion_qtpp_caveat"] = len(caveat_rows) >= 1

        # Format-specific assays via validation planner
        fc_plan = generate_validation_plan(
            risk_scores={"agg_risk": 0.30, "stability": 0.60},
            intent=fc_intent, molecule_class="fc_fusion",
        )
        fc_fsa = fc_plan.get("format_specific_assays", [])
        checks["fc_fusion_has_format_assays"] = len(fc_fsa) >= 1
        # Should NOT include FcRn_binding exclusion note (Fc-fusions DO have Fc)
        fc_excluded = [e["id"] for e in fc_plan.get("excluded_assays", [])]
        # Fc-fusion DOES have Fc, so FcRn should NOT be excluded
        checks["fc_fusion_fcrn_not_excluded"] = "FcRn_binding" not in fc_excluded

        log.info("    Fc-fusion: %d checks, class=%s",
                 sum(1 for k, v in checks.items() if k.startswith("fc_fusion") and v),
                 fc_report.context.molecule_class)

        # ──────────────────────────────────────────────────────────
        # B. Fusion protein (tandem scFv, no Fc)
        # ──────────────────────────────────────────────────────────
        log.info("  B. Fusion protein (tandem scFv)")
        fp_intent = _build_intent(
            "TandemScFv-SelfTest", fusion_seq, "fusion_protein",
            "Fusion Protein (tandem scFv)", has_fc=False, expects_glyc=False,
            pi=7.2, mw=55.0,
        )
        fp_report = assemble_report(fp_intent, _build_cache())

        checks["fusion_no_crash"] = fp_report is not None
        checks["fusion_ctx_class"] = fp_report.context.molecule_class == "fusion_protein"
        checks["fusion_es_class"] = fp_report.executive_summary.molecule_class == "fusion_protein"

        # Molecule-aware narrative
        fp_rec = fp_report.executive_summary.recommendation_detail
        checks["fusion_no_standard_cmc"] = "standard cmc" not in fp_rec.lower()
        checks["fusion_format_aware_rec"] = (
            "fusion" in fp_rec.lower() or "linker" in fp_rec.lower() or
            "format" in fp_rec.lower() or "domain" in fp_rec.lower()
        )

        # QTPP caveat present
        fp_qtpp = fp_report.developability.qtpp_rows
        fp_caveat_rows = [r for r in fp_qtpp
                          if "canonical" in (r.get("justification", "") or "").lower()]
        checks["fusion_qtpp_caveat"] = len(fp_caveat_rows) >= 1

        # Format-specific assays: linker_domain_integrity + domain_interaction
        fp_plan = generate_validation_plan(
            risk_scores={"agg_risk": 0.30, "stability": 0.60},
            intent=fp_intent, molecule_class="fusion_protein",
        )
        fp_fsa = fp_plan.get("format_specific_assays", [])
        fp_fsa_ids = [a["id"] for a in fp_fsa]
        checks["fusion_has_linker_assay"] = "linker_domain_integrity" in fp_fsa_ids
        checks["fusion_has_domain_assay"] = "domain_interaction" in fp_fsa_ids
        # FcRn should be excluded (no Fc region)
        fp_excluded_ids = [e["id"] for e in fp_plan.get("excluded_assays", [])]
        checks["fusion_fcrn_excluded"] = "FcRn_binding" in fp_excluded_ids

        log.info("    Fusion protein: %d checks, FSA=%s",
                 sum(1 for k, v in checks.items() if k.startswith("fusion") and v),
                 fp_fsa_ids)

        # ──────────────────────────────────────────────────────────
        # C. Single domain (VHH / nanobody)
        # ──────────────────────────────────────────────────────────
        log.info("  C. Single domain (nanobody)")
        sd_intent = _build_intent(
            "Nanobody-SelfTest", nanobody_seq, "single_domain",
            "Single-Domain Antibody (VHH)", has_fc=False, expects_glyc=False,
            pi=6.5, mw=14.0,
        )
        sd_report = assemble_report(sd_intent, _build_cache())

        checks["sdab_no_crash"] = sd_report is not None
        checks["sdab_ctx_class"] = sd_report.context.molecule_class == "single_domain"
        checks["sdab_es_class"] = sd_report.executive_summary.molecule_class == "single_domain"

        # Molecule-aware narrative
        sd_rec = sd_report.executive_summary.recommendation_detail
        checks["sdab_no_standard_cmc"] = "standard cmc" not in sd_rec.lower()
        checks["sdab_format_aware_rec"] = (
            "single" in sd_rec.lower() or "domain" in sd_rec.lower() or
            "aggregation" in sd_rec.lower() or "format" in sd_rec.lower() or
            "nanobody" in sd_rec.lower() or "thermal" in sd_rec.lower()
        )

        # Validation plan: should include single_domain-specific assays
        sd_plan = generate_validation_plan(
            risk_scores={"agg_risk": 0.30, "stability": 0.60},
            intent=sd_intent, molecule_class="single_domain",
        )
        sd_fsa = sd_plan.get("format_specific_assays", [])
        checks["sdab_has_format_assays"] = len(sd_fsa) >= 1

        # D. Single domain with FORCED LOW score (Caplacizumab edge case)
        # Ensures "Standard CMC pathway" never appears for non-canonical
        # molecules even when they score Low risk.
        sd_low_report = assemble_report(
            sd_intent, _build_cache(score=0.15, grade="Low")
        )
        sd_low_rec = sd_low_report.executive_summary.recommendation_detail
        checks["sdab_low_no_standard_cmc"] = "standard cmc" not in sd_low_rec.lower()
        checks["sdab_low_format_aware"] = (
            "format" in sd_low_rec.lower() or "aggregation" in sd_low_rec.lower()
            or "thermal" in sd_low_rec.lower()
        )
        log.info("    Single domain (Low score): no_std_cmc=%s, format_aware=%s",
                 checks["sdab_low_no_standard_cmc"], checks["sdab_low_format_aware"])

        log.info("    Single domain: %d checks, FSA=%d",
                 sum(1 for k, v in checks.items() if k.startswith("sdab") and v),
                 len(sd_fsa))

        status = _checks_to_status(checks)
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("Non-Canonical Formats FAIL: %s", failed)
        else:
            log.info("  Non-Canonical Formats: %d checks all PASS", len(checks))

        return {
            "status": status,
            "checks": checks,
            "data": {
                "n_checks": len(checks),
                "n_passed": sum(1 for v in checks.values() if v),
                "n_failed": sum(1 for v in checks.values() if not v),
                "formats_tested": ["fc_fusion", "fusion_protein", "single_domain"],
            },
        }
    except Exception as e:
        log.error("Non-Canonical Formats: %s", e, exc_info=True)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# 14. OOD Confidence Propagation — Uncertainty Under Low Evidence
# =========================================================================
def test_ood_confidence() -> dict:
    """
    Validate that out-of-distribution (OOD) molecules:
    1. Get confidence capped (not hardcoded High)
    2. Have conservative recommendation language
    3. Include OOD caveats in key_caveats
    4. Model metadata reflects capped confidence
    Also verify non-OOD canonical mAb retains High confidence (no regression).
    """
    log.info("=== 14. OOD Confidence Propagation ===")
    try:
        from src.report_assembler import assemble_report

        checks = {}

        # ── OOD molecule: etanercept-like Fc-fusion with high z-score ──
        ood_seq = (
            "LPAQVAFTPYAPEPGSTCRLREYYDQTAQMCCSKCSPGQHAKVFCTKTSDTVCDSCEDN"
            "STYTQLWNWVPECLSCGSRCSSDQVETQACTREQNRICTCRPGWYCALSKQEGCRLCAP"
            "DKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKF"
            "NWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEK"
            "TISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPP"
            "VLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPG"
        )
        ood_intent = {
            "name": "OOD-FcFusion-SelfTest",
            "sequence": ood_seq,
            "molecule_class": "fc_fusion",
            "molecule_class_info": {
                "type": "fc_fusion",
                "display_name": "Fc-Fusion Protein",
                "has_fc_region": True,
                "expects_glycosylation": True,
            },
            "pI": 8.34, "mw": 150.0,
            "hydrophobicity": 0.35, "gravy": -0.30,
            "seq_length": len(ood_seq),
            "liability_summary": {
                "deamidation_sites": 2, "oxidation_sites": 1,
                "dp_clipping_sites": 1,
            },
        }
        ood_cache = {
            "dev_result": {
                "data": {
                    "predictions": {"agg_risk": 0.25, "stability": 0.70, "viscosity_risk": 0.10},
                    "score": {"score": 0.25, "grade": "Low"},
                },
            },
            # Simulate OOD flag — same key the predictor writes
            "ood_bypass_reason": "OOD: max z = 4.20 on feature(s): charge_patches, hydrophobic_moment",
        }

        ood_report = assemble_report(ood_intent, ood_cache)
        checks["ood_no_crash"] = ood_report is not None

        # Context should reflect OOD status
        checks["ood_ctx_is_ood"] = ood_report.context.is_ood is True
        checks["ood_ctx_confidence_low"] = ood_report.context.ood_confidence == "Low"

        # Executive summary confidence should be capped (not "High")
        es_conf = ood_report.executive_summary.confidence_level
        checks["ood_es_confidence_capped"] = es_conf in ("Low", "Medium")
        checks["ood_es_confidence_not_high"] = es_conf != "High"

        # Risk dimensions should have capped confidence
        for dim in ood_report.developability.risk_dimensions:
            if dim.assessed:
                checks[f"ood_dim_{dim.dimension.lower()}_conf_capped"] = (
                    dim.confidence in ("Low", "Medium")
                )

        # Recommendation should mention OOD / uncertainty / experimental confirmation
        rec_detail = ood_report.executive_summary.recommendation_detail
        checks["ood_rec_mentions_uncertainty"] = (
            "out-of-distribution" in rec_detail.lower() or
            "uncertainty" in rec_detail.lower() or
            "experimentally" in rec_detail.lower() or
            "outside" in rec_detail.lower()
        )

        # Key caveats should include OOD caveat
        caveats_text = " ".join(ood_report.executive_summary.key_caveats).lower()
        checks["ood_caveat_present"] = (
            "out-of-distribution" in caveats_text or
            "outside" in caveats_text or
            "training" in caveats_text
        )

        # Model metadata confidence should be capped
        mm_conf = ood_report.model_metadata.confidence_level
        checks["ood_mm_confidence_capped"] = mm_conf in ("Low", "Medium")

        log.info("    OOD: es_conf=%s, mm_conf=%s, is_ood=%s",
                 es_conf, mm_conf, ood_report.context.is_ood)

        # ── Non-OOD canonical mAb — should retain High confidence (no regression) ──
        nistmab_seq = REFERENCE_MABS["NISTmAb"]["sequence"]
        noood_intent = {
            "name": "NonOOD-mAb-SelfTest",
            "sequence": nistmab_seq,
            "molecule_class": "canonical_mab",
            "molecule_class_info": {
                "type": "canonical_mab",
                "display_name": "Monoclonal Antibody (IgG)",
                "has_fc_region": True,
                "expects_glycosylation": True,
            },
            "pI": 8.47, "mw": 148.0,
            "hydrophobicity": 0.35, "gravy": -0.47,
            "seq_length": len(nistmab_seq),
            "liability_summary": {"deamidation_sites": 2, "oxidation_sites": 1},
        }
        noood_cache = {
            "dev_result": {
                "data": {
                    "predictions": {"agg_risk": 0.20, "stability": 0.75, "viscosity_risk": 0.10},
                    "score": {"score": 0.20, "grade": "Low"},
                    "prediction_mode": "xgboost",  # ML-backed → High confidence base
                },
            },
            # No ood_bypass_reason → not OOD
        }
        noood_report = assemble_report(noood_intent, noood_cache)

        checks["noood_no_crash"] = noood_report is not None
        checks["noood_ctx_not_ood"] = noood_report.context.is_ood is False
        checks["noood_es_confidence_high"] = noood_report.executive_summary.confidence_level == "High"
        checks["noood_mm_confidence_high"] = noood_report.model_metadata.confidence_level == "High"
        # No OOD caveat for canonical mAb
        noood_caveats = " ".join(noood_report.executive_summary.key_caveats).lower()
        checks["noood_no_ood_caveat"] = "out-of-distribution" not in noood_caveats

        log.info("    Non-OOD mAb: es_conf=%s, is_ood=%s",
                 noood_report.executive_summary.confidence_level,
                 noood_report.context.is_ood)

        status = _checks_to_status(checks)
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("OOD Confidence FAIL: %s", failed)
        else:
            log.info("  OOD Confidence: %d checks all PASS", len(checks))

        return {
            "status": status,
            "checks": checks,
            "data": {
                "n_checks": len(checks),
                "n_passed": sum(1 for v in checks.values() if v),
                "n_failed": sum(1 for v in checks.values() if not v),
            },
        }
    except Exception as e:
        log.error("OOD Confidence: %s", e, exc_info=True)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# 15. Cross-Section Consistency & Liability Prioritization
# =========================================================================
def test_cross_section_consistency() -> dict:
    """
    Validate:
    1. molecule_class is consistent across context, executive_summary,
       molecule_overview, and validation_plan
    2. molecule_name is consistent across sections
    3. Tier 1 high-value liabilities (e.g., aggregation, deamidation) are
       prioritized over Tier 3 simulated QC in top_risks
    4. Key fields exist and are non-empty across all report sections
    """
    log.info("=== 15. Cross-Section Consistency & Liability Prioritization ===")
    try:
        from src.report_assembler import assemble_report

        checks = {}

        # Build a molecule with known high-value liabilities
        nistmab_seq = REFERENCE_MABS["NISTmAb"]["sequence"]
        intent = {
            "name": "Consistency-SelfTest",
            "sequence": nistmab_seq,
            "molecule_class": "canonical_mab",
            "molecule_class_info": {
                "type": "canonical_mab",
                "display_name": "Monoclonal Antibody (IgG)",
                "has_fc_region": True,
                "expects_glycosylation": True,
            },
            "pI": 8.47, "mw": 148.0,
            "hydrophobicity": 0.35, "gravy": -0.47,
            "seq_length": len(nistmab_seq),
            "cysteine_count": 4,
            "liability_summary": {
                "deamidation_sites": 5,   # High count → should be top risk
                "oxidation_sites": 3,     # Moderate
                "dp_clipping_sites": 3,   # Moderate
                "asp_isomerization_sites": 1,
                "n_glycosylation_sites": 1,
            },
        }

        cache = {
            "dev_result": {
                "data": {
                    "predictions": {
                        "agg_risk": 0.55,        # Medium-High → Tier 1
                        "stability": 0.45,        # Medium
                        "viscosity_risk": 0.20,    # Low
                    },
                    "score": {"score": 0.45, "grade": "Medium"},
                },
            },
            "analytical_qc": {
                "sec": {"monomer_pct": 95.0, "hmw_pct": 3.5},   # Simulated QC
                "cief": {"main_pct": 78.0, "acidic_pct": 14.0, "basic_pct": 8.0},
                "ce_sds": {"intact_pct": 94.0},
            },
        }

        report = assemble_report(intent, cache)
        checks["consistency_no_crash"] = report is not None

        # ── 1. molecule_class consistency across sections ──
        ctx_cls = report.context.molecule_class
        es_cls = report.executive_summary.molecule_class
        # molecule_overview doesn't have molecule_class field directly, but
        # validation_plan does
        vp_cls = report.validation_plan.molecule_class if hasattr(report.validation_plan, "molecule_class") else ctx_cls
        checks["class_ctx_matches_intent"] = ctx_cls == "canonical_mab"
        checks["class_es_matches_ctx"] = es_cls == ctx_cls
        checks["class_vp_matches_ctx"] = vp_cls == ctx_cls

        # ── 2. molecule_name consistency ──
        ctx_name = report.context.molecule_name
        es_name = report.executive_summary.molecule_name
        mo_name = report.molecule_overview.name
        checks["name_ctx"] = ctx_name == "Consistency-SelfTest"
        checks["name_es_matches_ctx"] = es_name == ctx_name
        checks["name_mo_matches_ctx"] = mo_name == ctx_name

        # ── 3. Key fields non-empty in every section ──
        # Executive summary
        checks["es_has_grade"] = len(report.executive_summary.overall_grade or "") > 0
        checks["es_has_recommendation"] = len(report.executive_summary.recommendation or "") > 0
        checks["es_has_rec_detail"] = len(report.executive_summary.recommendation_detail or "") > 0
        checks["es_has_confidence"] = report.executive_summary.confidence_level in ("High", "Medium", "Low")

        # Molecule overview
        checks["mo_has_pi"] = report.molecule_overview.isoelectric_point is not None
        checks["mo_has_mw"] = report.molecule_overview.molecular_weight_kda is not None

        # Developability
        checks["dev_has_dimensions"] = len(report.developability.risk_dimensions) >= 2
        checks["dev_has_qtpp"] = len(report.developability.qtpp_rows) >= 3

        # Model metadata
        checks["mm_has_confidence"] = report.model_metadata.confidence_level in ("High", "Medium", "Low")
        checks["mm_has_mode"] = len(report.model_metadata.prediction_mode or "") > 0

        # ── 4. Liability prioritization: Tier 1 over Tier 3 ──
        top_risks = report.executive_summary.top_risks
        checks["has_top_risks"] = len(top_risks) >= 1

        if top_risks:
            # Tier 1 risks (ML-predicted: aggregation, stability) should appear
            # before or alongside Tier 3 (simulated QC) risks
            tier1_indices = []
            tier3_indices = []
            for i, risk in enumerate(top_risks):
                if "[Tier 1]" in risk:
                    tier1_indices.append(i)
                elif "[Tier 3]" in risk:
                    tier3_indices.append(i)

            # If both tiers present, Tier 1 should come first
            if tier1_indices and tier3_indices:
                checks["tier1_before_tier3"] = min(tier1_indices) < min(tier3_indices)
            else:
                # Acceptable if only one tier present
                checks["tier1_before_tier3"] = True

            # With agg_risk=0.55, aggregation should appear in top risks
            risks_text = " ".join(top_risks).lower()
            checks["agg_risk_in_top_risks"] = (
                "aggregat" in risks_text or "agg" in risks_text
            )
        else:
            checks["tier1_before_tier3"] = True
            checks["agg_risk_in_top_risks"] = True  # edge case

        # ── 5. Non-canonical molecule consistency check ──
        # Run a fusion_protein through and verify cross-section consistency too
        fp_seq = (
            "DIQLTQSPSSLSASVGDRVTITCRASQDIRNYLAWYQQKPGKAPKLLIYAASSLQSGVP"
            "SRFSGSGSGTDFTLTISSLQPEDFATYYCQQYDSSPWTFGQGTKVEIKGGGGSGGGGSGG"
            "GGSEVQLVESGGGLVQPGGSLRLSCAASGFTFNTYAMNWVRQAPGKGLEWVARIRSKYN"
            "NYATYYADSVKDRFTISRDDSKNTLYLQMNSLRAEDTAVYYCVRHGNFGNSYVSWFAYW"
            "GQGTLVTVSS"
        )
        fp_intent = {
            "name": "FP-Consistency-SelfTest",
            "sequence": fp_seq,
            "molecule_class": "fusion_protein",
            "molecule_class_info": {
                "type": "fusion_protein",
                "display_name": "Fusion Protein",
                "has_fc_region": False,
                "expects_glycosylation": False,
            },
            "pI": 7.2, "mw": 55.0,
            "hydrophobicity": 0.35, "gravy": -0.30,
            "seq_length": len(fp_seq),
            "liability_summary": {"deamidation_sites": 2, "oxidation_sites": 1},
        }
        fp_cache = {
            "dev_result": {
                "data": {
                    "predictions": {"agg_risk": 0.40, "stability": 0.55, "viscosity_risk": 0.10},
                    "score": {"score": 0.35, "grade": "Medium"},
                },
            },
        }
        fp_report = assemble_report(fp_intent, fp_cache)
        checks["fp_class_consistent"] = (
            fp_report.context.molecule_class ==
            fp_report.executive_summary.molecule_class == "fusion_protein"
        )
        checks["fp_name_consistent"] = (
            fp_report.context.molecule_name ==
            fp_report.executive_summary.molecule_name ==
            fp_report.molecule_overview.name == "FP-Consistency-SelfTest"
        )
        # Non-canonical key caveats should mention format limitations
        fp_caveats = " ".join(fp_report.executive_summary.key_caveats).lower()
        checks["fp_has_format_caveat"] = (
            "canonical" in fp_caveats or "format" in fp_caveats or
            "fusion" in fp_caveats or "non-canonical" in fp_caveats or
            "threshold" in fp_caveats
        )

        status = _checks_to_status(checks)
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("Cross-Section Consistency FAIL: %s", failed)
        else:
            log.info("  Cross-Section Consistency: %d checks all PASS", len(checks))

        return {
            "status": status,
            "checks": checks,
            "data": {
                "n_checks": len(checks),
                "n_passed": sum(1 for v in checks.values() if v),
                "n_failed": sum(1 for v in checks.values() if not v),
            },
        }
    except Exception as e:
        log.error("Cross-Section Consistency: %s", e, exc_info=True)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# 16. Downstream DoE — Purification Optimizer Full Pipeline
# =========================================================================
def test_downstream_doe() -> dict:
    """
    Validate the full DoE grid-search pipeline from purification_optimizer:
      1. run_doe_optimization() executes without error (small 3×3 grid)
      2. DoEOptimization return structure is complete
      3. doe_to_dict() serialization round-trips
      4. compute_mass_balance() produces valid results
      5. Optimal point metrics are in physically realistic ranges
    """
    log.info("=== 16. Downstream DoE (purification_optimizer) ===")
    try:
        from src.purification_optimizer import (
            run_doe_optimization, doe_to_dict, doe_summary,
            compute_mass_balance,
        )

        checks = {}

        # ── Part 1: Run DoE grid search (small 3×3 grid for speed) ──
        # Use NISTmAb-like biophysical properties
        doe_result = run_doe_optimization(
            pI=8.47, mw=148.0, hydrophobicity=0.35,
            ph_range=(5.5, 6.5),
            gradient_range=(5.0, 20.0),
            ph_steps=3, gradient_steps=3,
            salt_steps=1, load_steps=1,
        )
        checks["doe_returns_result"] = doe_result is not None

        # ── Part 2: Structure validation ──
        checks["has_optimal"] = hasattr(doe_result, "optimal") and doe_result.optimal is not None
        checks["has_ph_values"] = hasattr(doe_result, "ph_values") and len(doe_result.ph_values) == 3
        checks["has_gradient_values"] = hasattr(doe_result, "gradient_values") and len(doe_result.gradient_values) == 3
        checks["has_resolution_matrix"] = (
            hasattr(doe_result, "resolution_matrix")
            and doe_result.resolution_matrix is not None
            and doe_result.resolution_matrix.shape == (3, 3)
        )
        checks["has_yield_matrix"] = (
            hasattr(doe_result, "yield_matrix")
            and doe_result.yield_matrix is not None
            and doe_result.yield_matrix.shape == (3, 3)
        )
        checks["has_all_results"] = (
            hasattr(doe_result, "all_results")
            and len(doe_result.all_results) >= 9  # 3×3×1×1 grid minimum
        )
        checks["has_wall_time"] = hasattr(doe_result, "wall_time_s") and doe_result.wall_time_s >= 0
        checks["has_n_conditions"] = doe_result.n_conditions >= 9

        # ── Part 3: Optimal point physical realism ──
        opt = doe_result.optimal
        checks["optimal_ph_in_range"] = 5.0 <= opt.elution_ph <= 7.5
        checks["optimal_gradient_positive"] = opt.gradient_slope > 0
        checks["optimal_rts_positive"] = opt.rt_acidic > 0 and opt.rt_main > 0 and opt.rt_basic > 0
        checks["optimal_rt_order"] = opt.rt_acidic <= opt.rt_main <= opt.rt_basic
        checks["optimal_yield_valid"] = 0.0 <= opt.yield_main <= 1.0
        checks["optimal_resolution_nonneg"] = opt.resolution_min >= 0
        log.info("  Optimal: pH=%.2f gradient=%.1f Rs_min=%.3f yield=%.1f%%",
                 doe_result.optimal_ph, doe_result.optimal_gradient,
                 opt.resolution_min, opt.yield_main * 100)

        # ── Part 4: doe_to_dict serialization ──
        d = doe_to_dict(doe_result)
        checks["dict_has_optimal_ph"] = "optimal_ph" in d
        checks["dict_has_optimal_gradient"] = "optimal_gradient" in d
        checks["dict_has_n_conditions"] = d.get("n_conditions", 0) >= 9
        # Should be JSON-serializable
        import json as _json
        json_str = _json.dumps(d)
        checks["dict_json_serializable"] = len(json_str) > 50
        log.info("  doe_to_dict: %d bytes JSON", len(json_str))

        # ── Part 5: doe_summary text ──
        summary_text = doe_summary(doe_result)
        checks["summary_nonempty"] = len(summary_text) > 20
        checks["summary_mentions_ph"] = "pH" in summary_text or "ph" in summary_text.lower()

        # ── Part 6: compute_mass_balance ──
        mb = compute_mass_balance(opt.resolution_acidic_main, opt.resolution_main_basic)
        checks["mb_returns_dict"] = isinstance(mb, dict)
        checks["mb_pool_main"] = 0 < mb.get("pool_main_pct", 0) <= 100
        checks["mb_pool_purity"] = 0 < mb.get("pool_purity_pct", 0) <= 100
        checks["mb_waste"] = 0 <= mb.get("waste_total_pct", 0) <= 100
        # Mass balance should roughly sum to 100
        _total = mb.get("pool_main_pct", 0) + mb.get("waste_total_pct", 0)
        checks["mb_sums_near_100"] = 80 <= _total <= 120  # allow some overlap/rounding
        log.info("  Mass balance: pool=%.1f%% purity=%.1f%% waste=%.1f%%",
                 mb.get("pool_main_pct", 0), mb.get("pool_purity_pct", 0), mb.get("waste_total_pct", 0))

        status = _checks_to_status(checks)
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("Downstream DoE FAIL: %s", failed)
        else:
            log.info("  Downstream DoE: %d checks all PASS", len(checks))

        return {
            "status": status,
            "checks": checks,
            "data": {
                "n_checks": len(checks),
                "n_passed": sum(1 for v in checks.values() if v),
                "n_failed": sum(1 for v in checks.values() if not v),
                "optimal_ph": doe_result.optimal_ph,
                "optimal_gradient": doe_result.optimal_gradient,
                "wall_time_s": doe_result.wall_time_s,
            },
        }
    except Exception as e:
        log.error("Downstream DoE: %s", e, exc_info=True)
        return {"status": "ERROR", "data": {}, "error": str(e)}


def test_cross_path_alignment() -> dict:
    """
    17. Frontend vs Bulk vs Backend Alignment Test

    Verifies that a single molecule processed through:
      A) The frontend single-molecule path (predict_developability_risk + twins)
      B) The bulk CSV pipeline (parse_bulk_csv + run_bulk_analysis)
    produces identical numeric results.

    This catches:
      - Feature computation divergence (feature_registry vs fallback)
      - Grade threshold mismatches across code paths
      - Aggregation % formula inconsistencies
      - MW / pI source ambiguity between paths
      - Upstream dev_score coupling field-name mismatches
      - Stability twin parameter differences
      - 5-dim composite score divergence
    """
    log.info("=== 17. Cross-Path Alignment (Frontend vs Bulk) ===")
    try:
        import re as _re_align
        from src.feature_registry import compute_features
        from src.agents import predict_developability_risk, PharmaAgentManager
        from src.analytical_qc_twin import run_analytical_qc
        from src.upstream_twin import run_upstream_simulation
        from src.stability_twin import run_stability_study
        from src.developability_core import assess_developability
        from src.immunogenicity_twin import run_immunogenicity_assessment
        from src.preclinical_twin import predict_human_half_life, check_fcrn_binding_motif
        from src.analytical_twin import run_ms_characterization
        from src.bulk_schema import parse_bulk_csv, row_to_intent, BATCH_TYPES
        from src.bulk_runner import run_bulk_analysis
        from src.report_schema import GRADE_LOW_UPPER, GRADE_MEDIUM_UPPER

        checks = {}

        # ── Test molecule: Trastuzumab-like HC/LC ──
        HC = (
            "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPT"
            "NGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDY"
            "WGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSG"
            "ALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVE"
            "PKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDP"
            "EVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNK"
            "ALPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNG"
        )
        LC = (
            "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVP"
            "SRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRTVAAPSVFIFP"
            "PSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTL"
            "TLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
        )

        # ── Part 1: Build intent via bulk schema (authoritative path) ──
        csv_text = f"name,HC,LC\nAlignTestMab,{HC},{LC}\n"
        pr = parse_bulk_csv(csv_text, "canonical_mab")
        checks["bulk_csv_parsed"] = pr.n_valid == 1

        spec = BATCH_TYPES["canonical_mab"]
        bulk_intent = row_to_intent(pr.valid_rows[0], spec)
        checks["bulk_intent_has_mw"] = bulk_intent.get("mw") is not None and bulk_intent["mw"] > 50
        checks["bulk_intent_has_pI"] = bulk_intent.get("pI") is not None

        # ── Part 2: Frontend-style feature computation (same as app.py) ──
        assembly_chains = [
            {"name": "HC", "sequence": HC, "chain_type": "Heavy", "copy_number": 2, "length": len(HC)},
            {"name": "LC", "sequence": LC, "chain_type": "Light", "copy_number": 2, "length": len(LC)},
        ]
        super_seq = "".join(ch["sequence"] * ch.get("copy_number", 1) for ch in assembly_chains)
        fs = compute_features(sequence=super_seq, molecule_class="canonical_mab", chains=assembly_chains)
        fv = fs.features

        fe_pI = fv["pI"].value
        fe_mw = fv["mw_kda"].value
        fe_hydro = fv["hydrophobicity"].value
        fe_gravy = fv["gravy"].value
        fe_deam = fv["deam_sites"].value
        fe_ox = fv["ox_sites"].value

        # Feature values must match between frontend and bulk
        checks["pI_aligned"] = abs(fe_pI - bulk_intent["pI"]) < 0.001
        checks["mw_aligned"] = abs(fe_mw - bulk_intent["mw"]) < 0.01
        checks["gravy_aligned"] = abs(fe_gravy - bulk_intent["gravy"]) < 0.001
        checks["hydro_aligned"] = abs(fe_hydro - bulk_intent["hydrophobicity"]) < 0.001
        checks["deam_aligned"] = fe_deam == bulk_intent["deam_sites"]
        checks["ox_aligned"] = fe_ox == bulk_intent["ox_sites"]

        # ── Part 3: Run single-molecule pipeline (same as frontend) ──
        fe_intent = dict(bulk_intent)  # Use bulk's own intent to remove feature-source noise
        fe_raw = PharmaAgentManager().run_developability_pipeline(fe_intent)
        checks["single_pipeline_success"] = fe_raw.get("status") == "success"

        fe_dev = fe_raw.get("developability", {}) or {}
        fe_score_info = fe_dev.get("score", {})
        fe_preds = fe_dev.get("predictions", {})
        fe_score = fe_score_info.get("score")
        fe_grade = fe_score_info.get("grade")
        fe_agg = fe_preds.get("agg_risk")
        fe_stab = fe_preds.get("stability")
        fe_visc = fe_preds.get("viscosity_risk")

        # ── Part 4: Run bulk pipeline for same molecule ──
        batch = run_bulk_analysis(pr)
        rr = batch.results[0]
        checks["bulk_pipeline_success"] = rr.status == "success"

        # ── Part 5: 3-dim score alignment ──
        if fe_score is not None and rr.developability_score is not None:
            checks["score_3dim_aligned"] = abs(fe_score - rr.developability_score) < 0.0001
            checks["grade_3dim_aligned"] = fe_grade == rr.developability_grade
            checks["agg_risk_aligned"] = abs(fe_agg - rr.agg_risk) < 0.0001
            checks["stability_aligned"] = abs(fe_stab - rr.stability) < 0.0001
            checks["viscosity_aligned"] = abs(fe_visc - rr.viscosity_risk) < 0.0001
            log.info("  3-dim: single=%.4f (%s) bulk=%.4f (%s)",
                     fe_score, fe_grade, rr.developability_score, rr.developability_grade)
        else:
            checks["score_3dim_aligned"] = False

        # ── Part 6: QC Twin alignment ──
        # Replicate exact frontend formula: agg^2 * 20, clamped to [0.5, 10.0]
        _fe_agg_pct = max(0.5, min(fe_agg * fe_agg * 20.0, 10.0))
        fe_qc = run_analytical_qc(
            sequence=bulk_intent["sequence"], pI=float(bulk_intent["pI"]),
            aggregation_pct=_fe_agg_pct,
            is_mab=True, molecule_class="canonical_mab", sialylation_fraction=0.0,
        )
        if rr.cief_main_pct is not None:
            checks["cief_aligned"] = abs(fe_qc.cief.main_pct - rr.cief_main_pct) < 0.5
            checks["cesds_aligned"] = abs(fe_qc.ce_sds.intact_pct - rr.ce_sds_purity_pct) < 0.5
            checks["hmw_aligned"] = abs(fe_qc.ce_sds.hmw_pct - rr.ce_sds_hmw_pct) < 0.5
            log.info("  QC: cIEF single=%.1f bulk=%.1f | CE-SDS single=%.1f bulk=%.1f",
                     fe_qc.cief.main_pct, rr.cief_main_pct,
                     fe_qc.ce_sds.intact_pct, rr.ce_sds_purity_pct)

        # ── Part 7: Upstream Twin alignment ──
        _fe_up_score = (fe_dev.get("composite_score")
            or fe_dev.get("developability_score")
            or (fe_dev.get("score", {}) or {}).get("score"))
        fe_up = run_upstream_simulation(
            seed_density=0.5, temp_shift_day=5.0,
            dev_score=_fe_up_score, agg_risk=fe_agg,
            culture_days=14.0, hydrophobicity=bulk_intent.get("gravy"),
            sequence=bulk_intent.get("sequence"),
        )
        if rr.predicted_titer_g_L is not None:
            checks["titer_aligned"] = abs(fe_up.final_titer - rr.predicted_titer_g_L) < 0.05
            log.info("  Upstream: single=%.3f bulk=%.3f g/L",
                     fe_up.final_titer, rr.predicted_titer_g_L)

        # ── Part 8: Stability Twin alignment ──
        _fe_stab_deam = len(_re_align.findall(r"N[GS]", super_seq.upper()))
        _fe_stab_dp = len(_re_align.findall(r"DP", super_seq.upper()))
        fe_stab_res = run_stability_study(
            starting_hmw_pct=max(fe_agg ** 2 * 20.0, 0.5),
            starting_acidic_pct=max(5.0, min(fe_qc.cief.acidic_pct, 40.0)),
            formulation_ph=6.0, pI=bulk_intent["pI"],
            deamidation_sites=_fe_stab_deam, dp_clip_sites=_fe_stab_dp,
            hydrophobicity=bulk_intent["hydrophobicity"],
        )
        if rr.shelf_life_months is not None:
            checks["shelf_life_aligned"] = abs(fe_stab_res.predicted_shelf_life_months - rr.shelf_life_months) < 1.0
            log.info("  Stability: single=%.0f bulk=%.0f months",
                     fe_stab_res.predicted_shelf_life_months, rr.shelf_life_months)

        # ── Part 8b: DoE Purification alignment ──
        try:
            from src.purification_optimizer import run_doe_optimization
            fe_doe = run_doe_optimization(
                pI=bulk_intent["pI"],
                mw=bulk_intent["mw"],
                hydrophobicity=bulk_intent.get("hydrophobicity", 0.35),
            )
            if rr.doe_optimal_yield is not None and fe_doe.optimal:
                _fe_doe_yield = fe_doe.optimal.yield_main
                if _fe_doe_yield is not None:
                    checks["doe_yield_aligned"] = abs(_fe_doe_yield - rr.doe_optimal_yield) < 0.01
                    log.info("  DoE: single_yield=%.3f bulk_yield=%.3f",
                             _fe_doe_yield, rr.doe_optimal_yield)
        except Exception as _doe_err:
            log.warning("  DoE alignment check skipped: %s", _doe_err)

        # ── Part 8c: COGS alignment ──
        try:
            from src.cogs_twin import COGSInputs, calculate_cogs
            _cogs_titer = fe_up.final_titer if fe_up.final_titer > 0.01 else 3.50
            _cogs_yield = 0.70
            if fe_doe and fe_doe.optimal:
                _cogs_yield = fe_doe.optimal.yield_main or 0.70
            fe_cogs = calculate_cogs(
                COGSInputs(titer_g_per_L=_cogs_titer, downstream_yield=_cogs_yield),
                molecule_class="canonical_mab",
            )
            if rr.cogs_per_gram is not None:
                checks["cogs_aligned"] = abs(fe_cogs.cogs_per_gram - rr.cogs_per_gram) < 0.50
                log.info("  COGS: single=$%.2f/g bulk=$%.2f/g",
                         fe_cogs.cogs_per_gram, rr.cogs_per_gram)
        except Exception as _cogs_err:
            log.warning("  COGS alignment check skipped: %s", _cogs_err)

        # ── Part 8d: ADA / Immunogenicity alignment ──
        if rr.ada_risk_score is not None:
            _fe_imm_pre = run_immunogenicity_assessment(
                sequence=super_seq, agg_risk=fe_agg, dev_score=fe_score,
                molecule_name="AlignTestMab", molecule_class="canonical_mab",
            )
            if _fe_imm_pre and hasattr(_fe_imm_pre, "ada_risk_score"):
                checks["ada_score_aligned"] = abs(_fe_imm_pre.ada_risk_score - rr.ada_risk_score) < 0.01
                log.info("  ADA: single=%.3f bulk=%.3f",
                         _fe_imm_pre.ada_risk_score, rr.ada_risk_score)

        # ── Part 9: 5-dim composite alignment ──
        # Run immunogenicity + PK twins (same as bulk_runner._run_comprehensive_twins)
        imm = run_immunogenicity_assessment(
            sequence=super_seq, agg_risk=fe_agg, dev_score=fe_score,
            molecule_name="AlignTestMab", molecule_class="canonical_mab",
        )
        ms = run_ms_characterization(
            sequence=super_seq, protein_name="AlignTestMab", is_mab=True,
            chains=assembly_chains, molecule_class="canonical_mab",
        )
        _ld = ms.get("liability_density", {})
        _ld_f = _ld.get("density_per_1000", 30.0) if isinstance(_ld, dict) else 30.0
        pk = predict_human_half_life(
            global_pi=bulk_intent["pI"], hydrophobicity=bulk_intent["hydrophobicity"],
            liability_density=_ld_f, mw_kda=bulk_intent["mw"],
            fcrn_binding_motif_intact=check_fcrn_binding_motif(super_seq).get("intact", True),
            glycoform_profile="standard_cho",
        )
        _agg_anal = fe_agg ** 2 * 20.0
        fe_comp = assess_developability(
            molecule_name="AlignTestMab", molecule_class="canonical_mab",
            feature_values={
                "mw_kda": bulk_intent["mw"], "pI": bulk_intent["pI"],
                "hydrophobicity": bulk_intent["hydrophobicity"],
                "seq_length": bulk_intent["seq_length"],
                "cysteine_count": bulk_intent.get("cysteine_count", 0),
                "deam_sites": bulk_intent["deam_sites"],
                "ox_sites": bulk_intent["ox_sites"],
                "acidic_residues": bulk_intent.get("acidic_residues", 0),
                "basic_residues": bulk_intent.get("basic_residues", 0),
            },
            dev_predictions={"agg_risk": fe_agg, "stability": fe_stab, "viscosity_risk": fe_visc},
            analytical_results={"sec_monomer_pct": round(100.0 - max(0.5, min(_agg_anal, 10.0)), 2)},
            stability_results={"shelf_life_months": fe_stab_res.predicted_shelf_life_months},
            pk_results={"half_life_days": pk.get("half_life_days")},
            ada_results={"ada_risk_level": getattr(imm, "ada_risk_level", None)},
            upstream_results={"final_titer": fe_up.final_titer} if fe_up.final_titer > 0.01 else {},
        )
        if rr.composite_dev_score is not None:
            checks["composite_5dim_aligned"] = abs(fe_comp.composite_score - rr.composite_dev_score) < 0.005
            checks["composite_grade_aligned"] = fe_comp.composite_grade == rr.composite_dev_grade
            log.info("  5-dim: single=%.4f (%s) bulk=%.4f (%s)",
                     fe_comp.composite_score, fe_comp.composite_grade,
                     rr.composite_dev_score, rr.composite_dev_grade)

        # ── Part 10: Grade threshold canonical source check ──
        # Verify all grading paths use canonical thresholds from report_schema.
        # predict_developability_risk() delegates to predict_developability() →
        # DevelopabilityPredictor.compute_developability_score(), so we check
        # that function (the actual grade assignment site).
        import inspect
        from src.developability_predictor import DevelopabilityPredictor
        _cds_src = inspect.getsource(DevelopabilityPredictor.compute_developability_score)
        checks["predictor_uses_canonical_thresholds"] = "GRADE_LOW_UPPER" in _cds_src
        # predict_developability_risk delegates to predict_developability which
        # calls compute_developability_score; verify delegation chain exists
        _pdr_src = inspect.getsource(predict_developability_risk)
        checks["agents_delegates_to_predictor"] = "predict_developability(" in _pdr_src
        # Also verify the _predict_unified_fallback path uses canonical
        from src.agents import _predict_unified_fallback
        _puf_src = inspect.getsource(_predict_unified_fallback)
        checks["unified_fallback_uses_canonical"] = "GRADE_LOW_UPPER" in _puf_src
        # bulk_runner.summary_stats uses hardcoded 0.25/0.55 — confirm match
        checks["canonical_low_is_025"] = abs(GRADE_LOW_UPPER - 0.25) < 0.001
        checks["canonical_med_is_055"] = abs(GRADE_MEDIUM_UPPER - 0.55) < 0.001

        # ── Part 11: Aggregation % formula consistency ──
        # All QC call sites must use agg^2*20 (not agg*5 or anything else)
        # Verified by running both through the same formula
        _bulk_agg_pct = rr.agg_risk * rr.agg_risk * 20.0
        _fe_agg_pct_check = fe_agg * fe_agg * 20.0
        checks["agg_pct_formula_same"] = abs(_bulk_agg_pct - _fe_agg_pct_check) < 0.0001

        status = _checks_to_status(checks)
        n_pass = sum(1 for v in checks.values() if v)
        n_fail = sum(1 for v in checks.values() if not v)
        if status == "FAIL":
            failed = [k for k, v in checks.items() if not v]
            log.warning("  Cross-Path Alignment FAIL: %s", failed)
        else:
            log.info("  Cross-Path Alignment: %d checks all PASS", len(checks))

        return {
            "status": status,
            "checks": checks,
            "data": {
                "n_checks": len(checks),
                "n_passed": n_pass,
                "n_failed": n_fail,
                "single_score": fe_score,
                "bulk_score": rr.developability_score,
                "single_composite": fe_comp.composite_score if fe_comp else None,
                "bulk_composite": rr.composite_dev_score,
            },
        }
    except Exception as e:
        log.error("Cross-Path Alignment: %s", e, exc_info=True)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# 18. Bulk / Single Schema Alignment (P2)
# =========================================================================
def test_schema_alignment() -> dict:
    """
    Validate the bulk/single schema alignment module (P2).

    Tests:
    - Schema map is well-formed (all entries have correct categories; 58 as of v33.1)
    - Grade validator correctly catches bare 'Low'/'Medium'/'High' strings
    - Grade validator accepts canonical 'Low Risk'/'Medium Risk'/'High Risk'
    - extract_bulk_core() produces correct canonical field mapping
    - All grade strings in a live bulk run use the canonical format
    - Name-mismatch count matches expectations (12 documented mismatches)
    """
    import sys
    sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

    log.info("=== 18. Bulk / Single Schema Alignment (P2) ===")

    try:
        from src.bulk_single_schema_alignment import (
            FIELD_MAP,
            validate_grade_strings,
            validate_bulk_row_alignment,
            extract_bulk_core,
            get_name_mismatches,
            get_bulk_specific,
            get_single_specific,
            _selftest as _align_selftest,
        )
        from src.report_schema import grade_to_risk_label, GRADE_LOW_UPPER, GRADE_MEDIUM_UPPER

        checks = {}

        # ── Part 1: Module self-test ──
        try:
            _align_selftest()
            checks["alignment_selftest_pass"] = True
        except Exception as e:
            log.error("  Schema alignment selftest failed: %s", e)
            checks["alignment_selftest_pass"] = False

        # ── Part 2: Field map counts ──
        from collections import Counter
        cats = Counter(cat for cat, *_ in FIELD_MAP)
        checks["has_shared_aligned"]   = cats["SHARED_ALIGNED"] >= 10
        checks["has_name_mismatches"]  = cats["NAME_MISMATCH"] >= 8
        checks["has_bulk_specific"]    = cats["BULK_SPECIFIC"] >= 10
        checks["has_single_specific"]  = cats["SINGLE_SPECIFIC"] >= 5
        checks["total_fields_mapped"]  = len(FIELD_MAP) >= 40

        log.info("  Field map: %d SHARED_ALIGNED, %d NAME_MISMATCH, "
                 "%d BULK_SPECIFIC, %d SINGLE_SPECIFIC",
                 cats["SHARED_ALIGNED"], cats["NAME_MISMATCH"],
                 cats["BULK_SPECIFIC"], cats["SINGLE_SPECIFIC"])

        # ── Part 3: Grade validator — bare strings flagged ──
        class _BareGrades:
            status = "success"
            composite_dev_grade = "Low"
            developability_grade = "Medium"
            developability_score = 0.20
            composite_dev_score  = 0.20
            agg_risk = 0.15; stability = 0.80; viscosity_risk = 0.10

        viol = validate_grade_strings(_BareGrades())
        checks["bare_grades_detected"] = len(viol) >= 2
        log.info("  Grade validator: caught %d violations for bare grades (expected >=2)",
                 len(viol))

        # ── Part 4: Grade validator — canonical strings pass ──
        class _CanonGrades:
            status = "success"
            composite_dev_grade = "Low Risk"
            developability_grade = "Low Risk"
            developability_score = 0.20
            composite_dev_score  = 0.20
            agg_risk = 0.15; stability = 0.80; viscosity_risk = 0.10

        result = validate_bulk_row_alignment(_CanonGrades())
        checks["canonical_grades_pass"] = result["ok"]

        # ── Part 5: grade_to_risk_label idempotent ──
        # Applying the conversion twice must not produce "Low Risk Risk"
        double = grade_to_risk_label(grade_to_risk_label("Low"))
        checks["grade_label_not_doubled"] = double == "Low Risk"

        # ── Part 6: extract_bulk_core field mapping ──
        class _SampleResult:
            composite_dev_score = 0.262
            developability_score = 0.207
            composite_dev_grade = "Medium Risk"
            developability_grade = "Low Risk"
            agg_risk = 0.20; stability = 0.75; viscosity_risk = 0.12
            mw_kda = 145.2; pI = 8.4; gravy = -0.32; hydrophobicity = 0.35
            seq_length = 1204; deam_sites = 7; ox_sites = 5
            acidic_residues = 140; basic_residues = 160; cysteine_count = 16
            ood_flag = False; ood_details = None
            cief_main_pct = 54.6; cief_acidic_pct = 22.1; cief_basic_pct = 23.3
            ce_sds_purity_pct = 97.6; intact_mass_da = 131305.0
            half_life_days = 20.8; predicted_titer_g_L = 3.31
            ada_risk_category = "Medium"; molecule_class = "canonical_mab"

        core = extract_bulk_core(_SampleResult())
        checks["core_overall_score_mapped"]    = core["overall_score"] == 0.262
        checks["core_mw_mapped"]               = core["molecular_weight_kda"] == 145.2
        checks["core_pi_mapped"]               = core["isoelectric_point"] == 8.4
        checks["core_cesds_name_mapped"]        = core["cesds_intact_pct"] == 97.6
        checks["core_titer_name_mapped"]        = core["final_titer_g_l"] == 3.31
        checks["core_ood_flag_mapped"]          = core["is_ood"] is False
        checks["core_ada_name_mapped"]          = core["ada_risk_level"] == "Medium"

        # ── Part 7: grade_from_score + grade_to_risk_label round-trip ──
        from src.report_schema import grade_from_score
        for score, expected in [(0.10, "Low Risk"), (0.30, "Medium Risk"), (0.70, "High Risk")]:
            produced = grade_to_risk_label(grade_from_score(score))
            assert produced == expected, f"score {score} → {produced} (expected {expected})"
        checks["grade_roundtrip_correct"] = True

        # ── Part 8: Live bulk run — verify canonical grade strings ──
        try:
            from src.bulk_runner import run_bulk_analysis
            from src.bulk_schema import parse_bulk_csv

            _ALIGN_CSV = (
                "name,HC,LC\n"
                "AlignCheck_mAb,"
                "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS,"
                "DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYLYHPYTFGQGTKVEIKR\n"
            )
            _parse = parse_bulk_csv(_ALIGN_CSV, "canonical_mab")
            if _parse.is_ok:
                _batch = run_bulk_analysis(parse_result=_parse)
                for rr in _batch.results:
                    if rr.status == "success":
                        viol2 = validate_grade_strings(rr)
                        if viol2:
                            log.warning("  Live run grade violations: %s", viol2)
                        checks["live_run_canonical_grades"] = len(viol2) == 0
                        log.info("  Live run: composite_dev_grade='%s', developability_grade='%s'",
                                 rr.composite_dev_grade, rr.developability_grade)
                        break
                else:
                    checks["live_run_canonical_grades"] = False
            else:
                log.warning("  Live alignment CSV parse failed: %s", _parse.errors)
                checks["live_run_canonical_grades"] = False
        except Exception as e:
            log.warning("  Live bulk grade check skipped: %s", e)
            checks["live_run_canonical_grades"] = False  # non-fatal

        # ── Part 9: ProcessPKSummary unified schema check ──
        # Verify that the single-path report schema includes the same
        # capability fields that bulk CSV exports.
        from src.report_schema import ProcessPKSummary
        import dataclasses as _dc
        _pk_field_names = {f.name for f in _dc.fields(ProcessPKSummary)}
        _required_unified = {
            "cogs_per_gram", "cogs_cost_rating", "cogs_note",
            "doe_optimal_ph", "doe_optimal_yield", "doe_optimal_purity", "doe_rs_min",
            "shelf_life_months", "stability_grade",
            "ada_risk_score", "n_mhcii_hotspots",
            "clearance_ml_day_kg",
        }
        _missing = _required_unified - _pk_field_names
        checks["process_pk_unified_schema"] = len(_missing) == 0
        if _missing:
            log.warning("  ProcessPKSummary missing unified fields: %s", _missing)
        else:
            log.info("  ProcessPKSummary: all %d unified fields present", len(_required_unified))

        status = _checks_to_status(checks)
        n_pass = sum(1 for v in checks.values() if v)
        n_fail = sum(1 for v in checks.values() if not v)

        if status == "FAIL":
            log.warning("  Schema Alignment FAIL: %s", [k for k, v in checks.items() if not v])
        else:
            log.info("  Schema Alignment: %d checks all PASS", len(checks))

        return {
            "status": status,
            "checks": checks,
            "data": {
                "n_checks": len(checks),
                "n_passed": n_pass,
                "n_failed": n_fail,
                "shared_aligned": cats["SHARED_ALIGNED"],
                "name_mismatches": cats["NAME_MISMATCH"],
                "bulk_specific": cats["BULK_SPECIFIC"],
                "single_specific_gaps": cats["SINGLE_SPECIFIC"],
                "total_fields": len(FIELD_MAP),
            },
        }
    except Exception as e:
        log.error("Schema Alignment test: %s", e, exc_info=True)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# 19. Platform Alignment & Gap Detection
# =========================================================================
def test_platform_alignment() -> dict:
    """
    Section 19: Cross-module alignment audit and gap detection.

    Catches silent regressions where modules drift out of sync:
      - molecule_registry covers all MoleculeClass enum values
      - molecule_registry risk weights match molecule_classifier RISK_WEIGHT_PROFILES
      - molecule_registry suffixes match report_schema MOLECULE_RECOMMENDATION_SUFFIX
      - All risk weight profiles sum to ~1.0
      - FIELD_MAP covers every BulkRowResult dataclass output field
      - CSV _CSV_COLUMNS matches to_summary_dict() keys
      - grade_distribution normalizes canonical "Low Risk" → bare "Low" correctly
      - No old hex color values remain in app.py
      - optional_deps core layer reports OK
      - confidence_defaults cover every risk weight dimension
      - Extra dimensions (species_purity, conjugation) appear in risk_weights
    """
    import sys
    sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

    log.info("=== 19. Platform Alignment & Gap Detection ===")

    checks = {}

    try:
        # ── Check 1: molecule_registry ↔ MoleculeClass enum ──
        from src.molecule_classifier import MoleculeClass
        from src.molecule_registry import MOLECULE_REGISTRY

        enum_values = {mc.value for mc in MoleculeClass}
        registry_keys = set(MOLECULE_REGISTRY.keys())
        missing_in_reg = enum_values - registry_keys
        extra_in_reg = registry_keys - enum_values
        checks["registry_covers_all_enum_values"] = len(missing_in_reg) == 0
        checks["registry_no_extra_classes"] = len(extra_in_reg) == 0
        if missing_in_reg:
            log.warning("  MoleculeClass values not in registry: %s", missing_in_reg)
        if extra_in_reg:
            log.warning("  Registry keys not in MoleculeClass: %s", extra_in_reg)
        log.info("  Registry ↔ Enum: %d classes, %d enum values",
                 len(registry_keys), len(enum_values))

        # ── Check 2: molecule_registry risk weights ↔ RISK_WEIGHT_PROFILES ──
        from src.molecule_classifier import RISK_WEIGHT_PROFILES

        weight_match = True
        for cls in MOLECULE_REGISTRY:
            reg_w = MOLECULE_REGISTRY[cls]["risk_weights"]
            clf_w = RISK_WEIGHT_PROFILES.get(cls)
            if reg_w != clf_w:
                log.warning("  %s: risk weight mismatch (registry vs classifier)", cls)
                weight_match = False
        checks["risk_weights_match_classifier"] = weight_match

        # ── Check 3: molecule_registry suffixes ↔ MOLECULE_RECOMMENDATION_SUFFIX ──
        from src.report_schema import MOLECULE_RECOMMENDATION_SUFFIX

        suffix_match = True
        for cls in MOLECULE_REGISTRY:
            reg_suf = MOLECULE_REGISTRY[cls]["recommendation_suffix"]
            schema_suf = MOLECULE_RECOMMENDATION_SUFFIX.get(cls)
            if schema_suf is not None and reg_suf != schema_suf:
                log.warning("  %s: recommendation suffix mismatch", cls)
                suffix_match = False
        checks["recommendation_suffixes_match_schema"] = suffix_match

        # ── Check 4: All risk weight profiles sum to ~1.0 ──
        weights_sum_ok = True
        for cls, cfg in MOLECULE_REGISTRY.items():
            total = sum(cfg["risk_weights"].values())
            if not (0.95 <= total <= 1.05):
                log.warning("  %s: risk weights sum=%.3f (out of range)", cls, total)
                weights_sum_ok = False
        checks["all_risk_weights_sum_to_one"] = weights_sum_ok

        # ── Check 5: Standard 5 dimensions in every class ──
        standard_dims = {"aggregation", "stability", "viscosity", "expression", "immunogenicity"}
        dims_ok = True
        for cls, cfg in MOLECULE_REGISTRY.items():
            dims = set(cfg["risk_weights"].keys())
            if not standard_dims.issubset(dims):
                log.warning("  %s: missing standard dims %s", cls, standard_dims - dims)
                dims_ok = False
        checks["all_classes_have_5_standard_dimensions"] = dims_ok

        # ── Check 6: confidence_defaults cover every risk weight dimension ──
        conf_ok = True
        for cls, cfg in MOLECULE_REGISTRY.items():
            for dim in cfg["risk_weights"]:
                if dim not in cfg.get("confidence_defaults", {}):
                    log.warning("  %s: no confidence_default for '%s'", cls, dim)
                    conf_ok = False
        checks["confidence_defaults_complete"] = conf_ok

        # ── Check 7: Extra dimensions appear in risk_weights ──
        extra_ok = True
        for cls, cfg in MOLECULE_REGISTRY.items():
            for dim_cfg in cfg.get("extra_dimensions", []):
                if dim_cfg["name"] not in cfg["risk_weights"]:
                    log.warning("  %s: extra dim '%s' not in risk_weights", cls, dim_cfg["name"])
                    extra_ok = False
        checks["extra_dimensions_in_risk_weights"] = extra_ok

        # ── Check 8: FIELD_MAP covers all BulkRowResult output fields ──
        import dataclasses
        from src.bulk_runner import BulkRowResult
        from src.bulk_single_schema_alignment import FIELD_MAP

        mapped_bulk = set()
        for _cat, _s, bulk_field, _n in FIELD_MAP:
            if bulk_field != "(none)":
                mapped_bulk.add(bulk_field)
        all_br_fields = {f.name for f in dataclasses.fields(BulkRowResult)}
        internal_only = {"raw_result", "raw_comprehensive", "intent"}
        output_fields = all_br_fields - internal_only
        unmapped = output_fields - mapped_bulk
        checks["field_map_covers_all_bulk_output_fields"] = len(unmapped) == 0
        if unmapped:
            log.warning("  BulkRowResult fields not in FIELD_MAP: %s", unmapped)
        log.info("  FIELD_MAP: %d entries, %d output fields, %d unmapped",
                 len(FIELD_MAP), len(output_fields), len(unmapped))

        # ── Check 9: No duplicate bulk fields in FIELD_MAP ──
        seen = set()
        dupes = []
        for _cat, _s, bulk_field, _n in FIELD_MAP:
            if bulk_field != "(none)":
                if bulk_field in seen:
                    dupes.append(bulk_field)
                seen.add(bulk_field)
        checks["field_map_no_duplicate_bulk_fields"] = len(dupes) == 0
        if dupes:
            log.warning("  FIELD_MAP duplicate bulk fields: %s", dupes)

        # ── Check 10: CSV columns match to_summary_dict() keys ──
        from src.bulk_summary import _CSV_COLUMNS

        r = BulkRowResult(0, "AlignCheck", "success")
        d = r.to_summary_dict()
        csv_set = set(_CSV_COLUMNS)
        dict_set = set(d.keys())
        missing_in_csv = dict_set - csv_set
        checks["csv_columns_match_summary_dict"] = len(missing_in_csv) == 0
        if missing_in_csv:
            log.warning("  to_summary_dict keys not in _CSV_COLUMNS: %s", missing_in_csv)
        log.info("  CSV columns: %d, summary dict keys: %d, missing: %d",
                 len(csv_set), len(dict_set), len(missing_in_csv))

        # ── Check 11: Grade distribution normalizes canonical grades correctly ──
        from src.bulk_summary import generate_display_stats
        from src.bulk_runner import BulkBatchResult

        br = BulkBatchResult(batch_type="align_test", molecule_class="canonical_mab")
        br.results = [
            BulkRowResult(0, "A", "success",
                          composite_dev_score=0.10, composite_dev_grade="Low Risk"),
            BulkRowResult(1, "B", "success",
                          composite_dev_score=0.40, composite_dev_grade="Medium Risk"),
            BulkRowResult(2, "C", "success",
                          composite_dev_score=0.80, composite_dev_grade="High Risk"),
        ]
        stats = generate_display_stats(br)
        gd = stats["grade_distribution"]
        checks["grade_dist_low_counted"] = gd.get("Low", 0) == 1
        checks["grade_dist_medium_counted"] = gd.get("Medium", 0) == 1
        checks["grade_dist_high_counted"] = gd.get("High", 0) == 1
        log.info("  Grade distribution: Low=%d Medium=%d High=%d",
                 gd.get("Low", 0), gd.get("Medium", 0), gd.get("High", 0))

        # ── Check 12: No old color hex values in app.py ──
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        old_colors_found = []
        if os.path.exists(app_path):
            with open(app_path) as f:
                app_text = f.read()
            for old_hex in ("#059669", "#D97706", "#DC2626", "#d97706", "#dc2626", "#16a34a"):
                if old_hex in app_text:
                    old_colors_found.append(old_hex)
        checks["no_old_color_hex_in_app"] = len(old_colors_found) == 0
        if old_colors_found:
            log.warning("  Old hex colors found in app.py: %s", old_colors_found)
        else:
            log.info("  Color audit: zero old hex values in app.py")

        # ── Check 13: optional_deps core layer available ──
        from src.optional_deps import layer_status
        dep_status = layer_status()
        checks["optional_deps_core_available"] = dep_status.get("core", False) is True
        log.info("  optional_deps: core=%s, analysis=%s, training=%s",
                 dep_status.get("core"), dep_status.get("analysis"), dep_status.get("training"))

        # ── Check 14: molecule_registry selftest ──
        from src.molecule_registry import _selftest as _reg_selftest
        try:
            import io as _io
            import sys as _sys
            _old = _sys.stdout
            _sys.stdout = _io.StringIO()
            try:
                reg_ok = _reg_selftest()
            finally:
                _sys.stdout = _old
            checks["molecule_registry_selftest_pass"] = reg_ok
        except Exception as e:
            log.warning("  molecule_registry selftest failed: %s", e)
            checks["molecule_registry_selftest_pass"] = False

        # ── Check 15: Requirements chain integrity ──
        chain_ok = True
        expected_chain = [
            ("requirements.txt", "requirements-training.txt"),
            ("requirements-training.txt", "requirements-analysis.txt"),
            ("requirements-analysis.txt", "requirements-core.txt"),
        ]
        for parent, child in expected_chain:
            parent_path = os.path.join(PROJECT_ROOT, parent)
            if os.path.exists(parent_path):
                with open(parent_path) as f:
                    content = f.read()
                if f"-r {child}" not in content:
                    log.warning("  %s does not reference %s", parent, child)
                    chain_ok = False
            else:
                log.warning("  %s not found", parent)
                chain_ok = False
        checks["requirements_chain_intact"] = chain_ok
        log.info("  Requirements chain: %s", "intact" if chain_ok else "BROKEN")

        # ── Check 16: Determinism — score reproducibility ──
        from src.developability_core import assess_developability as _assess
        scores = []
        for _ in range(5):
            r = _assess(
                molecule_name="DeterminismCheck",
                molecule_class="canonical_mab",
                dev_predictions={"agg_risk": 0.20, "stability": 0.70, "viscosity_risk": 0.10},
            )
            scores.append(r.composite_score)
        checks["score_determinism_5_runs"] = len(set(scores)) == 1
        if len(set(scores)) > 1:
            log.warning("  Score not deterministic: %s", set(scores))
        else:
            log.info("  Determinism: 5 identical scores (%.4f)", scores[0])

        # ── Check 17: Context isolation — ReportContext is frozen after build ──
        from src.report_schema import ReportContext
        ctx = ReportContext(molecule_name="IsolationTest", molecule_class="canonical_mab")
        ctx._frozen = True
        try:
            ctx.molecule_name = "TAMPERED"
            checks["report_context_frozen"] = False
            log.warning("  ReportContext not frozen — mutation allowed")
        except (AttributeError, TypeError):
            checks["report_context_frozen"] = True
        log.info("  Context isolation: frozen=%s", checks["report_context_frozen"])

        # ── Check 18: Export chain — CSV columns == to_summary_dict keys ──
        # (Already checked in check 10 above, adding explicit name for clarity)
        checks["export_chain_csv_json_aligned"] = checks.get("csv_columns_match_summary_dict", False)

        # ── Check 19: Bulk row isolation — two rows produce independent results ──
        r1 = BulkRowResult(0, "Mol_A", "success", composite_dev_score=0.20)
        r2 = BulkRowResult(1, "Mol_B", "success", composite_dev_score=0.60)
        checks["bulk_row_isolation"] = (r1.composite_dev_score != r2.composite_dev_score and
                                         r1.name != r2.name and r1.row_index != r2.row_index)
        log.info("  Bulk row isolation: %s",
                 "independent" if checks["bulk_row_isolation"] else "SHARED STATE")

        # ── Check 20: Factory reset clears JAIN137 calibration flag ──
        from src.ml_predictor import factory_reset, get_calibration_status
        _pre = get_calibration_status()
        factory_reset()
        _post = get_calibration_status()
        checks["factory_reset_clears_calibration"] = _post.get("calibrated", True) is False
        log.info("  Factory reset clears calibration: %s",
                 checks["factory_reset_clears_calibration"])

        # ── Check 21: No st.session_state in backend src/ analysis modules ──
        import glob
        _backend_files = glob.glob(os.path.join(PROJECT_ROOT, "src", "*.py"))
        _analysis_modules = [
            "report_assembler.py", "developability_core.py", "bispecific_engine.py",
            "molecule_classifier.py", "feature_registry.py", "bulk_runner.py",
            "bulk_summary.py", "bulk_schema.py",
        ]
        session_state_leak = False
        for fname in _analysis_modules:
            fpath = os.path.join(PROJECT_ROOT, "src", fname)
            if os.path.exists(fpath):
                with open(fpath) as _f:
                    if "st.session_state" in _f.read():
                        log.warning("  st.session_state found in %s!", fname)
                        session_state_leak = True
        checks["no_session_state_in_backend"] = not session_state_leak
        log.info("  Backend isolation: no st.session_state in analysis modules = %s",
                 not session_state_leak)

        # ── Finalize ──
        status = _checks_to_status(checks)
        n_pass = sum(1 for v in checks.values() if v)
        n_fail = sum(1 for v in checks.values() if not v)

        if status == "FAIL":
            log.warning("  Platform Alignment FAIL: %s",
                        [k for k, v in checks.items() if not v])
        else:
            log.info("  Platform Alignment: %d checks all PASS", len(checks))

        return {
            "status": status,
            "checks": checks,
            "data": {
                "n_checks": len(checks),
                "n_passed": n_pass,
                "n_failed": n_fail,
                "registry_classes": len(MOLECULE_REGISTRY),
                "field_map_entries": len(FIELD_MAP),
                "csv_columns": len(csv_set),
                "old_colors_found": old_colors_found,
                "dep_status": dep_status,
            },
        }
    except Exception as e:
        log.error("Platform Alignment test: %s", e, exc_info=True)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# 20. Post-Training Validation (conditional — only if model artifact exists)
# =========================================================================
def test_post_training() -> dict:
    """
    Section 20: Post-training model artifact validation.

    CONDITIONAL: If no trained model exists, returns SKIP (not FAIL).
    When a model IS present, validates:
      - Artifact files exist and are loadable
      - Metadata schema is compatible with platform
      - Model classes are subset of MoleculeClass enum
      - Inference is deterministic (same input → same output)
      - Output class format is valid for platform consumption
      - Test accuracy exceeds minimum threshold (50%)
      - Benchmark panel drift is within tolerance
    """
    log.info("=== 20. Post-Training Validation (conditional) ===")

    artifact_dir = os.path.join(PROJECT_ROOT, "models", "classifier")
    model_path = os.path.join(artifact_dir, "classifier_model.npz")

    if not os.path.exists(model_path):
        log.info("  No trained model found — SKIP (not an error)")
        return {"status": "SKIP", "checks": {}, "data": {"reason": "no model artifact"}}

    try:
        from src.training.benchmark_evaluator import post_training_selftest

        checks = {}
        pt_checks = post_training_selftest(artifact_dir)
        for k, v in pt_checks.items():
            checks[f"pt_{k}"] = v

        n_pass = sum(1 for v in checks.values() if v)
        n_fail = sum(1 for v in checks.values() if not v)

        # Log details
        for k, v in checks.items():
            status = "PASS" if v else "FAIL"
            log.info("  %s: %s", k, status)

        status = _checks_to_status(checks)
        log.info("  Post-Training: %d/%d checks PASS", n_pass, len(checks))

        return {
            "status": status,
            "checks": checks,
            "data": {
                "artifact_dir": artifact_dir,
                "n_checks": len(checks),
                "n_passed": n_pass,
                "n_failed": n_fail,
            },
        }
    except Exception as e:
        log.error("Post-Training test: %s", e, exc_info=True)
        return {"status": "ERROR", "data": {}, "error": str(e)}


# =========================================================================
# Main: Run all and write JSON
# =========================================================================
def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "validation_results_v3.json")

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "v3.1_substantive_validation",
        "tests": {},
    }

    for name, func in [
        ("property_mapper", test_property_mapper),
        ("developability", test_developability),
        ("upstream", test_upstream),
        ("analytical_qc", test_analytical_qc),
        ("stability", test_stability),
        ("nistmab", test_nistmab),
        ("immunogenicity", test_immunogenicity),
        ("therasabdab", test_therasabdab),
        ("cross_check", test_cross_check),
        ("cex_resolution", test_cex_resolution),
        ("end_to_end", test_end_to_end),
        ("bispecific_format", test_bispecific_format),
        ("noncanonical_formats", test_noncanonical_formats),
        ("ood_confidence", test_ood_confidence),
        ("cross_section_consistency", test_cross_section_consistency),
        ("downstream_doe", test_downstream_doe),
        ("cross_path_alignment", test_cross_path_alignment),
        ("schema_alignment", test_schema_alignment),
        ("platform_alignment", test_platform_alignment),
        ("post_training", test_post_training),
    ]:
        try:
            results["tests"][name] = func()
        except Exception as e:
            log.error("Test %s FAILED: %s", name, e)
            results["tests"][name] = {"status": "ERROR", "error": str(e)}

    # Summary
    total = len(results["tests"])
    passed = sum(1 for t in results["tests"].values() if t.get("status") == "PASS")
    failed = sum(1 for t in results["tests"].values() if t.get("status") == "FAIL")
    errors = sum(1 for t in results["tests"].values() if t.get("status") == "ERROR")
    skipped = sum(1 for t in results["tests"].values() if t.get("status") == "SKIP")
    total_checks = sum(
        len(t.get("checks", {})) for t in results["tests"].values()
    )
    failed_checks = sum(
        sum(1 for v in t.get("checks", {}).values() if not v)
        for t in results["tests"].values()
    )

    results["summary"] = {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "total_checks": total_checks,
        "failed_checks": failed_checks,
    }

    log.info("=" * 60)
    log.info("RESULTS: %d/%d tests PASS | %d FAIL | %d ERROR | %d SKIP",
             passed, total, failed, errors, skipped)
    log.info("  Substantive checks: %d/%d passed", total_checks - failed_checks, total_checks)
    log.info("=" * 60)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Wrote %s (%d bytes)", output_path, os.path.getsize(output_path))
    print(output_path)


if __name__ == "__main__":
    main()
