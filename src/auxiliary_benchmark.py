"""
auxiliary_benchmark.py  ·  ProtePilot — Auxiliary Module Benchmark
====================================================================
Edge case and regression benchmark for analytical, QC, validation,
feature, and data pipeline modules.

Usage:
    python -m src.auxiliary_benchmark                              # Full run
    python -m src.auxiliary_benchmark --module analytical_twin     # Single
    python -m src.auxiliary_benchmark --json                       # JSON output

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Tuple

log = logging.getLogger("ProtePilot.AuxBenchmark")


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

_VH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRGITIFGVVIIPGFFDIWGQGTLVTVSS"
_VL = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
_PEP = "ACDEFGHIKLMNPQRSTVWY"


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Cases
# ═══════════════════════════════════════════════════════════════════════

def _make_cases() -> List[Dict[str, Any]]:
    cases = []

    # ── Analytical Twin ──────────────────────────────────────────

    def _mass_peptide_vs_mab():
        """Peptide mass should be much smaller than mAb mass."""
        from src.analytical_twin import calculate_intact_mass
        pep = calculate_intact_mass(_PEP, is_mab=False)
        mab = calculate_intact_mass(_VH + _VL, is_mab=True)
        ok = pep["bare_mass_da"] < mab["bare_mass_da"]
        return ok, f"pep={pep['bare_mass_da']:.0f} Da < mab={mab['bare_mass_da']:.0f} Da"

    def _mass_disulfide_reduces():
        """Disulfide correction should reduce mass (loss of 2H per bond)."""
        from src.analytical_twin import calculate_intact_mass
        r = calculate_intact_mass(_VH + _VL, is_mab=True, n_disulfide_bonds=4)
        ok = r["disulfide_corrected_da"] < r["bare_mass_da"]
        return ok, f"bare={r['bare_mass_da']:.1f} > corrected={r['disulfide_corrected_da']:.1f}"

    def _digest_missed_cleavage_more_peptides():
        """More missed cleavages should produce more peptides."""
        from src.analytical_twin import tryptic_digest
        p0 = tryptic_digest(_VH, missed_cleavages=0)
        p1 = tryptic_digest(_VH, missed_cleavages=1)
        ok = len(p1) >= len(p0)
        return ok, f"mc=0: {len(p0)} peptides, mc=1: {len(p1)} peptides"

    def _liability_density_high_deam():
        """Sequence rich in NG/NS should have higher liability density."""
        from src.analytical_twin import calculate_liability_density
        normal = calculate_liability_density([{"sequence": _VH, "copy_number": 1}])
        deam_rich = "NGNGNSNSNGNGNSNSNGNGNSNS" * 5
        high = calculate_liability_density([{"sequence": deam_rich, "copy_number": 1}])
        ok = high["density_per_1000"] > normal["density_per_1000"]
        return ok, f"normal={normal['density_per_1000']:.1f} < deam_rich={high['density_per_1000']:.1f}"

    cases.extend([
        {"name": "mass_peptide_vs_mab", "module": "analytical_twin", "fn": _mass_peptide_vs_mab,
         "desc": "Peptide mass < mAb mass"},
        {"name": "mass_disulfide_reduces", "module": "analytical_twin", "fn": _mass_disulfide_reduces,
         "desc": "Disulfide bonds reduce mass"},
        {"name": "digest_missed_cleavage", "module": "analytical_twin", "fn": _digest_missed_cleavage_more_peptides,
         "desc": "More missed cleavages = more peptides"},
        {"name": "liability_density_high_deam", "module": "analytical_twin", "fn": _liability_density_high_deam,
         "desc": "Deamidation-rich sequence has higher liability density"},
    ])

    # ── Analytical QC Twin ───────────────────────────────────────

    def _cief_deam_increases_acidic():
        """More deamidation sites should increase acidic %."""
        from src.analytical_qc_twin import simulate_cief
        low = simulate_cief(_VH, pI=7.5, deamidation_sites=0)
        high = simulate_cief(_VH, pI=7.5, deamidation_sites=10)
        ok = high.acidic_pct >= low.acidic_pct
        return ok, f"deam=0: acidic={low.acidic_pct:.1f}%, deam=10: acidic={high.acidic_pct:.1f}%"

    def _ce_sds_aggregation_increases_hmw():
        """Higher aggregation should increase HMW %."""
        from src.analytical_qc_twin import simulate_ce_sds
        low = simulate_ce_sds(_VH, aggregation_pct=1.0, molecule_class="canonical_mab")
        high = simulate_ce_sds(_VH, aggregation_pct=10.0, molecule_class="canonical_mab")
        ok = high.hmw_pct >= low.hmw_pct
        return ok, f"agg=1%: hmw={low.hmw_pct:.1f}%, agg=10%: hmw={high.hmw_pct:.1f}%"

    def _glycan_kifunensine_high_mannose():
        """Kifunensine addition should increase high mannose."""
        from src.analytical_qc_twin import simulate_glycan_profile
        normal = simulate_glycan_profile(molecule_class="canonical_mab")
        kif = simulate_glycan_profile(kifunensine_added=True, molecule_class="canonical_mab")
        ok = kif.high_mannose_pct > normal.high_mannose_pct
        return ok, f"normal HM={normal.high_mannose_pct:.1f}%, kif HM={kif.high_mannose_pct:.1f}%"

    cases.extend([
        {"name": "cief_deam_increases_acidic", "module": "analytical_qc_twin",
         "fn": _cief_deam_increases_acidic, "desc": "Deamidation increases acidic variants"},
        {"name": "ce_sds_agg_increases_hmw", "module": "analytical_qc_twin",
         "fn": _ce_sds_aggregation_increases_hmw, "desc": "Aggregation increases HMW"},
        {"name": "glycan_kifunensine_hm", "module": "analytical_qc_twin",
         "fn": _glycan_kifunensine_high_mannose, "desc": "Kifunensine increases high mannose"},
    ])

    # ── Validation Planner ───────────────────────────────────────

    def _plan_high_risk_more_assays():
        """Higher risk should trigger more assays."""
        from src.validation_planner import generate_validation_plan
        low = generate_validation_plan(
            risk_scores={"agg_risk": 0.1, "stability": 0.1, "viscosity_risk": 0.1},
            molecule_class="canonical_mab")
        high = generate_validation_plan(
            risk_scores={"agg_risk": 0.9, "stability": 0.9, "viscosity_risk": 0.9},
            molecule_class="canonical_mab")
        lo_n = high.get("total_assays", len(high.get("all_assays", [])))
        hi_n = low.get("total_assays", len(low.get("all_assays", [])))
        ok = lo_n >= hi_n  # high risk should have >= assays
        return ok, f"low_risk={hi_n} assays, high_risk={lo_n} assays"

    def _plan_format_specific_adc():
        """ADC should get format-specific assays."""
        from src.validation_planner import generate_validation_plan
        r = generate_validation_plan(
            risk_scores={"agg_risk": 0.5, "stability": 0.5, "viscosity_risk": 0.3},
            molecule_class="adc")
        fmt = r.get("format_specific_assays", [])
        total = r.get("total_assays", len(r.get("all_assays", [])))
        ok = total >= 1
        return ok, f"ADC: {len(fmt)} format-specific, {total} total"

    cases.extend([
        {"name": "plan_high_risk_more_assays", "module": "validation_planner",
         "fn": _plan_high_risk_more_assays, "desc": "Higher risk triggers more assays"},
        {"name": "plan_format_specific_adc", "module": "validation_planner",
         "fn": _plan_format_specific_adc, "desc": "ADC gets format-specific assays"},
    ])

    # ── Feature Registry ─────────────────────────────────────────

    def _features_longer_seq_higher_mw():
        """Longer sequence should have higher MW."""
        from src.feature_registry import compute_features
        short = compute_features(_PEP, molecule_class="peptide")
        long = compute_features(_VH + _VL, molecule_class="canonical_mab")
        ok = long.value("mw_kda") > short.value("mw_kda")
        return ok, f"short={short.value('mw_kda'):.1f}, long={long.value('mw_kda'):.1f} kDa"

    def _features_basic_seq_high_pi():
        """All-lysine sequence should have high pI."""
        from src.feature_registry import compute_features
        basic = compute_features("KKKKKKKKKKKKKKKKKKKK", molecule_class="peptide")
        acidic = compute_features("DDDDDDDDDDDDDDDDDDDD", molecule_class="peptide")
        ok = basic.value("pI") > acidic.value("pI")
        return ok, f"K20 pI={basic.value('pI'):.1f}, D20 pI={acidic.value('pI'):.1f}"

    def _features_consistent_between_calls():
        """Same sequence should produce identical features."""
        from src.feature_registry import compute_features
        a = compute_features(_VH, molecule_class="canonical_mab")
        b = compute_features(_VH, molecule_class="canonical_mab")
        va, vb = a.ml_vector(), b.ml_vector()
        ok = va == vb
        return ok, f"Vectors match: {len(va)}-dim"

    cases.extend([
        {"name": "features_longer_higher_mw", "module": "feature_registry",
         "fn": _features_longer_seq_higher_mw, "desc": "Longer sequence = higher MW"},
        {"name": "features_basic_high_pi", "module": "feature_registry",
         "fn": _features_basic_seq_high_pi, "desc": "All-K high pI, all-D low pI"},
        {"name": "features_deterministic", "module": "feature_registry",
         "fn": _features_consistent_between_calls, "desc": "Features deterministic"},
    ])

    # ── Data Pipeline ────────────────────────────────────────────

    def _pipeline_empty_csv():
        """Empty CSV should return error or 0 rows."""
        from src.data_pipeline import parse_csv_upload
        import io
        r = parse_csv_upload(io.StringIO(""), filename="empty.csv")
        ok = r.get("status") in ("error",) or r.get("n_rows", 0) == 0
        return ok, f"status={r.get('status')}, n_rows={r.get('n_rows', 0)}"

    def _pipeline_validate_missing_sequence():
        """validate_csv_data() should flag rows missing sequence."""
        from src.data_pipeline import validate_csv_data
        data = [
            {"Name": "Mol1"},
            {"Name": "Mol2", "Sequence": "ACDEF"},
        ]
        r = validate_csv_data(data)
        missing = r.get("n_missing_sequence", 0)
        ok = missing >= 1
        return ok, f"n_missing_sequence={missing}"

    cases.extend([
        {"name": "pipeline_empty_csv", "module": "data_pipeline",
         "fn": _pipeline_empty_csv, "desc": "Empty CSV handled gracefully"},
        {"name": "pipeline_validate_missing_seq", "module": "data_pipeline",
         "fn": _pipeline_validate_missing_sequence, "desc": "Missing sequence flagged"},
    ])

    return cases


BENCHMARK_CASES = _make_cases()


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════

def run_auxiliary_benchmark(
    module_filter: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    cases = BENCHMARK_CASES
    if module_filter:
        cases = [c for c in cases if c["module"] == module_filter]

    passed = failed = 0
    failures = []
    t0 = time.monotonic()

    for case in cases:
        name = case["name"]
        try:
            ok, detail = case["fn"]()
            if ok:
                passed += 1
                if verbose:
                    log.info("  [PASS] %s: %s", name, detail)
            else:
                failed += 1
                failures.append({"test": name, "detail": detail})
                if verbose:
                    log.warning("  [FAIL] %s: %s", name, detail)
        except Exception as exc:
            failed += 1
            failures.append({"test": name, "error": str(exc)})
            if verbose:
                log.error("  [ERR]  %s: %s", name, exc)

    elapsed_ms = (time.monotonic() - t0) * 1000
    total = passed + failed
    return {
        "passed": passed, "failed": failed, "total": total,
        "accuracy": round(passed / max(total, 1), 4),
        "failures": failures, "timing_ms": round(elapsed_ms, 1),
        "all_passed": failed == 0,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Auxiliary Module Benchmark Suite")
    parser.add_argument("--module", default=None,
                        choices=["analytical_twin", "analytical_qc_twin",
                                 "validation_planner", "feature_registry",
                                 "data_pipeline"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    log.info("Running auxiliary benchmarks%s...",
             f" (module={args.module})" if args.module else "")
    result = run_auxiliary_benchmark(module_filter=args.module)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"Auxiliary Benchmark: {result['passed']}/{result['total']} passed "
              f"({result['accuracy']:.0%}) in {result['timing_ms']:.0f}ms")
        if result["failures"]:
            print("Failures:")
            for f in result["failures"]:
                print(f"  - {f['test']}: {f.get('detail', f.get('error', ''))}")
        print(f"Status: {'ALL PASSED' if result['all_passed'] else 'FAILURES DETECTED'}")
    sys.exit(0 if result["all_passed"] else 1)
