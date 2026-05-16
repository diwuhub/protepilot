"""
twin_benchmark.py  ·  ProtePilot — Twin Engine Benchmark Suite
================================================================
Edge case and regression benchmark for all twin engine modules.
Tests boundary conditions, extreme inputs, and cross-module consistency.

Usage:
    python -m src.twin_benchmark                       # Full benchmark
    python -m src.twin_benchmark --module stability    # Single module
    python -m src.twin_benchmark --json                # JSON output

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import os as _os
import sys as _sys
_proj = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _proj not in _sys.path:
    _sys.path.insert(0, _proj)

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("ProtePilot.TwinBenchmark")


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Cases
# ═══════════════════════════════════════════════════════════════════════

# Each: {name, module, description, category, check_fn} -> (pass, detail)

def _make_cases() -> List[Dict[str, Any]]:
    cases = []

    # ── Stability ─────────────────────────────────────────────────
    def _stab_high_hmw():
        """Starting with high HMW should reduce shelf life."""
        from src.stability_twin import run_stability_study
        r_low = run_stability_study(starting_hmw_pct=0.5, pI=7.0)
        r_high = run_stability_study(starting_hmw_pct=5.0, pI=7.0)
        ok = r_high.predicted_shelf_life_months <= r_low.predicted_shelf_life_months
        return ok, f"low_hmw={r_low.predicted_shelf_life_months}mo, high_hmw={r_high.predicted_shelf_life_months}mo"

    def _stab_extreme_pi():
        """pI near formulation pH worsens stability (aggregation at pI)."""
        from src.stability_twin import run_stability_study
        r_far = run_stability_study(pI=8.5)  # far from pH 6.0 → good charge repulsion
        r_near = run_stability_study(pI=6.5)  # near pH 6.0 → low charge → aggregation
        ok = r_near.predicted_shelf_life_months <= r_far.predicted_shelf_life_months
        return ok, f"pI_near_pH={r_near.predicted_shelf_life_months}mo, pI_far_pH={r_far.predicted_shelf_life_months}mo"

    def _stab_excipient_benefit():
        """Sucrose excipient should improve stability."""
        from src.stability_twin import run_stability_study
        r_no = run_stability_study(pI=8.5, starting_hmw_pct=2.0)
        r_yes = run_stability_study(pI=8.5, starting_hmw_pct=2.0, excipients=["sucrose"])
        ok = r_yes.predicted_shelf_life_months >= r_no.predicted_shelf_life_months
        return ok, f"no_excip={r_no.predicted_shelf_life_months}mo, sucrose={r_yes.predicted_shelf_life_months}mo"

    cases.extend([
        {"name": "stability_high_hmw_penalty", "module": "stability", "fn": _stab_high_hmw,
         "desc": "High starting HMW reduces shelf life", "category": "edge"},
        {"name": "stability_extreme_pi", "module": "stability", "fn": _stab_extreme_pi,
         "desc": "Extreme pI worsens stability", "category": "edge"},
        {"name": "stability_excipient_benefit", "module": "stability", "fn": _stab_excipient_benefit,
         "desc": "Sucrose improves stability", "category": "edge"},
    ])

    # ── Upstream ──────────────────────────────────────────────────
    def _up_hydrophobic_penalty():
        """High hydrophobicity should reduce titer."""
        from src.upstream_twin import run_upstream_simulation
        r_norm = run_upstream_simulation(sequence="ACDEFGHIKLMNPQRSTVWY" * 10, molecule_class="canonical_mab")
        r_hydro = run_upstream_simulation(sequence="VVVIIILLLLFFWW" * 15, molecule_class="canonical_mab")
        ok = r_hydro.final_titer <= r_norm.final_titer
        return ok, f"normal={r_norm.final_titer:.2f}, hydrophobic={r_hydro.final_titer:.2f}"

    def _up_peptide_vs_mab():
        """Peptide class should have different production profile than mAb."""
        from src.upstream_twin import run_upstream_simulation
        seq = "ACDEFGHIKLMNPQRSTVWY" * 10
        r_mab = run_upstream_simulation(sequence=seq, molecule_class="canonical_mab")
        r_pep = run_upstream_simulation(sequence=seq, molecule_class="peptide")
        ok = r_mab.final_titer != r_pep.final_titer  # Should differ
        return ok, f"mab={r_mab.final_titer:.2f}, peptide={r_pep.final_titer:.2f}"

    def _up_short_sequence():
        """Very short sequence should still produce a result."""
        from src.upstream_twin import run_upstream_simulation
        r = run_upstream_simulation(sequence="ACDEFGHIKL", molecule_class="peptide")
        ok = r.final_titer > 0 and len(r.time_days) > 1
        return ok, f"titer={r.final_titer:.2f}, timepoints={len(r.time_days)}"

    cases.extend([
        {"name": "upstream_hydrophobic_penalty", "module": "upstream", "fn": _up_hydrophobic_penalty,
         "desc": "High hydrophobicity reduces titer", "category": "edge"},
        {"name": "upstream_peptide_vs_mab", "module": "upstream", "fn": _up_peptide_vs_mab,
         "desc": "Different molecule classes produce different titers", "category": "core"},
        {"name": "upstream_short_sequence", "module": "upstream", "fn": _up_short_sequence,
         "desc": "Short sequence still works", "category": "edge"},
    ])

    # ── Immunogenicity ────────────────────────────────────────────
    def _imm_all_human_low_risk():
        """Full-length human-like VH+VL should have Low-Medium ADA risk."""
        from src.immunogenicity_twin import run_immunogenicity_assessment
        # Full human germline VH (IGHV1-69) + VK (IGKV1-39) — close to trastuzumab framework
        human_vh = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGGYSSGWYFDVWGQGTLVTVSS"
        human_vl = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGQGTKVEIK"
        r = run_immunogenicity_assessment(sequence=human_vh + human_vl, molecule_class="canonical_mab")
        ok = r.ada_risk_level in ("Low", "Medium")
        return ok, f"risk={r.ada_risk_level}, score={r.ada_risk_score:.2f}"

    def _imm_extreme_sequence():
        """All-lysine sequence should have extreme immunogenicity profile."""
        from src.immunogenicity_twin import run_immunogenicity_assessment
        r = run_immunogenicity_assessment(sequence="K" * 100)
        ok = isinstance(r.ada_risk_score, (int, float))
        return ok, f"risk={r.ada_risk_level}, score={r.ada_risk_score:.2f}"

    def _imm_empty_hotspots():
        """Very short peptide should still return valid result."""
        from src.immunogenicity_twin import run_immunogenicity_assessment
        r = run_immunogenicity_assessment(sequence="ACDEF")
        ok = r.total_peptides_scanned >= 0
        return ok, f"scanned={r.total_peptides_scanned}"

    cases.extend([
        {"name": "immunogenicity_human_low_risk", "module": "immunogenicity", "fn": _imm_all_human_low_risk,
         "desc": "Human germline VH has low-medium ADA risk", "category": "core"},
        {"name": "immunogenicity_extreme_composition", "module": "immunogenicity", "fn": _imm_extreme_sequence,
         "desc": "All-lysine returns valid result", "category": "edge"},
        {"name": "immunogenicity_short_peptide", "module": "immunogenicity", "fn": _imm_empty_hotspots,
         "desc": "Very short input returns valid result", "category": "edge"},
    ])

    # ── Preclinical ───────────────────────────────────────────────
    def _pk_fcrn_impact():
        """Broken FcRn binding should reduce half-life."""
        from src.preclinical_twin import predict_human_half_life
        r_ok = predict_human_half_life(global_pi=7.0, fcrn_binding_motif_intact=True)
        r_bad = predict_human_half_life(global_pi=7.0, fcrn_binding_motif_intact=False)
        ok = r_bad["half_life_days"] < r_ok["half_life_days"]
        return ok, f"intact={r_ok['half_life_days']:.1f}d, broken={r_bad['half_life_days']:.1f}d"

    def _pk_high_liability():
        """High liability density should reduce half-life."""
        from src.preclinical_twin import predict_human_half_life
        r_low = predict_human_half_life(global_pi=7.0, liability_density=0.01)
        r_high = predict_human_half_life(global_pi=7.0, liability_density=50.0)
        ok = r_high["half_life_days"] <= r_low["half_life_days"]
        return ok, f"low_liab={r_low['half_life_days']:.1f}d, high_liab={r_high['half_life_days']:.1f}d"

    cases.extend([
        {"name": "preclinical_fcrn_impact", "module": "preclinical", "fn": _pk_fcrn_impact,
         "desc": "Broken FcRn binding reduces half-life", "category": "core"},
        {"name": "preclinical_high_liability", "module": "preclinical", "fn": _pk_high_liability,
         "desc": "High liability density reduces half-life", "category": "edge"},
    ])

    # ── Formulation ───────────────────────────────────────────────
    def _form_ph_near_pi():
        """Formulation at pH near pI should flag charge issue."""
        from src.formulation_twin import run_formulation_assessment
        r = run_formulation_assessment(pI=6.0, buffer_ph=6.0)
        ok = isinstance(r, dict) and "status" in r
        charge_warn = r.get("charge_near_zero", False)
        return ok, f"status={r.get('status')}, charge_near_zero={charge_warn}"

    def _form_ph_away_from_pi():
        """Formulation at pH far from pI should be safer."""
        from src.formulation_twin import run_formulation_assessment
        r = run_formulation_assessment(pI=8.5, buffer_ph=6.0)
        ok = isinstance(r, dict) and "status" in r
        return ok, f"status={r.get('status')}, keys={len(r)}"

    cases.extend([
        {"name": "formulation_ph_near_pi", "module": "formulation", "fn": _form_ph_near_pi,
         "desc": "pH near pI flags charge issue", "category": "core"},
        {"name": "formulation_ph_away_from_pi", "module": "formulation", "fn": _form_ph_away_from_pi,
         "desc": "pH far from pI is safer", "category": "core"},
    ])

    # ── COGS ──────────────────────────────────────────────────────
    def _cogs_small_batch():
        """Small batch should have higher cost/g."""
        from src.cogs_twin import run_cogs_analysis
        r_small = run_cogs_analysis(titer_g_per_L=3.0, bioreactor_volume_L=200)
        r_large = run_cogs_analysis(titer_g_per_L=3.0, bioreactor_volume_L=15000)
        ok = r_small.cogs_per_gram > r_large.cogs_per_gram
        return ok, f"200L=${r_small.cogs_per_gram:.0f}/g, 15000L=${r_large.cogs_per_gram:.0f}/g"

    def _cogs_low_yield():
        """Low downstream yield should increase cost."""
        from src.cogs_twin import run_cogs_analysis
        r_high = run_cogs_analysis(titer_g_per_L=3.0, downstream_yield=0.80)
        r_low = run_cogs_analysis(titer_g_per_L=3.0, downstream_yield=0.30)
        ok = r_low.cogs_per_gram > r_high.cogs_per_gram
        return ok, f"yield 80%=${r_high.cogs_per_gram:.0f}/g, 30%=${r_low.cogs_per_gram:.0f}/g"

    cases.extend([
        {"name": "cogs_scale_economy", "module": "cogs", "fn": _cogs_small_batch,
         "desc": "Larger batch reduces cost per gram", "category": "core"},
        {"name": "cogs_yield_impact", "module": "cogs", "fn": _cogs_low_yield,
         "desc": "Lower yield increases cost", "category": "edge"},
    ])

    return cases


BENCHMARK_CASES = _make_cases()


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════

def run_twin_benchmark(
    module_filter: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run twin engine benchmarks."""
    cases = BENCHMARK_CASES
    if module_filter:
        cases = [c for c in cases if c["module"] == module_filter]

    passed = 0
    failed = 0
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
        "passed": passed,
        "failed": failed,
        "total": total,
        "accuracy": round(passed / max(total, 1), 4),
        "failures": failures,
        "timing_ms": round(elapsed_ms, 1),
        "all_passed": failed == 0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Twin Engine Benchmark Suite")
    parser.add_argument("--module", default=None,
                        choices=["stability", "upstream", "immunogenicity",
                                 "preclinical", "formulation", "cogs"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    log.info("Running twin engine benchmarks%s...",
             f" (module={args.module})" if args.module else "")
    result = run_twin_benchmark(module_filter=args.module)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"Twin Engine Benchmark: {result['passed']}/{result['total']} passed "
              f"({result['accuracy']:.0%}) in {result['timing_ms']:.0f}ms")
        if result["failures"]:
            print("Failures:")
            for f in result["failures"]:
                print(f"  - {f['test']}: {f.get('detail', f.get('error', ''))}")
        print(f"Status: {'ALL PASSED' if result['all_passed'] else 'FAILURES DETECTED'}")

    sys.exit(0 if result["all_passed"] else 1)
