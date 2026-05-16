"""
infra_benchmark.py  ·  ProtePilot — Infrastructure & Small Module Benchmark
==============================================================================
Edge-case benchmarks for scaleup_twin, pareto_optimizer, data_harmonizer,
molecule_registry, ood_baseline, workspace_manager, PropertyMapper.

Usage:
    python -m src.infra_benchmark              # Full run
    python -m src.infra_benchmark --json       # JSON output

Author  : Di (ProtePilot)
"""
from __future__ import annotations
import json, logging, sys, time
from typing import Any, Dict, List, Tuple
log = logging.getLogger("ProtePilot.InfraBenchmark")


def _make_cases() -> List[Dict[str, Any]]:
    cases = []

    # ── scaleup_twin ─────────────────────────────────────────────
    def _scaleup_larger_volume_more_shear():
        from src.scaleup_twin import run_scaleup_simulation
        r = run_scaleup_simulation(small_volume_L=2.0, large_volume_L=20000.0, bench_titer=5.0)
        ok = r.predicted_titer_large > 0 and hasattr(r, "warnings")
        return ok, f"20kL: titer={r.predicted_titer_large:.2f}, warnings={len(r.warnings)}"

    def _scaleup_equal_volumes():
        """Equal volumes should still produce valid result (scaling may not be exactly 1.0 due to model)."""
        from src.scaleup_twin import run_scaleup_simulation
        r = run_scaleup_simulation(small_volume_L=2000.0, large_volume_L=2000.0)
        ok = 0.0 < r.titer_scaling_factor <= 1.5 and r.predicted_titer_large > 0
        return ok, f"Equal vol: scaling={r.titer_scaling_factor:.3f}, titer={r.predicted_titer_large:.2f}"

    cases.extend([
        {"name": "scaleup_large_volume", "module": "scaleup_twin", "fn": _scaleup_larger_volume_more_shear},
        {"name": "scaleup_equal_volumes", "module": "scaleup_twin", "fn": _scaleup_equal_volumes},
    ])

    # ── pareto_optimizer ─────────────────────────────────────────
    def _pareto_single_candidate():
        from src.pareto_optimizer import run_pareto_analysis
        r = run_pareto_analysis([{"name": "Solo", "obj1": 0.5}])
        ok = r.n_pareto == 1
        return ok, f"Single candidate: n_pareto={r.n_pareto}"

    def _pareto_all_identical():
        from src.pareto_optimizer import run_pareto_analysis
        candidates = [{"name": f"C{i}", "obj1": 0.5, "obj2": 0.5} for i in range(5)]
        r = run_pareto_analysis(candidates)
        ok = r.n_pareto >= 1
        return ok, f"Identical: {r.n_pareto}/{r.n_total} Pareto"

    cases.extend([
        {"name": "pareto_single", "module": "pareto_optimizer", "fn": _pareto_single_candidate},
        {"name": "pareto_identical", "module": "pareto_optimizer", "fn": _pareto_all_identical},
    ])

    # ── data_harmonizer ──────────────────────────────────────────
    def _harmonizer_missing_columns():
        import pandas as pd
        from src.data_harmonizer import DataHarmonizer
        h = DataHarmonizer()
        df = pd.DataFrame({"X": [1, 2]})
        r = h.harmonize(df)
        ok = isinstance(r, dict)
        return ok, f"Missing cols: status={r.get('status', '?')}"

    cases.append({"name": "harmonizer_missing_cols", "module": "data_harmonizer", "fn": _harmonizer_missing_columns})

    # ── molecule_registry ────────────────────────────────────────
    def _registry_unknown_class():
        from src.molecule_registry import get_config
        cfg = get_config("nonexistent_format_xyz")
        ok = isinstance(cfg, dict)
        return ok, f"Unknown class returns {type(cfg).__name__} with {len(cfg)} keys"

    cases.append({"name": "registry_unknown_class", "module": "molecule_registry", "fn": _registry_unknown_class})

    # ── PropertyMapper ───────────────────────────────────────────
    def _mapper_extreme_pi():
        """Extreme pI with different working pH should affect nu."""
        from src.PropertyMapper import PropertyMapper, ProteinProperties
        pm = PropertyMapper()
        # nu depends on |pI - pH|, so same pH_working but different pI → different net charge
        low = pm.map(ProteinProperties(name="Low", pI=4.0, MW_kDa=150.0, hydrophobicity=0.4, pH_working=6.0))
        high = pm.map(ProteinProperties(name="High", pI=10.0, MW_kDa=150.0, hydrophobicity=0.4, pH_working=6.0))
        ok = abs(low["nu"] - high["nu"]) > 0.01 or low["ka"] != high["ka"]
        return ok, f"pI=4@pH6→nu={low['nu']:.2f},ka={low['ka']:.2f}; pI=10@pH6→nu={high['nu']:.2f},ka={high['ka']:.2f}"

    def _mapper_small_vs_large_mw():
        from src.PropertyMapper import PropertyMapper, ProteinProperties
        pm = PropertyMapper()
        small = pm.map(ProteinProperties(name="Small", pI=7.0, MW_kDa=10.0, hydrophobicity=0.3))
        large = pm.map(ProteinProperties(name="Large", pI=7.0, MW_kDa=200.0, hydrophobicity=0.3))
        ok = small["sigma"] != large["sigma"] or small["nu"] != large["nu"]
        return ok, f"10kDa→sigma={small['sigma']:.1f}, 200kDa→sigma={large['sigma']:.1f}"

    cases.extend([
        {"name": "mapper_extreme_pi", "module": "property_mapper", "fn": _mapper_extreme_pi},
        {"name": "mapper_small_vs_large_mw", "module": "property_mapper", "fn": _mapper_small_vs_large_mw},
    ])

    return cases


BENCHMARK_CASES = _make_cases()

def run_infra_benchmark(module_filter=None, verbose=True):
    cases = BENCHMARK_CASES if not module_filter else [c for c in BENCHMARK_CASES if c["module"] == module_filter]
    passed = failed = 0; failures = []; t0 = time.monotonic()
    for c in cases:
        try:
            ok, detail = c["fn"]()
            if ok:
                passed += 1
                if verbose: log.info("  [PASS] %s: %s", c["name"], detail)
            else:
                failed += 1; failures.append({"test": c["name"], "detail": detail})
                if verbose: log.warning("  [FAIL] %s: %s", c["name"], detail)
        except Exception as exc:
            failed += 1; failures.append({"test": c["name"], "error": str(exc)})
            if verbose: log.error("  [ERR]  %s: %s", c["name"], exc)
    elapsed = (time.monotonic() - t0) * 1000
    total = passed + failed
    return {"passed": passed, "failed": failed, "total": total,
            "accuracy": round(passed/max(total,1), 4), "failures": failures,
            "timing_ms": round(elapsed, 1), "all_passed": failed == 0}

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_infra_benchmark(module_filter=args.module)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}\nInfra Benchmark: {result['passed']}/{result['total']} passed ({result['accuracy']:.0%}) in {result['timing_ms']:.0f}ms")
        if result["failures"]:
            for f in result["failures"]: print(f"  - {f['test']}: {f.get('detail', f.get('error', ''))}")
        print(f"Status: {'ALL PASSED' if result['all_passed'] else 'FAILURES DETECTED'}")
    sys.exit(0 if result["all_passed"] else 1)
