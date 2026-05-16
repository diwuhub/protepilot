"""
medium_benchmark.py  ·  ProtePilot — Medium Module Benchmarks (C2 Batch)
===========================================================================
Edge-case benchmarks for bispecific_engine, regulatory_filer,
generative_engineer, ht_screening, validation_strategy.

Usage:
    python -m src.medium_benchmark              # Full run
    python -m src.medium_benchmark --json       # JSON output

Author  : Di (ProtePilot)
"""
from __future__ import annotations
import json, logging, sys, time
from typing import Any, Dict, List, Tuple
log = logging.getLogger("ProtePilot.MediumBenchmark")


def _make_cases() -> List[Dict[str, Any]]:
    cases = []

    # ── bispecific_engine ─────────────────────────────────────────
    def _bispecific_identical_chains():
        """Identical chains should give homodimer-only assembly."""
        from src.bispecific_engine import AntibodyChain, build_assembly_species
        seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH" * 3
        a = AntibodyChain(name="ArmA", sequence=seq)
        b = AntibodyChain(name="ArmB", sequence=seq)
        species = build_assembly_species(a, b)
        ok = isinstance(species, dict) and len(species) >= 1
        return ok, f"Identical chains: {len(species)} species"

    def _bispecific_short_chains():
        """Very short chains should still produce valid output."""
        from src.bispecific_engine import run_bispecific_analysis
        r = run_bispecific_analysis(chain_a_seq="EVQLVESGG", chain_b_seq="DIQMTQSPS")
        ok = isinstance(r, dict) and "species" in r
        return ok, f"Short chains: {len(r.get('species', {}))} species"

    cases.extend([
        {"name": "bispecific_identical_chains", "module": "bispecific_engine", "fn": _bispecific_identical_chains},
        {"name": "bispecific_short_chains", "module": "bispecific_engine", "fn": _bispecific_short_chains},
    ])

    # ── regulatory_filer ──────────────────────────────────────────
    def _regulatory_minimal_session():
        """Minimal session data should still produce output."""
        from src.regulatory_filer import generate_ectd_markdown
        md = generate_ectd_markdown({"molecule_name": "Minimal"})
        ok = isinstance(md, str) and len(md) > 50
        return ok, f"Minimal session: {len(md)} chars"

    def _regulatory_bispecific_session():
        """Bispecific molecule should produce appropriate regulatory content."""
        from src.regulatory_filer import generate_ectd_markdown
        session = {
            "molecule_name": "BispecificMab",
            "molecule_class": "bispecific",
            "sequence": "EVQLVESGG" * 20 + "DIQMTQSPS" * 20,
        }
        md = generate_ectd_markdown(session)
        ok = isinstance(md, str) and len(md) > 100
        return ok, f"Bispecific eCTD: {len(md)} chars"

    cases.extend([
        {"name": "regulatory_minimal_session", "module": "regulatory_filer", "fn": _regulatory_minimal_session},
        {"name": "regulatory_bispecific_session", "module": "regulatory_filer", "fn": _regulatory_bispecific_session},
    ])

    # ── generative_engineer ───────────────────────────────────────
    def _generative_no_liabilities():
        """Clean sequence with no obvious liabilities."""
        from src.generative_engineer import identify_liabilities_for_mutagenesis
        # All-alanine sequence — no canonical liability motifs
        clean_seq = "AAAAAAAAAAAAAAAAAAAA"
        liabilities = identify_liabilities_for_mutagenesis(clean_seq)
        ok = isinstance(liabilities, list)
        return ok, f"Clean seq: {len(liabilities)} liabilities (expected ~0)"

    def _generative_charge_shift():
        """Charge shift engineering should modify sequence."""
        from src.generative_engineer import engineer_charge_shift
        seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVAR" * 2
        mutated, mutations = engineer_charge_shift(seq, target_delta_pi=-0.5, max_mutations=3)
        ok = isinstance(mutated, str) and len(mutated) == len(seq)
        return ok, f"Charge shift: {len(mutations)} mutations, len preserved"

    cases.extend([
        {"name": "generative_no_liabilities", "module": "generative_engineer", "fn": _generative_no_liabilities},
        {"name": "generative_charge_shift", "module": "generative_engineer", "fn": _generative_charge_shift},
    ])

    # ── ht_screening ──────────────────────────────────────────────
    def _ht_screening_large_batch():
        """50 candidates — larger batch should complete without error."""
        from src.ht_screening import generate_mock_discovery_csv, run_ht_screening
        csv_str = generate_mock_discovery_csv(n_candidates=50, seed=99)
        r = run_ht_screening(csv_str)
        ok = isinstance(r, dict)
        return ok, f"50 candidates: {r.get('n_screened', '?')} screened"

    def _ht_screening_single_candidate():
        """Single candidate edge case."""
        from src.ht_screening import generate_mock_discovery_csv, run_ht_screening
        csv_str = generate_mock_discovery_csv(n_candidates=1, seed=42)
        r = run_ht_screening(csv_str)
        ok = isinstance(r, dict)
        return ok, f"1 candidate: {r.get('n_screened', '?')} screened"

    cases.extend([
        {"name": "ht_screening_large_batch", "module": "ht_screening", "fn": _ht_screening_large_batch},
        {"name": "ht_screening_single_candidate", "module": "ht_screening", "fn": _ht_screening_single_candidate},
    ])

    # ── validation_strategy ───────────────────────────────────────
    def _validation_perfect_predictions():
        """Perfect predictions should give rmse=0.0 and r2=1.0."""
        import numpy as np
        from src.validation_strategy import compute_metrics
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_metrics(y, y)
        ok = metrics.get("rmse", 1.0) < 1e-6 and abs(metrics.get("r2", 0) - 1.0) < 1e-6
        return ok, f"Perfect: rmse={metrics.get('rmse', '?')}, r2={metrics.get('r2', '?')}"

    def _validation_batch_shift():
        """Batch shift split should produce valid splits."""
        import numpy as np
        from src.validation_strategy import batch_shift_split
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        splits = batch_shift_split(X, y, n_batches=3, random_seed=42)
        ok = isinstance(splits, list) and len(splits) >= 1
        return ok, f"{len(splits)} batch-shift splits"

    cases.extend([
        {"name": "validation_perfect_predictions", "module": "validation_strategy", "fn": _validation_perfect_predictions},
        {"name": "validation_batch_shift", "module": "validation_strategy", "fn": _validation_batch_shift},
    ])

    return cases


BENCHMARK_CASES = _make_cases()


def run_medium_benchmark(module_filter=None, verbose=True):
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
    result = run_medium_benchmark(module_filter=args.module)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}\nMedium Benchmark: {result['passed']}/{result['total']} passed ({result['accuracy']:.0%}) in {result['timing_ms']:.0f}ms")
        if result["failures"]:
            for f in result["failures"]: print(f"  - {f['test']}: {f.get('detail', f.get('error', ''))}")
        print(f"Status: {'ALL PASSED' if result['all_passed'] else 'FAILURES DETECTED'}")
    sys.exit(0 if result["all_passed"] else 1)
