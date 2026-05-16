"""
developability_benchmark.py  ·  ProtePilot — Developability Benchmark Suite
==============================================================================
Expanded benchmark for the developability assessment pipeline.

Tests:
  - All 9 molecule classes
  - Contract compliance for every output
  - Scoring invariants (grade boundaries, weight sums, inversion)
  - QTPP row counts and format-specific rows
  - Recommendation logic determinism
  - Edge cases (empty inputs, extreme scores)

Usage:
    python -m src.developability_benchmark               # Full benchmark
    python -m src.developability_benchmark --class adc    # Single class
    python -m src.developability_benchmark --quick        # Core only
    python -m src.developability_benchmark --json         # JSON output

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("ProtePilot.DevelopabilityBenchmark")


# ═══════════════════════════════════════════════════════════════════════
#  Reference Feature Sets
# ═══════════════════════════════════════════════════════════════════════

_BASE_FEATURES = {
    "pI": 8.44, "mw_kda": 148.0, "hydrophobicity": 0.34,
    "deam_sites": 2, "ox_sites": 5, "asp_isomerization_sites": 1,
    "beta_sheet_propensity": 1.08, "cdr_hydrophobicity": -0.3,
    "n_glycosylation_sites": 2, "pyroglutamate_risk": 0,
    "acidic_residues": 38, "basic_residues": 48,
    "seq_length": 450, "cysteine_count": 16,
}

_BASE_PREDS = {
    "agg_risk": 0.15, "stability": 0.85, "viscosity_risk": 0.10,
}


def _feats(**overrides):
    """Base features with overrides."""
    d = dict(_BASE_FEATURES)
    d.update(overrides)
    return d


def _preds(**overrides):
    """Base predictions with overrides."""
    d = dict(_BASE_PREDS)
    d.update(overrides)
    return d


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Panel
# ═══════════════════════════════════════════════════════════════════════

BENCHMARK_PANEL: List[Dict[str, Any]] = [
    # ── Standard classes ──────────────────────────────────────────
    {
        "name": "canonical_mab_standard",
        "kwargs": {
            "molecule_name": "Trastuzumab",
            "molecule_class": "canonical_mab",
            "feature_values": _feats(),
            "dev_predictions": _preds(),
        },
        "expected_recommendation": "Proceed",
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "core",
    },
    {
        "name": "bispecific_standard",
        "kwargs": {
            "molecule_name": "Emicizumab",
            "molecule_class": "bispecific",
            "feature_values": _feats(seq_length=900, cysteine_count=32),
            "dev_predictions": _preds(agg_risk=0.25, viscosity_risk=0.20),
        },
        "expected_recommendation": None,  # Any valid
        "expected_min_dims": 6,  # +species_purity
        "expected_min_qtpp": 12,
        "expected_dims_contain": ["species_purity"],
        "category": "core",
    },
    {
        "name": "adc_standard",
        "kwargs": {
            "molecule_name": "T-DM1",
            "molecule_class": "adc",
            "feature_values": _feats(),
            "dev_predictions": _preds(agg_risk=0.20, stability=0.80),
        },
        "expected_recommendation": None,
        "expected_min_dims": 6,  # +conjugation
        "expected_min_qtpp": 11,
        "expected_dims_contain": ["conjugation"],
        "category": "core",
    },
    {
        "name": "fc_fusion_standard",
        "kwargs": {
            "molecule_name": "Etanercept",
            "molecule_class": "fc_fusion",
            "feature_values": _feats(mw_kda=130, seq_length=480),
            "dev_predictions": _preds(agg_risk=0.30, stability=0.75),
        },
        "expected_recommendation": None,
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "core",
    },
    {
        "name": "peptide_standard",
        "kwargs": {
            "molecule_name": "Semaglutide",
            "molecule_class": "peptide",
            "feature_values": _feats(seq_length=31, mw_kda=4.1, cysteine_count=0),
            "dev_predictions": _preds(agg_risk=0.05, stability=0.70, viscosity_risk=0.02),
        },
        "expected_recommendation": None,
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "core",
    },
    {
        "name": "single_domain_standard",
        "kwargs": {
            "molecule_name": "Caplacizumab",
            "molecule_class": "single_domain",
            "feature_values": _feats(seq_length=120, mw_kda=12.8, cysteine_count=2),
            "dev_predictions": _preds(agg_risk=0.35, stability=0.80),
        },
        "expected_recommendation": None,
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "core",
    },
    {
        "name": "fusion_protein_standard",
        "kwargs": {
            "molecule_name": "BiTE-fusion",
            "molecule_class": "fusion_protein",
            "feature_values": _feats(seq_length=600, mw_kda=55),
            "dev_predictions": _preds(agg_risk=0.30, stability=0.70),
        },
        "expected_recommendation": None,
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "core",
    },
    {
        "name": "engineered_scaffold_standard",
        "kwargs": {
            "molecule_name": "DARPin-X",
            "molecule_class": "engineered_scaffold",
            "feature_values": _feats(seq_length=170, mw_kda=18, cysteine_count=0),
            "dev_predictions": _preds(agg_risk=0.30, stability=0.78),
        },
        "expected_recommendation": None,
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "core",
    },
    {
        "name": "unknown_class",
        "kwargs": {
            "molecule_name": "UnknownMol",
            "molecule_class": "unknown",
            "feature_values": _feats(),
            "dev_predictions": _preds(),
        },
        "expected_recommendation": None,
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "core",
    },

    # ── Recommendation logic tests ────────────────────────────────
    {
        "name": "high_risk_optimize",
        "kwargs": {
            "molecule_name": "HighRiskMol",
            "molecule_class": "canonical_mab",
            "feature_values": _feats(
                hydrophobicity=0.55, deam_sites=8, ox_sites=20,
                cysteine_count=17, seq_length=1400,
            ),
            "dev_predictions": _preds(agg_risk=0.70, stability=0.30, viscosity_risk=0.60),
        },
        "expected_recommendation": "Optimize before proceeding",
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "logic",
    },
    {
        "name": "low_risk_proceed",
        "kwargs": {
            "molecule_name": "PerfectMol",
            "molecule_class": "canonical_mab",
            "feature_values": _feats(
                hydrophobicity=0.28, deam_sites=1, ox_sites=3,
                cdr_hydrophobicity=-0.5,
            ),
            "dev_predictions": _preds(agg_risk=0.05, stability=0.95, viscosity_risk=0.05),
        },
        "expected_recommendation": "Proceed",
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "logic",
    },

    # ── Edge cases ────────────────────────────────────────────────
    {
        "name": "minimal_input",
        "kwargs": {
            "molecule_name": "MinimalMol",
            "molecule_class": "canonical_mab",
        },
        "expected_recommendation": None,  # Should still produce valid output
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "edge",
    },
    {
        "name": "with_analytical_data",
        "kwargs": {
            "molecule_name": "FullDataMol",
            "molecule_class": "canonical_mab",
            "feature_values": _feats(),
            "dev_predictions": _preds(),
            "analytical_results": {
                "sec_monomer_pct": 97.5, "sec_hmw_pct": 2.5,
                "cief_main_pct": 72.0, "cief_acidic_pct": 18.0,
                "cesds_intact_pct": 98.5,
            },
        },
        "expected_recommendation": None,
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "edge",
    },
    {
        "name": "with_stability_data",
        "kwargs": {
            "molecule_name": "StabMol",
            "molecule_class": "canonical_mab",
            "feature_values": _feats(),
            "dev_predictions": _preds(),
            "stability_results": {"shelf_life_months": 26},
        },
        "expected_recommendation": None,
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "edge",
    },
    {
        "name": "with_pk_data",
        "kwargs": {
            "molecule_name": "PKMol",
            "molecule_class": "canonical_mab",
            "feature_values": _feats(),
            "dev_predictions": _preds(),
            "pk_results": {"half_life_days": 18.5},
        },
        "expected_recommendation": None,
        "expected_min_dims": 5,
        "expected_min_qtpp": 11,  # +1 for PK row
        "category": "edge",
    },
    {
        "name": "with_ada_high",
        "kwargs": {
            "molecule_name": "ADAMol",
            "molecule_class": "canonical_mab",
            "feature_values": _feats(),
            "dev_predictions": _preds(),
            "ada_results": {"ada_risk_level": "High", "ada_risk_score": 0.85},
        },
        "expected_recommendation": None,
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "edge",
    },
    {
        "name": "odd_cysteine_aggregation",
        "kwargs": {
            "molecule_name": "OddCysMol",
            "molecule_class": "canonical_mab",
            "feature_values": _feats(cysteine_count=17),
            "dev_predictions": _preds(agg_risk=0.20),
        },
        "expected_recommendation": None,
        "expected_min_dims": 5,
        "expected_min_qtpp": 10,
        "category": "edge",
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark(
    panel: Optional[List[Dict[str, Any]]] = None,
    filter_class: Optional[str] = None,
    filter_category: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run developability benchmark and return results."""
    from src.developability_core import assess_developability
    from src.developability_contract import validate_assessment_output

    cases = panel or BENCHMARK_PANEL
    if filter_class:
        cases = [c for c in cases if c["kwargs"].get("molecule_class") == filter_class]
    if filter_category:
        cases = [c for c in cases if c.get("category") == filter_category]

    passed = 0
    failed = 0
    failures = []
    t0 = time.monotonic()

    for case in cases:
        test_name = case["name"]
        try:
            result = assess_developability(**case["kwargs"])
            result_dict = result.to_dict()
            errors = []

            # 1. Contract compliance
            violations = validate_assessment_output(result_dict)
            if violations:
                errors.extend([f"contract: {v}" for v in violations])

            # 2. Expected recommendation
            exp_rec = case.get("expected_recommendation")
            if exp_rec and result.recommendation != exp_rec:
                errors.append(f"recommendation: expected={exp_rec}, got={result.recommendation}")

            # 3. Minimum dimensions
            exp_dims = case.get("expected_min_dims", 5)
            if len(result.dimensions) < exp_dims:
                errors.append(f"dimensions: expected≥{exp_dims}, got={len(result.dimensions)}")

            # 4. Minimum QTPP rows
            exp_qtpp = case.get("expected_min_qtpp", 10)
            if len(result.qtpp) < exp_qtpp:
                errors.append(f"qtpp_rows: expected≥{exp_qtpp}, got={len(result.qtpp)}")

            # 5. Expected dimensions contain
            exp_contain = case.get("expected_dims_contain", [])
            dim_names = {d.name for d in result.dimensions}
            for c_dim in exp_contain:
                if c_dim not in dim_names:
                    errors.append(f"missing expected dimension: {c_dim}")

            if errors:
                failed += 1
                failures.append({"test": test_name, "errors": errors})
                if verbose:
                    log.warning("  [FAIL] %s: %s", test_name, "; ".join(errors))
            else:
                passed += 1
                if verbose:
                    log.info("  [PASS] %s: score=%.3f (%s), rec=%s, %d dims, %d qtpp",
                             test_name, result.composite_score, result.composite_grade,
                             result.recommendation, len(result.dimensions), len(result.qtpp))

        except Exception as exc:
            failed += 1
            failures.append({"test": test_name, "error": f"Exception: {exc}"})
            if verbose:
                log.error("  [ERR]  %s: %s", test_name, exc)

    elapsed_ms = (time.monotonic() - t0) * 1000
    total = passed + failed
    accuracy = passed / total if total > 0 else 0.0

    return {
        "passed": passed,
        "failed": failed,
        "total": total,
        "accuracy": round(accuracy, 4),
        "failures": failures,
        "timing_ms": round(elapsed_ms, 1),
        "all_passed": failed == 0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Scoring Invariant Checks
# ═══════════════════════════════════════════════════════════════════════

def run_invariant_checks(verbose: bool = True) -> Dict[str, Any]:
    """Verify scoring invariants hold across all classes."""
    from src.developability_core import assess_developability, _grade_score

    checks_passed = 0
    checks_failed = 0
    errors = []

    # 1. Grade boundary consistency
    try:
        assert _grade_score(0.0) == ("Low", "#10B981")
        assert _grade_score(0.24) == ("Low", "#10B981")
        assert _grade_score(0.25) == ("Medium", "#F59E0B")
        assert _grade_score(0.54) == ("Medium", "#F59E0B")
        assert _grade_score(0.55) == ("High", "#EF4444")
        assert _grade_score(1.0) == ("High", "#EF4444")
        checks_passed += 1
        if verbose:
            log.info("  [PASS] grade_boundary_consistency")
    except AssertionError as e:
        checks_failed += 1
        errors.append(f"grade_boundaries: {e}")

    # 2. Stability inversion
    try:
        result = assess_developability(
            molecule_name="InvTest",
            molecule_class="canonical_mab",
            feature_values=_feats(),
            dev_predictions={"agg_risk": 0.15, "stability": 0.90, "viscosity_risk": 0.10},
        )
        stab_dim = next(d for d in result.dimensions if d.name == "stability")
        # stability=0.90 → risk = 0.10, should be Low
        assert stab_dim.score < 0.25, f"stability=0.90 should give risk<0.25, got {stab_dim.score}"
        checks_passed += 1
        if verbose:
            log.info("  [PASS] stability_inversion (0.90 → risk=%.2f)", stab_dim.score)
    except Exception as e:
        checks_failed += 1
        errors.append(f"stability_inversion: {e}")

    # 3. Weight sum for all classes
    from src.molecule_classifier import get_risk_weights, MoleculeClass
    for mc in MoleculeClass:
        try:
            w = get_risk_weights(mc)
            total = sum(w.values())
            assert abs(total - 1.0) < 0.02, f"{mc.value}: weights sum to {total}"
            checks_passed += 1
        except Exception as e:
            checks_failed += 1
            errors.append(f"weight_sum({mc.value}): {e}")

    # 4. Format caveat for non-mAb
    try:
        result = assess_developability(
            molecule_name="CaveatTest",
            molecule_class="peptide",
            feature_values=_feats(seq_length=31),
            dev_predictions=_preds(),
        )
        # At least one QTPP row should have format caveat
        caveat_rows = [r for r in result.qtpp if "interpret with caution" in (r.justification or "")]
        assert len(caveat_rows) > 0, "No format caveat found for peptide QTPP"
        checks_passed += 1
        if verbose:
            log.info("  [PASS] format_caveat_peptide (%d rows with caveat)", len(caveat_rows))
    except Exception as e:
        checks_failed += 1
        errors.append(f"format_caveat: {e}")

    return {
        "passed": checks_passed,
        "failed": checks_failed,
        "total": checks_passed + checks_failed,
        "errors": errors,
        "all_passed": checks_failed == 0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ProtePilot — Developability Benchmark",
    )
    parser.add_argument("--class", dest="filter_class", default=None,
                        help="Run tests for a specific class only")
    parser.add_argument("--category", default=None,
                        help="Run tests in category: core, logic, edge")
    parser.add_argument("--quick", action="store_true",
                        help="Core tests only")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-test output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(name)s | %(message)s",
    )

    verbose = not args.quiet
    category = args.category or ("core" if args.quick else None)

    log.info("Running developability benchmark...")
    bench = run_benchmark(
        filter_class=args.filter_class,
        filter_category=category,
        verbose=verbose,
    )

    log.info("\nRunning scoring invariant checks...")
    invariants = run_invariant_checks(verbose=verbose)

    combined = {
        "benchmark": bench,
        "invariants": invariants,
        "overall_passed": bench["all_passed"] and invariants["all_passed"],
    }

    if args.json:
        print(json.dumps(combined, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Developability Benchmark Results")
        print(f"{'='*60}")
        print(f"  Panel:      {bench['passed']}/{bench['total']} passed "
              f"({bench['accuracy']:.0%}) in {bench['timing_ms']:.0f}ms")
        print(f"  Invariants: {invariants['passed']}/{invariants['total']} passed")
        if bench["failures"]:
            print(f"\n  Failures:")
            for f in bench["failures"]:
                print(f"    - {f['test']}: {f.get('errors', f.get('error', ''))}")
        if invariants["errors"]:
            print(f"\n  Invariant errors:")
            for e in invariants["errors"]:
                print(f"    - {e}")
        status = "ALL PASSED" if combined["overall_passed"] else "FAILURES DETECTED"
        print(f"\n  Status: {status}")

    sys.exit(0 if combined["overall_passed"] else 1)


if __name__ == "__main__":
    main()
