"""
twin_contracts.py  ·  ProtePilot — Twin Engine Behavioral Contracts
=====================================================================
Formal specification of behavioral guarantees for all twin engine
modules. Each guarantee is a testable assertion.

Covered modules:
    stability_twin      — ICH stability simulation
    upstream_twin       — Fed-batch bioreactor simulation
    immunogenicity_twin — T-cell epitope & ADA risk
    preclinical_twin    — PK clearance & half-life
    formulation_twin    — Buffer/excipient screening
    cogs_twin           — Manufacturing cost estimation

Usage:
    python -m src.twin_contracts                    # Run all contract checks
    python -m src.twin_contracts --module stability # Single module

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Tuple

log = logging.getLogger("ProtePilot.TwinContracts")


# ═══════════════════════════════════════════════════════════════════════
#  Contract Definitions
# ═══════════════════════════════════════════════════════════════════════

# Each contract: (name, module, description, check_function)
# check_function() -> (passed: bool, detail: str)

def _check_stability_result_schema() -> Tuple[bool, str]:
    """DualConditionResult must have shelf_life and grade."""
    from src.stability_twin import run_stability_study
    r = run_stability_study(pI=7.0)
    required = {"predicted_shelf_life_months", "overall_stability_grade", "long_term", "accelerated"}
    missing = required - set(vars(r).keys())
    if missing:
        return False, f"Missing fields: {missing}"
    if not isinstance(r.predicted_shelf_life_months, (int, float)):
        return False, f"shelf_life not numeric: {type(r.predicted_shelf_life_months)}"
    valid_grades = {"Excellent", "Good", "Acceptable", "At Risk", "Poor", "Fail", "A", "B", "C", "D", "F"}
    if r.overall_stability_grade not in valid_grades:
        return False, f"Invalid grade: {r.overall_stability_grade}"
    return True, f"shelf_life={r.predicted_shelf_life_months}mo, grade={r.overall_stability_grade}"


def _check_stability_shelf_life_range() -> Tuple[bool, str]:
    """Predicted shelf life must be 0-36 months."""
    from src.stability_twin import run_stability_study
    r = run_stability_study(pI=6.5)
    sl = r.predicted_shelf_life_months
    if not (0 <= sl <= 36):
        return False, f"shelf_life={sl} outside [0, 36]"
    return True, f"shelf_life={sl} within range"


def _check_stability_hmw_monotonic() -> Tuple[bool, str]:
    """HMW% must be non-decreasing over time."""
    from src.stability_twin import run_stability_study
    r = run_stability_study(pI=7.0)
    for cond in [r.long_term, r.accelerated]:
        hmw_values = [tp.sec_hmw_pct for tp in cond.timepoints]
        for i in range(1, len(hmw_values)):
            if hmw_values[i] < hmw_values[i-1] - 0.01:
                return False, f"HMW decreased: {hmw_values[i-1]:.2f} → {hmw_values[i]:.2f} at t={i}"
    return True, "HMW monotonically non-decreasing"


def _check_upstream_result_schema() -> Tuple[bool, str]:
    """BioreactorResult must have titer, VCD, and time arrays."""
    from src.upstream_twin import run_upstream_simulation
    r = run_upstream_simulation(sequence="ACDEFGHIKLMNPQRSTVWY" * 10, molecule_class="canonical_mab")
    required_attrs = {"final_titer", "peak_vcd", "time_days", "titer", "vcd"}
    missing = required_attrs - set(vars(r).keys())
    if missing:
        return False, f"Missing fields: {missing}"
    if r.final_titer <= 0:
        return False, f"final_titer={r.final_titer} <= 0"
    if len(r.time_days) < 2:
        return False, f"time_days has only {len(r.time_days)} points"
    return True, f"titer={r.final_titer:.2f} g/L, VCD_peak={r.peak_vcd:.1f}"


def _check_upstream_titer_positive() -> Tuple[bool, str]:
    """Final titer must be > 0 for any valid sequence."""
    from src.upstream_twin import run_upstream_simulation
    for cls in ["canonical_mab", "peptide", "fc_fusion"]:
        r = run_upstream_simulation(sequence="ACDEFGHIKLMNPQRSTVWY" * 10, molecule_class=cls)
        if r.final_titer <= 0:
            return False, f"titer <= 0 for {cls}"
    return True, "All classes produce positive titer"


def _check_immunogenicity_result_schema() -> Tuple[bool, str]:
    """ImmunogenicityResult must have ada_risk_level and hotspot count."""
    from src.immunogenicity_twin import run_immunogenicity_assessment
    r = run_immunogenicity_assessment(sequence="EVQLVESGGGLVQPGG" * 8)
    if r.ada_risk_level not in ("Low", "Medium", "High", "Very High"):
        return False, f"Invalid ada_risk_level: {r.ada_risk_level}"
    if not isinstance(r.n_high_risk, int):
        return False, f"n_high_risk not int: {type(r.n_high_risk)}"
    if not isinstance(r.ada_risk_score, (int, float)):
        return False, f"ada_risk_score not numeric"
    return True, f"ada_risk={r.ada_risk_level}, hotspots={r.n_high_risk}"


def _check_immunogenicity_score_range() -> Tuple[bool, str]:
    """ADA risk score must be >= 0."""
    from src.immunogenicity_twin import run_immunogenicity_assessment
    r = run_immunogenicity_assessment(sequence="ACDEFGHIKLMNPQRSTVWY" * 5)
    if r.ada_risk_score < 0:
        return False, f"ada_risk_score={r.ada_risk_score} < 0"
    return True, f"ada_risk_score={r.ada_risk_score:.2f} >= 0"


def _check_preclinical_half_life_range() -> Tuple[bool, str]:
    """Human half-life prediction must be in [0.1, 60] days."""
    from src.preclinical_twin import predict_human_half_life
    result = predict_human_half_life(global_pi=7.0, hydrophobicity=0.3, liability_density=0.01)
    hl = result["half_life_days"]
    if not (0.1 <= hl <= 60):
        return False, f"half_life={hl} outside [0.1, 60] days"
    return True, f"half_life={hl:.1f} days within range"


def _check_preclinical_pi_penalty() -> Tuple[bool, str]:
    """Extreme pI should reduce half-life vs optimal pI."""
    from src.preclinical_twin import predict_human_half_life
    r_normal = predict_human_half_life(global_pi=7.0, hydrophobicity=0.3, liability_density=0.01)
    r_extreme = predict_human_half_life(global_pi=9.5, hydrophobicity=0.3, liability_density=0.01)
    hl_normal = r_normal["half_life_days"]
    hl_extreme = r_extreme["half_life_days"]
    if hl_extreme >= hl_normal:
        return False, f"Extreme pI ({hl_extreme:.1f}d) not penalized vs normal ({hl_normal:.1f}d)"
    return True, f"pI penalty: normal={hl_normal:.1f}d > extreme={hl_extreme:.1f}d"


def _check_formulation_result_schema() -> Tuple[bool, str]:
    """FormulationAssessment must return dict with status."""
    from src.formulation_twin import run_formulation_assessment
    r = run_formulation_assessment(pI=7.0, buffer_ph=6.0, buffer_type="histidine")
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r)}"
    if "status" not in r:
        return False, f"No 'status' key in output: {list(r.keys())[:5]}"
    return True, f"Returned dict with keys: {sorted(r.keys())[:6]}"


def _check_cogs_result_schema() -> Tuple[bool, str]:
    """COGSResult must have cogs_per_gram and total_batch_cost."""
    from src.cogs_twin import run_cogs_analysis
    r = run_cogs_analysis(titer_g_per_L=3.0, bioreactor_volume_L=2000)
    required = {"cogs_per_gram", "total_batch_cost", "batch_output_g", "cost_rating"}
    missing = required - set(vars(r).keys())
    if missing:
        return False, f"Missing fields: {missing}"
    if r.cogs_per_gram <= 0:
        return False, f"cogs_per_gram={r.cogs_per_gram} <= 0"
    valid_ratings = {"Excellent", "Good", "Acceptable", "Poor", "Non-viable",
                      "Low", "Medium", "High", "Very High"}
    if r.cost_rating not in valid_ratings:
        return False, f"Invalid cost_rating: {r.cost_rating}"
    return True, f"cogs/g=${r.cogs_per_gram:.0f}, rating={r.cost_rating}"


def _check_cogs_titer_sensitivity() -> Tuple[bool, str]:
    """Higher titer should reduce cost per gram."""
    from src.cogs_twin import run_cogs_analysis
    r_low = run_cogs_analysis(titer_g_per_L=1.0, bioreactor_volume_L=2000)
    r_high = run_cogs_analysis(titer_g_per_L=5.0, bioreactor_volume_L=2000)
    if r_high.cogs_per_gram >= r_low.cogs_per_gram:
        return False, f"Higher titer didn't reduce cost: ${r_low.cogs_per_gram:.0f} vs ${r_high.cogs_per_gram:.0f}"
    return True, f"titer 1→5 g/L: ${r_low.cogs_per_gram:.0f} → ${r_high.cogs_per_gram:.0f}/g"


# ═══════════════════════════════════════════════════════════════════════
#  Contract Registry
# ═══════════════════════════════════════════════════════════════════════

CONTRACTS: List[Dict[str, Any]] = [
    # Stability
    {"name": "stability_result_schema", "module": "stability", "fn": _check_stability_result_schema,
     "desc": "DualConditionResult has shelf_life_months and grade"},
    {"name": "stability_shelf_life_range", "module": "stability", "fn": _check_stability_shelf_life_range,
     "desc": "Shelf life in [0, 36] months"},
    {"name": "stability_hmw_monotonic", "module": "stability", "fn": _check_stability_hmw_monotonic,
     "desc": "HMW% non-decreasing over time"},

    # Upstream
    {"name": "upstream_result_schema", "module": "upstream", "fn": _check_upstream_result_schema,
     "desc": "BioreactorResult has titer, VCD, time arrays"},
    {"name": "upstream_titer_positive", "module": "upstream", "fn": _check_upstream_titer_positive,
     "desc": "Final titer > 0 for all molecule classes"},

    # Immunogenicity
    {"name": "immunogenicity_result_schema", "module": "immunogenicity", "fn": _check_immunogenicity_result_schema,
     "desc": "ImmunogenicityResult has ada_risk_level and hotspots"},
    {"name": "immunogenicity_score_range", "module": "immunogenicity", "fn": _check_immunogenicity_score_range,
     "desc": "ADA risk score >= 0"},

    # Preclinical
    {"name": "preclinical_half_life_range", "module": "preclinical", "fn": _check_preclinical_half_life_range,
     "desc": "Half-life in [0.1, 60] days"},
    {"name": "preclinical_pi_penalty", "module": "preclinical", "fn": _check_preclinical_pi_penalty,
     "desc": "Extreme pI reduces half-life"},

    # Formulation
    {"name": "formulation_result_schema", "module": "formulation", "fn": _check_formulation_result_schema,
     "desc": "Returns dict with conditions/results"},

    # COGS
    {"name": "cogs_result_schema", "module": "cogs", "fn": _check_cogs_result_schema,
     "desc": "COGSResult has cogs_per_gram and cost_rating"},
    {"name": "cogs_titer_sensitivity", "module": "cogs", "fn": _check_cogs_titer_sensitivity,
     "desc": "Higher titer reduces cost per gram"},
]


# ═══════════════════════════════════════════════════════════════════════
#  Contract Runner
# ═══════════════════════════════════════════════════════════════════════

def run_twin_contracts(module_filter: str = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Run all twin engine contracts.

    Parameters
    ----------
    module_filter : str, optional
        Only run contracts for this module (stability, upstream, etc.)
    verbose : bool
        Print per-check results.

    Returns
    -------
    dict with passed, failed, total, errors, all_passed
    """
    checks = CONTRACTS
    if module_filter:
        checks = [c for c in checks if c["module"] == module_filter]

    passed = 0
    failed = 0
    errors = []

    for contract in checks:
        name = contract["name"]
        try:
            ok, detail = contract["fn"]()
            if ok:
                passed += 1
                if verbose:
                    log.info("  [PASS] %s: %s", name, detail)
            else:
                failed += 1
                errors.append(f"{name}: {detail}")
                if verbose:
                    log.warning("  [FAIL] %s: %s", name, detail)
        except Exception as exc:
            failed += 1
            errors.append(f"{name}: Exception: {exc}")
            if verbose:
                log.error("  [ERR]  %s: %s", name, exc)

    return {
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "errors": errors,
        "all_passed": failed == 0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Twin Engine Contract Checks")
    parser.add_argument("--module", default=None,
                        choices=["stability", "upstream", "immunogenicity",
                                 "preclinical", "formulation", "cogs"],
                        help="Check contracts for a single module")
    args = parser.parse_args()

    log.info("Running twin engine contracts%s...",
             f" (module={args.module})" if args.module else "")
    result = run_twin_contracts(module_filter=args.module)

    print(f"\n{'='*50}")
    print(f"Twin Engine Contracts: {result['passed']}/{result['total']} passed")
    if result["errors"]:
        print("Failures:")
        for e in result["errors"]:
            print(f"  - {e}")
    print(f"Status: {'ALL PASSED' if result['all_passed'] else 'FAILURES DETECTED'}")

    sys.exit(0 if result["all_passed"] else 1)
