#!/usr/bin/env python3
"""
run_all_checks.py  ·  ProtePilot — Unified Quality Gate
==========================================================
Runs all contract, benchmark, selftest, and audit suites in sequence.
Returns exit code 0 only if ALL suites pass.

Usage:
    python run_all_checks.py              # Full suite
    python run_all_checks.py --fast       # Contracts + audits only (skip benchmarks)
    python run_all_checks.py --json       # JSON output

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List

log = logging.getLogger("ProtePilot.QualityGate")
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")


# ═══════════════════════════════════════════════════════════════════════
#  Suite Registry
# ═══════════════════════════════════════════════════════════════════════

def _run_suite(name: str, run_fn, **kwargs) -> Dict[str, Any]:
    """Run a single suite and capture result."""
    t0 = time.monotonic()
    try:
        result = run_fn(**kwargs)
        elapsed = (time.monotonic() - t0) * 1000
        passed = result.get("passed", 0)
        total = result.get("total", 0)
        all_ok = result.get("all_passed", False)
        return {
            "name": name,
            "passed": passed,
            "total": total,
            "all_passed": all_ok,
            "timing_ms": round(elapsed, 1),
            "errors": result.get("errors", result.get("failures", [])),
        }
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "name": name,
            "passed": 0,
            "total": 1,
            "all_passed": False,
            "timing_ms": round(elapsed, 1),
            "errors": [f"Suite crashed: {exc}"],
        }


def run_all_checks(fast: bool = False) -> Dict[str, Any]:
    """
    Run all quality gate suites.

    Parameters
    ----------
    fast : bool
        If True, skip benchmark suites (only contracts + audits).

    Returns
    -------
    dict with suites, total_passed, total_total, all_passed, timing_ms
    """
    suites: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    # ── 1. Dependency Audit ──────────────────────────────────────
    log.info("=" * 60)
    log.info("Suite 1/11: Dependency Audit")
    log.info("=" * 60)
    try:
        from src.dependency_audit import run_full_audit
        r = run_full_audit()
        err_count = r.get("error_count", r.get("errors", 0))
        warn_count = r.get("warning_count", r.get("warnings", 0))
        all_ok = r.get("all_passed", err_count == 0)
        suites.append({
            "name": "dependency_audit",
            "passed": 1 if all_ok else 0,
            "total": 1,
            "all_passed": all_ok,
            "timing_ms": 0,
            "errors": [f"{err_count} errors, {warn_count} warnings"] if not all_ok else [],
        })
    except Exception as exc:
        suites.append({"name": "dependency_audit", "passed": 0, "total": 1,
                        "all_passed": False, "timing_ms": 0, "errors": [str(exc)]})

    # ── 2. Twin Contracts ────────────────────────────────────────
    log.info("=" * 60)
    log.info("Suite 2/11: Twin Engine Contracts")
    log.info("=" * 60)
    from src.twin_contracts import run_twin_contracts
    suites.append(_run_suite("twin_contracts", run_twin_contracts, verbose=True))

    # ── 3. Report & Bulk Contracts ───────────────────────────────
    log.info("=" * 60)
    log.info("Suite 3/11: Report & Bulk Contracts")
    log.info("=" * 60)
    from src.report_bulk_contracts import run_report_bulk_contracts
    suites.append(_run_suite("report_bulk_contracts", run_report_bulk_contracts, verbose=True))

    # ── 4. Auxiliary Contracts ───────────────────────────────────
    log.info("=" * 60)
    log.info("Suite 4/11: Auxiliary Contracts")
    log.info("=" * 60)
    from src.auxiliary_contracts import run_auxiliary_contracts
    suites.append(_run_suite("auxiliary_contracts", run_auxiliary_contracts, verbose=True))

    # ── 5. Infra Contracts ─────────────────────────────────────
    log.info("=" * 60)
    log.info("Suite 5/11: Infrastructure Contracts")
    log.info("=" * 60)
    from src.infra_contracts import run_infra_contracts
    suites.append(_run_suite("infra_contracts", run_infra_contracts, verbose=True))

    # ── 6. Medium Contracts ────────────────────────────────────
    log.info("=" * 60)
    log.info("Suite 6/11: Medium Module Contracts")
    log.info("=" * 60)
    from src.medium_contracts import run_medium_contracts
    suites.append(_run_suite("medium_contracts", run_medium_contracts, verbose=True))

    if not fast:
        # ── 7. Twin Benchmarks ───────────────────────────────────
        log.info("=" * 60)
        log.info("Suite 7/11: Twin Engine Benchmarks")
        log.info("=" * 60)
        from src.twin_benchmark import run_twin_benchmark
        suites.append(_run_suite("twin_benchmarks", run_twin_benchmark, verbose=True))

        # ── 8. Report & Bulk Benchmarks ──────────────────────────
        log.info("=" * 60)
        log.info("Suite 8/11: Report & Bulk Benchmarks")
        log.info("=" * 60)
        from src.report_bulk_benchmark import run_report_bulk_benchmark
        suites.append(_run_suite("report_bulk_benchmarks", run_report_bulk_benchmark, verbose=True))

        # ── 9. Auxiliary Benchmarks ──────────────────────────────
        log.info("=" * 60)
        log.info("Suite 9/11: Auxiliary Benchmarks")
        log.info("=" * 60)
        from src.auxiliary_benchmark import run_auxiliary_benchmark
        suites.append(_run_suite("auxiliary_benchmarks", run_auxiliary_benchmark, verbose=True))

        # ── 10. Infra Benchmarks ────────────────────────────────
        log.info("=" * 60)
        log.info("Suite 10/11: Infrastructure Benchmarks")
        log.info("=" * 60)
        from src.infra_benchmark import run_infra_benchmark
        suites.append(_run_suite("infra_benchmarks", run_infra_benchmark, verbose=True))

        # ── 11. Medium Benchmarks ───────────────────────────────
        log.info("=" * 60)
        log.info("Suite 11/11: Medium Module Benchmarks")
        log.info("=" * 60)
        from src.medium_benchmark import run_medium_benchmark
        suites.append(_run_suite("medium_benchmarks", run_medium_benchmark, verbose=True))

    total_time = (time.monotonic() - t0) * 1000
    total_passed = sum(s["passed"] for s in suites)
    total_total = sum(s["total"] for s in suites)
    all_passed = all(s["all_passed"] for s in suites)

    return {
        "suites": suites,
        "total_passed": total_passed,
        "total_total": total_total,
        "all_passed": all_passed,
        "timing_ms": round(total_time, 1),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Makefile-compatible CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProtePilot — Unified Quality Gate")
    parser.add_argument("--fast", action="store_true",
                        help="Skip benchmarks (contracts + audit only)")
    parser.add_argument("--json", action="store_true",
                        help="JSON output")
    args = parser.parse_args()

    result = run_all_checks(fast=args.fast)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print()
        print("=" * 60)
        print("  ProtePilot — Quality Gate Summary")
        print("=" * 60)
        for s in result["suites"]:
            icon = "PASS" if s["all_passed"] else "FAIL"
            print(f"  [{icon}] {s['name']}: {s['passed']}/{s['total']} "
                  f"({s['timing_ms']:.0f}ms)")
            if s["errors"]:
                for e in s["errors"][:3]:
                    err_str = e if isinstance(e, str) else str(e)
                    print(f"         → {err_str[:80]}")
        print("-" * 60)
        print(f"  Total: {result['total_passed']}/{result['total_total']} checks passed "
              f"in {result['timing_ms']/1000:.1f}s")
        print(f"  Status: {'ALL PASSED' if result['all_passed'] else 'FAILURES DETECTED'}")
        print("=" * 60)

    sys.exit(0 if result["all_passed"] else 1)
