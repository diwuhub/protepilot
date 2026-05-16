"""
run_module_selftests.py — Aggregate runner for all src/ module _selftest() functions.
=====================================================================================
v32.1 S7: Consolidation of scattered _selftest() smoke tests.

Each src module retains its _selftest() definition for standalone use:
    python -m src.developability_core   # runs just that module's selftest

This script runs ALL module selftests in a single invocation:
    python SelfTest/run_module_selftests.py
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import time

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
)
log = logging.getLogger("ProtePilot.ModuleSelfTests")

# Modules that define _selftest()
# Full inventory — 17 modules discovered via: grep -rl "def _selftest" src/
SELFTEST_MODULES = [
    # ── Core modules (original 7) ──────────────────────────────
    "src.developability_core",
    "src.feature_registry",
    "src.molecule_classifier",
    "src.report_assembler",
    "src.report_export",
    "src.report_schema",
    "src.validation_planner",
    # ── Contract modules ───────────────────────────────────────
    "src.analytical_contract",
    "src.classification_contract",
    "src.developability_contract",
    # ── Infrastructure modules ─────────────────────────────────
    "src.molecule_registry",
    "src.ood_baseline",
    "src.bulk_schema",
    "src.bulk_single_schema_alignment",
    # ── Training sub-modules ───────────────────────────────────
    "src.training.features",
    "src.training.feedback_store",
    "src.training.schema",
]


def main() -> int:
    passed = 0
    failed = 0
    errors = 0
    t0 = time.time()

    for mod_name in SELFTEST_MODULES:
        short = mod_name.split(".")[-1]
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, "_selftest", None)
            if fn is None:
                log.warning("  %s: no _selftest() found — SKIP", short)
                continue
            result = fn()
            if result is False:
                log.error("  %s: FAIL", short)
                failed += 1
            else:
                log.info("  %s: PASS", short)
                passed += 1
        except Exception as exc:
            log.error("  %s: ERROR — %s", short, exc)
            errors += 1

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(
        "Module selftests: %d PASS | %d FAIL | %d ERROR (%.1fs)",
        passed, failed, errors, elapsed,
    )
    log.info("=" * 60)
    return 0 if (failed == 0 and errors == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
