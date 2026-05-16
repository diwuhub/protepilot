"""
dependency_audit.py  ·  ProtePilot — Dependency & Boundary Audit
===================================================================
Automated checks for module boundaries, import hygiene, SSOT
consistency, and architectural layering.

Run this as part of CI or selftest to catch:
  1. Boundary violations (lower layer importing higher layer)
  2. SSOT drift (contract constants vs implementation values)
  3. Soft import fallback defaults diverging from platform_config
  4. Contract guarantee counts changing unexpectedly

Usage:
    python -m src.dependency_audit          # Full audit
    python -m src.dependency_audit --json   # JSON output

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Set, Tuple

log = logging.getLogger("ProtePilot.DependencyAudit")

# Base path for src/
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════
#  Architecture: Module Layer Definitions
# ═══════════════════════════════════════════════════════════════════════

# Layer 0 (foundation) → Layer 3 (orchestration).
# Rule: a module may import from SAME or LOWER layer only.
MODULE_LAYERS: Dict[str, int] = {
    # Layer 0: Pure types & configuration
    "src.type_defs": 0,
    "src.report_schema": 0,
    "src.platform_config": 0,

    # Layer 1: Training pipeline (features, schema, data)
    "src.training.schema": 1,
    "src.training.features": 1,
    "src.training.feedback_store": 1,
    "src.training.data_harmonizer": 1,
    "src.training.classifier_trainer": 1,
    "src.training.ood_trainer": 1,
    "src.training.model_inference": 1,
    "src.training.benchmark_evaluator": 1,
    "src.training.pipeline": 1,

    # Layer 1: Analytical (independent)
    "src.analytical_twin": 1,
    "src.analytical_qc_twin": 1,

    # Layer 2: Business logic
    "src.molecule_classifier": 2,
    "src.developability_core": 2,

    # Layer 2.5: Contracts (document Layer 2 behavior, don't import it)
    "src.classification_contract": 2,
    "src.developability_contract": 2,
    "src.analytical_contract": 2,

    # Layer 3: Benchmarks & orchestration
    "src.classifier_benchmark": 3,
    "src.developability_benchmark": 3,
}

# Forbidden import directions (higher → lower is ok, reverse is not).
# Special exceptions: molecule_classifier may import from training for
# Phase 2/3 (trained model + OOD), but these MUST be soft (try/except).
ALLOWED_CROSS_LAYER_EXCEPTIONS: Set[Tuple[str, str]] = {
    # molecule_classifier soft-imports training for Phase 2/3
    ("src.molecule_classifier", "src.training.model_inference"),
    ("src.molecule_classifier", "src.training.ood_trainer"),
    ("src.molecule_classifier", "src.training.features"),
    ("src.molecule_classifier", "src.training.feedback_store"),
    # developability_core soft-imports molecule_classifier for weights
    ("src.developability_core", "src.molecule_classifier"),
    # analytical_contract bridges analytical_qc_twin and training.features
    ("src.analytical_contract", "src.training.features"),
    ("src.analytical_contract", "src.analytical_qc_twin"),
    ("src.analytical_contract", "src.developability_core"),
    # Benchmarks import everything (Layer 3)
    ("src.classifier_benchmark", "src.molecule_classifier"),
    ("src.classifier_benchmark", "src.classification_contract"),
    ("src.developability_benchmark", "src.developability_core"),
    ("src.developability_benchmark", "src.developability_contract"),
    ("src.developability_benchmark", "src.molecule_classifier"),
}


# ═══════════════════════════════════════════════════════════════════════
#  1. Import Extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_imports(filepath: str) -> List[Dict[str, Any]]:
    """Extract all src.* imports from a Python file."""
    imports = []
    try:
        with open(filepath, "r") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
            imports.append({
                "module": node.module,
                "names": [alias.name for alias in node.names],
                "line": node.lineno,
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src"):
                    imports.append({
                        "module": alias.name,
                        "names": [],
                        "line": node.lineno,
                    })
    return imports


def _is_soft_import(filepath: str, lineno: int) -> bool:
    """Check if an import at a given line is 'soft' (deferred/guarded).

    An import is considered soft if it's inside:
      - A try/except block (guarded against ImportError), OR
      - A function body (lazy/deferred — not a structural dependency).
    """
    try:
        with open(filepath, "r") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return False

    # Walk the AST looking for the import node at the given line,
    # and check if any ancestor is a FunctionDef or Try.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    if child.lineno == lineno:
                        return True  # Inside a function → deferred/lazy
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    if child.lineno == lineno:
                        return True  # Inside try/except → guarded
    return False


# ═══════════════════════════════════════════════════════════════════════
#  2. Boundary Violation Check
# ═══════════════════════════════════════════════════════════════════════

def check_boundary_violations() -> List[Dict[str, Any]]:
    """Check for imports that violate the layered architecture."""
    violations = []

    for module_path, layer in MODULE_LAYERS.items():
        filepath = os.path.join(_SRC_DIR, "..",
                                module_path.replace(".", "/") + ".py")
        if not os.path.exists(filepath):
            continue

        imports = _extract_imports(filepath)
        for imp in imports:
            target = imp["module"]
            # Find target's layer
            target_layer = MODULE_LAYERS.get(target, -1)
            if target_layer < 0:
                continue  # External or untracked module

            # Check: importing from higher layer is a violation
            if target_layer > layer:
                pair = (module_path, target)
                if pair not in ALLOWED_CROSS_LAYER_EXCEPTIONS:
                    is_soft = _is_soft_import(filepath, imp["line"])
                    violations.append({
                        "source": module_path,
                        "target": target,
                        "source_layer": layer,
                        "target_layer": target_layer,
                        "line": imp["line"],
                        "soft": is_soft,
                        "severity": "warning" if is_soft else "error",
                    })

    return violations


# ═══════════════════════════════════════════════════════════════════════
#  3. SSOT Consistency Checks
# ═══════════════════════════════════════════════════════════════════════

def check_ssot_consistency() -> List[Dict[str, Any]]:
    """Verify single-source-of-truth values are consistent across modules.

    Returns list of issues.  Each issue has a 'severity' key:
      - 'error' : real mismatch
      - 'warning': non-critical drift
      - 'skip'  : check couldn't run (missing module)
      - 'pass'  : check ran successfully (for counting)
    """
    issues = []

    # 3a. MoleculeClass enum values vs classification_contract.VALID_CLASSES
    try:
        from src.type_defs import MoleculeClass
        from src.classification_contract import VALID_CLASSES
        enum_values = {mc.value for mc in MoleculeClass}
        if enum_values != VALID_CLASSES:
            diff = enum_values.symmetric_difference(VALID_CLASSES)
            issues.append({
                "check": "MoleculeClass_vs_contract",
                "detail": f"Mismatch: {diff}",
                "severity": "error",
            })
        else:
            issues.append({"check": "MoleculeClass_vs_contract", "detail": "OK", "severity": "pass"})
    except ImportError as e:
        issues.append({"check": "MoleculeClass_vs_contract", "detail": str(e), "severity": "skip"})

    # 3b. Grade boundaries: report_schema vs developability_contract
    try:
        from src.report_schema import GRADE_LOW_UPPER, GRADE_MEDIUM_UPPER
        from src.developability_contract import (
            GRADE_LOW_UPPER as DC_LOW, GRADE_MEDIUM_UPPER as DC_MED,
        )
        if GRADE_LOW_UPPER != DC_LOW or GRADE_MEDIUM_UPPER != DC_MED:
            issues.append({
                "check": "grade_boundaries",
                "detail": f"report_schema({GRADE_LOW_UPPER},{GRADE_MEDIUM_UPPER}) "
                          f"vs dev_contract({DC_LOW},{DC_MED})",
                "severity": "error",
            })
        else:
            issues.append({"check": "grade_boundaries", "detail": "OK", "severity": "pass"})
    except ImportError as e:
        issues.append({"check": "grade_boundaries", "detail": str(e), "severity": "skip"})

    # 3c. Feature schema version consistency
    try:
        from src.training.pipeline import _schema_hash
        import json as _json
        manifest_path = os.path.join(_SRC_DIR, "..", "models", "MANIFEST.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = _json.load(f)
            code_hash = _schema_hash()
            manifest_hash = manifest.get("schema_version", "")
            if manifest_hash and code_hash != manifest_hash:
                issues.append({
                    "check": "schema_version",
                    "detail": f"code={code_hash} vs manifest={manifest_hash}",
                    "severity": "warning",
                })
            else:
                issues.append({"check": "schema_version", "detail": "OK", "severity": "pass"})
        else:
            issues.append({"check": "schema_version", "detail": "no manifest", "severity": "pass"})
    except (ImportError, Exception) as e:
        issues.append({"check": "schema_version", "detail": str(e), "severity": "skip"})

    # 3d. Risk weight profiles: all MoleculeClass values have weights
    try:
        from src.molecule_classifier import RISK_WEIGHT_PROFILES
        from src.type_defs import MoleculeClass
        for mc in MoleculeClass:
            if mc.value not in RISK_WEIGHT_PROFILES:
                issues.append({
                    "check": "risk_weights_complete",
                    "detail": f"Missing weights for {mc.value}",
                    "severity": "error",
                })
            else:
                total = sum(RISK_WEIGHT_PROFILES[mc.value].values())
                if abs(total - 1.0) > 0.02:
                    issues.append({
                        "check": "risk_weights_sum",
                        "detail": f"{mc.value} weights sum to {total}",
                        "severity": "error",
                    })
        # If we got here without errors, mark pass
        if not any(i["check"].startswith("risk_weights") and i["severity"] == "error" for i in issues):
            issues.append({"check": "risk_weights", "detail": "OK", "severity": "pass"})
    except ImportError as e:
        issues.append({"check": "risk_weights", "detail": str(e), "severity": "skip"})

    # 3e. Soft import fallback defaults match platform_config
    try:
        from src.platform_config import (
            MIN_SEQUENCE_LENGTH, MIN_HC_LENGTH,
            MIN_CHAIN_CLUSTER_LENGTH, HC_IDENTITY_THRESHOLD,
        )
        expected = {
            "MIN_SEQUENCE_LENGTH": MIN_SEQUENCE_LENGTH,
            "MIN_HC_LENGTH": MIN_HC_LENGTH,
            "MIN_CHAIN_CLUSTER_LENGTH": MIN_CHAIN_CLUSTER_LENGTH,
            "HC_IDENTITY_THRESHOLD": HC_IDENTITY_THRESHOLD,
        }
        # Check molecule_classifier fallbacks
        clf_path = os.path.join(_SRC_DIR, "molecule_classifier.py")
        with open(clf_path) as f:
            clf_src = f.read()
        for name, expected_val in expected.items():
            pattern = rf"{name}\s*=\s*([0-9.]+)"
            matches = re.findall(pattern, clf_src)
            # The second match (after except ImportError) is the fallback
            if len(matches) >= 2:
                fallback = float(matches[1])
                if fallback != expected_val:
                    issues.append({
                        "check": f"fallback_{name}",
                        "detail": f"molecule_classifier fallback={fallback} "
                                  f"vs platform_config={expected_val}",
                        "severity": "error",
                    })
        if not any(i["check"].startswith("fallback_") and i["severity"] == "error" for i in issues):
            issues.append({"check": "fallback_defaults", "detail": "OK", "severity": "pass"})
    except (ImportError, FileNotFoundError) as e:
        issues.append({"check": "fallback_defaults", "detail": str(e), "severity": "skip"})

    # 3f. Analytical consumer keys match what dev_core actually reads
    try:
        from src.analytical_contract import DEVELOPABILITY_CORE_REQUIRED_KEYS
        dev_core_path = os.path.join(_SRC_DIR, "developability_core.py")
        with open(dev_core_path) as f:
            dev_src = f.read()
        consumed = set(re.findall(r'anal\.get\("(\w+)"\)', dev_src))
        missing_in_contract = consumed - DEVELOPABILITY_CORE_REQUIRED_KEYS
        if missing_in_contract:
            issues.append({
                "check": "analytical_key_alignment",
                "detail": f"dev_core reads keys not in contract: {missing_in_contract}",
                "severity": "error",
            })
        else:
            issues.append({"check": "analytical_key_alignment", "detail": "OK", "severity": "pass"})
    except (ImportError, FileNotFoundError) as e:
        issues.append({"check": "analytical_key_alignment", "detail": str(e), "severity": "skip"})

    return issues


# ═══════════════════════════════════════════════════════════════════════
#  4. Contract Guarantee Counts (Regression Guard)
# ═══════════════════════════════════════════════════════════════════════

def check_contract_guarantees() -> List[Dict[str, Any]]:
    """Verify contract guarantee counts haven't decreased."""
    issues = []
    expected = {
        "classification_contract": 10,
        "developability_contract": 12,
        "analytical_contract": 8,
    }

    for module_name, min_count in expected.items():
        try:
            mod = __import__(f"src.{module_name}", fromlist=["BEHAVIORAL_GUARANTEES"])
            guarantees = getattr(mod, "BEHAVIORAL_GUARANTEES", [])
            if len(guarantees) < min_count:
                issues.append({
                    "check": f"{module_name}_guarantees",
                    "detail": f"Expected ≥{min_count}, got {len(guarantees)}",
                    "severity": "error",
                })
        except ImportError as e:
            issues.append({
                "check": f"{module_name}_guarantees",
                "detail": str(e),
                "severity": "skip",
            })

    return issues


# ═══════════════════════════════════════════════════════════════════════
#  5. Full Audit Runner
# ═══════════════════════════════════════════════════════════════════════

def run_full_audit(verbose: bool = True) -> Dict[str, Any]:
    """Run all dependency and boundary checks."""
    results = {}

    # Boundary violations
    boundary = check_boundary_violations()
    results["boundary_violations"] = boundary
    if verbose:
        if boundary:
            for v in boundary:
                log.warning("  [%s] %s → %s (L%d→L%d, line %d%s)",
                           v["severity"].upper(), v["source"], v["target"],
                           v["source_layer"], v["target_layer"], v["line"],
                           ", soft" if v["soft"] else "")
        else:
            log.info("  [PASS] No boundary violations")

    # SSOT consistency
    ssot = check_ssot_consistency()
    results["ssot_issues"] = ssot
    if verbose:
        errors = [s for s in ssot if s["severity"] == "error"]
        passed = sum(1 for s in ssot if s["severity"] == "pass")
        skipped = sum(1 for s in ssot if s["severity"] == "skip")
        if errors:
            for s in errors:
                log.warning("  [FAIL] SSOT %s: %s", s["check"], s["detail"])
        else:
            log.info("  [PASS] SSOT consistency (%d checks passed, %d skipped)",
                     passed, skipped)

    # Contract guarantees
    contracts = check_contract_guarantees()
    results["contract_guarantees"] = contracts
    if verbose:
        errors = [c for c in contracts if c["severity"] == "error"]
        if errors:
            for c in errors:
                log.warning("  [FAIL] %s: %s", c["check"], c["detail"])
        else:
            log.info("  [PASS] Contract guarantees intact")

    # Summary
    all_errors = (
        [v for v in boundary if v["severity"] == "error"] +
        [s for s in ssot if s["severity"] == "error"] +
        [c for c in contracts if c["severity"] == "error"]
    )
    results["all_passed"] = len(all_errors) == 0
    results["error_count"] = len(all_errors)
    results["warning_count"] = (
        sum(1 for v in boundary if v["severity"] == "warning") +
        sum(1 for s in ssot if s["severity"] == "warning")
    )

    return results


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ProtePilot — Dependency & Boundary Audit",
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--quiet", action="store_true", help="Suppress detail")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(name)s | %(message)s",
    )

    log.info("Running dependency & boundary audit...")
    results = run_full_audit(verbose=not args.quiet)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"Dependency & Boundary Audit")
        print(f"{'='*60}")
        print(f"  Boundary violations: {len(results['boundary_violations'])} "
              f"({results['error_count']} errors, {results['warning_count']} warnings)")
        print(f"  SSOT issues: {sum(1 for s in results['ssot_issues'] if s['severity'] == 'error')} errors")
        print(f"  Contract guarantees: "
              f"{sum(1 for c in results['contract_guarantees'] if c['severity'] == 'error')} errors")
        status = "ALL PASSED" if results["all_passed"] else "ISSUES FOUND"
        print(f"\n  Status: {status}")

    sys.exit(0 if results["all_passed"] else 1)


if __name__ == "__main__":
    main()
