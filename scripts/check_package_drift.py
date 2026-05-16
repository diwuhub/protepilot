#!/usr/bin/env python3
"""
check_package_drift.py — Detect drift between Level-2 vendored copies and monorepo originals.

Level-2 packages contain standalone copies of monorepo source files.  This script
diffs each vendored file against its monorepo original (ignoring expected import
rewrites) and reports any *substantive* drift — i.e. logic changes in the monorepo
that haven't been propagated to the standalone package, or vice-versa.

Usage:
    python scripts/check_package_drift.py          # summary only
    python scripts/check_package_drift.py -v        # show diffs
    python scripts/check_package_drift.py --strict  # exit 1 on ANY diff (CI mode)

Exit codes:
    0 — no substantive drift detected (or --strict not set)
    1 — substantive drift detected (--strict mode)
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────
#  Mapping: (package_file, monorepo_original)
#  Paths are relative to the repo root.
# ───────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent

VENDORED_MAP = [
    # pharma_classifier
    {
        "package": "pharma_classifier",
        "file": "features.py",
        "vendored": "packages/pharma_classifier/src/pharma_classifier/features.py",
        "original": "src/training/features.py",
    },
    {
        "package": "pharma_classifier",
        "file": "schema.py",
        "vendored": "packages/pharma_classifier/src/pharma_classifier/schema.py",
        "original": "src/training/schema.py",
    },
    {
        "package": "pharma_classifier",
        "file": "ood.py",
        "vendored": "packages/pharma_classifier/src/pharma_classifier/ood.py",
        "original": "src/training/ood_trainer.py",
    },
    {
        "package": "pharma_classifier",
        "file": "inference.py",
        "vendored": "packages/pharma_classifier/src/pharma_classifier/inference.py",
        "original": "src/training/model_inference.py",
    },
    # pharma_harmonizer
    {
        "package": "pharma_harmonizer",
        "file": "features.py",
        "vendored": "packages/pharma_harmonizer/src/pharma_harmonizer/features.py",
        "original": "src/training/features.py",
    },
    {
        "package": "pharma_harmonizer",
        "file": "schema.py",
        "vendored": "packages/pharma_harmonizer/src/pharma_harmonizer/schema.py",
        "original": "src/training/schema.py",
    },
    {
        "package": "pharma_harmonizer",
        "file": "ood.py",
        "vendored": "packages/pharma_harmonizer/src/pharma_harmonizer/ood.py",
        "original": "src/training/ood_trainer.py",
    },
    {
        "package": "pharma_harmonizer",
        "file": "inference.py",
        "vendored": "packages/pharma_harmonizer/src/pharma_harmonizer/inference.py",
        "original": "src/training/model_inference.py",
    },
    {
        "package": "pharma_harmonizer",
        "file": "harmonizer.py",
        "vendored": "packages/pharma_harmonizer/src/pharma_harmonizer/harmonizer.py",
        "original": "src/training/data_harmonizer.py",
    },
    {
        "package": "pharma_harmonizer",
        "file": "trainer.py",
        "vendored": "packages/pharma_harmonizer/src/pharma_harmonizer/trainer.py",
        "original": "src/training/classifier_trainer.py",
    },
    {
        "package": "pharma_harmonizer",
        "file": "benchmark.py",
        "vendored": "packages/pharma_harmonizer/src/pharma_harmonizer/benchmark.py",
        "original": "src/training/benchmark_evaluator.py",
    },
    {
        "package": "pharma_harmonizer",
        "file": "feedback_store.py",
        "vendored": "packages/pharma_harmonizer/src/pharma_harmonizer/feedback_store.py",
        "original": "src/training/feedback_store.py",
    },
]

# ───────────────────────────────────────────────────────────────────────
#  Import-rewrite normalisation
#  Level-2 packages replace `from src.xxx import` / `import src.xxx`
#  with package-local imports.  We normalise both sides so that
#  expected import rewrites don't show up as drift.
# ───────────────────────────────────────────────────────────────────────

IMPORT_PATTERNS = [
    # from src.training.xxx import ... → from pharma_xxx.yyy import ...
    (re.compile(r"from\s+src\.\S+\s+import"), "from <pkg> import"),
    # from src.xxx import ...
    (re.compile(r"from\s+src\.\S+\s+import"), "from <pkg> import"),
    # import src.xxx
    (re.compile(r"import\s+src\.\S+"), "import <pkg>"),
    # from pharma_classifier.xxx import ...
    (re.compile(r"from\s+pharma_\w+\.\S*\s+import"), "from <pkg> import"),
    # from pharma_harmonizer.xxx import ...
    (re.compile(r"from\s+pharma_\w+\.\S*\s+import"), "from <pkg> import"),
]


def normalise_imports(lines: list[str]) -> list[str]:
    """Replace all src.* and pharma_* import lines with a canonical placeholder."""
    out = []
    for line in lines:
        norm = line
        for pat, repl in IMPORT_PATTERNS:
            if pat.search(norm):
                # Keep the imported names, just normalise the source
                norm = pat.sub(repl, norm)
                break
        out.append(norm)
    return out


def read_lines(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)


def compute_drift(entry: dict, verbose: bool = False) -> dict:
    """Compare one vendored file against its monorepo original."""
    vendored_path = REPO_ROOT / entry["vendored"]
    original_path = REPO_ROOT / entry["original"]

    result = {
        "package": entry["package"],
        "file": entry["file"],
        "vendored": entry["vendored"],
        "original": entry["original"],
        "status": "ok",
        "diff_lines": 0,
        "diff": "",
    }

    v_lines = read_lines(vendored_path)
    o_lines = read_lines(original_path)

    if v_lines is None:
        result["status"] = "vendored_missing"
        return result
    if o_lines is None:
        result["status"] = "original_missing"
        return result

    # Normalise imports on both sides
    v_norm = normalise_imports(v_lines)
    o_norm = normalise_imports(o_lines)

    diff = list(difflib.unified_diff(
        o_norm, v_norm,
        fromfile=f"monorepo/{entry['original']}",
        tofile=f"package/{entry['vendored']}",
        lineterm="",
    ))

    # Filter out diff header lines to count only actual changes
    change_lines = [l for l in diff if l.startswith("+") or l.startswith("-")]
    change_lines = [l for l in change_lines if not l.startswith("+++") and not l.startswith("---")]

    if change_lines:
        result["status"] = "drifted"
        result["diff_lines"] = len(change_lines)
        if verbose:
            result["diff"] = "\n".join(diff)

    return result


def main():
    parser = argparse.ArgumentParser(description="Detect drift between Level-2 vendored copies and monorepo originals")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full diffs")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any drift (CI mode)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Level-2 Package Drift Detection")
    print("=" * 60)

    results = [compute_drift(entry, verbose=args.verbose) for entry in VENDORED_MAP]

    ok_count = sum(1 for r in results if r["status"] == "ok")
    drift_count = sum(1 for r in results if r["status"] == "drifted")
    missing_count = sum(1 for r in results if "missing" in r["status"])

    for r in results:
        icon = {"ok": "✅", "drifted": "⚠️", "vendored_missing": "❌", "original_missing": "❌"}.get(r["status"], "?")
        line = f"  {icon} {r['package']}/{r['file']}"
        if r["status"] == "drifted":
            line += f"  ({r['diff_lines']} changed lines)"
        elif "missing" in r["status"]:
            line += f"  ({r['status']})"
        print(line)

        if args.verbose and r["diff"]:
            print(r["diff"])
            print()

    print()
    print(f"Summary: {ok_count} ok, {drift_count} drifted, {missing_count} missing")
    print(f"Total vendored files tracked: {len(VENDORED_MAP)}")

    if drift_count > 0:
        print("\n⚠️  Drift detected. Review changes and sync packages if needed.")
        print("   Run with -v to see full diffs.")
    elif missing_count > 0:
        print("\n❌ Missing files detected. Check vendored mapping.")
    else:
        print("\n✅ All vendored copies are in sync with monorepo originals (modulo import rewrites).")

    if args.strict and (drift_count > 0 or missing_count > 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
