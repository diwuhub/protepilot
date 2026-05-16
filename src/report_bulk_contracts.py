"""
report_bulk_contracts.py  ·  ProtePilot — Report & Bulk Analysis Contracts
=============================================================================
Formal specification of behavioral guarantees for report assembler,
bulk runner, bulk schema, and bulk summary modules.

Covered modules:
    report_assembler  — Full report generation from intent
    bulk_runner       — Batch processing pipeline
    bulk_schema       — CSV parsing and validation
    bulk_summary      — Export, ranking, and display stats

Usage:
    python -m src.report_bulk_contracts                        # Run all
    python -m src.report_bulk_contracts --module bulk_schema   # Single module

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Tuple

log = logging.getLogger("ProtePilot.ReportBulkContracts")


# ═══════════════════════════════════════════════════════════════════════
#  Helpers — minimal test fixtures
# ═══════════════════════════════════════════════════════════════════════

_MAB_CSV = """name,hc,lc
TestMab1,EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRGITIFGVVIIPGFFDIWGQGTLVTVSS,DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK
TestMab2,EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMSWIRQAPGKGLEWVSYISSSGSTIDYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDRGIATFAYWGQGTLVTVSS,DIQMTQSPSSLSASVGDRVTITCRASQGISSWLAWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQQANSFPLTFGGGTKVEIK
"""

_PEPTIDE_CSV = """name,peptide
TestPep1,ACDEFGHIKLMNPQRSTVWY
TestPep2,FWFWFWFWFWFWFWFWFWFW
"""


def _make_mab_parse_result():
    """Parse a minimal mAb CSV for testing."""
    from src.bulk_schema import parse_bulk_csv
    return parse_bulk_csv(_MAB_CSV.strip(), "canonical_mab")


def _make_peptide_parse_result():
    """Parse a minimal peptide CSV for testing."""
    from src.bulk_schema import parse_bulk_csv
    return parse_bulk_csv(_PEPTIDE_CSV.strip(), "peptide")


def _make_mab_intent() -> Dict[str, Any]:
    """Create a minimal report intent dict."""
    return {
        "name": "ContractTestMab",
        "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRGITIFGVVIIPGFFDIWGQGTLVTVSS",
        "molecule_class": "canonical_mab",
        "pH_working": 6.0,
        "pI": 7.5,
        "source": "contract_test",
    }


# ═══════════════════════════════════════════════════════════════════════
#  Contract Definitions
# ═══════════════════════════════════════════════════════════════════════

# ── Report Assembler ─────────────────────────────────────────────────

def _check_report_returns_report_object() -> Tuple[bool, str]:
    """assemble_report() must return a ReportObject with required sections."""
    from src.report_assembler import assemble_report, ReportObject
    intent = _make_mab_intent()
    r = assemble_report(intent)
    if not isinstance(r, ReportObject):
        return False, f"Expected ReportObject, got {type(r).__name__}"
    required = ["report_title", "generated_at", "executive_summary",
                 "molecule_overview", "developability"]
    missing = [a for a in required if not hasattr(r, a)]
    if missing:
        return False, f"Missing attrs: {missing}"
    return True, f"ReportObject with title='{r.report_title[:40]}...'"


def _check_report_to_dict() -> Tuple[bool, str]:
    """ReportObject.to_dict() must return a non-empty dict."""
    from src.report_assembler import assemble_report
    intent = _make_mab_intent()
    r = assemble_report(intent)
    d = r.to_dict()
    if not isinstance(d, dict):
        return False, f"to_dict() returned {type(d).__name__}"
    if len(d) < 3:
        return False, f"to_dict() has only {len(d)} keys"
    return True, f"to_dict() has {len(d)} top-level keys"


def _check_report_to_json() -> Tuple[bool, str]:
    """ReportObject.to_json() must produce valid JSON string."""
    import json as _json
    from src.report_assembler import assemble_report
    intent = _make_mab_intent()
    r = assemble_report(intent)
    j = r.to_json()
    if not isinstance(j, str):
        return False, f"to_json() returned {type(j).__name__}"
    try:
        parsed = _json.loads(j)
    except _json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    if not isinstance(parsed, dict):
        return False, f"JSON root is {type(parsed).__name__}, expected dict"
    return True, f"Valid JSON, {len(j)} chars"


# ── Bulk Schema ──────────────────────────────────────────────────────

def _check_bulk_schema_parse_mab() -> Tuple[bool, str]:
    """parse_bulk_csv() must parse canonical_mab CSV correctly."""
    pr = _make_mab_parse_result()
    if not hasattr(pr, "rows") or not hasattr(pr, "valid_rows"):
        return False, f"BulkParseResult missing rows/valid_rows"
    if pr.n_valid < 1:
        return False, f"No valid rows parsed (errors: {pr.errors})"
    if pr.n_total != 2:
        return False, f"Expected 2 rows, got {pr.n_total}"
    return True, f"{pr.n_valid}/{pr.n_total} valid, batch_type={pr.batch_type.key}"


def _check_bulk_schema_parse_peptide() -> Tuple[bool, str]:
    """parse_bulk_csv() must handle peptide batch type."""
    pr = _make_peptide_parse_result()
    if pr.n_valid < 1:
        return False, f"No valid peptide rows (errors: {pr.errors})"
    # Check that rows have sequences
    for row in pr.valid_rows:
        if not row.combined_sequence:
            return False, f"Row '{row.name}' missing combined_sequence"
    return True, f"{pr.n_valid} peptide rows parsed"


def _check_bulk_schema_template() -> Tuple[bool, str]:
    """generate_csv_template() must return non-empty CSV string."""
    from src.bulk_schema import generate_csv_template
    for btype in ["canonical_mab", "peptide", "scfv"]:
        tmpl = generate_csv_template(btype)
        if not isinstance(tmpl, str) or len(tmpl) < 10:
            return False, f"Template for {btype} too short: {len(tmpl) if tmpl else 0}"
        if "name" not in tmpl.lower():
            return False, f"Template for {btype} missing 'name' column"
    return True, "Templates valid for canonical_mab, peptide, scfv"


def _check_bulk_schema_batch_types_registry() -> Tuple[bool, str]:
    """BATCH_TYPES must contain expected keys."""
    from src.bulk_schema import BATCH_TYPES
    required = {"canonical_mab", "peptide"}
    missing = required - set(BATCH_TYPES.keys())
    if missing:
        return False, f"Missing batch types: {missing}"
    if len(BATCH_TYPES) < 5:
        return False, f"Only {len(BATCH_TYPES)} batch types, expected >= 5"
    return True, f"{len(BATCH_TYPES)} batch types registered"


def _check_bulk_schema_row_to_intent() -> Tuple[bool, str]:
    """row_to_intent() must produce dict with required keys."""
    from src.bulk_schema import row_to_intent
    pr = _make_mab_parse_result()
    if not pr.valid_rows:
        return False, "No valid rows to convert"
    intent = row_to_intent(pr.valid_rows[0], pr.batch_type)
    if not isinstance(intent, dict):
        return False, f"Expected dict, got {type(intent).__name__}"
    required_keys = {"name", "molecule_class", "sequence"}
    missing = required_keys - set(intent.keys())
    if missing:
        return False, f"Intent missing keys: {missing}"
    return True, f"Intent has {len(intent)} keys for '{intent.get('name', '?')}'"


# ── Bulk Runner ──────────────────────────────────────────────────────

def _check_bulk_runner_returns_batch_result() -> Tuple[bool, str]:
    """run_bulk_analysis() must return BulkBatchResult."""
    from src.bulk_runner import run_bulk_analysis, BulkBatchResult
    pr = _make_peptide_parse_result()
    br = run_bulk_analysis(pr)
    if not isinstance(br, BulkBatchResult):
        return False, f"Expected BulkBatchResult, got {type(br).__name__}"
    if br.n_total < 1:
        return False, f"No results (n_total={br.n_total})"
    return True, f"BulkBatchResult: {br.n_success}/{br.n_total} success"


def _check_bulk_runner_row_result_fields() -> Tuple[bool, str]:
    """BulkRowResult must have name, status, and key fields."""
    from src.bulk_runner import run_bulk_analysis
    pr = _make_peptide_parse_result()
    br = run_bulk_analysis(pr)
    if br.n_total < 1:
        return False, f"No rows to check (n_total=0)"
    for row in br.results:
        if not hasattr(row, "name") or not hasattr(row, "status"):
            return False, f"Row missing name/status"
        if row.status == "success":
            if row.pI is None and row.mw_kda is None:
                return False, f"Success row '{row.name}' missing biophysical data"
    return True, f"All {br.n_total} rows have required fields"


def _check_bulk_runner_success_rate() -> Tuple[bool, str]:
    """Success rate must be between 0 and 1."""
    from src.bulk_runner import run_bulk_analysis
    pr = _make_peptide_parse_result()
    br = run_bulk_analysis(pr)
    if not (0.0 <= br.success_rate <= 1.0):
        return False, f"success_rate={br.success_rate} outside [0, 1]"
    return True, f"success_rate={br.success_rate:.2f}"


# ── Bulk Summary ─────────────────────────────────────────────────────

def _check_summary_csv_export() -> Tuple[bool, str]:
    """export_summary_csv() must return non-empty CSV string."""
    from src.bulk_runner import run_bulk_analysis
    from src.bulk_summary import export_summary_csv
    pr = _make_peptide_parse_result()
    br = run_bulk_analysis(pr)
    csv_str = export_summary_csv(br)
    if not isinstance(csv_str, str):
        return False, f"Expected str, got {type(csv_str).__name__}"
    lines = csv_str.strip().split("\n")
    if len(lines) < 2:
        return False, f"CSV has only {len(lines)} lines (need header + data)"
    return True, f"CSV: {len(lines)} lines, header has {len(lines[0].split(','))} cols"


def _check_summary_display_stats() -> Tuple[bool, str]:
    """generate_display_stats() must return dict with overview section."""
    from src.bulk_runner import run_bulk_analysis
    from src.bulk_summary import generate_display_stats
    pr = _make_peptide_parse_result()
    br = run_bulk_analysis(pr)
    stats = generate_display_stats(br)
    if not isinstance(stats, dict):
        return False, f"Expected dict, got {type(stats).__name__}"
    if "overview" not in stats:
        return False, f"Missing 'overview' key, got: {list(stats.keys())[:5]}"
    ov = stats["overview"]
    if "n_total" not in ov:
        return False, f"overview missing n_total"
    return True, f"Stats: overview.n_total={ov.get('n_total')}, {len(stats)} sections"


def _check_summary_rank_candidates() -> Tuple[bool, str]:
    """rank_candidates() must return list sorted by score."""
    from src.bulk_runner import run_bulk_analysis
    from src.bulk_summary import rank_candidates
    pr = _make_peptide_parse_result()
    br = run_bulk_analysis(pr)
    ranked = rank_candidates(br)
    if not isinstance(ranked, list):
        return False, f"Expected list, got {type(ranked).__name__}"
    # Check sorted (ascending by default)
    if len(ranked) >= 2:
        scores = [r.get("dev_score", 0) or 0 for r in ranked]
        for i in range(1, len(scores)):
            if scores[i] < scores[i-1] - 0.001:
                return False, f"Not sorted ascending: {scores}"
    return True, f"{len(ranked)} candidates ranked"


def _check_summary_json_export() -> Tuple[bool, str]:
    """export_summary_json() must return valid JSON string."""
    import json as _json
    from src.bulk_runner import run_bulk_analysis
    from src.bulk_summary import export_summary_json
    pr = _make_peptide_parse_result()
    br = run_bulk_analysis(pr)
    j = export_summary_json(br)
    if not isinstance(j, str):
        return False, f"Expected str, got {type(j).__name__}"
    try:
        parsed = _json.loads(j)
    except _json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    return True, f"Valid JSON, {len(j)} chars"


# ═══════════════════════════════════════════════════════════════════════
#  Contract Registry
# ═══════════════════════════════════════════════════════════════════════

CONTRACTS: List[Dict[str, Any]] = [
    # Report Assembler
    {"name": "report_returns_report_object", "module": "report_assembler",
     "fn": _check_report_returns_report_object,
     "desc": "assemble_report() returns ReportObject with required sections"},
    {"name": "report_to_dict", "module": "report_assembler",
     "fn": _check_report_to_dict,
     "desc": "ReportObject.to_dict() returns non-empty dict"},
    {"name": "report_to_json", "module": "report_assembler",
     "fn": _check_report_to_json,
     "desc": "ReportObject.to_json() produces valid JSON"},

    # Bulk Schema
    {"name": "bulk_schema_parse_mab", "module": "bulk_schema",
     "fn": _check_bulk_schema_parse_mab,
     "desc": "parse_bulk_csv() handles canonical_mab CSV"},
    {"name": "bulk_schema_parse_peptide", "module": "bulk_schema",
     "fn": _check_bulk_schema_parse_peptide,
     "desc": "parse_bulk_csv() handles peptide CSV"},
    {"name": "bulk_schema_template", "module": "bulk_schema",
     "fn": _check_bulk_schema_template,
     "desc": "generate_csv_template() returns valid templates"},
    {"name": "bulk_schema_batch_types", "module": "bulk_schema",
     "fn": _check_bulk_schema_batch_types_registry,
     "desc": "BATCH_TYPES registry has required entries"},
    {"name": "bulk_schema_row_to_intent", "module": "bulk_schema",
     "fn": _check_bulk_schema_row_to_intent,
     "desc": "row_to_intent() produces valid pipeline intent"},

    # Bulk Runner
    {"name": "bulk_runner_returns_batch_result", "module": "bulk_runner",
     "fn": _check_bulk_runner_returns_batch_result,
     "desc": "run_bulk_analysis() returns BulkBatchResult"},
    {"name": "bulk_runner_row_result_fields", "module": "bulk_runner",
     "fn": _check_bulk_runner_row_result_fields,
     "desc": "BulkRowResult has name, status, biophysical data"},
    {"name": "bulk_runner_success_rate", "module": "bulk_runner",
     "fn": _check_bulk_runner_success_rate,
     "desc": "success_rate in [0, 1]"},

    # Bulk Summary
    {"name": "bulk_summary_csv_export", "module": "bulk_summary",
     "fn": _check_summary_csv_export,
     "desc": "export_summary_csv() returns valid CSV"},
    {"name": "bulk_summary_display_stats", "module": "bulk_summary",
     "fn": _check_summary_display_stats,
     "desc": "generate_display_stats() returns dict with overview"},
    {"name": "bulk_summary_rank_candidates", "module": "bulk_summary",
     "fn": _check_summary_rank_candidates,
     "desc": "rank_candidates() returns sorted list"},
    {"name": "bulk_summary_json_export", "module": "bulk_summary",
     "fn": _check_summary_json_export,
     "desc": "export_summary_json() returns valid JSON"},
]


# ═══════════════════════════════════════════════════════════════════════
#  Contract Runner
# ═══════════════════════════════════════════════════════════════════════

def run_report_bulk_contracts(
    module_filter: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all report & bulk analysis contracts.

    Parameters
    ----------
    module_filter : str, optional
        Only run contracts for this module.
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

    parser = argparse.ArgumentParser(description="Report & Bulk Analysis Contract Checks")
    parser.add_argument("--module", default=None,
                        choices=["report_assembler", "bulk_schema",
                                 "bulk_runner", "bulk_summary"],
                        help="Check contracts for a single module")
    args = parser.parse_args()

    log.info("Running report & bulk contracts%s...",
             f" (module={args.module})" if args.module else "")
    result = run_report_bulk_contracts(module_filter=args.module)

    print(f"\n{'='*50}")
    print(f"Report & Bulk Contracts: {result['passed']}/{result['total']} passed")
    if result["errors"]:
        print("Failures:")
        for e in result["errors"]:
            print(f"  - {e}")
    print(f"Status: {'ALL PASSED' if result['all_passed'] else 'FAILURES DETECTED'}")

    sys.exit(0 if result["all_passed"] else 1)
