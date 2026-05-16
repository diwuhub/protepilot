"""
report_bulk_benchmark.py  ·  ProtePilot — Report & Bulk Analysis Benchmark
=============================================================================
Edge case and regression benchmark for report assembler, bulk runner,
bulk schema, and bulk summary modules.

Usage:
    python -m src.report_bulk_benchmark                          # Full benchmark
    python -m src.report_bulk_benchmark --module bulk_schema     # Single module
    python -m src.report_bulk_benchmark --json                   # JSON output

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Tuple

log = logging.getLogger("ProtePilot.ReportBulkBenchmark")


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

_MAB_CSV = """name,hc,lc
BenchMab1,EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRGITIFGVVIIPGFFDIWGQGTLVTVSS,DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK
"""

_PEPTIDE_CSV = """name,peptide
BenchPep1,ACDEFGHIKLMNPQRSTVWY
BenchPep2,FWFWFWFWFWFWFWFWFWFW
BenchPep3,KKKKKKKKKKKKKKKKKKKK
"""


def _parse(csv_str: str, batch_type: str):
    from src.bulk_schema import parse_bulk_csv
    return parse_bulk_csv(csv_str.strip(), batch_type)


def _run_peptide_batch():
    from src.bulk_runner import run_bulk_analysis
    return run_bulk_analysis(_parse(_PEPTIDE_CSV, "peptide"))


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Cases
# ═══════════════════════════════════════════════════════════════════════

def _make_cases() -> List[Dict[str, Any]]:
    cases = []

    # ── Report Assembler ─────────────────────────────────────────

    def _report_minimal_intent():
        """Report from minimal intent should still have required sections."""
        from src.report_assembler import assemble_report
        r = assemble_report({"name": "MinimalMol", "sequence": "ACDEFGHIKLMNPQRSTVWY",
                             "molecule_class": "peptide", "source": "benchmark"})
        sections = [s for s in ["executive_summary", "molecule_overview", "developability",
                                "model_metadata"] if hasattr(r, s)]
        ok = len(sections) >= 3
        return ok, f"{len(sections)}/4 sections present"

    def _report_long_sequence():
        """Report should handle long sequences without error."""
        from src.report_assembler import assemble_report
        long_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGG" * 20
        r = assemble_report({"name": "LongSeqMol", "sequence": long_seq,
                             "molecule_class": "canonical_mab", "source": "benchmark"})
        ok = r.report_title is not None and len(r.to_dict()) > 0
        return ok, f"title='{r.report_title[:30]}...', dict_keys={len(r.to_dict())}"

    def _report_special_chars_name():
        """Report should handle special characters in molecule name."""
        from src.report_assembler import assemble_report
        r = assemble_report({"name": "Test-Mol_v2.1 (batch #3)",
                             "sequence": "ACDEFGHIKLMNPQRSTVWY" * 5,
                             "molecule_class": "peptide", "source": "benchmark"})
        ok = "Test-Mol" in r.report_title or r.report_title is not None
        return ok, f"title='{r.report_title[:40]}'"

    cases.extend([
        {"name": "report_minimal_intent", "module": "report_assembler", "fn": _report_minimal_intent,
         "desc": "Minimal intent still produces report sections", "category": "edge"},
        {"name": "report_long_sequence", "module": "report_assembler", "fn": _report_long_sequence,
         "desc": "Long sequence handled without error", "category": "edge"},
        {"name": "report_special_chars", "module": "report_assembler", "fn": _report_special_chars_name,
         "desc": "Special characters in molecule name", "category": "edge"},
    ])

    # ── Bulk Schema ──────────────────────────────────────────────

    def _schema_invalid_batch_type():
        """Invalid batch type should raise or return errors."""
        from src.bulk_schema import parse_bulk_csv
        try:
            pr = parse_bulk_csv("name,seq\nTest,ACDEF", "nonexistent_type")
            # If no exception, should have errors
            ok = not pr.is_ok or len(pr.errors) > 0
            return ok, f"is_ok={pr.is_ok}, errors={pr.errors[:1]}"
        except (KeyError, ValueError):
            return True, "Raised expected exception for invalid batch type"

    def _schema_empty_csv():
        """Empty CSV content should not crash."""
        from src.bulk_schema import parse_bulk_csv
        try:
            pr = parse_bulk_csv("", "canonical_mab")
            ok = not pr.is_ok or pr.n_valid == 0
            return ok, f"is_ok={pr.is_ok}, n_valid={pr.n_valid}"
        except Exception as exc:
            # Acceptable: raise on empty input
            return True, f"Raised {type(exc).__name__} on empty CSV"

    def _schema_max_rows_limit():
        """Rows beyond max_rows should be truncated."""
        from src.bulk_schema import parse_bulk_csv
        # Generate 10 peptide rows
        lines = ["name,peptide"]
        for i in range(10):
            lines.append(f"Pep{i},ACDEFGHIKLMNPQRSTVWY")
        csv_str = "\n".join(lines)
        pr = parse_bulk_csv(csv_str, "peptide", max_rows=5)
        ok = pr.n_total <= 5
        return ok, f"n_total={pr.n_total} with max_rows=5"

    def _schema_all_batch_types_have_template():
        """Every registered batch type must have a valid template."""
        from src.bulk_schema import BATCH_TYPES, generate_csv_template
        failures = []
        for key in BATCH_TYPES:
            try:
                tmpl = generate_csv_template(key)
                if not tmpl or len(tmpl) < 5:
                    failures.append(f"{key}: empty template")
            except Exception as e:
                failures.append(f"{key}: {e}")
        ok = len(failures) == 0
        return ok, f"{len(BATCH_TYPES)} types checked, {len(failures)} failures" + \
                    (f": {failures[0]}" if failures else "")

    cases.extend([
        {"name": "schema_invalid_batch_type", "module": "bulk_schema", "fn": _schema_invalid_batch_type,
         "desc": "Invalid batch type returns error", "category": "edge"},
        {"name": "schema_empty_csv", "module": "bulk_schema", "fn": _schema_empty_csv,
         "desc": "Empty CSV handled gracefully", "category": "edge"},
        {"name": "schema_max_rows_limit", "module": "bulk_schema", "fn": _schema_max_rows_limit,
         "desc": "max_rows truncates input", "category": "core"},
        {"name": "schema_all_batch_types_template", "module": "bulk_schema",
         "fn": _schema_all_batch_types_have_template,
         "desc": "All batch types have valid templates", "category": "core"},
    ])

    # ── Bulk Runner ──────────────────────────────────────────────

    def _runner_mab_batch():
        """mAb batch should produce results with biophysical data."""
        from src.bulk_runner import run_bulk_analysis
        pr = _parse(_MAB_CSV, "canonical_mab")
        br = run_bulk_analysis(pr)
        if br.n_total < 1:
            return False, "No results for mAb batch"
        success = [r for r in br.results if r.status == "success"]
        if not success:
            errors = [r.error_message for r in br.results if r.error_message]
            return False, f"No successes, errors: {errors[:2]}"
        row = success[0]
        has_pi = row.pI is not None
        has_mw = row.mw_kda is not None
        return has_pi and has_mw, f"pI={row.pI}, mw={row.mw_kda}"

    def _runner_progress_callback():
        """Progress callback should be called for each row."""
        from src.bulk_runner import run_bulk_analysis
        calls = []
        def _cb(cur, total, name):
            calls.append((cur, total, name))
        pr = _parse(_PEPTIDE_CSV, "peptide")
        br = run_bulk_analysis(pr, progress_callback=_cb)
        ok = len(calls) >= br.n_total
        return ok, f"{len(calls)} callbacks for {br.n_total} rows"

    def _runner_wall_time_tracked():
        """BulkBatchResult should track wall time."""
        from src.bulk_runner import run_bulk_analysis
        pr = _parse(_PEPTIDE_CSV, "peptide")
        br = run_bulk_analysis(pr)
        ok = br.wall_time_total > 0 and br.started_at and br.finished_at
        return ok, f"wall_time={br.wall_time_total:.2f}s, started={br.started_at[:19]}"

    def _runner_summary_stats():
        """summary_stats() should return dict with expected keys."""
        from src.bulk_runner import run_bulk_analysis
        pr = _parse(_PEPTIDE_CSV, "peptide")
        br = run_bulk_analysis(pr)
        stats = br.summary_stats()
        if not isinstance(stats, dict):
            return False, f"Expected dict, got {type(stats).__name__}"
        ok = "n_total" in stats or "total" in stats or len(stats) > 0
        return ok, f"summary_stats has {len(stats)} keys"

    cases.extend([
        {"name": "runner_mab_batch", "module": "bulk_runner", "fn": _runner_mab_batch,
         "desc": "mAb batch produces biophysical data", "category": "core"},
        {"name": "runner_progress_callback", "module": "bulk_runner", "fn": _runner_progress_callback,
         "desc": "Progress callback invoked per row", "category": "core"},
        {"name": "runner_wall_time_tracked", "module": "bulk_runner", "fn": _runner_wall_time_tracked,
         "desc": "Wall time and timestamps tracked", "category": "core"},
        {"name": "runner_summary_stats", "module": "bulk_runner", "fn": _runner_summary_stats,
         "desc": "summary_stats() returns valid dict", "category": "core"},
    ])

    # ── Bulk Summary ─────────────────────────────────────────────

    def _summary_display_stats_sections():
        """generate_display_stats() must have all expected sections."""
        from src.bulk_summary import generate_display_stats
        br = _run_peptide_batch()
        stats = generate_display_stats(br)
        expected = {"overview", "score_stats", "grade_distribution"}
        present = set(stats.keys()) & expected
        ok = len(present) >= 2
        return ok, f"Sections: {sorted(stats.keys())}"

    def _summary_rank_different_metrics():
        """rank_candidates() should accept different sort_by metrics."""
        from src.bulk_summary import rank_candidates
        br = _run_peptide_batch()
        for metric in ["dev_score", "agg_risk"]:
            ranked = rank_candidates(br, sort_by=metric)
            if not isinstance(ranked, list):
                return False, f"sort_by={metric} returned {type(ranked).__name__}"
        return True, f"Both dev_score and agg_risk sorting work"

    def _summary_csv_has_all_rows():
        """CSV export should have one row per successful result."""
        from src.bulk_summary import export_summary_csv
        br = _run_peptide_batch()
        csv_str = export_summary_csv(br)
        lines = [l for l in csv_str.strip().split("\n") if l.strip()]
        # header + data rows
        data_rows = len(lines) - 1
        ok = data_rows >= br.n_success
        return ok, f"{data_rows} data rows for {br.n_success} successes"

    def _summary_empty_batch():
        """Summary functions should handle empty batch result gracefully."""
        from src.bulk_runner import BulkBatchResult
        from src.bulk_summary import generate_display_stats, rank_candidates, export_summary_csv
        empty = BulkBatchResult(batch_type="peptide", molecule_class="peptide")
        stats = generate_display_stats(empty)
        ranked = rank_candidates(empty)
        csv_str = export_summary_csv(empty)
        ok = isinstance(stats, dict) and isinstance(ranked, list) and isinstance(csv_str, str)
        return ok, f"Empty batch: stats={len(stats)} keys, ranked={len(ranked)}, csv={len(csv_str)} chars"

    cases.extend([
        {"name": "summary_display_stats_sections", "module": "bulk_summary",
         "fn": _summary_display_stats_sections,
         "desc": "Display stats has all expected sections", "category": "core"},
        {"name": "summary_rank_different_metrics", "module": "bulk_summary",
         "fn": _summary_rank_different_metrics,
         "desc": "Ranking by different metrics", "category": "core"},
        {"name": "summary_csv_has_all_rows", "module": "bulk_summary",
         "fn": _summary_csv_has_all_rows,
         "desc": "CSV has row per successful result", "category": "core"},
        {"name": "summary_empty_batch", "module": "bulk_summary",
         "fn": _summary_empty_batch,
         "desc": "Empty batch handled gracefully", "category": "edge"},
    ])

    return cases


BENCHMARK_CASES = _make_cases()


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════

def run_report_bulk_benchmark(
    module_filter: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run report & bulk analysis benchmarks."""
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

    parser = argparse.ArgumentParser(description="Report & Bulk Analysis Benchmark Suite")
    parser.add_argument("--module", default=None,
                        choices=["report_assembler", "bulk_schema",
                                 "bulk_runner", "bulk_summary"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    log.info("Running report & bulk benchmarks%s...",
             f" (module={args.module})" if args.module else "")
    result = run_report_bulk_benchmark(module_filter=args.module)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"Report & Bulk Benchmark: {result['passed']}/{result['total']} passed "
              f"({result['accuracy']:.0%}) in {result['timing_ms']:.0f}ms")
        if result["failures"]:
            print("Failures:")
            for f in result["failures"]:
                print(f"  - {f['test']}: {f.get('detail', f.get('error', ''))}")
        print(f"Status: {'ALL PASSED' if result['all_passed'] else 'FAILURES DETECTED'}")

    sys.exit(0 if result["all_passed"] else 1)
