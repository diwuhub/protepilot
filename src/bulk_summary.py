"""
bulk_summary.py  ·  ProtePilot — Bulk Analysis Summary & Export
=================================================================
B9: Generate comparison CSV, aggregate statistics, and downloadable reports.

Design decision (B4): per-sample detail + batch summary.
- Per-sample: one row per molecule with key metrics
- Summary: aggregate statistics (mean, min, max, grade distribution)
- Export: CSV for downstream analysis, optional JSON for programmatic use
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from typing import Any, Dict, List, Optional

import os
import subprocess
import tempfile

from src.bulk_runner import BulkBatchResult, BulkRowResult

log = logging.getLogger("ProtePilot.BulkSummary")


# ═══════════════════════════════════════════════════════════════════════
#  CSV Export
# ═══════════════════════════════════════════════════════════════════════

_CSV_COLUMNS = [
    "row", "name", "status", "molecule_class",
    "seq_length", "mw_kda", "pI", "gravy", "hydrophobicity",
    "acidic_residues", "basic_residues",
    "deam_sites", "ox_sites", "cysteine_count",
    "dev_score", "dev_grade", "base_risk_score",
    "agg_risk", "stability", "viscosity_risk",
    "ood_flag",
    "intact_mass_da", "liability_density", "n_liabilities", "liability_summary",
    "cief_main_pct", "cief_acidic_pct", "cief_basic_pct",
    "ce_sds_purity_pct", "ce_sds_hmw_pct", "ce_sds_lmw_pct",
    "half_life_days", "clearance_ml_day_kg", "predicted_titer_g_L",
    "shelf_life_months", "stability_grade",
    "doe_optimal_ph", "doe_optimal_yield", "doe_optimal_purity", "doe_rs_min",
    "cogs_per_gram", "cogs_cost_rating",
    "ada_risk_score", "ada_risk_category", "n_mhcii_hotspots",
    "wall_time_s", "error",
]


def export_summary_csv(batch_result: BulkBatchResult) -> str:
    """
    Generate a summary CSV string from a BulkBatchResult.

    Returns
    -------
    str : CSV content ready for download.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row_dict in batch_result.to_summary_rows():
        writer.writerow(row_dict)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════
#  JSON Export
# ═══════════════════════════════════════════════════════════════════════

def export_summary_json(batch_result: BulkBatchResult) -> str:
    """
    Generate a detailed JSON report from a BulkBatchResult.

    Returns
    -------
    str : Pretty-printed JSON string.
    """
    report = {
        "report_type": "ProtePilot_Bulk_Analysis",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scoring_notes": {
            "dev_score": "5-dimension composite developability score (aggregation, stability, "
                         "viscosity, expression, immunogenicity). IDENTICAL to the score shown on the "
                         "single-molecule UI Developability page. Primary ranking metric.",
            "dev_grade": "Risk grade from dev_score: 'Low Risk' (<0.25), "
                         "'Medium Risk' (0.25-0.55), 'High Risk' (>=0.55). "
                         "Same thresholds and formula as the single-molecule grade.",
            "base_risk_score": "3-dimension predictor score (aggregation + stability + viscosity "
                               "only). Raw ML/heuristic output before expression and immunogenicity "
                               "adjustment. Retained for diagnostics — do NOT use for ranking.",
        },
        "batch_info": {
            "batch_type": batch_result.batch_type,
            "molecule_class": batch_result.molecule_class,
            "n_total": batch_result.n_total,
            "n_success": batch_result.n_success,
            "n_error": batch_result.n_error,
            "n_skipped": batch_result.n_skipped,
            "success_rate": round(batch_result.success_rate, 3),
            "wall_time_total_s": round(batch_result.wall_time_total, 2),
            "started_at": batch_result.started_at,
            "finished_at": batch_result.finished_at,
        },
        "statistics": batch_result.summary_stats(),
        "results": batch_result.to_summary_rows(),
    }
    return json.dumps(report, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
#  Statistics Summary (for UI display)
# ═══════════════════════════════════════════════════════════════════════

def generate_display_stats(batch_result: BulkBatchResult) -> Dict[str, Any]:
    """
    Generate display-ready statistics for the Streamlit UI.

    Returns
    -------
    dict with keys:
        - overview: {n_total, n_success, n_error, success_rate, wall_time}
        - score_stats: {mean, min, max, median}
        - grade_distribution: {Low: n, Medium: n, High: n}
        - risk_heatmap_data: list of {name, agg, stability, viscosity}
        - top_candidates: top 5 by score (ascending = lower risk = better)
        - flagged: OOD-flagged molecules
    """
    stats: Dict[str, Any] = {}

    # Overview
    stats["overview"] = {
        "n_total": batch_result.n_total,
        "n_success": batch_result.n_success,
        "n_error": batch_result.n_error,
        "n_skipped": batch_result.n_skipped,
        "success_rate": f"{batch_result.success_rate:.0%}",
        "wall_time": f"{batch_result.wall_time_total:.1f}s",
    }

    # Score statistics — uses 5-dim composite as PRIMARY, with 3-dim fallback
    success_results = [r for r in batch_result.results if r.status == "success"]

    def _primary_score(r):
        """Return 5-dim composite if available, else 3-dim predictor."""
        return r.composite_dev_score if r.composite_dev_score is not None else r.developability_score

    def _primary_grade(r):
        """Return composite grade if available, else predictor grade."""
        return r.composite_dev_grade or r.developability_grade

    def _bare_grade(g: str) -> str:
        """Normalize canonical 'Low Risk' → 'Low', 'Medium Risk' → 'Medium', etc.

        After the P1 grade-string fix, composite_dev_grade stores the canonical
        display form ('Low Risk'/'Medium Risk'/'High Risk').  grade_distribution
        uses bare keys ('Low', 'Medium', 'High') so downstream consumers
        (app.py metrics, tests) don't need updating.  This helper strips the
        ' Risk' suffix so both forms are handled transparently.
        """
        if g and g.endswith(" Risk"):
            return g[: -len(" Risk")]
        return g

    scores = [_primary_score(r) for r in success_results if _primary_score(r) is not None]

    if scores:
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        median = sorted_scores[n // 2] if n % 2 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
        stats["score_stats"] = {
            "mean": round(sum(scores) / len(scores), 3),
            "min": round(min(scores), 3),
            "max": round(max(scores), 3),
            "median": round(median, 3),
        }
    else:
        stats["score_stats"] = None

    # Grade distribution — uses composite grade (canonical or bare form both handled)
    grade_dist: Dict[str, int] = {"Low": 0, "Medium": 0, "High": 0}
    for r in success_results:
        g = _bare_grade(_primary_grade(r))  # normalizes "Low Risk" → "Low" etc.
        if g in grade_dist:
            grade_dist[g] += 1
        else:
            # Catch-all for unrecognized or None grades
            grade_dist.setdefault("Unknown", 0)
            grade_dist["Unknown"] += 1
    stats["grade_distribution"] = grade_dist

    # Risk heatmap data
    heatmap = []
    for r in success_results:
        heatmap.append({
            "name": r.name,
            "mw_kda": r.mw_kda,
            "pI": r.pI,
            "agg_risk": r.agg_risk,
            "stability": r.stability,
            "viscosity_risk": r.viscosity_risk,
            "dev_score": _primary_score(r),
            "grade": _primary_grade(r),
        })
    stats["risk_heatmap_data"] = heatmap

    # Top candidates (lowest dev score = best) — uses composite
    ranked = sorted(
        [r for r in success_results if _primary_score(r) is not None],
        key=lambda r: _primary_score(r),
    )
    stats["top_candidates"] = [
        {"name": r.name, "score": round(_primary_score(r), 3), "grade": _primary_grade(r)}
        for r in ranked[:5]
    ]

    # Flagged (OOD)
    stats["flagged"] = [
        {"name": r.name, "details": r.ood_details or "Out-of-distribution"}
        for r in batch_result.results if r.ood_flag
    ]

    return stats


# ═══════════════════════════════════════════════════════════════════════
#  Ranking Helper
# ═══════════════════════════════════════════════════════════════════════

def rank_candidates(
    batch_result: BulkBatchResult,
    sort_by: str = "dev_score",
    ascending: bool = True,
) -> List[Dict[str, Any]]:
    """
    Rank successful candidates by a given metric.

    Parameters
    ----------
    sort_by : str
        One of: "dev_score", "agg_risk", "stability", "viscosity_risk"
        "dev_score" uses the 5-dim composite (same as UI core page).
    ascending : bool
        True = lower is better (default for risk scores).

    Returns
    -------
    List of ranked dicts with rank, name, and scores.
    """

    def _primary_score(r):
        return r.composite_dev_score if r.composite_dev_score is not None else r.developability_score

    def _primary_grade(r):
        return r.composite_dev_grade or r.developability_grade

    if sort_by == "dev_score":
        # Use composite for ranking (PRIMARY score)
        valid = [
            r for r in batch_result.results
            if r.status == "success" and _primary_score(r) is not None
        ]
        valid.sort(key=lambda r: _primary_score(r), reverse=not ascending)
    else:
        _attr_map = {
            "agg_risk": "agg_risk",
            "stability": "stability",
            "viscosity_risk": "viscosity_risk",
        }
        attr = _attr_map.get(sort_by, "agg_risk")
        valid = [
            r for r in batch_result.results
            if r.status == "success" and getattr(r, attr) is not None
        ]
        valid.sort(key=lambda r: getattr(r, attr), reverse=not ascending)

    return [
        {
            "rank": i + 1,
            "name": r.name,
            "dev_score": round(_primary_score(r), 3) if _primary_score(r) is not None else None,
            "base_risk_score": round(r.developability_score, 3) if r.developability_score is not None else None,
            "agg_risk": round(r.agg_risk, 3) if r.agg_risk is not None else None,
            "stability": round(r.stability, 3) if r.stability is not None else None,
            "viscosity_risk": round(r.viscosity_risk, 3) if r.viscosity_risk is not None else None,
            "grade": _primary_grade(r),
            "ood": r.ood_flag,
        }
        for i, r in enumerate(valid)
    ]


# ═══════════════════════════════════════════════════════════════════════
#  DOCX Report Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_bulk_report_docx(batch_result: BulkBatchResult) -> Optional[bytes]:
    """
    Generate a DOCX report for bulk analysis results.

    Uses the docx-js Node script (src/bulk_report_docx.js) to create
    a professionally formatted Word document.

    Returns
    -------
    bytes or None : DOCX file content, or None if generation fails.
    """
    _script = os.path.join(os.path.dirname(__file__), "bulk_report_docx.js")
    if not os.path.exists(_script):
        log.error("DOCX generator script not found: %s", _script)
        return None

    json_str = export_summary_json(batch_result)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "bulk_data.json")
        docx_path = os.path.join(tmpdir, "bulk_report.docx")

        with open(json_path, "w") as f:
            f.write(json_str)

        try:
            proc = subprocess.run(
                ["node", _script, json_path, docx_path],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                log.error("DOCX generation failed: %s", proc.stderr)
                return None

            with open(docx_path, "rb") as f:
                return f.read()

        except Exception as e:
            log.error("DOCX generation exception: %s", e)
            return None


# ═══════════════════════════════════════════════════════════════════════
#  Save All Reports to Subfolder
# ═══════════════════════════════════════════════════════════════════════

def save_bulk_reports(
    batch_result: BulkBatchResult,
    output_dir: str,
) -> Dict[str, str]:
    """
    Save CSV, JSON, and DOCX reports to a bulk analysis subfolder.

    Parameters
    ----------
    batch_result : BulkBatchResult
    output_dir : str
        Base directory (e.g., workspace reports folder).
        A subfolder 'bulk_analysis/' is created inside.

    Returns
    -------
    dict : Paths of generated files keyed by type.
    """
    bulk_dir = os.path.join(output_dir, "bulk_analysis")
    os.makedirs(bulk_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"bulk_{batch_result.batch_type}_{timestamp}"
    generated: Dict[str, str] = {}

    # CSV
    csv_path = os.path.join(bulk_dir, f"{prefix}.csv")
    with open(csv_path, "w") as f:
        f.write(export_summary_csv(batch_result))
    generated["csv"] = csv_path
    log.info("Saved bulk CSV: %s", csv_path)

    # JSON
    json_path = os.path.join(bulk_dir, f"{prefix}.json")
    with open(json_path, "w") as f:
        f.write(export_summary_json(batch_result))
    generated["json"] = json_path
    log.info("Saved bulk JSON: %s", json_path)

    # DOCX
    docx_bytes = generate_bulk_report_docx(batch_result)
    if docx_bytes:
        docx_path = os.path.join(bulk_dir, f"{prefix}.docx")
        with open(docx_path, "wb") as f:
            f.write(docx_bytes)
        generated["docx"] = docx_path
        log.info("Saved bulk DOCX: %s", docx_path)
    else:
        log.warning("DOCX generation failed; skipping")

    return generated
