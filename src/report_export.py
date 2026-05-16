"""
report_export.py  ·  ProtePilot — Report Export Orchestrator
===========================================================
Coordinates: assemble → JSON → DOCX pipeline.

Usage from Streamlit:
    from src.report_export import export_global_report
    docx_path, json_path = export_global_report(intent, cache, extras)

Version : 1.0
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("ProtePilot.ReportExport")

# Path to the JS generator relative to project root
_JS_GENERATOR = os.path.join(os.path.dirname(__file__), "report_docx_generator.js")


def export_global_report(
    intent: Dict[str, Any],
    analysis_cache: Optional[Dict[str, Any]] = None,
    session_extras: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    filename_prefix: str = "ProtePilot_Report",
) -> Tuple[str, str]:
    """
    Full export pipeline: assemble → JSON + DOCX.

    Returns (docx_path, json_path).
    """
    from src.report_assembler import assemble_report

    # 1. Assemble
    report = assemble_report(intent, analysis_cache, session_extras)

    # 2. Determine output paths
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize molecule name for safe filesystem paths (remove path separators, special chars)
    raw_name = report.executive_summary.molecule_name or "molecule"
    mol_name = re.sub(r'[^\w\-.]', '_', raw_name).strip('_') or "molecule"
    base = f"{filename_prefix}_{mol_name}"

    json_path = os.path.join(output_dir, f"{base}.json")
    docx_path = os.path.join(output_dir, f"{base}.docx")

    # 3. Write JSON
    report.save_json(json_path)
    log.info("Report JSON saved: %s", json_path)

    # 4. Generate DOCX via Node.js
    try:
        result = subprocess.run(
            ["node", _JS_GENERATOR, json_path, docx_path],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(os.path.dirname(__file__)),  # project root
        )
        if result.returncode != 0:
            log.error("DOCX generation failed: %s", result.stderr)
            raise RuntimeError(f"DOCX generation error: {result.stderr[:500]}")
        log.info("Report DOCX saved: %s (%s)", docx_path, result.stdout.strip())
    except FileNotFoundError:
        log.error("Node.js not found — cannot generate DOCX")
        raise RuntimeError("Node.js is required for DOCX generation but was not found.")

    return docx_path, json_path


def _selftest():
    """Quick self-test: full export pipeline."""
    mock_intent = {
        "name": "Trastuzumab-test",
        "source": "fasta",
        "molecule_class": "canonical_mab",
        "molecule_class_info": {
            "display_name": "Canonical mAb",
            "confidence": "High",
            "has_fc_region": True,
            "expects_glycosylation": True,
            "evidence": ["IgG motifs"],
        },
        "pI": 8.45, "mw": 145.4, "gravy": -0.326,
        "hydrophobicity": 0.42, "deam_sites": 3, "ox_sites": 5,
        "acidic_residues": 80, "basic_residues": 95,
        "seq_length": 1320, "cysteine_count": 32,
        "chains": [
            {"name": "HC", "sequence": "EVQLVES...", "copy_number": 2},
            {"name": "LC", "sequence": "DIQMTQ...", "copy_number": 2},
        ],
        "chain_analyses": [
            {"name": "HC", "chain_type": "heavy", "length": 449,
             "cdrs": [{"name": "CDR-H1", "start": 26, "end": 35, "sequence": "GYTFTSYG"}],
             "liabilities": {"met_count": 3, "trp_count": 2, "deamidation_count": 2,
                             "n_glyco_count": 1, "dp_count": 0, "isomerization_count": 1,
                             "cys_count": 11, "acidic_count": 40, "basic_count": 48,
                             "risk_flags": []}},
        ],
        "liability_summary": {"deamidation_sites": 3, "oxidation_sites": 5},
    }

    mock_cache = {
        "dev_result": {
            "data": {
                "predictions": {"agg_risk": 0.18, "stability": 0.85, "viscosity_risk": 0.07},
                "score": {"score": 0.15, "grade": "Low Risk"},
                "mode": "rule_based",
            }
        },
        "predictor_source": "rule_based",
        "predictor_detail": "Rule-Based Heuristic (PropertyMapper v5.0)",
    }

    import tempfile
    out_dir = tempfile.mkdtemp(prefix="pharma_report_")
    docx_path, json_path = export_global_report(
        mock_intent, mock_cache, output_dir=out_dir,
    )

    assert os.path.exists(json_path), f"JSON not found: {json_path}"
    assert os.path.exists(docx_path), f"DOCX not found: {docx_path}"
    assert os.path.getsize(docx_path) > 5000, "DOCX too small"

    # Verify JSON structure
    with open(json_path) as f:
        data = json.load(f)
    assert data["executive_summary"]["molecule_name"] == "Trastuzumab-test"
    assert len(data["developability"]["risk_dimensions"]) == 3
    assert len(data["developability"]["qtpp_rows"]) > 0
    assert data["model_metadata"]["heuristic_ml_hybrid"] == "Heuristic"

    print(f"ReportExport _selftest PASS")
    print(f"  JSON: {json_path} ({os.path.getsize(json_path)} bytes)")
    print(f"  DOCX: {docx_path} ({os.path.getsize(docx_path)} bytes)")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
