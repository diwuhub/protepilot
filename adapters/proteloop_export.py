"""
Export ProtePilot developability assessment as ProteLoop-compatible JSON.

Usage:
    python adapters/proteloop_export.py --molecule trastuzumab --output risk_flags.json
    python adapters/proteloop_export.py <protepilot_result.json>

Output schema:
{
    "molecule_name": str,
    "molecule_format": str,
    "risk_flags": {
        "aggregation": {"score": float, "level": "high"|"medium"|"low"},
        "stability": {"score": float, "level": ...},
        "immunogenicity": {"score": float, "level": ...},
        "manufacturability": {"score": float, "level": ...},
        "expression": {"score": float, "level": ...}
    },
    "recommended_loops": [str],
    "protepilot_version": str,
    "timestamp": str
}
"""
import json
import sys
from datetime import datetime, timezone


# Map risk dimensions to ProteLoop loops that can help.
# Core 5 dimensions are always scored. Extended dimensions (glycosylation,
# hcp, artifact) are scored only if present in developability_scores.
RISK_TO_LOOP = {
    "aggregation": "formulation_dsf",
    "stability": "stability_trend",
    "immunogenicity": "ms_peptide_map",
    "manufacturability": "dsp_bispecific",
    "expression": "formulation_dsf",
    "glycosylation": "ms_o_glycan",
    "hcp": "ms_hcp",
    "artifact": "ms_artifact_triage",
}

# Core dimensions (always present in developability_scores)
_CORE_DIMS = ["aggregation", "stability", "immunogenicity", "manufacturability", "expression"]
# Extended dimensions (scored if present in input)
_EXTENDED_DIMS = ["glycosylation", "hcp", "artifact"]


def export_risk_flags(molecule_result: dict) -> dict:
    """Convert ProtePilot developability result to ProteLoop trigger format."""
    scores = molecule_result.get("developability_scores", {})
    flags = {}
    recommended = []

    # Score core dimensions (always present, default 0.5 if missing)
    # and extended dimensions (only if explicitly provided in scores)
    dims_to_score = list(_CORE_DIMS)
    for ext in _EXTENDED_DIMS:
        if ext in scores:
            dims_to_score.append(ext)

    for dim in dims_to_score:
        score = scores.get(dim, 0.5)
        level = "high" if score < 0.4 else "medium" if score < 0.7 else "low"
        flags[dim] = {"score": round(score, 4), "level": level}

        if level in ("high", "medium"):
            loop = RISK_TO_LOOP.get(dim)
            if loop and loop not in recommended:
                recommended.append(loop)

    return {
        "molecule_name": molecule_result.get("name", "unknown"),
        "molecule_format": molecule_result.get("format", "unknown"),
        "risk_flags": flags,
        "recommended_loops": recommended,
        "protepilot_version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            result = json.load(f)
        output = export_risk_flags(result)
        print(json.dumps(output, indent=2))
    else:
        print("Usage: python adapters/proteloop_export.py <protepilot_result.json>")
        sys.exit(1)
