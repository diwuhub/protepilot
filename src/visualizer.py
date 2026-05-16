"""
visualizer.py  ·  ProtePilot — Milestone 9
===========================================================
3D Protein Structure Visualization with Liability Mapping

Version   : 1.0
Author    : Di (ProtePilot)
Depends   : py3Dmol (optional), stmol (optional), numpy

Purpose
------------------------------------------------------------
Renders interactive 3D protein structures using py3Dmol / stmol
and maps sequence liabilities (oxidation, deamidation, Asp-Pro
clipping, aggregation hotspots) onto the 3D model with color-
coded residue highlighting.

Uses a default IgG1 placeholder PDB (1IGT) when no custom
structure is provided.  Falls back to a static HTML viewer
when py3Dmol is unavailable.

Color Scheme
------------------------------------------------------------
  Red     = High-risk liabilities (oxidation Met, Asp-Pro clip)
  Yellow  = Moderate-risk (deamidation hotspots, N-glycosylation)
  Orange  = SHAP-identified aggregation hotspots
  Cyan    = CDR regions
  Default = Light grey cartoon

References
------------------------------------------------------------
  py3Dmol  : https://3dmol.csb.pitt.edu/
  stmol    : Streamlit component for 3Dmol.js
  1IGT PDB : Murine IgG1 intact antibody (Harris et al., 1997)
"""

from __future__ import annotations

import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ProtePilot.Visualizer")

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------
_HAS_PY3DMOL = False
_HAS_STMOL = False

try:
    import py3Dmol
    _HAS_PY3DMOL = True
except ImportError:
    log.info("py3Dmol not installed — 3D viewer will use fallback HTML mode")

try:
    from stmol import showmol
    _HAS_STMOL = True
except ImportError:
    log.info("stmol not installed — Streamlit 3D component unavailable")


# ===========================================================================
# 1. Default PDB Data (1IGT excerpt — compact mock)
# ===========================================================================

# Minimal mock PDB representing an IgG1 Fab fragment
# Provides enough atoms to demonstrate 3D rendering and residue highlighting.
# In production, replace with actual fetched PDB or AlphaFold prediction.

_MOCK_PDB_1IGT = """\
HEADER    IMMUNOGLOBULIN                          01-MAR-97   1IGT
TITLE     INTACT IMMUNOGLOBULIN IgG1 (MOCK EXCERPT FOR PROTEPILOT AI)
REMARK   ProtePilot mock PDB — truncated for demonstration
REMARK   Real 1IGT has 1318 residues; this mock has representative atoms
ATOM      1  N   ASP H   1      27.340  24.430   2.614  1.00  9.67           N
ATOM      2  CA  ASP H   1      26.266  25.413   2.842  1.00 10.38           C
ATOM      3  C   ASP H   1      26.799  26.841   2.750  1.00  9.63           C
ATOM      4  O   ASP H   1      27.744  27.100   1.970  1.00  9.62           O
ATOM      5  N   ILE H   2      26.186  27.733   3.502  1.00  9.07           N
ATOM      6  CA  ILE H   2      26.558  29.148   3.472  1.00  8.39           C
ATOM      7  C   ILE H   2      25.424  29.974   2.892  1.00  8.78           C
ATOM      8  O   ILE H   2      24.257  29.546   2.963  1.00  9.24           O
ATOM      9  N   GLN H   3      25.765  31.117   2.317  1.00  8.00           N
ATOM     10  CA  GLN H   3      24.766  32.010   1.726  1.00  8.23           C
ATOM     11  C   GLN H   3      24.604  33.286   2.553  1.00  8.74           C
ATOM     12  O   GLN H   3      25.577  33.818   3.095  1.00 10.00           O
ATOM     13  N   MET H   4      23.405  33.821   2.621  1.00  8.16           N
ATOM     14  CA  MET H   4      23.128  35.033   3.380  1.00  8.53           C
ATOM     15  C   MET H   4      23.777  36.253   2.738  1.00  8.70           C
ATOM     16  O   MET H   4      23.685  36.413   1.520  1.00  9.30           O
ATOM     17  N   THR H   5      24.417  37.085   3.548  1.00  8.40           N
ATOM     18  CA  THR H   5      25.101  38.286   3.066  1.00  8.59           C
ATOM     19  C   THR H   5      26.614  38.171   3.229  1.00  8.14           C
ATOM     20  O   THR H   5      27.142  37.150   3.674  1.00  9.41           O
ATOM     21  N   GLN H   6      27.314  39.232   2.859  1.00  7.21           N
ATOM     22  CA  GLN H   6      28.763  39.243   2.957  1.00  7.49           C
ATOM     23  C   GLN H   6      29.224  40.695   3.039  1.00  7.86           C
ATOM     24  O   GLN H   6      28.468  41.594   3.410  1.00  8.90           O
ATOM     25  N   SER H   7      30.476  40.901   2.662  1.00  7.58           N
ATOM     26  CA  SER H   7      31.060  42.237   2.689  1.00  7.63           C
ATOM     27  C   SER H   7      30.551  43.061   1.512  1.00  7.48           C
ATOM     28  O   SER H   7      30.340  42.514   0.422  1.00  8.56           O
ATOM     29  N   PRO H   8      30.319  44.360   1.713  1.00  6.87           N
ATOM     30  CA  PRO H   8      29.816  45.263   0.669  1.00  7.06           C
ATOM     31  C   PRO H   8      30.777  45.404  -0.513  1.00  7.56           C
ATOM     32  O   PRO H   8      31.960  45.659  -0.310  1.00  8.55           O
ATOM     33  N   SER H   9      30.255  45.258  -1.735  1.00  7.73           N
ATOM     34  CA  SER H   9      31.055  45.361  -2.949  1.00  7.58           C
ATOM     35  C   SER H   9      31.563  46.790  -3.112  1.00  7.34           C
ATOM     36  O   SER H   9      30.794  47.750  -3.186  1.00  7.85           O
ATOM     37  N   SER H  10      32.881  46.905  -3.184  1.00  6.49           N
ATOM     38  CA  SER H  10      33.513  48.213  -3.364  1.00  6.68           C
ATOM     39  C   SER H  10      33.072  49.143  -2.228  1.00  6.64           C
ATOM     40  O   SER H  10      33.068  48.746  -1.063  1.00  7.75           O
ATOM     41  N   LEU H  11      32.700  50.368  -2.594  1.00  6.83           N
ATOM     42  CA  LEU H  11      32.232  51.351  -1.612  1.00  7.25           C
ATOM     43  C   LEU H  11      33.330  52.338  -1.236  1.00  7.23           C
ATOM     44  O   LEU H  11      34.297  52.487  -1.987  1.00  8.04           O
ATOM     45  N   SER H  12      33.152  53.000  -0.094  1.00  6.47           N
ATOM     46  CA  SER H  12      34.101  53.964   0.424  1.00  6.86           C
ATOM     47  C   SER H  12      33.474  55.349   0.557  1.00  7.34           C
ATOM     48  O   SER H  12      32.262  55.479   0.734  1.00  7.36           O
ATOM     49  N   ASN H  13      34.280  56.397   0.463  1.00  7.64           N
ATOM     50  CA  ASN H  13      33.791  57.767   0.574  1.00  7.81           C
ATOM     51  C   ASN H  13      33.971  58.549  -0.720  1.00  7.51           C
ATOM     52  O   ASN H  13      35.021  58.519  -1.368  1.00  7.75           O
ATOM     53  N   ALA H  14      32.951  59.261  -1.147  1.00  7.39           N
ATOM     54  CA  ALA H  14      33.013  60.086  -2.352  1.00  7.81           C
ATOM     55  C   ALA H  14      32.398  61.458  -2.104  1.00  7.63           C
ATOM     56  O   ALA H  14      31.394  61.548  -1.387  1.00  8.42           O
ATOM     57  N   ASP H  15      33.008  62.488  -2.693  1.00  8.05           N
ATOM     58  CA  ASP H  15      32.534  63.855  -2.536  1.00  8.44           C
ATOM     59  N   PRO H  16      33.200  64.700  -3.520  1.00  8.90           N
ATOM     60  CA  PRO H  16      32.850  66.100  -3.700  1.00  8.82           C
ATOM     61  N   GLY H  17      32.100  66.800  -2.700  1.00  8.50           N
ATOM     62  CA  GLY H  17      31.500  68.100  -2.900  1.00  8.60           C
ATOM     63  N   TRP H  18      31.900  68.800  -4.100  1.00  8.30           N
ATOM     64  CA  TRP H  18      31.400  70.100  -4.400  1.00  8.50           C
ATOM     65  N   TYR H  19      30.200  70.500  -3.700  1.00  8.10           N
ATOM     66  CA  TYR H  19      29.500  71.700  -4.100  1.00  8.20           C
ATOM     67  N   ARG H  20      28.300  71.500  -4.900  1.00  8.00           N
ATOM     68  CA  ARG H  20      27.400  72.600  -5.200  1.00  8.10           C
ATOM     69  N   GLN H  21      26.600  72.300  -6.350  1.00  7.90           N
ATOM     70  CA  GLN H  21      25.600  73.200  -6.900  1.00  8.00           C
ATOM     71  N   ASP H  22      25.100  74.200  -6.000  1.00  7.80           N
ATOM     72  CA  ASP H  22      24.100  75.100  -6.400  1.00  7.90           C
ATOM     73  N   PRO H  23      23.500  75.800  -5.300  1.00  7.70           N
ATOM     74  CA  PRO H  23      22.500  76.700  -5.600  1.00  7.80           C
ATOM     75  N   GLY H  24      22.000  77.400  -4.500  1.00  7.50           N
ATOM     76  CA  GLY H  24      21.000  78.300  -4.700  1.00  7.60           C
ATOM     77  N   LYS H  25      20.500  79.000  -3.600  1.00  7.30           N
ATOM     78  CA  LYS H  25      19.500  79.900  -3.800  1.00  7.40           C
ATOM     79  N   ALA H  26      19.000  80.600  -2.700  1.00  7.20           N
ATOM     80  CA  ALA H  26      18.000  81.500  -2.900  1.00  7.30           C
ATOM     81  N   THR H  27      17.500  82.200  -1.800  1.00  7.00           N
ATOM     82  CA  THR H  27      16.500  83.100  -2.000  1.00  7.10           C
ATOM     83  N   ILE H  28      16.000  83.800  -0.900  1.00  6.80           N
ATOM     84  CA  ILE H  28      15.000  84.700  -1.100  1.00  6.90           C
ATOM     85  N   SER H  29      14.500  85.400   0.000  1.00  6.60           N
ATOM     86  CA  SER H  29      13.500  86.300  -0.200  1.00  6.70           C
ATOM     87  N   CYS H  30      13.000  87.000   0.900  1.00  6.40           N
ATOM     88  CA  CYS H  30      12.000  87.900   0.700  1.00  6.50           C
ATOM     89  N   ASN H  31      11.500  88.600   1.800  1.00  6.40           N
ATOM     90  CA  ASN H  31      10.500  89.500   1.600  1.00  6.50           C
ATOM     91  N   ASP H  32      10.000  90.200   2.700  1.00  6.40           N
ATOM     92  CA  ASP H  32       9.000  91.100   2.500  1.00  6.50           C
ATOM     93  N   TYR H  33       8.500  91.800   3.600  1.00  6.40           N
ATOM     94  CA  TYR H  33       7.500  92.700   3.400  1.00  6.50           C
ATOM     95  N   GLY H  34       7.000  93.400   4.500  1.00  6.40           N
ATOM     96  CA  GLY H  34       6.000  94.300   4.300  1.00  6.50           C
ATOM     97  N   TYR H  35       5.500  95.000   5.400  1.00  6.40           N
ATOM     98  CA  TYR H  35       4.500  95.900   5.200  1.00  6.50           C
ATOM     99  N   PHE H  36       4.000  96.600   6.300  1.00  6.40           N
ATOM    100  CA  PHE H  36       3.000  97.500   6.100  1.00  6.50           C
ATOM    101  N   ASP L   1      40.340  24.430   2.614  1.00  9.67           N
ATOM    102  CA  ASP L   1      39.266  25.413   2.842  1.00 10.38           C
ATOM    103  C   ASP L   1      39.799  26.841   2.750  1.00  9.63           C
ATOM    104  O   ASP L   1      40.744  27.100   1.970  1.00  9.62           O
ATOM    105  N   ILE L   2      39.186  27.733   3.502  1.00  9.07           N
ATOM    106  CA  ILE L   2      39.558  29.148   3.472  1.00  8.39           C
ATOM    107  N   VAL L   3      38.765  31.117   2.317  1.00  8.00           N
ATOM    108  CA  VAL L   3      37.766  32.010   1.726  1.00  8.23           C
ATOM    109  N   MET L   4      36.405  33.821   2.621  1.00  8.16           N
ATOM    110  CA  MET L   4      36.128  35.033   3.380  1.00  8.53           C
ATOM    111  N   THR L   5      37.417  37.085   3.548  1.00  8.40           N
ATOM    112  CA  THR L   5      38.101  38.286   3.066  1.00  8.59           C
ATOM    113  N   GLN L   6      40.314  39.232   2.859  1.00  7.21           N
ATOM    114  CA  GLN L   6      41.763  39.243   2.957  1.00  7.49           C
ATOM    115  N   SER L   7      43.476  40.901   2.662  1.00  7.58           N
ATOM    116  CA  SER L   7      44.060  42.237   2.689  1.00  7.63           C
ATOM    117  N   PRO L   8      43.319  44.360   1.713  1.00  6.87           N
ATOM    118  CA  PRO L   8      42.816  45.263   0.669  1.00  7.06           C
ATOM    119  N   ALA L   9      43.255  45.258  -1.735  1.00  7.73           N
ATOM    120  CA  ALA L   9      44.055  45.361  -2.949  1.00  7.58           C
ATOM    121  N   LEU L  10      45.881  46.905  -3.184  1.00  6.49           N
ATOM    122  CA  LEU L  10      46.513  48.213  -3.364  1.00  6.68           C
ATOM    123  N   SER L  11      45.700  50.368  -2.594  1.00  6.83           N
ATOM    124  CA  SER L  11      45.232  51.351  -1.612  1.00  7.25           C
ATOM    125  N   CYS L  12      46.152  53.000  -0.094  1.00  6.47           N
ATOM    126  CA  CYS L  12      47.101  53.964   0.424  1.00  6.86           C
ATOM    127  N   ASN L  13      47.280  56.397   0.463  1.00  7.64           N
ATOM    128  CA  ASN L  13      46.791  57.767   0.574  1.00  7.81           C
ATOM    129  N   TYR L  14      45.951  59.261  -1.147  1.00  7.39           N
ATOM    130  CA  TYR L  14      46.013  60.086  -2.352  1.00  7.81           C
ATOM    131  N   GLY L  15      46.008  62.488  -2.693  1.00  8.05           N
ATOM    132  CA  GLY L  15      45.534  63.855  -2.536  1.00  8.44           C
END
"""


# ===========================================================================
# 2. Liability Mapping
# ===========================================================================

# Residue type → risk colors
LIABILITY_COLORS = {
    "oxidation_met": {"color": "red", "label": "Met Oxidation (High Risk)"},
    "oxidation_trp": {"color": "red", "label": "Trp Oxidation (High Risk)"},
    "deamidation":   {"color": "yellow", "label": "Deamidation Hotspot"},
    "asp_pro_clip":  {"color": "red", "label": "Asp-Pro Clipping Risk"},
    "n_glycosylation": {"color": "yellow", "label": "N-Glycosylation Motif"},
    "shap_hotspot":  {"color": "orange", "label": "SHAP Aggregation Hotspot"},
    "cdr_region":    {"color": "cyan", "label": "CDR Region"},
}


def extract_liability_residues(
    intent: Dict[str, Any],
    shap_hotspots: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract residue-level liabilities from the parsed intent dict.

    Returns a list of dicts with keys:
      - chain: "H" or "L"
      - resnum: 1-based residue number
      - resname: 3-letter AA code
      - liability_type: key from LIABILITY_COLORS
      - color: hex or named color for 3D highlighting
      - label: human-readable label
    """
    residues = []

    for ca in intent.get("chain_analyses", []):
        chain_type = ca.get("chain_type", "heavy")
        chain_id = "H" if "heavy" in chain_type.lower() else "L"
        liab = ca.get("liabilities", {})
        seq = ca.get("sequence", "")

        # Methionine oxidation sites
        for pos in liab.get("met_positions", []):
            if 0 < pos <= len(seq):
                residues.append({
                    "chain": chain_id,
                    "resnum": pos,
                    "resname": "MET",
                    "liability_type": "oxidation_met",
                    "color": LIABILITY_COLORS["oxidation_met"]["color"],
                    "label": f"Met{pos} oxidation ({chain_id})",
                })

        # Tryptophan oxidation sites
        for pos in liab.get("trp_positions", []):
            if 0 < pos <= len(seq):
                residues.append({
                    "chain": chain_id,
                    "resnum": pos,
                    "resname": "TRP",
                    "liability_type": "oxidation_trp",
                    "color": LIABILITY_COLORS["oxidation_trp"]["color"],
                    "label": f"Trp{pos} oxidation ({chain_id})",
                })

        # Deamidation hotspots
        for hotspot in liab.get("deamidation_hotspots", []):
            if isinstance(hotspot, dict):
                pos = hotspot.get("position", 0)
            else:
                pos = hotspot
            if 0 < pos <= len(seq):
                residues.append({
                    "chain": chain_id,
                    "resnum": pos,
                    "resname": "ASN",
                    "liability_type": "deamidation",
                    "color": LIABILITY_COLORS["deamidation"]["color"],
                    "label": f"Asn{pos} deamidation ({chain_id})",
                })

        # Asp-Pro clipping
        for pos in liab.get("dp_positions", []):
            if 0 < pos <= len(seq):
                residues.append({
                    "chain": chain_id,
                    "resnum": pos,
                    "resname": "ASP",
                    "liability_type": "asp_pro_clip",
                    "color": LIABILITY_COLORS["asp_pro_clip"]["color"],
                    "label": f"Asp{pos}-Pro clipping ({chain_id})",
                })

        # N-glycosylation motifs
        for motif in liab.get("n_glyco_motifs", []):
            if isinstance(motif, dict):
                pos = motif.get("position", 0)
            else:
                pos = motif
            if 0 < pos <= len(seq):
                residues.append({
                    "chain": chain_id,
                    "resnum": pos,
                    "resname": "ASN",
                    "liability_type": "n_glycosylation",
                    "color": LIABILITY_COLORS["n_glycosylation"]["color"],
                    "label": f"N-glyco motif at {pos} ({chain_id})",
                })

        # CDR regions
        for cdr in ca.get("cdrs", []):
            start = cdr.get("start", 0)
            end = cdr.get("end", 0)
            cdr_name = cdr.get("name", "CDR")
            for pos in range(start, end + 1):
                if 0 < pos <= len(seq):
                    residues.append({
                        "chain": chain_id,
                        "resnum": pos,
                        "resname": seq[pos - 1] if pos <= len(seq) else "X",
                        "liability_type": "cdr_region",
                        "color": LIABILITY_COLORS["cdr_region"]["color"],
                        "label": f"{cdr_name} ({chain_id})",
                    })

    # SHAP-identified aggregation hotspots
    if shap_hotspots:
        for pos in shap_hotspots:
            residues.append({
                "chain": "H",
                "resnum": pos,
                "resname": "UNK",
                "liability_type": "shap_hotspot",
                "color": LIABILITY_COLORS["shap_hotspot"]["color"],
                "label": f"SHAP hotspot at {pos}",
            })

    return residues


# ===========================================================================
# 3. py3Dmol Viewer Builder
# ===========================================================================

def _color_to_hex(name: str) -> str:
    """Convert named color to hex for py3Dmol."""
    mapping = {
        "red": "0xFF0000",
        "yellow": "0xFFD700",
        "orange": "0xFF8C00",
        "cyan": "0x00CED1",
        "green": "0x00FF00",
        "blue": "0x0000FF",
        "white": "0xFFFFFF",
        "grey": "0xC0C0C0",
    }
    return mapping.get(name, "0xFF00FF")


def build_3d_viewer(
    pdb_data: Optional[str] = None,
    liability_residues: Optional[List[Dict[str, Any]]] = None,
    width: int = 700,
    height: int = 500,
    style: str = "cartoon",
    background_color: str = "white",
) -> Any:
    """
    Build a py3Dmol viewer with liability residue highlighting.

    Parameters
    ----------
    pdb_data : PDB format string. Uses mock 1IGT if None.
    liability_residues : Output from extract_liability_residues()
    width, height : Viewer dimensions in pixels
    style : Base style ("cartoon", "stick", "sphere")
    background_color : Background color

    Returns
    -------
    py3Dmol.view object if py3Dmol available, else HTML string
    """
    if pdb_data is None:
        pdb_data = _MOCK_PDB_1IGT

    if liability_residues is None:
        liability_residues = []

    if not _HAS_PY3DMOL:
        return _build_fallback_html(pdb_data, liability_residues, width, height)

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, "pdb")

    # Base style: light grey cartoon
    view.setStyle(
        {},
        {"cartoon": {"color": "spectrum", "opacity": 0.85}}
    )

    # Color chain H and L differently
    view.setStyle(
        {"chain": "H"},
        {"cartoon": {"color": "#A8D8EA", "opacity": 0.80}}
    )
    view.setStyle(
        {"chain": "L"},
        {"cartoon": {"color": "#D4A5A5", "opacity": 0.80}}
    )

    # Highlight liability residues
    # Process by priority: CDRs first (lowest), then liabilities on top
    priority_order = [
        "cdr_region", "n_glycosylation", "deamidation",
        "shap_hotspot", "oxidation_trp", "oxidation_met", "asp_pro_clip"
    ]

    sorted_residues = sorted(
        liability_residues,
        key=lambda r: priority_order.index(r["liability_type"])
        if r["liability_type"] in priority_order else 99
    )

    for res in sorted_residues:
        color = _color_to_hex(res["color"]).replace("0x", "#")
        selector = {"chain": res["chain"], "resi": res["resnum"]}

        if res["liability_type"] == "cdr_region":
            view.addStyle(selector, {"cartoon": {"color": color, "opacity": 0.90}})
        else:
            # Show liabilities as sticks + spheres for emphasis
            view.addStyle(selector, {"stick": {"color": color, "radius": 0.3}})
            view.addStyle(selector, {"sphere": {"color": color, "radius": 0.6, "opacity": 0.7}})

        # Add label for non-CDR liabilities
        if res["liability_type"] != "cdr_region":
            view.addLabel(
                res["label"],
                {
                    "fontSize": 10,
                    "fontColor": "black",
                    "backgroundColor": color,
                    "backgroundOpacity": 0.6,
                    "position": {"x": 0, "y": 0, "z": 0},
                },
                selector,
            )

    view.setBackgroundColor(background_color)
    view.zoomTo()
    view.spin(False)

    return view


def _build_fallback_html(
    pdb_data: str,
    liability_residues: List[Dict[str, Any]],
    width: int,
    height: int,
) -> str:
    """
    Build a static HTML fallback when py3Dmol is not available.
    Uses 3Dmol.js CDN for client-side rendering.
    """
    # Filter non-CDR liabilities for the highlight list
    highlights = [r for r in liability_residues if r["liability_type"] != "cdr_region"]

    highlight_js = ""
    for res in highlights:
        color = _color_to_hex(res["color"])
        highlight_js += f"""
        viewer.addStyle(
            {{chain: "{res['chain']}", resi: {res['resnum']}}},
            {{stick: {{color: {color}, radius: 0.3}}, sphere: {{color: {color}, radius: 0.6, opacity: 0.7}}}}
        );
        viewer.addLabel(
            "{res['label']}",
            {{fontSize: 10, fontColor: "black", backgroundColor: "{res['color']}", backgroundOpacity: 0.6}},
            {{chain: "{res['chain']}", resi: {res['resnum']}}}
        );"""

    # Escape PDB for JS embedding
    pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html = f"""
    <div id="container-3d" style="width:{width}px;height:{height}px;position:relative;border:1px solid #ccc;border-radius:8px;overflow:hidden;">
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
    (function() {{
        var viewer = $3Dmol.createViewer("container-3d", {{
            defaultcolors: $3Dmol.rasmolElementColors,
            backgroundColor: "white"
        }});
        var pdbData = `{pdb_escaped}`;
        viewer.addModel(pdbData, "pdb");
        viewer.setStyle({{}}, {{cartoon: {{color: "spectrum", opacity: 0.85}}}});
        viewer.setStyle({{chain: "H"}}, {{cartoon: {{color: "#A8D8EA", opacity: 0.80}}}});
        viewer.setStyle({{chain: "L"}}, {{cartoon: {{color: "#D4A5A5", opacity: 0.80}}}});
        {highlight_js}
        viewer.zoomTo();
        viewer.render();
    }})();
    </script>
    """
    return html


# ===========================================================================
# 4. Streamlit Integration Helpers
# ===========================================================================

def render_3d_in_streamlit(
    view_or_html: Any,
    height: int = 500,
) -> None:
    """
    Render a 3D viewer in Streamlit using stmol or st.components.

    Parameters
    ----------
    view_or_html : py3Dmol.view object or HTML string
    height : Display height in Streamlit
    """
    import hashlib

    try:
        import streamlit as st
    except ImportError:
        log.warning("Streamlit not available for 3D rendering")
        return

    # Convert view_or_html to HTML string
    if isinstance(view_or_html, str):
        html_str = view_or_html
    elif _HAS_PY3DMOL:
        # Use _make_html() fallback to generate HTML from py3Dmol view
        html_str = view_or_html._make_html()
    elif _HAS_STMOL:
        # Fallback: try to generate HTML from stmol/py3Dmol view
        html_str = view_or_html._make_html() if hasattr(view_or_html, '_make_html') else str(view_or_html)
    else:
        st.info("Install py3Dmol and stmol for interactive 3D protein visualization.")
        return

    # Generate unique key based on HTML content hash to prevent stale renders
    content_hash = hashlib.md5(html_str.encode()).hexdigest()[:8]
    unique_key = f"3d_viewer_{content_hash}"

    # Always use HTML component with unique key for consistent rendering
    st.components.v1.html(html_str, height=height + 50, scrolling=False, key=unique_key)


def build_liability_legend() -> str:
    """Build an HTML legend for the liability color scheme."""
    items = []
    for key, info in LIABILITY_COLORS.items():
        if key == "cdr_region":
            continue  # Skip CDR in legend for clarity
        color_css = {
            "red": "#FF0000", "yellow": "#FFD700",
            "orange": "#FF8C00", "cyan": "#00CED1",
        }.get(info["color"], "#FF00FF")
        items.append(
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{color_css};border-radius:3px;margin-right:6px;'
            f'vertical-align:middle;"></span>'
            f'<span style="vertical-align:middle;font-size:13px;">{info["label"]}</span>'
        )

    return (
        '<div style="display:flex;flex-wrap:wrap;gap:16px;padding:8px 0;">'
        + "".join(f'<div style="display:flex;align-items:center;">{item}</div>' for item in items)
        + '</div>'
    )


def generate_liability_summary(
    liability_residues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarize liabilities for display in UI panels.

    Returns dict with counts and lists by category.
    """
    summary = {
        "total": 0,
        "by_type": {},
        "by_chain": {"H": [], "L": []},
        "high_risk": [],
        "moderate_risk": [],
    }

    for res in liability_residues:
        if res["liability_type"] == "cdr_region":
            continue  # Don't count CDRs as liabilities

        summary["total"] += 1

        lt = res["liability_type"]
        if lt not in summary["by_type"]:
            summary["by_type"][lt] = []
        summary["by_type"][lt].append(res)

        chain = res.get("chain", "H")
        if chain in summary["by_chain"]:
            summary["by_chain"][chain].append(res)

        if res["color"] == "red":
            summary["high_risk"].append(res)
        elif res["color"] in ("yellow", "orange"):
            summary["moderate_risk"].append(res)

    return summary


# ===========================================================================
# 5. SHAP-to-Residue Mapping (Heuristic)
# ===========================================================================

def map_shap_to_residues(
    shap_result: Optional[Dict[str, Any]] = None,
    intent: Optional[Dict[str, Any]] = None,
    top_n: int = 5,
) -> List[int]:
    """
    Map SHAP feature attributions to approximate residue positions.

    Since SHAP operates on embedding dimensions (not individual residues),
    this uses a heuristic mapping:
      - Top contributing embedding dimensions are mapped to residue
        positions in CDR regions (most variable regions).
      - If CDR info is available, distributes hotspots across CDRs.

    Returns list of 1-based residue positions.
    """
    if shap_result is None:
        return []

    hotspot_positions = []

    # Try to get CDR positions from intent
    cdr_ranges = []
    if intent:
        for ca in intent.get("chain_analyses", []):
            if "heavy" in ca.get("chain_type", "").lower():
                for cdr in ca.get("cdrs", []):
                    cdr_ranges.append((cdr.get("start", 1), cdr.get("end", 10)))

    if cdr_ranges:
        # Distribute hotspots across CDR midpoints
        for i, (start, end) in enumerate(cdr_ranges[:top_n]):
            mid = (start + end) // 2
            hotspot_positions.append(mid)
    else:
        # Fallback: use approximate CDR3 positions for typical IgG
        # VH CDR3 is roughly positions 95-102 (Kabat numbering)
        for offset in range(top_n):
            hotspot_positions.append(95 + offset * 2)

    return hotspot_positions[:top_n]


# ===========================================================================
# __main__: Standalone Test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  ProtePilot — Visualizer v1.0 Test")
    print("=" * 60)

    # Mock intent
    mock_intent = {
        "chain_analyses": [
            {
                "chain_type": "heavy",
                "sequence": "DIQMTQSPSSLSASVGDRVTITCRASQD" + "A" * 80,
                "liabilities": {
                    "met_positions": [4],
                    "trp_positions": [18],
                    "deamidation_hotspots": [13, 31],
                    "dp_positions": [15, 22],
                    "n_glyco_motifs": [],
                },
                "cdrs": [
                    {"name": "CDR-H1", "start": 26, "end": 35},
                    {"name": "CDR-H2", "start": 50, "end": 65},
                    {"name": "CDR-H3", "start": 95, "end": 102},
                ],
            },
            {
                "chain_type": "light",
                "sequence": "DIVMTQSPALMSASPGEKVTMTC" + "S" * 80,
                "liabilities": {
                    "met_positions": [4],
                    "trp_positions": [],
                    "deamidation_hotspots": [13],
                    "dp_positions": [],
                    "n_glyco_motifs": [{"position": 50}],
                },
                "cdrs": [
                    {"name": "CDR-L1", "start": 24, "end": 34},
                    {"name": "CDR-L2", "start": 50, "end": 56},
                    {"name": "CDR-L3", "start": 89, "end": 97},
                ],
            },
        ]
    }

    # Test liability extraction
    liabilities = extract_liability_residues(mock_intent, shap_hotspots=[97, 99, 101])
    print(f"\nExtracted {len(liabilities)} residue annotations")

    non_cdr = [r for r in liabilities if r["liability_type"] != "cdr_region"]
    print(f"  Liabilities (non-CDR): {len(non_cdr)}")
    for r in non_cdr:
        print(f"    {r['chain']}:{r['resnum']} {r['liability_type']} — {r['label']}")

    # Test summary
    summary = generate_liability_summary(liabilities)
    print(f"\nLiability Summary:")
    print(f"  Total: {summary['total']}")
    print(f"  High risk: {len(summary['high_risk'])}")
    print(f"  Moderate risk: {len(summary['moderate_risk'])}")

    # Test 3D viewer build
    viewer = build_3d_viewer(liability_residues=liabilities)
    print(f"\n3D viewer type: {type(viewer).__name__}")
    print(f"  py3Dmol available: {_HAS_PY3DMOL}")
    print(f"  stmol available: {_HAS_STMOL}")

    # Test SHAP mapping
    hotspots = map_shap_to_residues(shap_result={"agg_risk": {}}, intent=mock_intent)
    print(f"\nSHAP hotspot positions: {hotspots}")

    # Test legend
    legend = build_liability_legend()
    print(f"\nLegend HTML length: {len(legend)} chars")

    print("\nVisualizer v1.0 test complete")
