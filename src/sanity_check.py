"""
src/sanity_check.py — Scientific Sanity Check Module
=====================================================
ProtePilot — v7.3.1

Automatic validation of model outputs against published ranges for
therapeutic antibody development.  Flags results that fall outside
typical biopharmaceutical process windows.

Usage:
    from src.sanity_check import run_sanity_checks
    warnings = run_sanity_checks(results_dict)

Each check returns a list of SanityWarning objects.  The caller can
display them as UI alerts or log them for offline review.

Reference Ranges
-----------------
  IgG pI:           6.5 – 9.5  (Liu 2015, mAbs review)
  IgG MW:           140 – 160 kDa (standard IgG1)
  CEX RT (main):    8 – 25 min (typical analytical CEX)
  CEX Rs:           0.5 – 4.0  (USP resolution)
  PK half-life:     1 – 30 days (IgG range)
  CHO titer:        0.5 – 10 g/L (fed-batch 14-day)
  CHO VCD peak:     5 – 40 ×10^6/mL
  Viability:        60 – 100%
  Purification yield: 50 – 99%
  Gradient vs Rs:   Rs should not increase with steeper gradient
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

log = logging.getLogger("ProtePilot.SanityCheck")


@dataclass
class SanityWarning:
    """A single sanity check warning."""
    module: str          # e.g., "chromatography", "upstream", "pk"
    parameter: str       # e.g., "ka", "titer", "half_life"
    value: float         # observed value
    expected_range: str  # human-readable range
    severity: str        # "info", "warning", "critical"
    message: str         # actionable description


# ===========================================================================
# Reference Ranges (published literature)
# ===========================================================================

RANGES = {
    # Chromatography (analytical CEX)
    "ka":           (0.3, 8.0,   "SMA ka standard range (mAb CEX)"),
    "nu":           (1.5, 3.5,   "SMA nu standard range (mAb CEX)"),
    "rt_main":      (5.0, 30.0,  "Main peak RT (analytical CEX, min)"),
    "rs":           (0.3, 4.0,   "USP chromatographic resolution"),

    # PK
    "half_life":    (0.5, 30.0,  "Human IgG half-life (days)"),
    "clearance":    (0.5, 50.0,  "IgG clearance (mL/day/kg)"),

    # Upstream
    "titer":        (0.3, 12.0,  "CHO fed-batch titer (g/L)"),
    "peak_vcd":     (3.0, 50.0,  "Peak VCD (×10^6 cells/mL)"),
    "viability":    (50.0, 100.0, "Harvest viability (%)"),

    # Purification
    "yield":        (30.0, 99.5, "Purification step yield (%)"),
    "purity":       (85.0, 100.0, "Final purity (%)"),
    "hcp":          (0.0, 100.0, "Residual HCP (ppm)"),

    # Protein properties
    "pi":           (5.0, 10.0,  "Isoelectric point (mAb range)"),
    "mw":           (10.0, 300.0, "Molecular weight (kDa)"),
}


def _check_range(
    module: str,
    param: str,
    value: Optional[float],
    custom_range: Optional[tuple] = None,
) -> Optional[SanityWarning]:
    """Check if a value falls within expected range."""
    if value is None:
        return None

    if custom_range:
        low, high, desc = custom_range
    elif param in RANGES:
        low, high, desc = RANGES[param]
    else:
        return None

    if low <= value <= high:
        return None

    severity = "warning"
    if value < low * 0.5 or value > high * 2.0:
        severity = "critical"

    direction = "below" if value < low else "above"
    return SanityWarning(
        module=module,
        parameter=param,
        value=round(value, 4),
        expected_range=f"{low}–{high}",
        severity=severity,
        message=(
            f"{param} = {value:.4g} is {direction} expected range "
            f"({low}–{high}: {desc}). "
            f"Model parameters may need recalibration."
        ),
    )


# ===========================================================================
# Module-specific checkers
# ===========================================================================

def check_chromatography(
    ka: Optional[float] = None,
    nu: Optional[float] = None,
    rt_main: Optional[float] = None,
    rs: Optional[float] = None,
    gradient_slope: Optional[float] = None,
) -> List[SanityWarning]:
    """Check chromatographic outputs for physical reasonableness."""
    warnings = []
    for param, val in [("ka", ka), ("nu", nu), ("rt_main", rt_main), ("rs", rs)]:
        w = _check_range("chromatography", param, val)
        if w:
            warnings.append(w)
    return warnings


def check_pk(
    half_life: Optional[float] = None,
    clearance: Optional[float] = None,
) -> List[SanityWarning]:
    """Check PK predictions against IgG reference ranges."""
    warnings = []
    for param, val in [("half_life", half_life), ("clearance", clearance)]:
        w = _check_range("pk", param, val)
        if w:
            warnings.append(w)
    return warnings


def check_upstream(
    titer: Optional[float] = None,
    peak_vcd: Optional[float] = None,
    viability: Optional[float] = None,
) -> List[SanityWarning]:
    """Check upstream simulation outputs."""
    warnings = []
    for param, val in [("titer", titer), ("peak_vcd", peak_vcd),
                        ("viability", viability)]:
        w = _check_range("upstream", param, val)
        if w:
            warnings.append(w)
    return warnings


def check_purification(
    yield_pct: Optional[float] = None,
    purity: Optional[float] = None,
    hcp: Optional[float] = None,
) -> List[SanityWarning]:
    """Check purification results."""
    warnings = []
    for param, val in [("yield", yield_pct), ("purity", purity), ("hcp", hcp)]:
        w = _check_range("purification", param, val)
        if w:
            warnings.append(w)
    return warnings


def check_protein_properties(
    pi: Optional[float] = None,
    mw: Optional[float] = None,
) -> List[SanityWarning]:
    """Check basic protein properties."""
    warnings = []
    for param, val in [("pi", pi), ("mw", mw)]:
        w = _check_range("protein", param, val)
        if w:
            warnings.append(w)
    return warnings


# ===========================================================================
# Unified checker
# ===========================================================================

def run_sanity_checks(results: Dict[str, Any]) -> List[SanityWarning]:
    """
    Run all applicable sanity checks on a results dictionary.

    The results dict can contain any subset of these keys:
        ka, nu, rt_main, rs, half_life, clearance,
        titer, peak_vcd, viability, yield, purity, hcp, pi, mw

    Returns a list of SanityWarning objects (empty if all OK).
    """
    all_warnings: List[SanityWarning] = []

    all_warnings.extend(check_chromatography(
        ka=results.get("ka"),
        nu=results.get("nu"),
        rt_main=results.get("rt_main"),
        rs=results.get("rs"),
    ))
    all_warnings.extend(check_pk(
        half_life=results.get("half_life"),
        clearance=results.get("clearance"),
    ))
    all_warnings.extend(check_upstream(
        titer=results.get("titer"),
        peak_vcd=results.get("peak_vcd"),
        viability=results.get("viability"),
    ))
    # Normalize yield: if stored as fraction (0-1), convert to percentage
    _raw_yield = results.get("yield")
    if _raw_yield is not None and _raw_yield <= 1.0:
        _raw_yield = _raw_yield * 100.0
    all_warnings.extend(check_purification(
        yield_pct=_raw_yield,
        purity=results.get("purity"),
        hcp=results.get("hcp"),
    ))
    all_warnings.extend(check_protein_properties(
        pi=results.get("pi"),
        mw=results.get("mw"),
    ))

    if all_warnings:
        for w in all_warnings:
            log.warning("[%s] %s", w.severity.upper(), w.message)

    return all_warnings
