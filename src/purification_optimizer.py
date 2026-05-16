"""
src/purification_optimizer.py — GoSilico-Style In-Silico DoE for Purification
=============================================================================
ProtePilot — Milestone 19 · Version 2.0

Performs Design-of-Experiments (DoE) grid search over elution conditions
(pH, gradient steepness, salt elute, loading) to maximize chromatographic
Resolution (Rs), Yield, and Purity.  Produces contour-plot-ready matrices
and identifies the optimal "sweet spot" in the Design Space.

Version 2.0 enhancements
-------------------------
  - pI-adaptive pH range: DoE pH bounds auto-derived from molecule pI
  - salt_elute as optimization variable (previously hardcoded 500 mM)
  - c_load (loading density) as third DoE dimension
  - Multi-objective with purity hard constraint (≥95%) and Rs floor (≥1.5)
  - HCP clearance model: literature-calibrated baseline per step
  - Multi-segment gradient shapes: step + linear + concave + convex
  - Scale-up transfer function: van Deemter + axial dispersion correction

Architecture
------------
  pH grid (pI-adaptive)  ×  Gradient grid  ×  Salt_elute grid  ×  Load grid
       ↓ PropertyMapper (pH → SMA params per variant)
       ↓ estimate_rt_from_sma → retention times
       ↓ Peak shape model → Gaussian peaks → Resolution & Yield
       ↓ HCP clearance model → purity + impurity tracking
       ↓ Objective = w_Rs * Rs + w_Yield * Yield + w_Purity * Purity
       ↓ Hard constraints: Purity ≥ 95%, Rs ≥ 1.5
       → Optimal sweet spot + contour matrices

Graceful degradation
--------------------
• PropertyMapper unavailable → analytical SMA approximation from pI
• No external CADET binary needed — uses Yamamoto theory directly
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ===========================================================================
# 1. Data Classes
# ===========================================================================

@dataclass
class DoECondition:
    """Single DoE grid point."""
    elution_ph: float
    gradient_slope_mM_min: float


@dataclass
class DoEResult:
    """Result for a single DoE grid point."""
    elution_ph: float
    gradient_slope: float
    rt_acidic: float
    rt_main: float
    rt_basic: float
    fwhm_acidic: float
    fwhm_main: float
    fwhm_basic: float
    resolution_acidic_main: float
    resolution_main_basic: float
    resolution_min: float          # min(Rs_AM, Rs_MB)
    yield_main: float              # estimated main-peak recovery (0-1)
    objective: float               # combined score
    # v2.0 additions
    salt_elute: float = 500.0      # salt elute concentration (mM)
    c_load_mg_ml: float = 30.0     # column loading (mg/mL resin)
    pool_purity_pct: float = 0.0   # estimated pool purity (%)
    hcp_ppm: float = 0.0           # estimated residual HCP (ppm)
    gradient_shape: str = "linear" # "linear", "step", "concave", "convex"
    feasible: bool = True          # passes hard constraints?


@dataclass
class DoEOptimization:
    """Full DoE grid search result."""
    ph_values: List[float]
    gradient_values: List[float]
    resolution_matrix: np.ndarray   # shape (n_ph, n_grad)
    yield_matrix: np.ndarray        # shape (n_ph, n_grad)
    objective_matrix: np.ndarray    # shape (n_ph, n_grad)
    all_results: List[DoEResult]    # flattened list
    optimal: DoEResult              # best overall
    optimal_ph: float
    optimal_gradient: float
    wall_time_s: float
    n_conditions: int
    weights: Dict[str, float] = field(default_factory=lambda: {"resolution": 0.6, "yield": 0.4})
    # v2.0 additions
    salt_elute_values: List[float] = field(default_factory=list)
    load_values: List[float] = field(default_factory=list)
    purity_matrix: Optional[np.ndarray] = None
    hcp_matrix: Optional[np.ndarray] = None
    ph_range_source: str = "default"   # "pI_adaptive" | "default" | "user"


# ===========================================================================
# 1b. HCP Clearance Model (Literature-Calibrated Baseline)
# ===========================================================================
#
# Based on published data:
# - Protein A capture: 8000 ppm → 100-500 ppm (Urmann 2010, DOI:10.4161/mabs.12303)
# - CEX polishing: ~95% clearance, load-dependent (Wolfe 2014, DOI:10.1016/j.chroma.2014.02.086)
# - AEX flowthrough: ~70-90% clearance at optimal conditions (JCHROMB 2025, DOI:10.1016/j.jchromb.2025.124796)
# - Mixed-mode: reduces to 21-23 ppm from 180-190 ppm (JCHROMB 2021, DOI:10.1016/j.jchromb.2021.123072)

def estimate_hcp_clearance(
    step_type: str,
    hcp_in_ppm: float,
    c_load_mg_ml: float = 30.0,
    resolution: float = 1.5,
    ph: float = 6.0,
    pI: float = 8.5,
) -> Dict[str, float]:
    """
    Estimate HCP clearance for a chromatographic step.

    Literature baseline with load-dependent correction:
    - Higher loading → reduced HCP clearance (overloading effect)
    - Better resolution → better separation of HCP from product
    - pH closer to pI → stronger binding → more co-purification of weakly bound HCP

    Parameters
    ----------
    step_type     : "capture", "cex_polishing", "aex_flowthrough", "filtration"
    hcp_in_ppm    : Input HCP concentration (ppm)
    c_load_mg_ml  : Column loading (mg/mL resin)
    resolution    : Chromatographic resolution (for polishing steps)
    ph            : Operating pH
    pI            : Protein isoelectric point

    Returns
    -------
    dict with hcp_out_ppm, log_reduction, clearance_pct
    """
    if step_type == "capture":
        # Protein A: Baseline ~95% clearance at 30 mg/mL load
        # Literature: 8000 → 100-500 ppm; LRV ~1.2-1.9
        base_clearance = 0.95
        # Load correction: clearance drops at high loading
        load_factor = max(0.85, 1.0 - 0.003 * max(0, c_load_mg_ml - 30.0))
        clearance = base_clearance * load_factor

    elif step_type == "cex_polishing":
        # CEX: Baseline ~93-97% clearance depending on resolution and load
        # Literature: Wolfe 2014 — multimodal CEX achieved 12x reduction
        base_clearance = 0.93
        # Resolution bonus: better Rs → cleaner cut → less HCP in pool
        rs_bonus = min(0.04, 0.02 * max(0, resolution - 1.0))
        # Load penalty: >40 mg/mL reduces clearance
        load_penalty = max(0.0, 0.002 * max(0, c_load_mg_ml - 40.0))
        # pH-pI proximity penalty: operating closer to pI = weaker selectivity
        ph_penalty = max(0.0, 0.01 * max(0, 1.0 - abs(pI - ph)))
        clearance = min(0.99, base_clearance + rs_bonus - load_penalty - ph_penalty)

    elif step_type == "aex_flowthrough":
        # AEX FT: Baseline ~70-90% clearance
        # Literature: 3M Polisher ST achieved 8000 → 10 ppm at competitive loads
        # Standard Q Sepharose: ~70% clearance baseline
        base_clearance = 0.75
        # Load correction
        load_factor = max(0.6, 1.0 - 0.004 * max(0, c_load_mg_ml - 50.0))
        clearance = base_clearance * load_factor

    elif step_type == "filtration":
        # Viral filtration: minimal HCP removal (~5-10%)
        clearance = 0.08
    else:
        clearance = 0.0

    clearance = float(np.clip(clearance, 0.0, 0.999))
    hcp_out = hcp_in_ppm * (1.0 - clearance)
    hcp_out = max(0.5, hcp_out)

    log_reduction = np.log10(max(hcp_in_ppm, 1.0) / max(hcp_out, 0.1))

    return {
        "hcp_out_ppm": round(hcp_out, 1),
        "hcp_in_ppm": round(hcp_in_ppm, 1),
        "log_reduction": round(float(log_reduction), 2),
        "clearance_pct": round(clearance * 100, 1),
    }


# ===========================================================================
# 1c. Multi-Segment Gradient Shapes
# ===========================================================================
#
# Beyond simple linear gradients, real process development uses:
# - Step gradients: instantaneous salt jump (wash + elution plateau)
# - Concave gradients: slow start, rapid finish (better for closely eluting species)
# - Convex gradients: rapid start, slow finish (useful for broadly distributed variants)
# The shape affects peak spacing and resolution.

def gradient_salt_profile(
    t: np.ndarray,
    t_start: float,
    t_end: float,
    c_start: float,
    c_end: float,
    shape: str = "linear",
    step_fraction: float = 0.5,
) -> np.ndarray:
    """
    Generate salt concentration profile for a gradient segment.

    Parameters
    ----------
    t            : Time array (min)
    t_start      : Gradient start time (min)
    t_end        : Gradient end time (min)
    c_start      : Start salt concentration (mM)
    c_end        : End salt concentration (mM)
    shape        : "linear", "step", "concave", "convex"
    step_fraction: For step gradient, fraction of gradient range at which step occurs

    Returns
    -------
    np.ndarray : Salt concentration at each time point
    """
    c = np.full_like(t, c_start, dtype=float)
    mask = (t >= t_start) & (t <= t_end)
    t_grad = t[mask]

    if len(t_grad) == 0:
        c[t > t_end] = c_end
        return c

    # Normalized position in gradient (0→1)
    frac = (t_grad - t_start) / max(t_end - t_start, 0.01)
    frac = np.clip(frac, 0.0, 1.0)

    dc = c_end - c_start

    if shape == "linear":
        c[mask] = c_start + dc * frac

    elif shape == "step":
        # Step at step_fraction: hold c_start, then jump
        c[mask] = np.where(frac < step_fraction, c_start, c_end)

    elif shape == "concave":
        # Slow start, rapid finish: c = c_start + dc * frac^2
        c[mask] = c_start + dc * frac ** 2

    elif shape == "convex":
        # Rapid start, slow finish: c = c_start + dc * (1 - (1-frac)^2)
        c[mask] = c_start + dc * (1.0 - (1.0 - frac) ** 2)

    else:
        # Default to linear
        c[mask] = c_start + dc * frac

    # After gradient ends
    c[t > t_end] = c_end
    return c


def effective_gradient_slope(
    c_start: float,
    c_end: float,
    gradient_time: float,
    shape: str = "linear",
    elution_fraction: float = 0.5,
) -> float:
    """
    Compute the effective local gradient slope at the elution point.

    For non-linear gradients, the local slope differs from the average slope.
    This affects peak width and resolution.

    Parameters
    ----------
    c_start           : Start salt (mM)
    c_end             : End salt (mM)
    gradient_time     : Total gradient duration (min)
    shape             : Gradient shape
    elution_fraction  : Normalized elution position (0=start, 1=end)

    Returns
    -------
    float : Local gradient slope at elution point (mM/min)
    """
    avg_slope = (c_end - c_start) / max(gradient_time, 0.1)

    if shape == "linear":
        return avg_slope
    elif shape == "concave":
        # dc/dt at fraction f: 2*f * dc/dt_total
        return avg_slope * 2.0 * max(elution_fraction, 0.05)
    elif shape == "convex":
        # dc/dt at fraction f: 2*(1-f) * dc/dt_total
        return avg_slope * 2.0 * max(1.0 - elution_fraction, 0.05)
    elif shape == "step":
        # Step gradient: effectively infinite slope at transition
        # For RT estimation, use a high but finite slope
        return avg_slope * 5.0
    else:
        return avg_slope


# ===========================================================================
# 1d. Scale-Up Transfer Function
# ===========================================================================
#
# Translates lab-scale chromatography results to manufacturing scale.
# Key physics: constant L/u (bed height / linear velocity) preserves
# residence time, but axial dispersion increases with column diameter.

@dataclass
class ScaleUpParams:
    """Column geometry for scale-up transfer."""
    diameter_cm: float = 1.6        # Column diameter (cm)
    bed_height_cm: float = 20.0     # Bed height (cm)
    flow_rate_ml_min: float = 2.0   # Volumetric flow rate (mL/min)
    particle_dp_um: float = 34.0    # Resin particle diameter (μm) — e.g. SP Sepharose HP


def scale_up_transfer(
    lab: ScaleUpParams,
    mfg: ScaleUpParams,
    lab_rt: float,
    lab_fwhm: float,
) -> Dict[str, float]:
    """
    Transfer chromatographic performance from lab to manufacturing scale.

    Uses van Deemter equation to estimate plate height change:
      H = A + B/u + C*u
    where A = eddy diffusion (∝ dp), B = longitudinal diffusion, C = mass transfer.

    Maintains constant residence time (L/u), adjusts for:
    - Plate count change from different H at production flow rate
    - Wall effect reduction at larger diameter (improves packing)
    - Axial dispersion broadening from larger column

    Parameters
    ----------
    lab       : Lab-scale column parameters
    mfg       : Manufacturing-scale column parameters
    lab_rt    : Lab-scale retention time (min)
    lab_fwhm  : Lab-scale peak FWHM (min)

    Returns
    -------
    dict with mfg_rt, mfg_fwhm, mfg_plates, plate_ratio, scale_factor
    """
    import math

    # Cross-sectional areas (cm²)
    A_lab = math.pi * (lab.diameter_cm / 2) ** 2
    A_mfg = math.pi * (mfg.diameter_cm / 2) ** 2

    # Linear velocities (cm/min)
    u_lab = lab.flow_rate_ml_min / A_lab
    u_mfg = mfg.flow_rate_ml_min / A_mfg

    # van Deemter parameters (generic CEX resin)
    dp_cm = lab.particle_dp_um * 1e-4
    A_vd = 2.0 * dp_cm                     # Eddy diffusion
    B_vd = 0.002                            # Longitudinal diffusion (cm²/min)
    C_vd = 0.05 * dp_cm ** 2 / 1e-4        # Mass transfer resistance

    # Plate heights
    H_lab = A_vd + B_vd / max(u_lab, 0.01) + C_vd * u_lab
    H_mfg = A_vd + B_vd / max(u_mfg, 0.01) + C_vd * u_mfg

    # Plate counts
    N_lab = lab.bed_height_cm / max(H_lab, 1e-4)
    N_mfg = mfg.bed_height_cm / max(H_mfg, 1e-4)

    plate_ratio = N_mfg / max(N_lab, 1.0)

    # Wall effect correction: smaller columns have better wall support → higher N
    # At manufacturing scale (D > 20 cm), wall effect is negligible
    if lab.diameter_cm < 5.0 and mfg.diameter_cm > 10.0:
        wall_correction = 0.95  # lab N was slightly inflated by wall effect
    else:
        wall_correction = 1.0

    # Axial dispersion: broader columns have more radial flow distribution issues
    # Correction based on Peclet number approximation
    dispersion_broadening = 1.0 + 0.01 * max(0, mfg.diameter_cm - lab.diameter_cm)

    # RT: same if L/u ratio is maintained
    residence_ratio = (mfg.bed_height_cm / max(u_mfg, 0.01)) / \
                      (lab.bed_height_cm / max(u_lab, 0.01))
    mfg_rt = lab_rt * residence_ratio

    # FWHM: scales with sqrt(N_ratio) * dispersion correction
    sigma_lab = lab_fwhm / 2.355   # Gaussian sigma
    sigma_mfg = sigma_lab / max(plate_ratio * wall_correction, 0.01) ** 0.5 * dispersion_broadening
    mfg_fwhm = sigma_mfg * 2.355

    return {
        "mfg_rt": round(mfg_rt, 2),
        "mfg_fwhm": round(mfg_fwhm, 3),
        "mfg_plates": round(N_mfg, 0),
        "lab_plates": round(N_lab, 0),
        "plate_ratio": round(plate_ratio, 3),
        "wall_correction": wall_correction,
        "dispersion_broadening": round(dispersion_broadening, 3),
        "residence_ratio": round(residence_ratio, 3),
        "scale_factor_diameter": round(mfg.diameter_cm / lab.diameter_cm, 1),
        "scale_factor_volume": round(A_mfg * mfg.bed_height_cm /
                                     (A_lab * lab.bed_height_cm), 1),
    }


# ===========================================================================
# 1e. pI-Adaptive pH Range
# ===========================================================================

def compute_adaptive_ph_range(
    pI: float,
    mode: str = "cex",
) -> Tuple[float, float]:
    """
    Compute DoE pH search range based on molecule pI and chromatography mode.

    For CEX (cation exchange):
      - Protein must be positively charged → pH < pI
      - Optimal binding: pH 1.0-3.0 below pI
      - Lower bound: max(4.0, pI - 4.0) to avoid acid denaturation
      - Upper bound: pI - 0.5 to ensure positive charge

    For AEX (anion exchange):
      - Protein must be negatively charged → pH > pI
      - Not commonly used for mAb polishing (pI typically 7-9)

    Parameters
    ----------
    pI   : Isoelectric point
    mode : "cex" (default) or "aex"

    Returns
    -------
    (ph_min, ph_max) tuple
    """
    if mode == "cex":
        ph_max = pI - 0.5   # Must be below pI for cation binding
        ph_min = max(4.0, pI - 4.0)  # Don't go below pH 4 (denaturation risk)

        # Ensure at least 0.5 pH unit range for meaningful DoE
        if ph_max - ph_min < 0.5:
            ph_min = max(3.5, ph_max - 1.0)

        # Safety clamps
        ph_min = max(3.5, ph_min)
        ph_max = min(9.0, ph_max)

        if ph_min >= ph_max:
            # Fallback for unusual pI values
            ph_min = 5.0
            ph_max = 6.5
            log.warning("pI=%.1f produced invalid CEX pH range, using fallback [5.0, 6.5]", pI)

    elif mode == "aex":
        ph_min = pI + 0.5
        ph_max = min(10.0, pI + 3.0)
        ph_min = max(6.0, ph_min)
    else:
        ph_min, ph_max = 5.5, 6.8  # legacy default

    return (round(ph_min, 1), round(ph_max, 1))


# ===========================================================================
# 2. SMA Parameter Estimation from pH
# ===========================================================================

def _sma_params_for_variant(
    pI: float,
    mw: float,
    variant: str,
    elution_ph: float,
    hydrophobicity: float = 0.35,
) -> Dict[str, float]:
    """
    Estimate SMA parameters (ka, nu, sigma) for a charge variant
    at a given elution buffer pH.

    The pH affects the protein net charge, which modulates ka
    (adsorption equilibrium constant) through electrostatic interactions.

    Parameters
    ----------
    pI              : Isoelectric point of the main species
    mw              : Molecular weight (kDa)
    variant         : One of "acidic", "main", "basic"
    elution_ph      : Elution buffer pH (affects charge state)
    hydrophobicity  : Surface hydrophobicity fraction (0-1)

    Returns
    -------
    dict with keys: ka, nu, sigma, kd, lambda_
    """
    # =================================================================
    # v7.3: Standard-range PropertyMapper-aligned variant differentiation
    #
    # Strategy: Compute MAIN variant parameters from pI/MW/hydro,
    # then apply the SAME offset scheme PropertyMapper uses:
    #   - nu offsets:  acidic -0.4, main 0.0, basic +0.3
    #   - ka factors:  acidic 0.75x, main 1.0x, basic 1.25x
    # With standard ka (1-5), Keq~0.003 → c_elution~200 mM → RT~17 min.
    # Wide variant deltas produce Rs > 1.2 without inflated ka.
    # =================================================================

    # Net charge at elution pH (for main variant)
    delta_pH = pI - elution_ph
    net_charge = 10.0 * np.tanh(0.5 * delta_pH)  # smooth approximation

    # --- MAIN variant base parameters ---
    # ka: Standard SMA range (1-5), PropertyMapper v7.3 aligned
    ka_base = 1.0 + 0.012 * mw   # ~2.78 for 148 kDa
    charge_factor = max(0.3, 1.0 + 0.04 * net_charge)
    hydro_factor = 1.0 + 0.3 * hydrophobicity
    ka_main = ka_base * charge_factor * hydro_factor

    # nu: characteristic charge, must match PropertyMapper range (2.5-4.0)
    nu_main = 2.5 + 0.004 * mw  # ~3.1 for 148 kDa

    # sigma: steric shielding
    sigma = 6.0 + 0.015 * mw  # ~8.2 for 148 kDa

    # --- Variant offsets (matching PropertyMapper deamidation/oxidation) ---
    # PropertyMapper v7.3: acidic ka = ka_main * 0.75, basic ka = ka_main * 1.25
    # nu offsets: acidic -0.4, basic +0.3 (from PropertyMapper variant scheme)
    variant_ka_factors = {"acidic": 0.75, "main": 1.0, "basic": 1.25}
    variant_nu_offsets  = {"acidic": -0.4, "main": 0.0, "basic": +0.3}

    ka = ka_main * variant_ka_factors.get(variant, 1.0)
    nu = nu_main + variant_nu_offsets.get(variant, 0.0)

    # Clamp to physically reasonable ranges (standard-range, v7.3)
    ka = float(np.clip(ka, 0.3, 8.0))
    nu = float(np.clip(nu, 2.0, 4.5))
    sigma = float(np.clip(sigma, 5.0, 20.0))

    return {
        "ka": ka,
        "nu": nu,
        "sigma": sigma,
        "kd": 1000.0,        # desorption (fast, typical)
        "lambda_": 1200.0,   # ionic capacity
    }


# ===========================================================================
# 3. Resolution & Yield Calculators
# ===========================================================================

def _calculate_fwhm(rt: float, gradient_slope: float, nu: float) -> float:
    """
    Estimate Full Width at Half Maximum (FWHM) for a chromatographic peak.

    v7.3.1 Physics Correction
    -------------------------
    In real gradient elution, peak width scales approximately as:
        FWHM ∝ t_R / √N_eff
    where N_eff is the effective plate count. Both retention time and
    peak width compress proportionally with steeper gradients, so
    resolution (Rs = ΔtR / avg_width) is relatively insensitive to
    gradient steepness — or decreases slightly at very steep gradients
    due to reduced plate count efficiency.

    Previous (v7.2): had gradient_factor = 20/slope which made peaks
    too narrow at steep gradients → artificial Rs increase with gradient.

    Current: FWHM scales with RT (which already encodes gradient via
    estimate_rt_from_sma). A mild dispersion factor from nu captures
    the stronger-binding → broader-peak effect.
    """
    # FWHM scales with retention time (natural gradient compression)
    # Typical column: N ~ 5000-10000 plates, FWHM ~ 4*sigma ~ tR/sqrt(N)*4
    # For tR=15 min, N=5000: FWHM ~ 15/70*4 ≈ 0.85 min
    base_fwhm = 0.15 + 0.05 * rt  # ~0.90 min at RT=15 min

    # Higher nu → stronger binding → more mass-transfer broadening
    nu_factor = 1.0 + 0.10 * max(0.0, nu - 2.5)

    # Mild gradient correction: very steep gradients reduce plate efficiency
    # (faster elution → less equilibration time → broader relative to RT)
    # But this is a WEAK effect, not the dominant 1/slope relationship
    steep_penalty = 1.0 + 0.005 * max(0.0, gradient_slope - 15.0)

    fwhm = base_fwhm * nu_factor * steep_penalty
    return float(np.clip(fwhm, 0.1, 5.0))


def _resolution(rt1: float, rt2: float, fwhm1: float, fwhm2: float) -> float:
    """USP Resolution: Rs = 2 * (tR2 - tR1) / (w1 + w2)."""
    w_sum = fwhm1 * 2.355 + fwhm2 * 2.355  # FWHM to base width (Gaussian)
    if w_sum < 1e-6:
        return 0.0
    return abs(rt2 - rt1) * 2.0 / w_sum


def estimate_coelution_percent(rs: float) -> float:
    """
    Estimate the percentage of variant co-eluting with the main peak
    based on chromatographic resolution (Rs).

    Uses Gaussian overlap integral approximation:
      Rs >= 1.5  →  overlap < 2% (baseline resolved)
      Rs = 1.0   →  overlap ≈ 16%
      Rs = 0.5   →  overlap ≈ 48%
      Rs = 0.0   →  overlap = 100% (complete co-elution)

    Parameters
    ----------
    rs : float
        USP resolution between two adjacent peaks.

    Returns
    -------
    float : Estimated coelution percentage (0-100).
    """
    if rs <= 0:
        return 100.0
    # erfc approximation using numpy
    x = rs * np.sqrt(2) / 2.0
    # erfc(x) ≈ 2 * normal_sf(x * sqrt(2))
    # Use: erfc(x) = 1 - erf(x), with erf from polynomial approx
    # For simplicity: overlap ≈ 100 * exp(-2 * rs^2) is a good fit
    overlap = 100.0 * np.exp(-2.0 * rs ** 2)
    return float(np.clip(overlap, 0.0, 100.0))


def _estimate_yield(
    rt_main: float,
    rt_acidic: float,
    rt_basic: float,
    fwhm_main: float,
    fwhm_acidic: float,
    fwhm_basic: float,
    resolution_min: float,
) -> float:
    """
    Estimate main-peak recovery (yield) from peak overlap and resolution.

    Higher resolution → better separation → higher yield.
    Very steep gradients can cause poor binding → reduced yield.
    """
    # Base yield from resolution (sigmoid-like)
    if resolution_min >= 2.0:
        rs_yield = 0.98
    elif resolution_min >= 1.5:
        rs_yield = 0.90 + 0.08 * (resolution_min - 1.5) / 0.5
    elif resolution_min >= 1.0:
        rs_yield = 0.75 + 0.15 * (resolution_min - 1.0) / 0.5
    elif resolution_min >= 0.5:
        rs_yield = 0.55 + 0.20 * (resolution_min - 0.5) / 0.5
    else:
        rs_yield = 0.30 + 0.25 * resolution_min

    # Penalize if main peak is very close to column dead time (poor binding)
    if rt_main < 2.0:
        rs_yield *= 0.5

    return float(np.clip(rs_yield, 0.10, 0.99))


# ===========================================================================
# 4. Single-Condition Simulation
# ===========================================================================

def _simulate_condition(
    pI: float,
    mw: float,
    elution_ph: float,
    gradient_slope: float,
    hydrophobicity: float = 0.35,
    gradient_time: float = 30.0,
    c_salt_start: float = 50.0,
    c_salt_end: float = 500.0,
    w_resolution: float = 0.5,
    w_yield: float = 0.3,
    w_purity: float = 0.2,
    c_load_mg_ml: float = 30.0,
    gradient_shape: str = "linear",
    purity_floor: float = 95.0,
    rs_floor: float = 1.5,
    hcp_feed_ppm: float = 500.0,
) -> DoEResult:
    """
    Simulate a single chromatographic condition and compute Rs + Yield + Purity.

    Uses Yamamoto SMA theory (same as ml_predictor.estimate_rt_from_sma)
    but varies pH and gradient steepness across the DoE grid.

    v2.0: Added salt_elute, c_load, gradient_shape, purity constraint, HCP model.
    """
    from src.ml_predictor import estimate_rt_from_sma

    results = {}
    for variant in ("acidic", "main", "basic"):
        params = _sma_params_for_variant(pI, mw, variant, elution_ph, hydrophobicity)

        # Adjust gradient time to match requested gradient slope
        effective_gradient_time = (c_salt_end - c_salt_start) / max(gradient_slope, 0.1)

        rt = estimate_rt_from_sma(
            ka=params["ka"],
            nu=params["nu"],
            kd=params["kd"],
            lambda_=params["lambda_"],
            c_salt_start=c_salt_start,
            c_salt_end=c_salt_end,
            gradient_time=effective_gradient_time,
        )

        # For non-linear gradients, adjust FWHM using local slope at elution point
        elution_frac = (rt - 0.5) / max(effective_gradient_time, 0.1)  # approximate
        elution_frac = float(np.clip(elution_frac, 0.05, 0.95))
        local_slope = effective_gradient_slope(
            c_salt_start, c_salt_end, effective_gradient_time,
            shape=gradient_shape, elution_fraction=elution_frac,
        )

        fwhm = _calculate_fwhm(rt, local_slope, params["nu"])
        results[variant] = {"rt": rt, "fwhm": fwhm, "nu": params["nu"]}

    # Ensure ordering: acidic < main < basic (by RT)
    rt_a = results["acidic"]["rt"]
    rt_m = results["main"]["rt"]
    rt_b = results["basic"]["rt"]
    fwhm_a = results["acidic"]["fwhm"]
    fwhm_m = results["main"]["fwhm"]
    fwhm_b = results["basic"]["fwhm"]

    # Resolution
    rs_am = _resolution(rt_a, rt_m, fwhm_a, fwhm_m)
    rs_mb = _resolution(rt_m, rt_b, fwhm_m, fwhm_b)
    rs_min = min(rs_am, rs_mb)

    # Yield
    yield_main = _estimate_yield(rt_m, rt_a, rt_b, fwhm_m, fwhm_a, fwhm_b, rs_min)

    # Loading correction: higher load → band broadening → reduced resolution
    # Literature: DBC typically 30-50 mg/mL for SP Sepharose
    if c_load_mg_ml > 35.0:
        load_broadening = 1.0 + 0.008 * (c_load_mg_ml - 35.0)
        # Effective FWHM increases, recalculate Rs
        rs_am_adj = rs_am / load_broadening
        rs_mb_adj = rs_mb / load_broadening
        rs_min = min(rs_am_adj, rs_mb_adj)
        # Yield also drops slightly at high load (fronting)
        yield_main *= max(0.80, 1.0 - 0.003 * (c_load_mg_ml - 35.0))

    # Pool purity estimation from mass balance
    mb = compute_mass_balance(rs_am, rs_mb)
    pool_purity = mb["pool_purity_pct"]

    # HCP clearance for this CEX step
    hcp_result = estimate_hcp_clearance(
        step_type="cex_polishing",
        hcp_in_ppm=hcp_feed_ppm,
        c_load_mg_ml=c_load_mg_ml,
        resolution=rs_min,
        ph=elution_ph,
        pI=pI,
    )
    hcp_ppm = hcp_result["hcp_out_ppm"]

    # Throughput penalty: discourage impractically long runs
    throughput_factor = 1.0
    if rt_m > 25.0:
        throughput_factor = max(0.7, 1.0 - 0.02 * (rt_m - 25.0))

    # Hard constraint check
    feasible = (pool_purity >= purity_floor) and (rs_min >= rs_floor)

    # Objective: multi-criteria with throughput consideration
    # Normalize purity to 0-1 scale (90-100% → 0-1)
    purity_score = max(0.0, min(1.0, (pool_purity - 90.0) / 10.0))

    if feasible:
        objective = (w_resolution * rs_min + w_yield * yield_main
                     + w_purity * purity_score) * throughput_factor
    else:
        # Penalize infeasible solutions but don't zero them
        # (allows optimizer to find direction toward feasible region)
        penalty = 0.5
        if pool_purity < purity_floor:
            penalty *= max(0.3, pool_purity / purity_floor)
        if rs_min < rs_floor:
            penalty *= max(0.3, rs_min / rs_floor)
        objective = (w_resolution * rs_min + w_yield * yield_main
                     + w_purity * purity_score) * throughput_factor * penalty

    return DoEResult(
        elution_ph=elution_ph,
        gradient_slope=gradient_slope,
        rt_acidic=round(rt_a, 3),
        rt_main=round(rt_m, 3),
        rt_basic=round(rt_b, 3),
        fwhm_acidic=round(fwhm_a, 3),
        fwhm_main=round(fwhm_m, 3),
        fwhm_basic=round(fwhm_b, 3),
        resolution_acidic_main=round(rs_am, 4),
        resolution_main_basic=round(rs_mb, 4),
        resolution_min=round(rs_min, 4),
        yield_main=round(yield_main, 4),
        objective=round(objective, 4),
        salt_elute=c_salt_end,
        c_load_mg_ml=c_load_mg_ml,
        pool_purity_pct=round(pool_purity, 1),
        hcp_ppm=round(hcp_ppm, 1),
        gradient_shape=gradient_shape,
        feasible=feasible,
    )


# ===========================================================================
# 4b. Mass Balance Summary (v7.3.1 — suggestion #2)
# ===========================================================================

def compute_mass_balance(
    rs_acidic_main: float,
    rs_main_basic: float,
    variant_distribution: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute mass balance for chromatographic separation.

    Ensures that variant fractions + main peak + losses sum to 100%.
    Uses Gaussian overlap model from resolution values.

    Parameters
    ----------
    rs_acidic_main      : Resolution between acidic and main peaks
    rs_main_basic       : Resolution between main and basic peaks
    variant_distribution: Optional {acidic_pct, main_pct, basic_pct} feed composition.
                         Default: 15% acidic, 70% main, 15% basic (typical mAb).

    Returns
    -------
    dict with mass balance breakdown
    """
    if variant_distribution is None:
        variant_distribution = {"acidic_pct": 15.0, "main_pct": 70.0, "basic_pct": 15.0}

    feed_acidic = variant_distribution.get("acidic_pct", 15.0)
    feed_main = variant_distribution.get("main_pct", 70.0)
    feed_basic = variant_distribution.get("basic_pct", 15.0)
    feed_total = feed_acidic + feed_main + feed_basic

    # Coelution: fraction of variant that overlaps with main peak
    acidic_coelution = estimate_coelution_percent(rs_acidic_main) / 100.0
    basic_coelution = estimate_coelution_percent(rs_main_basic) / 100.0

    # Main peak collection efficiency: real pools use ±2σ cut window
    # which captures ~95.4% of main peak Gaussian area.  At higher Rs the
    # cut can be tighter to exclude variant tails → slight yield loss.
    rs_avg = (rs_acidic_main + rs_main_basic) / 2.0
    if rs_avg >= 2.0:
        collection_efficiency = 0.95          # tight cut, baseline-resolved
    elif rs_avg >= 1.5:
        collection_efficiency = 0.92          # moderate cut
    elif rs_avg >= 1.0:
        collection_efficiency = 0.88          # wider cut to recover main
    else:
        collection_efficiency = 0.85          # broad cut, partial overlap

    main_recovered = feed_main * collection_efficiency
    acidic_in_pool = feed_acidic * acidic_coelution
    basic_in_pool = feed_basic * basic_coelution

    pool_total = main_recovered + acidic_in_pool + basic_in_pool
    pool_purity = (main_recovered / pool_total * 100.0) if pool_total > 0 else 0.0
    pool_yield = (main_recovered / feed_main * 100.0) if feed_main > 0 else 0.0

    # Waste fractions: uncollected main + separated variants
    main_loss = feed_main * (1.0 - collection_efficiency)
    acidic_waste = feed_acidic * (1.0 - acidic_coelution)
    basic_waste = feed_basic * (1.0 - basic_coelution)
    waste_total = acidic_waste + basic_waste + main_loss

    # Mass balance check
    balance_check = abs((pool_total + waste_total) - feed_total)
    balanced = balance_check < 0.1

    return {
        "feed_total_pct": round(feed_total, 2),
        "pool_total_pct": round(pool_total, 2),
        "pool_main_pct": round(main_recovered, 2),
        "pool_acidic_impurity_pct": round(acidic_in_pool, 2),
        "pool_basic_impurity_pct": round(basic_in_pool, 2),
        "pool_purity_pct": round(pool_purity, 1),
        "pool_yield_pct": round(pool_yield, 1),
        "waste_total_pct": round(waste_total, 2),
        "mass_balance_error": round(balance_check, 3),
        "mass_balanced": balanced,
    }


# ===========================================================================
# 5. Full DoE Grid Search
# ===========================================================================

def run_doe_optimization(
    pI: float,
    mw: float = 150.0,
    hydrophobicity: float = 0.35,
    ph_range: Optional[Tuple[float, float]] = None,
    gradient_range: Tuple[float, float] = (5.0, 25.0),
    salt_elute_range: Tuple[float, float] = (300.0, 600.0),
    load_range: Tuple[float, float] = (20.0, 50.0),
    ph_steps: int = 12,
    gradient_steps: int = 10,
    salt_steps: int = 4,
    load_steps: int = 4,
    w_resolution: float = 0.5,
    w_yield: float = 0.3,
    w_purity: float = 0.2,
    purity_floor: float = 95.0,
    rs_floor: float = 1.5,
    gradient_shape: str = "linear",
    hcp_feed_ppm: float = 500.0,
) -> DoEOptimization:
    """
    Run a full Design-of-Experiments grid search over elution pH,
    gradient steepness, salt elute, and loading density.

    v2.0: pI-adaptive pH, 4-dimensional grid, purity constraint.

    Parameters
    ----------
    pI                : Protein isoelectric point
    mw                : Molecular weight (kDa)
    hydrophobicity    : Surface hydrophobicity fraction
    ph_range          : (min_pH, max_pH) — if None, auto-derived from pI
    gradient_range    : (min_slope, max_slope) in mM/min
    salt_elute_range  : (min_salt, max_salt) in mM for elution endpoint
    load_range        : (min_load, max_load) in mg/mL resin
    ph_steps          : Number of pH grid points
    gradient_steps    : Number of gradient grid points
    salt_steps        : Number of salt elute grid points
    load_steps        : Number of loading grid points
    w_resolution      : Weight for resolution in objective
    w_yield           : Weight for yield in objective
    w_purity          : Weight for purity in objective
    purity_floor      : Hard constraint: minimum pool purity (%)
    rs_floor          : Hard constraint: minimum resolution
    gradient_shape    : "linear", "concave", "convex", "step"
    hcp_feed_ppm      : HCP in CEX feed (ppm, after Protein A)

    Returns
    -------
    DoEOptimization with contour matrices and optimal sweet spot
    """
    t0 = time.time()

    # pI-adaptive pH range
    ph_range_source = "default"
    if ph_range is None:
        ph_range = compute_adaptive_ph_range(pI, mode="cex")
        ph_range_source = "pI_adaptive"
        log.info("pI-adaptive pH range for pI=%.2f: [%.1f, %.1f]", pI, ph_range[0], ph_range[1])
    else:
        ph_range_source = "user"

    ph_values = np.linspace(ph_range[0], ph_range[1], ph_steps).tolist()
    gradient_values = np.linspace(gradient_range[0], gradient_range[1], gradient_steps).tolist()
    salt_values = np.linspace(salt_elute_range[0], salt_elute_range[1], salt_steps).tolist()
    load_values = np.linspace(load_range[0], load_range[1], load_steps).tolist()

    # Primary matrices: pH × gradient (at best salt/load found)
    resolution_matrix = np.zeros((ph_steps, gradient_steps))
    yield_matrix = np.zeros((ph_steps, gradient_steps))
    objective_matrix = np.zeros((ph_steps, gradient_steps))
    purity_matrix = np.zeros((ph_steps, gradient_steps))
    hcp_matrix = np.zeros((ph_steps, gradient_steps))
    all_results: List[DoEResult] = []

    best_result: Optional[DoEResult] = None
    best_objective = -np.inf

    # 4D grid search: pH × gradient × salt_elute × load
    # For contour matrices, record the best salt/load at each (pH, gradient) point
    for i, ph in enumerate(ph_values):
        for j, grad in enumerate(gradient_values):
            best_at_ij: Optional[DoEResult] = None
            best_obj_ij = -np.inf

            for salt in salt_values:
                for load in load_values:
                    result = _simulate_condition(
                        pI=pI,
                        mw=mw,
                        elution_ph=ph,
                        gradient_slope=grad,
                        hydrophobicity=hydrophobicity,
                        c_salt_end=salt,
                        c_load_mg_ml=load,
                        w_resolution=w_resolution,
                        w_yield=w_yield,
                        w_purity=w_purity,
                        gradient_shape=gradient_shape,
                        purity_floor=purity_floor,
                        rs_floor=rs_floor,
                        hcp_feed_ppm=hcp_feed_ppm,
                    )
                    all_results.append(result)

                    if result.objective > best_obj_ij:
                        best_obj_ij = result.objective
                        best_at_ij = result

                    if result.objective > best_objective:
                        best_objective = result.objective
                        best_result = result

            # Record best salt/load for this (pH, gradient) cell
            if best_at_ij is not None:
                resolution_matrix[i, j] = best_at_ij.resolution_min
                yield_matrix[i, j] = best_at_ij.yield_main
                objective_matrix[i, j] = best_at_ij.objective
                purity_matrix[i, j] = best_at_ij.pool_purity_pct
                hcp_matrix[i, j] = best_at_ij.hcp_ppm

    wall_time = time.time() - t0

    if best_result is None:
        # Fallback: this shouldn't happen but guard against it
        best_result = all_results[0] if all_results else _simulate_condition(
            pI=pI, mw=mw, elution_ph=ph_range[0], gradient_slope=gradient_range[0])

    log.info(
        "DoE v2.0 complete: %d conditions in %.2fs | Optimal: pH=%.2f, Grad=%.1f mM/min, "
        "Salt=%.0f mM, Load=%.0f mg/mL, Rs=%.3f, Yield=%.1f%%, Purity=%.1f%%, "
        "HCP=%.0f ppm, Obj=%.4f %s",
        len(all_results), wall_time,
        best_result.elution_ph, best_result.gradient_slope,
        best_result.salt_elute, best_result.c_load_mg_ml,
        best_result.resolution_min, best_result.yield_main * 100,
        best_result.pool_purity_pct, best_result.hcp_ppm,
        best_result.objective,
        "[FEASIBLE]" if best_result.feasible else "[INFEASIBLE]",
    )

    return DoEOptimization(
        ph_values=ph_values,
        gradient_values=gradient_values,
        resolution_matrix=resolution_matrix,
        yield_matrix=yield_matrix,
        objective_matrix=objective_matrix,
        all_results=all_results,
        optimal=best_result,
        optimal_ph=best_result.elution_ph,
        optimal_gradient=best_result.gradient_slope,
        wall_time_s=round(wall_time, 3),
        n_conditions=len(all_results),
        weights={"resolution": w_resolution, "yield": w_yield, "purity": w_purity},
        salt_elute_values=salt_values,
        load_values=load_values,
        purity_matrix=purity_matrix,
        hcp_matrix=hcp_matrix,
        ph_range_source=ph_range_source,
    )


# ===========================================================================
# 6. Convenience: Summary & Serialization
# ===========================================================================

def doe_to_dict(opt: DoEOptimization) -> Dict[str, Any]:
    """Serialize DoE result for JSON/session-state storage."""
    d = {
        "optimal_ph": opt.optimal_ph,
        "optimal_gradient": opt.optimal_gradient,
        "optimal_resolution": opt.optimal.resolution_min,
        "optimal_yield": opt.optimal.yield_main,
        "optimal_objective": opt.optimal.objective,
        "optimal_rt_acidic": opt.optimal.rt_acidic,
        "optimal_rt_main": opt.optimal.rt_main,
        "optimal_rt_basic": opt.optimal.rt_basic,
        "optimal_rs_am": opt.optimal.resolution_acidic_main,
        "optimal_rs_mb": opt.optimal.resolution_main_basic,
        "ph_values": opt.ph_values,
        "gradient_values": opt.gradient_values,
        "resolution_matrix": opt.resolution_matrix.tolist(),
        "yield_matrix": opt.yield_matrix.tolist(),
        "objective_matrix": opt.objective_matrix.tolist(),
        "wall_time_s": opt.wall_time_s,
        "n_conditions": opt.n_conditions,
        "weights": opt.weights,
        # v2.0 additions
        "optimal_salt_elute": opt.optimal.salt_elute,
        "optimal_c_load": opt.optimal.c_load_mg_ml,
        "optimal_purity": opt.optimal.pool_purity_pct,
        "optimal_hcp_ppm": opt.optimal.hcp_ppm,
        "optimal_gradient_shape": opt.optimal.gradient_shape,
        "optimal_feasible": opt.optimal.feasible,
        "salt_elute_values": opt.salt_elute_values,
        "load_values": opt.load_values,
        "ph_range_source": opt.ph_range_source,
    }
    if opt.purity_matrix is not None:
        d["purity_matrix"] = opt.purity_matrix.tolist()
    if opt.hcp_matrix is not None:
        d["hcp_matrix"] = opt.hcp_matrix.tolist()
    return d


def doe_summary(opt: DoEOptimization) -> str:
    """Human-readable summary of the DoE optimization."""
    o = opt.optimal
    lines = [
        f"In-Silico DoE Optimization Complete (v2.0)",
        f"  Grid: {len(opt.ph_values)} pH × {len(opt.gradient_values)} gradient"
        f" × {len(opt.salt_elute_values) or 1} salt × {len(opt.load_values) or 1} load"
        f" = {opt.n_conditions} conditions ({opt.wall_time_s:.2f}s)",
        f"  pH range source: {opt.ph_range_source}",
        f"",
        f"  Optimal Sweet Spot:",
        f"    Elution pH:       {opt.optimal_ph:.2f}",
        f"    Gradient slope:   {opt.optimal_gradient:.1f} mM/min",
        f"    Salt elute:       {o.salt_elute:.0f} mM",
        f"    Loading:          {o.c_load_mg_ml:.0f} mg/mL",
        f"    Gradient shape:   {o.gradient_shape}",
        f"    Resolution (Rs):  {o.resolution_min:.3f}",
        f"    Main-Peak Yield:  {o.yield_main * 100:.1f}%",
        f"    Pool Purity:      {o.pool_purity_pct:.1f}%",
        f"    Residual HCP:     {o.hcp_ppm:.0f} ppm",
        f"    Feasible:         {'Yes' if o.feasible else 'No'}",
        f"    Objective Score:  {o.objective:.4f}",
        f"",
        f"  Peak Retention Times at Optimal:",
        f"    Acidic: {o.rt_acidic:.2f} min",
        f"    Main:   {o.rt_main:.2f} min",
        f"    Basic:  {o.rt_basic:.2f} min",
    ]
    return "\n".join(lines)


# ===========================================================================
# Multi-Step Purification Platform Process
# ===========================================================================

@dataclass
class PurificationStep:
    """Result from a single purification step."""
    step_name: str
    step_type: str          # "capture", "polishing", "flowthrough", "filtration"
    yield_pct: float        # step yield (%)
    purity_pct: float       # step purity (%)
    hcp_ppm: float          # residual HCP (ppm)
    log_clearance: float    # virus log clearance for this step
    notes: str = ""

    # Chromatogram data (CV-based)
    chrom_cv: Optional[np.ndarray] = None      # column volumes
    chrom_uv280: Optional[np.ndarray] = None   # UV280 signal (mAU)
    chrom_salt_or_ph: Optional[np.ndarray] = None  # overlay (conductivity or pH)
    chrom_label: str = ""                       # "salt (mS/cm)" or "pH"


@dataclass
class MultiStepResult:
    """Full multi-step purification result."""
    steps: List[PurificationStep]
    cumulative_yield: float     # overall yield (%)
    final_purity: float         # final purity (%)
    final_hcp_ppm: float        # final HCP (ppm)
    total_virus_clearance: float  # total log reduction
    summary_table: List[Dict[str, Any]]


def run_multistep_purification(
    pI: float = 8.5,
    mw_kda: float = 148.0,
    agg_risk: float = 0.05,
    doe_result: Optional[DoEResult] = None,
) -> MultiStepResult:
    """
    Simulate a 4-step monoclonal antibody purification platform process.

    Steps:
      1. Protein A Capture — affinity capture, low pH elution
      2. CEX Polishing — charge variant separation (uses DoE if available)
      3. AEX Flowthrough — DNA/endotoxin/HCP removal
      4. Viral Filtration — 20nm nanofiltration

    Parameters
    ----------
    pI : float
        Isoelectric point of the antibody.
    mw_kda : float
        Molecular weight in kDa.
    agg_risk : float
        Aggregation risk (0-1), affects Protein A yield.
    doe_result : DoEResult, optional
        CEX DoE result for polishing step parameters.

    Returns
    -------
    MultiStepResult with per-step metrics and chromatogram data.
    """
    steps = []

    # ---- Step 1: Protein A Capture ----
    proa_yield = 97.0 - 8.0 * min(agg_risk, 0.5)  # aggregation reduces yield
    proa_purity = 95.0 + 2.0 * (1.0 - agg_risk)
    # HCP clearance: literature-calibrated model
    _proa_hcp_result = estimate_hcp_clearance(
        step_type="capture",
        hcp_in_ppm=8000.0,  # typical HCCF HCP (literature baseline)
        c_load_mg_ml=35.0,
    )
    proa_hcp = _proa_hcp_result["hcp_out_ppm"]

    # ProA chromatogram: load → wash → elution at low pH
    cv = np.linspace(0, 25, 500)
    uv_proa = np.zeros_like(cv)
    ph_proa = np.ones_like(cv) * 7.0
    # Loading phase (0-5 CV): UV rises as flowthrough
    uv_proa[(cv >= 0) & (cv < 5)] = 20 * (1 - np.exp(-cv[(cv >= 0) & (cv < 5)]))
    # Wash (5-10 CV): low UV
    uv_proa[(cv >= 5) & (cv < 10)] = 5.0
    # Elution (10-15 CV): sharp peak at low pH
    _elution_mask = (cv >= 10) & (cv < 15)
    _elution_cv = cv[_elution_mask]
    uv_proa[_elution_mask] = 800 * np.exp(-0.5 * ((_elution_cv - 12.0) / 0.8) ** 2)
    # pH drops during elution
    ph_proa[(cv >= 9.5) & (cv < 15)] = 3.5
    ph_proa[(cv >= 15)] = 7.0
    # Strip (15-20 CV)
    uv_proa[(cv >= 15) & (cv < 20)] = 30 * np.exp(-0.3 * (cv[(cv >= 15) & (cv < 20)] - 15))
    # Re-equilibration
    ph_proa[(cv >= 15)] = np.linspace(3.5, 7.0, int(np.sum(cv >= 15)))[:int(np.sum(cv >= 15))]

    steps.append(PurificationStep(
        step_name="Protein A Capture",
        step_type="capture",
        yield_pct=round(proa_yield, 1),
        purity_pct=round(proa_purity, 1),
        hcp_ppm=round(proa_hcp, 0),
        log_clearance=0.0,
        notes=f"MabSelect SuRe, pH 3.5 elution, dynamic binding capacity ~35 g/L",
        chrom_cv=cv,
        chrom_uv280=uv_proa,
        chrom_salt_or_ph=ph_proa,
        chrom_label="pH",
    ))

    # ---- Step 2: CEX Polishing ----
    if doe_result is not None:
        cex_yield = doe_result.yield_main * 100
        cex_rs = doe_result.resolution_min
    else:
        cex_yield = 90.0
        cex_rs = 1.2
    cex_purity = min(99.5, proa_purity + 3.0 + cex_rs * 1.5)
    # HCP clearance: literature-calibrated model
    _cex_hcp_result = estimate_hcp_clearance(
        step_type="cex_polishing",
        hcp_in_ppm=proa_hcp,
        c_load_mg_ml=30.0,
        resolution=cex_rs,
        ph=6.0,
        pI=pI,
    )
    cex_hcp = _cex_hcp_result["hcp_out_ppm"]
    cex_coelution = estimate_coelution_percent(cex_rs)

    # CEX chromatogram: load → wash → gradient elution
    cv2 = np.linspace(0, 30, 600)
    uv_cex = np.zeros_like(cv2)
    salt_cex = np.ones_like(cv2) * 50.0  # mM
    # Loading (0-5 CV)
    uv_cex[(cv2 >= 0) & (cv2 < 5)] = 15 * (1 - np.exp(-cv2[(cv2 >= 0) & (cv2 < 5)]))
    # Wash (5-8 CV)
    uv_cex[(cv2 >= 5) & (cv2 < 8)] = 3.0
    # Gradient elution (8-25 CV)
    _grad_mask = (cv2 >= 8) & (cv2 < 25)
    _grad_cv = cv2[_grad_mask]
    salt_cex[_grad_mask] = 50.0 + (_grad_cv - 8.0) * 26.5  # 50→500 mM
    salt_cex[cv2 >= 25] = 500.0
    # Three peaks: acidic, main, basic
    uv_cex[_grad_mask] += 120 * np.exp(-0.5 * ((_grad_cv - 13.0) / 1.0) ** 2)  # acidic
    uv_cex[_grad_mask] += 600 * np.exp(-0.5 * ((_grad_cv - 16.0) / 1.2) ** 2)  # main
    uv_cex[_grad_mask] += 80 * np.exp(-0.5 * ((_grad_cv - 19.5) / 1.0) ** 2)   # basic

    steps.append(PurificationStep(
        step_name="CEX Polishing",
        step_type="polishing",
        yield_pct=round(cex_yield, 1),
        purity_pct=round(cex_purity, 1),
        hcp_ppm=round(cex_hcp, 0),
        log_clearance=0.0,
        notes=f"Rs={cex_rs:.2f}, coelution={cex_coelution:.1f}%, charge variant separation",
        chrom_cv=cv2,
        chrom_uv280=uv_cex,
        chrom_salt_or_ph=salt_cex,
        chrom_label="salt (mM NaCl)",
    ))

    # ---- Step 3: AEX Flowthrough ----
    aex_yield = 96.0
    aex_purity = min(99.8, cex_purity + 0.3)
    # HCP clearance: literature-calibrated model
    _aex_hcp_result = estimate_hcp_clearance(
        step_type="aex_flowthrough",
        hcp_in_ppm=cex_hcp,
    )
    aex_hcp = _aex_hcp_result["hcp_out_ppm"]

    # AEX chromatogram: flowthrough mode
    cv3 = np.linspace(0, 15, 300)
    uv_aex = np.zeros_like(cv3)
    salt_aex = np.ones_like(cv3) * 50.0
    # Product passes through (2-6 CV)
    _ft_mask = (cv3 >= 2) & (cv3 < 6)
    _ft_cv = cv3[_ft_mask]
    uv_aex[_ft_mask] = 500 * np.exp(-0.5 * ((_ft_cv - 4.0) / 1.0) ** 2)
    # Strip bound impurities (8-12 CV) with high salt
    _strip_mask = (cv3 >= 8) & (cv3 < 12)
    _strip_cv = cv3[_strip_mask]
    uv_aex[_strip_mask] = 50 * np.exp(-0.5 * ((_strip_cv - 10.0) / 0.8) ** 2)
    salt_aex[cv3 >= 7.5] = 1000.0

    steps.append(PurificationStep(
        step_name="AEX Flowthrough",
        step_type="flowthrough",
        yield_pct=round(aex_yield, 1),
        purity_pct=round(aex_purity, 1),
        hcp_ppm=round(aex_hcp, 0),
        log_clearance=1.5,
        notes="Q Sepharose FF, flowthrough mode — binds DNA, endotoxin, acidic HCP",
        chrom_cv=cv3,
        chrom_uv280=uv_aex,
        chrom_salt_or_ph=salt_aex,
        chrom_label="salt (mM NaCl)",
    ))

    # ---- Step 4: Viral Filtration ----
    vf_yield = 97.0
    vf_purity = aex_purity  # no change
    vf_hcp = aex_hcp        # no change

    steps.append(PurificationStep(
        step_name="Viral Filtration (20nm)",
        step_type="filtration",
        yield_pct=round(vf_yield, 1),
        purity_pct=round(vf_purity, 1),
        hcp_ppm=round(vf_hcp, 0),
        log_clearance=4.0,
        notes="Planova 20N, ≥4 log retrovirus clearance, 50 L/m²/h flux",
    ))

    # ---- Cumulative calculations ----
    cum_yield = 100.0
    for s in steps:
        cum_yield *= s.yield_pct / 100.0
    total_lrv = sum(s.log_clearance for s in steps)

    summary_table = []
    _running_yield = 100.0
    for s in steps:
        _running_yield *= s.yield_pct / 100.0
        summary_table.append({
            "Step": s.step_name,
            "Step Yield (%)": s.yield_pct,
            "Cumulative Yield (%)": round(_running_yield, 1),
            "Purity (%)": s.purity_pct,
            "HCP (ppm)": s.hcp_ppm,
            "Virus LRV": s.log_clearance,
            "Notes": s.notes,
        })

    return MultiStepResult(
        steps=steps,
        cumulative_yield=round(cum_yield, 1),
        final_purity=steps[-1].purity_pct,
        final_hcp_ppm=steps[-1].hcp_ppm,
        total_virus_clearance=round(total_lrv, 1),
        summary_table=summary_table,
    )


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    passed = 0
    total = 10

    # Test 1: Single condition (v2.0 with purity + HCP)
    r = _simulate_condition(pI=8.5, mw=150.0, elution_ph=6.0, gradient_slope=15.0)
    assert r.resolution_min >= 0, "Resolution must be non-negative"
    assert 0 < r.yield_main <= 1.0, "Yield must be in (0, 1]"
    assert r.pool_purity_pct > 0, "Purity must be computed"
    assert r.hcp_ppm >= 0, "HCP must be non-negative"
    print(f"Test 1 PASS: Single condition — Rs={r.resolution_min:.3f}, "
          f"Yield={r.yield_main:.1%}, Purity={r.pool_purity_pct:.1f}%, HCP={r.hcp_ppm:.0f}ppm")
    passed += 1

    # Test 2: pI-adaptive pH range
    ph_range = compute_adaptive_ph_range(8.5, mode="cex")
    assert ph_range[1] < 8.5, f"pH max must be < pI, got {ph_range[1]}"
    assert ph_range[0] >= 4.0, f"pH min must be >= 4.0, got {ph_range[0]}"
    ph_low = compute_adaptive_ph_range(5.5, mode="cex")
    assert ph_low[0] >= 3.5, "Low-pI molecule: pH min must be >= 3.5"
    print(f"Test 2 PASS: pI-adaptive pH — pI=8.5→{ph_range}, pI=5.5→{ph_low}")
    passed += 1

    # Test 3: HCP clearance model
    hcp_capture = estimate_hcp_clearance("capture", 8000.0, c_load_mg_ml=35.0)
    assert hcp_capture["hcp_out_ppm"] < 8000.0, "Capture must reduce HCP"
    assert hcp_capture["log_reduction"] > 1.0, "Capture LRV must be > 1.0"
    hcp_cex = estimate_hcp_clearance("cex_polishing", 400.0, resolution=1.8)
    assert hcp_cex["hcp_out_ppm"] < 400.0, "CEX must reduce HCP"
    print(f"Test 3 PASS: HCP model — Capture: {hcp_capture['hcp_out_ppm']:.0f}ppm, "
          f"CEX: {hcp_cex['hcp_out_ppm']:.0f}ppm")
    passed += 1

    # Test 4: Gradient shapes
    t = np.linspace(0, 30, 100)
    for shape in ("linear", "step", "concave", "convex"):
        c = gradient_salt_profile(t, 5.0, 25.0, 50.0, 500.0, shape=shape)
        assert c[0] == 50.0, f"{shape}: start must be 50 mM"
        assert c[-1] == 500.0, f"{shape}: end must be 500 mM"
    print("Test 4 PASS: Gradient shapes (linear, step, concave, convex)")
    passed += 1

    # Test 5: Scale-up transfer
    lab = ScaleUpParams(diameter_cm=1.6, bed_height_cm=20.0, flow_rate_ml_min=2.0)
    mfg = ScaleUpParams(diameter_cm=30.0, bed_height_cm=20.0, flow_rate_ml_min=700.0)
    su = scale_up_transfer(lab, mfg, lab_rt=15.0, lab_fwhm=1.0)
    assert su["mfg_rt"] > 0, "Mfg RT must be positive"
    assert su["mfg_fwhm"] > 0, "Mfg FWHM must be positive"
    assert su["scale_factor_volume"] > 100, "Volume scale factor should be large"
    print(f"Test 5 PASS: Scale-up — RT {15.0:.1f}→{su['mfg_rt']:.1f}min, "
          f"FWHM {1.0:.2f}→{su['mfg_fwhm']:.2f}min, Vol×{su['scale_factor_volume']:.0f}")
    passed += 1

    # Test 6: Full DoE v2.0 (4D grid with pI-adaptive pH)
    opt = run_doe_optimization(
        pI=8.5, mw=150.0,
        ph_steps=8, gradient_steps=6, salt_steps=3, load_steps=3,
    )
    expected_total = 8 * 6 * 3 * 3  # 432
    assert opt.n_conditions == expected_total, f"Expected {expected_total}, got {opt.n_conditions}"
    assert opt.resolution_matrix.shape == (8, 6), "Matrix shape mismatch"
    assert opt.purity_matrix is not None, "Purity matrix should exist"
    print(f"Test 6 PASS: Full DoE v2.0 — {opt.n_conditions} conditions in {opt.wall_time_s:.2f}s")
    passed += 1

    # Test 7: Optimal is meaningful and has new fields
    assert opt.optimal.resolution_min > 0, "Optimal resolution should be positive"
    assert opt.optimal.yield_main > 0, "Optimal yield should be positive"
    assert opt.optimal.pool_purity_pct > 0, "Purity must be computed"
    assert opt.ph_range_source == "pI_adaptive", "pH range should be pI-adaptive"
    print(f"Test 7 PASS: Optimal — pH={opt.optimal_ph:.2f}, Salt={opt.optimal.salt_elute:.0f}mM, "
          f"Load={opt.optimal.c_load_mg_ml:.0f}mg/mL, Purity={opt.optimal.pool_purity_pct:.1f}%")
    passed += 1

    # Test 8: Serialization with v2.0 fields
    d = doe_to_dict(opt)
    assert "optimal_ph" in d and "resolution_matrix" in d
    assert "optimal_salt_elute" in d and "optimal_purity" in d
    assert "purity_matrix" in d and "hcp_matrix" in d
    s = doe_summary(opt)
    assert "Sweet Spot" in s and "Purity" in s
    print("Test 8 PASS: Serialization with v2.0 fields OK")
    passed += 1

    # Test 9: Multi-step purification uses HCP model
    ms = run_multistep_purification(pI=8.5, mw_kda=148.0, agg_risk=0.05)
    assert ms.final_hcp_ppm < 100, f"Final HCP should be <100 ppm, got {ms.final_hcp_ppm}"
    assert ms.cumulative_yield > 50, "Cumulative yield should be >50%"
    print(f"Test 9 PASS: Multi-step — Yield={ms.cumulative_yield:.1f}%, "
          f"HCP={ms.final_hcp_ppm:.0f}ppm, LRV={ms.total_virus_clearance:.1f}")
    passed += 1

    # Test 10: Mass balance
    mb = compute_mass_balance(1.5, 1.8)
    assert mb["mass_balanced"], "Mass balance should close"
    assert mb["pool_purity_pct"] > 90.0, "Purity should be >90%"
    print(f"Test 10 PASS: Mass balance — Purity={mb['pool_purity_pct']:.1f}%, "
          f"Yield={mb['pool_yield_pct']:.1f}%")
    passed += 1

    print(f"\n{'='*50}")
    print(f"purification_optimizer v2.0 self-test: {passed}/{total} passed")
    print(doe_summary(opt))
    sys.exit(0 if passed == total else 1)
