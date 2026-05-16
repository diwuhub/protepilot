"""
stability_twin.py — Milestone 27
=====================================================================
Time-Series Stability & Degradation Kinetics Engine

Implements Arrhenius-based kinetic degradation models to project
the growth of aggregation (SEC HMW), sub-visible particles (SVP),
charge variant drift, and potency loss over time.

Provides:
    1. Aggregation kinetics: SEC HMW% growth via nucleation-growth model
    2. Sub-visible particle (SVP) projection over time
    3. Charge variant drift: cIEF acidic peak growth from deamidation
    4. Potency decay: binding activity loss over shelf-life
    5. Dual-condition simulation: 5C/24-month + 40C/3-month accelerated

Physics
------------------------------------------------------------
  - Arrhenius: k(T) = A * exp(-Ea / RT)
  - Aggregation: dC_agg/dt = k_agg * (C_monomer)^n (nucleation-growth)
  - Deamidation: first-order kinetics, t1/2 ~ 30-300 days depending on motif
  - Particle formation: secondary nucleation from aggregation intermediates
  - Excipient stabilization: Sucrose/Trehalose reduce k_agg by 30-60%
  - Polysorbate 80: reduces surface-induced aggregation

References
------------------------------------------------------------
  Roberts (2007) Biotechnol Bioeng 98:927 — Protein aggregation kinetics
  Hawe et al. (2009) J Pharm Sci 98:2509 — Sub-visible particles
  Wakankar & Borchardt (2006) J Pharm Sci 95:2321 — Deamidation kinetics
  ICH Q5C: Stability Testing of Biotechnological/Biological Products

Version : 1.0 (Analytical QC + Stability + Pareto — M27)
Author  : Di (ProtePilot)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# =========================================================================
# Constants
# =========================================================================

R_GAS = 8.314e-3    # kJ/(mol·K)
T_REF = 278.15       # 5°C in Kelvin (reference storage)
T_ACCEL = 313.15     # 40°C in Kelvin (accelerated)

# Activation energies (kJ/mol) — literature ranges for mAb degradation
EA_AGGREGATION = 85.0     # Aggregation (Roberts 2007): 60-120 kJ/mol
EA_DEAMIDATION = 90.0     # Deamidation (Wakankar 2006): 80-100 kJ/mol
EA_OXIDATION = 50.0       # Oxidation: 40-60 kJ/mol
EA_FRAGMENTATION = 100.0  # Backbone cleavage: 90-120 kJ/mol
EA_PARTICLE = 75.0        # Particle formation: 60-90 kJ/mol


# =========================================================================
# Data Classes
# =========================================================================

@dataclass
class StabilityTimepoint:
    """Single timepoint in stability trajectory."""
    month: float
    temperature_c: float
    sec_hmw_pct: float          # SEC high molecular weight %
    sec_monomer_pct: float      # SEC monomer %
    sec_lmw_pct: float          # SEC low molecular weight %
    svp_per_ml: float           # Sub-visible particles >=10um per mL
    cief_acidic_pct: float      # cIEF acidic variant %
    potency_pct: float          # Relative potency (% of T0)


@dataclass
class StabilityResult:
    """Full stability simulation result."""
    condition_name: str         # e.g., "5C Long-term" or "40C Accelerated"
    temperature_c: float
    duration_months: int
    timepoints: List[StabilityTimepoint]
    starting_hmw_pct: float
    final_hmw_pct: float
    hmw_growth_rate_pct_per_month: float
    final_svp_per_ml: float
    final_acidic_pct: float
    final_potency_pct: float
    shelf_life_months: Optional[float]  # months to reach 5% HMW (spec limit)
    passes_24month_spec: bool
    summary: str


@dataclass
class DualConditionResult:
    """Combined 5C + 40C stability results."""
    long_term: StabilityResult      # 5°C / 24 months
    accelerated: StabilityResult    # 40°C / 3 months
    predicted_shelf_life_months: float
    overall_stability_grade: str    # "Excellent", "Good", "At Risk", "Poor"
    summary: str


# =========================================================================
# 1. Arrhenius Rate Calculator
# =========================================================================

def arrhenius_rate(
    k_ref: float,
    ea_kj_mol: float,
    T_ref_K: float,
    T_target_K: float,
) -> float:
    """
    Calculate rate constant at target temperature via Arrhenius equation.

    k(T) = k_ref * exp[ -(Ea/R) * (1/T_target - 1/T_ref) ]

    Parameters
    ----------
    k_ref       : Rate constant at reference temperature
    ea_kj_mol   : Activation energy (kJ/mol)
    T_ref_K     : Reference temperature (K)
    T_target_K  : Target temperature (K)

    Returns
    -------
    Rate constant at target temperature
    """
    exponent = -(ea_kj_mol / R_GAS) * (1.0 / T_target_K - 1.0 / T_ref_K)
    return k_ref * math.exp(exponent)


# =========================================================================
# 2. Excipient Stabilization Modifiers
# =========================================================================

EXCIPIENT_EFFECTS = {
    "sucrose": {
        "agg_rate_multiplier": 0.70,    # reduces aggregation rate by 30%
        "deam_rate_multiplier": 0.90,   # minimal effect on deamidation
        "particle_rate_multiplier": 0.70,
        "mechanism": "Preferential exclusion stabilizes native fold",
    },
    "trehalose": {
        "agg_rate_multiplier": 0.50,    # reduces aggregation rate by 50%
        "deam_rate_multiplier": 0.88,
        "particle_rate_multiplier": 0.65,
        "mechanism": "Vitrification + preferential exclusion",
    },
    "ps80": {
        "agg_rate_multiplier": 0.80,    # reduces surface-induced aggregation by 20%
        "deam_rate_multiplier": 1.00,   # no effect on chemical degradation
        "particle_rate_multiplier": 0.40,  # strong particle reduction
        "mechanism": "Surfactant prevents adsorption-induced aggregation",
    },
    "arginine": {
        "agg_rate_multiplier": 0.65,
        "deam_rate_multiplier": 0.95,
        "particle_rate_multiplier": 0.80,
        "mechanism": "Suppresses protein-protein interactions",
    },
}


def _get_excipient_multiplier(
    excipients: List[str], rate_key: str,
) -> float:
    """Get combined excipient rate multiplier (multiplicative)."""
    multiplier = 1.0
    for exc in excipients:
        exc_lower = exc.lower().replace(" ", "").replace("-", "")
        for name, effects in EXCIPIENT_EFFECTS.items():
            if name in exc_lower:
                multiplier *= effects.get(rate_key, 1.0)
                break
    return multiplier


# =========================================================================
# 3. pH-Dependent Rate Modifiers
# =========================================================================

def _ph_aggregation_modifier(pH: float, pI: float) -> float:
    """
    pH-dependent aggregation rate modifier.
    Aggregation is fastest near pI (minimal electrostatic repulsion).
    """
    distance = abs(pH - pI)
    if distance < 0.5:
        return 2.5  # very close to pI → high aggregation risk
    elif distance < 1.0:
        return 1.5
    elif distance < 2.0:
        return 1.0
    else:
        return 0.7  # far from pI → good colloidal stability


def _ph_deamidation_modifier(pH: float) -> float:
    """
    pH-dependent deamidation rate modifier.
    Deamidation is faster at higher pH (base-catalyzed).
    """
    if pH < 5.0:
        return 0.3
    elif pH < 6.0:
        return 0.6
    elif pH < 7.0:
        return 1.0
    elif pH < 8.0:
        return 1.8
    else:
        return 3.0


# =========================================================================
# 4. Core Stability Simulation
# =========================================================================

def simulate_stability(
    starting_hmw_pct: float = 1.0,
    starting_acidic_pct: float = 15.0,
    formulation_ph: float = 6.0,
    pI: float = 8.5,
    excipients: Optional[List[str]] = None,
    temperature_c: float = 5.0,
    duration_months: int = 24,
    deamidation_sites: int = 5,
    dp_clip_sites: int = 1,
    hydrophobicity: float = 0.35,
    timepoint_interval_months: float = 1.0,
    condition_name: str = "",
    Tm: Optional[float] = None,
) -> StabilityResult:
    """
    Run kinetic degradation simulation over time.

    Models:
      - SEC HMW: nucleation-growth aggregation kinetics
      - SEC LMW: fragmentation kinetics
      - SVP: secondary nucleation from aggregation intermediates
      - cIEF acidic: deamidation-driven charge variant growth
      - Potency: composite decay from all degradation pathways

    Parameters
    ----------
    starting_hmw_pct     : Initial SEC HMW % (realistic range: 0.1-10.0)
    starting_acidic_pct  : Initial cIEF acidic %
    formulation_ph       : Buffer pH
    pI                   : Protein isoelectric point
    excipients           : List of excipient names
    temperature_c        : Storage temperature (°C)
    duration_months      : Simulation duration (months)
    deamidation_sites    : Number of deamidation hotspots
    dp_clip_sites        : Number of Asp-Pro clipping sites
    hydrophobicity       : Hydrophobicity score (0-1)
    timepoint_interval_months : Interval between timepoints
    condition_name       : Label for this condition
    Tm                   : Melting temperature (°C). Higher Tm → slower aggregation.
                           Typical mAb range: 60-85°C. Distinct from starting_hmw_pct.
    """
    if excipients is None:
        excipients = []

    # ── Input validation: prevent Tm/HMW cross-contamination ──
    if starting_hmw_pct > 20.0:
        log.warning(
            "starting_hmw_pct=%.1f is unrealistically high (>20%%). "
            "This may indicate a Tm value was passed as HMW. "
            "Clamping to 5.0%% and using the value as Tm instead.",
            starting_hmw_pct,
        )
        # Auto-recover: treat the erroneous value as Tm
        if Tm is None:
            Tm = starting_hmw_pct
        starting_hmw_pct = 5.0  # reasonable worst-case default

    if starting_hmw_pct < 0.0:
        raise ValueError(f"starting_hmw_pct={starting_hmw_pct} must be >= 0")

    T_K = temperature_c + 273.15

    # --- Base rate constants at 5°C (per month) ---
    # v7.4.1: Use multiplicative modifiers for both hydrophobicity and Tm
    # so that Tm dominates the correlation (fixes k_40c direction).
    # Previously, additive hydro formula created variance that overwhelmed Tm.
    #
    # Base rate: 0.08 %/month at 5°C (typical mAb, calibrated so that
    # a well-formulated mAb with sucrose+PS80 at 5°C → ~24-30 month shelf life)
    k_agg_ref = 0.08

    # Hydrophobicity modifier: multiplicative, mild effect (1.0 ± 0.3)
    # GRAVY range for mAbs: roughly -0.7 to +0.1. Reference: 0.0 (neutral)
    _hydro_factor = 1.0 + 0.4 * hydrophobicity  # hydro=-0.5 → 0.8; hydro=0 → 1.0; hydro=0.5 → 1.2
    _hydro_factor = max(0.5, min(2.0, _hydro_factor))
    k_agg_ref *= _hydro_factor

    # Tm-dependent modifier: stronger effect (dominates rate variation)
    # v7.4.1: Changed from 10°C to 5°C half-life to ensure Tm dominates
    # the k_40c correlation over hydrophobicity (empirical tuning against
    # PROPHET-Ab 246-antibody dataset: k_40c vs Tm1 rho=-0.624).
    # Literature range: 7-10°C (Roberts 2007, Biotechnol Bioeng 98:927).
    # Using 5°C is ~1.5x steeper than literature consensus but necessary
    # because the hydrophobicity term (GRAVY) positively correlates with
    # Tm1 in the PROPHET-Ab dataset (rho=+0.19), which inverts the k_40c
    # direction at the 10°C half-life setting. The 5°C half-life ensures
    # Tm variance (std=0.87) dominates hydro variance (std=0.16) by 5:1.
    # Reference Tm = 70°C; each 5°C deviation doubles/halves rate.
    if Tm is not None and Tm > 0:
        _tm_factor = 2.0 ** ((70.0 - Tm) / 5.0)
        _tm_factor = max(0.05, min(20.0, _tm_factor))
        k_agg_ref *= _tm_factor
    k_agg = arrhenius_rate(k_agg_ref, EA_AGGREGATION, T_REF, T_K)

    # Deamidation: ~0.1-0.5 %/month acidic growth at 5°C
    k_deam_ref = 0.05 + 0.03 * deamidation_sites
    k_deam = arrhenius_rate(k_deam_ref, EA_DEAMIDATION, T_REF, T_K)

    # Fragmentation: ~0.01-0.05 %/month at 5°C
    k_frag_ref = 0.01 + 0.015 * dp_clip_sites
    k_frag = arrhenius_rate(k_frag_ref, EA_FRAGMENTATION, T_REF, T_K)

    # Particle formation: linked to aggregation
    k_svp_ref = 50.0 + 100.0 * hydrophobicity  # particles/mL/month
    k_svp = arrhenius_rate(k_svp_ref, EA_PARTICLE, T_REF, T_K)

    # --- Apply modifiers ---
    ph_agg_mod = _ph_aggregation_modifier(formulation_ph, pI)
    ph_deam_mod = _ph_deamidation_modifier(formulation_ph)

    exc_agg_mod = _get_excipient_multiplier(excipients, "agg_rate_multiplier")
    exc_deam_mod = _get_excipient_multiplier(excipients, "deam_rate_multiplier")
    exc_svp_mod = _get_excipient_multiplier(excipients, "particle_rate_multiplier")

    k_agg *= ph_agg_mod * exc_agg_mod
    k_deam *= ph_deam_mod * exc_deam_mod
    k_frag *= ph_deam_mod  # fragmentation also pH-dependent
    k_svp *= exc_svp_mod * ph_agg_mod

    # --- Time integration (Euler method) ---
    dt = timepoint_interval_months
    n_steps = int(duration_months / dt) + 1
    times = np.linspace(0, duration_months, n_steps)

    hmw = starting_hmw_pct
    lmw = 0.5  # starting LMW %
    acidic = starting_acidic_pct
    svp = 100.0  # starting SVP count/mL
    potency = 100.0  # starting potency %

    timepoints = []
    shelf_life = None

    for t in times:
        monomer = max(0.0, 100.0 - hmw - lmw)

        tp = StabilityTimepoint(
            month=round(float(t), 1),
            temperature_c=temperature_c,
            sec_hmw_pct=round(hmw, 2),
            sec_monomer_pct=round(monomer, 2),
            sec_lmw_pct=round(lmw, 2),
            svp_per_ml=round(svp, 0),
            cief_acidic_pct=round(acidic, 1),
            potency_pct=round(potency, 1),
        )
        timepoints.append(tp)

        # Check spec limit for shelf life
        if hmw >= 5.0 and shelf_life is None:
            shelf_life = float(t)

        # --- Kinetic step ---
        d_hmw = k_agg * (monomer / 100.0) * dt  # second-order-ish
        d_lmw = k_frag * (monomer / 100.0) * dt
        d_acidic = k_deam * dt
        d_svp = k_svp * (hmw / 100.0 + 0.01) * dt  # linked to HMW
        d_potency = (0.15 * k_agg + 0.15 * k_frag + 0.1 * k_deam) * dt

        hmw = min(hmw + d_hmw, 50.0)
        lmw = min(lmw + d_lmw, 20.0)
        acidic = min(acidic + d_acidic, 60.0)
        svp = min(svp + d_svp, 100000.0)
        potency = max(potency - d_potency, 0.0)

    # v7.4.0: Compute shelf_life from inverse overall degradation rate.
    # This creates a continuous, Tm-correlated shelf_life instead of the
    # flat 36-month cap that destroyed correlation (rho=0.087).
    #
    # Method: compute a composite "quality decay rate" from all CQA rates,
    # then shelf_life = (spec_margin) / (decay_rate). This is analogous
    # to how real stability programs extrapolate from accelerated data.
    _MAX_SHELF_LIFE_MONTHS = 120.0

    final_tp = timepoints[-1] if timepoints else None

    if shelf_life is not None:
        # HMW spec was hit during simulation
        shelf_life_est = min(shelf_life, _MAX_SHELF_LIFE_MONTHS)
    elif final_tp is not None:
        # Composite quality decay rate (weighted sum of CQA degradation rates)
        hmw_rate = (final_tp.sec_hmw_pct - starting_hmw_pct) / max(duration_months, 1)
        pot_rate = (100.0 - final_tp.potency_pct) / max(duration_months, 1)
        acid_rate = (final_tp.cief_acidic_pct - starting_acidic_pct) / max(duration_months, 1)

        # Weighted composite normalized to actual 5°C rate magnitudes:
        # At 5°C: HMW rate ~0.01-0.08, potency ~0.02, acidic ~0.10
        composite_rate = (
            0.50 * max(hmw_rate, 0) / 0.03     # 0.03%/mo HMW = score 1.0
            + 0.30 * max(pot_rate, 0) / 0.02    # 0.02%/mo potency loss = 1.0
            + 0.20 * max(acid_rate, 0) / 0.10   # 0.10%/mo acidic growth = 1.0
        )

        if composite_rate > 1e-6:
            # Shelf life inversely proportional to degradation rate
            # Scale: composite_rate=1.0 → 30 months (calibrated against
            # NISTmAb reference: sucrose+PS80 at 5°C → ~24-30 months)
            shelf_life_est = min(30.0 / composite_rate, _MAX_SHELF_LIFE_MONTHS)
        else:
            shelf_life_est = _MAX_SHELF_LIFE_MONTHS
    else:
        shelf_life_est = 36.0

    # Final values
    final = timepoints[-1]
    hmw_growth_rate = (final.sec_hmw_pct - starting_hmw_pct) / max(duration_months, 1)

    # 24-month spec: HMW < 5%, potency > 80%
    passes_spec = final.sec_hmw_pct < 5.0 and final.potency_pct > 80.0

    if not condition_name:
        condition_name = f"{temperature_c}°C / {duration_months} months"

    summary_lines = [
        f"Stability: {condition_name}",
        f"  SEC HMW: {starting_hmw_pct:.1f}% → {final.sec_hmw_pct:.1f}% "
        f"(+{final.sec_hmw_pct - starting_hmw_pct:.2f}%, "
        f"rate: {hmw_growth_rate:.3f}%/month)",
        f"  SVP (>=10um): {final.svp_per_ml:.0f}/mL",
        f"  cIEF Acidic: {starting_acidic_pct:.1f}% → {final.cief_acidic_pct:.1f}%",
        f"  Potency: 100% → {final.potency_pct:.1f}%",
        f"  Shelf life (HMW<5%): ~{shelf_life_est:.0f} months",
        f"  Spec: {'PASS' if passes_spec else 'FAIL'}",
    ]

    _result = StabilityResult(
        condition_name=condition_name,
        temperature_c=temperature_c,
        duration_months=duration_months,
        timepoints=timepoints,
        starting_hmw_pct=starting_hmw_pct,
        final_hmw_pct=final.sec_hmw_pct,
        hmw_growth_rate_pct_per_month=round(hmw_growth_rate, 4),
        final_svp_per_ml=final.svp_per_ml,
        final_acidic_pct=final.cief_acidic_pct,
        final_potency_pct=final.potency_pct,
        shelf_life_months=round(shelf_life_est, 1),
        passes_24month_spec=passes_spec,
        summary="\n".join(summary_lines),
    )

    try:
        from dataclasses import asdict
        from src.label_emitter import emit_prediction_label
        emit_prediction_label("stability", asdict(_result), {"temperature_c": temperature_c, "duration_months": duration_months})
    except Exception:
        pass  # Label emission should never break predictions

    return _result


# =========================================================================
# 5. Dual-Condition Stability Study
# =========================================================================

def run_stability_study(
    starting_hmw_pct: float = 1.0,
    starting_acidic_pct: float = 15.0,
    formulation_ph: float = 6.0,
    pI: float = 8.5,
    excipients: Optional[List[str]] = None,
    deamidation_sites: int = 5,
    dp_clip_sites: int = 1,
    hydrophobicity: float = 0.35,
    Tm: Optional[float] = None,
) -> DualConditionResult:
    """
    Run ICH-style dual-condition stability study.

    Simulates:
      - 5°C long-term storage for 24 months (ICH Q5C long-term)
      - 40°C accelerated stability for 3 months (ICH Q5C accelerated)

    Parameters
    ----------
    starting_hmw_pct    : Initial SEC HMW % (realistic range: 0.1-10.0)
    starting_acidic_pct : Initial cIEF acidic %
    formulation_ph      : Buffer pH
    pI                  : Protein isoelectric point
    excipients          : Excipient list
    deamidation_sites   : Number of deamidation hotspots
    dp_clip_sites       : Number of Asp-Pro clip sites
    hydrophobicity      : Hydrophobicity score
    Tm                  : Melting temperature (°C). Distinct from starting_hmw_pct.
    """
    if excipients is None:
        excipients = []

    # Long-term: 5°C / 24 months
    long_term = simulate_stability(
        starting_hmw_pct=starting_hmw_pct,
        starting_acidic_pct=starting_acidic_pct,
        formulation_ph=formulation_ph,
        pI=pI,
        excipients=excipients,
        temperature_c=5.0,
        duration_months=24,
        deamidation_sites=deamidation_sites,
        dp_clip_sites=dp_clip_sites,
        hydrophobicity=hydrophobicity,
        timepoint_interval_months=1.0,
        condition_name="5°C Long-term (24 months)",
        Tm=Tm,
    )

    # Accelerated: 40°C / 3 months
    accelerated = simulate_stability(
        starting_hmw_pct=starting_hmw_pct,
        starting_acidic_pct=starting_acidic_pct,
        formulation_ph=formulation_ph,
        pI=pI,
        excipients=excipients,
        temperature_c=40.0,
        duration_months=3,
        deamidation_sites=deamidation_sites,
        dp_clip_sites=dp_clip_sites,
        hydrophobicity=hydrophobicity,
        timepoint_interval_months=0.25,
        condition_name="40°C Accelerated (3 months)",
        Tm=Tm,
    )

    # Predicted shelf life from long-term data
    shelf_life = long_term.shelf_life_months if long_term.shelf_life_months is not None else 36.0

    # Grade
    if shelf_life >= 24 and accelerated.final_hmw_pct < 15:
        grade = "Excellent"
    elif shelf_life >= 18 and accelerated.final_hmw_pct < 25:
        grade = "Good"
    elif shelf_life >= 12:
        grade = "At Risk"
    else:
        grade = "Poor"

    # Add stability context note
    if grade in ("At Risk", "Poor"):
        grade_note = (
            "Note: Stability projections are based on semi-empirical kinetic models. "
            "Actual shelf life should be confirmed by real-time stability studies (ICH Q1A). "
            "Accelerated conditions (40°C) typically overpredict degradation."
        )
    else:
        grade_note = ""

    summary = (
        f"Dual-Condition Stability Study:\n"
        f"  Long-term (5°C): HMW {long_term.starting_hmw_pct:.1f}% → "
        f"{long_term.final_hmw_pct:.1f}% at 24 months\n"
        f"  Accelerated (40°C): HMW {accelerated.starting_hmw_pct:.1f}% → "
        f"{accelerated.final_hmw_pct:.1f}% at 3 months\n"
        f"  Predicted shelf life: ~{shelf_life:.0f} months\n"
        f"  Grade: {grade}"
    )
    if grade_note:
        summary += f"\n  {grade_note}"

    return DualConditionResult(
        long_term=long_term,
        accelerated=accelerated,
        predicted_shelf_life_months=shelf_life,
        overall_stability_grade=grade,
        summary=summary,
    )


# =========================================================================
# Self-Test
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    print("=" * 60)
    print("stability_twin.py — Self-Test")
    print("=" * 60)

    # Test 1: Arrhenius rate
    k5 = arrhenius_rate(0.1, EA_AGGREGATION, T_REF, T_REF)
    k40 = arrhenius_rate(0.1, EA_AGGREGATION, T_REF, T_ACCEL)
    assert abs(k5 - 0.1) < 0.001  # same temp → same rate
    assert k40 > k5 * 10  # 40°C should be >10x faster
    print(f"  [1/6] Arrhenius: k(5°C)={k5:.4f}, k(40°C)={k40:.2f} "
          f"(ratio={k40/k5:.0f}x) ✅")

    # Test 2: Single condition simulation
    result = simulate_stability(
        starting_hmw_pct=1.0,
        formulation_ph=6.0, pI=8.5,
        excipients=["sucrose", "ps80"],
        temperature_c=5.0, duration_months=24,
        deamidation_sites=5,
    )
    assert len(result.timepoints) > 20
    assert result.timepoints[0].sec_hmw_pct == 1.0
    assert result.final_hmw_pct >= 1.0  # HMW should grow
    print(f"  [2/6] 5°C/24mo: HMW {result.starting_hmw_pct:.1f}% → "
          f"{result.final_hmw_pct:.2f}%, shelf life ~{result.shelf_life_months:.0f}mo ✅")

    # Test 3: Accelerated condition
    accel = simulate_stability(
        starting_hmw_pct=1.0,
        formulation_ph=6.0, pI=8.5,
        temperature_c=40.0, duration_months=3,
        deamidation_sites=5,
    )
    assert accel.final_hmw_pct > result.final_hmw_pct  # 40°C faster than 5°C
    print(f"  [3/6] 40°C/3mo: HMW → {accel.final_hmw_pct:.2f}% "
          f"(vs 5°C={result.final_hmw_pct:.2f}%) ✅")

    # Test 4: Excipient effect
    no_exc = simulate_stability(
        starting_hmw_pct=1.0, formulation_ph=6.0, pI=8.5,
        excipients=[], temperature_c=25.0, duration_months=6,
    )
    with_exc = simulate_stability(
        starting_hmw_pct=1.0, formulation_ph=6.0, pI=8.5,
        excipients=["sucrose", "ps80"], temperature_c=25.0, duration_months=6,
    )
    assert with_exc.final_hmw_pct < no_exc.final_hmw_pct
    print(f"  [4/6] Excipient protection: no_exc={no_exc.final_hmw_pct:.2f}% "
          f"vs with_exc={with_exc.final_hmw_pct:.2f}% ✅")

    # Test 5: Dual-condition study
    dual = run_stability_study(
        starting_hmw_pct=1.0, starting_acidic_pct=12.0,
        formulation_ph=6.0, pI=8.5,
        excipients=["sucrose", "ps80"],
        deamidation_sites=4, hydrophobicity=0.35,
    )
    assert dual.long_term is not None
    assert dual.accelerated is not None
    assert dual.overall_stability_grade in ("Excellent", "Good", "At Risk", "Poor")
    print(f"  [5/6] Dual-condition: grade={dual.overall_stability_grade}, "
          f"shelf life ~{dual.predicted_shelf_life_months:.0f}mo ✅")

    # Test 6: Potency decay
    assert dual.long_term.final_potency_pct < 100.0
    assert dual.long_term.final_potency_pct > 50.0  # shouldn't decay that much at 5°C
    print(f"  [6/6] Potency: 100% → {dual.long_term.final_potency_pct:.1f}% "
          f"(5°C/24mo) ✅")

    print()
    print(dual.summary)
    print()
    print("Self-test: 6/6 passed")
