"""
formulation_twin.py  ·  ProtePilot — Milestone 17
===========================================================
Formulation Digital Twin: Buffer & Excipient Simulation

Version   : 1.0
Author    : Di (ProtePilot)
Depends   : numpy, biopython (optional for Henderson-Hasselbalch)

Purpose
------------------------------------------------------------
Simulates the impact of formulation conditions (buffer pH,
buffer type, and excipients) on antibody developability risk.
Provides a physics-based feedback loop that updates aggregation
risk, thermal stability, and viscosity risk in real time as
users adjust formulation parameters.

Key Physics
------------------------------------------------------------
  1. Net Charge at Buffer pH
     Henderson-Hasselbalch titration of all titratable residues
     (Asp, Glu, His, Cys, Tyr, Lys, Arg, N-term, C-term).
     When |net_charge| -> 0 (pH near pI), colloidal stability
     drops and aggregation risk spikes.

  2. Buffer Ionic Strength & Type Effects
     - Histidine (pH 5.5-6.5): low viscosity, good for subQ
     - Citrate (pH 5.5-7.0): strong buffering, moderate viscosity
     - Phosphate (pH 6.5-8.0): wide range, higher viscosity risk at conc.

  3. Excipient Stabilization
     - Trehalose/Sucrose: preferential exclusion stabilizes native fold
       -> reduces aggregation risk, improves thermal stability
     - Polysorbate 80 (PS80): prevents surface adsorption and
       agitation-induced aggregation -> reduces aggregation risk

References
------------------------------------------------------------
  Ohtake et al. (2011) J Pharm Sci 100:2020 — Trehalose stabilization
  Wang et al. (2007) J Pharm Sci 96:1 — Formulation of mAbs
  He et al. (2011) J Pharm Sci 100:1330 — Buffer effects on mAb stability
  Shire et al. (2004) J Pharm Sci 93:1390 — mAb formulation challenges
  Kamerzell et al. (2011) Adv Drug Deliv Rev 63:1118 — Excipient effects
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ProtePilot.FormulationTwin")


# ===========================================================================
# 1. Constants — pKa Values for Henderson-Hasselbalch Titration
# ===========================================================================

# Standard pKa values for titratable amino acid residues (isolated)
# Source: Stryer Biochemistry, Lehninger Principles of Biochemistry
RESIDUE_PKA: Dict[str, float] = {
    "D": 3.65,    # Asp (carboxyl side chain)
    "E": 4.25,    # Glu (carboxyl side chain)
    "H": 6.00,    # His (imidazole)
    "C": 8.18,    # Cys (thiol)
    "Y": 10.07,   # Tyr (phenol)
    "K": 10.54,   # Lys (amino)
    "R": 12.48,   # Arg (guanidinium)
}

# Terminal pKa values
PKA_N_TERM = 9.69   # alpha-amino
PKA_C_TERM = 2.34   # alpha-carboxyl


# ===========================================================================
# 2. Buffer System Catalog
# ===========================================================================

@dataclass
class BufferSystem:
    """Represents a buffer system with its properties."""
    name: str
    full_name: str
    optimal_ph_low: float
    optimal_ph_high: float
    pka_values: List[float]
    ionic_strength_factor: float    # relative ionic strength at 20 mM
    viscosity_modifier: float       # relative viscosity contribution (0-1 scale)
    stabilization_bonus: float      # intrinsic stabilization (0-1 scale)
    notes: str = ""


BUFFER_CATALOG: Dict[str, BufferSystem] = {
    "histidine": BufferSystem(
        name="histidine",
        full_name="L-Histidine / Histidine-HCl",
        optimal_ph_low=5.5,
        optimal_ph_high=6.5,
        pka_values=[6.0],
        ionic_strength_factor=0.8,
        viscosity_modifier=0.05,
        stabilization_bonus=0.08,
        notes="Preferred for subcutaneous formulations due to low viscosity. "
              "Excellent buffering at pH 6.0.",
    ),
    "citrate": BufferSystem(
        name="citrate",
        full_name="Sodium Citrate / Citric Acid",
        optimal_ph_low=3.0,
        optimal_ph_high=6.5,
        pka_values=[3.13, 4.76, 6.40],
        ionic_strength_factor=1.2,
        viscosity_modifier=0.12,
        stabilization_bonus=0.05,
        notes="Strong buffering capacity across wide pH range. "
              "May chelate divalent metal ions. Slightly higher viscosity.",
    ),
    "phosphate": BufferSystem(
        name="phosphate",
        full_name="Sodium Phosphate (Mono/Dibasic)",
        optimal_ph_low=6.0,
        optimal_ph_high=8.0,
        pka_values=[2.15, 7.20, 12.35],
        ionic_strength_factor=1.0,
        viscosity_modifier=0.10,
        stabilization_bonus=0.03,
        notes="Widely used for IV formulations. Good buffering at pH 7.2. "
              "Higher viscosity risk at elevated concentrations.",
    ),
}


# ===========================================================================
# 3. Excipient Catalog
# ===========================================================================

@dataclass
class Excipient:
    """Represents a stabilizing excipient with its effects."""
    name: str
    category: str
    agg_risk_reduction: float       # fractional reduction in aggregation risk
    stability_boost: float          # additive improvement to thermal stability
    viscosity_modifier: float       # fractional change in viscosity risk (negative = lower)
    mechanism: str
    typical_conc: str


EXCIPIENT_CATALOG: Dict[str, Excipient] = {
    "trehalose": Excipient(
        name="Trehalose",
        category="Sugar (Disaccharide)",
        agg_risk_reduction=0.25,
        stability_boost=0.08,
        viscosity_modifier=0.05,     # slight viscosity increase from sugar
        mechanism="Preferential exclusion: trehalose is excluded from the protein "
                  "surface, thermodynamically favoring the compact native state and "
                  "suppressing aggregation-prone unfolded intermediates.",
        typical_conc="5-10% w/v (150-300 mM)",
    ),
    "sucrose": Excipient(
        name="Sucrose",
        category="Sugar (Disaccharide)",
        agg_risk_reduction=0.22,
        stability_boost=0.07,
        viscosity_modifier=0.06,
        mechanism="Preferential exclusion similar to trehalose. Slightly less "
                  "effective at equimolar concentration but widely used. "
                  "Also acts as a lyoprotectant during freeze-drying.",
        typical_conc="5-10% w/v (146-292 mM)",
    ),
    "ps80": Excipient(
        name="Polysorbate 80 (PS80)",
        category="Surfactant (Non-ionic)",
        agg_risk_reduction=0.15,
        stability_boost=0.02,
        viscosity_modifier=-0.03,    # slight viscosity reduction
        mechanism="Competes with protein for air-water and surface interfaces, "
                  "preventing adsorption-induced unfolding and aggregation. "
                  "Critical for protecting against agitation stress.",
        typical_conc="0.01-0.05% w/v",
    ),
}


# ===========================================================================
# 4. Henderson-Hasselbalch Net Charge Calculator
# ===========================================================================

def count_titratable_residues(sequence: str) -> Dict[str, int]:
    """
    Count titratable residues in a protein sequence.

    Parameters
    ----------
    sequence : Upper-case single-letter amino acid sequence

    Returns
    -------
    Dict mapping residue code -> count, plus 'N_term' and 'C_term'
    """
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    counts: Dict[str, int] = {}
    for res_code in RESIDUE_PKA:
        counts[res_code] = seq.count(res_code)
    counts["N_term"] = 1
    counts["C_term"] = 1
    return counts


def compute_net_charge_at_ph(
    sequence: str,
    ph: float,
    residue_counts: Optional[Dict[str, int]] = None,
) -> float:
    """
    Calculate protein net charge at a given pH using Henderson-Hasselbalch.

    For each titratable group:
      - Acidic groups (D, E, C-term): charge = -1 / (1 + 10^(pKa - pH))
      - Basic groups (K, R, H, N-term): charge = +1 / (1 + 10^(pH - pKa))
      - Cys and Tyr behave as weak acids (lose H+ at high pH)

    Parameters
    ----------
    sequence : Amino acid sequence (single-letter, upper case)
    ph       : Buffer pH value
    residue_counts : Pre-computed counts (optional, avoids re-counting)

    Returns
    -------
    Net charge (float). Positive = cationic, Negative = anionic.
    """
    if residue_counts is None:
        residue_counts = count_titratable_residues(sequence)

    net_charge = 0.0

    # Acidic residues (lose proton -> become negative)
    for res in ("D", "E"):
        pka = RESIDUE_PKA[res]
        count = residue_counts.get(res, 0)
        # Fraction deprotonated (negative charge)
        fraction_neg = 1.0 / (1.0 + 10.0 ** (pka - ph))
        net_charge -= count * fraction_neg

    # C-terminus (acidic)
    c_frac = 1.0 / (1.0 + 10.0 ** (PKA_C_TERM - ph))
    net_charge -= residue_counts.get("C_term", 1) * c_frac

    # Cys and Tyr (weak acids, lose proton at high pH)
    for res in ("C", "Y"):
        pka = RESIDUE_PKA[res]
        count = residue_counts.get(res, 0)
        fraction_neg = 1.0 / (1.0 + 10.0 ** (pka - ph))
        net_charge -= count * fraction_neg

    # Basic residues (gain proton -> become positive)
    for res in ("K", "R", "H"):
        pka = RESIDUE_PKA[res]
        count = residue_counts.get(res, 0)
        # Fraction protonated (positive charge)
        fraction_pos = 1.0 / (1.0 + 10.0 ** (ph - pka))
        net_charge += count * fraction_pos

    # N-terminus (basic)
    n_frac = 1.0 / (1.0 + 10.0 ** (ph - PKA_N_TERM))
    net_charge += residue_counts.get("N_term", 1) * n_frac

    return float(net_charge)


def compute_charge_curve(
    sequence: str,
    ph_range: Tuple[float, float] = (3.0, 12.0),
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute net charge vs pH curve for a protein sequence.

    Returns
    -------
    ph_values : (n_points,) array of pH values
    charges   : (n_points,) array of net charges
    """
    residue_counts = count_titratable_residues(sequence)
    ph_values = np.linspace(ph_range[0], ph_range[1], n_points)
    charges = np.array([
        compute_net_charge_at_ph(sequence, ph, residue_counts)
        for ph in ph_values
    ])
    return ph_values, charges


def estimate_pI_from_sequence(sequence: str, tolerance: float = 0.01) -> float:
    """
    Estimate isoelectric point by bisection on Henderson-Hasselbalch charges.

    Parameters
    ----------
    sequence  : Amino acid sequence
    tolerance : pH precision for bisection convergence

    Returns
    -------
    Estimated pI (float)
    """
    residue_counts = count_titratable_residues(sequence)
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        charge = compute_net_charge_at_ph(sequence, mid, residue_counts)
        if abs(charge) < tolerance:
            return round(mid, 2)
        if charge > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 2)


# ===========================================================================
# 5. Formulation Effect Engine
# ===========================================================================

@dataclass
class FormulationCondition:
    """User-selected formulation conditions."""
    buffer_ph: float = 6.0
    buffer_type: str = "histidine"
    excipients: List[str] = field(default_factory=list)
    # Optional: concentration, ionic strength, etc.
    buffer_concentration_mM: float = 20.0


@dataclass
class FormulationEffect:
    """Computed formulation effects on developability."""
    net_charge: float = 0.0
    charge_near_zero: bool = False
    ph_pI_distance: float = 0.0
    agg_risk_modifier: float = 0.0      # additive change to agg_risk
    stability_modifier: float = 0.0     # additive change to stability
    viscosity_modifier: float = 0.0     # additive change to viscosity_risk
    buffer_in_range: bool = True
    buffer_notes: str = ""
    excipient_effects: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    formulation_summary: str = ""


def compute_formulation_effects(
    condition: FormulationCondition,
    pI: float,
    sequence: Optional[str] = None,
    hydrophobicity: float = 0.35,
) -> FormulationEffect:
    """
    Compute the impact of formulation conditions on developability risk.

    This is the core physics & ML feedback loop. It calculates:
    1. Net charge at buffer pH (Henderson-Hasselbalch if sequence available)
    2. Aggregation rescue: pH near pI => risk spike; excipients => risk decrease
    3. Buffer type effects on viscosity and stability
    4. Excipient stabilization effects

    Parameters
    ----------
    condition      : FormulationCondition with user-selected parameters
    pI             : Protein isoelectric point
    sequence       : Full protein sequence (optional, for H-H charge calc)
    hydrophobicity : GRAVY-based hydrophobicity score

    Returns
    -------
    FormulationEffect with all modifiers and diagnostics
    """
    effect = FormulationEffect()
    ph = condition.buffer_ph

    # ---------------------------------------------------------------
    # 1. Net Charge at Buffer pH
    # ---------------------------------------------------------------
    if sequence and len(sequence) > 10:
        effect.net_charge = compute_net_charge_at_ph(sequence, ph)
    else:
        # Approximate net charge from pI and pH using simplified model
        # charge ~ (pI - pH) * estimated_slope
        # For a typical 150 kDa mAb (~1300 aa), slope ~ 5-15 charges per pH unit
        slope_estimate = 8.0
        effect.net_charge = slope_estimate * (pI - ph)

    effect.ph_pI_distance = abs(ph - pI)
    effect.charge_near_zero = abs(effect.net_charge) < 3.0 or effect.ph_pI_distance < 0.5

    # ---------------------------------------------------------------
    # 2. pH-Driven Aggregation Risk (Colloidal Stability)
    # ---------------------------------------------------------------
    # When pH is close to pI, net charge approaches zero,
    # electrostatic repulsion vanishes, and aggregation risk spikes.
    # This is the dominant formulation risk factor.

    if effect.ph_pI_distance < 0.3:
        # Danger zone: pH essentially at pI
        agg_penalty = 0.35
        effect.warnings.append(
            f"CRITICAL: Buffer pH ({ph:.1f}) is within 0.3 units of pI ({pI:.2f}). "
            f"Net charge ~ {effect.net_charge:.1f}. Colloidal stability is minimal — "
            f"high aggregation risk expected."
        )
    elif effect.ph_pI_distance < 0.8:
        # Warning zone
        agg_penalty = 0.20
        effect.warnings.append(
            f"WARNING: Buffer pH ({ph:.1f}) is close to pI ({pI:.2f}). "
            f"Net charge = {effect.net_charge:.1f}. Consider adjusting pH further "
            f"from pI for better colloidal stability."
        )
    elif effect.ph_pI_distance < 1.5:
        # Mild concern
        agg_penalty = 0.05
    else:
        # Good separation — net charge provides repulsion
        agg_penalty = -0.05  # slight benefit from good charge
        effect.recommendations.append(
            f"Buffer pH ({ph:.1f}) is well-separated from pI ({pI:.2f}). "
            f"Good electrostatic repulsion (net charge ~ {effect.net_charge:.1f})."
        )

    # Hydrophobicity amplification: hydrophobic proteins aggregate more at pI
    if hydrophobicity > 0.40 and effect.ph_pI_distance < 1.0:
        agg_penalty += 0.10
        effect.warnings.append(
            "High hydrophobicity combined with pH near pI amplifies aggregation risk."
        )

    effect.agg_risk_modifier += agg_penalty

    # ---------------------------------------------------------------
    # 3. Buffer Type Effects
    # ---------------------------------------------------------------
    buffer = BUFFER_CATALOG.get(condition.buffer_type)
    if buffer is None:
        buffer = BUFFER_CATALOG["histidine"]

    # Check if pH is within buffer's optimal range
    effect.buffer_in_range = buffer.optimal_ph_low <= ph <= buffer.optimal_ph_high
    if not effect.buffer_in_range:
        effect.warnings.append(
            f"{buffer.full_name} has optimal buffering at pH "
            f"{buffer.optimal_ph_low:.1f}-{buffer.optimal_ph_high:.1f}. "
            f"Current pH ({ph:.1f}) is outside this range — reduced buffering capacity."
        )

    # Buffer-specific modifiers
    effect.viscosity_modifier += buffer.viscosity_modifier
    effect.stability_modifier += buffer.stabilization_bonus
    effect.buffer_notes = buffer.notes

    # ---------------------------------------------------------------
    # 4. Excipient Stabilization Effects
    # ---------------------------------------------------------------
    for exc_name in condition.excipients:
        exc = EXCIPIENT_CATALOG.get(exc_name)
        if exc is None:
            continue

        # Aggregation rescue: excipients reduce aggregation risk
        effect.agg_risk_modifier -= exc.agg_risk_reduction
        # Stability boost
        effect.stability_modifier += exc.stability_boost
        # Viscosity impact
        effect.viscosity_modifier += exc.viscosity_modifier

        effect.excipient_effects.append({
            "name": exc.name,
            "category": exc.category,
            "agg_reduction": exc.agg_risk_reduction,
            "stability_boost": exc.stability_boost,
            "viscosity_change": exc.viscosity_modifier,
            "mechanism": exc.mechanism,
            "concentration": exc.typical_conc,
        })

        effect.recommendations.append(
            f"{exc.name} ({exc.typical_conc}): {exc.mechanism[:80]}..."
        )

    # ---------------------------------------------------------------
    # 5. Generate Formulation Summary
    # ---------------------------------------------------------------
    exc_names = ", ".join(
        EXCIPIENT_CATALOG[e].name for e in condition.excipients
        if e in EXCIPIENT_CATALOG
    ) or "None"

    buffer_label = buffer.full_name
    effect.formulation_summary = (
        f"Buffer: {buffer_label} pH {ph:.1f} ({condition.buffer_concentration_mM:.0f} mM) | "
        f"Excipients: {exc_names} | "
        f"Net charge at pH {ph:.1f}: {effect.net_charge:+.1f} | "
        f"Distance from pI: {effect.ph_pI_distance:.2f} pH units"
    )

    return effect


def apply_formulation_to_developability(
    base_predictions: Dict[str, float],
    formulation_effect: FormulationEffect,
) -> Dict[str, float]:
    """
    Apply formulation effects to base developability predictions.

    Takes the raw XGBoost/rule-based predictions and adjusts them
    based on formulation conditions. Clamps all values to valid ranges.

    Parameters
    ----------
    base_predictions : Dict with 'agg_risk', 'stability', 'viscosity_risk'
    formulation_effect : Computed FormulationEffect

    Returns
    -------
    Dict with adjusted predictions (same keys)
    """
    adjusted = {}

    # Aggregation risk: add formulation modifier
    raw_agg = base_predictions.get("agg_risk", 0.2)
    adjusted["agg_risk"] = float(np.clip(
        raw_agg + formulation_effect.agg_risk_modifier,
        0.02, 0.98,
    ))

    # Thermal stability: add modifier
    raw_stab = base_predictions.get("stability", 0.8)
    adjusted["stability"] = float(np.clip(
        raw_stab + formulation_effect.stability_modifier,
        0.20, 0.99,
    ))

    # Viscosity risk: add modifier
    raw_visc = base_predictions.get("viscosity_risk", 0.15)
    adjusted["viscosity_risk"] = float(np.clip(
        raw_visc + formulation_effect.viscosity_modifier,
        0.02, 0.95,
    ))

    return adjusted


def recompute_developability_score(
    predictions: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Recompute the composite Developability Score from adjusted predictions.

    Uses the same formula as developability_predictor.py:
        Score = w_agg * agg_risk + w_stab * (1 - stability) + w_visc * viscosity_risk

    Parameters
    ----------
    predictions : Dict with 'agg_risk', 'stability', 'viscosity_risk'
    weights     : Optional custom weights (default: 0.40 / 0.30 / 0.30)

    Returns
    -------
    Dict with 'score', 'grade', 'color'
    """
    if weights is None:
        weights = {"agg": 0.40, "stability": 0.30, "viscosity": 0.30}

    agg = predictions.get("agg_risk", 0.2)
    stab = predictions.get("stability", 0.8)
    visc = predictions.get("viscosity_risk", 0.15)

    score = (
        weights["agg"] * agg
        + weights["stability"] * (1.0 - stab)
        + weights["viscosity"] * visc
    )
    score = float(np.clip(score, 0.0, 1.0))

    if score < 0.3:
        grade = "Low Risk"
        color = "#10B981"
    elif score < 0.6:
        grade = "Medium Risk"
        color = "#F59E0B"
    else:
        grade = "High Risk"
        color = "#EF4444"

    return {
        "score": round(score, 4),
        "grade": grade,
        "color": color,
    }


# ===========================================================================
# 6. Convenience API
# ===========================================================================

def run_formulation_assessment(
    pI: float,
    buffer_ph: float = 6.0,
    buffer_type: str = "histidine",
    excipients: Optional[List[str]] = None,
    sequence: Optional[str] = None,
    hydrophobicity: float = 0.35,
    base_predictions: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    One-call API for formulation digital twin assessment.

    Parameters
    ----------
    pI              : Protein isoelectric point
    buffer_ph       : Buffer pH (4.5 to 8.0)
    buffer_type     : One of 'histidine', 'citrate', 'phosphate'
    excipients      : List of excipient keys ('trehalose', 'sucrose', 'ps80')
    sequence        : Optional protein sequence for Henderson-Hasselbalch
    hydrophobicity  : GRAVY hydrophobicity score
    base_predictions: Optional base developability predictions to adjust

    Returns
    -------
    Dict with complete assessment results
    """
    if excipients is None:
        excipients = []

    # Default base predictions if not provided
    if base_predictions is None:
        base_predictions = {
            "agg_risk": 0.20,
            "stability": 0.82,
            "viscosity_risk": 0.15,
        }

    condition = FormulationCondition(
        buffer_ph=buffer_ph,
        buffer_type=buffer_type,
        excipients=excipients,
    )

    # Compute formulation effects
    effect = compute_formulation_effects(
        condition=condition,
        pI=pI,
        sequence=sequence,
        hydrophobicity=hydrophobicity,
    )

    # Apply to base predictions
    adjusted = apply_formulation_to_developability(base_predictions, effect)

    # Recompute composite score
    score_info = recompute_developability_score(adjusted)

    # Also compute base score for comparison
    base_score_info = recompute_developability_score(base_predictions)

    return {
        "status": "success",
        "formulation": {
            "buffer_ph": buffer_ph,
            "buffer_type": buffer_type,
            "buffer_full_name": BUFFER_CATALOG.get(buffer_type, BUFFER_CATALOG["histidine"]).full_name,
            "excipients": excipients,
            "summary": effect.formulation_summary,
        },
        "net_charge": round(effect.net_charge, 2),
        "ph_pI_distance": round(effect.ph_pI_distance, 2),
        "charge_near_zero": effect.charge_near_zero,
        "base_predictions": base_predictions,
        "adjusted_predictions": adjusted,
        "base_score": base_score_info,
        "adjusted_score": score_info,
        "score_delta": round(score_info["score"] - base_score_info["score"], 4),
        "modifiers": {
            "agg_risk": round(effect.agg_risk_modifier, 4),
            "stability": round(effect.stability_modifier, 4),
            "viscosity_risk": round(effect.viscosity_modifier, 4),
        },
        "buffer_in_range": effect.buffer_in_range,
        "buffer_notes": effect.buffer_notes,
        "excipient_effects": effect.excipient_effects,
        "warnings": effect.warnings,
        "recommendations": effect.recommendations,
    }


# ===========================================================================
# 7. Self-Test
# ===========================================================================

if __name__ == "__main__":
    _passed = 0
    _total = 4

    print("=" * 70)
    print("  Formulation Digital Twin — Self-Test (with assertions)")
    print("=" * 70)

    # Test 1: Henderson-Hasselbalch charge calculation
    print("\n--- Test 1: pI estimation & charge at pI ≈ 0 ---")
    test_seq = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTY"
        "YADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGYDYWGQGTLVTVSS"
    )
    pi_est = estimate_pI_from_sequence(test_seq)
    charge_at_7 = compute_net_charge_at_ph(test_seq, 7.0)
    charge_at_pi = compute_net_charge_at_ph(test_seq, pi_est)
    assert 4.0 <= pi_est <= 11.0, f"pI out of reasonable range: {pi_est}"
    assert abs(charge_at_pi) < 1.0, f"Charge at pI should be near zero: {charge_at_pi}"
    assert isinstance(charge_at_7, float), "Charge must be numeric"
    print(f"  pI={pi_est}, charge@pI={charge_at_pi:+.2f}, charge@7={charge_at_7:+.2f}")
    _passed += 1

    # Test 2: Formulation effects — pH near pI (bad) → score should worsen
    print("\n--- Test 2: pH near pI (danger zone) → score worsens ---")
    result_bad = run_formulation_assessment(
        pI=8.5, buffer_ph=8.3, buffer_type="phosphate",
        excipients=[], hydrophobicity=0.40,
    )
    assert result_bad["status"] == "success", "Assessment failed"
    assert result_bad["score_delta"] >= 0, f"pH near pI should worsen (increase) score: delta={result_bad['score_delta']}"
    assert result_bad["charge_near_zero"] is True, "pH 8.3 with pI 8.5 should flag charge_near_zero"
    assert len(result_bad["warnings"]) > 0, "Should have warnings when pH is near pI"
    print(f"  Delta={result_bad['score_delta']:+.4f}, warnings={len(result_bad['warnings'])}")
    _passed += 1

    # Test 3: Formulation effects — pH away from pI + excipients (good) → score improves
    print("\n--- Test 3: pH 6.0 + Trehalose + PS80 (rescue) → score improves ---")
    result_good = run_formulation_assessment(
        pI=8.5, buffer_ph=6.0, buffer_type="histidine",
        excipients=["trehalose", "ps80"], hydrophobicity=0.40,
    )
    assert result_good["status"] == "success", "Assessment failed"
    assert result_good["score_delta"] < result_bad["score_delta"], \
        "Optimized formulation should have better delta than danger-zone formulation"
    assert result_good["charge_near_zero"] is False, "pH 6.0 with pI 8.5 should not be near-zero charge"
    assert "adjusted_predictions" in result_good, "Missing adjusted_predictions"
    print(f"  Delta={result_good['score_delta']:+.4f}, charge_near_zero={result_good['charge_near_zero']}")
    _passed += 1

    # Test 4: Buffer out of range → should flag warning
    print("\n--- Test 4: Histidine at pH 8.0 (out of range) → warns ---")
    result_oor = run_formulation_assessment(
        pI=8.5, buffer_ph=8.0, buffer_type="histidine",
    )
    assert result_oor["buffer_in_range"] is False, "Histidine at pH 8.0 should be out of range"
    assert any("range" in w.lower() or "outside" in w.lower() or "buffer" in w.lower()
               for w in result_oor["warnings"]), "Should warn about buffer out of range"
    print(f"  buffer_in_range={result_oor['buffer_in_range']}, warnings={len(result_oor['warnings'])}")
    _passed += 1

    print(f"\nformulation_twin selftest: {_passed}/{_total} passed")
