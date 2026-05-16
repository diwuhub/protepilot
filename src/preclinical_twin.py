"""
preclinical_twin.py  ·  ProtePilot — Milestone 13
===========================================================
Preclinical PK & Post-Translational Twin: In-Vivo Simulation

Version   : 1.0
Author    : Di (ProtePilot)
Depends   : numpy

Purpose
------------------------------------------------------------
Provides preclinical pharmacokinetics (PK) prediction for
therapeutic antibodies. Bridges CMC/analytical characterization
with in-vivo half-life estimation using empirical models
calibrated to published IgG clearance data.

Key Functions
------------------------------------------------------------
  1. predict_human_half_life()
     - Empirical model: baseline ~21 days for standard IgG1
     - Penalizes extreme pI (>9.0 or <5.5)
     - Penalizes high hydrophobicity
     - Penalizes high liability density (aggregation proxy)
     - Accounts for FcRn binding motif integrity
     - Returns predicted half-life (days) + PK risk assessment

  2. predict_pk_parameters()
     - Full PK parameter estimation: CL, Vd, AUC, Cmax, Tmax
     - Based on allometric scaling from IgG clearance models

  3. assess_glycoform_pk_impact()
     - Glycoform-specific PK modifiers
     - High-mannose => faster clearance (mannose receptor)
     - Afucosylated => enhanced ADCC but similar PK
     - Sialylated => slightly extended half-life

References
------------------------------------------------------------
  Wang et al. (2008) J Clin Pharmacol 48:538-552 — IgG PK
  Robbie et al. (2013) Antimicrob Agents Chemother 57:6147 — mAb PK
  Ternant et al. (2015) mAbs 7:372-379 — FcRn & pI effects
  Liu (2015) J Pharm Sci 104:1866 — pI clearance correlation
  Goetze et al. (2011) Glycobiology 21:949 — Glycan PK impact
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

log = logging.getLogger("ProtePilot.PreclinicalTwin")


# ===========================================================================
# 1. Constants & Reference Data
# ===========================================================================

# Standard IgG1 baseline half-life (days) in humans
# Based on endogenous IgG1 t1/2 ~ 21 days (range 18-26)
BASELINE_HALF_LIFE_DAYS = 21.0

# pI penalty parameters
# References: Liu (2015) showed pI > 9.0 correlates with faster clearance
# Mechanism: highly charged mAbs undergo rapid pinocytic uptake in tissues
PI_OPTIMAL_LOW = 6.0
PI_OPTIMAL_HIGH = 8.5
PI_SEVERE_LOW = 4.5
PI_SEVERE_HIGH = 9.5

# Hydrophobicity penalty
# References: High hydrophobicity leads to non-specific binding, polyreactivity,
# and accelerated tissue uptake / off-target clearance
HYDROPHOBICITY_THRESHOLD_MILD = 0.40
HYDROPHOBICITY_THRESHOLD_SEVERE = 0.60

# Liability density penalty (aggregation proxy)
# High liability density => aggregation risk => immune complex clearance
LIAB_DENSITY_THRESHOLD_MILD = 50.0    # motifs per 1000 aa
LIAB_DENSITY_THRESHOLD_SEVERE = 80.0

# Molecular weight impact
# Larger molecules (bispecifics, fusions) may have different PK
MW_STANDARD_IgG = 150.0  # kDa
MW_PENALTY_THRESHOLD = 200.0  # Reduced half-life for very large constructs

# FcRn binding motif (His310, His435, Ile253 in Fc CH2-CH3 interface)
# Intact FcRn binding is essential for IgG recycling and long half-life
FCRN_BINDING_WEIGHT = 0.30  # 30% of half-life depends on FcRn recycling

# Glycoform PK modifiers (multiplier on baseline half-life)
GLYCOFORM_PK_MODIFIERS: Dict[str, Dict[str, Any]] = {
    "standard_cho": {
        "name": "Standard CHO (G0F/G1F dominant)",
        "half_life_multiplier": 1.00,
        "description": "Normal IgG1 glycosylation; dominant G0F/G1F profile",
        "clearance_note": "Baseline clearance via FcRn-mediated recycling",
        "dominant_glycans": ["G0F", "G1F", "G2F"],
        "pi_shift": 0.0,
        "mass_shift_per_site": 1444.53,  # G0F average
    },
    "high_mannose": {
        "name": "High-Mannose (Man5/Man8/Man9)",
        "half_life_multiplier": 0.60,
        "description": "High-mannose glycans are cleared rapidly via mannose receptor (MR) on macrophages and liver sinusoidal endothelial cells",
        "clearance_note": "Accelerated clearance via mannose receptor binding — CL increases ~2x",
        "dominant_glycans": ["Man5", "Man8", "Man9"],
        "pi_shift": 0.0,
        "mass_shift_per_site": 1234.43,  # Man5 average
    },
    "afucosylated": {
        "name": "Afucosylated (G0, enhanced ADCC)",
        "half_life_multiplier": 0.95,
        "description": "Afucosylated mAbs have enhanced Fc-gamma-RIIIa binding (ADCC) with slightly faster clearance due to immune activation",
        "clearance_note": "Minimal PK impact; enhanced effector function may increase target-mediated clearance",
        "dominant_glycans": ["G0", "G1", "G2"],
        "pi_shift": 0.0,
        "mass_shift_per_site": 1298.47,  # G0 average
    },
    "highly_sialylated": {
        "name": "Highly Sialylated (G2F+2SA)",
        "half_life_multiplier": 1.10,
        "description": "Sialylated IgGs may have extended half-life via reduced asialoglycoprotein receptor binding. Sialic acid adds negative charge, lowering pI.",
        "clearance_note": "Slightly extended half-life; anti-inflammatory properties (IVIG mechanism)",
        "dominant_glycans": ["G1F_SA", "G2F_2SA"],
        "pi_shift": -0.3,  # Sialic acid negative charges lower pI
        "mass_shift_per_site": 1897.69,  # G1F+SA average
    },
    "none_aglycosylated": {
        "name": "Aglycosylated (No N-linked Glycans)",
        "half_life_multiplier": 0.50,
        "description": "Aglycosylated antibodies (e.g., E. coli-expressed or N297Q mutant) lack Fc glycans, losing FcRn co-engagement and Fc-gamma receptor binding. Dramatically reduced in-vivo stability and effector function.",
        "clearance_note": "Accelerated clearance — loss of Fc glycan destabilizes CH2 domain, reduces FcRn affinity",
        "dominant_glycans": [],
        "pi_shift": 0.0,
        "mass_shift_per_site": 0.0,
    },
}


# ===========================================================================
# 2. Core PK Prediction: Half-Life
# ===========================================================================

def predict_human_half_life(
    global_pi: float,
    hydrophobicity: float = 0.35,
    liability_density: float = 30.0,
    fcrn_binding_motif_intact: bool = True,
    mw_kda: float = 150.0,
    glycoform_profile: str = "standard_cho",
    assembly_half_life_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """
    Predict human half-life for a therapeutic antibody.

    Uses an empirical penalty model calibrated to published IgG PK data:
      t1/2 = baseline * glyco_modifier * prod(penalty_factors)

    Parameters
    ----------
    global_pi               : Theoretical isoelectric point of the assembled protein
    hydrophobicity          : Hydrophobicity score (0-1 scale, from GRAVY normalization)
    liability_density       : Liability motifs per 1000 assembled residues
    fcrn_binding_motif_intact : Whether FcRn binding site (His310/His435) is intact
    mw_kda                  : Molecular weight in kDa
    glycoform_profile       : Key from GLYCOFORM_PK_MODIFIERS

    Returns
    -------
    dict : {
        "half_life_days": float,
        "baseline_days": float,
        "penalties": list of {"factor": str, "multiplier": float, "reason": str},
        "risk_assessment": "Low" | "Medium" | "High" | "Very High",
        "risk_color": str (hex),
        "clearance_ml_day_kg": float,
        "vd_ml_kg": float,
        "glycoform_impact": dict,
        "recommendations": list of str,
        "summary": str,
    }
    """
    penalties: List[Dict[str, Any]] = []
    t_half = BASELINE_HALF_LIFE_DAYS

    # -- 1. Glycoform modifier ------------------------------------------------
    glyco_info = GLYCOFORM_PK_MODIFIERS.get(glycoform_profile, GLYCOFORM_PK_MODIFIERS["standard_cho"])
    glyco_mult = float(glyco_info["half_life_multiplier"])
    # Safeguard: standard G0F/G1F glycoforms MUST always yield 1.0x
    if glycoform_profile in ("standard_cho", None) or glyco_mult == 1.0:
        glyco_mult = 1.0  # Neutral — no penalty for standard glycosylation
    else:
        penalties.append({
            "factor": "Glycoform Profile",
            "multiplier": glyco_mult,
            "reason": glyco_info["clearance_note"],
        })

    # Apply pI shift from glycoform (e.g., sialylation lowers pI)
    effective_pi = global_pi + glyco_info.get("pi_shift", 0.0)

    # -- 2. pI penalty ---------------------------------------------------------
    pi_mult = 1.0
    pi_reason = ""
    if effective_pi > PI_SEVERE_HIGH:
        # Severe: pI > 9.5 => dramatic clearance increase
        overshoot = effective_pi - PI_SEVERE_HIGH
        pi_mult = max(0.30, 1.0 - 0.15 * overshoot)
        pi_reason = (f"Extreme high pI ({effective_pi:.1f}) — rapid pinocytic uptake "
                     f"and tissue sequestration reduce FcRn-mediated recycling")
    elif effective_pi > PI_OPTIMAL_HIGH:
        # Moderate: 8.5 < pI <= 9.5
        overshoot = effective_pi - PI_OPTIMAL_HIGH
        pi_mult = max(0.50, 1.0 - 0.10 * overshoot)
        pi_reason = (f"Elevated pI ({effective_pi:.1f}) — increased non-specific "
                     f"tissue binding and accelerated clearance")
    elif effective_pi < PI_SEVERE_LOW:
        # Very acidic pI (rare for mAbs)
        undershoot = PI_SEVERE_LOW - effective_pi
        pi_mult = max(0.50, 1.0 - 0.08 * undershoot)
        pi_reason = (f"Extreme low pI ({effective_pi:.1f}) — may affect FcRn binding "
                     f"at endosomal pH 6.0")
    elif effective_pi < PI_OPTIMAL_LOW:
        undershoot = PI_OPTIMAL_LOW - effective_pi
        pi_mult = max(0.70, 1.0 - 0.05 * undershoot)
        pi_reason = f"Low pI ({effective_pi:.1f}) — slightly reduced recycling efficiency"

    if pi_mult < 1.0:
        penalties.append({
            "factor": "Isoelectric Point (pI)",
            "multiplier": round(pi_mult, 3),
            "reason": pi_reason,
        })

    # -- 3. Hydrophobicity penalty ---------------------------------------------
    hydro_mult = 1.0
    if hydrophobicity > HYDROPHOBICITY_THRESHOLD_SEVERE:
        excess = hydrophobicity - HYDROPHOBICITY_THRESHOLD_SEVERE
        hydro_mult = max(0.50, 1.0 - 1.5 * excess)
        penalties.append({
            "factor": "Hydrophobicity",
            "multiplier": round(hydro_mult, 3),
            "reason": (f"High hydrophobicity ({hydrophobicity:.2f}) — elevated "
                       f"non-specific binding, polyreactivity, and aggregation-mediated clearance"),
        })
    elif hydrophobicity > HYDROPHOBICITY_THRESHOLD_MILD:
        excess = hydrophobicity - HYDROPHOBICITY_THRESHOLD_MILD
        hydro_mult = max(0.70, 1.0 - 0.5 * excess)
        penalties.append({
            "factor": "Hydrophobicity",
            "multiplier": round(hydro_mult, 3),
            "reason": (f"Moderate hydrophobicity ({hydrophobicity:.2f}) — some "
                       f"non-specific binding risk"),
        })

    # -- 4. Liability density penalty (aggregation proxy) ----------------------
    liab_mult = 1.0
    if liability_density > LIAB_DENSITY_THRESHOLD_SEVERE:
        excess = (liability_density - LIAB_DENSITY_THRESHOLD_SEVERE) / 100.0
        liab_mult = max(0.60, 1.0 - 0.5 * excess)
        penalties.append({
            "factor": "Liability Density",
            "multiplier": round(liab_mult, 3),
            "reason": (f"High liability density ({liability_density:.1f}/1k aa) — "
                       f"elevated aggregation risk leads to immune complex clearance"),
        })
    elif liability_density > LIAB_DENSITY_THRESHOLD_MILD:
        excess = (liability_density - LIAB_DENSITY_THRESHOLD_MILD) / 100.0
        liab_mult = max(0.80, 1.0 - 0.3 * excess)
        penalties.append({
            "factor": "Liability Density",
            "multiplier": round(liab_mult, 3),
            "reason": (f"Moderate liability density ({liability_density:.1f}/1k aa) — "
                       f"some PTM/aggregation risk"),
        })

    # -- 5. FcRn binding motif -------------------------------------------------
    fcrn_mult = 1.0
    if not fcrn_binding_motif_intact:
        fcrn_mult = 1.0 - FCRN_BINDING_WEIGHT  # 30% reduction
        penalties.append({
            "factor": "FcRn Binding",
            "multiplier": round(fcrn_mult, 3),
            "reason": ("Disrupted FcRn binding motif (His310/His435/Ile253) — "
                       "impaired neonatal Fc receptor recycling"),
        })

    # -- 6. MW penalty (large constructs) --------------------------------------
    mw_mult = 1.0
    if mw_kda > MW_PENALTY_THRESHOLD:
        excess = (mw_kda - MW_PENALTY_THRESHOLD) / 100.0
        mw_mult = max(0.70, 1.0 - 0.10 * excess)
        penalties.append({
            "factor": "Molecular Weight",
            "multiplier": round(mw_mult, 3),
            "reason": (f"Large construct ({mw_kda:.0f} kDa) — reduced tissue "
                       f"penetration and altered biodistribution"),
        })

    # -- 7. Assembly completeness penalty (v7.3.1) ------------------------------
    asm_mult = float(max(0.01, min(1.0, assembly_half_life_multiplier)))
    if asm_mult < 1.0:
        penalties.append({
            "factor": "Assembly Completeness",
            "multiplier": round(asm_mult, 3),
            "reason": (
                f"Incomplete IgG assembly — exposed hydrophobic interfaces cause "
                f"rapid aggregation and impaired FcRn recycling. "
                f"Half-life reduced to {asm_mult:.0%} of baseline."
            ),
        })

    # -- Compute final half-life -----------------------------------------------
    total_multiplier = glyco_mult * pi_mult * hydro_mult * liab_mult * fcrn_mult * mw_mult * asm_mult
    t_half = BASELINE_HALF_LIFE_DAYS * total_multiplier

    # Ensure minimum (biological floor — even fragments have some persistence)
    t_half = max(0.5, round(t_half, 1))

    # -- PK parameters (allometric, 70 kg human) -------------------------------
    # CL = 0.693 * Vd / t1/2   (linear PK approximation)
    # Vd for IgG ~ 3-5 L (central) or ~70-100 mL/kg (incl. tissue)
    vd_ml_kg = 70.0 + max(0, (mw_kda - MW_STANDARD_IgG) * 0.2)
    cl_ml_day_kg = 0.693 * vd_ml_kg / t_half if t_half > 0 else 999.0

    # -- Risk assessment -------------------------------------------------------
    if t_half >= 18:
        risk = "Low"
        risk_color = "#10B981"
    elif t_half >= 12:
        risk = "Medium"
        risk_color = "#F59E0B"
    elif t_half >= 6:
        risk = "High"
        risk_color = "#EF4444"
    else:
        risk = "Very High"
        risk_color = "#991B1B"

    # -- Recommendations -------------------------------------------------------
    recommendations = []
    if pi_mult < 0.85:
        recommendations.append(
            f"Consider charge engineering to bring pI into the optimal range "
            f"({PI_OPTIMAL_LOW}-{PI_OPTIMAL_HIGH}). Current effective pI: {effective_pi:.1f}."
        )
    if hydro_mult < 0.90:
        recommendations.append(
            "Reduce surface hydrophobicity by substituting exposed hydrophobic "
            "residues (V/I/L/F) with polar alternatives (T/S/N/Q) in CDR loops."
        )
    if liab_mult < 0.90:
        recommendations.append(
            "Address high liability density through targeted sequence engineering "
            "to remove deamidation (NG/NS) and oxidation (Met/Trp) hotspots."
        )
    if not fcrn_binding_motif_intact:
        recommendations.append(
            "Critical: FcRn binding motif is disrupted. Verify Fc region integrity "
            "(His310, His435, Ile253) for proper neonatal Fc receptor recycling."
        )
    if glyco_mult < 0.90:
        glyco_name = glyco_info["name"]
        recommendations.append(
            f"Glycoform profile ({glyco_name}) leads to accelerated clearance. "
            f"Consider cell line engineering or media optimization to shift toward "
            f"standard CHO glycosylation (G0F/G1F)."
        )
    if not recommendations:
        recommendations.append(
            "PK profile appears favorable. No major liabilities identified for "
            "in-vivo half-life."
        )

    # -- Summary ---------------------------------------------------------------
    summary = (
        f"Predicted half-life: {t_half:.1f} days (baseline {BASELINE_HALF_LIFE_DAYS:.0f} days, "
        f"{len(penalties)} penalty factor{'s' if len(penalties) != 1 else ''}). "
        f"PK Risk: {risk}. CL = {cl_ml_day_kg:.2f} mL/day/kg."
    )

    result = {
        "half_life_days": t_half,
        "baseline_days": BASELINE_HALF_LIFE_DAYS,
        "effective_pi": round(effective_pi, 2),
        "total_multiplier": round(total_multiplier, 4),
        "penalties": penalties,
        "risk_assessment": risk,
        "risk_color": risk_color,
        "clearance_ml_day_kg": round(cl_ml_day_kg, 3),
        "vd_ml_kg": round(vd_ml_kg, 1),
        "glycoform_impact": {
            "profile": glycoform_profile,
            "name": glyco_info["name"],
            "multiplier": glyco_mult,
            "pi_shift": glyco_info.get("pi_shift", 0.0),
            "mass_shift_per_site": glyco_info.get("mass_shift_per_site", 0),
            "description": glyco_info["description"],
        },
        "recommendations": recommendations,
        "summary": summary,
    }

    log.info("PK prediction: t1/2=%.1f days (%s risk), CL=%.3f mL/day/kg, "
             "effective pI=%.2f, glycoform=%s",
             t_half, risk, cl_ml_day_kg, effective_pi, glycoform_profile)
    return result


# ===========================================================================
# 3. Glycoform Impact on Chromatography
# ===========================================================================

def get_glycoform_pi_shift(glycoform_profile: str) -> float:
    """
    Get the pI shift caused by the selected glycoform profile (PK context).

    Sialylation adds negative charges (NeuAc = sialic acid), which
    lower the protein's isoelectric point. This affects IEX behavior.

    Note: This function reads from GLYCOFORM_PK_MODIFIERS (PK/half-life
    context). A same-name function exists in analytical_twin.py reading
    from HOST_CELL_GLYCOFORM_PROFILES (analytical context). The two dicts
    encode different assumptions — kept intentionally separate.

    Parameters
    ----------
    glycoform_profile : Key from GLYCOFORM_PK_MODIFIERS

    Returns
    -------
    float : pI shift (negative = more acidic)
    """
    info = GLYCOFORM_PK_MODIFIERS.get(glycoform_profile, {})
    return info.get("pi_shift", 0.0)


def get_glycoform_mass_per_site(glycoform_profile: str) -> float:
    """
    Get the glycan mass per N-glycosylation site for the selected profile.

    Parameters
    ----------
    glycoform_profile : Key from GLYCOFORM_PK_MODIFIERS

    Returns
    -------
    float : Mass in Daltons per glycosylation site
    """
    info = GLYCOFORM_PK_MODIFIERS.get(glycoform_profile, {})
    return info.get("mass_shift_per_site", 1444.53)


def get_glycoform_profiles() -> Dict[str, str]:
    """
    Return available glycoform profiles for UI dropdown.

    Returns
    -------
    dict : {key: display_name} for each available profile
    """
    return {
        key: info["name"]
        for key, info in GLYCOFORM_PK_MODIFIERS.items()
    }


def assess_glycoform_pk_impact(
    glycoform_profile: str,
    base_pi: float,
    base_half_life: float = BASELINE_HALF_LIFE_DAYS,
) -> Dict[str, Any]:
    """
    Assess how a glycoform profile affects PK and chromatography.

    Parameters
    ----------
    glycoform_profile : Key from GLYCOFORM_PK_MODIFIERS
    base_pi           : Original pI before glycoform effect
    base_half_life    : Reference half-life

    Returns
    -------
    dict : {
        "profile_name": str,
        "pi_shift": float,
        "effective_pi": float,
        "half_life_multiplier": float,
        "adjusted_half_life": float,
        "chromatography_impact": str,
        "description": str,
    }
    """
    info = GLYCOFORM_PK_MODIFIERS.get(glycoform_profile, GLYCOFORM_PK_MODIFIERS["standard_cho"])
    pi_shift = info.get("pi_shift", 0.0)
    effective_pi = base_pi + pi_shift
    hl_mult = info["half_life_multiplier"]
    adj_hl = round(base_half_life * hl_mult, 1)

    # Chromatography impact description
    if pi_shift < -0.1:
        chrom_impact = (
            f"pI lowered by {abs(pi_shift):.1f} units due to sialic acid negative charges. "
            f"In cation-exchange (CEX) chromatography, the molecule will elute EARLIER "
            f"(at lower salt concentration) due to reduced positive charge at working pH. "
            f"Effective pI: {effective_pi:.2f}."
        )
    elif pi_shift > 0.1:
        chrom_impact = (
            f"pI increased by {pi_shift:.1f} units. "
            f"In CEX, the molecule will elute LATER (higher salt). "
            f"Effective pI: {effective_pi:.2f}."
        )
    else:
        chrom_impact = (
            f"No significant pI shift. CEX retention unchanged. "
            f"Effective pI: {effective_pi:.2f}."
        )

    return {
        "profile_name": info["name"],
        "pi_shift": pi_shift,
        "effective_pi": round(effective_pi, 2),
        "half_life_multiplier": hl_mult,
        "adjusted_half_life": adj_hl,
        "chromatography_impact": chrom_impact,
        "description": info["description"],
        "clearance_note": info["clearance_note"],
        "dominant_glycans": info.get("dominant_glycans", []),
        "mass_per_site_da": info.get("mass_shift_per_site", 1444.53),
    }


# ===========================================================================
# 3B. Assembly Completeness Penalty
# ===========================================================================

def assess_assembly_completeness(
    chains: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Assess whether the antibody assembly is structurally complete.

    A standard IgG requires 2 HC + 2 LC for proper folding. Incomplete
    assemblies (e.g., half-antibody = 1HC+1LC) expose hydrophobic
    CH3 dimerization interfaces, causing severe aggregation and
    reduced in-vivo stability.

    Parameters
    ----------
    chains : List of chain dicts with 'chain_type' and 'copy_number'

    Returns
    -------
    dict : {
        "is_complete_igg": bool,
        "hc_copies": int,
        "lc_copies": int,
        "aggregation_penalty": float (0.0-1.0, added to agg_risk),
        "stability_penalty": float (multiplier on half-life, <1.0 = worse),
        "risk_level": str,
        "explanation": str,
    }
    """
    hc_copies = 0
    lc_copies = 0
    other_copies = 0

    for ch in chains:
        ctype = ch.get("chain_type", "").upper()
        copies = max(1, int(ch.get("copy_number", 1)))
        if ctype in ("HC", "HEAVY"):
            hc_copies += copies
        elif ctype in ("LC", "LIGHT"):
            lc_copies += copies
        else:
            other_copies += copies

    is_complete = (hc_copies == 2 and lc_copies == 2)

    if is_complete:
        return {
            "is_complete_igg": True,
            "hc_copies": hc_copies,
            "lc_copies": lc_copies,
            "aggregation_penalty": 0.0,
            "stability_penalty": 1.0,
            "assembly_half_life_multiplier": 1.0,
            "risk_level": "Low",
            "explanation": (
                "Complete IgG assembly (2HC + 2LC): fully paired CH3 interface, "
                "proper inter-chain disulfide bonding, standard FcRn recycling."
            ),
        }

    # Incomplete assembly — feature-driven risk assessment
    # Instead of hardcoded penalties, compute biophysical feature adjustments
    # that the scoring engine uses to predict risk naturally.
    #
    # Key principle: unpaired interfaces expose hydrophobic surface area.
    # We estimate the ADDITIONAL exposed hydrophobicity and liability density
    # that would result from incomplete assembly, then let the scoring model
    # (predict_human_half_life) handle the PK impact through its existing
    # penalty framework.

    # Estimate exposed hydrophobic surface area (fraction of total)
    # CH3 dimer interface: ~1800 Å² hydrophobic surface
    # VH-VL interface: ~700 Å² hydrophobic surface
    # Total IgG surface: ~60,000 Å²
    exposed_ch3 = 0.0   # fraction of hydrophobic surface newly exposed
    exposed_vhvl = 0.0
    missing_fc = False

    # -----------------------------------------------------------------------
    # Direct half-life multiplier for incomplete assemblies
    # (v7.3.1 — replaces ineffective feature-delta approach)
    #
    # Half-antibody (1HC+1LC): exposed CH3 interface causes rapid
    # aggregation and immune-complex clearance.  Real-world t½ ~ 2-5 days.
    # Multiplier 0.15 × 21 days → ~3.2 days.
    #
    # Single HC (no LC): CH3 + VH/CL unpaired → even worse.
    # No HC: Fc absent → no FcRn recycling → t½ < 1 day.
    # -----------------------------------------------------------------------
    assembly_mult = 1.0  # default: no penalty

    if hc_copies == 1 and lc_copies == 1:
        # Half-antibody: one CH3 interface unpaired
        exposed_ch3 = 1800.0 / 60000.0
        assembly_mult = 0.15   # ~3.2 days
        risk = "Very High"
        explanation = (
            "Half-antibody (1HC + 1LC): the unpaired CH3 domain exposes "
            "~1800 Å² of hydrophobic surface, causing rapid aggregation "
            "and dramatically reduced in-vivo half-life (~3 days vs ~21 days "
            "for complete IgG). FcRn homodimer recycling is impaired."
        )
    elif hc_copies == 1 and lc_copies == 0:
        exposed_ch3 = 1800.0 / 60000.0
        exposed_vhvl = 700.0 / 60000.0
        assembly_mult = 0.10   # ~2.1 days
        risk = "Very High"
        explanation = (
            "Single heavy chain only: both CH3 and VH-CL interfaces are "
            "unpaired, exposing ~2500 Å² additional hydrophobic surface. "
            "Predicted in-vivo half-life ~2 days."
        )
    elif hc_copies == 0:
        missing_fc = True
        assembly_mult = 0.05   # ~1 day
        risk = "High"
        explanation = (
            "No heavy chains detected: this is not an IgG assembly. "
            "Lacks Fc-mediated FcRn recycling entirely. "
            "Predicted in-vivo half-life < 1 day."
        )
    else:
        # Non-standard stoichiometry — moderate exposure
        unpaired_fraction = abs(hc_copies - 2) + abs(lc_copies - 2)
        exposed_ch3 = (unpaired_fraction * 900.0) / 60000.0
        assembly_mult = max(0.15, 1.0 - 0.25 * unpaired_fraction)
        risk = "Medium" if unpaired_fraction <= 2 else "High"
        explanation = (
            f"Non-standard stoichiometry ({hc_copies}HC + {lc_copies}LC): "
            f"assembly may have unpaired interfaces. "
            f"Half-life reduced to {assembly_mult:.0%} of baseline."
        )

    # Feature adjustments (added to base computed features by caller)
    hydrophobicity_delta = exposed_ch3 + exposed_vhvl  # ~0.03-0.04
    liability_density_delta = hydrophobicity_delta * 200.0  # proportional impact

    return {
        "is_complete_igg": False,
        "hc_copies": hc_copies,
        "lc_copies": lc_copies,
        "hydrophobicity_delta": round(hydrophobicity_delta, 4),
        "liability_density_delta": round(liability_density_delta, 2),
        "assembly_half_life_multiplier": round(assembly_mult, 3),
        "missing_fc": missing_fc,
        "risk_level": risk,
        "explanation": explanation,
    }


# ===========================================================================
# 4. FcRn Binding Motif Check
# ===========================================================================

def check_fcrn_binding_motif(sequence: str) -> Dict[str, Any]:
    """
    Check integrity of FcRn binding motif in the Fc region.

    The FcRn binding interface involves key residues at the CH2-CH3
    junction: His310, His435, Ile253 (EU numbering).

    For a full-length IgG1 HC (~450 aa), the Fc starts at ~position 231.
    We search for the conserved pattern in the expected region.

    This is a simplified heuristic check — not a full structural analysis.

    Parameters
    ----------
    sequence : Heavy chain or full antibody sequence

    Returns
    -------
    dict : {
        "intact": bool,
        "his_count_in_fc": int,
        "fc_region_detected": bool,
        "details": str,
    }
    """
    seq = sequence.upper()

    # Heuristic: for HC > 400 aa, assume Fc starts around position 230
    if len(seq) > 400:
        fc_region = seq[220:]
    elif len(seq) > 200:
        fc_region = seq[len(seq) // 2:]
    else:
        fc_region = seq

    # Count His residues in Fc (should be >= 6 for proper FcRn binding)
    his_count = fc_region.count("H")
    # Check for CPPC hinge motif (IgG1 indicator)
    has_hinge = "CPPC" in seq or "CPPCP" in seq

    fc_detected = len(seq) > 400 or has_hinge

    # Simplified check: if we have enough His in Fc and a proper hinge,
    # we assume FcRn binding is intact
    intact = his_count >= 4 and (fc_detected or len(seq) > 300)

    details = (
        f"Fc region {'detected' if fc_detected else 'not clearly detected'}. "
        f"His residues in Fc: {his_count}. "
        f"Hinge motif (CPPC): {'present' if has_hinge else 'absent'}. "
        f"FcRn binding: {'likely intact' if intact else 'potentially disrupted'}."
    )

    return {
        "intact": intact,
        "his_count_in_fc": his_count,
        "fc_region_detected": fc_detected,
        "has_hinge_motif": has_hinge,
        "details": details,
    }


# ===========================================================================
# __main__: Test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    _passed = 0
    _total = 7

    print("=" * 60)
    print("  Preclinical Twin v1.0 — Self-Test (with assertions)")
    print("=" * 60)

    # Test 1: Standard IgG1 — good properties → Low/Medium risk, t1/2 near baseline
    print("\n--- Test 1: Standard IgG1 (favorable) ---")
    pk1 = predict_human_half_life(
        global_pi=7.8, hydrophobicity=0.30,
        liability_density=35.0, fcrn_binding_motif_intact=True,
        mw_kda=148.0, glycoform_profile="standard_cho",
    )
    assert pk1["half_life_days"] >= 18.0, f"Good IgG1 t1/2 too low: {pk1['half_life_days']}"
    assert pk1["risk_assessment"] in ("Low", "Medium"), f"Good IgG1 risk unexpectedly high: {pk1['risk_assessment']}"
    assert pk1["clearance_ml_day_kg"] > 0, "Clearance must be positive"
    assert "half_life_days" in pk1 and "summary" in pk1, "Missing required keys"
    print(f"  Half-life: {pk1['half_life_days']} days, Risk: {pk1['risk_assessment']}")
    _passed += 1

    # Test 2: High pI mAb → should have penalties and shorter t1/2
    print("\n--- Test 2: High pI mAb ---")
    pk2 = predict_human_half_life(
        global_pi=9.5, hydrophobicity=0.45,
        liability_density=65.0, fcrn_binding_motif_intact=True,
        mw_kda=150.0, glycoform_profile="standard_cho",
    )
    assert pk2["half_life_days"] < pk1["half_life_days"], "High pI should reduce t1/2"
    assert pk2["risk_assessment"] in ("Medium", "High", "Very High"), f"Expected elevated risk: {pk2['risk_assessment']}"
    assert len(pk2["penalties"]) >= 2, "Should have multiple penalties (pI + hydrophobicity + liability)"
    print(f"  Half-life: {pk2['half_life_days']} days, Risk: {pk2['risk_assessment']}, Penalties: {len(pk2['penalties'])}")
    _passed += 1

    # Test 3: High-mannose glycoform → faster clearance
    print("\n--- Test 3: High-Mannose Glycoform ---")
    pk3 = predict_human_half_life(
        global_pi=7.5, hydrophobicity=0.30,
        liability_density=30.0, fcrn_binding_motif_intact=True,
        mw_kda=148.0, glycoform_profile="high_mannose",
    )
    assert pk3["half_life_days"] < 21.0, f"High-mannose should reduce t1/2: {pk3['half_life_days']}"
    assert "glycoform_impact" in pk3, "Missing glycoform_impact"
    print(f"  Half-life: {pk3['half_life_days']} days, Risk: {pk3['risk_assessment']}")
    _passed += 1

    # Test 4: Sialylated glycoform → extended or neutral t1/2
    print("\n--- Test 4: Highly Sialylated ---")
    pk4 = predict_human_half_life(
        global_pi=8.0, hydrophobicity=0.30,
        liability_density=30.0, fcrn_binding_motif_intact=True,
        mw_kda=148.0, glycoform_profile="highly_sialylated",
    )
    assert pk4["half_life_days"] >= 20.0, f"Sialylated should not drastically reduce t1/2: {pk4['half_life_days']}"
    assert "effective_pi" in pk4, "Missing effective_pi key"
    print(f"  Half-life: {pk4['half_life_days']} days, Effective pI: {pk4['effective_pi']}")
    _passed += 1

    # Test 5: Disrupted FcRn binding → severe penalty
    print("\n--- Test 5: Disrupted FcRn Binding ---")
    pk5 = predict_human_half_life(
        global_pi=7.5, hydrophobicity=0.30,
        liability_density=30.0, fcrn_binding_motif_intact=False,
        mw_kda=148.0, glycoform_profile="standard_cho",
    )
    assert pk5["half_life_days"] < 15.0, f"Disrupted FcRn should drastically reduce t1/2: {pk5['half_life_days']}"
    assert any("FcRn" in p["factor"] for p in pk5["penalties"]), "Should have FcRn penalty"
    print(f"  Half-life: {pk5['half_life_days']} days, Risk: {pk5['risk_assessment']}")
    _passed += 1

    # Test 6: Glycoform chromatography impact — all profiles return valid dicts
    print("\n--- Test 6: Glycoform Chromatography Impact ---")
    for profile_key in GLYCOFORM_PK_MODIFIERS:
        impact = assess_glycoform_pk_impact(profile_key, base_pi=8.0)
        assert "profile_name" in impact and "pi_shift" in impact, f"Missing keys for {profile_key}"
        assert isinstance(impact["effective_pi"], float), f"effective_pi not float for {profile_key}"
    print(f"  All {len(GLYCOFORM_PK_MODIFIERS)} profiles validated")
    _passed += 1

    # Test 7: Available profiles — non-empty
    print("\n--- Test 7: Available Glycoform Profiles ---")
    profiles = get_glycoform_profiles()
    assert len(profiles) >= 4, f"Expected ≥4 glycoform profiles, got {len(profiles)}"
    assert "standard_cho" in profiles, "Missing standard_cho profile"
    print(f"  {len(profiles)} profiles available")
    _passed += 1

    print(f"\npreclinical_twin selftest: {_passed}/{_total} passed")
