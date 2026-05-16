"""
validation_planner.py  ·  ProtePilot — Milestone 8 → Phase 3A
===========================================================
Analytical Validation Plan Generator — Molecule-Class-Aware

Version   : 2.0
Author    : Di (ProtePilot)
Depends   : molecule_classifier (optional, for MoleculeClass enum)

Purpose
------------------------------------------------------------
Translates developability risk scores AND molecule class into
a recommended analytical testing plan. Different molecule formats
require different analytical strategies:

  - Canonical mAb  → standard ICH Q6B panel
  - Bispecific      → + species purity, mispairing assays
  - ADC             → + DAR, free drug, linker stability
  - Fc-fusion       → + fusion junction integrity, glycan profiling
  - Single-domain   → + oligomerization, renal clearance markers
  - Peptide         → + chemical stability, HPLC purity

Assay Catalog
------------------------------------------------------------
Includes standard methods from ICH Q6B, FDA CMC guidance,
and industry best practices for mAb characterization:
  - SEC, DLS, DSF, DSC, CE-SDS, icIEF, MAM, Viscometry, etc.
  - Format-specific: HIC, rCE-SDS, SPR, peptide mapping

References
------------------------------------------------------------
  ICH Q6B: Specifications for Biotechnological Products
  USP <1032>: Design and Development of Biological Assays
  Jarasch et al., J. Pharm. Sci. 104:1885 (2015) — Developability assessment
  ICH S9: Nonclinical evaluation for ADC safety
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("ProtePilot.ValidationPlanner")


# ===========================================================================
# 1. Assay Catalog
# ===========================================================================

ASSAY_CATALOG = {
    "SEC": {
        "full_name": "Size-Exclusion Chromatography (SEC-HPLC)",
        "measures": "Aggregation, fragments, monomer purity",
        "risk_category": "aggregation",
        "threshold_key": "agg_risk",
        "threshold_value": 0.20,
        "priority": "critical",
        "typical_timeline": "Week 1-2",
        "description": (
            "Gold-standard method for quantifying monomer, dimer, and higher-order "
            "aggregates. Essential for all mAb candidates. Run under native conditions "
            "(PBS, pH 7.4) with UV detection at 280 nm."
        ),
        "always_required": True,
    },
    "DLS": {
        "full_name": "Dynamic Light Scattering",
        "measures": "Hydrodynamic diameter, polydispersity, onset of aggregation",
        "risk_category": "aggregation",
        "threshold_key": "agg_risk",
        "threshold_value": 0.40,
        "priority": "high",
        "typical_timeline": "Week 1-2",
        "description": (
            "Non-invasive measurement of particle size distribution. Useful for "
            "detecting sub-visible aggregates and monitoring colloidal stability. "
            "Recommended when aggregation risk is elevated."
        ),
        "always_required": False,
    },
    "DSF": {
        "full_name": "Differential Scanning Fluorimetry (nanoDSF)",
        "measures": "Thermal unfolding temperature (Tm), colloidal stability (Tagg)",
        "risk_category": "stability",
        "threshold_key": "stability",
        "threshold_value": 0.75,
        "threshold_direction": "below",
        "priority": "high",
        "typical_timeline": "Week 1-3",
        "description": (
            "High-throughput thermal stability screening. Measures Tm (onset of unfolding) "
            "and Tagg (onset of aggregation). Ideal for formulation screening. "
            "Target: Tm > 65C for IgG1."
        ),
        "always_required": False,
    },
    "DSC": {
        "full_name": "Differential Scanning Calorimetry",
        "measures": "Thermodynamic stability (Tm1, Tm2, Tm3), enthalpy of unfolding",
        "risk_category": "stability",
        "threshold_key": "stability",
        "threshold_value": 0.60,
        "threshold_direction": "below",
        "priority": "medium",
        "typical_timeline": "Week 2-4",
        "description": (
            "Detailed thermodynamic characterization of domain-level unfolding. "
            "Provides Tm values for CH2, Fab, and CH3 domains. Recommended when "
            "stability concerns are significant."
        ),
        "always_required": False,
    },
    "CE-SDS": {
        "full_name": "Capillary Electrophoresis with SDS (CE-SDS)",
        "measures": "Purity, fragmentation, non-glycosylated heavy chain",
        "risk_category": "purity",
        "threshold_key": None,
        "threshold_value": None,
        "priority": "critical",
        "typical_timeline": "Week 1-2",
        "description": (
            "Separates intact/reduced antibody species by molecular weight under "
            "denaturing conditions. Detects heavy chain, light chain, and fragment "
            "impurities. Required for all mAb candidates per ICH Q6B."
        ),
        "always_required": True,
    },
    "icIEF": {
        "full_name": "Imaged Capillary Isoelectric Focusing (icIEF)",
        "measures": "Charge heterogeneity, acidic/basic variants, deamidation",
        "risk_category": "charge_heterogeneity",
        "threshold_key": None,
        "threshold_value": None,
        "priority": "critical",
        "typical_timeline": "Week 1-3",
        "description": (
            "High-resolution charge variant analysis. Quantifies acidic, main, "
            "and basic species. Essential for monitoring deamidation, oxidation, "
            "and other charge-altering modifications. Required for lot release."
        ),
        "always_required": True,
    },
    "MAM": {
        "full_name": "Multi-Attribute Method (LC-MS/MS)",
        "measures": "Post-translational modifications (PTMs), glycosylation, oxidation, deamidation",
        "risk_category": "ptm",
        "threshold_key": "ptm_sites",
        "threshold_value": 3,
        "priority": "high",
        "typical_timeline": "Week 2-4",
        "description": (
            "Mass spectrometry-based multi-attribute monitoring. Simultaneously "
            "quantifies oxidation, deamidation, glycosylation, and other PTMs "
            "at the peptide level. Replacing multiple conventional methods."
        ),
        "always_required": False,
    },
    "Viscometry": {
        "full_name": "Cone-and-Plate Viscometry / Capillary Viscometry",
        "measures": "Solution viscosity at target concentration",
        "risk_category": "viscosity",
        "threshold_key": "viscosity_risk",
        "threshold_value": 0.30,
        "priority": "high",
        "typical_timeline": "Week 2-4",
        "description": (
            "Measures dynamic viscosity of concentrated antibody solutions. "
            "Critical for subcutaneous delivery (target <20 cP at >100 mg/mL). "
            "Recommended when viscosity risk is elevated."
        ),
        "always_required": False,
    },
    "SE-AUC": {
        "full_name": "Sedimentation Velocity Analytical Ultracentrifugation",
        "measures": "Aggregate characterization, stoichiometry, reversibility",
        "risk_category": "aggregation",
        "threshold_key": "agg_risk",
        "threshold_value": 0.60,
        "priority": "medium",
        "typical_timeline": "Week 3-6",
        "description": (
            "Orthogonal method for aggregate characterization without matrix effects. "
            "Provides sedimentation coefficient distribution, revealing aggregate "
            "species and their relative abundance. Recommended for high-risk candidates."
        ),
        "always_required": False,
    },
    "FcRn_binding": {
        "full_name": "FcRn Binding Assay (SPR/BLI)",
        "measures": "Neonatal Fc receptor binding, half-life prediction",
        "risk_category": "pharmacokinetics",
        "threshold_key": "stability",
        "threshold_value": 0.65,
        "threshold_direction": "below",
        "priority": "medium",
        "typical_timeline": "Week 2-4",
        "description": (
            "Surface plasmon resonance or bio-layer interferometry measurement "
            "of FcRn binding kinetics. Predicts in vivo half-life. Important when "
            "stability modifications may affect Fc region integrity."
        ),
        "always_required": False,
    },
}


# ===========================================================================
# 1b. Format-Specific Assay Extensions
# ===========================================================================

FORMAT_SPECIFIC_ASSAYS: Dict[str, Dict[str, Any]] = {
    # ── Bispecific-specific ──────────────────────────────
    "HIC_species": {
        "full_name": "Hydrophobic Interaction Chromatography (Species Purity)",
        "measures": "AA/AB/BB homodimer vs heterodimer ratio",
        "risk_category": "species_purity",
        "priority": "critical",
        "typical_timeline": "Week 1-2",
        "molecule_classes": ["bispecific"],
        "description": (
            "Separates homodimer (AA, BB) from heterodimer (AB) species based on "
            "surface hydrophobicity differences. Essential for bispecific quality "
            "control — target heterodimer purity ≥ 90%."
        ),
        "explanation": (
            "Bispecific antibodies are produced as a mixture of three species: "
            "two homodimers (AA, BB) and the desired heterodimer (AB). HIC resolves "
            "these species to confirm correct assembly and monitor mispairing."
        ),
    },
    "rCE-SDS_bispecific": {
        "full_name": "Reduced CE-SDS (Bispecific Chain Integrity)",
        "measures": "Individual chain identity and stoichiometry",
        "risk_category": "species_purity",
        "priority": "high",
        "typical_timeline": "Week 1-3",
        "molecule_classes": ["bispecific"],
        "description": (
            "Reduced CE-SDS confirms that both distinct heavy chains (or half-antibodies) "
            "are present in the correct stoichiometric ratio. Detects incomplete assembly."
        ),
        "explanation": (
            "Unlike conventional mAbs with identical heavy chains, bispecifics require "
            "verification that both distinct chains are present and correctly paired."
        ),
    },
    "SPR_dual_binding": {
        "full_name": "Dual-Target Binding (SPR/BLI)",
        "measures": "Simultaneous binding to both target antigens",
        "risk_category": "potency",
        "priority": "critical",
        "typical_timeline": "Week 2-4",
        "molecule_classes": ["bispecific"],
        "description": (
            "Surface plasmon resonance or BLI measurement confirming simultaneous "
            "binding to both target antigens. Validates functional bispecificity."
        ),
        "explanation": (
            "The core therapeutic premise of a bispecific is simultaneous engagement "
            "of two targets. This bridging assay is critical to confirm that both "
            "binding arms are functional and not sterically blocked."
        ),
    },

    # ── ADC-specific ─────────────────────────────────────
    "DAR_analysis": {
        "full_name": "Drug-to-Antibody Ratio (DAR) Analysis (HIC/RP-HPLC)",
        "measures": "Average DAR, DAR distribution, naked antibody fraction",
        "risk_category": "conjugation",
        "priority": "critical",
        "typical_timeline": "Week 1-2",
        "molecule_classes": ["adc"],
        "description": (
            "Measures the number of drug molecules conjugated per antibody. "
            "DAR uniformity is critical for consistent potency and safety. "
            "Target: DAR within ±0.5 of nominal, naked Ab < 5%."
        ),
        "explanation": (
            "ADC efficacy depends on delivering the right amount of cytotoxic payload. "
            "Under-conjugated species lack potency; over-conjugated species increase "
            "toxicity and aggregation. DAR must be tightly controlled."
        ),
    },
    "free_drug_assay": {
        "full_name": "Free Drug / Unconjugated Payload Assay (RP-HPLC)",
        "measures": "Free cytotoxic payload, linker-drug species",
        "risk_category": "conjugation",
        "priority": "high",
        "typical_timeline": "Week 1-3",
        "molecule_classes": ["adc"],
        "description": (
            "Quantifies unconjugated cytotoxic drug in the ADC preparation. "
            "Free drug contributes to off-target toxicity and must be ≤ 1%."
        ),
        "explanation": (
            "Even small amounts of free cytotoxic drug create significant safety "
            "concerns. This assay monitors conjugation efficiency and linker integrity."
        ),
    },
    "linker_stability": {
        "full_name": "Linker Stability Assessment (Plasma Incubation + LC-MS)",
        "measures": "Linker cleavage rate, drug release kinetics in plasma",
        "risk_category": "conjugation",
        "priority": "high",
        "typical_timeline": "Week 2-4",
        "molecule_classes": ["adc"],
        "description": (
            "Assesses stability of the linker-payload in human and mouse plasma. "
            "Premature drug release in circulation causes systemic toxicity."
        ),
        "explanation": (
            "The linker must remain stable in circulation but release payload at the "
            "target site. Premature cleavage negates the ADC targeting advantage."
        ),
    },

    # ── Fc-fusion-specific ───────────────────────────────
    "junction_integrity": {
        "full_name": "Fusion Junction Integrity (Peptide Mapping LC-MS)",
        "measures": "Cleavage at fusion junction, junction-related modifications",
        "risk_category": "purity",
        "priority": "high",
        "typical_timeline": "Week 2-4",
        "molecule_classes": ["fc_fusion"],
        "description": (
            "Targeted peptide mapping around the fusion junction to detect cleavage, "
            "deamidation, or other modifications at the engineered junction site."
        ),
        "explanation": (
            "Fc-fusion proteins have an engineered junction between the therapeutic "
            "domain and Fc. This junction can be a hotspot for proteolytic cleavage "
            "and chemical modifications that affect potency."
        ),
    },
    "glycan_profiling": {
        "full_name": "Extended Glycan Profiling (HILIC-FLR-MS)",
        "measures": "N-glycan fingerprint, sialylation, O-glycan analysis",
        "risk_category": "glycosylation",
        "priority": "high",
        "typical_timeline": "Week 2-4",
        "molecule_classes": ["fc_fusion"],
        "description": (
            "Comprehensive glycan analysis including both Fc and fusion domain "
            "glycosylation. Fc-fusion proteins often have additional glycosylation "
            "sites requiring extended characterization."
        ),
        "explanation": (
            "Many Fc-fusion proteins carry glycosylation on the therapeutic domain "
            "in addition to the Fc. These glycans can affect activity, clearance, "
            "and immunogenicity — requiring more detailed profiling than standard mAbs."
        ),
    },

    # ── Fusion protein (non-Fc) — tandem scFv, FabFab, etc. ──
    "linker_domain_integrity": {
        "full_name": "Linker & Domain Integrity (Peptide Mapping LC-MS)",
        "measures": "Linker clipping, domain boundary modifications, inter-domain cleavage",
        "risk_category": "purity",
        "priority": "high",
        "typical_timeline": "Week 2-4",
        "molecule_classes": ["fusion_protein"],
        "description": (
            "Targeted peptide mapping at linker regions and domain junctions to detect "
            "proteolytic clipping, deamidation, or oxidation at engineered boundaries."
        ),
        "explanation": (
            "Multi-domain fusion proteins (tandem scFv, FabFab, etc.) rely on "
            "flexible linkers (e.g., (G4S)n) that can be vulnerable to clipping "
            "under manufacturing stress. Early detection prevents late-stage failures."
        ),
    },
    "domain_interaction": {
        "full_name": "Domain Interaction & Folding Assessment (DSC/DSF + SEC-MALS)",
        "measures": "Thermal unfolding domains, inter-domain cooperativity, aggregation onset",
        "risk_category": "stability",
        "priority": "high",
        "typical_timeline": "Week 1-3",
        "molecule_classes": ["fusion_protein"],
        "description": (
            "Differential scanning analysis to resolve individual domain Tm values "
            "and SEC-MALS to confirm proper folding and monomer integrity."
        ),
        "explanation": (
            "Fusion proteins can exhibit independent domain unfolding, misfolding, "
            "or domain-swapped aggregation. Multi-domain DSC resolves whether each "
            "domain folds cooperatively, critical for shelf-life prediction."
        ),
    },

    # ── Single-domain / nanobody-specific ────────────────
    "SEC_oligomer": {
        "full_name": "SEC-MALS (Multi-Angle Light Scattering)",
        "measures": "Oligomeric state, absolute molar mass in solution",
        "risk_category": "aggregation",
        "priority": "high",
        "typical_timeline": "Week 1-2",
        "molecule_classes": ["single_domain"],
        "description": (
            "SEC coupled with MALS to determine absolute molar mass and oligomeric "
            "state. Single-domain antibodies can form reversible dimers/trimers "
            "that standard SEC may not fully resolve."
        ),
        "explanation": (
            "Due to their small size (~15 kDa), nanobodies/VHHs are prone to "
            "reversible self-association that may not be captured by standard SEC. "
            "MALS provides absolute mass determination independent of calibration."
        ),
    },
    "renal_clearance": {
        "full_name": "In Vitro Renal Filtration Assessment",
        "measures": "Apparent hydrodynamic radius, albumin binding",
        "risk_category": "pharmacokinetics",
        "priority": "medium",
        "typical_timeline": "Week 2-4",
        "molecule_classes": ["single_domain"],
        "description": (
            "Evaluates risk of rapid renal clearance based on hydrodynamic radius. "
            "Molecules < 60 kDa are subject to renal filtration (t½ < 2 hours "
            "for unmodified nanobodies)."
        ),
        "explanation": (
            "Single-domain antibodies (~15 kDa) fall well below the renal filtration "
            "threshold (~60 kDa), leading to rapid clearance. Half-life extension "
            "strategies (PEG, albumin binding, Fc fusion) must be verified early."
        ),
    },

    # ── Peptide-specific ─────────────────────────────────
    "RP_HPLC_purity": {
        "full_name": "Reversed-Phase HPLC Purity Analysis",
        "measures": "Chemical purity, related substances, degradation products",
        "risk_category": "purity",
        "priority": "critical",
        "typical_timeline": "Week 1-2",
        "molecule_classes": ["peptide"],
        "description": (
            "Gold-standard purity method for peptides. Resolves synthesis-related "
            "impurities (deletion, truncation, modification products) and degradation "
            "products with high resolution."
        ),
        "explanation": (
            "Peptides are typically produced by chemical synthesis, which introduces "
            "impurity profiles very different from recombinant proteins. RP-HPLC is "
            "the primary purity method replacing SEC/CE-SDS used for larger biologics."
        ),
    },
    "chemical_stability": {
        "full_name": "Forced Degradation / Chemical Stability Panel",
        "measures": "Oxidation, deamidation, racemization, diketopiperazine formation",
        "risk_category": "stability",
        "priority": "high",
        "typical_timeline": "Week 1-3",
        "molecule_classes": ["peptide"],
        "description": (
            "Accelerated degradation under stress conditions (heat, light, pH, peroxide) "
            "to identify primary degradation pathways. Essential for defining storage "
            "conditions and shelf life."
        ),
        "explanation": (
            "Peptides are more susceptible to chemical degradation than larger proteins "
            "due to their exposed backbone. Identifying degradation hotspots early "
            "guides formulation and sequence optimization."
        ),
    },
    "MS_identity": {
        "full_name": "Intact Mass Spectrometry (ESI-MS / MALDI-TOF)",
        "measures": "Molecular weight confirmation, disulfide mapping",
        "risk_category": "identity",
        "priority": "critical",
        "typical_timeline": "Week 1-2",
        "molecule_classes": ["peptide"],
        "description": (
            "Accurate mass determination for identity confirmation. For cyclic or "
            "disulfide-containing peptides, also confirms correct cyclization / "
            "disulfide connectivity."
        ),
        "explanation": (
            "Peptide synthesis can produce species with correct sequence but incorrect "
            "disulfide pairing. Intact mass and fragmentation MS confirm both identity "
            "and correct post-synthetic modifications."
        ),
    },
}


# ===========================================================================
# 1c. Molecule-Class Assay Exclusions (assays NOT relevant for certain types)
# ===========================================================================

# Assays from the standard catalog that should be skipped for certain molecule classes
_ASSAY_EXCLUSIONS: Dict[str, List[str]] = {
    "peptide": ["SEC", "CE-SDS", "SE-AUC", "FcRn_binding"],  # peptides use RP-HPLC instead
    "single_domain": ["FcRn_binding"],  # no Fc region (unless Fc-fused)
    "fusion_protein": ["FcRn_binding"],  # no Fc region — no FcRn recycling
}

# Which standard assays get replaced and by what (for explanation text)
_ASSAY_REPLACEMENTS: Dict[str, Dict[str, str]] = {
    "peptide": {
        "SEC": "Replaced by RP-HPLC for purity analysis (peptides < 10 kDa)",
        "CE-SDS": "Replaced by RP-HPLC and intact MS (peptides analyzed differently than proteins)",
    },
}


# ===========================================================================
# 2. Validation Plan Generator (v2.0 — molecule-class-aware)
# ===========================================================================

def generate_validation_plan(
    risk_scores: Dict[str, float],
    intent: Optional[Dict[str, Any]] = None,
    include_timeline: bool = True,
    molecule_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an analytical validation plan from developability risk scores.

    Maps risk predictions to specific bioanalytical assays based on
    defined thresholds, regulatory requirements, AND molecule class.

    Parameters
    ----------
    risk_scores     : Dict with keys: agg_risk, stability, viscosity_risk (all in [0,1])
    intent          : Parsed intent dict (optional, for PTM site info)
    include_timeline: Whether to include recommended timeline
    molecule_class  : str, e.g. "canonical_mab", "bispecific", "adc", "peptide", etc.

    Returns
    -------
    dict with:
        - required_assays: List of always-required assays
        - risk_triggered_assays: List of risk-triggered assays
        - format_specific_assays: List of molecule-format-specific assays
        - excluded_assays: List of assays excluded for this format (with reason)
        - all_assays: Combined list
        - total_assays: Total count
        - risk_summary: Human-readable risk summary
        - format_note: Molecule-class-specific guidance note
        - timeline: Recommended testing timeline (if requested)
        - recommendations: High-level recommendations
    """
    agg = risk_scores.get("agg_risk", 0.2)
    stab = risk_scores.get("stability", 0.8)
    visc = risk_scores.get("viscosity_risk", 0.15)
    mol_cls = molecule_class or "canonical_mab"

    # Count PTM sites from intent
    ptm_sites = 0
    if intent:
        ptm_sites = intent.get("deam_sites", 1) + intent.get("ox_sites", 1)
        # Also check chain_analyses for more detailed PTM counts
        for ca in intent.get("chain_analyses", []):
            liab = ca.get("liabilities", {})
            ptm_sites = max(ptm_sites,
                           liab.get("deamidation_count", 0) + liab.get("met_count", 0))

    # Determine which standard assays to exclude for this molecule class
    exclusions = set(_ASSAY_EXCLUSIONS.get(mol_cls, []))
    replacements = _ASSAY_REPLACEMENTS.get(mol_cls, {})

    required_assays: List[Dict[str, Any]] = []
    risk_triggered: List[Dict[str, Any]] = []
    excluded_assays: List[Dict[str, Any]] = []

    for assay_id, assay in ASSAY_CATALOG.items():
        # Check if this assay is excluded for this molecule class
        if assay_id in exclusions:
            reason = replacements.get(assay_id, f"Not applicable for {mol_cls} format")
            excluded_assays.append({
                "id": assay_id,
                "name": assay["full_name"],
                "reason": reason,
            })
            continue

        assay_entry = {
            "id": assay_id,
            "name": assay["full_name"],
            "measures": assay["measures"],
            "priority": assay["priority"],
            "timeline": assay.get("typical_timeline", "TBD"),
            "description": assay["description"],
            "risk_category": assay["risk_category"],
        }

        if assay["always_required"]:
            assay_entry["trigger_reason"] = "Required per ICH Q6B / standard panel"
            required_assays.append(assay_entry)
            continue

        # Check risk-based triggers
        triggered = False
        trigger_reason = ""

        threshold_key = assay.get("threshold_key")
        threshold_val = assay.get("threshold_value")
        direction = assay.get("threshold_direction", "above")

        if threshold_key and threshold_val is not None:
            if threshold_key == "agg_risk" and direction == "above":
                if agg > threshold_val:
                    triggered = True
                    trigger_reason = f"Aggregation risk ({agg:.2f}) > threshold ({threshold_val})"
            elif threshold_key == "stability" and direction == "below":
                if stab < threshold_val:
                    triggered = True
                    trigger_reason = f"Stability ({stab:.2f}) < threshold ({threshold_val})"
            elif threshold_key == "viscosity_risk" and direction == "above":
                if visc > threshold_val:
                    triggered = True
                    trigger_reason = f"Viscosity risk ({visc:.2f}) > threshold ({threshold_val})"
            elif threshold_key == "ptm_sites":
                if ptm_sites > threshold_val:
                    triggered = True
                    trigger_reason = f"PTM sites ({ptm_sites}) > threshold ({threshold_val})"

        if triggered:
            assay_entry["trigger_reason"] = trigger_reason
            risk_triggered.append(assay_entry)

    # ── Add format-specific assays ──────────────────────────────────────
    format_specific: List[Dict[str, Any]] = []
    for assay_id, assay in FORMAT_SPECIFIC_ASSAYS.items():
        if mol_cls in assay.get("molecule_classes", []):
            format_specific.append({
                "id": assay_id,
                "name": assay["full_name"],
                "measures": assay["measures"],
                "priority": assay["priority"],
                "timeline": assay.get("typical_timeline", "TBD"),
                "description": assay["description"],
                "risk_category": assay["risk_category"],
                "explanation": assay.get("explanation", ""),
                "trigger_reason": f"Required for {mol_cls} format",
            })

    # ── Unknown class + multi-chain safety net ────────────────────────
    n_chains = len(intent.get("chains", [])) if intent else 0
    if mol_cls == "unknown" and n_chains > 2:
        _safety_net_assays = [
            {
                "id": "intact_mass_unknown",
                "name": "Intact Mass Confirmation (Multi-Chain Safety Net)",
                "measures": "Confirm expected MW for multi-chain assembly",
                "priority": "high",
                "timeline": "Week 1-2",
                "description": "Intact mass spectrometry to verify assembly stoichiometry.",
                "risk_category": "identity",
                "explanation": (
                    "Classification unknown with >2 chains detected. Intact mass "
                    "confirms correct assembly before relying on downstream predictions."
                ),
                "trigger_reason": f"Unknown class with {n_chains} chains — assembly verification required",
            },
            {
                "id": "species_purity_unknown",
                "name": "Species Purity Analysis (Multi-Chain Safety Net)",
                "measures": "Detect misassembled species, homodimers, half-molecules",
                "priority": "high",
                "timeline": "Week 1-2",
                "description": "HIC or non-reduced CE-SDS to quantify product-related species.",
                "risk_category": "purity",
                "explanation": (
                    "Multi-chain molecules of unknown class are at elevated risk of "
                    "mispaired or incompletely assembled species."
                ),
                "trigger_reason": f"Unknown class with {n_chains} chains — species heterogeneity risk",
            },
            {
                "id": "subunit_analysis_unknown",
                "name": "Subunit / Reduced Mass Analysis (Multi-Chain Safety Net)",
                "measures": "Confirm individual chain identities and modifications",
                "priority": "medium",
                "timeline": "Week 2-4",
                "description": "Reduced LC-MS to verify each chain's identity and PTM state.",
                "risk_category": "identity",
                "explanation": (
                    "Orthogonal confirmation that each chain in the multi-chain complex "
                    "matches expected sequence and modification state."
                ),
                "trigger_reason": f"Unknown class with {n_chains} chains — orthogonal identity check",
            },
        ]
        format_specific.extend(_safety_net_assays)

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    required_assays.sort(key=lambda x: priority_order.get(x["priority"], 99))
    risk_triggered.sort(key=lambda x: priority_order.get(x["priority"], 99))
    format_specific.sort(key=lambda x: priority_order.get(x["priority"], 99))

    all_assays = required_assays + format_specific + risk_triggered

    # ── Risk summary ────────────────────────────────────────────────────
    risk_levels = []
    if agg > 0.4:
        risk_levels.append(f"HIGH aggregation risk ({agg:.2f})")
    elif agg > 0.2:
        risk_levels.append(f"moderate aggregation risk ({agg:.2f})")

    if stab < 0.65:
        risk_levels.append(f"LOW stability ({stab:.2f})")
    elif stab < 0.8:
        risk_levels.append(f"moderate stability concerns ({stab:.2f})")

    if visc > 0.4:
        risk_levels.append(f"HIGH viscosity risk ({visc:.2f})")
    elif visc > 0.2:
        risk_levels.append(f"moderate viscosity risk ({visc:.2f})")

    if not risk_levels:
        risk_summary = "Low overall risk profile. Standard characterization panel recommended."
    else:
        risk_summary = "Risk factors identified: " + "; ".join(risk_levels) + "."

    # ── Format-specific guidance note ───────────────────────────────────
    _FORMAT_NOTES: Dict[str, str] = {
        "canonical_mab": (
            "Standard IgG monoclonal antibody — ICH Q6B panel applies directly. "
            "Focus on aggregation, charge heterogeneity, and glycosylation."
        ),
        "bispecific": (
            "Bispecific format requires additional species purity assays to confirm "
            "heterodimer assembly. HIC and dual-binding assays are critical for "
            "product quality. Mispairing is the primary manufacturing challenge."
        ),
        "adc": (
            "Antibody-Drug Conjugate requires conjugation-specific assays in addition "
            "to standard antibody testing. DAR uniformity, free drug levels, and linker "
            "stability are critical quality attributes per ICH S9."
        ),
        "fc_fusion": (
            "Fc-fusion protein may have additional glycosylation sites and a vulnerable "
            "fusion junction. Extended glycan profiling and junction integrity testing "
            "are recommended beyond standard IgG assays."
        ),
        "fusion_protein": (
            "Multi-domain fusion protein (e.g., tandem scFv, FabFab, bispecific fragment) "
            "requires linker integrity assessment and domain interaction characterization. "
            "Without Fc region, half-life and effector function are format-dependent. "
            "Standard IgG thresholds may not apply — interpret predictions with caution."
        ),
        "single_domain": (
            "Single-domain antibody (nanobody/VHH) has unique challenges: small size "
            "enables renal clearance, lack of Fc may affect effector functions, and "
            "reversible self-association is common. Consider half-life extension strategy."
        ),
        "peptide": (
            "Peptide therapeutics use a fundamentally different analytical panel than "
            "proteins. RP-HPLC replaces SEC/CE-SDS for purity. Chemical stability "
            "(not thermal unfolding) is the primary degradation concern. MS identity "
            "confirmation is essential for synthetic products."
        ),
        "engineered_scaffold": (
            "Engineered scaffold protein — analytical strategy depends on specific "
            "format. Consider both protein-like (aggregation, stability) and unique "
            "format-specific properties. Consult format developer for recommended panel."
        ),
    }
    if mol_cls == "unknown":
        if n_chains > 2:
            format_note = (
                f"Molecule class: unknown ({n_chains} chains detected). "
                "Applying standard ICH Q6B panel with conservative multi-chain "
                "verification assays. Assembly integrity and species purity assays "
                "added as a precautionary measure. Verify molecule format to refine plan."
            )
        else:
            format_note = (
                "Molecule class: unknown. Applying standard ICH Q6B panel "
                "using canonical IgG assumptions. Verify molecule format to determine "
                "if additional format-specific assays are needed."
            )
    else:
        format_note = _FORMAT_NOTES.get(mol_cls, (
            f"Molecule class '{mol_cls}' — applying standard ICH Q6B panel. "
            "Consider format-specific assays based on structural features."
        ))

    # ── Recommendations ─────────────────────────────────────────────────
    recommendations = []

    # Risk-driven recommendations
    if agg > 0.3:
        recommendations.append(
            "Prioritize aggregation characterization (SEC, DLS) in early development."
        )
    if stab < 0.75:
        recommendations.append(
            "Conduct thermal stability screening (DSF) to optimize formulation."
        )
    if visc > 0.3:
        recommendations.append(
            "Measure viscosity at target concentration before advancing to tox studies."
        )
    if ptm_sites > 3:
        recommendations.append(
            "Implement MAM for comprehensive PTM monitoring across development."
        )

    # Format-driven recommendations
    if mol_cls == "bispecific":
        recommendations.append(
            "Establish species purity (AA/AB/BB) monitoring from early development. "
            "Target heterodimer (AB) purity ≥ 90%."
        )
        recommendations.append(
            "Develop bridging potency assay to confirm dual-target engagement."
        )
    elif mol_cls == "adc":
        recommendations.append(
            "Monitor DAR throughout process development — DAR drift indicates "
            "conjugation instability. Target DAR within ±0.5 of nominal."
        )
        recommendations.append(
            "Conduct forced degradation with focus on linker cleavage pathways."
        )
    elif mol_cls == "fc_fusion":
        recommendations.append(
            "Monitor fusion junction integrity across storage conditions — "
            "junction clipping is a common degradation pathway for Fc-fusions."
        )
    elif mol_cls == "single_domain":
        recommendations.append(
            "Evaluate half-life extension strategy early (PEG, albumin-binding domain, "
            "or Fc fusion) — unmodified VHH has t½ < 2 hours."
        )
    elif mol_cls == "peptide":
        recommendations.append(
            "Prioritize chemical stability (deamidation, oxidation, racemization) "
            "over conformational stability for peptide candidates."
        )
        recommendations.append(
            "Develop stability-indicating RP-HPLC method as primary purity assay."
        )

    if not recommendations:
        recommendations.append(
            "Standard ICH characterization panel sufficient. No elevated risks detected."
        )

    # ── Timeline ────────────────────────────────────────────────────────
    timeline = None
    if include_timeline:
        timeline = {
            "week_1_2": [a["name"] for a in all_assays
                        if a.get("timeline", "").startswith("Week 1")],
            "week_2_4": [a["name"] for a in all_assays
                        if a.get("timeline", "").startswith("Week 2")],
            "week_3_6": [a["name"] for a in all_assays
                        if a.get("timeline", "").startswith("Week 3")],
        }

    result = {
        "required_assays": required_assays,
        "risk_triggered_assays": risk_triggered,
        "format_specific_assays": format_specific,
        "excluded_assays": excluded_assays,
        "all_assays": all_assays,
        "total_assays": len(all_assays),
        "risk_summary": risk_summary,
        "format_note": format_note,
        "molecule_class": mol_cls,
        "recommendations": recommendations,
        "risk_scores": {
            "agg_risk": round(agg, 4),
            "stability": round(stab, 4),
            "viscosity_risk": round(visc, 4),
            "ptm_sites": ptm_sites,
        },
    }

    if timeline:
        result["timeline"] = timeline

    log.info("Validation plan generated: %d assays (%d required, %d format-specific, "
             "%d risk-triggered) for %s",
             len(all_assays), len(required_assays), len(format_specific),
             len(risk_triggered), mol_cls)

    return result


# ===========================================================================
# __main__: Standalone Test
# ===========================================================================

def _selftest():
    """Quick self-test for molecule-class-aware validation planner."""
    ok = True

    # 1. Canonical mAb — low risk → only 3 required assays
    plan = generate_validation_plan(
        risk_scores={"agg_risk": 0.12, "stability": 0.88, "viscosity_risk": 0.08},
        intent={"deam_sites": 1, "ox_sites": 1},
        molecule_class="canonical_mab",
    )
    if plan["total_assays"] < 3:
        print("FAIL: canonical_mab low risk should have >= 3 assays")
        ok = False
    if plan.get("format_specific_assays"):
        print("FAIL: canonical_mab should have no format-specific assays")
        ok = False

    # 2. Bispecific → should add HIC_species, rCE-SDS_bispecific, SPR_dual_binding
    plan_bi = generate_validation_plan(
        risk_scores={"agg_risk": 0.30, "stability": 0.75, "viscosity_risk": 0.15},
        molecule_class="bispecific",
    )
    fmt_ids = [a["id"] for a in plan_bi.get("format_specific_assays", [])]
    for needed in ["HIC_species", "SPR_dual_binding"]:
        if needed not in fmt_ids:
            print(f"FAIL: bispecific missing {needed}")
            ok = False
    if "ispecific" not in plan_bi.get("format_note", "").lower():
        print("FAIL: bispecific format_note missing")
        ok = False

    # 3. ADC → should add DAR, free_drug, linker_stability
    plan_adc = generate_validation_plan(
        risk_scores={"agg_risk": 0.25, "stability": 0.70, "viscosity_risk": 0.10},
        molecule_class="adc",
    )
    fmt_ids_adc = [a["id"] for a in plan_adc.get("format_specific_assays", [])]
    if "DAR_analysis" not in fmt_ids_adc:
        print("FAIL: ADC missing DAR_analysis")
        ok = False

    # 4. Peptide → should EXCLUDE SEC and CE-SDS, ADD RP-HPLC and MS
    plan_pep = generate_validation_plan(
        risk_scores={"agg_risk": 0.10, "stability": 0.80, "viscosity_risk": 0.05},
        molecule_class="peptide",
    )
    req_ids = [a["id"] for a in plan_pep.get("required_assays", [])]
    excl_ids = [a["id"] for a in plan_pep.get("excluded_assays", [])]
    if "SEC" not in excl_ids:
        print("FAIL: peptide should exclude SEC")
        ok = False
    if "SEC" in req_ids:
        print("FAIL: peptide should not require SEC")
        ok = False
    fmt_ids_pep = [a["id"] for a in plan_pep.get("format_specific_assays", [])]
    if "RP_HPLC_purity" not in fmt_ids_pep:
        print("FAIL: peptide missing RP_HPLC_purity")
        ok = False

    # 5. All plans should have recommendations list (non-empty)
    for p in [plan, plan_bi, plan_adc, plan_pep]:
        if not p.get("recommendations"):
            print("FAIL: missing recommendations")
            ok = False

    if ok:
        print("ValidationPlanner v2.0 selftest PASS")
    else:
        print("ValidationPlanner v2.0 selftest FAIL")
    return ok


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _selftest()
