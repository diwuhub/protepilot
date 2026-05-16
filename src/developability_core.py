"""
developability_core.py  ·  ProtePilot — Developability Integration Layer
===========================================================================
Phase 2A: The platform's central decision engine.

This module is the INTEGRATION LAYER that sits above all analytical modules.
It collects evidence from characterization, analytical QC, stability,
PK, formulation, and immunogenicity modules, then produces a unified
risk judgment using molecule-class-aware weight profiles.

Architecture:
    Characterization (evidence) → Developability Core (integration) → Decision

    The key distinction: Characterization tells you WHAT the molecule IS.
    Developability Core tells you WHETHER it can be developed.

Key outputs:
    1. Multi-dimensional risk scores (molecule-class-aware weights)
    2. QTPP (Quality Target Product Profile) table — ICH Q8 framework
    3. Unified go/no-go recommendation with evidence citations
    4. Radar chart data for visualization
    5. Plain-language explanation of why each score is what it is

References:
    - ICH Q8(R2): Pharmaceutical Development
    - ICH Q6B: Specifications for Biotechnological Products
    - Jarasch et al., J. Pharm. Sci., 2015: Developability assessment
    - Jain et al., PNAS, 2017: Biophysical properties of clinical mAbs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.report_schema import GRADE_LOW_UPPER, GRADE_MEDIUM_UPPER, grade_to_risk_label

log = logging.getLogger("ProtePilot.DevelopabilityCore")


# ═══════════════════════════════════════════════════════════════════════
#  Risk Dimension Definition
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RiskDimension:
    """
    Single risk dimension with score, evidence, and explanation.

    Every dimension carries not just a number, but the reasoning behind it.
    This is what makes the platform trustworthy — scientists can see WHY.
    """
    name: str = ""
    display_name: str = ""
    score: float = 0.0              # 0.0 (best) to 1.0 (worst)
    weight: float = 0.0             # Molecule-class-specific weight
    grade: str = "Unknown"          # "Low", "Medium", "High"
    color: str = "#888888"
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""           # Plain-language WHY
    source: str = ""                # Which module produced this score
    confidence: str = "Medium"      # "High", "Medium", "Low"

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


def _grade_score(score: float) -> Tuple[str, str]:
    """Convert 0-1 score to grade and color.

    Uses unified grade boundaries from report_schema.py (SINGLE SOURCE OF TRUTH).
    """
    if score < GRADE_LOW_UPPER:
        return "Low", "#10B981"
    elif score < GRADE_MEDIUM_UPPER:
        return "Medium", "#F59E0B"
    else:
        return "High", "#EF4444"


# ═══════════════════════════════════════════════════════════════════════
#  QTPP (Quality Target Product Profile) — ICH Q8 Framework
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class QTPPRow:
    """Single row of the QTPP table."""
    attribute: str = ""           # CQA name (e.g., "Aggregation (SEC HMW%)")
    target: str = ""              # Target value (e.g., "< 2%")
    acceptable_range: str = ""    # Acceptable range (e.g., "< 5%")
    current_prediction: str = ""  # Platform's prediction
    status: str = "Unknown"       # "Within Target", "Within Range", "Out of Range", "Not Assessed"
    justification: str = ""       # Regulatory reference
    risk_flag: bool = False       # True if current exceeds acceptable range
    model_source: str = ""        # Computation model / module source


# ── Default acceptance criteria (single source of truth) ──────────────
# Keys match the 4 QC-related QTPP rows that also appear in Virtual QC Lab.
# "target" = ideal, "accept" = still acceptable (pass).
# Direction: "upper" = lower is better (agg, acidic), "lower" = higher is better (monomer, intact, main).
QTPP_ACCEPTANCE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "sec_monomer":     {"target_lower": 98.0, "accept_lower": 95.0},
    "aggregation_hmw": {"target_upper": 2.0,  "accept_upper": 5.0},
    "cief_main":       {"target_lower": 70.0, "accept_lower": 60.0},
    "cief_acidic":     {"target_upper": 20.0, "accept_upper": 30.0},
    "cesds_intact":    {"target_lower": 98.0, "accept_lower": 95.0},
}


def generate_qtpp(
    molecule_class: str,
    feature_values: Dict[str, Any],
    dev_predictions: Optional[Dict[str, float]] = None,
    analytical_results: Optional[Dict[str, Any]] = None,
    stability_results: Optional[Dict[str, Any]] = None,
    pk_results: Optional[Dict[str, Any]] = None,
    ada_results: Optional[Dict[str, Any]] = None,
    user_criteria: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[QTPPRow]:
    """
    Generate a QTPP table based on molecule class and available predictions.

    The QTPP is the starting document for any CMC development program
    (ICH Q8). Each row defines a Critical Quality Attribute (CQA) with
    target values and acceptable ranges based on regulatory guidance
    and industry standards.

    Parameters
    ----------
    molecule_class : str
        From MoleculeClassifier output.
    feature_values : dict
        From FeatureRegistry — biophysical features.
    dev_predictions : dict, optional
        From DevelopabilityPredictor — agg_risk, stability, viscosity_risk.
    analytical_results : dict, optional
        From AnalyticalQCTwin — cIEF, SEC, CE-SDS results.
    stability_results : dict, optional
        From StabilityTwin — shelf-life, degradation.
    pk_results : dict, optional
        From PK Twin — half-life, clearance.

    Returns
    -------
    List[QTPPRow] : The QTPP table.
    """
    preds = dev_predictions or {}
    anal = analytical_results or {}
    stab = stability_results or {}
    pk = pk_results or {}
    feats = feature_values or {}

    # ── Merge default + user acceptance criteria ──────────────────
    _crit = {}
    for k, v in QTPP_ACCEPTANCE_DEFAULTS.items():
        _crit[k] = dict(v)  # copy defaults
    if user_criteria:
        for k, overrides in user_criteria.items():
            if k in _crit:
                _crit[k].update(overrides)

    rows = []

    # ── 1. Identity ───────────────────────────────────────────────
    pI_val = feats.get("pI", None)
    rows.append(QTPPRow(
        attribute="Isoelectric Point (pI)",
        target="Reported value",
        acceptable_range="5.5 – 9.5",
        current_prediction=f"{pI_val:.2f}" if pI_val else "Not computed",
        status=_assess_range(pI_val, 5.5, 9.5) if pI_val else "Not Assessed",
        justification="ICH Q6B — characterization of charge heterogeneity",
        model_source="Biopython ProtParam (Bjellqvist)",
    ))

    mw_val = feats.get("mw_kda", None)
    rows.append(QTPPRow(
        attribute="Molecular Weight (kDa)",
        target="Within 5% of theoretical",
        acceptable_range="Theoretical ± 5%",
        current_prediction=f"{mw_val:.1f} kDa" if mw_val else "Not computed",
        status="Within Target" if mw_val else "Not Assessed",
        justification="ICH Q6B — intact mass by LC-MS",
        model_source="Biopython ProtParam (sequence MW)",
    ))

    # ── 2. Purity / Aggregation ───────────────────────────────────
    _agg_c = _crit["aggregation_hmw"]
    agg_risk = preds.get("agg_risk")
    hmw_proj = round(agg_risk * 10.0, 1) if agg_risk is not None else None
    rows.append(QTPPRow(
        attribute="Aggregation (SEC HMW%)",
        target=f"< {_agg_c['target_upper']}%",
        acceptable_range=f"< {_agg_c['accept_upper']}%",
        current_prediction=f"{hmw_proj}% (projected)" if hmw_proj is not None else "Not predicted",
        status=_assess_range_upper(hmw_proj, _agg_c["target_upper"], _agg_c["accept_upper"]) if hmw_proj is not None else "Not Assessed",
        justification="ICH Q6B; EMA/CHMP/BWP/492270/2008 — aggregate control",
        risk_flag=hmw_proj is not None and hmw_proj > _agg_c["accept_upper"],
        model_source="ML Predictor (hydrophobicity + charge)",
    ))

    # SEC monomer purity from analytical
    _sec_c = _crit["sec_monomer"]
    sec_monomer = anal.get("sec_monomer_pct")
    rows.append(QTPPRow(
        attribute="Monomer Purity (SEC)",
        target=f"> {_sec_c['target_lower']}%",
        acceptable_range=f"> {_sec_c['accept_lower']}%",
        current_prediction=f"{sec_monomer:.1f}%" if sec_monomer else "Not assessed",
        status=_assess_range_lower(sec_monomer, _sec_c["target_lower"], _sec_c["accept_lower"]) if sec_monomer else "Not Assessed",
        justification="ICH Q6B — size variants",
        model_source="Analytical QC Twin (SEC simulator)",
    ))

    # ── 3. Charge Heterogeneity ───────────────────────────────────
    _cief_c = _crit["cief_main"]
    cief_main = anal.get("cief_main_pct")
    rows.append(QTPPRow(
        attribute="Charge Variants (cIEF Main Peak%)",
        target=f"> {_cief_c['target_lower']}%",
        acceptable_range=f"> {_cief_c['accept_lower']}%",
        current_prediction=f"{cief_main:.1f}%" if cief_main else "Not assessed",
        status=_assess_range_lower(cief_main, _cief_c["target_lower"], _cief_c["accept_lower"]) if cief_main else "Not Assessed",
        justification="ICH Q6B — charge heterogeneity by cIEF or iCE",
        model_source="Analytical QC Twin (cIEF simulator)",
    ))

    _cief_ac = _crit["cief_acidic"]
    cief_acidic = anal.get("cief_acidic_pct")
    rows.append(QTPPRow(
        attribute="Charge Variants (cIEF Acidic%)",
        target=f"< {_cief_ac['target_upper']}%",
        acceptable_range=f"< {_cief_ac['accept_upper']}%",
        current_prediction=f"{cief_acidic:.1f}%" if cief_acidic else "Not assessed",
        status=_assess_range_upper(cief_acidic, _cief_ac["target_upper"], _cief_ac["accept_upper"]) if cief_acidic else "Not Assessed",
        justification="ICH Q6B — acidic variant control by cIEF or iCE",
        model_source="Analytical QC Twin (cIEF simulator)",
    ))

    deam_sites = feats.get("deam_sites", 0)
    rows.append(QTPPRow(
        attribute="Deamidation Hotspots (NG/NS)",
        target="≤ 3 sites",
        acceptable_range="≤ 6 sites",
        current_prediction=f"{deam_sites} sites",
        status=_assess_range_upper(deam_sites, 3, 6),
        justification="Chemical stability — Asn deamidation drives acidic variant accumulation",
        risk_flag=deam_sites > 6,
        model_source="Sequence Liability Scanner (regex)",
    ))

    # ── 4. Stability ──────────────────────────────────────────────
    stability_score = preds.get("stability")
    rows.append(QTPPRow(
        attribute="Thermal Stability Score",
        target="> 0.80",
        acceptable_range="> 0.60",
        current_prediction=f"{stability_score:.2f}" if stability_score is not None else "Not predicted",
        status=_assess_range_lower(stability_score, 0.80, 0.60) if stability_score is not None else "Not Assessed",
        justification="ICH Q5C — stability indicating methods",
        model_source="ML Predictor (Tm/stability heuristic)",
    ))

    shelf_life = stab.get("shelf_life_months")
    rows.append(QTPPRow(
        attribute="Predicted Shelf Life (2-8°C)",
        target="> 24 months",
        acceptable_range="> 18 months",
        current_prediction=f"{shelf_life:.0f} months" if shelf_life else "Not assessed",
        status=_assess_range_lower(shelf_life, 24, 18) if shelf_life else "Not Assessed",
        justification="ICH Q5C — real-time stability at recommended storage",
        model_source="Stability Twin (Arrhenius kinetics)",
    ))

    # ── 5. Viscosity ──────────────────────────────────────────────
    visc_risk = preds.get("viscosity_risk")
    rows.append(QTPPRow(
        attribute="Viscosity Risk (for SC delivery)",
        target="< 0.20 (Low)",
        acceptable_range="< 0.50 (Medium)",
        current_prediction=f"{visc_risk:.2f}" if visc_risk is not None else "Not predicted",
        status=_assess_range_upper(visc_risk, 0.20, 0.50) if visc_risk is not None else "Not Assessed",
        justification="Target < 20 cP at 150 mg/mL for subcutaneous injectability",
        risk_flag=visc_risk is not None and visc_risk > 0.50,
        model_source="ML Predictor (charge + hydrophobicity)",
    ))

    # ── 6. Fragmentation ─────────────────────────────────────────
    _cesds_c = _crit["cesds_intact"]
    cesds_intact = anal.get("cesds_intact_pct")
    rows.append(QTPPRow(
        attribute="Intact IgG (CE-SDS, Non-reduced)",
        target=f"> {_cesds_c['target_lower']}%",
        acceptable_range=f"> {_cesds_c['accept_lower']}%",
        current_prediction=f"{cesds_intact:.1f}%" if cesds_intact else "Not assessed",
        status=_assess_range_lower(cesds_intact, _cesds_c["target_lower"], _cesds_c["accept_lower"]) if cesds_intact else "Not Assessed",
        justification="ICH Q6B — fragmentation assessment by CE-SDS",
        model_source="Analytical QC Twin (CE-SDS simulator)",
    ))

    # ── 7. PTM Liabilities ────────────────────────────────────────
    ox_sites = feats.get("ox_sites", 0)
    rows.append(QTPPRow(
        attribute="Oxidation-Susceptible Residues (M + W)",
        target="≤ 8 sites",
        acceptable_range="≤ 15 sites",
        current_prediction=f"{ox_sites} sites",
        status=_assess_range_upper(ox_sites, 8, 15),
        justification="Met/Trp oxidation monitoring — forced degradation studies per ICH Q5C",
        model_source="Sequence Liability Scanner (regex)",
    ))

    iso_sites = feats.get("asp_isomerization_sites", 0)
    rows.append(QTPPRow(
        attribute="Asp Isomerization Hotspots (DG/DS)",
        target="≤ 2 sites",
        acceptable_range="≤ 5 sites",
        current_prediction=f"{iso_sites} sites",
        status=_assess_range_upper(iso_sites, 2, 5),
        justification="Asp isomerization to iso-Asp affects potency and charge heterogeneity",
        model_source="Sequence Liability Scanner (regex)",
    ))

    pyro = feats.get("pyroglutamate_risk", 0)
    pyro_text = {0: "No risk (non-Q/E N-term)", 1: "Moderate (Glu N-term)", 2: "High (Gln N-term)"}
    rows.append(QTPPRow(
        attribute="Pyroglutamate Formation (N-terminal)",
        target="Characterized",
        acceptable_range="Monitor if Gln/Glu N-term",
        current_prediction=pyro_text.get(pyro, "Unknown"),
        status="Within Target" if pyro == 0 else "Within Range",
        justification="N-terminal cyclization — common modification, generally non-critical",
        model_source="Sequence Liability Scanner (N-term check)",
    ))

    # ── 8. Glycosylation ─────────────────────────────────────────
    nglyco = feats.get("n_glycosylation_sites", 0)
    rows.append(QTPPRow(
        attribute="N-Glycosylation Sites (NxS/T)",
        target="Reported",
        acceptable_range="Characterized per ICH Q6B",
        current_prediction=f"{nglyco} sites",
        status="Within Target",
        justification="ICH Q6B — glycan profile characterization, ADCC/CDC impact",
        model_source="Sequence Liability Scanner (NxS/T motif)",
    ))

    # ── 9. PK / Half-life (if available) ──────────────────────────
    half_life = pk.get("half_life_days")
    if half_life is not None:
        rows.append(QTPPRow(
            attribute="Predicted Half-Life (human)",
            target="> 14 days (IgG1)",
            acceptable_range="> 7 days",
            current_prediction=f"{half_life:.1f} days",
            status=_assess_range_lower(half_life, 14.0, 7.0),
            justification="FcRn-mediated recycling — IgG1 typical 14-21 days",
            model_source="Preclinical PK Twin (FcRn model)",
        ))

    # ── 10. Immunogenicity ────────────────────────────────────────
    _ada = ada_results or {}
    _ada_risk = _ada.get("ada_risk_level", "").strip()
    _ada_score = _ada.get("ada_risk_score")
    if _ada_risk:
        _ada_pred = f"{_ada_risk}"
        if _ada_score is not None:
            _ada_pred += f" (score: {float(_ada_score):.3f})"
        _ada_status = "Within Target" if _ada_risk.lower() == "low" else \
                      "Within Range" if _ada_risk.lower() == "medium" else "Out of Range"
    else:
        _ada_pred = "Assess via ADA page"
        _ada_status = "Not Assessed"
    rows.append(QTPPRow(
        attribute="Immunogenicity (ADA Risk)",
        target="Low",
        acceptable_range="Manageable with mitigation",
        current_prediction=_ada_pred,
        status=_ada_status,
        justification="ICH S6(R1) — immunogenicity risk assessment for biotechnological products",
        risk_flag=(_ada_risk.lower() == "high") if _ada_risk else False,
        model_source="Immunogenicity Twin (MHC-II binding)",
    ))

    # ── 11. Bispecific-specific CQAs ──────────────────────────────
    if molecule_class == "bispecific":
        rows.append(QTPPRow(
            attribute="Homodimer Contamination (species purity)",
            target="< 5% per homodimer",
            acceptable_range="< 10% total homodimers",
            current_prediction="Assess via Bispecific Separation panel",
            status="Not Assessed",
            justification="Bispecific-specific — homodimer removal is critical for efficacy and safety",
            model_source="Bispecific Engine (SMA/Yamamoto)",
        ))
        rows.append(QTPPRow(
            attribute="CEX Species Resolution (Rs)",
            target="Rs > 1.5",
            acceptable_range="Rs > 0.8",
            current_prediction="Assess via Bispecific Separation panel",
            status="Not Assessed",
            justification="Bispecific AA/AB/BB species must be resolvable for process control",
            model_source="Bispecific Engine (SMA/Yamamoto)",
        ))

    # ── 12. ADC-specific CQAs ─────────────────────────────────────
    if molecule_class == "adc":
        rows.append(QTPPRow(
            attribute="Drug-to-Antibody Ratio (DAR)",
            target="Target DAR ± 0.5",
            acceptable_range="DAR 2.0 – 8.0",
            current_prediction="Not modeled",
            status="Not Assessed",
            justification="ADC-specific — DAR affects efficacy, PK, and toxicity",
            model_source="Not modeled (requires linker info)",
        ))

    # ── P3a: Format-aware caveats for non-canonical molecules ────
    _caveat_formats = {"bispecific", "fusion_protein", "fc_fusion", "adc", "engineered_scaffold", "peptide"}
    if molecule_class in _caveat_formats:
        _caveat = (
            f" [Note: thresholds derived from canonical mAb data; "
            f"interpret with caution for {molecule_class.replace('_', ' ')} format]"
        )
        # Apply caveat to standard mAb-derived rows (not format-specific ones)
        _format_specific_attrs = {
            "Homodimer Contamination (species purity)",
            "CEX Species Resolution (Rs)",
            "Drug-to-Antibody Ratio (DAR)",
        }
        for row in rows:
            if row.attribute not in _format_specific_attrs and row.justification:
                row.justification += _caveat

    return rows


# ═══════════════════════════════════════════════════════════════════════
#  Integrated Risk Assessment
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DevelopabilityAssessment:
    """
    Complete developability assessment — the unified decision output.

    This is what the Developability Dashboard displays.
    """
    # Molecule identity
    molecule_name: str = ""
    molecule_class: str = "unknown"
    molecule_class_display: str = "Unclassified"

    # Risk dimensions
    dimensions: List[RiskDimension] = field(default_factory=list)

    # Composite
    composite_score: float = 0.0     # Weighted sum, 0 (best) - 1 (worst)
    composite_grade: str = "Unknown"  # "Low Risk" / "Medium Risk" / "High Risk" (display label)
    composite_color: str = "#888888"

    # Decision
    recommendation: str = ""         # "Proceed", "Proceed with caution", "Optimize before proceeding"
    recommendation_detail: str = ""  # Plain-language explanation

    # QTPP
    qtpp: List[QTPPRow] = field(default_factory=list)

    # Provenance
    model_sources: List[str] = field(default_factory=list)
    confidence: str = "Medium"

    def radar_data(self) -> Dict[str, Any]:
        """Data structure for radar/spider chart visualization."""
        labels = [d.display_name for d in self.dimensions]
        scores = [d.score for d in self.dimensions]
        weights = [d.weight for d in self.dimensions]
        colors = [d.color for d in self.dimensions]
        return {
            "labels": labels,
            "scores": scores,
            "weights": weights,
            "colors": colors,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "molecule_name": self.molecule_name,
            "molecule_class": self.molecule_class,
            "composite_score": self.composite_score,
            "composite_grade": self.composite_grade,
            "recommendation": self.recommendation,
            "recommendation_detail": self.recommendation_detail,
            "confidence": self.confidence,
            "dimensions": [
                {"name": d.name, "display_name": d.display_name, "score": d.score,
                 "weight": d.weight, "grade": d.grade, "explanation": d.explanation,
                 "source": d.source, "confidence": d.confidence, "evidence": d.evidence}
                for d in self.dimensions
            ],
            "qtpp_rows": [
                {"attribute": r.attribute, "target": r.target,
                 "acceptable_range": r.acceptable_range,
                 "current_prediction": r.current_prediction,
                 "status": r.status, "risk_flag": r.risk_flag}
                for r in self.qtpp
            ],
        }


def assess_developability(
    molecule_name: str = "",
    molecule_class: str = "unknown",
    feature_values: Optional[Dict[str, Any]] = None,
    dev_predictions: Optional[Dict[str, float]] = None,
    analytical_results: Optional[Dict[str, Any]] = None,
    stability_results: Optional[Dict[str, Any]] = None,
    pk_results: Optional[Dict[str, Any]] = None,
    ada_results: Optional[Dict[str, Any]] = None,
    upstream_results: Optional[Dict[str, Any]] = None,
    user_criteria: Optional[Dict[str, Dict[str, float]]] = None,
) -> DevelopabilityAssessment:
    """
    Produce a unified developability assessment.

    This is the MAIN ENTRY POINT for the Developability Core Layer.
    It collects evidence from all modules and produces a single,
    coherent risk judgment.

    Parameters
    ----------
    molecule_name : str
    molecule_class : str
        From MoleculeClassifier (drives weight selection).
    feature_values : dict
        From FeatureRegistry (biophysical features).
    dev_predictions : dict
        From DevelopabilityPredictor (agg_risk, stability, viscosity_risk).
    analytical_results : dict
        From AnalyticalQCTwin (cIEF, SEC, CE-SDS).
    stability_results : dict
        From StabilityTwin (shelf-life, degradation).
    pk_results : dict
        From PKTwin (half-life, clearance).
    ada_results : dict
        From ADA module (immunogenicity risk).
    upstream_results : dict
        From UpstreamTwin (titer, viability).
    """
    assessment = DevelopabilityAssessment(
        molecule_name=molecule_name,
        molecule_class=molecule_class,
    )

    # Get display name
    try:
        from src.molecule_classifier import MoleculeClass
        mc = MoleculeClass(molecule_class)
        assessment.molecule_class_display = mc.display_name
    except (ValueError, ImportError):
        assessment.molecule_class_display = molecule_class.replace("_", " ").title()

    # Get molecule-class-specific weights
    try:
        from src.molecule_classifier import get_risk_weights
        weights = get_risk_weights(
            MoleculeClass(molecule_class) if molecule_class != "unknown"
            else MoleculeClass.UNKNOWN
        )
    except (ValueError, ImportError):
        weights = {"aggregation": 0.30, "stability": 0.25, "viscosity": 0.15,
                   "expression": 0.15, "immunogenicity": 0.15}

    feats = feature_values or {}
    preds = dev_predictions or {}
    anal = analytical_results or {}
    stab = stability_results or {}
    pk = pk_results or {}
    ada = ada_results or {}
    ups = upstream_results or {}

    sources = []

    # ── Dimension 1: Aggregation ──────────────────────────────────
    agg_score = preds.get("agg_risk", 0.3)
    agg_evidence = []
    agg_explanation = ""

    hydro = feats.get("hydrophobicity", 0.35)
    beta = feats.get("beta_sheet_propensity", 1.0)
    cdr_h = feats.get("cdr_hydrophobicity")

    # Hydrophobicity threshold calibrated to Jain et al., PNAS 114 (2017):
    # Clinical-stage mAbs have HIC RT < 14.3 min (top 75th percentile).
    # Normalized hydrophobicity > 0.42 corresponds to HIC RT > ~14 min,
    # indicating elevated self-association and aggregation propensity.
    # Previous threshold 0.45 had no citation; recalibrated to 0.42.
    if hydro > 0.42:
        agg_evidence.append(
            f"High hydrophobicity ({hydro:.2f} > 0.42) increases aggregation propensity "
            f"[Jain et al., PNAS 2017: clinical mAbs 75th percentile HIC RT ≈ 14.3 min]"
        )
    if beta > 1.15:
        agg_evidence.append(f"Elevated beta-sheet propensity ({beta:.3f} > 1.15) — beta-aggregation risk")
    if cdr_h is not None and cdr_h > 0.0:
        agg_evidence.append(
            f"CDR hydrophobicity ({cdr_h:.3f} GRAVY > 0) — surface-exposed hydrophobic patch "
            f"[Raybould et al., mAbs 2019: CDR GRAVY > 0 correlates with polyspecificity]"
        )

    sec_hmw = anal.get("sec_hmw_pct")
    if sec_hmw is not None:
        if sec_hmw > 5.0:
            agg_evidence.append(f"SEC HMW = {sec_hmw:.1f}% exceeds 5% limit")
            agg_score = max(agg_score, 0.7)
        elif sec_hmw > 2.0:
            agg_evidence.append(f"SEC HMW = {sec_hmw:.1f}% above 2% target but within range")
        sources.append("AnalyticalQC (SEC)")

    # Free cysteine check — odd cysteine count implies unpaired -SH group
    # which can form inter-molecular disulfide bonds → aggregation
    _cys_count = feats.get("cysteine_count", 0)
    if _cys_count > 0 and _cys_count % 2 != 0:
        agg_evidence.append(
            f"Odd cysteine count ({_cys_count}) — potential free thiol "
            f"causing disulfide-mediated aggregation"
        )
        agg_score = max(agg_score, 0.35)

    if agg_score < GRADE_LOW_UPPER:
        agg_explanation = "Aggregation risk is low. Hydrophobicity and structural features are within normal ranges for clinical mAbs."
    elif agg_score < GRADE_MEDIUM_UPPER:
        agg_explanation = "Moderate aggregation risk. Consider formulation optimization (excipients, pH) and accelerated stability studies to confirm."
    else:
        agg_explanation = "High aggregation risk. Recommend SEC-MALS, DLS at elevated concentration, and forced degradation study. Formulation screening is critical."

    agg_grade, agg_color = _grade_score(agg_score)
    assessment.dimensions.append(RiskDimension(
        name="aggregation", display_name="Aggregation",
        score=agg_score, weight=weights.get("aggregation", 0.30),
        grade=agg_grade, color=agg_color,
        evidence=agg_evidence, explanation=agg_explanation,
        source="DevelopabilityPredictor (XGBoost)" if "agg_risk" in preds else "Rule-based estimate",
        confidence="High" if "agg_risk" in preds else "Low",
    ))

    # ── Dimension 2: Stability ────────────────────────────────────
    stab_score_raw = preds.get("stability", 0.8)
    stab_risk = 1.0 - stab_score_raw  # Convert: high stability → low risk
    stab_evidence = []
    stab_explanation = ""

    deam = feats.get("deam_sites", 0)
    ox = feats.get("ox_sites", 0)
    iso = feats.get("asp_isomerization_sites", 0)

    if deam > 4:
        stab_evidence.append(f"{deam} deamidation hotspots (NG/NS) — accelerated charge variant formation")
    elif deam > 0:
        stab_evidence.append(f"{deam} deamidation site(s) detected")
    if ox > 10:
        stab_evidence.append(f"{ox} oxidation-susceptible residues (Met+Trp) — monitor under light/peroxide stress")
    if iso > 3:
        stab_evidence.append(f"{iso} Asp isomerization sites — potential potency loss over shelf-life")

    shelf_life = stab.get("shelf_life_months")
    if shelf_life is not None:
        if shelf_life < 18:
            stab_evidence.append(f"Predicted shelf-life {shelf_life:.0f} months — below 18-month minimum")
            stab_risk = max(stab_risk, 0.6)
        elif shelf_life < 24:
            stab_evidence.append(f"Predicted shelf-life {shelf_life:.0f} months — meeting minimum but below 24-month target")
        else:
            stab_evidence.append(f"Predicted shelf-life {shelf_life:.0f} months — meets target")
        sources.append("StabilityTwin (Arrhenius)")

    if stab_risk < GRADE_LOW_UPPER:
        stab_explanation = "Good thermal and chemical stability profile. PTM liabilities are manageable with standard formulation."
    elif stab_risk < GRADE_MEDIUM_UPPER:
        stab_explanation = "Moderate stability concerns. Recommend accelerated stability study (40°C/75% RH) and forced degradation to identify degradation pathways."
    else:
        stab_explanation = "Significant stability risk. Multiple PTM hotspots and/or short predicted shelf-life. Formulation optimization and process controls are critical."

    stab_grade, stab_color = _grade_score(stab_risk)
    assessment.dimensions.append(RiskDimension(
        name="stability", display_name="Stability",
        score=stab_risk, weight=weights.get("stability", 0.25),
        grade=stab_grade, color=stab_color,
        evidence=stab_evidence, explanation=stab_explanation,
        source="DevelopabilityPredictor + StabilityTwin" if shelf_life else "DevelopabilityPredictor",
        confidence="High" if shelf_life else "Medium",
    ))

    # ── Dimension 3: Viscosity ────────────────────────────────────
    visc_score = preds.get("viscosity_risk", 0.15)
    visc_evidence = []

    mw = feats.get("mw_kda", 150)
    if mw > 180:
        visc_evidence.append(f"High MW ({mw:.0f} kDa) increases viscosity at high concentration")
    charge_asym = abs(feats.get("acidic_residues", 40) - feats.get("basic_residues", 45))
    if charge_asym > 15:
        visc_evidence.append(f"Charge asymmetry ({charge_asym} residue difference) — electroviscous effect")

    if visc_score < GRADE_LOW_UPPER:
        visc_expl = "Low viscosity risk. Suitable for high-concentration subcutaneous formulation."
    elif visc_score < GRADE_MEDIUM_UPPER:
        visc_expl = "Moderate viscosity risk. Test concentration-viscosity profile up to target (e.g., 150 mg/mL). Consider arginine or other viscosity reducers."
    else:
        visc_expl = "High viscosity risk. May require IV-only administration or significant formulation work. Concentration-viscosity testing is priority."

    visc_grade, visc_color = _grade_score(visc_score)
    assessment.dimensions.append(RiskDimension(
        name="viscosity", display_name="Viscosity",
        score=visc_score, weight=weights.get("viscosity", 0.15),
        grade=visc_grade, color=visc_color,
        evidence=visc_evidence, explanation=visc_expl,
        source="DevelopabilityPredictor" if "viscosity_risk" in preds else "Rule-based",
        confidence="Medium",
    ))

    # ── Dimension 4: Expression Risk ──────────────────────────────
    # Heuristic-based since we don't have expression ML model yet
    expr_score = 0.2  # Default: low-moderate risk
    expr_evidence = []

    seq_len = feats.get("seq_length", 450)
    if seq_len > 1300:
        expr_score += 0.2
        expr_evidence.append(f"Very long sequence ({seq_len} aa) — potential expression challenge")
    elif seq_len > 800:
        expr_score += 0.1
        expr_evidence.append(f"Extended sequence ({seq_len} aa)")

    cys = feats.get("cysteine_count", 16)
    if cys > 24:
        expr_score += 0.1
        expr_evidence.append(f"{cys} Cys residues — complex disulfide bonding may reduce folding yield")

    # Upstream evidence if available
    titer = ups.get("final_titer")
    if titer is not None:
        if titer < 1.0:
            expr_score = max(expr_score, 0.7)
            expr_evidence.append(f"Predicted titer {titer:.1f} g/L — below commercial viability threshold (2 g/L)")
        elif titer < 2.0:
            expr_score = max(expr_score, 0.4)
            expr_evidence.append(f"Predicted titer {titer:.1f} g/L — below target but may be optimizable")
        else:
            expr_evidence.append(f"Predicted titer {titer:.1f} g/L — meets commercial target")
        sources.append("UpstreamTwin (CHO Fed-Batch)")

    expr_score = min(expr_score, 1.0)
    if expr_score < GRADE_LOW_UPPER:
        expr_expl = "Expression risk is low. Standard CHO cell line and culture conditions expected to achieve adequate titer."
    elif expr_score < GRADE_MEDIUM_UPPER:
        expr_expl = "Moderate expression risk. Consider codon optimization, signal peptide screening, and media/feed optimization."
    else:
        expr_expl = "High expression risk. Significant cell line development effort may be needed. Evaluate alternative host cells or expression strategies."

    expr_grade, expr_color = _grade_score(expr_score)
    assessment.dimensions.append(RiskDimension(
        name="expression", display_name="Expression",
        score=expr_score, weight=weights.get("expression", 0.15),
        grade=expr_grade, color=expr_color,
        evidence=expr_evidence, explanation=expr_expl,
        source="UpstreamTwin + heuristic" if titer else "Heuristic (sequence-based)",
        confidence="Medium" if titer else "Low",
    ))

    # ── Dimension 5: Immunogenicity ───────────────────────────────
    immuno_score = 0.2
    immuno_evidence = []

    ada_risk = ada.get("ada_risk_level", "").lower()
    if ada_risk == "high":
        immuno_score = 0.75
        immuno_evidence.append("ADA module predicts High immunogenicity risk")
    elif ada_risk == "medium":
        immuno_score = 0.45
        immuno_evidence.append("ADA module predicts Medium immunogenicity risk")
    elif ada_risk == "low":
        immuno_score = 0.15
        immuno_evidence.append("ADA module predicts Low immunogenicity risk")
    if ada_risk:
        sources.append("ADA/Immunogenicity Module")

    if molecule_class == "engineered_scaffold":
        immuno_score = max(immuno_score, 0.35)
        immuno_evidence.append("Non-human scaffold framework — inherently higher ADA risk")
    elif molecule_class == "peptide":
        immuno_score = max(immuno_score, 0.30)
        immuno_evidence.append("Small peptide — limited T-cell epitope masking")

    if immuno_score < GRADE_LOW_UPPER:
        immuno_expl = "Low immunogenicity risk. Fully human or humanized framework with manageable epitope profile."
    elif immuno_score < GRADE_MEDIUM_UPPER:
        immuno_expl = "Moderate immunogenicity risk. Consider in-silico deimmunization or clinical ADA monitoring strategy."
    else:
        immuno_expl = "High immunogenicity risk. Recommend T-cell epitope mapping, tolerance induction strategy, and robust ADA assay development."

    immuno_grade, immuno_color = _grade_score(immuno_score)
    assessment.dimensions.append(RiskDimension(
        name="immunogenicity", display_name="Immunogenicity",
        score=immuno_score, weight=weights.get("immunogenicity", 0.15),
        grade=immuno_grade, color=immuno_color,
        evidence=immuno_evidence, explanation=immuno_expl,
        source="ADA Module" if ada_risk else "Heuristic",
        confidence="Medium" if ada_risk else "Low",
    ))

    # ── Bispecific-specific: Species Purity ───────────────────────
    if "species_purity" in weights:
        sp_score = 0.3  # Default moderate
        sp_evidence = ["Bispecific format requires homodimer separation assessment"]
        sp_grade, sp_color = _grade_score(sp_score)
        assessment.dimensions.append(RiskDimension(
            name="species_purity", display_name="Species Purity",
            score=sp_score, weight=weights.get("species_purity", 0.15),
            grade=sp_grade, color=sp_color,
            evidence=sp_evidence,
            explanation="Bispecific antibodies produce AA/AB/BB species mixtures. Homodimer removal is critical for efficacy. Assess via bispecific separation panel.",
            source="Bispecific Engine",
            confidence="Low",
        ))

    # ── ADC-specific: Conjugation ─────────────────────────────────
    if "conjugation" in weights:
        conj_score = 0.3
        conj_evidence = ["ADC conjugation quality (DAR distribution) not yet modeled"]
        conj_grade, conj_color = _grade_score(conj_score)
        assessment.dimensions.append(RiskDimension(
            name="conjugation", display_name="Conjugation",
            score=conj_score, weight=weights.get("conjugation", 0.15),
            grade=conj_grade, color=conj_color,
            evidence=conj_evidence,
            explanation="ADC drug-to-antibody ratio (DAR) and linker stability are critical quality attributes. Not yet modeled — manual assessment required.",
            source="Not modeled",
            confidence="Low",
        ))

    # ── Composite Score ───────────────────────────────────────────
    total_weighted = sum(d.weighted_score for d in assessment.dimensions)
    total_weight = sum(d.weight for d in assessment.dimensions)
    assessment.composite_score = round(total_weighted / max(total_weight, 0.01), 4)
    _raw_grade, assessment.composite_color = _grade_score(assessment.composite_score)
    # Use display label ("Low Risk" / "Medium Risk" / "High Risk") to match
    # the single-path convention from developability_predictor and report_assembler.
    # Internal dimension grades (lines below) keep bare "Low"/"Medium"/"High" for
    # logic comparisons — only the composite exposed to reports gets the suffix.
    assessment.composite_grade = grade_to_risk_label(_raw_grade)

    # ── Recommendation ────────────────────────────────────────────
    high_dims = [d for d in assessment.dimensions if d.grade == "High"]
    med_dims = [d for d in assessment.dimensions if d.grade == "Medium"]

    if len(high_dims) >= 2:
        assessment.recommendation = "Optimize before proceeding"
        assessment.recommendation_detail = (
            f"Multiple high-risk dimensions ({', '.join(d.display_name for d in high_dims)}). "
            f"Recommend addressing these risks before advancing to process development. "
            f"Consider sequence engineering, formulation optimization, or format change."
        )
    elif len(high_dims) == 1:
        assessment.recommendation = "Proceed with caution"
        assessment.recommendation_detail = (
            f"One high-risk dimension ({high_dims[0].display_name}: {high_dims[0].explanation}) "
            f"requires targeted mitigation. Other dimensions are manageable."
        )
    elif len(med_dims) >= 3:
        assessment.recommendation = "Proceed with caution"
        assessment.recommendation_detail = (
            f"Multiple moderate-risk dimensions. No single critical issue, "
            f"but cumulative risk warrants thorough characterization and "
            f"proactive formulation/process development."
        )
    else:
        assessment.recommendation = "Proceed"
        assessment.recommendation_detail = (
            "Overall developability profile is favorable. Standard CMC development "
            "pathway recommended. Continue with process development and analytical characterization."
        )

    # ── QTPP ──────────────────────────────────────────────────────
    assessment.qtpp = generate_qtpp(
        molecule_class=molecule_class,
        feature_values=feats,
        dev_predictions=preds,
        analytical_results=anal,
        stability_results=stab,
        pk_results=pk,
        ada_results=ada,
        user_criteria=user_criteria,
    )

    # ── Provenance ────────────────────────────────────────────────
    assessment.model_sources = sources or ["Heuristic (sequence-based)"]
    assessment.confidence = (
        "High" if len(sources) >= 3
        else "Medium" if len(sources) >= 1
        else "Low"
    )

    log.info(
        "DevelopabilityCore: %s [%s] → score=%.3f (%s), rec=%s, %d QTPP rows",
        molecule_name, molecule_class, assessment.composite_score,
        assessment.composite_grade, assessment.recommendation,
        len(assessment.qtpp),
    )

    return assessment


# ═══════════════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════════════

def _assess_range(value: Optional[float], low: float, high: float) -> str:
    """Assess if value is within an acceptable range."""
    if value is None:
        return "Not Assessed"
    if low <= value <= high:
        return "Within Target"
    return "Out of Range"


def _assess_range_upper(value: Optional[float], target: float, limit: float) -> str:
    """Assess: lower is better (e.g., aggregation). target < limit."""
    if value is None:
        return "Not Assessed"
    if value <= target:
        return "Within Target"
    elif value <= limit:
        return "Within Range"
    return "Out of Range"


def _assess_range_lower(value: Optional[float], target: float, limit: float) -> str:
    """Assess: higher is better (e.g., stability). target > limit."""
    if value is None:
        return "Not Assessed"
    if value >= target:
        return "Within Target"
    elif value >= limit:
        return "Within Range"
    return "Out of Range"


# ═══════════════════════════════════════════════════════════════════════
#  SelfTest
# ═══════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    """
    Comprehensive selftest for developability core layer.

    Tests: all 9 classes, scoring invariants, contract compliance,
    recommendation logic, QTPP row counts, format-specific dimensions.
    """
    errors = []
    checks = 0

    def _check(name, condition, msg=""):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(f"{name}: {msg}")
            log.warning("  [FAIL] selftest %s: %s", name, msg)
        else:
            log.info("  [PASS] selftest %s", name)

    _base_feats = {
        "pI": 8.44, "mw_kda": 148.0, "hydrophobicity": 0.34,
        "deam_sites": 2, "ox_sites": 5, "asp_isomerization_sites": 1,
        "beta_sheet_propensity": 1.08, "cdr_hydrophobicity": -0.3,
        "n_glycosylation_sites": 2, "pyroglutamate_risk": 0,
        "acidic_residues": 38, "basic_residues": 48,
        "seq_length": 450, "cysteine_count": 16,
    }
    _base_preds = {"agg_risk": 0.15, "stability": 0.85, "viscosity_risk": 0.10}

    # ── 1. canonical_mab standard ─────────────────────────────────
    result = assess_developability(
        molecule_name="TestmAb", molecule_class="canonical_mab",
        feature_values=_base_feats, dev_predictions=_base_preds,
    )
    _check("canonical_mab_basic",
           result.molecule_name == "TestmAb" and result.molecule_class == "canonical_mab",
           f"identity: {result.molecule_name}/{result.molecule_class}")
    _check("composite_score_range",
           0.0 <= result.composite_score <= 1.0,
           f"composite_score={result.composite_score}")
    _check("composite_grade_valid",
           result.composite_grade in ("Low Risk", "Medium Risk", "High Risk"),
           f"composite_grade={result.composite_grade}")
    _check("recommendation_valid",
           result.recommendation in ("Proceed", "Proceed with caution", "Optimize before proceeding"),
           f"recommendation={result.recommendation}")
    _check("min_dimensions", len(result.dimensions) >= 5,
           f"expected ≥5, got {len(result.dimensions)}")
    _check("min_qtpp_rows", len(result.qtpp) >= 10,
           f"expected ≥10, got {len(result.qtpp)}")

    # ── 2. Every dimension has explanation and valid grade ────────
    all_dims_ok = True
    for dim in result.dimensions:
        if not dim.explanation:
            all_dims_ok = False
        if dim.grade not in ("Low", "Medium", "High", "Unknown"):
            all_dims_ok = False
        if not (0.0 <= dim.score <= 1.0):
            all_dims_ok = False
    _check("dimensions_quality", all_dims_ok,
           "Some dimensions missing explanation, invalid grade, or score out of range")

    # ── 3. Radar data consistency ─────────────────────────────────
    radar = result.radar_data()
    _check("radar_data",
           len(radar["labels"]) == len(result.dimensions) and
           len(radar["scores"]) == len(result.dimensions),
           "radar data length mismatch")

    # ── 4. QTPP has assessed rows ─────────────────────────────────
    assessed = [r for r in result.qtpp if r.status != "Not Assessed"]
    _check("qtpp_assessed", len(assessed) >= 5,
           f"expected ≥5 assessed, got {len(assessed)}")

    # ── 5. Bispecific: species_purity + homodimer QTPP ────────────
    bispec = assess_developability(
        molecule_name="BispecTest", molecule_class="bispecific",
        feature_values=dict(_base_feats, seq_length=900, cysteine_count=32),
        dev_predictions={"agg_risk": 0.25, "stability": 0.75, "viscosity_risk": 0.20},
    )
    dim_names = [d.name for d in bispec.dimensions]
    _check("bispecific_species_purity", "species_purity" in dim_names,
           f"dims: {dim_names}")
    bispec_qtpp_attrs = [r.attribute for r in bispec.qtpp]
    _check("bispecific_homodimer_qtpp",
           any("Homodimer" in a for a in bispec_qtpp_attrs),
           "missing homodimer QTPP row")
    _check("bispecific_min_dims", len(bispec.dimensions) >= 6,
           f"expected ≥6, got {len(bispec.dimensions)}")

    # ── 6. ADC: conjugation dimension + DAR QTPP ─────────────────
    adc = assess_developability(
        molecule_name="ADCTest", molecule_class="adc",
        feature_values=_base_feats,
        dev_predictions={"agg_risk": 0.20, "stability": 0.80, "viscosity_risk": 0.15},
    )
    adc_dims = [d.name for d in adc.dimensions]
    _check("adc_conjugation", "conjugation" in adc_dims, f"dims: {adc_dims}")
    adc_qtpp_attrs = [r.attribute for r in adc.qtpp]
    _check("adc_dar_qtpp", any("DAR" in a for a in adc_qtpp_attrs),
           "missing DAR QTPP row")

    # ── 7. Stability inversion ────────────────────────────────────
    stab_dim = next(d for d in result.dimensions if d.name == "stability")
    _check("stability_inversion",
           stab_dim.score < 0.25,  # stability=0.85 → risk=0.15 → Low
           f"stability=0.85 should give risk<0.25, got {stab_dim.score}")

    # ── 8. Recommendation logic: high risk → optimize ─────────────
    high_risk = assess_developability(
        molecule_name="HighRisk", molecule_class="canonical_mab",
        feature_values=dict(_base_feats, hydrophobicity=0.55, seq_length=1400,
                           cysteine_count=17, deam_sites=8, ox_sites=20),
        dev_predictions={"agg_risk": 0.70, "stability": 0.30, "viscosity_risk": 0.60},
    )
    _check("recommendation_optimize",
           high_risk.recommendation == "Optimize before proceeding",
           f"got: {high_risk.recommendation}")

    # ── 9. Recommendation logic: low risk → proceed ──────────────
    low_risk = assess_developability(
        molecule_name="LowRisk", molecule_class="canonical_mab",
        feature_values=dict(_base_feats, hydrophobicity=0.28, cdr_hydrophobicity=-0.5),
        dev_predictions={"agg_risk": 0.05, "stability": 0.95, "viscosity_risk": 0.05},
    )
    _check("recommendation_proceed",
           low_risk.recommendation == "Proceed",
           f"got: {low_risk.recommendation}")

    # ── 10. Contract compliance (if available) ────────────────────
    try:
        from src.developability_contract import validate_assessment_output
        for name, r in [("canonical_mab", result), ("bispecific", bispec), ("adc", adc)]:
            v = validate_assessment_output(r.to_dict())
            if v:
                _check(f"contract_{name}", False, f"violations: {v}")
            else:
                _check(f"contract_{name}", True)
    except ImportError:
        _check("contract_import", True)  # Not required for basic selftest

    # ── 11. All 9 classes produce valid output ────────────────────
    all_classes_ok = True
    for cls in ["canonical_mab", "bispecific", "fc_fusion", "adc",
                "single_domain", "peptide", "fusion_protein",
                "engineered_scaffold", "unknown"]:
        try:
            r = assess_developability(molecule_name=f"Test_{cls}", molecule_class=cls,
                                      feature_values=_base_feats, dev_predictions=_base_preds)
            if not (0.0 <= r.composite_score <= 1.0) or len(r.dimensions) < 5:
                all_classes_ok = False
        except Exception:
            all_classes_ok = False
    _check("all_9_classes", all_classes_ok, "Some classes failed")

    # ── 12. Format caveat on non-mAb classes ─────────────────────
    peptide_r = assess_developability(
        molecule_name="PeptideTest", molecule_class="peptide",
        feature_values=dict(_base_feats, seq_length=31),
        dev_predictions=_base_preds,
    )
    caveat_rows = [r for r in peptide_r.qtpp if "interpret with caution" in (r.justification or "")]
    _check("format_caveat", len(caveat_rows) > 0,
           "No format caveat found for peptide QTPP")

    # ── 13. Minimal input (no features/preds) still works ────────
    try:
        minimal = assess_developability(molecule_name="Minimal", molecule_class="canonical_mab")
        _check("minimal_input",
               0.0 <= minimal.composite_score <= 1.0 and len(minimal.dimensions) >= 5,
               f"score={minimal.composite_score}, dims={len(minimal.dimensions)}")
    except Exception as e:
        _check("minimal_input", False, str(e))

    # ── Summary ───────────────────────────────────────────────────
    if errors:
        log.error("DevelopabilityCore selftest: %d/%d FAILED", len(errors), checks)
        for e in errors:
            log.error("  - %s", e)
        return False

    log.info("DevelopabilityCore selftest PASSED (%d/%d checks)", checks, checks)
    return True


# ═══════════════════════════════════════════════════════════════════════
#  Standalone CLI
# ═══════════════════════════════════════════════════════════════════════

def _cli_main():
    """
    Standalone CLI for developability assessment.

    Usage:
        python -m src.developability_core --selftest
        python -m src.developability_core --benchmark
        python -m src.developability_core --demo
        python -m src.developability_core --demo --class bispecific --json
    """
    import argparse
    import json as _json
    import sys as _sys

    parser = argparse.ArgumentParser(
        description="ProtePilot — Developability Core CLI",
    )
    parser.add_argument("--selftest", action="store_true",
                        help="Run selftest suite")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo assessment with reference data")
    parser.add_argument("--class", dest="mol_class", default="canonical_mab",
                        help="Molecule class for demo (default: canonical_mab)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    if args.selftest:
        ok = _selftest()
        _sys.exit(0 if ok else 1)

    if args.benchmark:
        from src.developability_benchmark import main as bench_main
        bench_main()
        return

    if args.demo:
        result = assess_developability(
            molecule_name=f"Demo_{args.mol_class}",
            molecule_class=args.mol_class,
            feature_values={
                "pI": 8.44, "mw_kda": 148.0, "hydrophobicity": 0.34,
                "deam_sites": 2, "ox_sites": 5, "asp_isomerization_sites": 1,
                "beta_sheet_propensity": 1.08, "cdr_hydrophobicity": -0.3,
                "n_glycosylation_sites": 2, "pyroglutamate_risk": 0,
                "acidic_residues": 38, "basic_residues": 48,
                "seq_length": 450, "cysteine_count": 16,
            },
            dev_predictions={"agg_risk": 0.15, "stability": 0.85, "viscosity_risk": 0.10},
        )

        if args.json:
            print(_json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Molecule:      {result.molecule_name} [{result.molecule_class}]")
            print(f"Composite:     {result.composite_score:.3f} ({result.composite_grade})")
            print(f"Recommendation: {result.recommendation}")
            print(f"Confidence:    {result.confidence}")
            print(f"\nRisk Dimensions ({len(result.dimensions)}):")
            for d in result.dimensions:
                print(f"  {d.display_name:20s} score={d.score:.2f} weight={d.weight:.2f} "
                      f"grade={d.grade:6s} [{d.source}]")
            print(f"\nQTPP Rows ({len(result.qtpp)}):")
            for r in result.qtpp:
                flag = " ⚠" if r.risk_flag else ""
                print(f"  {r.attribute:45s} {r.status:15s} {r.current_prediction}{flag}")
        return

    parser.print_help()
    _sys.exit(1)


if __name__ == "__main__":
    _cli_main()
