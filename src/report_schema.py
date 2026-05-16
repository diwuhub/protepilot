"""
report_schema.py  ·  ProtePilot — Global Report Data Schema
===========================================================
Defines the standardized data structure for the Global Report.

This schema is the SINGLE SOURCE OF TRUTH for report generation.
Both DOCX and JSON exports read from the same ReportObject instance.

Field categories:
  - final_summary   : User-facing conclusions (shown prominently)
  - evidence        : Supporting data that justifies conclusions
  - metadata        : Model route, confidence, technical provenance

Version 2.0 — Unified Data Source Architecture
-----------------------------------------------
  - ReportContext: single authoritative source for all cross-section fields
  - Evidence tiers: Tier 1 (Primary), Tier 2 (Supporting), Tier 3 (Simulated)
  - Narrative-grade mapping: language intensity aligned with risk grade
  - Cross-section consistency validation pass

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# =====================================================================
# 0. Evidence Tier Constants
# =====================================================================
# All evidence in the report belongs to exactly one tier.
# Higher tiers carry more weight in recommendation and top_risks.

EVIDENCE_TIER_1 = "primary"     # Sequence-derived, high confidence
EVIDENCE_TIER_2 = "supporting"  # Predicted, moderate confidence
EVIDENCE_TIER_3 = "simulated"   # Virtual QC, low discriminatory

# Grade boundary constants — SINGLE SOURCE for both report_assembler
# and developability_core. score is a RISK score (0 = best, 1 = worst).
# Calibrated against Jain et al., PNAS 114 (2017) clinical-stage mAb distribution:
#   - Low risk: < 0.30 (corresponds to ~70th percentile of clinical mAbs)
#   - Medium risk: 0.30-0.60 (marginal; recommend mitigation studies)
#   - High risk: ≥ 0.60 (above clinical norms; requires engineering or formulation)
GRADE_LOW_UPPER = 0.30          # score < 0.30  → "Low Risk"
GRADE_MEDIUM_UPPER = 0.60       # 0.30 ≤ score < 0.60 → "Medium Risk"
#                               # score ≥ 0.60  → "High Risk"

# Narrative language intensity mapping
NARRATIVE_MAP = {
    "Low": {
        "tone": ["favorable", "manageable", "low concern"],
        "banned": ["elevated", "significant", "action required"],
        "recommendation_base": "proceed",
    },
    "Medium": {
        "tone": ["moderate", "monitor", "additional mitigation"],
        "banned": ["favorable", "excellent", "negligible"],
        "recommendation_base": "proceed with caution",
    },
    "High": {
        "tone": ["elevated", "significant", "action required"],
        "banned": ["favorable", "manageable", "standard"],
        "recommendation_base": "optimize before proceeding",
    },
}

# Molecule-aware recommendation suffixes
MOLECULE_RECOMMENDATION_SUFFIX = {
    "canonical_mab": "",
    "bispecific": (
        " Recommend format-specific characterization including species-control "
        "strategy, homodimer quantification, and bispecific-specific potency assays."
    ),
    "fc_fusion": (
        " Recommend Fc-fusion linker stability assessment and half-life "
        "extension validation with FcRn binding confirmation."
    ),
    "adc": (
        " Recommend conjugation chemistry DAR consistency, payload stability, "
        "and linker-drug pharmacokinetic characterization."
    ),
    "single_domain": (
        " Recommend aggregation-prone format monitoring, thermal stability "
        "confirmation, and half-life extension strategy assessment."
    ),
    "peptide": (
        " Recommend peptide-specific stability profiling, protease susceptibility, "
        "and formulation compatibility for subcutaneous delivery."
    ),
    "fusion_protein": (
        " Recommend domain interface stability, linker integrity, and "
        "functional activity of each fusion partner independently."
    ),
    "engineered_scaffold": (
        " Recommend scaffold-specific immunogenicity assessment and "
        "comparability to published safety data for this format."
    ),
    "unknown": (
        " Format identification is a prerequisite — all predictions assume "
        "canonical IgG behavior and may not apply. Confirm molecule format "
        "experimentally before relying on risk assessments."
    ),
}


# =====================================================================
# 0b. ReportContext — Unified Data Source
# =====================================================================

@dataclass
class ReportContext:
    """
    Single authoritative source for all cross-section fields.

    Built ONCE at the start of report generation from the latest
    workspace context. All section builders read from this object,
    never directly from intent/cache/extras.

    If a field is None, the section builder MUST mark it as
    'Not assessed' — never silently fall back to a default.
    """
    # ── Molecule identity ──
    molecule_name: str = ""
    molecule_class: str = "unknown"
    molecule_class_display: str = "Unclassified"
    molecule_class_info: Dict[str, Any] = field(default_factory=dict)
    assembly_class: str = ""
    has_fc_region: bool = False
    expects_glycosylation: bool = False

    # ── Biophysical (from feature_set_obj or intent, NEVER hardcoded) ──
    molecular_weight_kda: Optional[float] = None
    isoelectric_point: Optional[float] = None
    gravy_score: Optional[float] = None
    hydrophobicity: Optional[float] = None
    sequence_length: int = 0
    cysteine_count: int = 0

    # ── Liabilities (Tier 1 evidence) ──
    deam_sites: Optional[int] = None       # None means not assessed
    ox_sites: Optional[int] = None
    isomerization_sites: Optional[int] = None
    dp_clip_sites: Optional[int] = None
    free_cysteine_risk: bool = False
    n_glycosylation_sites: Optional[int] = None
    pyroglutamate_risk: bool = False

    # ── Phase 1B biophysical features ──
    beta_sheet_propensity: Optional[float] = None
    cdr_hydrophobicity: Optional[float] = None

    # ── Residue counts (for 5-dim composite) ──
    acidic_residues: Optional[int] = None
    basic_residues: Optional[int] = None

    # ── Developability predictions ──
    agg_risk: Optional[float] = None       # None means not assessed
    stability: Optional[float] = None
    viscosity_risk: Optional[float] = None
    overall_score: Optional[float] = None  # 5-dim composite (primary, matches UI core page)
    base_risk_score: Optional[float] = None  # 3-dim predictor (agg+stability+viscosity only)
    overall_grade: str = ""
    prediction_mode: str = "rule_based"    # "xgboost" | "rule_based"

    # ── OOD (Out-of-Distribution) status ──
    is_ood: bool = False
    ood_confidence: str = "High"           # OOD detector's confidence: High/Medium/Low
    ood_reason: str = ""

    # ── Evidence availability ──
    has_qc_data: bool = False
    qc_is_simulated: bool = False
    has_upstream: bool = False
    upstream_data: Dict[str, Any] = field(default_factory=dict)
    has_pk: bool = False
    has_ada: bool = False

    # ── Chain / structural info ──
    chains: List[Dict[str, Any]] = field(default_factory=list)
    chain_analyses: List[Dict[str, Any]] = field(default_factory=list)
    liability_summary: Dict[str, Any] = field(default_factory=dict)

    # ── Source tracking ──
    source: str = "text"                   # "fasta" | "text" | "assembly"
    context_timestamp: str = ""

    # ── Immutability guard ──
    _frozen: bool = field(default=False, repr=False)

    def freeze(self) -> "ReportContext":
        """Mark this context as immutable. Any subsequent __setattr__ will raise."""
        object.__setattr__(self, "_frozen", True)
        return self

    def __setattr__(self, name: str, value):
        if getattr(self, "_frozen", False) and name != "_frozen":
            raise AttributeError(
                f"ReportContext is frozen — cannot set '{name}'. "
                "Build a new context if you need different values."
            )
        super().__setattr__(name, value)


def grade_from_score(score: float) -> str:
    """Convert risk score (0=best, 1=worst) to grade string.

    Uses the SINGLE authoritative threshold constants defined above.
    """
    if score < GRADE_LOW_UPPER:
        return "Low"
    elif score < GRADE_MEDIUM_UPPER:
        return "Medium"
    else:
        return "High"


def grade_to_risk_label(grade: str) -> str:
    """Convert grade to display label: 'Low' → 'Low Risk'."""
    return f"{grade} Risk" if grade in ("Low", "Medium", "High") else grade


# =====================================================================
# 1. Sub-schemas (Section-level data containers)
# =====================================================================

@dataclass
class ExecutiveSummary:
    """Section 1: One-page executive summary."""
    molecule_name: str = ""
    molecule_class: str = "unknown"
    molecule_class_display: str = "Unclassified"
    assembly_class: str = ""               # e.g. "homodimer", "heterodimer", "monomer"
    overall_score: float = 0.0
    overall_grade: str = ""                # "Low Risk", "Medium Risk", "High Risk"
    recommendation: str = ""               # "Proceed", "Proceed with caution", "Optimize"
    recommendation_detail: str = ""        # Plain-language explanation
    top_risks: List[str] = field(default_factory=list)   # Top 3 risk statements
    top_strengths: List[str] = field(default_factory=list)
    confidence_level: str = "Medium"       # Overall analysis confidence
    key_caveats: List[str] = field(default_factory=list)
    limitations_upfront: List[str] = field(default_factory=list)  # First-page limitations
    analysis_date: str = ""


@dataclass
class MoleculeOverview:
    """Section 2: Molecule identity and biophysical profile."""
    name: str = ""
    molecule_class: str = ""
    format_description: str = ""           # e.g. "IgG1 κ monoclonal antibody"
    chain_composition: List[Dict[str, Any]] = field(default_factory=list)
    # e.g. [{"name":"HC","type":"heavy","length":449,"copies":2}, ...]
    stoichiometry: str = ""                # e.g. "2HC + 2LC (homodimer)"
    molecular_weight_kda: Optional[float] = None
    isoelectric_point: Optional[float] = None
    gravy_score: Optional[float] = None
    hydrophobicity_normalized: Optional[float] = None
    sequence_length: int = 0
    cysteine_count: int = 0
    has_fc_region: bool = False
    expects_glycosylation: bool = False


@dataclass
class RiskItem:
    """A single risk dimension with explanation."""
    dimension: str = ""                    # e.g. "Aggregation"
    score: float = 0.0                     # 0-1
    grade: str = ""                        # "Low" / "Medium" / "High"
    weight: float = 0.0                    # Molecule-class-specific weight
    primary_drivers: List[str] = field(default_factory=list)
    explanation: str = ""                  # Why this matters
    source: str = ""                       # Which module produced this
    confidence: str = "Medium"
    evidence_tier: str = EVIDENCE_TIER_2   # "primary" | "supporting" | "simulated"
    assessed: bool = True                  # False = not assessed (no data)


@dataclass
class DevelopabilitySection:
    """Section 3: Developability risk assessment."""
    composite_score: float = 0.0
    composite_grade: str = ""
    recommendation: str = ""
    risk_dimensions: List[RiskItem] = field(default_factory=list)
    liability_summary: Dict[str, int] = field(default_factory=dict)
    # e.g. {"deamidation_sites": 3, "oxidation_sites": 5, ...}
    qtpp_rows: List[Dict[str, str]] = field(default_factory=list)
    # e.g. [{"attribute":"Aggregation","target":"<2%","current":"2.7%","status":"Within Range"},...]
    evidence_basis: str = ""               # "sequence-only" | "sequence+experimental" | "experimental"


@dataclass
class AnalyticalSummary:
    """Section 4: Analytical / Characterization key findings."""
    sec_monomer_pct: Optional[float] = None
    sec_hmw_pct: Optional[float] = None
    cesds_intact_pct: Optional[float] = None
    cief_main_pct: Optional[float] = None
    cief_acidic_pct: Optional[float] = None
    cief_basic_pct: Optional[float] = None
    glycan_highlights: List[str] = field(default_factory=list)
    charge_variant_note: str = ""
    purity_note: str = ""
    evidence_status: str = ""              # "assessed" | "not_yet_assessed" | "partial"
    ms_intact_mass_da: Optional[float] = None
    ms_note: str = ""


@dataclass
class ProcessPKSummary:
    """Section 5: Process / PK / Downstream summary.

    Unified schema — all fields populated by BOTH single-path (via
    _analysis_cache from app.py auto-triggers) and bulk-path (via
    _run_comprehensive_twins in bulk_runner.py).
    """
    # PK
    half_life_days: Optional[float] = None
    clearance_ml_day_kg: Optional[float] = None
    pk_risk_level: str = ""
    pk_note: str = ""
    pk_evidence_status: str = ""           # "predicted" | "predicted_low_confidence" | "measured"
    # Chromatography
    cex_summary: str = ""                  # e.g. "3-peak elution, Rs > 1.5"
    # DoE Purification Optimization
    doe_optimal_ph: Optional[float] = None
    doe_optimal_yield: Optional[float] = None
    doe_optimal_purity: Optional[float] = None
    doe_rs_min: Optional[float] = None
    doe_note: str = ""
    # Upstream
    final_titer_g_l: Optional[float] = None
    upstream_note: str = ""
    # Stability Twin (ICH shelf-life projection)
    shelf_life_months: Optional[float] = None
    stability_grade: Optional[str] = None
    stability_note: str = ""
    # ADA / Immunogenicity
    ada_risk_level: str = ""
    ada_risk_score: Optional[float] = None
    n_mhcii_hotspots: Optional[int] = None
    ada_note: str = ""
    # COGS (commercial manufacturing cost)
    cogs_per_gram: Optional[float] = None
    cogs_cost_rating: Optional[str] = None
    cogs_note: str = ""


@dataclass
class ValidationPlanSection:
    """Section 6: Recommended analytical validation plan."""
    molecule_class_impact: str = ""        # How molecule class affected the plan
    total_assays: int = 0
    required_assays: List[Dict[str, str]] = field(default_factory=list)
    # [{"name":"SEC","priority":"critical","reason":"ICH Q6B"}]
    format_specific_assays: List[Dict[str, str]] = field(default_factory=list)
    risk_triggered_assays: List[Dict[str, str]] = field(default_factory=list)
    excluded_assays: List[Dict[str, str]] = field(default_factory=list)
    key_recommendations: List[str] = field(default_factory=list)


@dataclass
class ModelMetadata:
    """Section 7: Analysis route, model provenance, confidence."""
    analysis_route: str = ""               # "FASTA → pLM → XGBoost" or "Text → Rule-Based"
    model_source: str = ""                 # "XGBoost v1.0" or "Rule-Based Heuristic"
    embedding_mode: str = ""               # "ESM-2" or "Mock (Composition)"
    prediction_mode: str = ""              # "xgboost" or "rule_based"
    is_ood: bool = False                   # Out-of-distribution flag
    ood_reason: str = ""
    confidence_level: str = "Medium"       # High / Medium / Low
    confidence_rationale: str = ""
    heuristic_ml_hybrid: str = ""          # "heuristic" / "ML" / "hybrid"
    molecule_class_benchmark_note: str = ""
    caveats: List[str] = field(default_factory=list)
    high_confidence_conclusions: List[str] = field(default_factory=list)
    low_confidence_conclusions: List[str] = field(default_factory=list)


@dataclass
class RegulatoryContextSection:
    """Section 8b: Regulatory signal classification (optional).

    Populated by the cross-repo bridge to reg-intel-biopharma's
    Policy-Signal Classifier. Absent when the classifier is unavailable.
    """
    signal_class: str = "unknown"          # new_requirement | relaxation | maintenance | ambiguous | unknown
    probabilities: Dict[str, float] = field(default_factory=dict)
    source: str = ""                       # "policy_signal_classifier" | "fallback"
    note: str = ""                         # Human-readable note (e.g. fallback reason)
    assessed: bool = False                 # True only when classifier ran successfully


@dataclass
class AppendixData:
    """Section 8: Selected raw metrics for reviewer reference."""
    biophysical_features: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"pI": 8.45, "MW_kDa": 145.4, "GRAVY": -0.326, ...}
    liability_counts: Dict[str, int] = field(default_factory=dict)
    cdr_regions: List[Dict[str, str]] = field(default_factory=list)
    chain_details: List[Dict[str, Any]] = field(default_factory=list)


# =====================================================================
# 2. Top-Level Report Object
# =====================================================================

@dataclass
class ReportObject:
    """
    The unified Global Report data object.

    This is assembled by report_assembler.py and consumed by
    report_generator.py. Both DOCX and JSON exports derive from
    the SAME instance of this object.
    """
    # Header
    report_title: str = "ProtePilot — Global Analysis Report"
    report_version: str = "2.0"
    generated_at: str = ""
    platform_version: str = "ProtePilot v3.0"

    # Unified context (v2.0)
    context: ReportContext = field(default_factory=ReportContext)

    # Sections
    executive_summary: ExecutiveSummary = field(default_factory=ExecutiveSummary)
    molecule_overview: MoleculeOverview = field(default_factory=MoleculeOverview)
    developability: DevelopabilitySection = field(default_factory=DevelopabilitySection)
    analytical: AnalyticalSummary = field(default_factory=AnalyticalSummary)
    process_pk: ProcessPKSummary = field(default_factory=ProcessPKSummary)
    validation_plan: ValidationPlanSection = field(default_factory=ValidationPlanSection)
    model_metadata: ModelMetadata = field(default_factory=ModelMetadata)
    regulatory_context: RegulatoryContextSection = field(default_factory=RegulatoryContextSection)
    appendix: AppendixData = field(default_factory=AppendixData)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def save_json(self, path: str) -> None:
        """Write JSON to file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


# =====================================================================
# 3. Self-test
# =====================================================================

def _selftest():
    """Validate schema instantiation and serialization."""
    r = ReportObject(
        generated_at=datetime.now().isoformat(),
    )
    r.executive_summary.molecule_name = "Trastuzumab"
    r.executive_summary.molecule_class = "canonical_mab"
    r.executive_summary.overall_score = 0.78
    r.executive_summary.top_risks = ["Moderate deamidation risk"]

    r.developability.risk_dimensions.append(RiskItem(
        dimension="Aggregation", score=0.15, grade="Low", weight=0.30,
        primary_drivers=["Low hydrophobicity"], explanation="Favorable aggregation profile",
    ))

    d = r.to_dict()
    assert d["executive_summary"]["molecule_name"] == "Trastuzumab"
    assert len(d["developability"]["risk_dimensions"]) == 1
    assert d["developability"]["risk_dimensions"][0]["score"] == 0.15

    j = r.to_json()
    assert "Trastuzumab" in j
    assert "Aggregation" in j

    print("ReportSchema _selftest PASS")
    return True


if __name__ == "__main__":
    _selftest()
