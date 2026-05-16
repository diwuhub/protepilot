"""
bulk_runner.py  ·  ProtePilot — Bulk Analysis Runner
======================================================
B8: Sequential row-by-row execution with per-row error isolation.

Design decisions (from Architecture Plan):
- B2: skip-and-log — failed rows produce partial report, don't block others
- B3: each row creates a fresh pipeline context (no cross-contamination)
- B5: OOD/unusual flagged and proceeds (consistent with single-molecule)
- B6: sequential loop first (no concurrency complexity)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.bulk_schema import BatchTypeSpec, BulkParseResult, BulkRow
from src.report_schema import GRADE_LOW_UPPER, GRADE_MEDIUM_UPPER
from src.bulk_schema import row_to_intent  # module-level for mockability

log = logging.getLogger("ProtePilot.BulkRunner")


# ═══════════════════════════════════════════════════════════════════════
#  Row Result — result for a single molecule in the batch
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BulkRowResult:
    """Analysis result for one molecule in a bulk batch."""
    row_index: int
    name: str
    status: str = "pending"           # "success" | "error" | "skipped"
    error_message: Optional[str] = None
    wall_time: float = 0.0

    # Computed biophysical properties (from feature_registry)
    mw_kda: Optional[float] = None
    pI: Optional[float] = None
    gravy: Optional[float] = None
    hydrophobicity: Optional[float] = None
    seq_length: Optional[int] = None
    acidic_residues: Optional[int] = None
    basic_residues: Optional[int] = None
    deam_sites: Optional[int] = None
    ox_sites: Optional[int] = None
    cysteine_count: Optional[int] = None

    # Pipeline outputs (populated on success)
    developability_score: Optional[float] = None
    developability_grade: Optional[str] = None
    agg_risk: Optional[float] = None
    stability: Optional[float] = None
    viscosity_risk: Optional[float] = None
    molecule_class: Optional[str] = None
    ood_flag: bool = False
    ood_details: Optional[str] = None

    # Comprehensive analysis results (populated when comprehensive mode enabled)
    # Analytical Twin (MS characterization / liabilities)
    intact_mass_da: Optional[float] = None
    liability_density: Optional[float] = None
    n_liabilities: Optional[int] = None
    liability_summary: Optional[str] = None

    # Analytical QC Twin (cIEF, CE-SDS)
    cief_main_pct: Optional[float] = None
    cief_acidic_pct: Optional[float] = None
    cief_basic_pct: Optional[float] = None
    ce_sds_purity_pct: Optional[float] = None
    ce_sds_hmw_pct: Optional[float] = None
    ce_sds_lmw_pct: Optional[float] = None

    # Preclinical Twin (PK)
    half_life_days: Optional[float] = None
    clearance_ml_day_kg: Optional[float] = None

    # Immunogenicity Twin
    ada_risk_score: Optional[float] = None
    ada_risk_category: Optional[str] = None
    n_mhcii_hotspots: Optional[int] = None

    # Upstream Twin (titer prediction)
    predicted_titer_g_L: Optional[float] = None

    # Stability Twin (ICH shelf-life prediction)
    shelf_life_months: Optional[float] = None
    stability_grade: Optional[str] = None

    # DoE Purification Twin
    doe_optimal_ph: Optional[float] = None
    doe_optimal_gradient: Optional[float] = None
    doe_optimal_yield: Optional[float] = None
    doe_optimal_purity: Optional[float] = None
    doe_rs_min: Optional[float] = None

    # COGS Twin
    cogs_per_gram: Optional[float] = None
    cogs_cost_rating: Optional[str] = None
    cogs_batch_output_g: Optional[float] = None

    # 5-dimension composite score (from assess_developability)
    composite_dev_score: Optional[float] = None
    composite_dev_grade: Optional[str] = None

    # Raw pipeline result for detailed export
    raw_result: Optional[Dict[str, Any]] = field(default=None, repr=False)
    raw_comprehensive: Optional[Dict[str, Any]] = field(default=None, repr=False)
    intent: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Flat dict for CSV/DataFrame export."""
        d = {
            "row": self.row_index + 1,
            "name": self.name,
            "status": self.status,
            "molecule_class": self.molecule_class or "",
            "seq_length": self.seq_length or "",
            "mw_kda": round(self.mw_kda, 2) if self.mw_kda is not None else "",
            "pI": round(self.pI, 2) if self.pI is not None else "",
            "gravy": round(self.gravy, 4) if self.gravy is not None else "",
            "hydrophobicity": round(self.hydrophobicity, 3) if self.hydrophobicity is not None else "",
            "acidic_residues": self.acidic_residues if self.acidic_residues is not None else "",
            "basic_residues": self.basic_residues if self.basic_residues is not None else "",
            # PRIMARY: 5-dim composite from assess_developability() — matches UI single-molecule page
            "dev_score": round(self.composite_dev_score, 3) if self.composite_dev_score is not None else (
                round(self.developability_score, 3) if self.developability_score is not None else ""),
            "dev_grade": self.composite_dev_grade or self.developability_grade or "",
            # REFERENCE: 3-dim predictor (agg+stability+viscosity only) — for diagnostics
            "base_risk_score": round(self.developability_score, 3) if self.developability_score is not None else "",
            "agg_risk": round(self.agg_risk, 3) if self.agg_risk is not None else "",
            "stability": round(self.stability, 3) if self.stability is not None else "",
            "viscosity_risk": round(self.viscosity_risk, 3) if self.viscosity_risk is not None else "",
            "ood_flag": "Y" if self.ood_flag else "",
            # Comprehensive fields
            "intact_mass_da": round(self.intact_mass_da, 1) if self.intact_mass_da is not None else "",
            "liability_density": round(self.liability_density, 1) if self.liability_density is not None else "",
            "n_liabilities": self.n_liabilities if self.n_liabilities is not None else "",
            "cief_main_pct": round(self.cief_main_pct, 1) if self.cief_main_pct is not None else "",
            "cief_acidic_pct": round(self.cief_acidic_pct, 1) if self.cief_acidic_pct is not None else "",
            "cief_basic_pct": round(self.cief_basic_pct, 1) if self.cief_basic_pct is not None else "",
            "ce_sds_purity_pct": round(self.ce_sds_purity_pct, 1) if self.ce_sds_purity_pct is not None else "",
            "ce_sds_hmw_pct": round(self.ce_sds_hmw_pct, 1) if self.ce_sds_hmw_pct is not None else "",
            "ce_sds_lmw_pct": round(self.ce_sds_lmw_pct, 1) if self.ce_sds_lmw_pct is not None else "",
            "deam_sites": self.deam_sites if self.deam_sites is not None else "",
            "ox_sites": self.ox_sites if self.ox_sites is not None else "",
            "cysteine_count": self.cysteine_count if self.cysteine_count is not None else "",
            "half_life_days": round(self.half_life_days, 1) if self.half_life_days is not None else "",
            "predicted_titer_g_L": round(self.predicted_titer_g_L, 2) if self.predicted_titer_g_L is not None else "",
            "clearance_ml_day_kg": round(self.clearance_ml_day_kg, 2) if self.clearance_ml_day_kg is not None else "",
            "ada_risk_score": round(self.ada_risk_score, 3) if self.ada_risk_score is not None else "",
            "ada_risk_category": self.ada_risk_category or "",
            "n_mhcii_hotspots": self.n_mhcii_hotspots if self.n_mhcii_hotspots is not None else "",
            "liability_summary": self.liability_summary or "",
            "shelf_life_months": round(self.shelf_life_months, 0) if self.shelf_life_months is not None else "",
            "stability_grade": self.stability_grade or "",
            "doe_optimal_ph": round(self.doe_optimal_ph, 2) if self.doe_optimal_ph is not None else "",
            "doe_optimal_yield": round(self.doe_optimal_yield * 100, 1) if self.doe_optimal_yield is not None else "",
            "doe_optimal_purity": round(self.doe_optimal_purity, 1) if self.doe_optimal_purity is not None else "",
            "doe_rs_min": round(self.doe_rs_min, 3) if self.doe_rs_min is not None else "",
            "cogs_per_gram": round(self.cogs_per_gram, 2) if self.cogs_per_gram is not None else "",
            "cogs_cost_rating": self.cogs_cost_rating or "",
            "wall_time_s": round(self.wall_time, 2),
            "error": self.error_message or "",
        }
        return d


# ═══════════════════════════════════════════════════════════════════════
#  Batch Result — aggregate of all row results
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BulkBatchResult:
    """Aggregate result for the entire batch."""
    batch_type: str
    molecule_class: str
    results: List[BulkRowResult] = field(default_factory=list)
    wall_time_total: float = 0.0
    started_at: str = ""
    finished_at: str = ""

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.results if r.status == "success")

    @property
    def n_error(self) -> int:
        return sum(1 for r in self.results if r.status == "error")

    @property
    def n_skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def success_rate(self) -> float:
        return self.n_success / max(self.n_total, 1)

    def summary_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics across successful results.

        Uses the 5-dim composite score (composite_dev_score) as the primary
        metric, falling back to the 3-dim predictor (developability_score)
        when composite is not available.
        """
        scores = [
            (r.composite_dev_score if r.composite_dev_score is not None
             else r.developability_score)
            for r in self.results
            if r.status == "success"
            and (r.composite_dev_score is not None or r.developability_score is not None)
        ]
        if not scores:
            return {"n_scored": 0}

        return {
            "n_scored": len(scores),
            "mean_score": round(sum(scores) / len(scores), 3),
            "min_score": round(min(scores), 3),
            "max_score": round(max(scores), 3),
            "n_low_risk": sum(1 for s in scores if s < GRADE_LOW_UPPER),
            "n_medium_risk": sum(1 for s in scores if GRADE_LOW_UPPER <= s < GRADE_MEDIUM_UPPER),
            "n_high_risk": sum(1 for s in scores if s >= GRADE_MEDIUM_UPPER),
            "n_ood": sum(1 for r in self.results if r.ood_flag),
        }

    def to_summary_rows(self) -> List[Dict[str, Any]]:
        """All row results as flat dicts for DataFrame/CSV export."""
        return [r.to_summary_dict() for r in self.results]


# ═══════════════════════════════════════════════════════════════════════
#  Runner — sequential execution engine
# ═══════════════════════════════════════════════════════════════════════

def _extract_results(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a raw pipeline result dict."""
    extracted: Dict[str, Any] = {}

    dev = raw.get("developability")
    if dev:
        score_info = dev.get("score", {})
        extracted["developability_score"] = score_info.get("score")
        extracted["developability_grade"] = score_info.get("grade")

        predictions = dev.get("predictions", {})
        extracted["agg_risk"] = predictions.get("agg_risk")
        extracted["stability"] = predictions.get("stability")
        extracted["viscosity_risk"] = predictions.get("viscosity_risk")

        ood = dev.get("ood_info") or predictions.get("ood_info", {})
        if ood:
            extracted["ood_flag"] = ood.get("is_ood", False)
            flags = ood.get("flags", [])
            if flags:
                # flags may be list of dicts with "metric" key, or list of strings
                parts = []
                for f in flags[:3]:
                    if isinstance(f, dict):
                        parts.append(f.get("metric", str(f)))
                    else:
                        parts.append(str(f))
                extracted["ood_details"] = "; ".join(parts)

    return extracted


def _run_comprehensive_twins(
    row_result: BulkRowResult,
    intent: Dict[str, Any],
    extracted: Dict[str, Any],
) -> None:
    """
    Run additional analysis twins for comprehensive per-molecule results.

    Each twin is isolated with try/except so a failure in one twin
    does not affect the others or the overall row status.
    """
    seq = intent.get("sequence", "")
    name = intent.get("name", "")
    pI_val = intent.get("pI", 8.0)
    mw_val = intent.get("mw", 150.0)
    hydro = intent.get("hydrophobicity", 0.35)    # normalized 0-1 (for cIEF, composite)
    gravy_raw = intent.get("gravy", -0.4)          # raw GRAVY score (for upstream twin)
    chains = intent.get("assembly_chains") or intent.get("chains", [])
    mol_class = intent.get("molecule_class", "canonical_mab")
    # Use MoleculeClass enum for robust type-aware decisions (not string matching)
    try:
        from src.molecule_classifier import MoleculeClass as _MC
        _mc_enum = _MC(mol_class)
        is_mab = _mc_enum.is_mab_like
        has_fc = _mc_enum.has_fc_region
    except (ValueError, KeyError, ImportError):
        is_mab = "mab" in mol_class.lower() or "bispecific" in mol_class.lower()
        has_fc = is_mab or "fc" in mol_class.lower()
    comprehensive = {}

    # ── 1. Analytical Twin: MS characterization & liability scanning ──
    try:
        from src.analytical_twin import run_ms_characterization
        ms_result = run_ms_characterization(
            sequence=seq,
            protein_name=name,
            is_mab=is_mab,
            chains=chains if chains else None,
            molecule_class=mol_class,
        )
        if ms_result and isinstance(ms_result, dict):
            # intact_mass is a nested dict: {bare_mass_da, disulfide_corrected_da, ...}
            im = ms_result.get("intact_mass", {})
            if isinstance(im, dict):
                row_result.intact_mass_da = im.get("disulfide_corrected_da") or im.get("bare_mass_da")
            # liability_density is a dict: {total_residues, total_motifs, density_per_1000, ...}
            ld = ms_result.get("liability_density", {})
            if isinstance(ld, dict):
                row_result.liability_density = ld.get("density_per_1000")
                row_result.n_liabilities = ld.get("total_motifs", 0)
                # Build summary of per-type breakdown
                per_type = ld.get("per_type_density", {})
                if per_type:
                    parts = [f"{k}:{v:.0f}" for k, v in list(per_type.items())[:4] if v > 0]
                    row_result.liability_summary = "; ".join(parts) if parts else None
            comprehensive["analytical"] = ms_result
        log.info("  %s: Analytical Twin OK (mass=%.0f Da, liab_density=%.1f)",
                 name, row_result.intact_mass_da or 0, row_result.liability_density or 0)
    except Exception as e:
        log.warning("  %s: Analytical Twin failed: %s", name, e)

    # ── 2. Analytical QC Twin: cIEF & CE-SDS profiles ──
    try:
        from src.analytical_qc_twin import run_analytical_qc
        # Unified formula: quadratic conversion (matches app.py single-path logic)
        # Low agg risk → very low aggregation %; high risk scales up steeply
        _agg_risk_val = extracted.get("agg_risk", 0.0) or 0.0
        agg_pct = _agg_risk_val * _agg_risk_val * 20.0  # same as app.py line 7549
        qc_result = run_analytical_qc(
            sequence=seq,
            pI=pI_val,
            aggregation_pct=max(0.5, min(agg_pct, 10.0)),
            is_mab=is_mab,
            molecule_class=mol_class,
            sialylation_fraction=0.0,  # standard CHO default (matches single-path)
        )
        if qc_result:
            # cIEF — result has .cief attribute (CIEFResult dataclass)
            cief = getattr(qc_result, "cief", None)
            if cief:
                row_result.cief_main_pct = getattr(cief, "main_pct", None)
                row_result.cief_acidic_pct = getattr(cief, "acidic_pct", None)
                row_result.cief_basic_pct = getattr(cief, "basic_pct", None)

            # CE-SDS — result has .ce_sds attribute (CESDSResult dataclass)
            cesds = getattr(qc_result, "ce_sds", None)
            if cesds:
                row_result.ce_sds_purity_pct = getattr(cesds, "intact_pct", None)
                row_result.ce_sds_hmw_pct = getattr(cesds, "hmw_pct", None)
                row_result.ce_sds_lmw_pct = getattr(cesds, "lmw_pct", None)

            comprehensive["analytical_qc"] = qc_result.summary if hasattr(qc_result, "summary") else str(qc_result)
        log.info("  %s: QC Twin OK (cIEF main=%.1f%%, CE-SDS intact=%.1f%%)",
                 name, row_result.cief_main_pct or 0, row_result.ce_sds_purity_pct or 0)
    except Exception as e:
        log.warning("  %s: Analytical QC Twin failed: %s", name, e)

    # ── 3. Preclinical Twin: PK / half-life prediction ──
    try:
        from src.preclinical_twin import predict_human_half_life, check_fcrn_binding_motif
        # liability_density must be a float, not a dict
        _ld = row_result.liability_density
        _ld_float = float(_ld) if isinstance(_ld, (int, float)) and _ld is not None else 30.0
        # FcRn motif check — critical for single_domain / peptide (no Fc → short half-life)
        _fcrn_intact = True  # default for Fc-bearing
        if seq:
            _fcrn_result = check_fcrn_binding_motif(seq)
            _fcrn_intact = _fcrn_result.get("intact", True) if has_fc else False
        pk_result = predict_human_half_life(
            global_pi=pI_val,
            hydrophobicity=hydro,
            liability_density=_ld_float,
            mw_kda=mw_val,
            fcrn_binding_motif_intact=_fcrn_intact,
            glycoform_profile=intent.get("glycoform_profile", "standard_cho"),
        )
        if pk_result:
            row_result.half_life_days = pk_result.get("half_life_days")
            row_result.clearance_ml_day_kg = pk_result.get("clearance_ml_day_kg")
            comprehensive["preclinical"] = pk_result
        log.info("  %s: Preclinical Twin OK (t1/2=%.1f d)",
                 name, row_result.half_life_days or 0)
    except Exception as e:
        log.warning("  %s: Preclinical Twin failed: %s", name, e)

    # ── 4. Immunogenicity Twin: ADA risk & MHC-II hotspots ──
    try:
        from src.immunogenicity_twin import run_immunogenicity_assessment
        imm_result = run_immunogenicity_assessment(
            sequence=seq,
            agg_risk=extracted.get("agg_risk"),
            dev_score=extracted.get("developability_score"),
            molecule_name=name,
            molecule_class=mol_class,
        )
        if imm_result:
            if hasattr(imm_result, "ada_risk_score"):
                row_result.ada_risk_score = imm_result.ada_risk_score
                # Attribute is ada_risk_level (not ada_risk_category)
                row_result.ada_risk_category = getattr(imm_result, "ada_risk_level", None)
                row_result.n_mhcii_hotspots = getattr(imm_result, "n_high_risk", 0) + getattr(imm_result, "n_medium_risk", 0)
            elif isinstance(imm_result, dict):
                row_result.ada_risk_score = imm_result.get("ada_risk_score")
                row_result.ada_risk_category = imm_result.get("ada_risk_level")
                row_result.n_mhcii_hotspots = imm_result.get("n_high_risk", 0) + imm_result.get("n_medium_risk", 0)
            comprehensive["immunogenicity"] = imm_result.summary if hasattr(imm_result, "summary") else str(imm_result)
        log.info("  %s: Immunogenicity Twin OK (ADA=%s, score=%.3f)",
                 name, row_result.ada_risk_category or "?", row_result.ada_risk_score or 0)
    except Exception as e:
        log.warning("  %s: Immunogenicity Twin failed: %s", name, e)

    # ── 5. Upstream Twin: Fed-batch titer prediction ──
    try:
        from src.upstream_twin import run_upstream_simulation
        # 3-level fallback for dev_score (matches app.py single-path logic):
        #   1. extracted["developability_score"]  (fallback pipeline path)
        #   2. row_result.developability_score    (already parsed)
        #   3. row_result.composite_dev_score     (5-dim composite if set earlier)
        _up_dev_score = (
            extracted.get("developability_score")
            or row_result.developability_score
        )
        _up_agg_risk = (
            extracted.get("agg_risk")
            or row_result.agg_risk
        )
        up_result = run_upstream_simulation(
            seed_density=0.5,
            temp_shift_day=5.0,
            dev_score=_up_dev_score,
            agg_risk=_up_agg_risk,
            culture_days=14.0,
            hydrophobicity=gravy_raw,  # raw GRAVY (not normalized); matches app.py single-path
            sequence=seq,
            molecule_class=mol_class,
        )
        if up_result:
            _titer_val = getattr(up_result, "final_titer", None)
            # Only store meaningful titer (> 0.01 g/L); near-zero = simulation issue
            if _titer_val is not None and _titer_val > 0.01:
                row_result.predicted_titer_g_L = _titer_val
            comprehensive["upstream"] = {
                "final_titer_g_L": _titer_val,
                "peak_vcd": getattr(up_result, "peak_vcd", None),
            }
        log.info("  %s: Upstream Twin OK (titer=%.2f g/L)",
                 name, row_result.predicted_titer_g_L or 0)
    except Exception as e:
        log.warning("  %s: Upstream Twin failed: %s", name, e)

    # ── 6. Stability Twin: ICH shelf-life projection ──
    try:
        from src.stability_twin import run_stability_study
        _stab_deam = row_result.deam_sites or 5
        _stab_hmw = (row_result.ce_sds_hmw_pct or 1.0)
        _stab_acidic = (row_result.cief_acidic_pct or 15.0)
        stab_result = run_stability_study(
            starting_hmw_pct=max(0.1, min(_stab_hmw, 10.0)),
            starting_acidic_pct=max(5.0, min(_stab_acidic, 40.0)),
            formulation_ph=6.0,
            pI=pI_val,
            deamidation_sites=_stab_deam,
            dp_clip_sites=1,
            hydrophobicity=hydro,
        )
        if stab_result:
            row_result.shelf_life_months = getattr(stab_result, "predicted_shelf_life_months", None)
            row_result.stability_grade = getattr(stab_result, "overall_stability_grade", None)
            comprehensive["stability"] = {
                "shelf_life_months": row_result.shelf_life_months,
                "stability_grade": row_result.stability_grade,
            }
        log.info("  %s: Stability Twin OK (shelf_life=%.0f mo, grade=%s)",
                 name, row_result.shelf_life_months or 0, row_result.stability_grade or "?")
    except Exception as e:
        log.warning("  %s: Stability Twin failed: %s", name, e)

    # ── 6b. DoE Purification Twin: pI-adaptive grid search ──
    try:
        from src.purification_optimizer import run_doe_optimization, doe_to_dict
        doe_result = run_doe_optimization(
            pI=pI_val,
            mw=mw_val,
            hydrophobicity=hydro,
            # ph_range=None → pI-adaptive (same as single-path Step 1.10d)
        )
        if doe_result:
            row_result.doe_optimal_ph = doe_result.optimal_ph
            row_result.doe_optimal_gradient = doe_result.optimal_gradient
            row_result.doe_optimal_yield = doe_result.optimal.yield_main if doe_result.optimal else None
            row_result.doe_optimal_purity = doe_result.optimal.pool_purity_pct if doe_result.optimal else None
            row_result.doe_rs_min = doe_result.optimal.resolution_min if doe_result.optimal else None
            comprehensive["doe_purification"] = doe_to_dict(doe_result)
        log.info("  %s: DoE Twin OK (pH=%.2f, yield=%.1f%%)",
                 name, row_result.doe_optimal_ph or 0,
                 (row_result.doe_optimal_yield or 0) * 100)
    except Exception as e:
        log.warning("  %s: DoE Purification Twin failed: %s", name, e)

    # ── 6c. COGS Twin: manufacturing cost calculation ──
    try:
        from src.cogs_twin import COGSInputs, calculate_cogs, cogs_to_dict
        _cogs_titer = row_result.predicted_titer_g_L or 5.0
        _cogs_yield = row_result.doe_optimal_yield or 0.70
        _cogs_inputs = COGSInputs(
            titer_g_per_L=_cogs_titer,
            downstream_yield=_cogs_yield,
        )
        _cogs_result = calculate_cogs(_cogs_inputs, molecule_class=mol_class)
        if _cogs_result:
            row_result.cogs_per_gram = _cogs_result.cogs_per_gram
            row_result.cogs_cost_rating = _cogs_result.cost_rating
            row_result.cogs_batch_output_g = _cogs_result.batch_output_g
            comprehensive["cogs"] = cogs_to_dict(_cogs_result)
        log.info("  %s: COGS Twin OK ($%.2f/g, %s)",
                 name, row_result.cogs_per_gram or 0, row_result.cogs_cost_rating or "?")
    except Exception as e:
        log.warning("  %s: COGS Twin failed: %s", name, e)

    # ── 7. Composite developability score (5-dimension, same as UI core page) ──
    try:
        from src.developability_core import assess_developability
        # Enriched feature_values — aligned with single-path (app.py line 9157)
        _feats = {
            "mw_kda": mw_val,
            "pI": pI_val,
            "hydrophobicity": hydro,
            "seq_length": len(seq) if seq else 0,
            "cysteine_count": intent.get("cysteine_count", 0),
            "deam_sites": intent.get("deam_sites", 0),
            "ox_sites": intent.get("ox_sites", 0),
            "acidic_residues": intent.get("acidic_residues", 0),
            "basic_residues": intent.get("basic_residues", 0),
        }
        # Add enriched features from feature_set if available
        _fs_obj = intent.get("feature_set_obj")
        if _fs_obj and hasattr(_fs_obj, "features"):
            for _fk in ("beta_sheet_propensity", "asp_isomerization_sites",
                         "pyroglutamate_risk", "n_glycosylation_sites"):
                _fvx = _fs_obj.features.get(_fk)
                if _fvx and getattr(_fvx, "value", None) is not None:
                    _feats[_fk] = _fvx.value
        _preds = {}
        if row_result.agg_risk is not None:
            _preds["agg_risk"] = row_result.agg_risk
        if row_result.stability is not None:
            _preds["stability"] = row_result.stability
        if row_result.viscosity_risk is not None:
            _preds["viscosity_risk"] = row_result.viscosity_risk
        _stab_data = {}
        if row_result.shelf_life_months is not None:
            _stab_data["shelf_life_months"] = row_result.shelf_life_months
        _pk_data = {}
        if row_result.half_life_days is not None:
            _pk_data["half_life_days"] = row_result.half_life_days
        _ada_data = {}
        if row_result.ada_risk_category:
            _ada_data["ada_risk_level"] = row_result.ada_risk_category
        _ups_data = {}
        # Only pass titer if it's a meaningful value (> 0.01 g/L);
        # near-zero values indicate simulation didn't run properly
        if row_result.predicted_titer_g_L is not None and row_result.predicted_titer_g_L > 0.01:
            _ups_data["final_titer"] = row_result.predicted_titer_g_L

        # Analytical QC evidence (SEC monomer %)
        _anal_data = {}
        if row_result.cief_main_pct is not None:
            # Derive SEC monomer from the same agg_pct used in QC simulation
            _agg_risk_for_anal = row_result.agg_risk or 0.0
            _agg_pct_for_anal = _agg_risk_for_anal * _agg_risk_for_anal * 20.0
            _anal_data["sec_monomer_pct"] = round(100.0 - max(0.5, min(_agg_pct_for_anal, 10.0)), 2)

        _assessment = assess_developability(
            molecule_name=name,
            molecule_class=mol_class,
            feature_values=_feats,
            dev_predictions=_preds,
            analytical_results=_anal_data or None,
            stability_results=_stab_data,
            pk_results=_pk_data,
            ada_results=_ada_data,
            upstream_results=_ups_data,
        )
        row_result.composite_dev_score = _assessment.composite_score
        row_result.composite_dev_grade = _assessment.composite_grade
        log.info("  %s: Composite score=%.3f (%s)",
                 name, _assessment.composite_score, _assessment.composite_grade)
    except Exception as e:
        log.warning("  %s: Composite score computation failed: %s", name, e)

    row_result.raw_comprehensive = comprehensive


def run_bulk_analysis(
    parse_result: BulkParseResult,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BulkBatchResult:
    """
    Execute the developability pipeline for each valid row in a bulk batch.

    Parameters
    ----------
    parse_result : BulkParseResult
        Validated CSV parse result from bulk_schema.parse_bulk_csv().
    progress_callback : callable, optional
        Called as callback(current_row, total_rows, molecule_name) for UI updates.

    Returns
    -------
    BulkBatchResult with per-row results and aggregate statistics.
    """
    import src.agents as _agents_mod
    PharmaAgentManager = _agents_mod.PharmaAgentManager

    spec = parse_result.batch_type
    batch_result = BulkBatchResult(
        batch_type=spec.key,
        molecule_class=spec.molecule_class,
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    total = len(parse_result.rows)
    manager = PharmaAgentManager()

    log.info("=" * 60)
    log.info("Bulk Analysis: %d molecules, type=%s", total, spec.display_name)
    log.info("=" * 60)

    t_batch_start = time.time()

    for i, row in enumerate(parse_result.rows):
        if progress_callback:
            progress_callback(i + 1, total, row.name)

        # Skip rows with validation errors
        if row.error:
            row_result = BulkRowResult(
                row_index=row.row_index,
                name=row.name,
                status="skipped",
                error_message=row.error,
                molecule_class=spec.molecule_class,
            )
            batch_result.results.append(row_result)
            log.warning("Row %d (%s): skipped — %s", row.row_index + 1, row.name, row.error)
            continue

        # Build intent dict for this row
        t_row_start = time.time()
        row_result = BulkRowResult(
            row_index=row.row_index,
            name=row.name,
            molecule_class=spec.molecule_class,
        )

        try:
            intent = row_to_intent(row, spec)
            row_result.intent = intent

            # Store computed biophysical properties so they appear in exports
            row_result.mw_kda = intent.get("mw")
            row_result.pI = intent.get("pI")
            row_result.gravy = intent.get("gravy")
            row_result.hydrophobicity = intent.get("hydrophobicity")
            row_result.seq_length = intent.get("seq_length")
            row_result.acidic_residues = intent.get("acidic_residues")
            row_result.basic_residues = intent.get("basic_residues")
            row_result.deam_sites = intent.get("deam_sites")
            row_result.ox_sites = intent.get("ox_sites")
            row_result.cysteine_count = intent.get("cysteine_count")

            # Run the pipeline
            raw = manager.run_developability_pipeline(intent)
            row_result.raw_result = raw

            if raw.get("status") == "success":
                row_result.status = "success"
                extracted = _extract_results(raw)
                row_result.developability_score = extracted.get("developability_score")
                row_result.developability_grade = extracted.get("developability_grade")
                row_result.agg_risk = extracted.get("agg_risk")
                row_result.stability = extracted.get("stability")
                row_result.viscosity_risk = extracted.get("viscosity_risk")
                row_result.ood_flag = extracted.get("ood_flag", False)
                row_result.ood_details = extracted.get("ood_details")

                # ── Comprehensive analysis twins ──────────────────
                _run_comprehensive_twins(row_result, intent, extracted)
            else:
                row_result.status = "error"
                row_result.error_message = raw.get("summary", "Pipeline returned error status")

        except Exception as e:
            row_result.status = "error"
            row_result.error_message = f"Exception: {type(e).__name__}: {e}"
            log.exception("Row %d (%s): pipeline exception", row.row_index + 1, row.name)

        row_result.wall_time = time.time() - t_row_start
        batch_result.results.append(row_result)

        log.info(
            "Row %d/%d (%s): %s  [%.1fs]",
            i + 1, total, row.name, row_result.status, row_result.wall_time,
        )

    batch_result.wall_time_total = time.time() - t_batch_start
    batch_result.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")

    log.info("=" * 60)
    log.info(
        "Bulk Analysis complete: %d/%d success, %d errors, %d skipped  [%.1fs]",
        batch_result.n_success, batch_result.n_total,
        batch_result.n_error, batch_result.n_skipped,
        batch_result.wall_time_total,
    )
    log.info("=" * 60)

    if progress_callback:
        progress_callback(total, total, "Complete")

    return batch_result
