"""
report_assembler.py  ·  ProtePilot — Report Assembly Layer v2.0
===========================================================
Collects results from ALL system modules and assembles them
into a standardized ReportObject (defined in report_schema.py).

Version 2.0 — Unified Data Source Architecture
-----------------------------------------------
  1. ReportContext built ONCE from latest workspace context
  2. All section builders read from ReportContext, never from raw dicts
  3. No hardcoded defaults — missing data → "Not assessed"
  4. Evidence tiers (Tier 1 > Tier 2 > Tier 3) for top_risks/recommendation
  5. Narrative-grade mapping: language intensity matches risk grade
  6. Molecule-aware recommendation templates
  7. Cross-section consistency pass at the end

Usage:
    from src.report_assembler import assemble_report
    report = assemble_report(intent, analysis_cache, session_state)
    report.save_json("report.json")

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.report_schema import (
    ReportObject, ReportContext, ExecutiveSummary, MoleculeOverview,
    DevelopabilitySection, RiskItem, AnalyticalSummary,
    ProcessPKSummary, ValidationPlanSection, ModelMetadata,
    AppendixData, RegulatoryContextSection,
    # Constants
    EVIDENCE_TIER_1, EVIDENCE_TIER_2, EVIDENCE_TIER_3,
    GRADE_LOW_UPPER, GRADE_MEDIUM_UPPER,
    NARRATIVE_MAP, MOLECULE_RECOMMENDATION_SUFFIX,
    grade_from_score, grade_to_risk_label,
)

log = logging.getLogger("ProtePilot.ReportAssembler")


# =====================================================================
# Main Entry Point
# =====================================================================

def assemble_report(
    intent: Dict[str, Any],
    analysis_cache: Optional[Dict[str, Any]] = None,
    session_extras: Optional[Dict[str, Any]] = None,
) -> ReportObject:
    """
    Assemble a complete ReportObject from current analysis state.

    v2.0: All data flows through a single ReportContext object.
    """
    cache = analysis_cache or {}
    extras = session_extras or {}

    # ── Guard: reject if intent looks stale (bulk safety) ──────────
    _cache_intent = cache.get("intent")
    if _cache_intent and intent:
        _ci_name = _cache_intent.get("name", "")
        _i_name = intent.get("name", "")
        if _ci_name and _i_name and _ci_name != _i_name:
            log.warning(
                "Intent/cache mismatch: intent.name='%s' but cache.intent.name='%s'. "
                "Using intent as primary (cache may be stale).",
                _i_name, _ci_name,
            )

    # ── Step 0: Build unified context ──────────────────────────────
    ctx = _build_report_context(intent, cache, extras)

    # ── Step 0b: Upgrade overall_score to 5-dim composite ─────────
    # The 3-dim predictor score (agg+stability+viscosity) is stored as
    # ctx.overall_score during _build_report_context. We now compute
    # the full 5-dim composite (matching the UI developability core
    # page) using assess_developability(), and store it back. The 3-dim
    # predictor is preserved as ctx.base_risk_score for reference.
    ctx.base_risk_score = ctx.overall_score  # keep 3-dim predictor
    try:
        from src.developability_core import assess_developability as _assess
        _feats = {
            "mw_kda": ctx.molecular_weight_kda,
            "pI": ctx.isoelectric_point,
            "hydrophobicity": ctx.hydrophobicity,
            "seq_length": ctx.sequence_length,
            "cysteine_count": ctx.cysteine_count,
            "deam_sites": ctx.deam_sites,
            "ox_sites": ctx.ox_sites,
            "acidic_residues": ctx.acidic_residues,
            "basic_residues": ctx.basic_residues,
        }
        _dev_data_0b = _get_dev_data(cache)
        _preds = _dev_data_0b.get("predictions", {})
        if ctx.agg_risk is not None:
            _preds.setdefault("agg_risk", ctx.agg_risk)
        if ctx.stability is not None:
            _preds.setdefault("stability", ctx.stability)
        if ctx.viscosity_risk is not None:
            _preds.setdefault("viscosity_risk", ctx.viscosity_risk)

        # ── Extract stability evidence from cache ──
        _stab_raw_0b = cache.get("stability_result", {})
        _stab = None
        if isinstance(_stab_raw_0b, dict) and _stab_raw_0b:
            _stab = {"shelf_life_months": _stab_raw_0b.get("shelf_life_months")}

        # ── Extract PK evidence from cache ──
        _pk_raw_0b = cache.get("pk_result", {})
        _pk_data_0b = _pk_raw_0b.get("data", _pk_raw_0b) if isinstance(_pk_raw_0b, dict) else {}
        _pk = {"half_life_days": _pk_data_0b.get("half_life_days")} if _pk_data_0b.get("half_life_days") else {}

        # ── Extract ADA / immunogenicity evidence from cache ──
        _ada_raw_0b = cache.get("ada_result") or cache.get("immunogenicity_result") or extras.get("ada_result")
        _ada = None
        if isinstance(_ada_raw_0b, dict) and _ada_raw_0b:
            _ada = {
                "ada_risk_level": _ada_raw_0b.get("ada_risk_level",
                    _ada_raw_0b.get("risk_category", "")),
            }

        # ── Extract upstream titer evidence from cache ──
        _ups = None
        if ctx.has_upstream and ctx.upstream_data:
            _titer_val = ctx.upstream_data.get("final_titer")
            if _titer_val is not None and _titer_val > 0.01:
                _ups = {"final_titer": _titer_val}

        # ── Extract analytical QC evidence from cache ──
        _anal_0b = None
        _qc_raw_0b = cache.get("analytical_qc")
        if isinstance(_qc_raw_0b, dict):
            _sec_0b = _qc_raw_0b.get("sec", {})
            _anal_0b = {
                "sec_monomer_pct": _sec_0b.get("monomer_pct") if isinstance(_sec_0b, dict)
                    else getattr(_sec_0b, "monomer_pct", None),
            }

        _composite = _assess(
            molecule_name=ctx.molecule_name,
            molecule_class=ctx.molecule_class,
            feature_values=_feats,
            dev_predictions=_preds,
            analytical_results=_anal_0b,
            stability_results=_stab,
            pk_results=_pk,
            ada_results=_ada,
            upstream_results=_ups,
        )
        ctx.overall_score = _composite.composite_score
        ctx.overall_grade = grade_to_risk_label(grade_from_score(_composite.composite_score))
        log.debug("Report composite score: %.3f (%s) [was 3-dim: %.3f]",
                  ctx.overall_score, ctx.overall_grade, ctx.base_risk_score or 0)
    except Exception as _e:
        log.warning("Could not compute 5-dim composite for report: %s. "
                    "Falling back to 3-dim predictor score.", _e)
        # Keep the 3-dim score as fallback

    ctx.freeze()  # Immutable from this point — prevents accidental mutation

    report = ReportObject(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        context=ctx,
    )

    # ── Section builders — all read from ctx ──────────────────────
    report.executive_summary = _build_executive_summary(ctx, cache)
    report.molecule_overview = _build_molecule_overview(ctx)
    report.developability = _build_developability(ctx, cache, intent)
    report.analytical = _build_analytical(ctx, cache)
    report.process_pk = _build_process_pk(ctx, cache, extras)
    report.validation_plan = _build_validation_plan(ctx, cache, intent)
    report.model_metadata = _build_model_metadata(ctx, cache, intent)
    report.regulatory_context = _build_regulatory_context(ctx, intent)
    report.appendix = _build_appendix(ctx, intent)

    # ── Final: Cross-section consistency pass ─────────────────────
    _validate_cross_section_consistency(report)

    log.info("Report v2.0 assembled for '%s' (%s) — grade=%s, score=%.2f",
             ctx.molecule_name, ctx.molecule_class,
             ctx.overall_grade, ctx.overall_score or 0)

    return report


# =====================================================================
# Step 0: Build ReportContext (Single Source of Truth)
# =====================================================================

def _build_report_context(
    intent: Dict, cache: Dict, extras: Dict,
) -> ReportContext:
    """
    Extract ALL cross-section fields from the latest workspace context.

    Priority order for each field:
      1. feature_set_obj (FeatureRegistry computed values)
      2. intent dict (from parse_intent / molecule setup)
      3. None (NOT a hardcoded default)

    If a value is not available, it stays None.
    """
    ctx = ReportContext()
    ctx.context_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ctx.source = intent.get("source", "text")

    # ── Molecule identity ──
    ctx.molecule_name = intent.get("name", "Unknown Molecule")
    mol_cls = intent.get("molecule_class", "unknown")
    if hasattr(mol_cls, "value"):
        mol_cls = mol_cls.value
    ctx.molecule_class = mol_cls

    cls_info = intent.get("molecule_class_info", {})
    ctx.molecule_class_info = cls_info
    ctx.molecule_class_display = cls_info.get(
        "display_name", mol_cls.replace("_", " ").title()
    )
    ctx.has_fc_region = cls_info.get("has_fc_region", False)
    ctx.expects_glycosylation = cls_info.get("expects_glycosylation", False)

    # Assembly class
    chains = intent.get("chains", [])
    ctx.chains = chains
    if len(chains) == 1:
        ctx.assembly_class = "monomer"
    elif len(chains) == 2:
        seqs = [c.get("sequence", "") for c in chains]
        ctx.assembly_class = "homodimer" if seqs[0] == seqs[1] else "heterodimer"
    elif len(chains) > 2:
        ctx.assembly_class = f"multi-chain ({len(chains)} chains)"

    # ── Biophysical — priority: feature_set_obj > intent > None ──
    fs_obj = intent.get("feature_set_obj")
    if fs_obj and hasattr(fs_obj, "value"):
        ctx.molecular_weight_kda = fs_obj.value("mw_kda")
        ctx.isoelectric_point = fs_obj.value("pI")
        ctx.hydrophobicity = fs_obj.value("hydrophobicity")
        ctx.gravy_score = intent.get("gravy")
        ctx.sequence_length = intent.get("seq_length", 0)
        ctx.cysteine_count = intent.get("cysteine_count", 0)
        ctx.acidic_residues = _int_or_none(intent.get("acidic_residues"))
        ctx.basic_residues = _int_or_none(intent.get("basic_residues"))
        # Liabilities from feature_set_obj
        ctx.deam_sites = _int_or_none(fs_obj.value("deam_sites"))
        ctx.ox_sites = _int_or_none(fs_obj.value("ox_sites"))
    else:
        ctx.molecular_weight_kda = intent.get("mw")  # may be None
        ctx.isoelectric_point = intent.get("pI")
        ctx.hydrophobicity = intent.get("hydrophobicity")
        ctx.gravy_score = intent.get("gravy")
        ctx.sequence_length = intent.get("seq_length", 0)
        ctx.cysteine_count = intent.get("cysteine_count", 0)
        ctx.acidic_residues = _int_or_none(intent.get("acidic_residues"))
        ctx.basic_residues = _int_or_none(intent.get("basic_residues"))
        # Liabilities from intent (may be None if not computed)
        ctx.deam_sites = _int_or_none(intent.get("deam_sites"))
        ctx.ox_sites = _int_or_none(intent.get("ox_sites"))

    # ── Phase 1B features from feature_set_obj ──
    if fs_obj and hasattr(fs_obj, "value"):
        ctx.beta_sheet_propensity = fs_obj.value("beta_sheet_propensity")
        ctx.cdr_hydrophobicity = fs_obj.value("cdr_hydrophobicity")
        _pyro = fs_obj.value("pyroglutamate_risk")
        if _pyro is not None:
            ctx.pyroglutamate_risk = bool(_pyro)

    # ── Liabilities from liability_summary ──
    liab = intent.get("liability_summary", {})
    ctx.liability_summary = liab
    if liab:
        if ctx.deam_sites is None:
            ctx.deam_sites = _int_or_none(liab.get("deamidation_sites"))
        if ctx.ox_sites is None:
            ctx.ox_sites = _int_or_none(liab.get("oxidation_sites"))
        ctx.isomerization_sites = _int_or_none(liab.get("asp_isomerization_sites"))
        ctx.n_glycosylation_sites = _int_or_none(liab.get("n_glycosylation_sites"))
        if ctx.dp_clip_sites is None:
            # Try all known key variants: dp_clipping_sites (primary), dp_sites (legacy)
            ctx.dp_clip_sites = _int_or_none(
                liab.get("dp_clipping_sites") or liab.get("dp_sites")
            )

    ctx.chain_analyses = intent.get("chain_analyses", [])

    # ── P1c: Propagate dp_clip_sites from chain_analyses if still None ──
    # Chain analyses may store dp count in nested liabilities dict
    if ctx.dp_clip_sites is None and ctx.chain_analyses:
        _total_dp = 0
        for _ca in ctx.chain_analyses:
            # Try top-level "dp" first, then liabilities.dp_count
            _dp_val = _ca.get("dp")
            if _dp_val is None:
                _ca_liab = _ca.get("liabilities", {})
                if isinstance(_ca_liab, dict):
                    _dp_val = _ca_liab.get("dp_count") or _ca_liab.get("dp_clipping_sites")
            _total_dp += (_dp_val or 0)
        if _total_dp > 0:
            ctx.dp_clip_sites = _total_dp

    # ── Developability predictions ──
    dev_data = _get_dev_data(cache)
    preds = dev_data.get("predictions", {})

    # Use None for missing predictions, NOT hardcoded defaults
    ctx.agg_risk = preds.get("agg_risk") if "agg_risk" in preds else None
    ctx.stability = preds.get("stability") if "stability" in preds else None
    ctx.viscosity_risk = preds.get("viscosity_risk") if "viscosity_risk" in preds else None

    score_info = dev_data.get("score", {})
    ctx.overall_score = score_info.get("score")
    raw_grade = score_info.get("grade", "")
    # Normalize grade using unified thresholds
    if ctx.overall_score is not None:
        ctx.overall_grade = grade_to_risk_label(grade_from_score(ctx.overall_score))
    else:
        ctx.overall_grade = raw_grade or "Unknown"

    # The developability result stores the key as "prediction_mode" (not "mode").
    # Fall back to cache-level "predictor_source" (chromatography model source) only
    # if neither key is present.
    ctx.prediction_mode = (
        dev_data.get("prediction_mode")
        or dev_data.get("mode")
        or cache.get("predictor_source", "rule_based")
    )
    ctx.embedding_mode = dev_data.get("embedding_mode", cache.get("embedding_mode", "mock"))

    # ── OOD (Out-of-Distribution) status ──
    _ood_reason = cache.get("ood_bypass_reason") or ""
    ctx.is_ood = bool(_ood_reason)
    ctx.ood_reason = _ood_reason
    # OOD detector stores confidence in dev_data; parse from ood_reason z-score
    if ctx.is_ood:
        import re as _re
        _z_match = _re.search(r"max z[=\s]*([\d.]+)", _ood_reason)
        if _z_match:
            _max_z = float(_z_match.group(1))
            ctx.ood_confidence = "Low" if _max_z >= 3.0 else ("Medium" if _max_z >= 2.0 else "High")
        else:
            ctx.ood_confidence = "Medium"  # OOD but z not parseable → conservative

    # ── Evidence availability ──
    qc_data = cache.get("analytical_qc")
    ctx.has_qc_data = bool(qc_data)
    ctx.qc_is_simulated = (
        isinstance(qc_data, dict) and qc_data.get("source") == "simulated"
    ) if ctx.has_qc_data else False
    # Upstream data may live in extras (from session_state) or cache
    _ups_dict = extras.get("upstream_result_dict", {})
    if not (isinstance(_ups_dict, dict) and _ups_dict.get("final_titer")):
        _ups_dict = cache.get("upstream_result") or cache.get("upstream_result_dict") or {}
    ctx.has_upstream = bool(isinstance(_ups_dict, dict) and _ups_dict.get("final_titer"))
    ctx.upstream_data = _ups_dict if ctx.has_upstream else {}
    ctx.has_pk = bool(cache.get("pk_result"))
    # ADA: check both extras and cache — app.py stores in both locations,
    # but some code paths may only populate one.
    ctx.has_ada = bool(extras.get("ada_result") or cache.get("ada_result"))

    return ctx


# =====================================================================
# Section Builders (all read from ReportContext)
# =====================================================================

def _build_executive_summary(ctx: ReportContext, cache: Dict) -> ExecutiveSummary:
    """Build Section 1: Executive Summary."""
    es = ExecutiveSummary()
    es.molecule_name = ctx.molecule_name
    es.analysis_date = datetime.now().strftime("%Y-%m-%d")
    es.molecule_class = ctx.molecule_class
    es.molecule_class_display = ctx.molecule_class_display
    es.assembly_class = ctx.assembly_class

    # Score and grade from unified context
    # P1c: Preserve None vs 0.0 — None means "not computed", 0.0 is a real score
    es.overall_score = ctx.overall_score if ctx.overall_score is not None else None
    es.overall_grade = ctx.overall_grade

    # ── Recommendation — unified logic ──
    es.recommendation, es.recommendation_detail = _generate_recommendation(ctx)

    # ── Confidence — capped by OOD status ──
    _model_conf = "High" if ctx.prediction_mode == "xgboost" else "Medium"
    es.confidence_level = _effective_confidence(_model_conf, ctx)

    # ── Top Risks — Tier 1 evidence first, then Tier 2 ──
    es.top_risks = _generate_top_risks(ctx)

    # ── Top Strengths — only from assessed dimensions ──
    es.top_strengths = _generate_top_strengths(ctx)

    # ── Caveats ──
    es.key_caveats = _generate_caveats(ctx)

    # ── Upfront limitations ──
    es.limitations_upfront = _generate_limitations(ctx)

    return es


def _build_molecule_overview(ctx: ReportContext) -> MoleculeOverview:
    """Build Section 2: Molecule Overview — all from ReportContext."""
    mo = MoleculeOverview()
    mo.name = ctx.molecule_name
    mo.molecule_class = ctx.molecule_class
    mo.format_description = ctx.molecule_class_info.get("description", "")
    mo.has_fc_region = ctx.has_fc_region
    mo.expects_glycosylation = ctx.expects_glycosylation

    for ca in ctx.chain_analyses:
        mo.chain_composition.append({
            "name": ca.get("name", ""),
            "type": ca.get("chain_type", "unknown"),
            "length": ca.get("length", 0),
        })

    if ctx.chains:
        parts = []
        for c in ctx.chains:
            copies = c.get("copy_number", 1)
            name = c.get("name", "chain")
            parts.append(f"{copies}\u00d7{name}" if copies > 1 else name)
        mo.stoichiometry = " + ".join(parts)

    mo.molecular_weight_kda = round(ctx.molecular_weight_kda, 1) if ctx.molecular_weight_kda is not None else None
    mo.isoelectric_point = ctx.isoelectric_point
    mo.gravy_score = ctx.gravy_score
    mo.hydrophobicity_normalized = ctx.hydrophobicity
    mo.sequence_length = ctx.sequence_length
    mo.cysteine_count = ctx.cysteine_count

    return mo


def _build_developability(
    ctx: ReportContext, cache: Dict, intent: Dict,
) -> DevelopabilitySection:
    """Build Section 3: Developability Assessment."""
    ds = DevelopabilitySection()

    # P1c: Preserve None vs 0.0 semantics
    ds.composite_score = ctx.overall_score  # None = not computed; 0.0 = real score
    # Use unified grade
    if ctx.overall_score is not None:
        ds.composite_grade = grade_to_risk_label(grade_from_score(ctx.overall_score))
    else:
        ds.composite_grade = "Unknown"

    # Recommendation from unified logic (same as executive summary)
    ds.recommendation, _ = _generate_recommendation(ctx)

    # ── Risk Dimensions with evidence tiers and narrative mapping ──
    _dim_defs = [
        ("Aggregation", "agg_risk", True, EVIDENCE_TIER_2),
        ("Thermal Stability", "stability", False, EVIDENCE_TIER_2),
        ("Viscosity", "viscosity_risk", True, EVIDENCE_TIER_2),
    ]

    # Weights from molecule_classifier
    # NOTE: get_risk_weights returns keys like "aggregation", "stability", "viscosity"
    # (not abbreviated "agg"/"visc"), so we must use the full key names.
    try:
        from src.molecule_classifier import get_risk_weights
        weights = get_risk_weights(ctx.molecule_class)
    except Exception:
        weights = {"aggregation": 0.30, "stability": 0.25, "viscosity": 0.20}

    weight_map = {
        "agg_risk": weights.get("aggregation", 0.30),
        "stability": weights.get("stability", 0.25),
        "viscosity_risk": weights.get("viscosity", 0.20),
    }

    for dim_name, key, higher_worse, tier in _dim_defs:
        raw_val = getattr(ctx, key, None) if key in ("agg_risk", "stability", "viscosity_risk") else None
        if raw_val is None:
            # Not assessed — mark explicitly, don't use default
            ds.risk_dimensions.append(RiskItem(
                dimension=dim_name, score=0.0, grade="Not assessed",
                weight=weight_map.get(key, 0.33), primary_drivers=[],
                explanation=f"{dim_name} data not available. Cannot assess this dimension.",
                source="none", confidence="None",
                evidence_tier=tier, assessed=False,
            ))
            continue

        val = raw_val
        w = weight_map.get(key, 0.33)
        grade_val = (1.0 - val) if not higher_worse else val
        grade = grade_from_score(grade_val)

        # Narrative-aligned explanation with evidence tier label
        explanation = _build_dimension_explanation(dim_name, key, val, grade, ctx)
        drivers = _build_dimension_drivers(key, val, ctx)
        _tier_labels = {
            EVIDENCE_TIER_1: "Primary evidence",
            EVIDENCE_TIER_2: "Supporting evidence (predicted)",
            EVIDENCE_TIER_3: "Simulated / supportive only",
        }
        _tier_prefix = f"[{_tier_labels.get(tier, tier)}] "
        explanation = _tier_prefix + explanation

        ds.risk_dimensions.append(RiskItem(
            dimension=dim_name, score=round(val, 4), grade=grade,
            weight=round(w, 3), primary_drivers=drivers,
            explanation=explanation, source="developability_predictor",
            confidence=_effective_confidence(
                "High" if ctx.prediction_mode == "xgboost" else "Medium", ctx),
            evidence_tier=tier, assessed=True,
        ))

    # ── Evidence basis ──
    if ctx.has_qc_data and not ctx.qc_is_simulated:
        ds.evidence_basis = "sequence+experimental"
    elif ctx.has_qc_data and ctx.qc_is_simulated:
        ds.evidence_basis = "sequence+simulated"
    else:
        ds.evidence_basis = "sequence-only"

    # ── Liability Summary (Tier 1 evidence — from ctx, not hardcoded) ──
    ds.liability_summary = {}
    if ctx.deam_sites is not None:
        ds.liability_summary["deamidation_sites"] = ctx.deam_sites
    if ctx.ox_sites is not None:
        ds.liability_summary["oxidation_sites"] = ctx.ox_sites
    if ctx.n_glycosylation_sites is not None:
        ds.liability_summary["glycosylation_sites"] = ctx.n_glycosylation_sites
    if ctx.dp_clip_sites is not None:
        ds.liability_summary["dp_clip_sites"] = ctx.dp_clip_sites
    if ctx.isomerization_sites is not None:
        ds.liability_summary["isomerization_sites"] = ctx.isomerization_sites

    # ── QTPP rows (from developability_core) ──
    # v2.0 fix: Build feat_vals from ReportContext (single source of truth),
    # NOT from feature_set_obj or intent. This ensures QTPP current_prediction
    # values are identical to what Molecule Overview and Liability Summary show.
    try:
        from src.developability_core import assess_developability

        # ── Authoritative feature map from ctx ──
        feat_vals = {}
        _ctx_field_map = {
            "pI": ctx.isoelectric_point,
            "mw_kda": ctx.molecular_weight_kda,
            "hydrophobicity": ctx.hydrophobicity,
            "deam_sites": ctx.deam_sites,
            "ox_sites": ctx.ox_sites,
            "asp_isomerization_sites": ctx.isomerization_sites,
            "n_glycosylation_sites": ctx.n_glycosylation_sites,
            "dp_sites": ctx.dp_clip_sites,
            "seq_length": ctx.sequence_length,
            "cysteine_count": ctx.cysteine_count,
            # Phase 1B features — previously missing from report
            "beta_sheet_propensity": ctx.beta_sheet_propensity,
            "cdr_hydrophobicity": ctx.cdr_hydrophobicity,
            "pyroglutamate_risk": 1.0 if ctx.pyroglutamate_risk else 0.0,
        }
        for _fk, _fv in _ctx_field_map.items():
            if _fv is not None:
                feat_vals[_fk] = _fv

        # acidic/basic residues: try feature_set_obj first (ctx doesn't store these),
        # then fall back to liability_summary or chain_analyses
        fs_obj = intent.get("feature_set_obj")
        if fs_obj and hasattr(fs_obj, "value"):
            for _rk in ("acidic_residues", "basic_residues"):
                _rv = fs_obj.value(_rk)
                if _rv is not None:
                    feat_vals[_rk] = _rv
        if "acidic_residues" not in feat_vals:
            # Derive from chain_analyses if possible
            _total_acidic = 0
            _total_basic = 0
            for _ca in ctx.chain_analyses:
                _seq = _ca.get("sequence", "").upper()
                _total_acidic += _seq.count("D") + _seq.count("E")
                _total_basic += _seq.count("K") + _seq.count("R") + _seq.count("H")
            if _total_acidic > 0:
                feat_vals["acidic_residues"] = _total_acidic
                feat_vals["basic_residues"] = _total_basic

        # Free cysteine risk
        if ctx.free_cysteine_risk:
            feat_vals["free_cysteine_risk"] = True

        dev_data = _get_dev_data(cache)
        preds = dev_data.get("predictions", {})
        pk_raw = cache.get("pk_result", {})
        pk_data = pk_raw.get("data", pk_raw) if isinstance(pk_raw, dict) else {}

        # ── Build analytical_results dict from cache (P1a fix) ──
        _qc_raw = cache.get("analytical_qc")
        _anal_for_assess = None
        if isinstance(_qc_raw, dict):
            _sec = _qc_raw.get("sec", {})
            _cief = _qc_raw.get("cief", {})
            _cesds = _qc_raw.get("ce_sds", {})
            _anal_for_assess = {
                "sec_monomer_pct": _sec.get("monomer_pct") if isinstance(_sec, dict)
                    else getattr(_sec, "monomer_pct", None),
                "sec_hmw_pct": _sec.get("hmw_pct") if isinstance(_sec, dict)
                    else getattr(_sec, "hmw_pct", None),
                "cief_main_pct": _cief.get("main_pct") if isinstance(_cief, dict)
                    else getattr(_cief, "main_pct", None),
                "cesds_intact_pct": _cesds.get("intact_pct") if isinstance(_cesds, dict)
                    else getattr(_cesds, "intact_pct", None),
            }

        # ── Build stability_results dict from cache (P1a fix) ──
        _stab_raw = cache.get("stability_result", {})
        _stab_for_assess = None
        if isinstance(_stab_raw, dict) and _stab_raw:
            _stab_for_assess = {
                "shelf_life_months": _stab_raw.get("shelf_life_months"),
            }

        # ── Build ada_results from extras (P1a fix) ──
        _ada_for_assess = None
        _ada_raw = cache.get("ada_result") or cache.get("immunogenicity_result")
        if isinstance(_ada_raw, dict) and _ada_raw:
            _ada_for_assess = {
                "ada_risk_level": _ada_raw.get("ada_risk_level",
                    _ada_raw.get("risk_category", "")),
            }

        # ── Build upstream_results from ctx (populated from extras/cache) ──
        _ups_for_assess = None
        if ctx.has_upstream and ctx.upstream_data:
            _ups_for_assess = {
                "final_titer": ctx.upstream_data.get("final_titer"),
            }

        assessment = assess_developability(
            molecule_name=ctx.molecule_name,
            molecule_class=ctx.molecule_class,
            feature_values=feat_vals,
            dev_predictions=preds,
            analytical_results=_anal_for_assess,
            stability_results=_stab_for_assess,
            pk_results={"half_life_days": pk_data.get("half_life_days")},
            ada_results=_ada_for_assess,
            upstream_results=_ups_for_assess,
        )
        for row in assessment.qtpp:
            ds.qtpp_rows.append({
                "attribute": row.attribute,
                "target": row.target,
                "acceptable_range": row.acceptable_range,
                "current_prediction": row.current_prediction,
                "status": row.status,
                "justification": row.justification,
            })
    except Exception as e:
        log.warning("QTPP generation failed: %s", e)

    return ds


def _build_analytical(ctx: ReportContext, cache: Dict) -> AnalyticalSummary:
    """Build Section 4: Analytical QC Summary."""
    an = AnalyticalSummary()
    qc = cache.get("analytical_qc")
    if not qc:
        an.evidence_status = "not_yet_assessed"
        an.purity_note = (
            "Analytical characterization not yet performed. SEC, CE-SDS, and cIEF data "
            "will be populated after experimental runs. Current developability assessment "
            "is based on sequence-derived predictions only."
        )
        return an

    _qc_source = qc.get("source", "experimental") if isinstance(qc, dict) else "experimental"
    an.evidence_status = "simulated" if _qc_source == "simulated" else "assessed"

    # SEC
    sec = qc.get("sec", {})
    if hasattr(sec, "monomer_pct"):
        an.sec_monomer_pct = sec.monomer_pct
        an.sec_hmw_pct = sec.hmw_pct
    elif isinstance(sec, dict):
        an.sec_monomer_pct = sec.get("monomer_pct")
        an.sec_hmw_pct = sec.get("hmw_pct")

    # CE-SDS
    cesds = qc.get("ce_sds", {})
    if hasattr(cesds, "intact_pct"):
        an.cesds_intact_pct = cesds.intact_pct
    elif isinstance(cesds, dict):
        an.cesds_intact_pct = cesds.get("intact_pct")

    # cIEF
    cief = qc.get("cief", {})
    if hasattr(cief, "main_pct"):
        an.cief_main_pct = cief.main_pct
        an.cief_acidic_pct = cief.acidic_pct
        an.cief_basic_pct = cief.basic_pct
    elif isinstance(cief, dict):
        an.cief_main_pct = cief.get("main_pct")
        an.cief_acidic_pct = cief.get("acidic_pct")
        an.cief_basic_pct = cief.get("basic_pct")

    # Purity note
    if an.sec_monomer_pct is not None:
        if an.sec_monomer_pct >= 98:
            an.purity_note = f"High monomer purity ({an.sec_monomer_pct:.1f}%). Favorable SEC profile."
        elif an.sec_monomer_pct >= 95:
            an.purity_note = f"Acceptable monomer purity ({an.sec_monomer_pct:.1f}%). Minor aggregates detected."
        else:
            an.purity_note = f"Low monomer purity ({an.sec_monomer_pct:.1f}%). Aggregation mitigation needed."

    if an.cief_main_pct is not None:
        an.charge_variant_note = (
            f"Main peak: {an.cief_main_pct:.1f}%, "
            f"Acidic: {an.cief_acidic_pct:.1f}%, "
            f"Basic: {an.cief_basic_pct:.1f}%"
        )

    # Tier 3 disclaimer for simulated data
    if _qc_source == "simulated" and an.purity_note:
        an.purity_note = (
            "[Simulated / Supportive only] " + an.purity_note +
            " Values are sequence-derived predictions with limited discriminatory power. "
            "Do not use as primary decision drivers. Confirm with laboratory characterization."
        )

    return an


def _build_process_pk(
    ctx: ReportContext, cache: Dict, extras: Dict,
) -> ProcessPKSummary:
    """Build Section 5: Process / PK summary."""
    pk = ProcessPKSummary()

    # PK — class-aware
    pk_raw = cache.get("pk_result", {})
    if isinstance(pk_raw, dict):
        pk_data = pk_raw.get("data", pk_raw)
        pk.half_life_days = pk_data.get("half_life_days")
        if pk.half_life_days:
            pk.pk_evidence_status = (
                "predicted_low_confidence" if ctx.molecule_class == "unknown" else "predicted"
            )

            _prefix = ""
            if ctx.molecule_class == "unknown":
                _prefix = (
                    "Molecule format unclassified — PK model assumes canonical IgG "
                    "(Fc-mediated clearance). This prediction may not apply. "
                )

            _fc_note = "FcRn-mediated recycling expected" if ctx.has_fc_region else \
                       "No Fc region — renal/proteolytic clearance likely dominates"

            if pk.half_life_days > 14:
                pk.pk_risk_level = "Low"
                pk.pk_note = (
                    f"{_prefix}Predicted t\u00bd = {pk.half_life_days:.1f} days. "
                    f"Favorable PK profile for {ctx.molecule_class_display}. {_fc_note}."
                )
            elif pk.half_life_days > 7:
                pk.pk_risk_level = "Medium"
                pk.pk_note = (
                    f"{_prefix}Predicted t\u00bd = {pk.half_life_days:.1f} days. "
                    f"Moderate for {ctx.molecule_class_display} — may need dose optimization. {_fc_note}."
                )
            else:
                pk.pk_risk_level = "High"
                pk.pk_note = (
                    f"{_prefix}Predicted t\u00bd = {pk.half_life_days:.1f} days. "
                    f"Short half-life for {ctx.molecule_class_display}."
                )
                if ctx.has_fc_region:
                    pk.pk_note += " Investigate FcRn binding affinity and Fc integrity."

    # Chromatography
    variants = cache.get("variants")
    if variants:
        source = cache.get("source_label", "")
        # Normalize internal labels to user-facing descriptions
        _source_labels = {
            "ML OVERRIDE": "ML-predicted SMA parameters (PyTorch MLP)",
            "ML_OVERRIDE": "ML-predicted SMA parameters (PyTorch MLP)",
            "static_v5": "Heuristic SMA parameters (PropertyMapper v5)",
            "static_v5_gravy": "Heuristic SMA parameters (PropertyMapper v5, GRAVY-adjusted)",
        }
        source_display = _source_labels.get(source, source) if source else "Analytical model"
        pk.cex_summary = f"CEX model: {source_display}"

    # Upstream — use ctx.upstream_data (unified from extras/cache in _build_report_context)
    if ctx.has_upstream and ctx.upstream_data:
        _titer = ctx.upstream_data.get("final_titer")
        if _titer is not None:
            pk.final_titer_g_l = _titer
            pk.upstream_note = f"Predicted final titer: {_titer:.2f} g/L"
            # Include additional upstream metrics if available
            _vcd = ctx.upstream_data.get("peak_vcd")
            _viab = ctx.upstream_data.get("viability_at_harvest")
            if _vcd:
                pk.upstream_note += f" (peak VCD: {_vcd:.1f} × 10⁶/mL"
                if _viab:
                    pk.upstream_note += f", harvest viability: {_viab:.0f}%"
                pk.upstream_note += ")"

    # PK clearance
    if isinstance(pk_raw, dict):
        pk_data = pk_raw.get("data", pk_raw)
        pk.clearance_ml_day_kg = pk_data.get("clearance_ml_day_kg")

    # ADA — P1d/P3b: explicit "Not assessed" instead of empty string
    ada = cache.get("ada_result") or extras.get("ada_result", {})
    if isinstance(ada, dict) and ada:
        _ada_level = ada.get("ada_risk_level", "")
        if _ada_level:
            pk.ada_risk_level = _ada_level
            pk.ada_risk_score = ada.get("ada_risk_score")
            pk.n_mhcii_hotspots = (ada.get("n_high_risk", 0) or 0) + (ada.get("n_medium_risk", 0) or 0)
            pk.ada_note = f"ADA risk: {_ada_level}"
        else:
            pk.ada_risk_level = "Not assessed"
            pk.ada_note = "Immunogenicity assessment not yet performed."
    else:
        pk.ada_risk_level = "Not assessed"
        pk.ada_note = "Immunogenicity assessment not yet performed."

    # DoE Purification Optimization
    # doe_to_dict() uses keys: optimal_ph, optimal_yield, optimal_purity, optimal_resolution
    form = cache.get("formulation_result")
    if isinstance(form, dict):
        pk.doe_optimal_ph = form.get("optimal_ph")
        pk.doe_optimal_yield = form.get("optimal_yield")
        pk.doe_optimal_purity = form.get("optimal_purity")
        pk.doe_rs_min = form.get("optimal_resolution") or form.get("rs_min")
        if pk.doe_optimal_yield is not None:
            pk.doe_note = (
                f"DoE optimization: pH {pk.doe_optimal_ph or '?'}, "
                f"yield {pk.doe_optimal_yield:.1%}, "
                f"purity {pk.doe_optimal_purity or '?'}%"
            )

    # Stability Twin (ICH shelf-life)
    stab = cache.get("stability_result", {})
    if isinstance(stab, dict) and stab:
        pk.shelf_life_months = stab.get("shelf_life_months")
        pk.stability_grade = stab.get("stability_grade")
        if pk.shelf_life_months is not None:
            pk.stability_note = (
                f"Predicted shelf life: {pk.shelf_life_months:.0f} months "
                f"(grade: {pk.stability_grade or 'N/A'})"
            )

    # COGS (commercial manufacturing cost)
    cogs = cache.get("cogs_result", {})
    if isinstance(cogs, dict) and cogs:
        pk.cogs_per_gram = cogs.get("cogs_per_gram")
        pk.cogs_cost_rating = cogs.get("cost_rating")
        if pk.cogs_per_gram is not None:
            pk.cogs_note = (
                f"Estimated COGS: ${pk.cogs_per_gram:.2f}/g "
                f"({pk.cogs_cost_rating or 'N/A'})"
            )

    return pk


def _build_validation_plan(
    ctx: ReportContext, cache: Dict, intent: Dict,
) -> ValidationPlanSection:
    """Build Section 6: Validation Plan."""
    vp = ValidationPlanSection()

    try:
        from src.validation_planner import generate_validation_plan
        dev_data = _get_dev_data(cache)
        preds = dev_data.get("predictions", {})

        plan = generate_validation_plan(
            risk_scores=preds,
            intent=intent,
            molecule_class=ctx.molecule_class,
        )

        vp.total_assays = plan.get("total_assays", 0)
        vp.molecule_class_impact = plan.get("format_note", "")

        for a in plan.get("required_assays", []):
            vp.required_assays.append({
                "name": a["name"], "priority": a["priority"],
                "reason": a.get("trigger_reason", "ICH Q6B standard"),
            })
        for a in plan.get("format_specific_assays", []):
            vp.format_specific_assays.append({
                "name": a["name"], "priority": a["priority"],
                "reason": a.get("trigger_reason", ""),
                "explanation": a.get("explanation", ""),
            })
        for a in plan.get("risk_triggered_assays", []):
            vp.risk_triggered_assays.append({
                "name": a["name"], "priority": a["priority"],
                "reason": a.get("trigger_reason", ""),
            })
        for a in plan.get("excluded_assays", []):
            vp.excluded_assays.append({
                "name": a["name"], "reason": a.get("reason", ""),
            })
        vp.key_recommendations = plan.get("recommendations", [])

    except Exception as e:
        log.warning("Validation plan assembly failed: %s", e)

    return vp


def _build_model_metadata(
    ctx: ReportContext, cache: Dict, intent: Dict,
) -> ModelMetadata:
    """Build Section 7: Model / Route Metadata."""
    mm = ModelMetadata()

    # Read embedding_mode from dev_result data (preferred), then cache fallback
    _dev_data_mm = _get_dev_data(cache)
    embed_mode = (
        _dev_data_mm.get("embedding_mode")
        or cache.get("embedding_mode", "mock")
    )
    pred_source = ctx.prediction_mode
    pred_detail = cache.get("predictor_detail", "")
    ood_reason = cache.get("ood_bypass_reason")

    if ctx.source == "fasta":
        _embed_label = "ESM-2 Embedding" if embed_mode == "esm2" else "Composition Embedding"
        _pred_label = "XGBoost" if pred_source == "xgboost" else "Rule-Based"
        mm.analysis_route = f"FASTA \u2192 Biopython \u2192 {_embed_label} \u2192 {_pred_label}"
    else:
        mm.analysis_route = "Text Parameters \u2192 Rule-Based Heuristic"

    mm.model_source = pred_detail or pred_source
    mm.embedding_mode = "ESM-2" if embed_mode == "esm2" else "Mock (Composition)"
    mm.prediction_mode = pred_source

    if pred_source == "xgboost":
        mm.heuristic_ml_hybrid = "ML"
        _base_conf = "High"
        mm.confidence_rationale = (
            "Predictions backed by XGBoost model trained on biophysical features."
        )
    else:
        mm.heuristic_ml_hybrid = "Heuristic"
        _base_conf = "Medium"
        mm.confidence_rationale = (
            "Predictions from rule-based heuristics using biophysical property mapping."
        )

    if ood_reason:
        mm.is_ood = True
        mm.ood_reason = ood_reason
        mm.caveats.append(f"Out-of-distribution warning: {ood_reason}")

    # OOD caps confidence — model may be ML-backed but sequence is outside training domain
    mm.confidence_level = _effective_confidence(_base_conf, ctx)
    if ctx.is_ood and mm.confidence_level != _base_conf:
        mm.confidence_rationale += (
            f" Confidence capped to {mm.confidence_level} due to out-of-distribution "
            f"sequence features ({ctx.ood_reason[:80]}...)."
        )

    # Molecule class benchmark note
    if ctx.molecule_class == "canonical_mab":
        mm.molecule_class_benchmark_note = (
            "Canonical mAb \u2014 well-represented in training data and validation benchmarks."
        )
    elif ctx.molecule_class == "unknown":
        mm.molecule_class_benchmark_note = (
            f"Molecule class: unknown ({len(ctx.chains)} chain(s) detected). "
            "Analysis defaults to canonical IgG assumptions. All risk scores, PK predictions, "
            "and validation recommendations assume standard mAb behavior."
        )
        mm.caveats.append(
            "Classification uncertain \u2014 all downstream analysis uses canonical IgG defaults."
        )
    elif ctx.molecule_class in ("bispecific", "adc", "fc_fusion"):
        mm.molecule_class_benchmark_note = (
            f"{ctx.molecule_class_display} format \u2014 limited training data representation. "
            "Risk weights adjusted; absolute scores may be less calibrated."
        )
        mm.caveats.append(
            f"Benchmark coverage for {ctx.molecule_class_display} is limited."
        )
    else:
        mm.molecule_class_benchmark_note = (
            f"{ctx.molecule_class_display} \u2014 minimal benchmark coverage. "
            "Results should be treated as exploratory."
        )

    # Confidence conclusions
    if ctx.source == "fasta":
        mm.high_confidence_conclusions.append("Sequence-derived properties (pI, MW, liabilities)")
        mm.high_confidence_conclusions.append("PTM hotspot identification")
    if ctx.prediction_mode == "xgboost":
        mm.high_confidence_conclusions.append("Developability risk scores (ML-backed)")
    else:
        mm.low_confidence_conclusions.append("Developability risk scores (heuristic)")
    mm.low_confidence_conclusions.append("Absolute shelf-life predictions")
    mm.low_confidence_conclusions.append("In vivo PK extrapolations")

    return mm


def _build_regulatory_context(
    ctx: ReportContext, intent: Dict,
) -> RegulatoryContextSection:
    """Build Section 8b: Regulatory signal classification (optional).

    Uses the cross-repo bridge to reg-intel-biopharma's Policy-Signal
    Classifier. If the classifier is unavailable, returns an unassessed
    section so the report is still complete.
    """
    section = RegulatoryContextSection()

    # Build a representative text from the molecule context
    claim_text = intent.get("regulatory_claim", "")
    if not claim_text:
        # Fallback: synthesize from molecule name and recommendation
        name = ctx.molecule_name or intent.get("name", "")
        rec = ctx.overall_grade or ""
        if name:
            claim_text = f"Regulatory assessment for {name}. Risk grade: {rec}."

    if not claim_text.strip():
        section.note = "no regulatory text available for classification"
        return section

    try:
        from src.regulatory_context import assess_regulatory_context

        result = assess_regulatory_context(claim_text)
        section.signal_class = result.get("signal_class", "unknown")
        section.probabilities = result.get("probabilities", {})
        section.source = result.get("source", "fallback")
        section.note = result.get("note", "")
        section.assessed = section.source == "policy_signal_classifier"
    except Exception as exc:
        log.debug("Regulatory context section skipped: %s", exc)
        section.note = f"classifier unavailable: {exc}"

    return section


def _build_appendix(ctx: ReportContext, intent: Dict) -> AppendixData:
    """
    Build Section 8: Selected raw metrics.

    v2.0 fix: Two-layer liability structure:
      1. liability_counts["_global_*"] — from ctx (same source as Developability section)
      2. liability_counts["{chain}_*"] — per-chain breakdown for detailed view
    This prevents chain-level re-counting from conflicting with global summary.
    """
    ap = AppendixData()

    _precision = {"pI": 4, "mw": 1, "gravy": 4, "hydrophobicity": 4}
    # Use ctx fields as authoritative source
    _fields = [
        ("pI", ctx.isoelectric_point),
        ("mw", ctx.molecular_weight_kda),
        ("gravy", ctx.gravy_score),
        ("hydrophobicity", ctx.hydrophobicity),
        ("deam_sites", ctx.deam_sites),
        ("ox_sites", ctx.ox_sites),
        ("isomerization_sites", ctx.isomerization_sites),
        ("n_glycosylation_sites", ctx.n_glycosylation_sites),
        ("dp_clip_sites", ctx.dp_clip_sites),
        ("seq_length", ctx.sequence_length),
        ("cysteine_count", ctx.cysteine_count),
    ]
    for key, val in _fields:
        if val is not None:
            if isinstance(val, float) and key in _precision:
                val = round(val, _precision[key])
            ap.biophysical_features[key] = val

    # ── Layer 1: Global liability summary (from ctx — matches Developability section) ──
    if ctx.deam_sites is not None:
        ap.liability_counts["_global_deamidation"] = ctx.deam_sites
    if ctx.ox_sites is not None:
        ap.liability_counts["_global_oxidation"] = ctx.ox_sites
    if ctx.isomerization_sites is not None:
        ap.liability_counts["_global_isomerization"] = ctx.isomerization_sites
    if ctx.n_glycosylation_sites is not None:
        ap.liability_counts["_global_n_glycosylation"] = ctx.n_glycosylation_sites
    if ctx.dp_clip_sites is not None:
        ap.liability_counts["_global_dp_clip"] = ctx.dp_clip_sites

    # ── Layer 2: Per-chain breakdown (for detailed view only) ──
    for ca in ctx.chain_analyses:
        liab = ca.get("liabilities", {})
        cname = ca.get("name", "?")
        ap.liability_counts[f"{cname}_met"] = liab.get("met_count", 0)
        ap.liability_counts[f"{cname}_trp"] = liab.get("trp_count", 0)
        ap.liability_counts[f"{cname}_deam"] = liab.get("deamidation_count", 0)
        ap.liability_counts[f"{cname}_nglyco"] = liab.get("n_glyco_count", 0)
        ap.liability_counts[f"{cname}_dp"] = liab.get("dp_count", 0)
        ap.liability_counts[f"{cname}_isomerization"] = liab.get("isomerization_count", 0)

    # CDR regions
    for ca in ctx.chain_analyses:
        for cdr in ca.get("cdrs", []):
            ap.cdr_regions.append({
                "chain": ca.get("name", ""),
                "cdr": cdr.get("name", ""),
                "sequence": cdr.get("sequence", ""),
                "length": cdr.get("end", 0) - cdr.get("start", 0),
            })

    for ca in ctx.chain_analyses:
        ap.chain_details.append({
            "name": ca.get("name", ""),
            "type": ca.get("chain_type", ""),
            "length": ca.get("length", 0),
        })

    return ap


# =====================================================================
# Unified Recommendation Logic
# =====================================================================

def _generate_recommendation(ctx: ReportContext) -> tuple:
    """
    Generate recommendation and detail text.

    Decision tree:
      1. overall_score → base recommendation (Proceed / Caution / Optimize)
      2. molecule_class → format-specific suffix
      3. prediction_mode → confidence qualifier
      4. evidence_completeness → completeness caveat

    Returns (recommendation_str, detail_str)
    """
    score = ctx.overall_score
    mol_cls = ctx.molecule_class
    is_rule_based = ctx.prediction_mode != "xgboost"

    if score is None:
        # P4a: Even without a score, provide format-aware guidance
        rec = "Assessment incomplete"
        if mol_cls in ("bispecific", "fusion_protein", "adc"):
            detail = (
                f"Developability score not available for this {mol_cls.replace('_', ' ')} format. "
                "Sequence-derived liabilities have been assessed but quantitative risk scoring "
                "requires model training on format-specific data. Prioritize: "
                "(1) species purity and assembly characterization, "
                "(2) format-specific PTM monitoring, "
                "(3) accelerated stability with format-relevant stress conditions."
            )
        else:
            detail = (
                "Developability score not available. Cannot generate recommendation. "
                "Run developability assessment to obtain risk scoring."
            )
        return rec, detail

    # Base recommendation from score (using unified thresholds)
    # Non-canonical formats get format-aware base language instead of
    # "Standard CMC pathway" to avoid over-optimistic framing.
    # Inverted logic: only canonical_mab gets standard wording; everything
    # else (single_domain, peptide, fc_fusion, bispecific, adc, unknown, …)
    # receives format-aware language automatically — prevents new molecule
    # classes from silently inheriting inappropriate mAb-centric text.
    _is_non_canonical = mol_cls not in ("canonical_mab",)
    grade = grade_from_score(score)
    if grade == "Low":
        rec = "Proceed"
        if _is_non_canonical:
            if mol_cls == "single_domain":
                detail = (
                    "Low sequence-derived developability risk for this nanobody/VHH. "
                    "Single-domain antibodies exhibit distinct aggregation, renal-clearance, "
                    "and thermal-stability profiles; format-specific characterization is recommended."
                )
            elif mol_cls == "peptide":
                detail = (
                    "Low sequence-derived developability risk for this peptide therapeutic. "
                    "Peptide-specific liabilities (protease susceptibility, formulation stability) "
                    "should be assessed with format-appropriate assays before advancement."
                )
            elif mol_cls == "engineered_scaffold":
                detail = (
                    "Low sequence-derived developability risk for this engineered scaffold format. "
                    "Scaffold-specific immunogenicity baseline and benchmark comparability "
                    "should be confirmed alongside standard characterization."
                )
            else:
                detail = (
                    "Low sequence-derived developability risk; however, format-specific "
                    "characterization pathways are recommended given limited benchmark "
                    "coverage for this molecule class."
                )
        else:
            detail = (
                "Low developability risk based on sequence-derived predictions. "
                "Standard CMC pathway recommended with routine characterization."
            )
    elif grade == "Medium":
        rec = "Proceed with caution"
        if _is_non_canonical:
            if mol_cls == "single_domain":
                detail = (
                    "Moderate developability risks identified for this nanobody/VHH. "
                    "Aggregation hotspot engineering, thermal stability optimization, and "
                    "half-life extension strategy should be addressed before late-stage advancement."
                )
            elif mol_cls == "peptide":
                detail = (
                    "Moderate developability risks identified for this peptide therapeutic. "
                    "Prioritize protease susceptibility mitigation, modification strategies, "
                    "and formulation optimization before late-stage advancement."
                )
            elif mol_cls == "engineered_scaffold":
                detail = (
                    "Moderate developability risks identified for this engineered scaffold. "
                    "Scaffold-specific optimization and immunogenicity assessment should be "
                    "prioritized before late-stage development."
                )
            else:
                detail = (
                    "Moderate risks identified in sequence-derived predictions. Format-specific "
                    "optimization of flagged dimensions is recommended, with emphasis on "
                    "characterization unique to this molecule class."
                )
        else:
            detail = (
                "Moderate risks identified in sequence-derived predictions. Targeted optimization "
                "of flagged dimensions recommended before late-stage advancement."
            )
    else:  # grade == "High"
        rec = "Optimize before proceeding"
        if _is_non_canonical:
            if mol_cls == "single_domain":
                detail = (
                    "Significant developability concerns for this nanobody/VHH from sequence-derived "
                    "predictions. Aggregation hotspot engineering, thermal stability improvement, "
                    "and half-life extension strategy are required before advancing."
                )
            elif mol_cls == "peptide":
                detail = (
                    "Significant developability concerns for this peptide therapeutic from "
                    "sequence-derived predictions. Protease susceptibility, poor predicted "
                    "solubility, or formulation incompatibility require engineering solutions "
                    "before advancing."
                )
            elif mol_cls == "engineered_scaffold":
                detail = (
                    "Significant developability concerns for this engineered scaffold from "
                    "sequence-derived predictions. Scaffold re-engineering or immunogenicity "
                    "mitigation is required before advancement."
                )
            else:
                detail = (
                    "Significant developability concerns from sequence-derived predictions for "
                    "this non-standard molecule format. Format-specific engineering modifications "
                    "or variant screening are recommended before advancing."
                )
        else:
            detail = (
                "Significant developability concerns from sequence-derived predictions. Engineering "
                "modifications or variant screening recommended before advancing."
            )

    # Molecule-specific detail enrichment
    if mol_cls == "bispecific":
        detail += (
            " For this bispecific format: establish species purity (AA/AB/BB) monitoring "
            "from early development; validate dual-target binding by SPR/BLI; "
            "prioritize assembly-related PTM and clipping control; and plan for "
            "homodimer/heterodimer separation characterization."
        )
    elif mol_cls in ("fusion_protein", "fc_fusion"):
        detail += (
            " For this fusion protein: confirm linker integrity under stress; "
            "assess domain-domain interactions; monitor for clipping at fusion junctions."
        )
    elif mol_cls == "adc":
        detail += (
            " For this ADC: establish DAR distribution control; "
            "validate conjugation site integrity and payload stability."
        )
    else:
        # Canonical or other — use existing suffix if available
        suffix = MOLECULE_RECOMMENDATION_SUFFIX.get(mol_cls, "")
        if suffix:
            detail += suffix

    # Confidence qualifier for heuristic-only predictions
    if is_rule_based:
        detail += (
            " Note: predictions are heuristic-based (no trained ML model); "
            "experimental confirmation is strongly recommended."
        )

    # OOD uncertainty qualifier — inject when sequence is outside training domain
    if ctx.is_ood:
        detail += (
            " Note: this molecule's sequence features fall outside the training "
            "distribution (out-of-distribution detected); quantitative risk scores "
            "carry additional uncertainty and should be confirmed experimentally."
        )

    return rec, detail


# =====================================================================
# Top Risks Generator (Tier-prioritized)
# =====================================================================

def _generate_top_risks(ctx: ReportContext) -> List[str]:
    """
    Generate top risk statements ranked by scientific severity.

    Each risk gets a severity score = base_severity × tier_multiplier × format_relevance.
    - Tier 1 (sequence-derived) gets tier_multiplier = 1.0
    - Tier 2 (predicted)        gets tier_multiplier = 0.8
    - format_relevance: for bispecific, assembly/species risks get 1.5× boost

    Results are sorted by severity (descending), then formatted with [Tier N] labels.
    """
    # (severity_score, tier_label, text)
    scored_risks: list = []
    _is_bispecific = ctx.molecule_class in ("bispecific", "unknown") and len(ctx.chains) > 2

    # ── Tier 1: Sequence-derived liabilities ──
    # Multi-chain assembly risk — structural integrity, ranked HIGH for bispecific
    if len(ctx.chains) > 2 and _is_bispecific:
        _fmt_boost = 1.5  # format-relevance boost
        scored_risks.append((
            0.9 * _fmt_boost,
            "Tier 1",
            f"Multi-chain assembly ({len(ctx.chains)} chains) \u2014 species heterogeneity "
            "and mispairing must be monitored; establish AA/AB/BB species control early"
        ))

    if ctx.dp_clip_sites is not None and ctx.dp_clip_sites > 0:
        _base = min(0.4 + ctx.dp_clip_sites * 0.15, 0.95)
        _boost = 1.3 if _is_bispecific else 1.0  # clipping more impactful for complex formats
        scored_risks.append((
            _base * _boost,
            "Tier 1",
            f"Asp-Pro clipping risk ({ctx.dp_clip_sites} DP sites) \u2014 "
            "potential fragment generation during processing; monitor by CE-SDS"
        ))

    if ctx.isomerization_sites is not None and ctx.isomerization_sites > 2:
        _base = min(0.3 + ctx.isomerization_sites * 0.05, 0.85)
        scored_risks.append((
            _base,
            "Tier 1",
            f"Asp isomerization hotspots ({ctx.isomerization_sites} DG/DS/DD sites) \u2014 "
            "potential potency loss over shelf-life"
        ))

    if ctx.deam_sites is not None and ctx.deam_sites > 3:
        _base = min(0.25 + (ctx.deam_sites - 3) * 0.04, 0.8)
        scored_risks.append((
            _base,
            "Tier 1",
            f"Elevated deamidation liability ({ctx.deam_sites} hotspots) \u2014 "
            "monitor with MAM/peptide mapping during development"
        ))

    if ctx.ox_sites is not None and ctx.ox_sites > 6:
        _base = min(0.2 + (ctx.ox_sites - 6) * 0.02, 0.75)
        scored_risks.append((
            _base,
            "Tier 1",
            f"Multiple oxidation-prone residues ({ctx.ox_sites} Met+Trp sites)"
        ))

    if ctx.cysteine_count > 0 and ctx.molecule_class == "canonical_mab":
        # Stoichiometry-aware: compute expected Cys from actual chains × copy_number.
        # IgG1 tetramer (2HC + 2LC) has ~32 Cys total, not 16 per monomer.
        if ctx.chains and len(ctx.chains) > 0:
            expected_cys = sum(
                c.get("sequence", "").upper().count("C") * c.get("copy_number", 1)
                for c in ctx.chains
            )
            # Fallback: if chains lack sequences, estimate from molecule class
            if expected_cys == 0:
                expected_cys = 32  # standard IgG1 tetramer default
        else:
            expected_cys = 32  # standard IgG1 tetramer default
        if ctx.cysteine_count != expected_cys:
            scored_risks.append((
                0.65,
                "Tier 1",
                f"Unexpected cysteine count ({ctx.cysteine_count} vs {expected_cys} "
                f"expected for assembled stoichiometry) — "
                "free cysteine or disulfide mis-pairing risk"
            ))

    # ── Tier 2: Predicted risks (moderate confidence, 0.8× tier penalty) ──
    if ctx.agg_risk is not None and ctx.agg_risk > 0.4:
        _sev = ctx.agg_risk * 0.8
        detail = f"High aggregation risk (score {ctx.agg_risk:.2f})"
        if ctx.hydrophobicity is not None and ctx.hydrophobicity > 0.45:
            detail += f" \u2014 elevated surface hydrophobicity ({ctx.hydrophobicity:.3f})"
        scored_risks.append((_sev, "Tier 2", detail))

    if ctx.stability is not None and ctx.stability < 0.65:
        _sev = (1.0 - ctx.stability) * 0.8
        scored_risks.append((
            _sev,
            "Tier 2",
            f"Low thermal stability (score {ctx.stability:.2f}) \u2014 risk of degradation "
            "during processing/storage"
        ))

    if ctx.viscosity_risk is not None and ctx.viscosity_risk > 0.4:
        _sev = ctx.viscosity_risk * 0.8
        detail = f"High viscosity risk (score {ctx.viscosity_risk:.2f})"
        if ctx.molecular_weight_kda and ctx.molecular_weight_kda > 160:
            detail += f" \u2014 large MW ({ctx.molecular_weight_kda:.0f} kDa)"
        scored_risks.append((_sev, "Tier 2", detail))

    # Sort by severity descending
    scored_risks.sort(key=lambda x: x[0], reverse=True)

    if not scored_risks:
        return ["No major risks identified"]

    # Format with tier labels
    return [f"[{tier}] {text}" for _, tier, text in scored_risks[:5]]


# =====================================================================
# Top Strengths Generator
# =====================================================================

def _generate_top_strengths(ctx: ReportContext) -> List[str]:
    """Generate strengths — only from assessed dimensions."""
    strengths: List[str] = []

    if ctx.agg_risk is not None and ctx.agg_risk <= 0.2:
        strengths.append(f"Low aggregation propensity (score {ctx.agg_risk:.2f})")
    if ctx.stability is not None and ctx.stability >= 0.8:
        strengths.append(f"Good thermal stability (score {ctx.stability:.2f})")
    if ctx.viscosity_risk is not None and ctx.viscosity_risk <= 0.15:
        s = f"Low viscosity risk (score {ctx.viscosity_risk:.2f})"
        if ctx.molecular_weight_kda and ctx.molecular_weight_kda < 160:
            s += f" \u2014 MW {ctx.molecular_weight_kda:.0f} kDa favorable for SC delivery"
        strengths.append(s)
    if ctx.isoelectric_point and 6.5 <= ctx.isoelectric_point <= 9.0:
        strengths.append(
            f"pI {ctx.isoelectric_point:.1f} within favorable range for standard purification"
        )

    return strengths


# =====================================================================
# Caveats and Limitations
# =====================================================================

def _generate_caveats(ctx: ReportContext) -> List[str]:
    caveats: List[str] = []

    if ctx.molecule_class == "unknown":
        n = len(ctx.chains)
        caveats.append(
            f"Molecule classification: unknown ({n} chain(s) detected). "
            "Analysis defaults to canonical IgG assumptions. Verify format experimentally."
        )
    elif ctx.molecule_class != "canonical_mab":
        caveats.append(
            f"Molecule classified as {ctx.molecule_class_display}. "
            "Risk weights adjusted for this format; absolute scores may be less calibrated."
        )

    if ctx.prediction_mode != "xgboost":
        caveats.append(
            "Predictions based on rule-based heuristics (no ML model trained). "
            "Confidence is lower than XGBoost-backed predictions."
        )

    if ctx.is_ood:
        caveats.append(
            f"Out-of-distribution molecule detected — sequence features fall outside "
            f"the training domain. Risk scores carry additional uncertainty."
        )

    return caveats


def _generate_limitations(ctx: ReportContext) -> List[str]:
    limitations: List[str] = []

    if ctx.molecule_class == "unknown":
        limitations.append(
            "Molecule format unclassified \u2014 all predictions default to canonical IgG assumptions."
        )
    if ctx.prediction_mode != "xgboost":
        limitations.append(
            "Analysis uses rule-based heuristics only (no trained ML model). "
            "Absolute risk scores are low-confidence estimates."
        )
    if not ctx.has_qc_data:
        limitations.append(
            "No analytical data (SEC, CE-SDS, cIEF) \u2014 "
            "all quality metrics are sequence-derived predictions."
        )
    elif ctx.qc_is_simulated:
        limitations.append(
            "Analytical QC data is from virtual simulation (Tier 3 evidence), "
            "not experimental measurement. Limited discriminatory power."
        )

    return limitations


# =====================================================================
# Narrative-Aligned Dimension Explanations
# =====================================================================

def _build_dimension_explanation(
    dim_name: str, key: str, val: float, grade: str, ctx: ReportContext,
) -> str:
    """
    Build explanation text with language intensity aligned to grade.

    NARRATIVE_MAP ensures:
    - Low grade → favorable/manageable language
    - Medium grade → moderate/monitor language
    - High grade → elevated/significant language
    """
    tone = NARRATIVE_MAP.get(grade, NARRATIVE_MAP["Medium"])["tone"]

    if key == "agg_risk":
        if grade == "High":
            return (
                f"Elevated aggregation risk driven by surface hydrophobic patches. "
                f"Significant concern for manufacturing and storage stability. "
                f"Formulation optimization and aggregation mitigation are required."
            )
        elif grade == "Medium":
            return (
                f"Moderate aggregation propensity. Monitor with SEC during development. "
                f"Additional formulation screening may help reduce risk."
            )
        else:
            return (
                f"Low aggregation risk. Favorable for manufacturing and storage."
            )

    elif key == "stability":
        if grade == "High":
            return (
                "Low thermal stability indicates significant risk of degradation "
                "during processing and storage. Action required: formulation optimization."
            )
        elif grade == "Medium":
            return (
                "Moderate stability. Formulation optimization (pH, excipients) may "
                "be needed. Monitor during accelerated stability studies."
            )
        else:
            return "Good thermal stability expected under standard conditions."

    elif key == "viscosity_risk":
        if grade == "High":
            return (
                "Elevated viscosity risk may significantly impede subcutaneous delivery "
                "at high concentration. Action required: evaluate concentration limits."
            )
        elif grade == "Medium":
            return (
                "Moderate viscosity concern. Measure experimentally at target concentration."
            )
        else:
            return "Low viscosity risk. Suitable for high-concentration formulation."

    return f"{dim_name}: {grade} risk (score {val:.2f})"


def _build_dimension_drivers(
    key: str, val: float, ctx: ReportContext,
) -> List[str]:
    """Build primary driver list for a risk dimension."""
    drivers: List[str] = []
    if key == "agg_risk":
        if ctx.hydrophobicity is not None:
            drivers.append(f"Hydrophobicity: {ctx.hydrophobicity:.3f}")
    elif key == "stability":
        drivers.append(f"Stability score: {val:.3f}")
    elif key == "viscosity_risk":
        if ctx.molecular_weight_kda:
            drivers.append(f"MW: {ctx.molecular_weight_kda:.1f} kDa")
    return drivers


# =====================================================================
# Cross-Section Consistency Validation
# =====================================================================

def _validate_cross_section_consistency(report: ReportObject) -> None:
    """
    Final pass to detect and fix cross-section inconsistencies.

    v2.0 expanded: 10 checks covering all critical cross-section fields.
    Does NOT recalculate science — only ensures fields agree across sections.
    """
    ctx = report.context
    es = report.executive_summary
    mo = report.molecule_overview
    ds = report.developability
    mm = report.model_metadata
    ap = report.appendix

    issues: List[str] = []

    # ── 1. Molecule class consistency (ES = MO = ctx) ──
    if es.molecule_class != mo.molecule_class:
        issues.append(f"molecule_class: ES={es.molecule_class} vs MO={mo.molecule_class}")
        mo.molecule_class = es.molecule_class

    # ── 2. MW consistency between MO and Appendix ──
    ap_mw = ap.biophysical_features.get("mw")
    if mo.molecular_weight_kda is not None and ap_mw is not None:
        if abs(mo.molecular_weight_kda - ap_mw) > 0.5:
            issues.append(f"MW mismatch: MO={mo.molecular_weight_kda} vs Appendix={ap_mw}")
            ap.biophysical_features["mw"] = round(mo.molecular_weight_kda, 1)

    # ── 3. Grade/recommendation consistency (ES = DS) ──
    if es.overall_grade and ds.composite_grade:
        if es.overall_grade != ds.composite_grade:
            issues.append(f"Grade mismatch: ES={es.overall_grade} vs DS={ds.composite_grade}")
            ds.composite_grade = es.overall_grade

    if es.recommendation and ds.recommendation:
        if es.recommendation != ds.recommendation:
            issues.append(f"Recommendation mismatch: ES={es.recommendation} vs DS={ds.recommendation}")
            ds.recommendation = es.recommendation

    # ── 4. Liability summary vs QTPP consistency (deamidation) ──
    if ds.liability_summary and ds.qtpp_rows:
        deam_in_liab = ds.liability_summary.get("deamidation_sites")
        for row in ds.qtpp_rows:
            if "deamidation" in row.get("attribute", "").lower() and deam_in_liab is not None:
                expected_current = str(deam_in_liab)
                if expected_current not in str(row.get("current_prediction", "")):
                    issues.append(
                        f"Deam mismatch: liability={deam_in_liab} vs QTPP={row.get('current_prediction')}"
                    )

    # ── 5. Glycosylation consistency (DS liability = Appendix global = QTPP) ──
    ds_glyco = ds.liability_summary.get("glycosylation_sites")
    ap_glyco = ap.liability_counts.get("_global_n_glycosylation")
    if ds_glyco is not None and ap_glyco is not None and ds_glyco != ap_glyco:
        issues.append(f"Glycosylation mismatch: DS={ds_glyco} vs Appendix._global={ap_glyco}")
        ap.liability_counts["_global_n_glycosylation"] = ds_glyco

    # ── 6. Isomerization consistency (DS liability = Appendix global) ──
    ds_iso = ds.liability_summary.get("isomerization_sites")
    ap_iso = ap.liability_counts.get("_global_isomerization")
    if ds_iso is not None and ap_iso is not None and ds_iso != ap_iso:
        issues.append(f"Isomerization mismatch: DS={ds_iso} vs Appendix._global={ap_iso}")
        ap.liability_counts["_global_isomerization"] = ds_iso

    # ── 7. Appendix global deam vs DS liability summary ──
    ds_deam = ds.liability_summary.get("deamidation_sites")
    ap_deam = ap.liability_counts.get("_global_deamidation")
    if ds_deam is not None and ap_deam is not None and ds_deam != ap_deam:
        issues.append(f"Deam (global) mismatch: DS={ds_deam} vs Appendix._global={ap_deam}")
        ap.liability_counts["_global_deamidation"] = ds_deam

    # ── 8. Appendix chain-level sum vs global cross-check (warn only, don't fix) ──
    _chain_deam_sum = sum(
        v for k, v in ap.liability_counts.items()
        if k.endswith("_deam") and not k.startswith("_global")
    )
    if ds_deam is not None and _chain_deam_sum > 0 and _chain_deam_sum != ds_deam:
        issues.append(
            f"Deam chain-sum={_chain_deam_sum} differs from global={ds_deam} "
            f"(expected — global may apply deduplication or different counting rules)"
        )
        # NOTE: Do NOT auto-fix. This is an informational warning. Chain-level
        # and global may differ legitimately due to overlapping motifs, etc.

    # ── 9. Unknown/OOD: no overconfident language ──
    if ctx.molecule_class == "unknown" or mm.is_ood:
        overconfident_terms = ["confirmed", "validated", "definitive", "conclusive"]
        for term in overconfident_terms:
            if term in es.recommendation_detail.lower():
                issues.append(f"Overconfident term '{term}' in ES for unknown/OOD molecule")
                es.recommendation_detail = es.recommendation_detail.replace(
                    term, f"{term} (pending format confirmation)"
                )

    # ── 10. Recommendation detail vs grade/confidence/OOD coherence ──
    if es.overall_grade in ("High Risk", "Elevated Risk"):
        # Recommendation should NOT say "Proceed" without qualification
        if es.recommendation and "proceed" in es.recommendation.lower():
            if "caution" not in es.recommendation.lower() and "optimize" not in es.recommendation.lower():
                issues.append(
                    f"Grade={es.overall_grade} but recommendation='{es.recommendation}' — "
                    f"expected 'Proceed with caution' or 'Optimize'"
                )

    # ── 11. Molecule class vs validation language consistency ──
    vp = report.validation_plan
    if vp and hasattr(vp, "assays"):
        # If molecule is bispecific, validation should mention species purity
        if ctx.molecule_class == "bispecific":
            _vp_text = " ".join(
                a.get("name", "") + " " + a.get("rationale", "")
                for a in (vp.assays if hasattr(vp, "assays") else [])
                if isinstance(a, dict)
            ).lower()
            if not any(kw in _vp_text for kw in ["species", "homodimer", "bispecific", "heterodimer"]):
                issues.append(
                    "Bispecific molecule but validation plan has no species purity mention"
                )

    if issues:
        log.warning("Cross-section consistency: %d issues found and corrected: %s",
                     len(issues), "; ".join(issues))


# =====================================================================
# Helpers
# =====================================================================

def _get_dev_data(cache: Dict) -> Dict:
    """Extract dev_result data dict from cache."""
    raw = cache.get("dev_result", {})
    if isinstance(raw, dict):
        return raw.get("data", raw)
    return {}


def _effective_confidence(model_conf: str, ctx) -> str:
    """Return the lower of model confidence and OOD-derived confidence.

    Confidence ranking: High > Medium > Low.
    When a molecule is OOD (out-of-distribution), its confidence cap may be
    lower than the ML model's intrinsic confidence. We take the minimum.
    """
    _rank = {"High": 3, "Medium": 2, "Low": 1}
    if not getattr(ctx, "is_ood", False):
        return model_conf
    ood_conf = getattr(ctx, "ood_confidence", "High")
    if _rank.get(ood_conf, 3) < _rank.get(model_conf, 3):
        return ood_conf
    return model_conf


def _int_or_none(val) -> Optional[int]:
    """Convert to int or None."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# =====================================================================
# Self-test
# =====================================================================

def _selftest():
    """Minimal self-test: assemble a report from mock data."""
    mock_intent = {
        "name": "Trastuzumab-test",
        "source": "fasta",
        "molecule_class": "canonical_mab",
        "molecule_class_info": {
            "display_name": "Canonical mAb (IgG)",
            "confidence": "High",
            "has_fc_region": True,
            "expects_glycosylation": True,
            "evidence": ["IgG motifs detected"],
        },
        "pI": 8.45, "mw": 145.4, "gravy": -0.326,
        "hydrophobicity": 0.42, "deam_sites": 3, "ox_sites": 5,
        "acidic_residues": 80, "basic_residues": 95,
        "seq_length": 1320, "cysteine_count": 32,
        "chains": [
            {"name": "HC", "sequence": "EVQLVES...", "copy_number": 2},
            {"name": "LC", "sequence": "DIQMTQ...", "copy_number": 2},
        ],
        "chain_analyses": [
            {"name": "HC", "chain_type": "heavy", "length": 449,
             "cdrs": [{"name": "CDR-H1", "start": 26, "end": 35, "sequence": "GYTFTSYG"}],
             "liabilities": {"met_count": 3, "trp_count": 2, "deamidation_count": 2,
                             "n_glyco_count": 1, "dp_count": 0, "isomerization_count": 1,
                             "cys_count": 11, "acidic_count": 40, "basic_count": 48}},
        ],
        "liability_summary": {
            "deamidation_sites": 3, "oxidation_sites": 5,
            "asp_isomerization_sites": 2, "dp_sites": 0,
            "n_glycosylation_sites": 2,
        },
    }

    mock_cache = {
        "dev_result": {
            "data": {
                "predictions": {"agg_risk": 0.18, "stability": 0.85, "viscosity_risk": 0.07},
                "score": {"score": 0.15, "grade": "Low Risk"},
                "mode": "rule_based",
            }
        },
        "predictor_source": "rule_based",
        "predictor_detail": "Rule-Based Heuristic (PropertyMapper v5.0)",
    }

    report = assemble_report(mock_intent, mock_cache)

    # Verify ReportContext is authoritative
    assert report.context.molecule_name == "Trastuzumab-test"
    assert report.context.molecule_class == "canonical_mab"
    assert report.context.deam_sites == 3
    assert report.context.ox_sites == 5
    assert report.context.agg_risk == 0.18
    assert report.context.stability == 0.85

    # Verify sections read from ctx
    assert report.executive_summary.molecule_name == "Trastuzumab-test"
    # 5-dim composite replaces 3-dim predictor score; base_risk_score keeps original
    assert report.context.base_risk_score == 0.15, \
        f"base_risk_score should be 0.15 (3-dim), got {report.context.base_risk_score}"
    assert report.executive_summary.overall_score is not None, \
        "overall_score should not be None after 5-dim composite"
    assert 0.0 <= report.executive_summary.overall_score <= 1.0, \
        f"overall_score out of range: {report.executive_summary.overall_score}"
    assert report.molecule_overview.molecular_weight_kda == 145.4
    assert len(report.developability.risk_dimensions) == 3

    # Verify no "Not assessed" for fully-assessed dimensions
    for rd in report.developability.risk_dimensions:
        assert rd.assessed, f"{rd.dimension} should be assessed"
        assert rd.grade != "Not assessed"

    # Verify grade uses unified thresholds
    assert report.executive_summary.overall_grade == "Low Risk"
    assert report.developability.composite_grade == "Low Risk"

    # Verify recommendation consistency (cross-section check)
    assert report.executive_summary.recommendation == report.developability.recommendation

    # Verify evidence tiers
    for rd in report.developability.risk_dimensions:
        assert rd.evidence_tier == EVIDENCE_TIER_2

    # Verify molecule-aware recommendation (canonical_mab has no suffix)
    assert "format-specific" not in report.executive_summary.recommendation_detail

    # Verify top_risks use tier tags
    for risk in report.executive_summary.top_risks:
        if risk != "No major risks identified":
            assert "[" in risk, f"Risk should have tier tag: {risk}"

    # Verify model metadata
    assert report.model_metadata.heuristic_ml_hybrid == "Heuristic"

    # Verify validation plan
    assert report.validation_plan.total_assays >= 3

    # Verify appendix uses ctx values
    assert report.appendix.biophysical_features.get("pI") == 8.45

    # JSON round-trip
    j = report.to_json()
    assert "Trastuzumab-test" in j
    assert "canonical_mab" in j
    assert "evidence_tier" in j

    # ── Test with missing predictions ──
    sparse_intent = {
        "name": "Sparse-molecule",
        "source": "text",
        "molecule_class": "unknown",
    }
    sparse_cache = {}
    sparse_report = assemble_report(sparse_intent, sparse_cache)

    # Should NOT crash; should show "Not assessed" for all dimensions
    assert sparse_report.context.agg_risk is None
    assert sparse_report.context.overall_score is None
    for rd in sparse_report.developability.risk_dimensions:
        assert not rd.assessed
        assert rd.grade == "Not assessed"

    # Recommendation should indicate incomplete
    assert "incomplete" in sparse_report.executive_summary.recommendation.lower() or \
           "Assessment" in sparse_report.executive_summary.recommendation

    print("ReportAssembler v2.0 _selftest PASS")
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    _selftest()
