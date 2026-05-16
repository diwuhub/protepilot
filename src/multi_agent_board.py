"""
src/multi_agent_board.py — Multi-Agent CMC Board
==================================================
ProtePilot — Milestone 22 · Version 1.0

Simulates a cross-functional CMC Board meeting with three distinct
AI personas that assess the current molecule from their domain
perspective, debate risks, and reach a regulatory verdict.

Agents
------
  1. Upstream Expert   — Titer, viability, scale-up feasibility
  2. Downstream Expert — Resolution, yield, liability clearance
  3. Regulatory / QA — Safety, aggregation/oxidation risk, quality attributes, filing

Pipeline: Upstream → Downstream critiques → Regulatory delivers verdict

When an OpenAI key is available, each agent is a separate API call
with a domain-specific system prompt. Without a key, a robust mock
engine generates context-aware assessments from session data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ===========================================================================
# 1. Data Classes
# ===========================================================================

@dataclass
class AgentStatement:
    """One agent's contribution to the board meeting."""
    agent_name: str
    agent_role: str
    icon: str
    statement: str
    risk_flags: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class BoardMeetingResult:
    """Result of a CMC Board meeting."""
    statements: List[AgentStatement]
    consensus: str
    risk_level: str                              # "Low" / "Medium" / "High" / "Critical"
    risk_assessment: str                         # "Favorable" / "Manageable" / "Elevated" / "Significant"
    mitigation_recommendations: List[str]        # Specific mitigations
    next_steps: str                              # Suggested path forward
    summary: str
    wall_time_s: float


# ===========================================================================
# 2. Session Data Extractor
# ===========================================================================

def _extract_board_data(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize all relevant data for the board meeting."""
    intent = session_data.get("last_intent") or {}
    # Fallback: if last_intent is empty/missing, pull from workspace store
    if not intent or not intent.get("sequence"):
        _ws = session_data.get("_workspace_store")
        if _ws and hasattr(_ws, "get_active"):
            _active_ws = _ws.get_active()
            if _active_ws and isinstance(_active_ws, dict):
                _ws_intent = _active_ws.get("intent")
                if _ws_intent and isinstance(_ws_intent, dict):
                    intent = _ws_intent
    up = session_data.get("upstream_result_dict") or {}
    doe = session_data.get("doe_result_dict") or {}
    cogs = session_data.get("cogs_result_dict") or {}

    # Developability
    dev_score = None
    agg_risk = None
    stability = None
    _ws = session_data.get("_workspace_store")
    _active = None
    if _ws and hasattr(_ws, "get_active"):
        _active = _ws.get_active()
    if _active and isinstance(_active, dict):
        _cache = _active.get("analysis_cache") or {}
        _dev = (_cache.get("dev_result") or {}).get("data", {})
        if _dev:
            dev_score = _dev.get("composite_score")
            preds = _dev.get("predictions", {})
            agg_risk = preds.get("agg_risk")
            stability = preds.get("stability")

    # ADA / Immunogenicity
    ada_risk_level = None
    _ada_result = session_data.get("ada_result")
    if _ada_result and hasattr(_ada_result, "ada_risk_level"):
        ada_risk_level = _ada_result.ada_risk_level
    elif isinstance(_ada_result, dict):
        ada_risk_level = _ada_result.get("ada_risk_level")

    # Sequence composition analysis for quality assessment
    sequence = intent.get("sequence", "")
    seq_upper = sequence.upper() if sequence else ""
    seq_len = len(seq_upper)
    aromatic_frac = ((seq_upper.count("W") + seq_upper.count("Y") + seq_upper.count("F"))
                     / max(seq_len, 1)) if seq_len > 0 else 0.0
    cys_frac = seq_upper.count("C") / max(seq_len, 1) if seq_len > 0 else 0.0

    # Repeat / low-complexity detection (Shannon entropy + consecutive repeats)
    repeat_alert = False
    repeat_severity = None   # "warning" or "critical"
    if seq_len > 20:
        # 1. Shannon entropy of amino acid composition
        from collections import Counter as _Counter
        _aa_counts = _Counter(seq_upper)
        _aa_freqs = [c / seq_len for c in _aa_counts.values()]
        _shannon = -sum(f * np.log2(f) for f in _aa_freqs if f > 0)
        # Max entropy for 20 AA ≈ 4.32; normal IgGs typically > 3.8
        if _shannon < 3.0:
            repeat_alert = True
            repeat_severity = "warning"

        # 2. Consecutive identical residue runs (e.g., GGGGGG)
        _max_run = 1
        _cur_run = 1
        for _ci in range(1, seq_len):
            if seq_upper[_ci] == seq_upper[_ci - 1]:
                _cur_run += 1
                _max_run = max(_max_run, _cur_run)
            else:
                _cur_run = 1
        if _max_run >= 6:
            repeat_alert = True
            repeat_severity = "critical"

    # OOD info from workspace
    ood_info = {}
    if _active and isinstance(_active, dict):
        _cache = _active.get("analysis_cache") or {}
        _dev = (_cache.get("dev_result") or {}).get("data", {})
        ood_info = _dev.get("ood_info", {})

    return {
        "name": intent.get("name", "Unknown"),
        "pI": intent.get("pI"),
        "mw": intent.get("mw"),
        "sequence_length": seq_len,
        "hydrophobicity": intent.get("hydrophobicity"),
        "gravy": intent.get("gravy"),
        "aromatic_frac": aromatic_frac,
        "cys_frac": cys_frac,
        "repeat_alert": repeat_alert,
        "repeat_severity": repeat_severity,
        "ood_is_ood": ood_info.get("is_ood", False),
        "ood_confidence": ood_info.get("confidence", "Unknown"),
        "titer": up.get("final_titer"),
        "peak_vcd": up.get("peak_vcd"),
        "viability": up.get("viability_at_harvest"),
        "dev_penalty": up.get("dev_penalty_applied"),
        "optimal_ph": doe.get("optimal_ph"),
        "optimal_gradient": doe.get("optimal_gradient"),
        "resolution": doe.get("optimal_resolution"),
        "ds_yield": doe.get("optimal_yield"),
        "cogs_per_gram": cogs.get("cogs_per_gram"),
        "cost_rating": cogs.get("cost_rating"),
        "batch_output_g": cogs.get("batch_output_g"),
        "dev_score": dev_score,
        "agg_risk": agg_risk,
        "stability": stability,
        "ada_risk_level": ada_risk_level,
        "formulation_ph": session_data.get("formulation_buffer_ph"),
        "formulation_buffer": session_data.get("formulation_buffer_type"),
        "excipients": session_data.get("formulation_excipients", []),
        "glycoform": session_data.get("glycoform_profile"),
    }


# ===========================================================================
# 3. Mock Agent Engines (no API key)
# ===========================================================================

def _mock_upstream_agent(d: Dict[str, Any]) -> AgentStatement:
    """Upstream Expert assessment."""
    flags = []
    parts = [f"**Upstream Assessment for {d['name']}:**\n"]

    titer = d.get("titer")
    if titer:
        parts.append(f"The CHO fed-batch process achieves a harvest titer of **{titer:.2f} g/L**.")
        if titer >= 5.0:
            parts.append("This is a strong titer suitable for commercial-scale manufacturing at 2000L.")
        elif titer >= 2.0:
            parts.append("Titer is moderate. Process intensification (perfusion or enriched media) should be evaluated.")
            flags.append("Moderate titer — consider process optimization")
        else:
            parts.append("**Titer is critically low.** This poses significant risk to process robustness and scale-up feasibility. Recommend clone re-screening or media optimization.")
            flags.append("LOW TITER: <2 g/L — commercial viability at risk")
    else:
        parts.append("No upstream simulation data available. I recommend running the fed-batch simulation before proceeding.")
        flags.append("No upstream data — simulation required")

    vcd = d.get("peak_vcd")
    viab = d.get("viability")
    if vcd:
        parts.append(f"\nPeak VCD: {vcd:.1f} x10^6 cells/mL.")
    if viab:
        parts.append(f"Harvest viability: {viab:.0f}%.")
        if viab < 70:
            flags.append("Low harvest viability (<70%)")

    penalty = d.get("dev_penalty")
    if penalty and penalty < 0.8:
        parts.append(f"\n**Warning:** Developability penalty is {penalty:.0%} — the molecule's high aggregation propensity causes ER stress, suppressing specific productivity.")
        flags.append(f"Dev penalty = {penalty:.0%} — ER stress reducing qp")

    # Sequence composition flags
    aro = d.get("aromatic_frac", 0.0)
    if aro > 0.20:
        parts.append(f"\n**CRITICAL: Aromatic content is {aro:.0%}** (W+Y+F). Normal IgGs have 10-15%. Excessive aromatics cause protein misfolding, ER stress, and aggregation during expression.")
        flags.append(f"CRITICAL aromatic overload: {aro:.0%} (W+Y+F)")
    if d.get("repeat_alert"):
        _rep_sev = d.get("repeat_severity", "warning")
        if _rep_sev == "critical":
            parts.append("\n**CRITICAL: Consecutive repeat region detected (≥6 identical residues).** This causes ribosomal stalling and misfolding in CHO cells.")
            flags.append("CRITICAL: consecutive repeat region")
        else:
            parts.append("\n**Warning: Low-complexity region detected (Shannon entropy < 3.0).** Sequence has unusually biased amino acid composition that may affect expression.")
            flags.append("Low-complexity sequence — monitor expression")
    if d.get("ood_is_ood"):
        parts.append(f"\n**OOD Warning:** This sequence is flagged as Out-of-Distribution (confidence: {d.get('ood_confidence', 'Low')}). Predictions may be unreliable.")
        flags.append("OOD sequence — predictions unreliable")

    rec = "Proceed to downstream" if titer and titer >= 2.0 and aro <= 0.20 else "Optimize upstream before proceeding"

    return AgentStatement(
        agent_name="Upstream Expert",
        agent_role="Cell Culture & Bioprocess",
        icon="UP",
        statement="\n".join(parts),
        risk_flags=flags,
        recommendation=rec,
    )


def _mock_downstream_agent(d: Dict[str, Any]) -> AgentStatement:
    """Downstream Expert assessment."""
    flags = []
    parts = [f"**Downstream Assessment for {d['name']}:**\n"]

    rs = d.get("resolution")
    yld = d.get("ds_yield")
    opt_ph = d.get("optimal_ph")
    opt_grad = d.get("optimal_gradient")

    if rs is not None:
        parts.append(f"DoE optimization identifies the sweet spot at pH **{opt_ph:.2f}**, gradient **{opt_grad:.1f} mM/min**.")
        parts.append(f"Resolution (Rs): **{rs:.3f}** | Yield: **{yld:.1%}**.")

        # Quantified coelution analysis
        try:
            from src.purification_optimizer import estimate_coelution_percent
            _coelution_pct = estimate_coelution_percent(rs)
        except Exception:
            _coelution_pct = None

        if rs >= 1.5:
            _co_txt = f" (estimated coelution: {_coelution_pct:.1f}%)" if _coelution_pct is not None else ""
            parts.append(f"Baseline resolution achieved — acidic and basic charge variants are well-separated from the main peak{_co_txt}.")
        elif rs >= 0.8:
            _co_txt = f" **{_coelution_pct:.1f}%** estimated coelution" if _coelution_pct is not None else ""
            parts.append(
                f"Partial overlap between acidic variants (deamidated, sialylated species) "
                f"and the main peak —{_co_txt}. "
                f"Consider shallower gradient (current: {opt_grad:.1f} mM/min) or extended column length to improve separation."
            )
            flags.append(f"Partial resolution (Rs={rs:.2f}) — acidic/main overlap ~{_coelution_pct:.0f}%" if _coelution_pct else "Partial resolution — variant overlap risk")
        else:
            _co_txt = f" **{_coelution_pct:.1f}%**" if _coelution_pct is not None else ""
            parts.append(
                f"**Resolution is poor (Rs={rs:.3f}).** Acidic charge variants (deamidated, glycated species) "
                f"and basic variants (C-terminal Lys, succinimide) co-elute with the main peak — "
                f"estimated coelution{_co_txt}. "
                f"Recommend: (1) column screening (CEX resin with higher selectivity), "
                f"(2) mixed-mode chromatography, or (3) pH gradient optimization."
            )
            flags.append(f"POOR RESOLUTION: Rs={rs:.3f}, coelution ~{_coelution_pct:.0f}% — acidic/basic variants contaminate main peak" if _coelution_pct else "POOR RESOLUTION: variants co-elute")

        if yld and yld < 0.5:
            parts.append(
                f"Note: Cumulative yield ({yld:.0%}) is below typical platform range (50-70%). "
                f"This is a COGS/economic parameter and should be addressed in process optimization, "
                f"not treated as a product quality or safety concern."
            )
    else:
        parts.append("No DoE data available. I recommend running the In-Silico DoE on the Downstream page.")
        flags.append("No downstream DoE data")

    agg = d.get("agg_risk")  # ML risk index [0-1]
    if agg and agg > 0.3:
        # Map to projected physical aggregation (SEC-HPLC HMW%)
        _projected_hmw = agg * agg * 20.0  # Quadratic mapping: 0.3→1.8%, 0.5→5%, 0.8→12.8%
        parts.append(f"\n**Aggregation risk index: {agg:.2f}** (projected {_projected_hmw:.1f}% HMW) may cause product loss on Protein A and require additional polishing steps to meet ICH Q6B purity specifications.")
        flags.append(f"Agg risk index {agg:.2f} (proj. {_projected_hmw:.1f}% HMW) — may need extra polishing")

    pI = d.get("pI")
    form_ph = d.get("formulation_ph")
    if pI and form_ph:
        gap = abs(pI - form_ph)
        if gap < 1.0:
            parts.append(f"\nFormulation pH ({form_ph}) is dangerously close to pI ({pI}). Colloidal instability risk in the drug product.")
            flags.append("pH near pI — colloidal instability")

    rec = "Downstream process is acceptable" if rs and rs >= 0.8 else "Downstream optimization required"

    return AgentStatement(
        agent_name="Downstream Expert",
        agent_role="Purification & Formulation",
        icon="DS",
        statement="\n".join(parts),
        risk_flags=flags,
        recommendation=rec,
    )


def _mock_regulatory_agent(d: Dict[str, Any], up_flags: List[str], ds_flags: List[str]) -> AgentStatement:
    """Regulatory/QA assessment and final verdict."""
    flags = []
    parts = [f"**Regulatory & QA Assessment for {d['name']}:**\n"]

    all_flags = up_flags + ds_flags

    # Aggregation
    agg = d.get("agg_risk")  # ML risk index [0-1]
    if agg:
        if agg > 0.4:
            _projected_hmw = agg * agg * 20.0  # Quadratic mapping
            parts.append(f"**Critical safety concern:** Aggregation risk index is {agg:.2f} (projected {_projected_hmw:.1f}% HMW). Aggregates are immunogenic and can cause adverse events. This level would likely trigger an FDA hold or require extensive characterization.")
            flags.append(f"HIGH AGGREGATION: risk index {agg:.2f} (proj. {_projected_hmw:.1f}% HMW) — potential clinical hold")
        elif agg > 0.2:
            _projected_hmw = agg * agg * 20.0  # Quadratic mapping
            parts.append(f"Aggregation risk index {agg:.2f} (projected {_projected_hmw:.1f}% HMW) is above preferred threshold. Ensure SEC-HPLC specifications are in place.")
            flags.append(f"Moderate aggregation — risk index {agg:.2f} (proj. {_projected_hmw:.1f}% HMW) — additional characterization needed")

    # Developability
    dev = d.get("dev_score")
    if dev is not None:
        if dev >= 0.7:
            parts.append(f"\nDevelopability score ({dev:.2f}) is in the acceptable range for IND filing.")
        else:
            parts.append(f"\n**Developability score ({dev:.2f}) is below 0.7.** Recommend molecular engineering to improve the candidate before IND submission.")
            flags.append(f"Low dev score ({dev:.2f}) — engineering recommended")
    else:
        parts.append("\nDevelopability score: not available (run main analysis pipeline first). CMC assessment based on available data only.")

    # ADA / Immunogenicity risk (patient safety — replaces COGS evaluation)
    ada_risk = d.get("ada_risk_level")
    if ada_risk:
        if ada_risk in ("High", "Critical"):
            parts.append(
                f"\n**Critical immunogenicity concern:** ADA risk is **{ada_risk}**. "
                f"High anti-drug antibody formation will lead to loss of efficacy and "
                f"potential adverse events. This molecule is unsuitable for clinical "
                f"development in its current form."
            )
            flags.append("CRITICAL ADA RISK: patient immunogenicity threat")
        elif ada_risk == "Medium":
            parts.append(
                f"\nADA risk is **{ada_risk}**. Recommend humanization and MHC-II "
                f"epitope mapping before IND filing."
            )
            flags.append("Note: Moderate ADA risk — consider humanization assessment")
        else:
            parts.append(f"\nADA risk is {ada_risk}. Acceptable for IND filing.")

    # Sequence composition flags at regulatory level
    aro = d.get("aromatic_frac", 0.0)
    if aro > 0.20:
        parts.append(f"\n**Sequence composition is non-standard.** Aromatic residue content ({aro:.0%}) is far above IgG norms (10-15%). Per ICH Q6B, product-related impurities from misfolding are unacceptable. This molecule cannot be manufactured to GMP standards.")
        flags.append("CRITICAL: non-standard sequence composition")
    if d.get("repeat_alert"):
        _rep_sev = d.get("repeat_severity", "warning")
        if _rep_sev == "critical":
            flags.append("CRITICAL: consecutive repeat region — GMP manufacturing infeasible")
        else:
            flags.append("Warning: Low-complexity region — expression may be affected")
    if d.get("ood_is_ood") and d.get("ood_confidence") == "Low":
        flags.append("OOD: all assessments are low-confidence")

    # ── Risk Assessment (color-coded, no binary Go/No-Go) ──
    n_critical = sum(1 for f in (all_flags + flags) if any(w in f.upper() for w in ["CRITICAL", "HOLD", "STOP"]))
    n_moderate = len(all_flags) + len(flags) - n_critical

    # Build mitigation recommendations (ICH-referenced investigations, not filing decisions)
    mitigations = []
    for f in (all_flags + flags):
        f_lower = f.lower()
        if "agg" in f_lower:
            mitigations.append("Conduct detailed aggregation profiling (SEC-HPLC, DLS) and develop specifications per ICH Q6B. Optimize formulation buffer (pH/excipient screening) to minimize aggregation under ICH Q5C storage conditions.")
        elif "ada" in f_lower or "immuno" in f_lower:
            mitigations.append("Conduct comprehensive MHC-II epitope mapping per ICH Q5E and establish baseline immunogenicity profile. Consider humanization or epitope removal assessment for clinical risk stratification.")
        elif "titer" in f_lower:
            mitigations.append("Optimize upstream cell culture parameters (temperature shift, feed strategy, osmolarity) per ICH Q5A. Characterize process robustness and scale-up feasibility across 100 L to 2000 L scales.")
        elif "resolution" in f_lower or "coelution" in f_lower:
            mitigations.append("Characterize charge variants (acidic/basic species) by CIE-MS per ICH Q6B. Evaluate additional polishing steps, mixed-mode chromatography, or alternative gradient strategies to ensure product homogeneity.")
        elif "ood" in f_lower:
            mitigations.append("Validate in vitro predictions empirically — sequence is outside training distribution. Conduct small-scale expression and characterization studies to confirm model predictions.")
        elif "composition" in f_lower or "aromatic" in f_lower:
            mitigations.append("Assess expression feasibility with small-scale transient or stable expression studies. Evaluate protein folding by DSF/DSC and characterize post-translational modifications per ICH Q6B.")
        elif "repeat" in f_lower:
            mitigations.append("Verify protein expression and correct folding with small-scale expression studies. Conduct peptide mapping and mass spectrometry analysis to confirm sequence integrity and PTM profiles.")
    # Deduplicate
    mitigations = list(dict.fromkeys(mitigations))

    if n_critical >= 2:
        risk_level = "Critical"
        risk_assessment = "Significant"
        verdict = (
            "**Risk Assessment: SIGNIFICANT** — Multiple critical quality or safety concerns identified. "
            "Comprehensive investigation and risk mitigation required per ICH Q8/Q9 framework before advancement."
        )
        next_steps = "Prioritize detailed characterization of critical issues. Engage development and analytics teams. Re-assess after targeted investigations."
    elif n_critical == 1:
        risk_level = "High"
        risk_assessment = "Elevated"
        verdict = (
            "**Risk Assessment: ELEVATED** — One critical quality attribute or safety concern requires focused investigation. "
            "Proceed with targeted mitigation studies per ICH Q6B and Q5E."
        )
        next_steps = "Develop and execute focused investigation plan for the critical attribute. Parallel characterization and development activities recommended."
    elif n_moderate >= 4:
        risk_level = "High"
        risk_assessment = "Elevated"
        verdict = (
            "**Risk Assessment: ELEVATED** — Multiple moderate quality or process risks identified. "
            "Recommend addressing key items through additional characterization studies per ICH Q6B and Q5C."
        )
        next_steps = "Prioritize and characterize top-risk attributes. Establish provisional specifications. Continue development with risk monitoring."
    elif n_moderate >= 1:
        risk_level = "Medium"
        risk_assessment = "Manageable"
        verdict = (
            "**Risk Assessment: MANAGEABLE** — Quality attributes are within acceptable ranges for development stage. "
            "Risks are addressable through standard ICH Q8 development activities."
        )
        next_steps = "Proceed with standard pharmaceutical development activities. Establish preliminary specifications per ICH Q6B. Monitor flagged attributes in pre-clinical characterization studies."
    else:
        risk_level = "Low"
        risk_assessment = "Favorable"
        verdict = (
            "**Risk Assessment: FAVORABLE** — Quality profile is within expected range for this development stage. "
            "No critical concerns identified. Proceed with confidence."
        )
        next_steps = "Proceed with standard development path. Comprehensive CMC strategy per ICH Q8 recommended. Establish specifications per ICH Q6B based on pre-clinical and clinical data."

    parts.append(f"\n---\n{verdict}")

    if mitigations:
        parts.append("\n**Recommended Mitigations:**")
        for i, m in enumerate(mitigations, 1):
            parts.append(f"  {i}. {m}")

    rec = next_steps

    return AgentStatement(
        agent_name="QA / Regulatory",
        agent_role="Quality Assurance & Regulatory Affairs",
        icon="QA",
        statement="\n".join(parts),
        risk_flags=flags,
        recommendation=rec,
    ), risk_level, risk_assessment, mitigations, next_steps


# ===========================================================================
# 4. OpenAI-Powered Board (when key available)
# ===========================================================================

_UPSTREAM_SYSTEM = """You are the Upstream Process Development expert on a CMC Board.
Domain: CHO cell culture, fed-batch and perfusion bioprocessing, titer optimisation, cell line
development, and scale-up from bench to 2000 L production bioreactors.
Regulatory frame: ICH Q5A (viral safety), Q5B (expression construct), Q5D (cell substrates).
Assess the molecule's upstream manufacturability using the provided session data. Reference
specific numbers (titer in g/L, peak VCD, viability, dev penalty). Identify risks to process
robustness and scale-up. Do NOT comment on cost-of-goods or commercial pricing.
Keep your response under 200 words. End with a clear recommendation."""

_DOWNSTREAM_SYSTEM = """You are the Downstream Process Development expert on a CMC Board.
Domain: Protein A capture, ion-exchange polishing, viral inactivation/filtration, UF/DF,
Design-of-Experiments optimisation, formulation development, and drug product fill-finish.
Regulatory frame: ICH Q5A (viral clearance), Q6B (specifications for biotech products).
Assess purification feasibility using the provided DoE and analytical data. Reference specific
numbers (resolution, yield, aggregation clearance). Critique upstream risks where relevant.
Do NOT comment on cost-of-goods or commercial pricing.
Keep your response under 200 words. End with a clear recommendation."""

_REGULATORY_SYSTEM = """You are the Quality Assurance and Regulatory Affairs expert consultant on a CMC Board.
Domain: Patient safety, immunogenicity (ADA risk), post-translational modifications, aggregation,
charge heterogeneity, product-related impurities, and specification setting.
Regulatory frame: ICH Q5C (stability), Q5E (comparability), Q6B (specifications), Q8/Q9/Q10
(pharmaceutical development, quality risk management, quality systems).

Your role is to provide comprehensive risk assessment and expert consultation, NOT to issue
go/no-go decisions. Do NOT evaluate cost-of-goods, commercial pricing, or business strategy.

Focus strictly on section-by-section risk analysis:
  - Manufacturability per ICH Q8 (Quality by Design framework)
  - Product quality and purity per ICH Q6B (acceptance criteria for biotech products)
  - Comparability assessment per ICH Q5E (analytical procedures, potential impurities)
  - Patient safety and immunogenicity (ADA risk characterization, aggregation profile)
  - Charge variants and PTM-related quality attributes (glycosylation, oxidation, deamidation)

For each risk identified, explain the severity, underlying cause, and recommended investigation
or mitigation strategy. Rate risk severity as Favorable / Manageable / Elevated / Significant.

Keep your response under 250 words. End with a clear summary of your consultative assessment."""


def _call_openai_agent(system: str, context: str, prior: str, api_key: str) -> Optional[str]:
    """Call a single agent via OpenAI."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Session Data:\n{context}\n\nPrior Board Discussion:\n{prior}\n\nProvide your assessment."},
        ]
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=500, temperature=0.7)
        return resp.choices[0].message.content
    except Exception as e:
        log.warning("OpenAI agent call failed: %s", e)
        return None


# ===========================================================================
# 5. Board Meeting Orchestrator
# ===========================================================================

def run_cmc_board_meeting(
    session_data: Dict[str, Any],
    api_key: Optional[str] = None,
) -> BoardMeetingResult:
    """
    Orchestrate a CMC Board meeting with 3 agents.

    Parameters
    ----------
    session_data : Streamlit session state (or dict)
    api_key      : OpenAI API key (None → mock agents)

    Returns
    -------
    BoardMeetingResult with all agent statements and verdict
    """
    t0 = time.time()
    d = _extract_board_data(session_data)

    if api_key and api_key.strip().startswith("sk-"):
        # Format context for OpenAI
        ctx_lines = []
        for k, v in d.items():
            if v is not None:
                ctx_lines.append(f"  {k}: {v}")
        context = "\n".join(ctx_lines)

        # Agent 1: Upstream
        up_text = _call_openai_agent(_UPSTREAM_SYSTEM, context, "(First speaker)", api_key)
        if up_text is None:
            up_text = "Upstream assessment unavailable."
        up_stmt = AgentStatement("Upstream Expert", "Cell Culture & Bioprocess", "UP", up_text, recommendation="See above")

        # Agent 2: Downstream
        ds_text = _call_openai_agent(_DOWNSTREAM_SYSTEM, context, f"Upstream Expert said:\n{up_text}", api_key)
        if ds_text is None:
            ds_text = "Downstream assessment unavailable."
        ds_stmt = AgentStatement("Downstream Expert", "Purification & Formulation", "DS", ds_text, recommendation="See above")

        # Agent 3: Regulatory
        prior = f"Upstream Expert:\n{up_text}\n\nDownstream Expert:\n{ds_text}"
        reg_text = _call_openai_agent(_REGULATORY_SYSTEM, context, prior, api_key)
        if reg_text is None:
            reg_text = "Regulatory assessment unavailable."

        # Parse risk level and risk assessment from text (consultative output, not filing decision)
        risk_level = "Medium"
        risk_assessment = "Manageable"
        mitigations = []
        next_steps = "Proceed with standard development path"

        for lvl in ["Critical", "High", "Medium", "Low"]:
            if lvl.lower() in reg_text.lower():
                risk_level = lvl
                break

        if risk_level == "Critical":
            risk_assessment = "Significant"
            next_steps = "Prioritize detailed characterization of critical issues. Engage development and analytics teams. Re-assess after targeted investigations."
        elif risk_level == "High":
            risk_assessment = "Elevated"
            next_steps = "Develop and execute focused investigation plan for the critical attributes. Parallel characterization and development activities recommended."
        elif risk_level == "Medium":
            risk_assessment = "Manageable"
            next_steps = "Proceed with standard pharmaceutical development activities. Establish preliminary specifications per ICH Q6B. Monitor flagged attributes in pre-clinical characterization studies."
        else:
            risk_assessment = "Favorable"
            next_steps = "Proceed with standard development path. Comprehensive CMC strategy per ICH Q8 recommended. Establish specifications per ICH Q6B based on pre-clinical and clinical data."

        reg_stmt = AgentStatement("QA / Regulatory", "Quality Assurance & Regulatory Affairs", "QA",
                                   reg_text, recommendation="See verdict above")

        statements = [up_stmt, ds_stmt, reg_stmt]
        consensus = reg_text.split("VERDICT")[-1] if "VERDICT" in reg_text else reg_text[-200:]

    else:
        # Mock agents
        up_stmt = _mock_upstream_agent(d)
        ds_stmt = _mock_downstream_agent(d)
        reg_stmt, risk_level, risk_assessment, mitigations, next_steps = _mock_regulatory_agent(d, up_stmt.risk_flags, ds_stmt.risk_flags)
        statements = [up_stmt, ds_stmt, reg_stmt]
        consensus = reg_stmt.statement.split("---")[-1].strip() if "---" in reg_stmt.statement else reg_stmt.recommendation

    wall = time.time() - t0

    all_flags = []
    for s in statements:
        all_flags.extend(s.risk_flags)

    summary = (
        f"CMC Board for {d['name']}: {len(statements)} agents, "
        f"{len(all_flags)} risk flags, verdict = {risk_level}, "
        f"risk_assessment = {risk_assessment}"
    )

    log.info(summary)

    return BoardMeetingResult(
        statements=statements,
        consensus=consensus,
        risk_level=risk_level,
        risk_assessment=risk_assessment,
        mitigation_recommendations=mitigations,
        next_steps=next_steps,
        summary=summary,
        wall_time_s=round(wall, 2),
    )


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    passed = 0

    mock = {
        "last_intent": {"name": "TestmAb", "pI": 8.5, "mw": 150.0, "sequence": "A" * 450},
        "upstream_result_dict": {"final_titer": 9.05, "peak_vcd": 76.0, "viability_at_harvest": 85.0, "dev_penalty_applied": 0.79},
        "doe_result_dict": {"optimal_ph": 6.5, "optimal_gradient": 22.6, "optimal_resolution": 0.44, "optimal_yield": 0.41},
        "cogs_result_dict": {"cogs_per_gram": 40.43, "cost_rating": "Excellent", "batch_output_g": 7000},
        "formulation_buffer_ph": 6.0,
        "formulation_buffer_type": "histidine",
        "formulation_excipients": ["trehalose"],
        "glycoform_profile": "standard_cho",
    }

    # Test 1: Board meeting runs
    r = run_cmc_board_meeting(mock)
    assert len(r.statements) == 3
    print(f"Test 1 PASS: Board meeting — {len(r.statements)} agents, {r.risk_level}")
    passed += 1

    # Test 2: All agents have statements
    for s in r.statements:
        assert len(s.statement) > 50
        assert s.agent_name
    print(f"Test 2 PASS: All agents produced statements")
    passed += 1

    # Test 3: Risk level and assessment are valid
    assert r.risk_level in ("Low", "Medium", "High", "Critical")
    assert r.risk_assessment in ("Favorable", "Manageable", "Elevated", "Significant")
    print(f"Test 3 PASS: Risk = {r.risk_level}, assessment = {r.risk_assessment}")
    passed += 1

    # Test 4: Data references in statements
    full_text = " ".join(s.statement for s in r.statements)
    assert "9.05" in full_text or "titer" in full_text.lower()
    print(f"Test 4 PASS: Statements reference session data")
    passed += 1

    # Test 5: Bad molecule triggers critical
    bad = dict(mock)
    bad["upstream_result_dict"] = {"final_titer": 0.3, "peak_vcd": 5.0, "viability_at_harvest": 45.0, "dev_penalty_applied": 0.35}
    bad["cogs_result_dict"] = {"cogs_per_gram": 950, "cost_rating": "Non-viable", "batch_output_g": 200}
    r_bad = run_cmc_board_meeting(bad)
    assert r_bad.risk_level in ("High", "Critical")
    print(f"Test 5 PASS: Bad molecule → {r_bad.risk_level}")
    passed += 1

    print(f"\n{'='*50}")
    print(f"multi_agent_board self-test: {passed}/5 passed")
    sys.exit(0 if passed == 5 else 1)
