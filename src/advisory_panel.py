"""
src/advisory_panel.py — ProtePilot Advisory Panel
===================================================
ProtePilot Platform · v32.0

Three domain-expert roles provide structured, data-referenced assessments
of a molecule's developability. These are *roles* (deterministic heuristic
evaluators), not autonomous agents — they read computed results from the
session and produce concise, actionable summaries.

Roles
-----
  1. Upstream Process Scientist   — expression, titer, scale-up feasibility
  2. Downstream & Formulation Sci — purification, COGS, stability, formulation
  3. Quality & Analytical Scientist— CQA control, PTM, immunogenicity, spec-setting

Each role outputs a structured AdvisorySummary with:
  - key_findings   : list[str]  (1-2 sentences each, max 3)
  - risk_flags     : list[str]  (specific, data-referenced)
  - recommendation : str        (one-sentence action item)
  - risk_level     : str        (Low / Medium / High / Critical)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("ProtePilot.AdvisoryPanel")


# ===========================================================================
# 1. Data Classes
# ===========================================================================

@dataclass
class AdvisorySummary:
    """One role's structured assessment."""
    role_name: str
    role_title: str
    key_findings: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    recommendation: str = ""
    risk_level: str = "Low"          # Low / Medium / High / Critical


@dataclass
class PanelResult:
    """Combined advisory panel output."""
    upstream: AdvisorySummary
    downstream: AdvisorySummary
    quality: AdvisorySummary
    overall_risk: str = "Low"        # worst of the three
    overall_recommendation: str = ""
    all_risk_flags: List[str] = field(default_factory=list)


# ===========================================================================
# 2. Session Data Extractor (reused from multi_agent_board pattern)
# ===========================================================================

def _extract_panel_data(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize all relevant data for the advisory panel."""
    intent = session_data.get("last_intent") or {}
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

    # Developability from workspace
    dev_score = None
    agg_risk = None
    stability = None
    viscosity_risk = None
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
            viscosity_risk = preds.get("viscosity_risk")

    # ADA / Immunogenicity
    ada_risk_level = None
    _ada_result = session_data.get("ada_result")
    if _ada_result and hasattr(_ada_result, "ada_risk_level"):
        ada_risk_level = _ada_result.ada_risk_level
    elif isinstance(_ada_result, dict):
        ada_risk_level = _ada_result.get("ada_risk_level")

    # Analytical QC
    _qc = session_data.get("analytical_qc_result") or {}
    if not _qc and _active and isinstance(_active, dict):
        _cache = _active.get("analysis_cache") or {}
        _qc = _cache.get("analytical_qc") or {}
    # Extract QC pass/fail flag
    _qc_pass = True
    if isinstance(_qc, dict):
        _qc_pass = _qc.get("overall_qc_pass", _qc.get("overall_pass", True))

    # Stability result
    stab = session_data.get("stability_result_dict") or {}

    return {
        "name": intent.get("name", "Unknown"),
        "molecule_class": intent.get("molecule_class", "unknown"),
        "pI": intent.get("pI"),
        "mw": intent.get("mw"),
        "hydrophobicity": intent.get("hydrophobicity"),
        "sequence_length": len(intent.get("sequence", "")),
        "n_chains": intent.get("n_chains"),
        # Upstream
        "titer": up.get("final_titer"),
        "peak_vcd": up.get("peak_vcd"),
        "viability": up.get("viability_at_harvest"),
        "dev_penalty": up.get("dev_penalty_applied"),
        # Downstream
        "optimal_ph": doe.get("optimal_ph"),
        "optimal_gradient": doe.get("optimal_gradient"),
        "resolution": doe.get("optimal_resolution"),
        "ds_yield": doe.get("optimal_yield"),
        "cogs_per_gram": cogs.get("cogs_per_gram"),
        "cost_rating": cogs.get("cost_rating"),
        # Developability
        "dev_score": dev_score,
        "agg_risk": agg_risk,
        "stability_score": stability,
        "viscosity_risk": viscosity_risk,
        # Clinical
        "ada_risk_level": ada_risk_level,
        # Formulation
        "formulation_ph": session_data.get("formulation_buffer_ph"),
        # Analytical QC
        "qc_pass": _qc_pass,
        "sec_monomer": (_qc.get("sec", {}).get("monomer_pct") if isinstance(_qc.get("sec"), dict) else _qc.get("sec_monomer_pct")) if isinstance(_qc, dict) else None,
        "cief_main": (_qc.get("cief", {}).get("main_pct") if isinstance(_qc.get("cief"), dict) else _qc.get("main_pct")) if isinstance(_qc, dict) else None,
        # Stability
        "shelf_life_months": stab.get("shelf_life_months"),
        "stability_grade": stab.get("grade"),
    }


# ===========================================================================
# 3. Role Assessors
# ===========================================================================

def _assess_upstream(d: Dict[str, Any]) -> AdvisorySummary:
    """Upstream Process Scientist assessment."""
    findings = []
    flags = []

    titer = d.get("titer")
    if titer is not None:
        if titer >= 5.0:
            findings.append(f"Titer of {titer:.1f} g/L is strong for commercial manufacturing at 2000L scale.")
        elif titer >= 2.0:
            findings.append(f"Titer of {titer:.1f} g/L is moderate. Process intensification may improve yield.")
            flags.append(f"Moderate titer ({titer:.1f} g/L)")
        else:
            findings.append(f"Titer of {titer:.1f} g/L is critically low — commercial viability at risk.")
            flags.append(f"LOW TITER: {titer:.1f} g/L")

    viability = d.get("viability")
    if viability is not None:
        if viability < 70:
            findings.append(f"Harvest viability {viability:.0f}% is below 70% threshold — elevated HCP and DNA impurities expected.")
            flags.append(f"Low viability ({viability:.0f}%)")
        elif viability < 80:
            flags.append(f"Borderline viability ({viability:.0f}%)")

    penalty = d.get("dev_penalty")
    if penalty and penalty > 0:
        findings.append(f"Developability penalty of {penalty:.0f}% applied to titer due to sequence complexity.")

    if not findings:
        findings.append("Upstream simulation not yet run. Run CHO fed-batch simulation for expression feasibility.")

    # Risk level
    risk = "Low"
    if any("LOW TITER" in f for f in flags) or (viability and viability < 70):
        risk = "High"
    elif flags:
        risk = "Medium"

    rec = "Proceed with standard CHO process development." if risk == "Low" else \
          "Evaluate process intensification or media optimization." if risk == "Medium" else \
          "Investigate alternative expression systems or sequence engineering."

    return AdvisorySummary(
        role_name="Upstream Process Scientist",
        role_title="Expression & Cell Culture",
        key_findings=findings[:3],
        risk_flags=flags,
        recommendation=rec,
        risk_level=risk,
    )


def _assess_downstream(d: Dict[str, Any]) -> AdvisorySummary:
    """Downstream & Formulation Scientist assessment."""
    findings = []
    flags = []

    rs = d.get("resolution")
    yld = d.get("ds_yield")
    if rs is not None:
        if rs >= 1.5:
            findings.append(f"DoE resolution Rs={rs:.2f} — baseline separation achieved.")
        elif rs >= 0.8:
            findings.append(f"DoE resolution Rs={rs:.2f} — partial separation, may require optimized gradient.")
            flags.append(f"Partial resolution (Rs={rs:.2f})")
        else:
            findings.append(f"DoE resolution Rs={rs:.2f} — co-elution risk, polishing step likely needed.")
            flags.append(f"Poor resolution (Rs={rs:.2f})")

    if yld is not None:
        if yld < 0.50:
            flags.append(f"Low yield ({yld:.0%})")

    agg = d.get("agg_risk")
    if agg is not None and agg > 0.4:
        findings.append(f"Aggregation risk {agg:.2f} is elevated — additional polishing or formulation optimization needed.")
        flags.append(f"Aggregation risk {agg:.2f}")

    cogs = d.get("cogs_per_gram")
    if cogs is not None:
        rating = d.get("cost_rating", "")
        if cogs > 150:
            findings.append(f"COGS ${cogs:.0f}/g exceeds commercial viability threshold.")
            flags.append(f"COGS ${cogs:.0f}/g — non-viable")
        elif cogs > 80:
            findings.append(f"COGS ${cogs:.0f}/g ({rating}) — monitor cost drivers.")

    shelf = d.get("shelf_life_months")
    if shelf is not None and shelf < 18:
        flags.append(f"Short predicted shelf life ({shelf} months)")

    if not findings:
        findings.append("Downstream DoE not yet run. Run purification optimization for feasibility assessment.")

    risk = "Low"
    if any("Poor resolution" in f or "non-viable" in f for f in flags):
        risk = "High"
    elif flags:
        risk = "Medium"

    rec = "Standard Protein A + IEX polishing platform is suitable." if risk == "Low" else \
          "Optimize gradient conditions and evaluate formulation stabilizers." if risk == "Medium" else \
          "Investigate alternative purification strategies or sequence engineering to reduce aggregation."

    return AdvisorySummary(
        role_name="Downstream & Formulation Scientist",
        role_title="Purification, Formulation & COGS",
        key_findings=findings[:3],
        risk_flags=flags,
        recommendation=rec,
        risk_level=risk,
    )


def _assess_quality(d: Dict[str, Any]) -> AdvisorySummary:
    """Quality & Analytical Scientist assessment."""
    findings = []
    flags = []

    # QC pass/fail — takes precedence over composite score
    qc_pass = d.get("qc_pass", True)
    if not qc_pass:
        findings.append("Virtual QC status: **FAIL** — one or more analytical assays outside acceptance criteria.")
        flags.append("QC FAIL — assay(s) out of specification")

    dev = d.get("dev_score")
    if dev is not None:
        if dev > 0.55:
            findings.append(f"Composite developability score {dev:.2f} indicates elevated overall risk profile.")
            flags.append(f"High developability risk ({dev:.2f})")
        elif dev > 0.25:
            findings.append(f"Composite developability score {dev:.2f} — moderate risk, standard characterization recommended.")
        else:
            findings.append(f"Composite developability score {dev:.2f} — favorable profile.")

    ada = d.get("ada_risk_level")
    if ada and ada.lower() == "high":
        findings.append("ADA immunogenicity risk is HIGH — deimmunization engineering recommended.")
        flags.append("HIGH ADA risk")
    elif ada and ada.lower() == "medium":
        flags.append("Medium ADA risk — monitor in Phase I")

    sec = d.get("sec_monomer")
    if sec is not None and sec < 95.0:
        flags.append(f"SEC monomer {sec:.1f}% below 95% target")

    cief = d.get("cief_main")
    if cief is not None and cief < 55.0:
        flags.append(f"cIEF main peak {cief:.1f}% — heterogeneous charge profile")

    visc = d.get("viscosity_risk")
    if visc is not None and visc > 0.5:
        flags.append(f"Viscosity risk {visc:.2f} — may limit high-concentration formulation")

    if not findings:
        findings.append("Run developability assessment to enable quality evaluation.")

    risk = "Low"
    if any("HIGH ADA" in f or "High developability" in f or "QC FAIL" in f for f in flags):
        risk = "High"
    elif flags:
        risk = "Medium"

    rec = "Standard analytical characterization package is sufficient." if risk == "Low" else \
          "Expand CQA characterization and establish preliminary specifications." if risk == "Medium" else \
          "Prioritize detailed characterization. Consider deimmunization or sequence optimization."

    return AdvisorySummary(
        role_name="Quality & Analytical Scientist",
        role_title="CQA Control & Immunogenicity",
        key_findings=findings[:3],
        risk_flags=flags,
        recommendation=rec,
        risk_level=risk,
    )


# ===========================================================================
# 4. Panel Orchestrator
# ===========================================================================

_RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def run_advisory_panel(session_data: Dict[str, Any]) -> PanelResult:
    """
    Run the three-role advisory panel and return structured results.

    Parameters
    ----------
    session_data : dict  (typically st.session_state snapshot + _workspace_store)

    Returns
    -------
    PanelResult with per-role summaries and overall risk
    """
    d = _extract_panel_data(session_data)

    up = _assess_upstream(d)
    ds = _assess_downstream(d)
    qa = _assess_quality(d)

    # Overall risk = worst of the three
    roles = [up, ds, qa]
    worst = max(roles, key=lambda r: _RISK_ORDER.get(r.risk_level, 0))
    overall_risk = worst.risk_level

    all_flags = []
    for r in roles:
        all_flags.extend(r.risk_flags)

    # Overall recommendation
    if overall_risk == "Low":
        overall_rec = "Proceed with standard development. All domains report favorable profiles."
    elif overall_risk == "Medium":
        overall_rec = "Proceed with targeted investigations on flagged attributes."
    elif overall_risk == "High":
        overall_rec = "Address critical findings before advancing. Focused characterization required."
    else:
        overall_rec = "Major concerns identified. Re-evaluate molecule design or development strategy."

    result = PanelResult(
        upstream=up,
        downstream=ds,
        quality=qa,
        overall_risk=overall_risk,
        overall_recommendation=overall_rec,
        all_risk_flags=all_flags,
    )

    log.info(
        "Advisory Panel for %s: UP=%s DS=%s QA=%s → Overall=%s (%d flags)",
        d["name"], up.risk_level, ds.risk_level, qa.risk_level,
        overall_risk, len(all_flags),
    )

    return result


# ===========================================================================
# 5. Self-test
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ProtePilot Advisory Panel — Self-Test")
    print("=" * 60)

    # Mock session data
    mock_session = {
        "last_intent": {
            "name": "Trastuzumab-test",
            "pI": 8.45,
            "mw": 148000,
            "hydrophobicity": 0.42,
            "sequence": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK",
            "molecule_class": "canonical_mab",
            "n_chains": 4,
        },
        "upstream_result_dict": {
            "final_titer": 5.2,
            "peak_vcd": 22.5,
            "viability_at_harvest": 85.0,
            "dev_penalty_applied": 0,
        },
        "doe_result_dict": {
            "optimal_ph": 5.8,
            "optimal_gradient": 12.0,
            "optimal_resolution": 1.85,
            "optimal_yield": 0.78,
        },
        "cogs_result_dict": {
            "cogs_per_gram": 42.5,
            "cost_rating": "Good",
            "batch_output_g": 8112,
        },
    }

    result = run_advisory_panel(mock_session)

    for role in [result.upstream, result.downstream, result.quality]:
        print(f"\n{'='*50}")
        print(f"  {role.role_name} ({role.role_title})")
        print(f"  Risk: {role.risk_level}")
        print(f"  Findings:")
        for f in role.key_findings:
            print(f"    - {f}")
        if role.risk_flags:
            print(f"  Flags: {role.risk_flags}")
        print(f"  Rec: {role.recommendation}")

    print(f"\n{'='*50}")
    print(f"  OVERALL: {result.overall_risk}")
    print(f"  {result.overall_recommendation}")
    print(f"  Total flags: {len(result.all_risk_flags)}")
    print("\nAdvisory Panel self-test PASS")
