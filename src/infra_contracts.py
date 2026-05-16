"""
infra_contracts.py  ·  ProtePilot — Infrastructure & Small Module Contracts
==============================================================================
Behavioral contracts for scaleup_twin, pareto_optimizer, data_harmonizer,
molecule_registry, ood_baseline, workspace_manager, PropertyMapper.

Usage:
    python -m src.infra_contracts                          # Run all
    python -m src.infra_contracts --module scaleup_twin    # Single module

Author  : Di (ProtePilot)
"""
from __future__ import annotations
import logging, sys
from typing import Any, Dict, List, Tuple
log = logging.getLogger("ProtePilot.InfraContracts")


# ═══════════════════════════════════════════════════════════════════════
#  Contracts
# ═══════════════════════════════════════════════════════════════════════

# ── scaleup_twin ─────────────────────────────────────────────────────

def _check_scaleup_returns_result() -> Tuple[bool, str]:
    from src.scaleup_twin import run_scaleup_simulation
    r = run_scaleup_simulation(small_volume_L=2.0, large_volume_L=2000.0)
    for attr in ["recommended_strategy", "predicted_titer_large", "summary"]:
        if not hasattr(r, attr):
            return False, f"Missing attr: {attr}"
    if r.predicted_titer_large <= 0:
        return False, f"Titer={r.predicted_titer_large}, expected > 0"
    return True, f"titer={r.predicted_titer_large:.2f} g/L, strategy={r.recommended_strategy}"

def _check_scaleup_scaling_factor() -> Tuple[bool, str]:
    from src.scaleup_twin import run_scaleup_simulation
    r = run_scaleup_simulation(small_volume_L=2.0, large_volume_L=2000.0, bench_titer=5.0)
    if not (0.5 <= r.titer_scaling_factor <= 1.5):
        return False, f"Scaling factor {r.titer_scaling_factor} outside [0.5, 1.5]"
    return True, f"scaling_factor={r.titer_scaling_factor:.3f}"

# ── pareto_optimizer ─────────────────────────────────────────────────

def _check_pareto_returns_result() -> Tuple[bool, str]:
    from src.pareto_optimizer import run_pareto_analysis
    candidates = [
        {"name": "A", "agg_risk": 0.2, "stability": 0.3, "cost": 100},
        {"name": "B", "agg_risk": 0.8, "stability": 0.1, "cost": 50},
        {"name": "C", "agg_risk": 0.5, "stability": 0.5, "cost": 75},
    ]
    r = run_pareto_analysis(candidates)
    if not hasattr(r, "frontier") or not hasattr(r, "n_pareto"):
        return False, f"Missing frontier/n_pareto"
    if r.n_pareto < 1:
        return False, f"No Pareto-optimal candidates"
    return True, f"{r.n_pareto}/{r.n_total} Pareto-optimal"

def _check_pareto_dominance() -> Tuple[bool, str]:
    from src.pareto_optimizer import run_pareto_analysis
    # B dominates A on all objectives (lower = better)
    candidates = [
        {"name": "A", "obj1": 0.9, "obj2": 0.9},
        {"name": "B", "obj1": 0.1, "obj2": 0.1},
    ]
    r = run_pareto_analysis(candidates)
    frontier_names = [c.name for c in r.frontier]
    ok = "B" in frontier_names
    return ok, f"Frontier: {frontier_names}"

# ── data_harmonizer ──────────────────────────────────────────────────

def _check_harmonizer_returns_dict() -> Tuple[bool, str]:
    import pandas as pd
    from src.data_harmonizer import DataHarmonizer
    h = DataHarmonizer()
    df = pd.DataFrame({
        "Name": ["Mol1", "Mol2"],
        "Sequence": ["ACDEFGHIKLMNPQRSTVWY", "FWFWFWFWFWFWFWFWFWFW"],
    })
    r = h.harmonize(df)
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    if r.get("status") == "error":
        return False, f"Error: {r.get('warnings', '')}"
    return True, f"status={r['status']}, n_valid={r.get('n_valid', '?')}"

# ── molecule_registry ────────────────────────────────────────────────

def _check_registry_known_classes() -> Tuple[bool, str]:
    from src.molecule_registry import get_config
    for cls in ["canonical_mab", "peptide", "fc_fusion"]:
        cfg = get_config(cls)
        if not isinstance(cfg, dict):
            return False, f"{cls}: expected dict, got {type(cfg).__name__}"
        if "display_name" not in cfg:
            return False, f"{cls}: missing display_name"
    return True, "canonical_mab, peptide, fc_fusion configs OK"

def _check_registry_risk_weights() -> Tuple[bool, str]:
    from src.molecule_registry import get_config
    cfg = get_config("canonical_mab")
    weights = cfg.get("risk_weights", {})
    if not weights:
        return False, "No risk_weights for canonical_mab"
    total = sum(weights.values())
    if total < 0.5:
        return False, f"Risk weights sum={total}, too low"
    return True, f"risk_weights: {len(weights)} dims, sum={total:.2f}"

# ── ood_baseline ─────────────────────────────────────────────────────

def _check_ood_baseline_compute() -> Tuple[bool, str]:
    import pandas as pd
    from src.ood_baseline import OODBaselineCalculator
    calc = OODBaselineCalculator()
    df = pd.DataFrame({
        "Combined_Sequence": ["ACDEFGHIKLMNPQRSTVWY" * 5] * 15,
    })
    r = calc.compute_from_training_data(df)
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    if "length" not in r and "pI" not in r:
        return False, f"Missing expected keys, got: {list(r.keys())[:3]}"
    return True, f"Baseline: {len(r)} features computed"

# ── workspace_manager ────────────────────────────────────────────────

def _check_workspace_create() -> Tuple[bool, str]:
    from src.workspace_manager import create_workspace
    ws = create_workspace(display_name="ContractTest")
    if not isinstance(ws, dict):
        return False, f"Expected dict, got {type(ws).__name__}"
    if "id" not in ws and "context_id" not in ws:
        return False, f"Missing id/context_id"
    return True, f"workspace created: {ws.get('display_name', ws.get('id', '?'))}"

# ── PropertyMapper ───────────────────────────────────────────────────

def _check_property_mapper_map() -> Tuple[bool, str]:
    from src.PropertyMapper import PropertyMapper, ProteinProperties
    pm = PropertyMapper()
    protein = ProteinProperties(name="Test", pI=7.5, MW_kDa=150.0, hydrophobicity=0.4)
    r = pm.map(protein)
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    for key in ["nu", "ka", "sigma"]:
        if key not in r:
            return False, f"Missing key: {key}"
    if r["nu"] <= 0:
        return False, f"nu={r['nu']}, expected > 0"
    return True, f"nu={r['nu']:.2f}, ka={r['ka']:.4f}, sigma={r['sigma']:.2f}"

def _check_property_mapper_ph_sensitivity() -> Tuple[bool, str]:
    from src.PropertyMapper import PropertyMapper, ProteinProperties
    pm = PropertyMapper()
    p1 = ProteinProperties(name="pH6", pI=7.5, MW_kDa=150.0, hydrophobicity=0.4, pH_working=6.0)
    p2 = ProteinProperties(name="pH8", pI=7.5, MW_kDa=150.0, hydrophobicity=0.4, pH_working=8.0)
    r1, r2 = pm.map(p1), pm.map(p2)
    ok = r1["nu"] != r2["nu"]
    return ok, f"pH 6→nu={r1['nu']:.2f}, pH 8→nu={r2['nu']:.2f}"


# ── report_export ──────────────────────────────────────────────────────

def _check_report_export_returns_paths() -> Tuple[bool, str]:
    """export_global_report returns (docx_path, json_path) with valid JSON."""
    import tempfile, json, os
    from src.report_export import export_global_report
    mock_intent = {
        "name": "ContractTest", "source": "fasta", "molecule_class": "canonical_mab",
        "molecule_class_info": {"display_name": "Canonical mAb", "confidence": "High",
                                "has_fc_region": True, "expects_glycosylation": True, "evidence": ["IgG"]},
        "pI": 8.5, "mw": 148.0, "gravy": -0.4, "hydrophobicity": 0.42,
        "deam_sites": 3, "ox_sites": 5, "acidic_residues": 80, "basic_residues": 95,
        "seq_length": 1320, "cysteine_count": 32,
        "chains": [{"name": "HC", "sequence": "EVQLVES...", "copy_number": 2},
                   {"name": "LC", "sequence": "DIQMTQ...", "copy_number": 2}],
        "chain_analyses": [{"name": "HC", "chain_type": "heavy", "length": 449,
            "cdrs": [{"name": "CDR-H1", "start": 26, "end": 35, "sequence": "GYTFTSYG"}],
            "liabilities": {"met_count": 3, "trp_count": 2, "deamidation_count": 2,
                            "n_glyco_count": 1, "dp_count": 0, "isomerization_count": 1,
                            "cys_count": 11, "acidic_count": 40, "basic_count": 48}}],
        "liability_summary": {"deamidation_sites": 3, "oxidation_sites": 5,
                              "asp_isomerization_sites": 2, "dp_sites": 0, "n_glycosylation_sites": 2},
    }
    mock_cache = {
        "dev_result": {"data": {"predictions": {"agg_risk": 0.18, "stability": 0.85, "viscosity_risk": 0.07},
                                "score": {"score": 0.15, "grade": "Low Risk"}, "mode": "rule_based"}},
        "predictor_source": "rule_based",
        "predictor_detail": "Rule-Based Heuristic (PropertyMapper v5.0)",
    }
    with tempfile.TemporaryDirectory() as td:
        docx_path, json_path = export_global_report(mock_intent, mock_cache, output_dir=td)
        if not os.path.isfile(json_path):
            return False, f"JSON not created: {json_path}"
        with open(json_path) as f:
            data = json.load(f)
        if "executive_summary" not in data:
            return False, "JSON missing executive_summary"
        if "context" not in data:
            return False, "JSON missing context"
        return True, f"JSON={os.path.getsize(json_path)}B, DOCX={'exists' if os.path.isfile(docx_path) else 'missing'}"


def _check_report_export_json_schema() -> Tuple[bool, str]:
    """Exported JSON has all required top-level sections."""
    import tempfile, json
    from src.report_export import export_global_report
    mock_intent = {
        "name": "SchemaCheck", "source": "fasta", "molecule_class": "canonical_mab",
        "molecule_class_info": {"display_name": "Canonical mAb", "confidence": "High",
                                "has_fc_region": True, "expects_glycosylation": True, "evidence": []},
        "pI": 8.0, "mw": 150.0, "gravy": -0.35, "hydrophobicity": 0.41,
        "deam_sites": 2, "ox_sites": 4, "acidic_residues": 75, "basic_residues": 90,
        "seq_length": 1300, "cysteine_count": 30,
        "chains": [{"name": "HC", "sequence": "EVQLVES...", "copy_number": 2}],
        "chain_analyses": [],
    }
    mock_cache = {
        "dev_result": {"data": {"predictions": {"agg_risk": 0.2, "stability": 0.8},
                                "score": {"score": 0.2, "grade": "Low Risk"}, "mode": "rule_based"}},
    }
    with tempfile.TemporaryDirectory() as td:
        _, json_path = export_global_report(mock_intent, mock_cache, output_dir=td)
        with open(json_path) as f:
            data = json.load(f)
        required = ["context", "executive_summary", "molecule_overview", "developability"]
        missing = [k for k in required if k not in data]
        if missing:
            return False, f"Missing sections: {missing}"
        return True, f"All {len(required)} required sections present"


# ── purification_optimizer ────────────────────────────────────────────

def _check_doe_returns_valid_result() -> Tuple[bool, str]:
    """run_doe_optimization returns DoEOptimization with valid optimal result."""
    from src.purification_optimizer import run_doe_optimization
    opt = run_doe_optimization(pI=8.5, mw=150.0, ph_steps=5, gradient_steps=5,
                               salt_steps=2, load_steps=2)
    if opt.optimal is None:
        return False, "No optimal result found"
    o = opt.optimal
    if not (0.0 <= o.yield_main <= 100.0):
        return False, f"Yield out of range: {o.yield_main}"
    if not (0.0 <= o.pool_purity_pct <= 100.0):
        return False, f"Purity out of range: {o.pool_purity_pct}"
    if o.resolution_min < 0:
        return False, f"Negative resolution: {o.resolution_min}"
    return True, f"Optimal: pH={o.elution_ph:.1f}, yield={o.yield_main:.1f}%, purity={o.pool_purity_pct:.1f}%, Rs={o.resolution_min:.2f}"


def _check_doe_ph_adaptive_range() -> Tuple[bool, str]:
    """pH range auto-adapts based on pI."""
    from src.purification_optimizer import run_doe_optimization
    opt_low = run_doe_optimization(pI=6.0, ph_steps=3, gradient_steps=3, salt_steps=2, load_steps=2)
    opt_high = run_doe_optimization(pI=9.0, ph_steps=3, gradient_steps=3, salt_steps=2, load_steps=2)
    if opt_low.optimal is None or opt_high.optimal is None:
        return False, "Missing optimal for pI=6.0 or pI=9.0"
    ph_diff = abs(opt_high.optimal.elution_ph - opt_low.optimal.elution_ph)
    if ph_diff < 0.1:
        return False, f"No pH discrimination: pI6→pH={opt_low.optimal.elution_ph:.1f}, pI9→pH={opt_high.optimal.elution_ph:.1f}"
    return True, f"pI-adaptive: pI6→pH={opt_low.optimal.elution_ph:.1f}, pI9→pH={opt_high.optimal.elution_ph:.1f} (Δ={ph_diff:.1f})"


def _check_doe_mass_balance() -> Tuple[bool, str]:
    """Mass balance: yield_main should be in physically reasonable range."""
    from src.purification_optimizer import run_doe_optimization
    opt = run_doe_optimization(pI=8.0, ph_steps=4, gradient_steps=4, salt_steps=2, load_steps=2)
    if opt.optimal is None:
        return False, "No optimal for mass balance check"
    o = opt.optimal
    # yield_main is on a 0-1 scale (0.93 = 93% recovery)
    if o.yield_main < 0.30:
        return False, f"Unrealistically low yield: {o.yield_main * 100:.1f}%"
    if o.yield_main > 1.0:
        return False, f"Yield exceeds 100%: {o.yield_main * 100:.1f}%"
    return True, f"Yield={o.yield_main * 100:.1f}%, purity={o.pool_purity_pct:.1f}%, feasible={o.feasible}"


# ── batch_processor ───────────────────────────────────────────────────

def _check_batch_orchestrator_import() -> Tuple[bool, str]:
    """HighThroughputOrchestrator can be imported and instantiated."""
    from src.batch_processor import HighThroughputOrchestrator
    orch = HighThroughputOrchestrator(max_workers=1)
    for attr in ["run_batch", "run_batch_sequential"]:
        if not hasattr(orch, attr):
            return False, f"Missing method: {attr}"
    return True, "HighThroughputOrchestrator importable with run_batch + run_batch_sequential"


# ═══════════════════════════════════════════════════════════════════════
#  Registry + Runner
# ═══════════════════════════════════════════════════════════════════════

CONTRACTS = [
    {"name": "scaleup_returns_result", "module": "scaleup_twin", "fn": _check_scaleup_returns_result},
    {"name": "scaleup_scaling_factor", "module": "scaleup_twin", "fn": _check_scaleup_scaling_factor},
    {"name": "pareto_returns_result", "module": "pareto_optimizer", "fn": _check_pareto_returns_result},
    {"name": "pareto_dominance", "module": "pareto_optimizer", "fn": _check_pareto_dominance},
    {"name": "harmonizer_returns_dict", "module": "data_harmonizer", "fn": _check_harmonizer_returns_dict},
    {"name": "registry_known_classes", "module": "molecule_registry", "fn": _check_registry_known_classes},
    {"name": "registry_risk_weights", "module": "molecule_registry", "fn": _check_registry_risk_weights},
    {"name": "ood_baseline_compute", "module": "ood_baseline", "fn": _check_ood_baseline_compute},
    {"name": "workspace_create", "module": "workspace_manager", "fn": _check_workspace_create},
    {"name": "property_mapper_map", "module": "property_mapper", "fn": _check_property_mapper_map},
    {"name": "property_mapper_ph_sensitivity", "module": "property_mapper", "fn": _check_property_mapper_ph_sensitivity},
    {"name": "report_export_returns_paths", "module": "report_export", "fn": _check_report_export_returns_paths},
    {"name": "report_export_json_schema", "module": "report_export", "fn": _check_report_export_json_schema},
    {"name": "doe_returns_valid_result", "module": "purification_optimizer", "fn": _check_doe_returns_valid_result},
    {"name": "doe_ph_adaptive_range", "module": "purification_optimizer", "fn": _check_doe_ph_adaptive_range},
    {"name": "doe_mass_balance", "module": "purification_optimizer", "fn": _check_doe_mass_balance},
    {"name": "batch_orchestrator_import", "module": "batch_processor", "fn": _check_batch_orchestrator_import},
]

def run_infra_contracts(module_filter=None, verbose=True):
    checks = CONTRACTS if not module_filter else [c for c in CONTRACTS if c["module"] == module_filter]
    passed = failed = 0
    errors = []
    for c in checks:
        try:
            ok, detail = c["fn"]()
            if ok:
                passed += 1
                if verbose: log.info("  [PASS] %s: %s", c["name"], detail)
            else:
                failed += 1; errors.append(f"{c['name']}: {detail}")
                if verbose: log.warning("  [FAIL] %s: %s", c["name"], detail)
        except Exception as exc:
            failed += 1; errors.append(f"{c['name']}: {exc}")
            if verbose: log.error("  [ERR]  %s: %s", c["name"], exc)
    return {"passed": passed, "failed": failed, "total": passed+failed, "errors": errors, "all_passed": failed==0}

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default=None)
    args = parser.parse_args()
    result = run_infra_contracts(module_filter=args.module)
    print(f"\n{'='*50}\nInfra Contracts: {result['passed']}/{result['total']} passed")
    if result["errors"]:
        for e in result["errors"]: print(f"  - {e}")
    print(f"Status: {'ALL PASSED' if result['all_passed'] else 'FAILURES DETECTED'}")
    sys.exit(0 if result["all_passed"] else 1)
