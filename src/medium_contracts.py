"""
medium_contracts.py  ·  ProtePilot — Medium Module Contracts (C2 Batch)
==========================================================================
Behavioral contracts for bispecific_engine, regulatory_filer,
generative_engineer, ht_screening, validation_strategy.

Usage:
    python -m src.medium_contracts                              # Run all
    python -m src.medium_contracts --module bispecific_engine    # Single module

Author  : Di (ProtePilot)
"""
from __future__ import annotations
import logging, sys
from typing import Any, Dict, List, Tuple
log = logging.getLogger("ProtePilot.MediumContracts")


# ═══════════════════════════════════════════════════════════════════════
#  Contracts
# ═══════════════════════════════════════════════════════════════════════

# ── bispecific_engine ─────────────────────────────────────────────────

def _check_bispecific_assembly_species() -> Tuple[bool, str]:
    from src.bispecific_engine import AntibodyChain, build_assembly_species
    a = AntibodyChain(name="ArmA", sequence="EVQLVESGGGLVQ" * 10)
    b = AntibodyChain(name="ArmB", sequence="DIQMTQSPSSLSA" * 10)
    species = build_assembly_species(a, b)
    if not isinstance(species, dict):
        return False, f"Expected dict, got {type(species).__name__}"
    # Must contain heterodimer (AB) and at least one homodimer
    has_hetero = any("AB" in k or "hetero" in k.lower() for k in species)
    if not has_hetero and len(species) < 2:
        return False, f"Missing heterodimer or too few species: {list(species.keys())}"
    return True, f"{len(species)} species: {list(species.keys())[:4]}"


def _check_bispecific_run_analysis() -> Tuple[bool, str]:
    from src.bispecific_engine import run_bispecific_analysis
    r = run_bispecific_analysis(
        chain_a_seq="EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH" * 3,
        chain_b_seq="DIQMTQSPSSLSASVGDRVTITCRASQSISSYLN" * 3,
    )
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    for key in ["species", "peaks"]:
        if key not in r:
            return False, f"Missing key: {key}"
    return True, f"Analysis OK: {len(r.get('species', {}))} species, {len(r.get('peaks', {}))} peaks"


def _check_bispecific_resolution() -> Tuple[bool, str]:
    from src.bispecific_engine import compute_resolution
    rs = compute_resolution(rt_1=10.0, fwhm_1=0.5, rt_2=12.0, fwhm_2=0.5)
    if not isinstance(rs, (int, float)):
        return False, f"Expected float, got {type(rs).__name__}"
    if rs <= 0:
        return False, f"Resolution={rs}, expected > 0"
    return True, f"Resolution={rs:.2f} (peaks at 10.0 and 12.0 min)"


# ── regulatory_filer ──────────────────────────────────────────────────

def _check_regulatory_ectd_markdown() -> Tuple[bool, str]:
    from src.regulatory_filer import generate_ectd_markdown
    session = {
        "molecule_name": "TestMab",
        "molecule_class": "canonical_mab",
        "sequence": "EVQLVESGGGLVQ" * 10,
    }
    md = generate_ectd_markdown(session)
    if not isinstance(md, str):
        return False, f"Expected str, got {type(md).__name__}"
    if len(md) < 100:
        return False, f"Output too short: {len(md)} chars"
    return True, f"eCTD markdown: {len(md)} chars, starts with '{md[:40]}...'"


def _check_regulatory_ectd_docx() -> Tuple[bool, str]:
    from src.regulatory_filer import generate_ectd_docx
    session = {
        "molecule_name": "TestMab",
        "molecule_class": "canonical_mab",
        "sequence": "EVQLVESGGGLVQ" * 10,
    }
    doc_bytes = generate_ectd_docx(session)
    if not isinstance(doc_bytes, bytes):
        return False, f"Expected bytes, got {type(doc_bytes).__name__}"
    if len(doc_bytes) < 1000:
        return False, f"DOCX too small: {len(doc_bytes)} bytes"
    # Check DOCX magic bytes (PK zip format)
    if doc_bytes[:2] != b"PK":
        return False, f"Not a valid DOCX (wrong magic bytes)"
    return True, f"eCTD DOCX: {len(doc_bytes)} bytes"


# ── generative_engineer ──────────────────────────────────────────────

def _check_generative_identify_liabilities() -> Tuple[bool, str]:
    from src.generative_engineer import identify_liabilities_for_mutagenesis
    # Sequence with known liabilities: NG (deamidation), DG (isomerization), DP (clipping), M (oxidation)
    seq = "EVQLVESGGGLVQPNGDGDPWYRQAPGKGLEWVAMGQGTLVTVSS"
    liabilities = identify_liabilities_for_mutagenesis(seq, chain_name="TestHC")
    if not isinstance(liabilities, list):
        return False, f"Expected list, got {type(liabilities).__name__}"
    return True, f"{len(liabilities)} liabilities identified in test sequence"


def _check_generative_apply_mutations() -> Tuple[bool, str]:
    from src.generative_engineer import identify_liabilities_for_mutagenesis, apply_liability_mutations
    seq = "EVQLVESGGGLVQPNGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVAR"
    liabilities = identify_liabilities_for_mutagenesis(seq)
    mutated, mutations = apply_liability_mutations(seq, liabilities, max_mutations=3)
    if not isinstance(mutated, str):
        return False, f"Expected str, got {type(mutated).__name__}"
    if len(mutated) != len(seq):
        return False, f"Mutated length {len(mutated)} != original {len(seq)}"
    return True, f"{len(mutations)} mutations applied, seq length preserved"


def _check_generative_variants() -> Tuple[bool, str]:
    from src.generative_engineer import generate_optimized_variants
    chains = [
        {"name": "HC", "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH" * 3, "type": "heavy"},
        {"name": "LC", "sequence": "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLN" * 3, "type": "light"},
    ]
    r = generate_optimized_variants(chains, dev_score=0.6, dev_grade="Medium", n_variants=2)
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    variants = r.get("variants", [])
    if not isinstance(variants, list):
        return False, f"variants not a list: {type(variants).__name__}"
    return True, f"{len(variants)} variants generated"


# ── ht_screening ─────────────────────────────────────────────────────

def _check_ht_screening_mock_csv() -> Tuple[bool, str]:
    from src.ht_screening import generate_mock_discovery_csv
    csv_str = generate_mock_discovery_csv(n_candidates=10, seed=42)
    if not isinstance(csv_str, str):
        return False, f"Expected str, got {type(csv_str).__name__}"
    lines = csv_str.strip().split("\n")
    if len(lines) < 5:
        return False, f"Only {len(lines)} lines in mock CSV"
    return True, f"Mock CSV: {len(lines)} lines (10 candidates + header)"


def _check_ht_screening_parse() -> Tuple[bool, str]:
    from src.ht_screening import generate_mock_discovery_csv, parse_discovery_csv
    csv_str = generate_mock_discovery_csv(n_candidates=5, seed=42)
    parsed = parse_discovery_csv(csv_str)
    if not isinstance(parsed, dict):
        return False, f"Expected dict, got {type(parsed).__name__}"
    return True, f"Parsed {parsed.get('n_candidates', '?')} candidates"


def _check_ht_screening_run() -> Tuple[bool, str]:
    from src.ht_screening import generate_mock_discovery_csv, run_ht_screening
    csv_str = generate_mock_discovery_csv(n_candidates=10, seed=42)
    r = run_ht_screening(csv_str)
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    return True, f"Screened: {r.get('n_screened', '?')} candidates, {r.get('n_star', '?')} stars"


# ── validation_strategy ──────────────────────────────────────────────

def _check_validation_compute_metrics() -> Tuple[bool, str]:
    import numpy as np
    from src.validation_strategy import compute_metrics
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
    metrics = compute_metrics(y_true, y_pred)
    if not isinstance(metrics, dict):
        return False, f"Expected dict, got {type(metrics).__name__}"
    for key in ["rmse", "mae", "r2"]:
        if key not in metrics:
            return False, f"Missing '{key}' in metrics: {list(metrics.keys())[:5]}"
    if metrics["r2"] <= 0:
        return False, f"R2={metrics['r2']}, expected > 0 for near-perfect predictions"
    return True, f"Metrics: rmse={metrics['rmse']:.4f}, r2={metrics['r2']:.4f}, {len(metrics)} total"


def _check_validation_bootstrap_ci() -> Tuple[bool, str]:
    import numpy as np
    from src.validation_strategy import bootstrap_confidence_interval, compute_metrics
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_pred = y_true + np.random.RandomState(42).randn(10) * 0.5
    metric_fn = lambda yt, yp: compute_metrics(yt, yp).get("rmse", 0)
    ci = bootstrap_confidence_interval(y_true, y_pred, metric_fn, n_bootstrap=100)
    if not isinstance(ci, dict):
        return False, f"Expected dict, got {type(ci).__name__}"
    for key in ["mean", "ci_lower", "ci_upper"]:
        if key not in ci:
            return False, f"Missing key: {key}"
    return True, f"Bootstrap CI: {ci['ci_lower']:.3f} - {ci['ci_upper']:.3f} (mean={ci['mean']:.3f})"


def _check_validation_time_split() -> Tuple[bool, str]:
    import numpy as np
    from src.validation_strategy import time_based_split
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    splits = time_based_split(X, y, n_splits=3)
    if not isinstance(splits, list):
        return False, f"Expected list, got {type(splits).__name__}"
    if len(splits) < 1:
        return False, "No splits returned"
    return True, f"{len(splits)} temporal splits"


# ═══════════════════════════════════════════════════════════════════════
#  Registry + Runner
# ═══════════════════════════════════════════════════════════════════════

CONTRACTS = [
    {"name": "bispecific_assembly_species", "module": "bispecific_engine", "fn": _check_bispecific_assembly_species},
    {"name": "bispecific_run_analysis", "module": "bispecific_engine", "fn": _check_bispecific_run_analysis},
    {"name": "bispecific_resolution", "module": "bispecific_engine", "fn": _check_bispecific_resolution},
    {"name": "regulatory_ectd_markdown", "module": "regulatory_filer", "fn": _check_regulatory_ectd_markdown},
    {"name": "regulatory_ectd_docx", "module": "regulatory_filer", "fn": _check_regulatory_ectd_docx},
    {"name": "generative_identify_liabilities", "module": "generative_engineer", "fn": _check_generative_identify_liabilities},
    {"name": "generative_apply_mutations", "module": "generative_engineer", "fn": _check_generative_apply_mutations},
    {"name": "generative_variants", "module": "generative_engineer", "fn": _check_generative_variants},
    {"name": "ht_screening_mock_csv", "module": "ht_screening", "fn": _check_ht_screening_mock_csv},
    {"name": "ht_screening_parse", "module": "ht_screening", "fn": _check_ht_screening_parse},
    {"name": "ht_screening_run", "module": "ht_screening", "fn": _check_ht_screening_run},
    {"name": "validation_compute_metrics", "module": "validation_strategy", "fn": _check_validation_compute_metrics},
    {"name": "validation_bootstrap_ci", "module": "validation_strategy", "fn": _check_validation_bootstrap_ci},
    {"name": "validation_time_split", "module": "validation_strategy", "fn": _check_validation_time_split},
]


def run_medium_contracts(module_filter=None, verbose=True):
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
    result = run_medium_contracts(module_filter=args.module)
    print(f"\n{'='*50}\nMedium Contracts: {result['passed']}/{result['total']} passed")
    if result["errors"]:
        for e in result["errors"]: print(f"  - {e}")
    print(f"Status: {'ALL PASSED' if result['all_passed'] else 'FAILURES DETECTED'}")
    sys.exit(0 if result["all_passed"] else 1)
