"""
auxiliary_contracts.py  ·  ProtePilot — Auxiliary Module Contracts
====================================================================
Formal behavioral contracts for analytical, QC, validation, feature,
and data pipeline modules.

Covered modules:
    analytical_twin       — MS characterization, intact mass, liability density
    analytical_qc_twin    — cIEF, CE-SDS, glycan simulation
    validation_planner    — Validation plan generation
    feature_registry      — Biophysical feature computation
    data_pipeline         — CSV parsing and validation

Usage:
    python -m src.auxiliary_contracts                              # Run all
    python -m src.auxiliary_contracts --module analytical_twin     # Single module

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Tuple

log = logging.getLogger("ProtePilot.AuxContracts")


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

_TEST_SEQ_VH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRGITIFGVVIIPGFFDIWGQGTLVTVSS"
_TEST_SEQ_VL = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
_TEST_PEPTIDE = "ACDEFGHIKLMNPQRSTVWY"

_MAB_CHAINS = [
    {"sequence": _TEST_SEQ_VH, "copy_number": 1, "name": "HC", "chain_type": "heavy"},
    {"sequence": _TEST_SEQ_VL, "copy_number": 1, "name": "LC", "chain_type": "light"},
]


# ═══════════════════════════════════════════════════════════════════════
#  Contract Definitions
# ═══════════════════════════════════════════════════════════════════════

# ── Analytical Twin ──────────────────────────────────────────────────

def _check_intact_mass_returns_dict() -> Tuple[bool, str]:
    """calculate_intact_mass() must return dict with mass fields."""
    from src.analytical_twin import calculate_intact_mass
    r = calculate_intact_mass(_TEST_SEQ_VH + _TEST_SEQ_VL, is_mab=True)
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    for key in ["bare_mass_da", "disulfide_corrected_da", "sequence_length"]:
        if key not in r:
            return False, f"Missing key: {key}"
    mass = r["bare_mass_da"]
    if not (1000 < mass < 500000):
        return False, f"Mass {mass} Da out of plausible range"
    return True, f"mass={mass:.1f} Da, len={r['sequence_length']}"


def _check_stoichiometric_mass() -> Tuple[bool, str]:
    """calculate_stoichiometric_intact_mass() must work with chain list."""
    from src.analytical_twin import calculate_stoichiometric_intact_mass
    r = calculate_stoichiometric_intact_mass(_MAB_CHAINS, is_mab=True)
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    if "bare_mass_da" not in r:
        return False, f"Missing bare_mass_da, keys: {list(r.keys())[:5]}"
    return True, f"stoich_mass={r['bare_mass_da']:.1f} Da"


def _check_liability_density() -> Tuple[bool, str]:
    """calculate_liability_density() must return density per 1000 residues."""
    from src.analytical_twin import calculate_liability_density
    r = calculate_liability_density(_MAB_CHAINS)
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    if "density_per_1000" not in r:
        return False, f"Missing density_per_1000"
    d = r["density_per_1000"]
    if not (0 <= d <= 500):
        return False, f"density={d} outside [0, 500]"
    return True, f"density={d:.1f}/1000 res, risk={r.get('risk_level', '?')}"


def _check_tryptic_digest() -> Tuple[bool, str]:
    """tryptic_digest() must return list of peptide dicts."""
    from src.analytical_twin import tryptic_digest
    peptides = tryptic_digest(_TEST_SEQ_VH, missed_cleavages=1)
    if not isinstance(peptides, list):
        return False, f"Expected list, got {type(peptides).__name__}"
    if len(peptides) < 1:
        return False, "No peptides generated"
    p = peptides[0]
    for key in ["sequence", "mass_avg_da", "length"]:
        if key not in p:
            return False, f"Peptide missing key: {key}"
    return True, f"{len(peptides)} peptides, first={p['sequence'][:10]}..."


def _check_ms_characterization() -> Tuple[bool, str]:
    """run_ms_characterization() must return success dict."""
    from src.analytical_twin import run_ms_characterization
    r = run_ms_characterization(
        sequence=_TEST_SEQ_VH + _TEST_SEQ_VL,
        protein_name="ContractTest",
        is_mab=True,
        chains=_MAB_CHAINS,
    )
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    if r.get("status") == "error":
        return False, f"Error: {r.get('message', '?')}"
    if "intact_mass" not in r:
        return False, f"Missing intact_mass key"
    return True, f"MS char OK, {len(r.get('peptide_map', []))} peptides"


def _check_build_super_sequence() -> Tuple[bool, str]:
    """build_super_sequence() must return concatenated string."""
    from src.analytical_twin import build_super_sequence
    s = build_super_sequence(_MAB_CHAINS)
    if not isinstance(s, str):
        return False, f"Expected str, got {type(s).__name__}"
    expected_len = len(_TEST_SEQ_VH) + len(_TEST_SEQ_VL)
    if len(s) < expected_len * 0.8:
        return False, f"Super sequence too short: {len(s)} vs expected ~{expected_len}"
    return True, f"super_seq len={len(s)}"


# ── Analytical QC Twin ───────────────────────────────────────────────

def _check_run_analytical_qc() -> Tuple[bool, str]:
    """run_analytical_qc() must return AnalyticalQCResult with sub-results."""
    from src.analytical_qc_twin import run_analytical_qc
    r = run_analytical_qc(
        sequence=_TEST_SEQ_VH + _TEST_SEQ_VL,
        pI=7.5,
        is_mab=True,
        molecule_class="canonical_mab",
    )
    for attr in ["cief", "ce_sds", "glycan", "overall_qc_pass"]:
        if not hasattr(r, attr):
            return False, f"Missing attr: {attr}"
    return True, f"QC pass={r.overall_qc_pass}, cIEF main={r.cief.main_pct:.1f}%"


def _check_cief_percentages_sum() -> Tuple[bool, str]:
    """cIEF acidic + main + basic must sum to ~100%."""
    from src.analytical_qc_twin import simulate_cief
    r = simulate_cief(
        sequence=_TEST_SEQ_VH + _TEST_SEQ_VL,
        pI=7.5,
        deamidation_sites=3,
    )
    total = r.acidic_pct + r.main_pct + r.basic_pct
    ok = 98.0 <= total <= 102.0
    return ok, f"acidic={r.acidic_pct:.1f}% + main={r.main_pct:.1f}% + basic={r.basic_pct:.1f}% = {total:.1f}%"


def _check_ce_sds_purity() -> Tuple[bool, str]:
    """CE-SDS intact + fragment + lmw + hmw must be plausible."""
    from src.analytical_qc_twin import simulate_ce_sds
    r = simulate_ce_sds(
        sequence=_TEST_SEQ_VH + _TEST_SEQ_VL,
        is_mab=True,
        molecule_class="canonical_mab",
    )
    total = r.intact_pct + r.fragment_pct + r.lmw_pct + r.hmw_pct
    ok = 90.0 <= total <= 110.0 and r.intact_pct > 0
    return ok, f"intact={r.intact_pct:.1f}%, hmw={r.hmw_pct:.1f}%, total={total:.1f}%"


def _check_glycan_profile() -> Tuple[bool, str]:
    """Glycan percentages must be non-negative."""
    from src.analytical_qc_twin import simulate_glycan_profile
    r = simulate_glycan_profile(
        n_glycosylation_sites=2,
        molecule_class="canonical_mab",
    )
    for attr in ["g0f_pct", "g1f_pct", "g2f_pct", "high_mannose_pct"]:
        v = getattr(r, attr, None)
        if v is None or v < 0:
            return False, f"{attr}={v}, expected >= 0"
    return True, f"G0F={r.g0f_pct:.1f}%, G1F={r.g1f_pct:.1f}%, dominant={r.dominant_species}"


# ── Validation Planner ───────────────────────────────────────────────

def _check_validation_plan_structure() -> Tuple[bool, str]:
    """generate_validation_plan() must return dict with assay lists."""
    from src.validation_planner import generate_validation_plan
    r = generate_validation_plan(
        risk_scores={"agg_risk": 0.3, "stability": 0.2, "viscosity_risk": 0.1},
        molecule_class="canonical_mab",
    )
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    if "required_assays" not in r and "all_assays" not in r:
        return False, f"Missing assay lists, keys: {list(r.keys())[:5]}"
    total = r.get("total_assays", len(r.get("all_assays", [])))
    if total < 1:
        return False, f"No assays generated"
    return True, f"{total} assays for canonical_mab"


def _check_validation_plan_peptide() -> Tuple[bool, str]:
    """Validation plan for peptide should differ from mAb."""
    from src.validation_planner import generate_validation_plan
    r = generate_validation_plan(
        risk_scores={"agg_risk": 0.5, "stability": 0.5, "viscosity_risk": 0.3},
        molecule_class="peptide",
    )
    total = r.get("total_assays", len(r.get("all_assays", [])))
    ok = total >= 1
    return ok, f"{total} assays for peptide"


# ── Feature Registry ─────────────────────────────────────────────────

def _check_compute_features_returns_featureset() -> Tuple[bool, str]:
    """compute_features() must return FeatureSet with core features."""
    from src.feature_registry import compute_features
    fs = compute_features(_TEST_SEQ_VH + _TEST_SEQ_VL, molecule_class="canonical_mab")
    if not hasattr(fs, "features"):
        return False, f"No 'features' attribute on {type(fs).__name__}"
    core = ["pI", "mw_kda", "hydrophobicity", "gravy"]
    missing = [f for f in core if f not in fs.features]
    if missing:
        return False, f"Missing features: {missing}"
    pi = fs.value("pI")
    if pi is None or not (2 < pi < 14):
        return False, f"pI={pi} out of range"
    return True, f"pI={pi:.2f}, mw={fs.value('mw_kda'):.1f} kDa, {len(fs.features)} features"


def _check_feature_ml_vector() -> Tuple[bool, str]:
    """ml_vector() must return numeric list of consistent length."""
    from src.feature_registry import compute_features
    fs = compute_features(_TEST_PEPTIDE, molecule_class="peptide")
    vec = fs.ml_vector()
    if not isinstance(vec, list):
        return False, f"ml_vector() returned {type(vec).__name__}"
    if len(vec) < 5:
        return False, f"ml_vector has only {len(vec)} elements"
    if any(v is None for v in vec):
        return False, f"ml_vector contains None"
    return True, f"ml_vector: {len(vec)}-dim, first 3 = {vec[:3]}"


def _check_feature_report_dict() -> Tuple[bool, str]:
    """report_dict() must return display-ready dict."""
    from src.feature_registry import compute_features
    fs = compute_features(_TEST_SEQ_VH, molecule_class="canonical_mab")
    rd = fs.report_dict()
    if not isinstance(rd, dict):
        return False, f"Expected dict, got {type(rd).__name__}"
    if len(rd) < 3:
        return False, f"Only {len(rd)} entries"
    return True, f"report_dict: {len(rd)} entries"


# ── Data Pipeline ────────────────────────────────────────────────────

def _check_parse_csv_jain137() -> Tuple[bool, str]:
    """parse_csv_upload() must parse Jain-137 format."""
    from src.data_pipeline import parse_csv_upload
    import io
    csv = "Name,Sequence,Exp_Aggregation_Percent\nTestMol,ACDEFGHIKLMNPQRSTVWY,2.5\n"
    r = parse_csv_upload(io.StringIO(csv), filename="test.csv")
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    if r.get("status") == "error":
        return False, f"Parse error: {r.get('message', '?')}"
    if r.get("n_rows", 0) < 1:
        return False, f"No rows parsed"
    return True, f"status={r['status']}, n_rows={r.get('n_rows')}, schema={r.get('schema', '?')}"


def _check_validate_csv_data() -> Tuple[bool, str]:
    """validate_csv_data() must return status dict."""
    from src.data_pipeline import validate_csv_data
    data = [
        {"Name": "Mol1", "Sequence": "ACDEFGHIKLMNPQRSTVWY", "pI": 5.5},
        {"Name": "Mol2", "Sequence": "FWFWFWFWFW", "pI": 6.0},
    ]
    r = validate_csv_data(data)
    if not isinstance(r, dict):
        return False, f"Expected dict, got {type(r).__name__}"
    if "status" not in r:
        return False, f"Missing 'status' key"
    return True, f"status={r['status']}, n_rows={r.get('n_rows', '?')}"


# ═══════════════════════════════════════════════════════════════════════
#  Contract Registry
# ═══════════════════════════════════════════════════════════════════════

CONTRACTS: List[Dict[str, Any]] = [
    # Analytical Twin
    {"name": "intact_mass_returns_dict", "module": "analytical_twin",
     "fn": _check_intact_mass_returns_dict,
     "desc": "calculate_intact_mass() returns dict with mass fields"},
    {"name": "stoichiometric_mass", "module": "analytical_twin",
     "fn": _check_stoichiometric_mass,
     "desc": "calculate_stoichiometric_intact_mass() with chain list"},
    {"name": "liability_density", "module": "analytical_twin",
     "fn": _check_liability_density,
     "desc": "calculate_liability_density() returns density per 1000 res"},
    {"name": "tryptic_digest", "module": "analytical_twin",
     "fn": _check_tryptic_digest,
     "desc": "tryptic_digest() returns peptide list"},
    {"name": "ms_characterization", "module": "analytical_twin",
     "fn": _check_ms_characterization,
     "desc": "run_ms_characterization() returns success dict"},
    {"name": "build_super_sequence", "module": "analytical_twin",
     "fn": _check_build_super_sequence,
     "desc": "build_super_sequence() returns concatenated string"},

    # Analytical QC Twin
    {"name": "run_analytical_qc", "module": "analytical_qc_twin",
     "fn": _check_run_analytical_qc,
     "desc": "run_analytical_qc() returns AnalyticalQCResult"},
    {"name": "cief_percentages_sum", "module": "analytical_qc_twin",
     "fn": _check_cief_percentages_sum,
     "desc": "cIEF acidic + main + basic ≈ 100%"},
    {"name": "ce_sds_purity", "module": "analytical_qc_twin",
     "fn": _check_ce_sds_purity,
     "desc": "CE-SDS purity components plausible"},
    {"name": "glycan_profile", "module": "analytical_qc_twin",
     "fn": _check_glycan_profile,
     "desc": "Glycan percentages non-negative"},

    # Validation Planner
    {"name": "validation_plan_structure", "module": "validation_planner",
     "fn": _check_validation_plan_structure,
     "desc": "generate_validation_plan() returns assay lists"},
    {"name": "validation_plan_peptide", "module": "validation_planner",
     "fn": _check_validation_plan_peptide,
     "desc": "Peptide validation plan differs from mAb"},

    # Feature Registry
    {"name": "compute_features_featureset", "module": "feature_registry",
     "fn": _check_compute_features_returns_featureset,
     "desc": "compute_features() returns FeatureSet with core features"},
    {"name": "feature_ml_vector", "module": "feature_registry",
     "fn": _check_feature_ml_vector,
     "desc": "ml_vector() returns numeric list"},
    {"name": "feature_report_dict", "module": "feature_registry",
     "fn": _check_feature_report_dict,
     "desc": "report_dict() returns display-ready dict"},

    # Data Pipeline
    {"name": "parse_csv_jain137", "module": "data_pipeline",
     "fn": _check_parse_csv_jain137,
     "desc": "parse_csv_upload() parses Jain-137 format"},
    {"name": "validate_csv_data", "module": "data_pipeline",
     "fn": _check_validate_csv_data,
     "desc": "validate_csv_data() returns status dict"},
]


# ═══════════════════════════════════════════════════════════════════════
#  Contract Runner
# ═══════════════════════════════════════════════════════════════════════

def run_auxiliary_contracts(
    module_filter: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    checks = CONTRACTS
    if module_filter:
        checks = [c for c in checks if c["module"] == module_filter]

    passed = failed = 0
    errors = []

    for contract in checks:
        name = contract["name"]
        try:
            ok, detail = contract["fn"]()
            if ok:
                passed += 1
                if verbose:
                    log.info("  [PASS] %s: %s", name, detail)
            else:
                failed += 1
                errors.append(f"{name}: {detail}")
                if verbose:
                    log.warning("  [FAIL] %s: %s", name, detail)
        except Exception as exc:
            failed += 1
            errors.append(f"{name}: Exception: {exc}")
            if verbose:
                log.error("  [ERR]  %s: %s", name, exc)

    return {
        "passed": passed, "failed": failed, "total": passed + failed,
        "errors": errors, "all_passed": failed == 0,
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Auxiliary Module Contract Checks")
    parser.add_argument("--module", default=None,
                        choices=["analytical_twin", "analytical_qc_twin",
                                 "validation_planner", "feature_registry",
                                 "data_pipeline"])
    args = parser.parse_args()

    log.info("Running auxiliary contracts%s...",
             f" (module={args.module})" if args.module else "")
    result = run_auxiliary_contracts(module_filter=args.module)

    print(f"\n{'='*50}")
    print(f"Auxiliary Contracts: {result['passed']}/{result['total']} passed")
    if result["errors"]:
        print("Failures:")
        for e in result["errors"]:
            print(f"  - {e}")
    print(f"Status: {'ALL PASSED' if result['all_passed'] else 'FAILURES DETECTED'}")
    sys.exit(0 if result["all_passed"] else 1)
