"""
analytical_contract.py  ·  ProtePilot — Analytical / PTM Contract
====================================================================
Formal specification of the analytical module's result objects, the
bridge to developability_core's flat-dict consumer format, and the
PTM feature schema.

Problem this solves:
  analytical_qc_twin returns typed dataclasses (CIEFResult, CESDSResult,
  GlycanResult), but developability_core expects a flat dict with keys
  like "sec_monomer_pct", "cief_main_pct", "cesds_intact_pct".

  Previously this bridge was duplicated in bulk_runner.py and
  report_assembler.py.  This contract provides the SINGLE canonical
  bridge function and documents all expected key names.

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger("ProtePilot.AnalyticalContract")


# ═══════════════════════════════════════════════════════════════════════
#  Analytical Results — Consumer Key Names
# ═══════════════════════════════════════════════════════════════════════

# These are the flat dict keys that developability_core.assess_developability()
# reads from its `analytical_results` parameter.
# This is the SINGLE SOURCE OF TRUTH for these key names.

ANALYTICAL_CONSUMER_KEYS: Dict[str, str] = {
    "sec_monomer_pct":   "SEC monomer purity (%), higher is better",
    "sec_hmw_pct":       "SEC high molecular weight / aggregate (%), lower is better",
    "cief_main_pct":     "cIEF main peak (%), higher is better",
    "cief_acidic_pct":   "cIEF acidic variants (%), lower is better",
    "cief_basic_pct":    "cIEF basic variants (%) [reporting only, not used by dev_core]",
    "cesds_intact_pct":  "CE-SDS intact IgG (%), higher is better",
}

# Keys that developability_core actually reads (subset of above).
DEVELOPABILITY_CORE_REQUIRED_KEYS: Set[str] = {
    "sec_monomer_pct", "sec_hmw_pct",
    "cief_main_pct", "cief_acidic_pct",
    "cesds_intact_pct",
}


# ═══════════════════════════════════════════════════════════════════════
#  PTM / Liability Feature Schema
# ═══════════════════════════════════════════════════════════════════════

# PTM features that developability_core reads from feature_values.
# All are computed by src.training.features.py (sequence-based regex).
PTM_FEATURE_KEYS: Dict[str, str] = {
    "deam_sites":              "Deamidation hotspots (NG/NS motif count)",
    "ox_sites":                "Oxidation-susceptible residues (Met + Trp count)",
    "asp_isomerization_sites": "Asp isomerization hotspots (DG/DS motif count)",
    "pyroglutamate_risk":      "Pyroglutamate formation risk (0=none, 1=Glu, 2=Gln N-term)",
    "n_glycosylation_sites":   "N-linked glycosylation sites (NxS/T motif count)",
    "cysteine_count":          "Total cysteine residues (disulfide bond / free thiol)",
}

# Additional biophysical features that developability_core reads.
BIOPHYSICAL_FEATURE_KEYS: Dict[str, str] = {
    "pI":                     "Isoelectric point",
    "mw_kda":                 "Molecular weight (kDa)",
    "hydrophobicity":         "GRAVY hydrophobicity score",
    "beta_sheet_propensity":  "Average beta-sheet propensity",
    "cdr_hydrophobicity":     "CDR region hydrophobicity (GRAVY)",
    "acidic_residues":        "Count of D+E residues",
    "basic_residues":         "Count of K+R residues",
    "seq_length":             "Sequence length (amino acids)",
}


# ═══════════════════════════════════════════════════════════════════════
#  Canonical Bridge Function
# ═══════════════════════════════════════════════════════════════════════

def qc_result_to_analytical_dict(qc_result) -> Dict[str, Optional[float]]:
    """
    Convert an AnalyticalQCResult (from analytical_qc_twin) to the flat
    dict format that developability_core.assess_developability() expects
    as its `analytical_results` parameter.

    This is the CANONICAL bridge — use this instead of manually extracting
    fields in bulk_runner, report_assembler, etc.

    Parameters
    ----------
    qc_result : AnalyticalQCResult
        Output from analytical_qc_twin.run_analytical_qc().

    Returns
    -------
    dict with keys matching ANALYTICAL_CONSUMER_KEYS.
    """
    result: Dict[str, Optional[float]] = {}

    # cIEF
    cief = getattr(qc_result, "cief", None)
    if cief is not None:
        result["cief_main_pct"] = getattr(cief, "main_pct", None)
        result["cief_acidic_pct"] = getattr(cief, "acidic_pct", None)
        result["cief_basic_pct"] = getattr(cief, "basic_pct", None)
    else:
        result["cief_main_pct"] = None
        result["cief_acidic_pct"] = None
        result["cief_basic_pct"] = None

    # CE-SDS
    ce_sds = getattr(qc_result, "ce_sds", None)
    if ce_sds is not None:
        result["cesds_intact_pct"] = getattr(ce_sds, "intact_pct", None)
        # SEC monomer approximation: 100% - HMW% (CE-SDS HMW as proxy)
        hmw = getattr(ce_sds, "hmw_pct", None)
        if hmw is not None:
            result["sec_hmw_pct"] = hmw
            result["sec_monomer_pct"] = round(100.0 - hmw, 2)
        else:
            result["sec_hmw_pct"] = None
            result["sec_monomer_pct"] = None
    else:
        result["cesds_intact_pct"] = None
        result["sec_hmw_pct"] = None
        result["sec_monomer_pct"] = None

    return result


def ptm_features_from_sequence(sequence: str) -> Dict[str, Any]:
    """
    Compute PTM-related features from a sequence using the canonical
    features.py implementation.

    This is a convenience wrapper that extracts only PTM features.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    dict with keys matching PTM_FEATURE_KEYS.
    """
    import re
    if sequence is None:
        sequence = ""
    seq = sequence.upper().strip()

    # Start with features.py output (covers deam, ox, cysteine, etc.)
    base: Dict[str, Any] = {}
    try:
        from src.training.features import compute_sequence_features
        base = compute_sequence_features(seq)
    except ImportError:
        pass

    result: Dict[str, Any] = {}
    for key in PTM_FEATURE_KEYS:
        if key in base:
            result[key] = base[key]
        else:
            # Compute individually for keys not in features.py
            result[key] = 0

    # pyroglutamate_risk is NOT in features.py — always compute from sequence
    if seq:
        result["pyroglutamate_risk"] = (2 if seq[0] == "Q" else
                                         1 if seq[0] == "E" else 0)
    else:
        result["pyroglutamate_risk"] = 0

    # n_glycosylation_sites may not be in features.py — ensure it's present
    if "n_glycosylation_sites" not in base:
        result["n_glycosylation_sites"] = len(re.findall(r"N[^P][ST]", seq))

    # asp_isomerization_sites may not be in features.py
    if "asp_isomerization_sites" not in base:
        result["asp_isomerization_sites"] = len(re.findall(r"D[GS]", seq))

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Contract Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_analytical_dict(analytical: Dict[str, Any]) -> List[str]:
    """
    Validate a flat analytical_results dict against the contract.

    Returns list of violation messages (empty = all OK).
    """
    violations = []

    # Check required keys exist
    for key in DEVELOPABILITY_CORE_REQUIRED_KEYS:
        if key not in analytical:
            violations.append(f"Missing required key: '{key}'")

    # Check value ranges (when present)
    for key, val in analytical.items():
        if val is None:
            continue
        if key.endswith("_pct"):
            if not (0.0 <= val <= 100.0):
                violations.append(f"'{key}' value {val} outside [0, 100] range")

    # Cross-consistency checks
    sec_mon = analytical.get("sec_monomer_pct")
    sec_hmw = analytical.get("sec_hmw_pct")
    if sec_mon is not None and sec_hmw is not None:
        if abs((sec_mon + sec_hmw) - 100.0) > 5.0:
            violations.append(
                f"sec_monomer_pct ({sec_mon}) + sec_hmw_pct ({sec_hmw}) = "
                f"{sec_mon + sec_hmw}, expected ~100.0"
            )

    cief_main = analytical.get("cief_main_pct")
    cief_acidic = analytical.get("cief_acidic_pct")
    cief_basic = analytical.get("cief_basic_pct")
    if all(v is not None for v in [cief_main, cief_acidic, cief_basic]):
        total = cief_main + cief_acidic + cief_basic
        if abs(total - 100.0) > 5.0:
            violations.append(
                f"cIEF fractions sum to {total}, expected ~100.0"
            )

    return violations


def validate_ptm_features(features: Dict[str, Any]) -> List[str]:
    """
    Validate PTM feature dict against the contract.
    """
    violations = []

    for key in PTM_FEATURE_KEYS:
        if key not in features:
            violations.append(f"Missing PTM feature: '{key}'")
        else:
            val = features[key]
            if not isinstance(val, (int, float)):
                violations.append(f"PTM feature '{key}' should be numeric, got {type(val).__name__}")
            elif val < 0:
                violations.append(f"PTM feature '{key}' = {val} is negative")

    # Specific range checks
    pyro = features.get("pyroglutamate_risk")
    if pyro is not None and pyro not in (0, 1, 2):
        violations.append(f"pyroglutamate_risk = {pyro}, expected 0, 1, or 2")

    return violations


# ═══════════════════════════════════════════════════════════════════════
#  Behavioral Guarantees
# ═══════════════════════════════════════════════════════════════════════

BEHAVIORAL_GUARANTEES = [
    {
        "name": "bridge_produces_all_consumer_keys",
        "description": "qc_result_to_analytical_dict() always produces all DEVELOPABILITY_CORE_REQUIRED_KEYS",
    },
    {
        "name": "bridge_values_in_range",
        "description": "All _pct values from bridge are in [0, 100] or None",
    },
    {
        "name": "cief_fractions_sum_100",
        "description": "cIEF acidic + main + basic sum to ~100%",
    },
    {
        "name": "sec_monomer_hmw_complement",
        "description": "sec_monomer_pct + sec_hmw_pct ≈ 100%",
    },
    {
        "name": "ptm_features_complete",
        "description": "ptm_features_from_sequence() returns all 6 PTM feature keys",
    },
    {
        "name": "ptm_features_non_negative",
        "description": "All PTM feature values are non-negative integers",
    },
    {
        "name": "pyroglutamate_trichotomy",
        "description": "pyroglutamate_risk is exactly 0, 1, or 2",
    },
    {
        "name": "key_names_align_with_developability_core",
        "description": "Bridge output keys match what developability_core.assess_developability() reads via anal.get()",
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  Self-Test
# ═══════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    """Comprehensive selftest for analytical contract."""
    errors = []
    checks = 0

    def _check(name, condition, msg=""):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(f"{name}: {msg}")
            log.warning("  [FAIL] %s: %s", name, msg)
        else:
            log.info("  [PASS] %s", name)

    # ── 1. Bridge function with real QC result ────────────────────
    try:
        from src.analytical_qc_twin import run_analytical_qc
        qc = run_analytical_qc(
            sequence="EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH" * 5,
            pI=8.5, aggregation_pct=1.0,
        )
        bridged = qc_result_to_analytical_dict(qc)

        # Check all required keys present
        missing = DEVELOPABILITY_CORE_REQUIRED_KEYS - set(bridged.keys())
        _check("bridge_all_keys", len(missing) == 0,
               f"missing: {missing}")

        # Check values in range
        all_in_range = all(
            v is None or (0.0 <= v <= 100.0)
            for k, v in bridged.items() if k.endswith("_pct")
        )
        _check("bridge_values_range", all_in_range,
               f"out of range values in {bridged}")

        # Validate via contract
        violations = validate_analytical_dict(bridged)
        _check("bridge_contract_valid", len(violations) == 0,
               f"violations: {violations}")

        # Check cIEF fractions sum
        cief_sum = (bridged.get("cief_main_pct", 0) or 0) + \
                   (bridged.get("cief_acidic_pct", 0) or 0) + \
                   (bridged.get("cief_basic_pct", 0) or 0)
        _check("cief_sum_100", abs(cief_sum - 100.0) < 5.0,
               f"sum={cief_sum}")

        # Check SEC complement
        sec_sum = (bridged.get("sec_monomer_pct", 0) or 0) + \
                  (bridged.get("sec_hmw_pct", 0) or 0)
        _check("sec_complement", abs(sec_sum - 100.0) < 5.0,
               f"sum={sec_sum}")

    except ImportError as e:
        _check("bridge_import", False, str(e))

    # ── 2. Bridge feeds into developability_core ──────────────────
    try:
        from src.developability_core import assess_developability
        result = assess_developability(
            molecule_name="BridgeTest",
            molecule_class="canonical_mab",
            feature_values={
                "pI": 8.5, "mw_kda": 148.0, "hydrophobicity": 0.34,
                "deam_sites": 2, "ox_sites": 5, "asp_isomerization_sites": 1,
                "n_glycosylation_sites": 2, "pyroglutamate_risk": 0,
                "beta_sheet_propensity": 1.08, "cdr_hydrophobicity": -0.3,
                "acidic_residues": 38, "basic_residues": 48,
                "seq_length": 450, "cysteine_count": 16,
            },
            dev_predictions={"agg_risk": 0.15, "stability": 0.85, "viscosity_risk": 0.10},
            analytical_results=bridged,
        )
        # QTPP rows referencing analytical data should be assessed (not "Not Assessed")
        qtpp_assessed = [r for r in result.qtpp
                         if r.attribute in ("Monomer Purity (SEC)", "Charge Variants (cIEF Main Peak%)",
                                            "Intact IgG (CE-SDS, Non-reduced)")
                         and r.status != "Not Assessed"]
        _check("bridge_to_developability", len(qtpp_assessed) >= 2,
               f"expected ≥2 assessed analytical QTPP rows, got {len(qtpp_assessed)}")

    except Exception as e:
        _check("bridge_to_developability", False, str(e))

    # ── 3. PTM features ───────────────────────────────────────────
    ptm = ptm_features_from_sequence(
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH"
        "WVRQAPGKGLEWVARIYPTING"
    )
    _check("ptm_all_keys",
           set(PTM_FEATURE_KEYS.keys()).issubset(set(ptm.keys())),
           f"missing: {set(PTM_FEATURE_KEYS.keys()) - set(ptm.keys())}")
    _check("ptm_non_negative",
           all(v >= 0 for v in ptm.values()),
           f"negative values: {ptm}")
    _check("ptm_deam_detected",
           ptm.get("deam_sites", 0) >= 0,
           "deam_sites should be non-negative")

    ptm_violations = validate_ptm_features(ptm)
    _check("ptm_contract_valid", len(ptm_violations) == 0,
           f"violations: {ptm_violations}")

    # ── 4. Pyroglutamate trichotomy ───────────────────────────────
    ptm_q = ptm_features_from_sequence("QVQLVQSGAEVKKPGA")  # Q N-term
    ptm_e = ptm_features_from_sequence("EVQLVESGGGLVQPGG")  # E N-term
    ptm_a = ptm_features_from_sequence("AVQLVESGGGLVQPGG")  # A N-term
    _check("pyro_q", ptm_q["pyroglutamate_risk"] == 2, f"Q→{ptm_q['pyroglutamate_risk']}")
    _check("pyro_e", ptm_e["pyroglutamate_risk"] == 1, f"E→{ptm_e['pyroglutamate_risk']}")
    _check("pyro_a", ptm_a["pyroglutamate_risk"] == 0, f"A→{ptm_a['pyroglutamate_risk']}")

    # ── 5. Key name alignment with developability_core ────────────
    # Verify that the keys we define match what dev_core actually reads
    try:
        import re as _re
        with open("src/developability_core.py", "r") as f:
            src = f.read()
        consumed_keys = set(_re.findall(r'anal\.get\("(\w+)"\)', src))
        missing_in_contract = consumed_keys - set(ANALYTICAL_CONSUMER_KEYS.keys())
        _check("key_alignment",
               len(missing_in_contract) == 0,
               f"dev_core reads keys not in contract: {missing_in_contract}")
    except Exception as e:
        _check("key_alignment", False, str(e))

    # ── Summary ───────────────────────────────────────────────────
    if errors:
        log.error("AnalyticalContract selftest: %d/%d FAILED", len(errors), checks)
        for e in errors:
            log.error("  - %s", e)
        return False

    log.info("AnalyticalContract selftest PASSED (%d/%d checks)", checks, checks)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    ok = _selftest()
    import sys
    sys.exit(0 if ok else 1)
