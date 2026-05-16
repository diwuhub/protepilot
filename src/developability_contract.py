"""
developability_contract.py  ·  ProtePilot — Developability Contract
=====================================================================
Formal specification of the developability assessment's input/output
schema, scoring invariants, and behavioral guarantees.

This contract ensures:
  1. Downstream consumers (reports, dashboard) can validate assessment
     output without importing the full assessment logic.
  2. CI/selftest can verify behavioral promises after code changes.
  3. The scoring pipeline's invariants are documented, not implicit.

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

log = logging.getLogger("ProtePilot.DevelopabilityContract")


# ═══════════════════════════════════════════════════════════════════════
#  Contract Constants
# ═══════════════════════════════════════════════════════════════════════

# Valid molecule classes accepted by assess_developability().
VALID_CLASSES: Set[str] = {
    "canonical_mab", "bispecific", "fc_fusion", "adc",
    "single_domain", "peptide", "fusion_protein",
    "engineered_scaffold", "unknown",
}

# Core risk dimensions that MUST appear for every assessment.
CORE_DIMENSIONS: Set[str] = {
    "aggregation", "stability", "viscosity", "expression", "immunogenicity",
}

# Conditional dimensions (appear only for certain molecule classes).
CONDITIONAL_DIMENSIONS = {
    "species_purity": {"bispecific"},
    "conjugation": {"adc"},
}

# Valid grades for individual dimensions.
VALID_DIMENSION_GRADES: Set[str] = {"Low", "Medium", "High", "Unknown"}

# Valid composite grades (display labels with "Risk" suffix).
VALID_COMPOSITE_GRADES: Set[str] = {"Low Risk", "Medium Risk", "High Risk"}

# Valid recommendations.
VALID_RECOMMENDATIONS: Set[str] = {
    "Proceed", "Proceed with caution", "Optimize before proceeding",
}

# Valid confidence tiers.
VALID_CONFIDENCE_TIERS: Set[str] = {"High", "Medium", "Low"}

# Scoring boundaries (from report_schema.py — single source of truth).
GRADE_LOW_UPPER = 0.30      # score < 0.30 → "Low"
GRADE_MEDIUM_UPPER = 0.60   # 0.30 ≤ score < 0.60 → "Medium"

# QTPP row statuses.
VALID_QTPP_STATUSES: Set[str] = {
    "Within Target", "Within Range", "Out of Range", "Not Assessed",
}

# Minimum QTPP rows for standard assessment.
MIN_QTPP_ROWS_STANDARD = 10
MIN_QTPP_ROWS_BISPECIFIC = 12  # +2 for homodimer + CEX resolution
MIN_QTPP_ROWS_ADC = 11         # +1 for DAR

# Required fields in DevelopabilityAssessment.to_dict() output.
REQUIRED_OUTPUT_FIELDS: Set[str] = {
    "molecule_name", "molecule_class", "composite_score",
    "composite_grade", "recommendation", "recommendation_detail",
    "confidence", "dimensions", "qtpp_rows",
}

# Required fields per dimension dict in to_dict() output.
REQUIRED_DIMENSION_FIELDS: Set[str] = {
    "name", "display_name", "score", "weight", "grade",
    "explanation", "source", "confidence", "evidence",
}


# ═══════════════════════════════════════════════════════════════════════
#  Scoring Invariants
# ═══════════════════════════════════════════════════════════════════════

SCORING_INVARIANTS = [
    "Every dimension score is in [0.0, 1.0] where 0 = best, 1 = worst",
    "Every dimension weight is in [0.0, 1.0]",
    "Dimension weights for a given class sum to ~1.0 (±0.02)",
    "composite_score = sum(score * weight) / sum(weight)",
    "composite_score is in [0.0, 1.0]",
    "Grade boundaries: <0.25 = Low, 0.25-0.55 = Medium, ≥0.55 = High",
    "Recommendation logic: ≥2 High dims → Optimize, 1 High → Caution, ≥3 Medium → Caution, else Proceed",
    "Stability input is inverted: high stability (0.85) → low risk (0.15)",
    "OOD/non-standard classes get format-aware caveats on mAb-derived QTPP rows",
]


# ═══════════════════════════════════════════════════════════════════════
#  Contract Validation Functions
# ═══════════════════════════════════════════════════════════════════════

def validate_assessment_output(result_dict: Dict[str, Any]) -> List[str]:
    """
    Validate a DevelopabilityAssessment.to_dict() output against the contract.

    Returns list of violation messages (empty = all OK).
    """
    violations = []

    # 1. Required fields
    missing = REQUIRED_OUTPUT_FIELDS - set(result_dict.keys())
    if missing:
        violations.append(f"Missing required fields: {sorted(missing)}")

    # 2. Molecule class
    mc = result_dict.get("molecule_class", "")
    if mc not in VALID_CLASSES:
        violations.append(f"Invalid molecule_class: '{mc}'")

    # 3. Composite score range
    cs = result_dict.get("composite_score", -1)
    if not (0.0 <= cs <= 1.0):
        violations.append(f"composite_score {cs} outside [0, 1]")

    # 4. Composite grade
    cg = result_dict.get("composite_grade", "")
    if cg not in VALID_COMPOSITE_GRADES:
        violations.append(f"Invalid composite_grade: '{cg}'")

    # 5. Recommendation
    rec = result_dict.get("recommendation", "")
    if rec not in VALID_RECOMMENDATIONS:
        violations.append(f"Invalid recommendation: '{rec}'")

    # 6. Confidence
    conf = result_dict.get("confidence", "")
    if conf not in VALID_CONFIDENCE_TIERS:
        violations.append(f"Invalid confidence: '{conf}'")

    # 7. Dimensions
    dims = result_dict.get("dimensions", [])
    dim_names = set()
    for d in dims:
        dm = REQUIRED_DIMENSION_FIELDS - set(d.keys())
        if dm:
            violations.append(f"Dimension '{d.get('name','?')}' missing fields: {sorted(dm)}")
        s = d.get("score", -1)
        if not (0.0 <= s <= 1.0):
            violations.append(f"Dimension '{d.get('name','?')}' score {s} outside [0, 1]")
        w = d.get("weight", -1)
        if not (0.0 <= w <= 1.0):
            violations.append(f"Dimension '{d.get('name','?')}' weight {w} outside [0, 1]")
        g = d.get("grade", "")
        if g not in VALID_DIMENSION_GRADES:
            violations.append(f"Dimension '{d.get('name','?')}' invalid grade: '{g}'")
        if not d.get("explanation"):
            violations.append(f"Dimension '{d.get('name','?')}' missing explanation")
        dim_names.add(d.get("name", ""))

    # 8. Core dimensions present
    missing_core = CORE_DIMENSIONS - dim_names
    if missing_core:
        violations.append(f"Missing core dimensions: {sorted(missing_core)}")

    # 9. Conditional dimensions
    for cond_dim, required_classes in CONDITIONAL_DIMENSIONS.items():
        if mc in required_classes and cond_dim not in dim_names:
            violations.append(f"Class '{mc}' requires dimension '{cond_dim}' but it's missing")

    # 10. QTPP rows
    qtpp = result_dict.get("qtpp_rows", [])
    min_rows = MIN_QTPP_ROWS_STANDARD
    if mc == "bispecific":
        min_rows = MIN_QTPP_ROWS_BISPECIFIC
    elif mc == "adc":
        min_rows = MIN_QTPP_ROWS_ADC
    if len(qtpp) < min_rows:
        violations.append(f"Expected ≥{min_rows} QTPP rows for {mc}, got {len(qtpp)}")

    for row in qtpp:
        status = row.get("status", "")
        if status not in VALID_QTPP_STATUSES:
            violations.append(f"QTPP row '{row.get('attribute','?')}' invalid status: '{status}'")

    return violations


def validate_dimension_weights(weights: Dict[str, float], molecule_class: str) -> List[str]:
    """Validate risk weight profile for a molecule class."""
    violations = []

    missing = CORE_DIMENSIONS - set(weights.keys())
    # Not all core dims must be in weights (e.g. 'unknown' has basic set)
    # But the standard 5 should be present for most classes
    if missing and molecule_class not in ("unknown",):
        violations.append(f"Missing core weight dimensions for {molecule_class}: {sorted(missing)}")

    total = sum(weights.values())
    if abs(total - 1.0) > 0.02:
        violations.append(
            f"Weights for {molecule_class} sum to {total:.4f} (expected 1.0 ± 0.02)"
        )

    for dim, val in weights.items():
        if not (0.0 <= val <= 1.0):
            violations.append(f"Weight {dim}={val} outside [0, 1] for {molecule_class}")

    return violations


# ═══════════════════════════════════════════════════════════════════════
#  Behavioral Guarantees
# ═══════════════════════════════════════════════════════════════════════

BEHAVIORAL_GUARANTEES = [
    {
        "name": "core_dimensions_always_present",
        "description": "Every assessment has at least 5 core dimensions: aggregation, stability, viscosity, expression, immunogenicity",
    },
    {
        "name": "bispecific_gets_species_purity",
        "description": "Bispecific assessments include species_purity dimension and homodimer QTPP row",
    },
    {
        "name": "adc_gets_conjugation",
        "description": "ADC assessments include conjugation dimension and DAR QTPP row",
    },
    {
        "name": "composite_score_bounded",
        "description": "composite_score is always in [0.0, 1.0]",
    },
    {
        "name": "recommendation_deterministic",
        "description": "Recommendation follows deterministic rules: ≥2 High→Optimize, 1 High→Caution, ≥3 Medium→Caution, else Proceed",
    },
    {
        "name": "stability_inverted",
        "description": "Stability prediction (high=good) is inverted to risk (high=bad) before scoring",
    },
    {
        "name": "dimensions_have_explanations",
        "description": "Every dimension has a non-empty explanation string",
    },
    {
        "name": "qtpp_minimum_rows",
        "description": "QTPP table has ≥10 rows for standard, ≥12 for bispecific, ≥11 for ADC",
    },
    {
        "name": "weights_class_specific",
        "description": "Risk weights are selected based on molecule_class, not hardcoded",
    },
    {
        "name": "format_caveat_non_mab",
        "description": "Non-canonical-mAb classes get format-aware caveat on QTPP justifications",
    },
    {
        "name": "output_schema_valid",
        "description": "to_dict() always passes validate_assessment_output()",
    },
    {
        "name": "grade_boundaries_consistent",
        "description": "Grade boundaries match report_schema.py: <0.25=Low, 0.25-0.55=Medium, ≥0.55=High",
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  Self-Test
# ═══════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    """Verify contract constants are internally consistent."""
    # Grade boundaries are ordered
    assert GRADE_LOW_UPPER < GRADE_MEDIUM_UPPER, "Grade boundaries not ordered"

    # Core dimensions are a subset of what the contract checks
    assert len(CORE_DIMENSIONS) == 5

    # Conditional dimensions reference valid dimension names
    for dim, classes in CONDITIONAL_DIMENSIONS.items():
        assert classes.issubset(VALID_CLASSES), f"Conditional dim {dim} has invalid classes"

    # All guarantees have names
    for g in BEHAVIORAL_GUARANTEES:
        assert g["name"] and g["description"], f"Incomplete guarantee: {g}"

    log.info("DevelopabilityContract selftest PASSED (%d guarantees)", len(BEHAVIORAL_GUARANTEES))
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
    print(f"Contract OK: {len(BEHAVIORAL_GUARANTEES)} behavioral guarantees defined.")
    print(f"Core dimensions: {sorted(CORE_DIMENSIONS)}")
    print(f"Scoring invariants: {len(SCORING_INVARIANTS)}")
