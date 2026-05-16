"""
classification_contract.py  ·  ProtePilot — Classifier Contract
===================================================================
Formal specification of the molecule classifier's input/output schema,
fusion strategy, and behavioral guarantees.

This contract exists so that:
  1. Downstream modules can validate classifier output without
     importing the classifier itself (decoupled dependency).
  2. CI/selftest can verify that the classifier still meets its
     behavioral promises after any code change.
  3. New engineers understand the 3-phase fusion strategy without
     reading 1000+ lines of implementation.

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

log = logging.getLogger("ProtePilot.ClassificationContract")


# ═══════════════════════════════════════════════════════════════════════
#  Contract Constants
# ═══════════════════════════════════════════════════════════════════════

# All valid molecule classes that the classifier may return.
VALID_CLASSES: Set[str] = {
    "canonical_mab", "bispecific", "fc_fusion", "adc",
    "single_domain", "peptide", "fusion_protein",
    "engineered_scaffold", "unknown",
}

# Confidence tiers (High > Medium > Low).
VALID_CONFIDENCE_TIERS: Set[str] = {"High", "Medium", "Low"}

# Minimum fields that ClassificationResult.to_dict() MUST contain.
REQUIRED_OUTPUT_FIELDS: Set[str] = {
    "molecule_class", "display_name", "confidence", "confidence_score",
    "evidence", "warnings", "n_chains", "n_unique_chains",
    "chain_types", "chain_lengths", "user_override",
    "has_fc_region", "expects_glycosylation",
}

# Classes that carry Fc-region by default.
FC_BEARING_CLASSES: Set[str] = {"canonical_mab", "bispecific", "fc_fusion", "adc"}

# Risk weight dimensions that MUST exist for every class.
REQUIRED_RISK_DIMENSIONS: Set[str] = {
    "aggregation", "stability", "viscosity", "expression", "immunogenicity",
}

# Risk weights must sum to within this tolerance of 1.0.
RISK_WEIGHT_SUM_TOLERANCE: float = 0.02


# ═══════════════════════════════════════════════════════════════════════
#  Fusion Strategy Specification
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FusionStrategy:
    """
    Documents the 3-phase classification fusion strategy.

    Phase 1 — Rule-Based (authoritative):
        Sequence length, chain count, motif detection, and name hints
        determine the class.  This is the ground truth — Phases 2 and 3
        never override it.

    Phase 2 — Trained Model Second Opinion:
        XGBoost (24 features, 8 classes) adjusts confidence but NEVER
        changes the class label.
        - High-confidence rule → skip model entirely
        - Both agree → boost confidence
        - Disagree → flag warning, keep rule-based

    Phase 3 — OOD Detection:
        Ensemble (global Mahalanobis + IsolationForest + per-class
        Mahalanobis) with majority vote.  If OOD detected:
        - Cap confidence_score to ≤0.30
        - Add OOD warning
        - Do NOT change class label

    Invariants:
        1. Rule-based class is NEVER overridden by Phase 2 or 3.
        2. user_hint (if valid) takes absolute priority over all phases.
        3. When user_hint overrides, feedback is recorded (fire-and-forget).
        4. OOD skips peptide and single_domain (they are inherently
           outside the mAb training distribution by design).
    """
    phases: List[str] = field(default_factory=lambda: [
        "rule_based",          # Phase 1
        "trained_model",       # Phase 2
        "ood_detection",       # Phase 3
    ])
    rule_based_is_authoritative: bool = True
    trained_model_can_override_class: bool = False
    ood_can_override_class: bool = False
    ood_skip_classes: Set[str] = field(default_factory=lambda: {"peptide", "single_domain"})
    user_hint_priority: str = "absolute"
    feedback_on_override: bool = True


# ═══════════════════════════════════════════════════════════════════════
#  Contract Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_output(result_dict: Dict[str, Any]) -> List[str]:
    """
    Validate a ClassificationResult.to_dict() against the contract.

    Returns a list of violation messages (empty = all OK).
    """
    violations = []

    # 1. Required fields
    missing = REQUIRED_OUTPUT_FIELDS - set(result_dict.keys())
    if missing:
        violations.append(f"Missing required fields: {sorted(missing)}")

    # 2. Class value
    cls_val = result_dict.get("molecule_class", "")
    if cls_val not in VALID_CLASSES:
        violations.append(f"Invalid molecule_class: '{cls_val}'")

    # 3. Confidence tier
    conf = result_dict.get("confidence", "")
    if conf not in VALID_CONFIDENCE_TIERS:
        violations.append(f"Invalid confidence tier: '{conf}'")

    # 4. Confidence score range
    score = result_dict.get("confidence_score", -1)
    if not (0.0 <= score <= 1.0):
        violations.append(f"confidence_score {score} outside [0, 1]")

    # 5. Fc consistency
    if cls_val in FC_BEARING_CLASSES and result_dict.get("has_fc_region") is False:
        violations.append(f"Class '{cls_val}' should have has_fc_region=True")

    # 6. Evidence is non-empty
    if not result_dict.get("evidence"):
        violations.append("Evidence list is empty — classifier must explain its decision")

    return violations


def validate_risk_weights(weights: Dict[str, float], molecule_class: str) -> List[str]:
    """
    Validate risk weight profile against the contract.
    """
    violations = []

    missing = REQUIRED_RISK_DIMENSIONS - set(weights.keys())
    if missing:
        violations.append(f"Missing risk dimensions for {molecule_class}: {sorted(missing)}")

    total = sum(weights.values())
    if abs(total - 1.0) > RISK_WEIGHT_SUM_TOLERANCE:
        violations.append(
            f"Risk weights for {molecule_class} sum to {total:.4f} "
            f"(expected 1.0 ± {RISK_WEIGHT_SUM_TOLERANCE})"
        )

    for dim, val in weights.items():
        if not (0.0 <= val <= 1.0):
            violations.append(f"Weight {dim}={val} outside [0, 1] for {molecule_class}")

    return violations


# ═══════════════════════════════════════════════════════════════════════
#  Behavioral Guarantees (for selftest use)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BehavioralGuarantee:
    """A single testable classifier guarantee."""
    name: str
    description: str
    test_fn: str  # Name of the function in selftest that verifies this


BEHAVIORAL_GUARANTEES = [
    BehavioralGuarantee(
        name="peptide_under_80aa",
        description="Any single-chain sequence <80 aa → peptide with High confidence",
        test_fn="test_peptide_threshold",
    ),
    BehavioralGuarantee(
        name="bispecific_two_distinct_hc",
        description="Two HC chains with <85% identity → bispecific with High confidence",
        test_fn="test_bispecific_detection",
    ),
    BehavioralGuarantee(
        name="canonical_mab_hc_lc_motifs",
        description="HC + LC chains with ≥3 antibody motifs → canonical_mab",
        test_fn="test_canonical_mab_detection",
    ),
    BehavioralGuarantee(
        name="fc_no_cl_means_fusion",
        description="Fc motifs present but no CL domain → fc_fusion",
        test_fn="test_fc_fusion_detection",
    ),
    BehavioralGuarantee(
        name="user_hint_absolute",
        description="Valid user_hint always takes priority; feedback recorded on disagreement",
        test_fn="test_user_override",
    ),
    BehavioralGuarantee(
        name="ood_never_overrides_class",
        description="OOD detection caps confidence but never changes molecule_class",
        test_fn="test_ood_confidence_cap",
    ),
    BehavioralGuarantee(
        name="risk_weights_complete",
        description="Every MoleculeClass has risk weights summing to ~1.0",
        test_fn="test_risk_weights",
    ),
    BehavioralGuarantee(
        name="output_schema_valid",
        description="ClassificationResult.to_dict() always passes validate_output()",
        test_fn="test_output_schema",
    ),
    BehavioralGuarantee(
        name="empty_input_returns_unknown",
        description="Empty or None sequence returns unknown with Low confidence",
        test_fn="test_empty_input",
    ),
    BehavioralGuarantee(
        name="validation_corpus_accuracy",
        description="Built-in validation corpus achieves ≥85% accuracy",
        test_fn="test_validation_corpus",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
#  Self-Test
# ═══════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    """Verify contract constants are internally consistent."""
    # All FC_BEARING_CLASSES are valid
    assert FC_BEARING_CLASSES.issubset(VALID_CLASSES), \
        f"FC_BEARING_CLASSES has invalid entries: {FC_BEARING_CLASSES - VALID_CLASSES}"

    # FusionStrategy defaults are consistent
    fs = FusionStrategy()
    assert fs.rule_based_is_authoritative
    assert not fs.trained_model_can_override_class
    assert not fs.ood_can_override_class
    assert fs.ood_skip_classes.issubset(VALID_CLASSES)

    # Every guarantee has a name and test_fn
    for g in BEHAVIORAL_GUARANTEES:
        assert g.name and g.test_fn, f"Incomplete guarantee: {g}"

    log.info("ClassificationContract selftest PASSED (%d guarantees)", len(BEHAVIORAL_GUARANTEES))
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
    print(f"Contract OK: {len(BEHAVIORAL_GUARANTEES)} behavioral guarantees defined.")
    print(f"Valid classes: {sorted(VALID_CLASSES)}")
    print(f"Fusion strategy: {FusionStrategy().phases}")
