"""
platform_config.py  ·  ProtePilot — Cross-Module Constants
==========================================================
Constants that are referenced by 2+ modules and MUST stay synchronized.

Module-specific constants (kinetics, pKa, buffer catalogs, etc.) remain
in their respective modules.  See config/platform_constants_reference.yaml
for the full audit inventory.

Design decision: Python module (not YAML) for type safety, IDE support,
and zero-dependency loading.  All values are frozen at import time.

Author  : Di (ProtePilot)
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# 1. Grade Boundaries
# ═══════════════════════════════════════════════════════════════
# Re-exported from report_schema.py (which remains the single source
# of truth).  Other modules should import from EITHER report_schema
# OR platform_config — both resolve to the same values.

from src.report_schema import GRADE_LOW_UPPER, GRADE_MEDIUM_UPPER  # noqa: F401

# ═══════════════════════════════════════════════════════════════
# 2. Confidence Thresholds
# ═══════════════════════════════════════════════════════════════
# Used by: model_inference.py, benchmark_evaluator.py, molecule_classifier.py

CONFIDENCE_HIGH: float = 0.80
CONFIDENCE_MEDIUM: float = 0.50

# ═══════════════════════════════════════════════════════════════
# 3. Model Validation Gates
# ═══════════════════════════════════════════════════════════════
# Used by: benchmark_evaluator.py, SelfTest/run_validation.py

MIN_TEST_ACCURACY: float = 0.50
MAX_BENCHMARK_DRIFT: int = 4            # max class changes before alert
MAX_ACCURACY_DEGRADATION: float = 0.05  # vs baseline (must be >= baseline - this)

# ═══════════════════════════════════════════════════════════════
# 4. Chain Detection Thresholds
# ═══════════════════════════════════════════════════════════════
# Used by: molecule_classifier.py, bulk_schema.py, data_harmonizer.py,
#          benchmark_evaluator.py

MIN_SEQUENCE_LENGTH: int = 10     # aa — below this, sequence is invalid
MIN_HC_LENGTH: int = 200          # aa — for bispecific heavy-chain detection
MIN_CHAIN_CLUSTER_LENGTH: int = 80  # aa — for unique-chain clustering
HC_IDENTITY_THRESHOLD: float = 0.85  # bispecific HC1 vs HC2 pairwise identity

# ═══════════════════════════════════════════════════════════════
# 5. Evidence Tiers (re-exported from report_schema)
# ═══════════════════════════════════════════════════════════════
from src.report_schema import EVIDENCE_TIER_1, EVIDENCE_TIER_2, EVIDENCE_TIER_3  # noqa: F401
