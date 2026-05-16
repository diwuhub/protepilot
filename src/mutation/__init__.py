"""Mutation predictor for antibody developability screening.

Pipeline:
    sequence (VH/VL) → enumerate single-AA candidates →
    score with ESM-2 t12 masked marginals (Meier 2021) →
    apply ab-benchmark developability guardrails (TAP/DI/CamSol) →
    rank, emit top-N table with rationale + uncertainty

The masked-marginal score is a **ranking heuristic**, not a ΔΔG prediction.
No claim of binding-affinity improvement. No wet-lab validation implied.
"""

from src.mutation.schema import (
    CDR_REGIONS,
    FRAMEWORK_REGIONS,
    AntibodyChain,
    MutationCandidate,
    ScoredMutation,
)

__all__ = [
    "AntibodyChain",
    "CDR_REGIONS",
    "FRAMEWORK_REGIONS",
    "MutationCandidate",
    "ScoredMutation",
]
