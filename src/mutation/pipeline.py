"""End-to-end mutation-predictor pipeline.

Single entry point that wraps:
    enumerate → score (masked marginal) → guardrails → rank

Usage:
    from src.mutation.pipeline import rank_mutations
    scored = rank_mutations(vh, vl, cdr_only=True)
    report_df = top_n(scored, n=20)
"""

from __future__ import annotations

from src.mutation.enumerate import enumerate_single_mutations
from src.mutation.guardrails import GuardrailThresholds, apply_guardrails
from src.mutation.masked_marginal import MaskedMarginalConfig, MaskedMarginalScorer
from src.mutation.schema import AntibodyChain, ScoredMutation


def rank_mutations(
    vh: str,
    vl: str = "",
    *,
    cdr_only: bool = False,
    restrict_to_positions: dict[AntibodyChain, list[int]] | None = None,
    scorer: MaskedMarginalScorer | None = None,
    scorer_config: MaskedMarginalConfig | None = None,
    thresholds: GuardrailThresholds | None = None,
    apply_developability_guardrails: bool = True,
) -> list[ScoredMutation]:
    """Run the full Phase 2 mutation-predictor pipeline.

    Parameters
    ----------
    vh, vl : parent sequences
    cdr_only : restrict to CDR positions (recommended for affinity matur.)
    restrict_to_positions : per-chain position allow-list (overrides cdr_only)
    scorer : pre-instantiated MaskedMarginalScorer (amortizes weight load)
    scorer_config : used if scorer is None
    thresholds : custom guardrail thresholds
    apply_developability_guardrails : skip guardrails if False

    Returns
    -------
    list[ScoredMutation] — unsorted; use src.mutation.report.top_n to sort.
    """
    candidates = enumerate_single_mutations(
        vh=vh, vl=vl,
        exclude_framework=cdr_only,
        restrict_to_positions=restrict_to_positions,
    )
    scorer = scorer or MaskedMarginalScorer(config=scorer_config)
    scored = scorer.score(vh=vh, vl=vl, candidates=candidates)
    if apply_developability_guardrails:
        scored = apply_guardrails(vh=vh, vl=vl, scored=scored, thresholds=thresholds)
    return scored
