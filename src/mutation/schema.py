"""Mutation-predictor schema.

A `MutationCandidate` describes ONE single-residue substitution on an
antibody: which chain, which position (0-indexed within the chain),
what the wild-type and mutant residues are, and which structural region
(framework vs a named CDR) the position falls in.

A `ScoredMutation` augments a candidate with:
    - the raw masked-marginal ranking score (log-likelihood ratio)
    - guardrail flags: developability regressions vs the parent
    - a rationale string
    - `uncertainty`: the confidence category ("high" / "medium" / "low")
      derived from the magnitude of the score and guardrail status.

Design notes:
    - 0-indexed positions are enforced so numpy slicing lines up with
      the underlying sequence. The Kabat/IMGT number is held as a
      separate string annotation, never as the primary key.
    - The region enum uses lowercased strings ("fr1", "cdr_h1", …) so it
      serializes cleanly to CSV/JSON and aligns with the regex-based
      CDR extractor already in ab_benchmark.seqprops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AntibodyChain(str, Enum):
    VH = "vh"
    VL = "vl"


# Region labels cover both conventional numbering schemes at the granularity
# we need. For Phase 2 with regex-based CDR extraction, only the CDR labels
# are ever set with certainty; framework residues are bucketed as 'framework'.
CDR_REGIONS = frozenset(
    {"cdr_h1", "cdr_h2", "cdr_h3", "cdr_l1", "cdr_l2", "cdr_l3"}
)
FRAMEWORK_REGIONS = frozenset({"framework"})

# All 20 standard amino acids, canonical order (matches BLOSUM indexing).
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class MutationCandidate:
    """One single-residue substitution on a parent antibody."""

    chain: AntibodyChain
    position: int                  # 0-indexed within the chain
    wildtype_aa: str               # single letter, STANDARD_AA
    mutant_aa: str                 # single letter, STANDARD_AA
    region: str                    # "cdr_h1" | ... | "cdr_l3" | "framework"
    numbering_label: str = ""      # optional Kabat/IMGT string, e.g. "H:27"

    def __post_init__(self) -> None:
        if self.wildtype_aa not in STANDARD_AA:
            raise ValueError(f"wildtype_aa {self.wildtype_aa!r} not a standard AA")
        if self.mutant_aa not in STANDARD_AA:
            raise ValueError(f"mutant_aa {self.mutant_aa!r} not a standard AA")
        if self.wildtype_aa == self.mutant_aa:
            raise ValueError("wildtype_aa == mutant_aa (not a mutation)")
        if self.position < 0:
            raise ValueError(f"position must be non-negative, got {self.position}")
        if self.region not in CDR_REGIONS | FRAMEWORK_REGIONS:
            raise ValueError(f"region {self.region!r} not recognized")

    @property
    def mutation_label(self) -> str:
        """Short string like 'VH:S27A' — suitable for tables and logs."""
        prefix = self.chain.value.upper()
        return f"{prefix}:{self.wildtype_aa}{self.position}{self.mutant_aa}"

    def is_in_cdr(self) -> bool:
        return self.region in CDR_REGIONS


@dataclass
class ScoredMutation:
    """A candidate after scoring + guardrail evaluation."""

    candidate: MutationCandidate

    # Meier 2021 masked-marginal log-likelihood ratio
    #   LLR = log P(mutant | context) - log P(wildtype | context)
    # Positive => mutation is favored by the PLM at this position.
    llr: float

    # Per-chain masked probabilities for ALL 20 AAs at this position,
    # retained so downstream analyses (heatmaps) don't need a re-run.
    wildtype_prob: float
    mutant_prob: float

    # Developability deltas (mutant minus parent); NaN if not computed.
    tap_risk_flag_delta: float = float("nan")
    di_seq_proxy_delta: float = float("nan")
    camsol_intrinsic_mean_delta: float = float("nan")

    # Boolean guardrail flags — True means "this mutation would regress
    # the named property vs the parent beyond the configured threshold".
    regresses_tap: bool = False
    regresses_di: bool = False
    regresses_camsol: bool = False

    # Final verdict after combining all guardrails.
    passes_guardrails: bool = True
    rationale: str = ""

    # Confidence classification: "high" | "medium" | "low".
    uncertainty: str = "medium"

    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        """Single scalar used to sort candidates.

        This is the masked-marginal LLR with a hard zero for candidates
        that fail guardrails. Sorting descending by composite_score puts
        PLM-favored mutations that pass developability at the top.
        """
        if not self.passes_guardrails:
            return -1e9  # sink failed guardrails to the bottom
        return self.llr

    def to_row(self) -> dict[str, Any]:
        """Flat row for CSV/DataFrame export."""
        c = self.candidate
        return {
            "chain": c.chain.value,
            "position": c.position,
            "wildtype_aa": c.wildtype_aa,
            "mutant_aa": c.mutant_aa,
            "region": c.region,
            "numbering_label": c.numbering_label,
            "mutation_label": c.mutation_label,
            "llr": self.llr,
            "wildtype_prob": self.wildtype_prob,
            "mutant_prob": self.mutant_prob,
            "tap_risk_flag_delta": self.tap_risk_flag_delta,
            "di_seq_proxy_delta": self.di_seq_proxy_delta,
            "camsol_intrinsic_mean_delta": self.camsol_intrinsic_mean_delta,
            "regresses_tap": self.regresses_tap,
            "regresses_di": self.regresses_di,
            "regresses_camsol": self.regresses_camsol,
            "passes_guardrails": self.passes_guardrails,
            "composite_score": self.composite_score,
            "uncertainty": self.uncertainty,
            "rationale": self.rationale,
        }
