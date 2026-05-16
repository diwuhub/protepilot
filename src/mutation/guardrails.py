"""Developability guardrails for mutation candidates.

Applies ab-benchmark's sequence-proxy TAP, Developability Index, and
CamSol-intrinsic baselines to (parent, mutant) sequences and flags any
mutation whose delta regresses developability beyond configured thresholds.

Design choices:
    - Guardrails operate on the *full VH+VL* of parent and mutant — they
      are NOT per-residue predictions. A single substitution perturbs
      global metrics by a small amount; the thresholds below are tuned
      so that only meaningful regressions trip the flag.
    - Guardrail failure sinks the mutation in ranking but does not drop
      it from the output. The report still shows failed candidates with
      a reason string, so users can audit why a PLM-favored mutation was
      down-ranked.
    - The baseline metric pulled from ab_benchmark.baselines.* is the
      ONE metric per baseline that the benchmark audit showed
      correlates with Jain-endpoint signals:
          TAP           → tap_risk_flag_count  (sum of amber-band flags)
          DI            → di_seq_proxy
          CamSol        → camsol_intrinsic_mean

Thresholds (defaults — reasonable for sequence-proxy metrics on Jain 137):
    TAP     Δ(risk_flag_count)       > +1         → regresses
    DI      Δ(di_seq_proxy)          > +0.3       → regresses
    CamSol  Δ(camsol_intrinsic_mean) < -0.03      → regresses

If the ab-benchmark package is not importable, this module returns
unperturbed ScoredMutations with `passes_guardrails=True` and a note in
the rationale. That lets the pipeline still emit a ranking in environments
without ab-benchmark (e.g. CI).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass

from src.mutation.schema import (
    AntibodyChain,
    MutationCandidate,
    ScoredMutation,
)


def _ensure_ab_benchmark_on_path() -> None:
    """Lazy sys.path shim — called only when we actually need ab_benchmark.

    Done inside a function, not at module import time, to avoid shadowing
    ProtePilot's own `tests` namespace package during pytest collection
    (ab-benchmark ships its own tests/__init__.py).
    """
    _AB_BENCHMARK = os.environ.get(
        "AB_BENCHMARK_PATH",
        str(Path(__file__).resolve().parents[3] / "ab-benchmark"),
    )
    if os.path.isdir(_AB_BENCHMARK) and _AB_BENCHMARK not in sys.path:
        sys.path.append(_AB_BENCHMARK)


@dataclass
class GuardrailThresholds:
    tap_risk_delta_max: float = 1.0         # +1 risk flag
    di_seq_proxy_delta_max: float = 0.3     # +0.3 units
    camsol_mean_delta_min: float = -0.03    # allow small drop only


def _apply_mutation(vh: str, vl: str, c: MutationCandidate) -> tuple[str, str]:
    """Return (vh', vl') after applying a single-residue substitution."""
    if c.chain is AntibodyChain.VH:
        if c.position >= len(vh):
            raise ValueError(f"VH position {c.position} out of range (len={len(vh)})")
        mut_vh = vh[: c.position] + c.mutant_aa + vh[c.position + 1 :]
        return mut_vh, vl
    if c.position >= len(vl):
        raise ValueError(f"VL position {c.position} out of range (len={len(vl)})")
    mut_vl = vl[: c.position] + c.mutant_aa + vl[c.position + 1 :]
    return vh, mut_vl


def _compute_parent_metrics(vh: str, vl: str) -> dict:
    """Return parent metrics for TAP, DI, CamSol. Empty dict if ab-benchmark missing."""
    _ensure_ab_benchmark_on_path()
    try:
        from ab_benchmark.baselines.camsol import compute_camsol_intrinsic
        from ab_benchmark.baselines.developability_index import compute_developability_index
        from ab_benchmark.baselines.tap import compute_tap
        from ab_benchmark.schema import AntibodyRecord, SourceDataset
    except ImportError:
        return {}

    record = AntibodyRecord(
        ab_id="_parent", source=SourceDataset.JAIN_2017, vh=vh, vl=vl
    )
    out: dict[str, dict] = {}
    tap_res = compute_tap(record)
    if tap_res.available:
        out["tap"] = tap_res.metrics
    di_res = compute_developability_index(record)
    if di_res.available:
        out["di"] = di_res.metrics
    camsol_res = compute_camsol_intrinsic(record)
    if camsol_res.available:
        out["camsol"] = camsol_res.metrics
    return out


def apply_guardrails(
    vh: str,
    vl: str,
    scored: list[ScoredMutation],
    thresholds: GuardrailThresholds | None = None,
) -> list[ScoredMutation]:
    """Augment each scored mutation with developability delta flags.

    Mutates each ScoredMutation in-place (setting tap/di/camsol deltas,
    boolean regression flags, `passes_guardrails`, rationale, and
    uncertainty) and returns the same list for convenience.
    """
    thresholds = thresholds or GuardrailThresholds()
    _ensure_ab_benchmark_on_path()

    try:
        from ab_benchmark.baselines.camsol import compute_camsol_intrinsic
        from ab_benchmark.baselines.developability_index import compute_developability_index
        from ab_benchmark.baselines.tap import compute_tap
        from ab_benchmark.schema import AntibodyRecord, SourceDataset
    except ImportError:
        for s in scored:
            s.rationale = f"{s.rationale} | guardrails skipped (ab-benchmark not importable)".strip(" |")
        return scored

    parent_metrics = _compute_parent_metrics(vh, vl)
    if not parent_metrics:
        for s in scored:
            s.rationale = f"{s.rationale} | guardrails skipped (parent metrics unavailable)".strip(" |")
        return scored

    for s in scored:
        mut_vh, mut_vl = _apply_mutation(vh, vl, s.candidate)
        mut_record = AntibodyRecord(
            ab_id="_mutant", source=SourceDataset.JAIN_2017, vh=mut_vh, vl=mut_vl
        )

        regressions: list[str] = []

        # TAP
        if "tap" in parent_metrics:
            tap_res = compute_tap(mut_record)
            if tap_res.available:
                delta = (
                    tap_res.metrics["tap_risk_flag_count"]
                    - parent_metrics["tap"]["tap_risk_flag_count"]
                )
                s.tap_risk_flag_delta = float(delta)
                if delta > thresholds.tap_risk_delta_max:
                    s.regresses_tap = True
                    regressions.append(f"TAP risk flags +{delta:.0f}")

        # DI
        if "di" in parent_metrics:
            di_res = compute_developability_index(mut_record)
            if di_res.available:
                delta = (
                    di_res.metrics["di_seq_proxy"]
                    - parent_metrics["di"]["di_seq_proxy"]
                )
                s.di_seq_proxy_delta = float(delta)
                if delta > thresholds.di_seq_proxy_delta_max:
                    s.regresses_di = True
                    regressions.append(f"DI Δ={delta:+.2f}")

        # CamSol
        if "camsol" in parent_metrics:
            camsol_res = compute_camsol_intrinsic(mut_record)
            if camsol_res.available:
                delta = (
                    camsol_res.metrics["camsol_intrinsic_mean"]
                    - parent_metrics["camsol"]["camsol_intrinsic_mean"]
                )
                s.camsol_intrinsic_mean_delta = float(delta)
                if delta < thresholds.camsol_mean_delta_min:
                    s.regresses_camsol = True
                    regressions.append(f"CamSol Δ={delta:+.3f}")

        s.passes_guardrails = not (s.regresses_tap or s.regresses_di or s.regresses_camsol)

        # Uncertainty refinement: passes guardrails ∧ strong signal → high.
        if s.passes_guardrails and abs(s.llr) >= 2.0:
            s.uncertainty = "high"
        elif not s.passes_guardrails:
            s.uncertainty = "low"

        if regressions:
            s.rationale = f"{s.rationale} | regressions: {', '.join(regressions)}"
        else:
            s.rationale = f"{s.rationale} | developability preserved"

    return scored
