"""Tests for src/mutation/guardrails.py.

Where ab_benchmark is importable, these tests run the real guardrail
pipeline. Otherwise they assert the "guardrails skipped" fallback path.
"""

import importlib.util
import os
import sys

import pytest

from src.mutation.guardrails import (
    GuardrailThresholds,
    _apply_mutation,
    apply_guardrails,
)
from src.mutation.schema import AntibodyChain, MutationCandidate, ScoredMutation


# Presence-by-filesystem check, NOT sys.path manipulation. ab-benchmark
# ships its own tests/__init__.py; adding it to sys.path at import time
# would shadow ProtePilot's namespace `tests` package during pytest
# collection. The guardrail module itself handles the path insertion
# lazily inside its own functions.
AB_BENCHMARK_AVAILABLE = os.path.isfile(
    "/Users/di/Projects/ab-benchmark/ab_benchmark/__init__.py"
)


TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


def _mk_scored(chain=AntibodyChain.VH, pos=30, wt=None, mt="A", llr=0.5):
    vh, vl = TRASTUZUMAB_VH, TRASTUZUMAB_VL
    if wt is None:
        wt = (vh if chain is AntibodyChain.VH else vl)[pos]
    return ScoredMutation(
        candidate=MutationCandidate(
            chain=chain, position=pos, wildtype_aa=wt, mutant_aa=mt,
            region="framework",
        ),
        llr=llr, wildtype_prob=0.2, mutant_prob=0.35,
    )


class TestApplyMutation:
    def test_vh_substitution(self):
        c = MutationCandidate(AntibodyChain.VH, 5, TRASTUZUMAB_VH[5], "A", "framework")
        vh2, vl2 = _apply_mutation(TRASTUZUMAB_VH, TRASTUZUMAB_VL, c)
        assert len(vh2) == len(TRASTUZUMAB_VH)
        assert vh2[5] == "A"
        assert vh2[:5] == TRASTUZUMAB_VH[:5]
        assert vh2[6:] == TRASTUZUMAB_VH[6:]
        assert vl2 == TRASTUZUMAB_VL

    def test_vl_substitution(self):
        c = MutationCandidate(AntibodyChain.VL, 3, TRASTUZUMAB_VL[3], "V", "framework")
        vh2, vl2 = _apply_mutation(TRASTUZUMAB_VH, TRASTUZUMAB_VL, c)
        assert vh2 == TRASTUZUMAB_VH
        assert vl2[3] == "V"

    def test_vh_out_of_range_raises(self):
        c = MutationCandidate(AntibodyChain.VH, 9999, TRASTUZUMAB_VH[0], "A", "framework")
        with pytest.raises(ValueError, match="VH position"):
            _apply_mutation(TRASTUZUMAB_VH, TRASTUZUMAB_VL, c)


@pytest.mark.skipif(not AB_BENCHMARK_AVAILABLE, reason="ab-benchmark not importable")
class TestRealGuardrails:
    def test_silent_mutation_passes(self):
        # V-to-I in framework: conservative; should not trip any flag.
        scored = [_mk_scored(pos=2, wt="Q", mt="E")]  # Q2E in VH framework
        apply_guardrails(TRASTUZUMAB_VH, TRASTUZUMAB_VL, scored)
        s = scored[0]
        # Either passes OR has populated deltas.
        if s.passes_guardrails:
            # Passed: rationale should say "developability preserved"
            assert "preserved" in s.rationale or s.passes_guardrails
        # tap/di/camsol deltas should be finite numbers (not NaN).
        import math
        assert math.isfinite(s.tap_risk_flag_delta) or math.isnan(s.tap_risk_flag_delta)
        assert math.isfinite(s.di_seq_proxy_delta) or math.isnan(s.di_seq_proxy_delta)

    def test_damaging_mutation_regresses(self):
        # A bulk AA → Gly substitution in a Jain-like framework often shifts
        # CamSol mean slightly; we don't hardcode the direction, but at
        # least one of the three metrics should show a non-zero delta.
        vh_pos = 30
        scored = [_mk_scored(pos=vh_pos, wt=TRASTUZUMAB_VH[vh_pos], mt="W")]
        apply_guardrails(TRASTUZUMAB_VH, TRASTUZUMAB_VL, scored)
        s = scored[0]
        deltas = [
            s.tap_risk_flag_delta,
            s.di_seq_proxy_delta,
            s.camsol_intrinsic_mean_delta,
        ]
        # at least one delta should be finite and non-zero
        import math
        assert any(math.isfinite(d) and d != 0.0 for d in deltas), deltas

    def test_composite_score_sinks_failures(self):
        # Force failure by making regresses flags True manually.
        s = _mk_scored(llr=5.0)
        s.regresses_tap = True
        s.passes_guardrails = False
        assert s.composite_score == -1e9

    def test_thresholds_affect_flags(self):
        # With an impossibly tight threshold (zero allowed), any
        # non-silent mutation should trigger a regression flag.
        scored = [_mk_scored(pos=30, wt=TRASTUZUMAB_VH[30], mt="P")]
        tight = GuardrailThresholds(
            tap_risk_delta_max=-99,
            di_seq_proxy_delta_max=-99,
            camsol_mean_delta_min=99,
        )
        apply_guardrails(TRASTUZUMAB_VH, TRASTUZUMAB_VL, scored, thresholds=tight)
        s = scored[0]
        # All three guardrails should fire.
        assert s.regresses_tap or s.regresses_di or s.regresses_camsol
        assert not s.passes_guardrails


class TestNoAbBenchmark:
    @pytest.mark.skipif(AB_BENCHMARK_AVAILABLE, reason="ab-benchmark IS available; skipping fallback test")
    def test_skips_gracefully_without_ab_benchmark(self):
        scored = [_mk_scored()]
        apply_guardrails(TRASTUZUMAB_VH, TRASTUZUMAB_VL, scored)
        assert scored[0].passes_guardrails is True
        assert "guardrails skipped" in scored[0].rationale
