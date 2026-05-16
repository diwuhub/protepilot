"""Tests for src/mutation/schema.py."""

import pytest

from src.mutation.schema import (
    CDR_REGIONS,
    FRAMEWORK_REGIONS,
    STANDARD_AA,
    AntibodyChain,
    MutationCandidate,
    ScoredMutation,
)


class TestMutationCandidate:
    def test_valid_vh_mutation(self):
        c = MutationCandidate(
            chain=AntibodyChain.VH, position=27, wildtype_aa="S", mutant_aa="A",
            region="cdr_h1",
        )
        assert c.mutation_label == "VH:S27A"
        assert c.is_in_cdr()

    def test_valid_framework_mutation(self):
        c = MutationCandidate(
            chain=AntibodyChain.VL, position=40, wildtype_aa="T", mutant_aa="V",
            region="framework",
        )
        assert not c.is_in_cdr()
        assert c.mutation_label == "VL:T40V"

    def test_rejects_non_standard_wildtype(self):
        with pytest.raises(ValueError, match="wildtype_aa"):
            MutationCandidate(
                chain=AntibodyChain.VH, position=1, wildtype_aa="X", mutant_aa="A",
                region="framework",
            )

    def test_rejects_non_standard_mutant(self):
        with pytest.raises(ValueError, match="mutant_aa"):
            MutationCandidate(
                chain=AntibodyChain.VH, position=1, wildtype_aa="S", mutant_aa="Z",
                region="framework",
            )

    def test_rejects_identity_mutation(self):
        with pytest.raises(ValueError, match="wildtype_aa == mutant_aa"):
            MutationCandidate(
                chain=AntibodyChain.VH, position=1, wildtype_aa="S", mutant_aa="S",
                region="framework",
            )

    def test_rejects_negative_position(self):
        with pytest.raises(ValueError, match="position"):
            MutationCandidate(
                chain=AntibodyChain.VH, position=-1, wildtype_aa="S", mutant_aa="A",
                region="framework",
            )

    def test_rejects_unknown_region(self):
        with pytest.raises(ValueError, match="region"):
            MutationCandidate(
                chain=AntibodyChain.VH, position=1, wildtype_aa="S", mutant_aa="A",
                region="fictional_region",
            )


class TestScoredMutation:
    def _candidate(self):
        return MutationCandidate(
            chain=AntibodyChain.VH, position=27, wildtype_aa="S", mutant_aa="A",
            region="cdr_h1",
        )

    def test_composite_score_passing_equals_llr(self):
        s = ScoredMutation(
            candidate=self._candidate(),
            llr=1.5, wildtype_prob=0.1, mutant_prob=0.45,
            passes_guardrails=True,
        )
        assert s.composite_score == 1.5

    def test_composite_score_failing_is_sunk(self):
        s = ScoredMutation(
            candidate=self._candidate(),
            llr=2.5, wildtype_prob=0.02, mutant_prob=0.3,
            passes_guardrails=False,
        )
        assert s.composite_score == -1e9

    def test_to_row_shape(self):
        s = ScoredMutation(
            candidate=self._candidate(),
            llr=0.8, wildtype_prob=0.2, mutant_prob=0.45,
        )
        row = s.to_row()
        expected_keys = {
            "chain", "position", "wildtype_aa", "mutant_aa",
            "region", "mutation_label", "llr", "wildtype_prob",
            "mutant_prob", "tap_risk_flag_delta", "di_seq_proxy_delta",
            "camsol_intrinsic_mean_delta", "regresses_tap", "regresses_di",
            "regresses_camsol", "passes_guardrails", "composite_score",
            "uncertainty", "rationale", "numbering_label",
        }
        assert expected_keys.issubset(set(row))


class TestEnums:
    def test_cdr_regions_cover_six_cdrs(self):
        assert len(CDR_REGIONS) == 6
        assert "cdr_h3" in CDR_REGIONS

    def test_framework_region_is_single(self):
        assert FRAMEWORK_REGIONS == frozenset({"framework"})

    def test_standard_aa_length(self):
        assert len(STANDARD_AA) == 20
        assert set(STANDARD_AA) == set("ACDEFGHIKLMNPQRSTVWY")
