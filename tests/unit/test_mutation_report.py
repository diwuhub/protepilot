"""Tests for src/mutation/report.py."""

import pandas as pd
import pytest

from src.mutation.report import (
    save_report,
    scored_to_dataframe,
    summarize,
    top_n,
)
from src.mutation.schema import (
    AntibodyChain,
    MutationCandidate,
    ScoredMutation,
)


def _mk(llr, mutant="A", passes=True, region="cdr_h1"):
    """Build a ScoredMutation. `mutant` must be a standard AA (not S, the WT)."""
    c = MutationCandidate(
        chain=AntibodyChain.VH, position=27,
        wildtype_aa="S", mutant_aa=mutant,
        region=region,
    )
    return ScoredMutation(
        candidate=c, llr=llr, wildtype_prob=0.1, mutant_prob=0.3,
        passes_guardrails=passes,
    )


class TestScoredToDataFrame:
    def test_empty(self):
        df = scored_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_deterministic_order(self):
        a = _mk(llr=0.5, mutant="A")
        b = _mk(llr=2.0, mutant="G")
        c = _mk(llr=0.5, mutant="C")
        d1 = scored_to_dataframe([a, b, c])
        d2 = scored_to_dataframe([c, b, a])
        # Deterministic by composite_score desc, then mutation_label asc.
        # At llr=2.0, only one candidate (G). At llr=0.5, "A" before "C" by label.
        assert list(d1["mutation_label"]) == ["VH:S27G", "VH:S27A", "VH:S27C"]
        assert list(d1["mutation_label"]) == list(d2["mutation_label"])

    def test_failing_guardrail_sinks_to_bottom(self):
        a = _mk(llr=5.0, mutant="A", passes=False)
        b = _mk(llr=0.1, mutant="G", passes=True)
        df = scored_to_dataframe([a, b])
        assert df.iloc[0]["mutation_label"] == "VH:S27G"  # lower LLR but passes
        assert df.iloc[1]["mutation_label"] == "VH:S27A"


class TestTopN:
    def test_n_limit(self):
        items = [_mk(llr=i * 0.1, mutant=aa) for i, aa in enumerate("ACDEFGHI")]
        df = top_n(items, n=3)
        assert len(df) == 3

    def test_passing_only(self):
        items = [
            _mk(llr=1.0, mutant="A", passes=True),
            _mk(llr=2.0, mutant="G", passes=False),
            _mk(llr=0.5, mutant="C", passes=True),
        ]
        df = top_n(items, n=10, passing_only=True)
        assert len(df) == 2
        assert set(df["mutation_label"]) == {"VH:S27A", "VH:S27C"}

    def test_cdr_only(self):
        items = [
            _mk(llr=1.0, mutant="A", region="cdr_h1"),
            _mk(llr=2.0, mutant="G", region="framework"),
        ]
        df = top_n(items, n=10, cdr_only=True)
        assert len(df) == 1
        assert df.iloc[0]["region"] == "cdr_h1"


class TestSaveReport:
    def test_save(self, tmp_path):
        items = [_mk(llr=i * 0.1, mutant=aa) for i, aa in enumerate("ACDE")]
        out = save_report(items, tmp_path / "report.csv")
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 4

    def test_deterministic_content(self, tmp_path):
        items_a = [_mk(llr=1.0, mutant="A"),
                   _mk(llr=2.0, mutant="G")]
        items_b = list(reversed(items_a))
        p1 = save_report(items_a, tmp_path / "r1.csv")
        p2 = save_report(items_b, tmp_path / "r2.csv")
        assert p1.read_text() == p2.read_text()


class TestSummarize:
    def test_empty(self):
        s = summarize([])
        assert s == {"n_candidates": 0}

    def test_counts(self):
        items = [
            _mk(llr=1.0, mutant="A", passes=True, region="cdr_h1"),
            _mk(llr=-1.0, mutant="G", passes=True, region="cdr_h1"),
            _mk(llr=2.0, mutant="C", passes=False, region="framework"),
        ]
        s = summarize(items)
        assert s["n_candidates"] == 3
        assert s["n_passing_guardrails"] == 2
        assert s["n_cdr"] == 2
        assert s["llr_max_passing"] == pytest.approx(1.0)
        assert s["llr_min_passing"] == pytest.approx(-1.0)
