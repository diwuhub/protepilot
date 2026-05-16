"""Top-N mutation report generator.

Deterministic (stable sort, tie-break on mutation_label) so the same
inputs produce the same table every run — a Phase 2 self-assessment
requirement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mutation.schema import ScoredMutation


def scored_to_dataframe(scored: list[ScoredMutation]) -> pd.DataFrame:
    """Convert a list of ScoredMutation to a pandas DataFrame."""
    rows = [s.to_row() for s in scored]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Deterministic ordering: composite_score desc, then mutation_label asc
    # (lexicographic) so ties break predictably.
    df = df.sort_values(
        by=["composite_score", "mutation_label"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def top_n(
    scored: list[ScoredMutation],
    n: int = 20,
    passing_only: bool = False,
    cdr_only: bool = False,
) -> pd.DataFrame:
    """Return the top-N rows of the ranked DataFrame.

    Parameters
    ----------
    scored : list from the scoring + guardrail pipeline
    n : number of rows to keep (default 20)
    passing_only : if True, drop rows where passes_guardrails is False
    cdr_only : if True, keep only CDR-region mutations
    """
    df = scored_to_dataframe(scored)
    if df.empty:
        return df
    if passing_only:
        df = df[df["passes_guardrails"]]
    if cdr_only:
        df = df[df["region"].str.startswith("cdr_")]
    return df.head(n).reset_index(drop=True)


def save_report(
    scored: list[ScoredMutation],
    out_path: Path | str,
    top_n_rows: int = 50,
) -> Path:
    """Save a deterministic CSV report; return the written path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = top_n(scored, n=top_n_rows)
    df.to_csv(out_path, index=False)
    return out_path


def summarize(scored: list[ScoredMutation]) -> dict:
    """Compact per-run summary for logging / notebook display."""
    df = scored_to_dataframe(scored)
    if df.empty:
        return {"n_candidates": 0}

    return {
        "n_candidates": int(len(df)),
        "n_passing_guardrails": int(df["passes_guardrails"].sum()),
        "n_regresses_tap": int(df["regresses_tap"].sum()),
        "n_regresses_di": int(df["regresses_di"].sum()),
        "n_regresses_camsol": int(df["regresses_camsol"].sum()),
        "n_cdr": int(df["region"].str.startswith("cdr_").sum()),
        "llr_max_passing": float(
            df.loc[df["passes_guardrails"], "llr"].max()
        ) if df["passes_guardrails"].any() else float("nan"),
        "llr_min_passing": float(
            df.loc[df["passes_guardrails"], "llr"].min()
        ) if df["passes_guardrails"].any() else float("nan"),
    }
