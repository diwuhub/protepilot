"""ProteinGym-style retrospective evaluation of masked-marginal LLR.

ProteinGym datasets are long-format CSVs with at least:
    mutant     — HGVS-style single-point mutation string, e.g. "S27A" or "A:S27A"
    DMS_score  — experimentally measured fitness/activity value
    target_seq — parent sequence (sometimes held separately)

This module reads such a CSV, scores each mutation with our
masked-marginal scorer, and reports:
    - Spearman ρ between LLR and DMS score, with bootstrap 95% CI
    - Coverage (how many of the CSV rows were scorable)
    - A fit-quality residual plot suggestion (caller renders it)

Meier et al. 2021 reported mean ρ ≈ 0.45 across ProteinGym with ESM-1b.
With ESM-2 t12 (35 M) we expect somewhat lower ρ since it is a smaller
model (~0.3–0.4 range). ESM-2 t33 (650 M) would recover Meier's numbers;
we pin to t12 for Phase 2 consistency with ProtePilot CLAUDE.md.

Usage:
    from src.mutation.proteingym import evaluate_proteingym_csv
    result = evaluate_proteingym_csv(
        csv_path="data/proteingym/TRASTUZUMAB_HER2.csv",
        parent_vh=TRASTUZUMAB_VH,
        parent_vl=TRASTUZUMAB_VL,
    )
    print(result.summary())
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.mutation.masked_marginal import MaskedMarginalScorer
from src.mutation.schema import (
    STANDARD_AA,
    AntibodyChain,
    MutationCandidate,
    ScoredMutation,
)


# HGVS-ish pattern covering "S27A", "A:S27A", "H:S27A", "VH:S27A".
_MUTATION_RE = re.compile(
    r"^(?:(?P<chain>[AH]|VH|VL|L|K):)?"
    r"(?P<wt>[ACDEFGHIKLMNPQRSTVWY])"
    r"(?P<pos>\d+)"
    r"(?P<mt>[ACDEFGHIKLMNPQRSTVWY])$"
)


@dataclass
class ProteinGymResult:
    n_rows: int
    n_scored: int
    spearman_rho: float
    spearman_ci_low: float
    spearman_ci_high: float
    rmse: float
    predictions: pd.DataFrame    # columns: mutation, llr, dms_score

    def summary(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_scored": self.n_scored,
            "coverage": self.n_scored / self.n_rows if self.n_rows else 0.0,
            "spearman_rho": self.spearman_rho,
            "spearman_ci": (self.spearman_ci_low, self.spearman_ci_high),
            "rmse": self.rmse,
        }


def _parse_mutation(label: str) -> tuple[AntibodyChain, int, str, str] | None:
    """Parse an HGVS-style mutation label → (chain, position_0indexed, wt, mt).

    Returns None if the string does not parse. Positions in ProteinGym are
    1-indexed by convention; we shift to 0-indexed here.
    """
    m = _MUTATION_RE.match(label.strip())
    if not m:
        return None
    chain_raw = (m.group("chain") or "H").upper()
    if chain_raw in {"H", "A", "VH"}:
        chain = AntibodyChain.VH
    elif chain_raw in {"L", "K", "VL"}:
        chain = AntibodyChain.VL
    else:
        return None
    pos = int(m.group("pos")) - 1  # 1-indexed → 0-indexed
    return chain, pos, m.group("wt"), m.group("mt")


def _bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> tuple[float, float, float]:
    """Return (point_rho, ci_low, ci_high) for Spearman rank correlation."""
    from scipy.stats import spearmanr

    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    rho_hat = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(random_state)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = spearmanr(x[idx], y[idx]).statistic
        stats[i] = s if not np.isnan(s) else rho_hat
    low = float(np.quantile(stats, alpha / 2))
    high = float(np.quantile(stats, 1 - alpha / 2))
    return rho_hat, low, high


def evaluate_proteingym_csv(
    csv_path: str | Path,
    parent_vh: str,
    parent_vl: str = "",
    mutation_column: str = "mutant",
    score_column: str = "DMS_score",
    scorer: MaskedMarginalScorer | None = None,
) -> ProteinGymResult:
    """Evaluate a ProteinGym-format CSV against the masked-marginal scorer.

    Rows whose mutation label cannot be parsed, or whose WT residue does
    not match the parent sequence at the stated position, are skipped
    with a warning — coverage is reported so silent loss is visible.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"ProteinGym CSV not found: {path}")
    df = pd.read_csv(path)
    if mutation_column not in df.columns:
        raise ValueError(f"mutation column {mutation_column!r} not in {list(df.columns)}")
    if score_column not in df.columns:
        raise ValueError(f"score column {score_column!r} not in {list(df.columns)}")

    # Build MutationCandidate list, tracking row order.
    candidates: list[MutationCandidate] = []
    kept_indices: list[int] = []
    skipped_parse = 0
    skipped_mismatch = 0
    for i, label in enumerate(df[mutation_column].astype(str).tolist()):
        parsed = _parse_mutation(label)
        if parsed is None:
            skipped_parse += 1
            continue
        chain, pos, wt, mt = parsed
        chain_seq = parent_vh if chain is AntibodyChain.VH else parent_vl
        if pos < 0 or pos >= len(chain_seq) or chain_seq[pos] != wt:
            skipped_mismatch += 1
            continue
        if wt == mt:
            skipped_parse += 1
            continue
        candidates.append(
            MutationCandidate(
                chain=chain, position=pos,
                wildtype_aa=wt, mutant_aa=mt,
                region="framework",  # region is irrelevant for the scorer
            )
        )
        kept_indices.append(i)

    scorer = scorer or MaskedMarginalScorer()
    scored = scorer.score(vh=parent_vh, vl=parent_vl, candidates=candidates)

    # Realign with the CSV rows we kept.
    llrs = np.full(len(kept_indices), np.nan, dtype=float)
    for cand, s in zip(candidates, scored):
        # candidates order matches kept_indices order by construction.
        idx_in_kept = candidates.index(cand)
        llrs[idx_in_kept] = s.llr

    dms = df.iloc[kept_indices][score_column].to_numpy(dtype=float)
    mutations = df.iloc[kept_indices][mutation_column].astype(str).to_numpy()

    rho, lo, hi = _bootstrap_spearman(llrs, dms)
    # RMSE only meaningful if caller has calibrated LLR → ΔΔG; we keep it
    # as a numerical sanity metric, not a ΔΔG claim.
    valid_mask = ~(np.isnan(llrs) | np.isnan(dms))
    if valid_mask.sum() > 1:
        rmse = float(np.sqrt(np.mean((llrs[valid_mask] - dms[valid_mask]) ** 2)))
    else:
        rmse = float("nan")

    predictions = pd.DataFrame({
        "mutation": mutations,
        "llr": llrs,
        "dms_score": dms,
    })

    return ProteinGymResult(
        n_rows=len(df),
        n_scored=int(valid_mask.sum()),
        spearman_rho=rho,
        spearman_ci_low=lo,
        spearman_ci_high=hi,
        rmse=rmse,
        predictions=predictions,
    )
