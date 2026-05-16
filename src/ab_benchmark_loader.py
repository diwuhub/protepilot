"""Bridge to the ab-benchmark harmonized parquet.

ProtePilot consumes the harmonized antibody developability dataset produced
by the sibling `ab-benchmark` repo. This module reads the parquet, filters to rows with a requested
endpoint, and returns an array view suitable for sklearn training plus
leakage-resistant group labels.

Design:
    - Additive only — does not modify any existing ProtePilot module.
    - Reads the wide-format parquet from a sibling `ab-benchmark` checkout by
      default. A PROTEPILOT_AB_BENCHMARK_PARQUET env var overrides.
    - Cluster assignment: prefers `ab_benchmark.eval.splits.assign_clusters`
      if the sibling package is importable, else falls back to clustering
      by `ab_id_canonical` alone (weaker but safe).
    - Fails gracefully with FileNotFoundError + guidance if the parquet
      is missing (e.g. ab-benchmark not yet built).

Canonical endpoints (per ab_benchmark.schema.EndpointKind):
    tm_onset_c, hic_rt, ac_sins, bvp_score, psr_score, expression_mgl,
    solubility, viscosity_cp, clearance
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PARQUET_PATH = (
    Path(__file__).resolve().parents[2]
    / "ab-benchmark"
    / "data"
    / "processed"
    / "harmonized_antibody_dev_wide.parquet"
)

# Endpoints that should always be present in the harmonized wide parquet.
# If this list drifts from ab_benchmark.schema.EndpointKind we add a Phase 0
# test there; see ab-benchmark/tests/test_schema.py::TestEnums.
KNOWN_ENDPOINTS = {
    "tm_onset_c",
    "hic_rt",
    "ac_sins",
    "bvp_score",
    "psr_score",
    "expression_mgl",
    "solubility",
    "viscosity_cp",
    "clearance",
}


@dataclass
class HarmonizedBatch:
    """Training batch for one endpoint from the harmonized benchmark."""

    endpoint: str
    ab_ids: np.ndarray          # source-native identifiers
    canonical_ids: np.ndarray   # cross-source canonical IDs (str)
    sources: np.ndarray         # SourceDataset values (str)
    vh: np.ndarray              # VH sequences (str)
    vl: np.ndarray              # VL sequences (str)
    y: np.ndarray               # measured endpoint values (float)
    groups: np.ndarray          # cluster IDs for GroupKFold (int)

    def __len__(self) -> int:
        return len(self.y)

    def summary(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "n": len(self.y),
            "n_clusters": int(len(np.unique(self.groups))),
            "sources": pd.Series(self.sources).value_counts().to_dict(),
            "y_min": float(np.min(self.y)),
            "y_max": float(np.max(self.y)),
            "y_mean": float(np.mean(self.y)),
        }


def _resolve_parquet_path(parquet_path: Path | str | None) -> Path:
    if parquet_path is not None:
        return Path(parquet_path)
    env = os.environ.get("PROTEPILOT_AB_BENCHMARK_PARQUET")
    if env:
        return Path(env)
    return DEFAULT_PARQUET_PATH


def load_harmonized_wide(parquet_path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw wide parquet from ab-benchmark. Does not filter.

    Raises FileNotFoundError with guidance if the parquet is missing.
    """
    path = _resolve_parquet_path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ab-benchmark harmonized parquet not found at {path}. "
            f"Build it by running, in the sibling repo:\n"
            f"    cd ../ab-benchmark\n"
            f"    .venv/bin/python -m ab_benchmark.data.build_harmonized\n"
            f"Or set PROTEPILOT_AB_BENCHMARK_PARQUET to point at an alternative location."
        )
    return pd.read_parquet(path)


def _compute_groups(df: pd.DataFrame) -> np.ndarray:
    """Cluster antibodies into groups for leakage-resistant CV.

    Prefers ab_benchmark.eval.splits.assign_clusters (which groups on
    V-gene + CDR-H3-length + 50% identity) when the sibling package is
    importable; otherwise falls back to `ab_id_canonical` alone.
    """
    try:
        from ab_benchmark.eval.splits import assign_clusters  # type: ignore

        required = {"ab_id_canonical", "v_gene_heavy", "cdr_h3", "cdr_h3_length"}
        if required.issubset(df.columns):
            return assign_clusters(df, identity_threshold=0.5).to_numpy()
    except ImportError:
        pass

    # Fallback: group by canonical ID only. Same antibody across sources
    # stays in one cluster; similar CDR-H3s in different ab_ids do not.
    if "ab_id_canonical" in df.columns:
        codes, _ = pd.factorize(df["ab_id_canonical"])
        return codes
    # Last resort: every row is its own cluster (no grouping).
    return np.arange(len(df))


def load_harmonized_training_set(
    endpoint: str,
    parquet_path: Path | str | None = None,
) -> HarmonizedBatch:
    """Load a one-endpoint training batch from the harmonized parquet.

    Filters rows where the endpoint is NaN. Both VH and VL are required
    (otherwise downstream ESM-2 concatenation fails), so rows missing
    either sequence are also dropped.

    Parameters
    ----------
    endpoint : str
        One of KNOWN_ENDPOINTS (e.g. 'tm_onset_c', 'hic_rt').
    parquet_path : Path | str | None
        Override the default ab-benchmark parquet location.

    Returns
    -------
    HarmonizedBatch
        Parallel arrays ready for sklearn; `groups` is a leakage-resistant
        cluster assignment for GroupKFold.
    """
    if endpoint not in KNOWN_ENDPOINTS:
        raise ValueError(
            f"Unknown endpoint {endpoint!r}. Known endpoints: {sorted(KNOWN_ENDPOINTS)}"
        )

    df = load_harmonized_wide(parquet_path)
    if endpoint not in df.columns:
        raise ValueError(
            f"Endpoint column {endpoint!r} not present in parquet. "
            f"Columns: {sorted(df.columns)}"
        )

    mask = df[endpoint].notna() & df["vh"].notna() & df["vl"].notna()
    # Drop empty-string VH/VL too.
    mask &= df["vh"].astype(str).str.len() > 0
    mask &= df["vl"].astype(str).str.len() > 0
    subset = df.loc[mask].reset_index(drop=True)

    if subset.empty:
        raise ValueError(
            f"No rows with non-null {endpoint!r} and both VH and VL present. "
            f"Check upstream harmonization."
        )

    groups = _compute_groups(subset)

    return HarmonizedBatch(
        endpoint=endpoint,
        ab_ids=subset["ab_id"].to_numpy(),
        canonical_ids=subset["ab_id_canonical"].to_numpy(),
        sources=subset["source"].to_numpy(),
        vh=subset["vh"].to_numpy(),
        vl=subset["vl"].to_numpy(),
        y=subset[endpoint].to_numpy(dtype=float),
        groups=np.asarray(groups, dtype=int),
    )


def available_endpoints(parquet_path: Path | str | None = None) -> dict[str, int]:
    """Return a {endpoint: n_rows_with_that_endpoint} map for quick inspection."""
    df = load_harmonized_wide(parquet_path)
    out: dict[str, int] = {}
    for ep in KNOWN_ENDPOINTS:
        if ep in df.columns:
            mask = df[ep].notna() & df["vh"].notna() & df["vl"].notna()
            out[ep] = int(mask.sum())
    return out
