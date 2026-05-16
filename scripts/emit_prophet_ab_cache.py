"""Emit PROPHET-Ab ridge out-of-fold predictions to the TSV format expected
by ab-benchmark's baselines.prophet_ab wrapper.

After running this, point ab-benchmark at the TSV:

    export PROPHET_AB_CACHE_TSV=/Users/di/Projects/ProtePilot/data/prophet_ab_predictions.tsv
    # then rerun:
    cd /Users/di/Projects/ab-benchmark
    .venv/bin/python -m ab_benchmark.baselines.run_all

The Phase 0 "PROPHET-Ab: unavailable — requires Phase 1 ESM-2 embeddings"
row will flip to available with OOF predictions for each Jain 2017 endpoint.

TSV columns:
    ab_id       # source-native identifier (matches ab-benchmark harmonized row)
    endpoint    # canonical endpoint kind (tm_onset_c, hic_rt, ...)
    prediction  # OOF prediction from ridge on 960-dim ESM-2 t12 VH+VL

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/emit_prophet_ab_cache.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from ab_benchmark_loader import load_harmonized_training_set  # noqa: E402
from baseline_runner import evaluate_prophet_ab  # noqa: E402
from esm2_features import get_esm2_embeddings  # noqa: E402


DEFAULT_ENDPOINTS = [
    "tm_onset_c",
    "hic_rt",
    "ac_sins",
    "bvp_score",
    "psr_score",
    "expression_mgl",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=_PROJECT_ROOT / "data" / "prophet_ab_predictions.tsv",
        help="Output TSV path (default: data/prophet_ab_predictions.tsv).",
    )
    ap.add_argument(
        "--endpoints",
        nargs="+",
        default=DEFAULT_ENDPOINTS,
        help="Endpoints to produce predictions for.",
    )
    ap.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="GroupKFold splits for OOF predictions (default 5).",
    )
    args = ap.parse_args(argv)

    rows: list[dict] = []
    summary: list[str] = []
    for endpoint in args.endpoints:
        batch = load_harmonized_training_set(endpoint)
        X = get_esm2_embeddings(batch.vh, batch.vl, allow_build=False)
        result = evaluate_prophet_ab(batch, X, n_splits=args.n_splits)
        summary.append(
            f"  {endpoint:<18s} n={result.n:>3}  ρ={result.spearman.point:+.3f} "
            f"[{result.spearman.low:+.2f},{result.spearman.high:+.2f}]"
        )
        for i, pred in enumerate(result.oof_predictions):
            rows.append({
                "ab_id": batch.ab_ids[i],
                "endpoint": endpoint,
                "prediction": float(pred),
            })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)

    print(f"Wrote {len(df)} predictions across {len(args.endpoints)} endpoints to {args.out}")
    print("\nPer-endpoint OOF Spearman ρ (ridge on 960-dim ESM-2 t12 VH+VL):")
    for line in summary:
        print(line)
    print()
    print("To enable in ab-benchmark:")
    print(f"  export PROPHET_AB_CACHE_TSV={args.out}")
    print("  cd /Users/di/Projects/ab-benchmark && .venv/bin/python -m ab_benchmark.baselines.run_all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
