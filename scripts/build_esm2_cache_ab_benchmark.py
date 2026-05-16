"""Pre-compute ESM-2 t12 VH+VL embeddings for the ab-benchmark harmonized parquet.

Default target cache: data/esm2_cache_ab_benchmark.pt (content-addressed,
distinct from the legacy data/esm2_embeddings_cache.pt).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/build_esm2_cache_ab_benchmark.py
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/build_esm2_cache_ab_benchmark.py --batch-size 16
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/build_esm2_cache_ab_benchmark.py --only-labeled

Idempotent: already-cached (vh, vl) pairs are skipped. Re-running on the
same parquet produces no new forward passes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from ab_benchmark_loader import (  # noqa: E402
    DEFAULT_PARQUET_PATH,
    KNOWN_ENDPOINTS,
    load_harmonized_wide,
)
from esm2_features import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    EMBEDDING_DIM,
    MODEL_NAME,
    cache_info,
    get_esm2_embeddings,
)


def _select_rows(df: pd.DataFrame, only_labeled: bool) -> pd.DataFrame:
    # Always require both chains.
    mask = df["vh"].notna() & df["vl"].notna()
    mask &= df["vh"].astype(str).str.len() > 0
    mask &= df["vl"].astype(str).str.len() > 0

    if only_labeled:
        ep_cols = [c for c in KNOWN_ENDPOINTS if c in df.columns]
        labeled = df[ep_cols].notna().any(axis=1) if ep_cols else pd.Series([False] * len(df))
        mask &= labeled

    return df.loc[mask].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET_PATH,
        help=f"Path to ab-benchmark wide parquet (default: {DEFAULT_PARQUET_PATH}).",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Output cache path (default: {DEFAULT_CACHE_PATH}).",
    )
    ap.add_argument(
        "--only-labeled",
        action="store_true",
        help="Only embed rows that have at least one biophysical endpoint measurement.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="ESM-2 forward-pass batch size (default 16; lower if OOM).",
    )
    args = ap.parse_args(argv)

    df = load_harmonized_wide(args.parquet)
    rows = _select_rows(df, only_labeled=args.only_labeled)

    vh = rows["vh"].astype(str).tolist()
    vl = rows["vl"].astype(str).tolist()
    print(f"Parquet:   {args.parquet}")
    print(f"Cache:     {args.cache}")
    print(f"Model:     {MODEL_NAME}")
    print(f"Dim:       {EMBEDDING_DIM}")
    print(f"Rows to embed (after filter): {len(vh)}")

    info_before = cache_info(args.cache)
    print(f"Cache before: {info_before}")

    t0 = time.time()
    embeddings = get_esm2_embeddings(vh, vl, cache_path=args.cache, batch_size=args.batch_size)
    elapsed = time.time() - t0

    info_after = cache_info(args.cache)
    newly = info_after["n_cached"] - info_before["n_cached"]
    print(f"Cache after:  {info_after}")
    print(f"Newly embedded: {newly}")
    print(f"Embeddings output shape: {embeddings.shape}")
    print(f"Elapsed: {elapsed:.1f}s " + (f"(~{elapsed/max(newly,1)*1000:.0f}ms/new pair)" if newly else "(all cached)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
