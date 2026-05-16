"""
precompute_esm2_embeddings.py
===========================================================
Run ESM-2 once on all sequences in the training CSV and cache
the resulting embeddings to a .pt file.

This avoids running ESM-2 on every forward pass during training,
reducing epoch time from minutes to milliseconds on CPU.

Usage:
    python scripts/precompute_esm2_embeddings.py [--data path/to/csv] [--out path/to/cache.pt]
"""

import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
import torch

from esm2_hybrid_encoder import ESM2HybridEncoder, ESM2_EMBED_DIM


def main():
    parser = argparse.ArgumentParser(description="Precompute ESM-2 embeddings")
    parser.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "unified_training_data.csv"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "esm2_embeddings_cache.pt"))
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    n = len(df)
    print(f"  Samples: {n}")

    # Detect sequence columns
    if "hc_sequence" in df.columns and "lc_sequence" in df.columns:
        hc_seqs = df["hc_sequence"].astype(str).tolist()
        lc_seqs = df["lc_sequence"].astype(str).tolist()
    elif "sequence" in df.columns:
        seqs = df["sequence"].astype(str).tolist()
        hc_seqs = seqs
        lc_seqs = seqs
    else:
        raise ValueError("CSV must have hc_sequence/lc_sequence or sequence column")

    # Initialize encoder (just for ESM-2 embedding, not training)
    print("Initializing ESM-2 encoder...")
    encoder = ESM2HybridEncoder()

    # Embed one-by-one with progress (ESM-2 is the bottleneck)
    print(f"Computing ESM-2 embeddings for {n} sequences...")
    embeddings = []
    start = time.time()

    for i in range(n):
        emb = encoder.encode_sequences([hc_seqs[i]], [lc_seqs[i]])  # (1, 640)
        embeddings.append(emb.squeeze(0))  # (640,)

        if (i + 1) % 10 == 0 or i == n - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {elapsed:.1f}s elapsed, ~{eta:.0f}s remaining ({rate:.1f} seq/s)")

    # Stack into single tensor
    all_embeddings = torch.stack(embeddings)  # (N, 640)
    elapsed = time.time() - start
    print(f"\nDone! {n} embeddings computed in {elapsed:.1f}s")
    print(f"  Shape: {all_embeddings.shape}")

    # Save cache
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "embeddings": all_embeddings,       # (N, 640)
        "hc_seqs": hc_seqs,
        "lc_seqs": lc_seqs,
        "source_csv": args.data,
        "n_samples": n,
        "embed_dim": ESM2_EMBED_DIM,
    }, args.out)
    print(f"  Saved to: {args.out}")
    print(f"  File size: {os.path.getsize(args.out) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
