"""
build_esm2_cache.py — Pre-compute ESM-2 embeddings for unified_training_data.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main():
    data_path = os.path.join(PROJECT_ROOT, "data", "unified_training_data.csv")
    cache_path = os.path.join(PROJECT_ROOT, "data", "esm2_embeddings_cache.pt")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    hc_seqs = df["hc_sequence"].fillna("").astype(str).tolist()
    lc_seqs = df["lc_sequence"].fillna("").astype(str).tolist()

    # Get ESM-2 embedder
    from esm2_hybrid_encoder import _get_esm2_embedder
    embedder = _get_esm2_embedder()
    if embedder is None:
        print("ERROR: ESM-2 not available")
        return

    print(f"ESM-2 embedder ready. Computing embeddings for {len(df)} sequences...")

    from esm2_hybrid_encoder import _embed_antibody_batch
    t0 = time.time()
    embeddings = _embed_antibody_batch(hc_seqs, lc_seqs, embedder=embedder)
    elapsed = time.time() - t0

    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(df)*1000:.0f}ms/seq)")

    # Save cache
    torch.save({
        "embeddings": torch.tensor(embeddings, dtype=torch.float32),
        "n_samples": len(df),
        "data_path": data_path,
    }, cache_path)
    size_mb = os.path.getsize(cache_path) / 1024 / 1024
    print(f"  Cache saved: {cache_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
