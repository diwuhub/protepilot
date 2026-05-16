"""ESM-2 t12 VH+VL antibody features with transparent caching.

Thin wrapper around ProtePilot's existing `esm2_hybrid_encoder` that:

1. Commits to ONE antibody embedding shape everywhere: **960-dim**
   (VH 480-dim + VL 480-dim concatenation, mean-pooled per chain from
   ESM-2 t12 / esm2_t12_35M_UR50D). Documented in model metadata
   and this module.

2. Provides content-addressed caching keyed on `(vh, vl)` — not on row
   index — so embeddings computed for one dataset (Jain 137) can be
   reused by any downstream consumer with the same sequences, including
   the new ab-benchmark harmonized parquet.

3. Never rebuilds an embedding that is already cached. First build is
   expensive (~100ms/pair on CPU); cache hits are ~free.

Cache file format (torch.save dict):
    {
      "version":    int,
      "model_name": str,   # "facebook/esm2_t12_35M_UR50D"
      "dim":        int,   # 960
      "hash_to_row": dict[str, int],
      "embeddings": torch.Tensor  # (N, 960) float32
    }

Cache path: `data/esm2_cache_ab_benchmark.pt` (distinct from the legacy
`data/esm2_embeddings_cache.pt` which is keyed by row index of
`unified_training_data.csv`). Override via PROTEPILOT_ESM2_CACHE env var.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Resolve ProtePilot project root without hard-coding.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
EMBEDDING_DIM = 960  # VH 480 + VL 480 concatenated
CACHE_VERSION = 1

DEFAULT_CACHE_PATH = _PROJECT_ROOT / "data" / "esm2_cache_ab_benchmark.pt"


def _resolve_cache_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("PROTEPILOT_ESM2_CACHE")
    if env:
        return Path(env)
    return DEFAULT_CACHE_PATH


# --- content-addressed hash -------------------------------------------------


def seq_pair_hash(vh: str, vl: str) -> str:
    """Stable SHA-256 of a (VH, VL) pair. Used as cache key."""
    h = hashlib.sha256()
    h.update(vh.strip().upper().encode("utf-8"))
    h.update(b"|")
    h.update(vl.strip().upper().encode("utf-8"))
    return h.hexdigest()


# --- cache I/O --------------------------------------------------------------


@dataclass
class _CacheState:
    hash_to_row: dict[str, int]
    embeddings: torch.Tensor

    def size(self) -> int:
        return self.embeddings.shape[0]


def _empty_cache() -> _CacheState:
    return _CacheState(
        hash_to_row={},
        embeddings=torch.zeros((0, EMBEDDING_DIM), dtype=torch.float32),
    )


def load_cache(path: Path | str | None = None) -> _CacheState:
    p = _resolve_cache_path(path)
    if not p.exists():
        return _empty_cache()
    blob = torch.load(p, weights_only=False)
    if blob.get("version") != CACHE_VERSION:
        raise RuntimeError(
            f"ESM-2 cache at {p} has version {blob.get('version')!r}, "
            f"expected {CACHE_VERSION}. Delete it and rebuild."
        )
    if blob.get("dim") != EMBEDDING_DIM:
        raise RuntimeError(
            f"ESM-2 cache at {p} has dim {blob.get('dim')!r}, expected {EMBEDDING_DIM}."
        )
    return _CacheState(
        hash_to_row=dict(blob["hash_to_row"]),
        embeddings=blob["embeddings"],
    )


def save_cache(state: _CacheState, path: Path | str | None = None) -> Path:
    p = _resolve_cache_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CACHE_VERSION,
            "model_name": MODEL_NAME,
            "dim": EMBEDDING_DIM,
            "hash_to_row": state.hash_to_row,
            "embeddings": state.embeddings,
        },
        p,
    )
    return p


# --- public API -------------------------------------------------------------


def get_esm2_embeddings(
    vh_list: list[str] | np.ndarray,
    vl_list: list[str] | np.ndarray,
    cache_path: Path | str | None = None,
    allow_build: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    """Return a (N, 960) array of ESM-2 t12 VH+VL embeddings.

    Transparent load-or-build:
      1. Look up each (vh, vl) by content-hash in the cache.
      2. If any misses, build them via esm2_hybrid_encoder._embed_antibody_batch
         (in chunks of batch_size), persist to cache, and append to the return.
      3. Return results in the input order.

    Parameters
    ----------
    vh_list, vl_list : parallel lists/arrays of VH and VL sequences
    cache_path : optional override
    allow_build : if False and there are misses, raise RuntimeError instead
        of computing (useful for CI/test determinism)
    batch_size : chunk size for the ESM-2 forward pass
    """
    vh = [str(s).strip().upper() for s in vh_list]
    vl = [str(s).strip().upper() for s in vl_list]
    if len(vh) != len(vl):
        raise ValueError(f"vh and vl must be the same length (got {len(vh)} vs {len(vl)})")
    if len(vh) == 0:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    cache = load_cache(cache_path)
    hashes = [seq_pair_hash(h, l) for h, l in zip(vh, vl)]

    # Build: any (vh, vl) not yet in cache.
    missing_unique: dict[str, tuple[str, str]] = {}
    for hv, h, l in zip(hashes, vh, vl):
        if hv not in cache.hash_to_row and hv not in missing_unique:
            missing_unique[hv] = (h, l)

    if missing_unique:
        if not allow_build:
            raise RuntimeError(
                f"{len(missing_unique)} sequences not in cache and allow_build=False. "
                f"Populate the cache first via scripts/build_esm2_cache_ab_benchmark.py."
            )

        from esm2_hybrid_encoder import _embed_antibody_batch, _get_esm2_embedder

        embedder = _get_esm2_embedder()
        if embedder is None:
            raise RuntimeError(
                "ESM-2 embedder unavailable — transformers or torch missing? "
                "Install requirements-training.txt."
            )

        # Build in order for reproducibility.
        hv_order = list(missing_unique.keys())
        hc_batch = [missing_unique[hv][0] for hv in hv_order]
        lc_batch = [missing_unique[hv][1] for hv in hv_order]

        # Chunked embedding.
        new_embeds = np.zeros((len(hv_order), EMBEDDING_DIM), dtype=np.float32)
        for i in range(0, len(hv_order), batch_size):
            chunk_hc = hc_batch[i : i + batch_size]
            chunk_lc = lc_batch[i : i + batch_size]
            chunk = _embed_antibody_batch(chunk_hc, chunk_lc, embedder=embedder)
            if chunk.shape != (len(chunk_hc), EMBEDDING_DIM):
                raise RuntimeError(
                    f"ESM-2 backend returned shape {chunk.shape}, "
                    f"expected ({len(chunk_hc)}, {EMBEDDING_DIM})."
                )
            new_embeds[i : i + len(chunk_hc)] = chunk

        # Append to cache.
        start = cache.size()
        cache.embeddings = torch.cat(
            [cache.embeddings, torch.from_numpy(new_embeds)],
            dim=0,
        )
        for offset, hv in enumerate(hv_order):
            cache.hash_to_row[hv] = start + offset
        save_cache(cache, cache_path)

    # Assemble output in input order.
    out = np.zeros((len(hashes), EMBEDDING_DIM), dtype=np.float32)
    for i, hv in enumerate(hashes):
        out[i] = cache.embeddings[cache.hash_to_row[hv]].numpy()
    return out


def cache_info(cache_path: Path | str | None = None) -> dict:
    """Summarize the current cache state without modifying it."""
    p = _resolve_cache_path(cache_path)
    if not p.exists():
        return {"path": str(p), "exists": False, "n_cached": 0}
    cache = load_cache(p)
    return {
        "path": str(p),
        "exists": True,
        "n_cached": cache.size(),
        "dim": int(cache.embeddings.shape[1]) if cache.embeddings.ndim == 2 else 0,
        "model_name": MODEL_NAME,
        "cache_version": CACHE_VERSION,
    }
