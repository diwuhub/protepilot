"""
esm2_hybrid_encoder.py  ·  ProtePilot — Unified Integration
===========================================================
Dual-Channel Encoder: ESM-2 Protein Language Model + Biophysical Features

Architecture
------------------------------------------------------------
Channel 1 (ESM-2):
    VH sequence → ESM-2 → (480,)  ──┐
                                     ├→ concat → (960,) → projection → (hidden_dim,)
    VL sequence → ESM-2 → (480,)  ──┘

Channel 2 (Biophysical):
    [pI, MW, deam, ox, acidic, basic, hydro] → (7,) → MLP → (hidden_dim,)

Fusion:
    element-wise add → LayerNorm → (hidden_dim,)

Graceful Degradation
------------------------------------------------------------
If ESM-2 / transformers is unavailable, falls back to mock embeddings
from pLM_embedder.py, preserving full pipeline functionality.

Origin: Integrates Biologics AI SimpleEncoder concept with ProtePilot ESM-2.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger("ProtePilot.ESM2HybridEncoder")

# ---------------------------------------------------------------------------
# Constants (aligned with ProtePilot pLM_embedder.py)
# ---------------------------------------------------------------------------
ESM2_EMBED_DIM = 960          # VH (480) + VL (480) concatenated
BIOPHYS_DIM = 7               # pI, MW_kDa, deam_sites, ox_sites, acidic, basic, hydro
BIOPHYS_NAMES = [
    "pI", "MW_kDa", "deam_sites", "ox_sites",
    "acidic_residues", "basic_residues", "hydrophobicity",
]


# ---------------------------------------------------------------------------
# Lazy ESM-2 embedder access (singleton from pLM_embedder.py)
# ---------------------------------------------------------------------------
def _get_esm2_embedder():
    """Lazily import and return the ProtePilot ESM2Embedder singleton."""
    try:
        from pLM_embedder import get_embedder
        return get_embedder()
    except ImportError:
        log.warning("pLM_embedder not available; using mock embeddings")
        return None


def _embed_antibody_batch(
    hc_seqs: List[str],
    lc_seqs: List[str],
    embedder=None,
) -> np.ndarray:
    """
    Embed a batch of antibody HC/LC pairs → (B, 960) numpy array.

    CRITICAL: All embeddings MUST be mean-pooled to (960,) shape before stacking.
    Falls back to mock embeddings if ESM-2 is unavailable or fails.
    """
    if embedder is None:
        embedder = _get_esm2_embedder()

    embeddings = []
    for hc, lc in zip(hc_seqs, lc_seqs):
        try:
            if embedder is not None:
                # embed_antibody returns (960,) — always mean-pooled
                emb = embedder.embed_antibody(hc, lc)
                # Validate shape
                if emb.shape != (ESM2_EMBED_DIM,):
                    log.warning(
                        f"ESM-2 returned invalid shape {emb.shape}, "
                        f"expected ({ESM2_EMBED_DIM},), using mock"
                    )
                    emb = _mock_antibody(hc, lc)
            else:
                emb = _mock_antibody(hc, lc)
        except Exception as e:
            log.warning(f"ESM-2 embed failed: {e} — falling back to mock")
            emb = _mock_antibody(hc, lc)

        # Ensure we have a valid (960,) numpy array
        if not isinstance(emb, np.ndarray) or emb.shape != (ESM2_EMBED_DIM,):
            log.warning(f"Invalid embedding shape {emb.shape}, using fresh mock")
            emb = _mock_antibody(hc, lc)

        embeddings.append(emb)

    # Stack all (960,) embeddings to (B, 960)
    batch_embeddings = np.stack(embeddings, axis=0)
    assert batch_embeddings.shape[0] == len(hc_seqs), \
        f"Batch size mismatch: {batch_embeddings.shape[0]} vs {len(hc_seqs)}"
    assert batch_embeddings.shape[1] == ESM2_EMBED_DIM, \
        f"Embedding dimension mismatch: {batch_embeddings.shape[1]} vs {ESM2_EMBED_DIM}"

    return batch_embeddings


def _mock_antibody(hc: str, lc: str) -> np.ndarray:
    """Deterministic mock embedding when ESM-2 is unavailable."""
    try:
        from pLM_embedder import mock_antibody_embedding
        return mock_antibody_embedding(hc, lc, dim=ESM2_EMBED_DIM)
    except ImportError:
        # Minimal fallback: AA composition + hash features
        return _minimal_mock(hc, lc)


def _minimal_mock(hc: str, lc: str, dim: int = 960) -> np.ndarray:
    """Ultra-minimal mock when pLM_embedder.py itself is unavailable."""
    import hashlib
    AA = "ACDEFGHIKLMNPQRSTVWY"
    half = dim // 2
    result = np.zeros(dim, dtype=np.float32)
    for idx, seq in enumerate([hc, lc]):
        seq_upper = seq.upper()
        total = max(len(seq_upper), 1)
        # AA composition (20-dim)
        comp = np.array([seq_upper.count(aa) / total for aa in AA], dtype=np.float32)
        # Hash features
        h = hashlib.md5(seq_upper.encode()).hexdigest()
        seed = int(h[:8], 16)
        rng = np.random.RandomState(seed)
        hash_feats = rng.randn(half - 20).astype(np.float32) * 0.1
        offset = idx * half
        result[offset:offset + 20] = comp
        result[offset + 20:offset + half] = hash_feats
    return result


# ---------------------------------------------------------------------------
# ESM-2 Hybrid Encoder (PyTorch Module)
# ---------------------------------------------------------------------------
class ESM2HybridEncoder(nn.Module):
    """
    Dual-channel encoder fusing ESM-2 protein embeddings with biophysical
    features for multi-task antibody property prediction.

    Parameters
    ----------
    esm2_dim : int
        Dimensionality of the ESM-2 antibody embedding (default 960).
    biophys_dim : int
        Number of biophysical features (default 7).
    hidden_dim : int
        Output embedding dimensionality (default 256).
    dropout : float
        Dropout rate applied in both channels.
    """

    def __init__(
        self,
        esm2_dim: int = ESM2_EMBED_DIM,
        biophys_dim: int = BIOPHYS_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.esm2_dim = esm2_dim
        self.biophys_dim = biophys_dim
        self.hidden_dim = hidden_dim

        # Channel 1: ESM-2 embedding projection
        self.esm2_proj = nn.Sequential(
            nn.Linear(esm2_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Channel 2: Biophysical feature projection
        self.biophys_proj = nn.Sequential(
            nn.Linear(biophys_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # Cache for ESM-2 embedder
        self._embedder = None

    @property
    def embedding_dim(self) -> int:
        """Output dimensionality of the encoder."""
        return self.hidden_dim

    def _get_embedder(self):
        """Lazy-load ESM-2 embedder."""
        if self._embedder is None:
            self._embedder = _get_esm2_embedder()
        return self._embedder

    def encode_sequences(
        self,
        hc_seqs: List[str],
        lc_seqs: List[str],
    ) -> Tensor:
        """
        Encode HC/LC sequence pairs into ESM-2 embeddings.

        CRITICAL: Returns mean-pooled embeddings with shape (batch, esm2_dim).

        Returns
        -------
        Tensor, shape ``(batch, esm2_dim)`` where esm2_dim=960
            All embeddings are mean-pooled to fixed-size vectors (no per-token embeddings).
        """
        embedder = self._get_embedder()
        emb_np = _embed_antibody_batch(hc_seqs, lc_seqs, embedder=embedder)

        # Validate shape before converting to tensor
        assert emb_np.shape[0] == len(hc_seqs), \
            f"Batch size mismatch: {emb_np.shape[0]} != {len(hc_seqs)}"
        assert emb_np.shape[1] == self.esm2_dim, \
            f"ESM-2 dim mismatch: {emb_np.shape[1]} != {self.esm2_dim}"

        tensor = torch.from_numpy(emb_np).float()
        return tensor

    def forward(
        self,
        hc_seqs: List[str],
        lc_seqs: List[str],
        biophys_features: Optional[Tensor] = None,
        cached_esm2_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode antibody sequences and optional biophysical features.

        Parameters
        ----------
        hc_seqs : List[str]
            Heavy chain amino acid sequences.
        lc_seqs : List[str]
            Light chain amino acid sequences.
        biophys_features : Tensor, shape ``(batch, 7)``, optional
            Biophysical features [pI, MW_kDa, deam_sites, ox_sites,
            acidic_residues, basic_residues, hydrophobicity].
        cached_esm2_emb : Tensor, shape ``(batch, 960)``, optional
            Pre-computed ESM-2 embeddings. If provided, skips running
            ESM-2 on sequences (massive speedup during training).

        Returns
        -------
        Tensor, shape ``(batch, hidden_dim)``
            Fused representation ready for downstream task heads.

        Raises
        ------
        RuntimeError
            If tensor shapes are incompatible for fusion.
        """
        batch_size = len(hc_seqs)

        # Channel 1: ESM-2 embeddings (use cache if available)
        if cached_esm2_emb is not None:
            esm2_emb = cached_esm2_emb
            # Validate cached embedding shape
            assert esm2_emb.shape[0] == batch_size, \
                f"Cached ESM-2 batch size {esm2_emb.shape[0]} != {batch_size}"
            assert esm2_emb.shape[1] == self.esm2_dim, \
                f"Cached ESM-2 dim {esm2_emb.shape[1]} != {self.esm2_dim}"
        else:
            esm2_emb = self.encode_sequences(hc_seqs, lc_seqs)  # (B, 960)

        # Move to same device as model parameters
        device = next(self.parameters()).device
        esm2_emb = esm2_emb.to(device)

        # Project ESM-2 embeddings
        h = self.esm2_proj(esm2_emb)  # (B, hidden_dim)
        assert h.shape == (batch_size, self.hidden_dim), \
            f"ESM-2 projection shape mismatch: {h.shape} != ({batch_size}, {self.hidden_dim})"

        # Channel 2: Biophysical features (if provided)
        if biophys_features is not None:
            biophys_features = biophys_features.to(device)

            # Validate biophysical features shape
            assert biophys_features.shape[0] == batch_size, \
                f"Biophys batch size {biophys_features.shape[0]} != {batch_size}"
            assert biophys_features.shape[1] == self.biophys_dim, \
                f"Biophys dim {biophys_features.shape[1]} != {self.biophys_dim}"

            h_bio = self.biophys_proj(biophys_features)  # (B, hidden_dim)
            assert h_bio.shape == (batch_size, self.hidden_dim), \
                f"Biophys projection shape mismatch: {h_bio.shape} != ({batch_size}, {self.hidden_dim})"

            # Element-wise fusion (now safe — shapes are guaranteed compatible)
            h = h + h_bio

        h = self.fusion_norm(h)
        assert h.shape == (batch_size, self.hidden_dim), \
            f"Final output shape mismatch: {h.shape} != ({batch_size}, {self.hidden_dim})"

        return h  # (B, hidden_dim)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"esm2_dim={self.esm2_dim}, "
            f"biophys_dim={self.biophys_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )
