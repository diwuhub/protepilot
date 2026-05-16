"""
pLM_embedder.py  ·  ProtePilot — Milestone 8
===========================================================
Protein Language Model (ESM-2) Embedding Module

Version   : 1.1
Author    : Di (ProtePilot)
Depends   : transformers (optional), numpy

Architecture
------------------------------------------------------------
Uses facebook/esm2_t12_35M_UR50D (35M parameter ESM-2 model)
to generate per-residue embeddings, then average-pools to
produce fixed-length sequence representations.

    Single chain:  sequence -> ESM-2 -> mean-pool -> (480,)
    Antibody:      VH (480) + VL (480) -> concatenate -> (960,)

Upgrade History
------------------------------------------------------------
  v1.0: esm2_t6_8M_UR50D  (320-dim hidden, 640-dim antibody)
  v1.1: esm2_t12_35M_UR50D (480-dim hidden, 960-dim antibody)
        NOTE: Dimension change requires full model retrain.

Graceful Degradation
------------------------------------------------------------
If HuggingFace transformers is not installed, the module
falls back to a deterministic mock embedding based on amino
acid composition features. This allows the downstream
XGBoost predictor to still function (with reduced accuracy).

References
------------------------------------------------------------
  ESM-2: Lin et al., Science 379 (2023)
  HuggingFace: https://huggingface.co/facebook/esm2_t12_35M_UR50D
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ProtePilot.pLM")

# Embedding dimensions
ESM2_HIDDEN_DIM = 480       # ESM-2 t12_35M hidden size
ANTIBODY_EMBED_DIM = 960    # VH (480) + VL (480)

# Standard amino acid alphabet
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


# ===========================================================================
# Mock Embedding (fallback when transformers unavailable)
# ===========================================================================

def _aa_composition(sequence: str) -> np.ndarray:
    """
    Compute amino acid composition vector (20-dim, normalized).

    Returns fraction of each standard amino acid in the sequence.
    """
    seq = sequence.upper()
    total = max(len(seq), 1)
    comp = np.zeros(20, dtype=np.float32)
    for i, aa in enumerate(AA_ALPHABET):
        comp[i] = seq.count(aa) / total
    return comp


def _sequence_hash_features(sequence: str, dim: int) -> np.ndarray:
    """
    Generate deterministic pseudo-random features from sequence hash.

    Uses MD5 hash of the sequence to seed a PRNG, producing
    reproducible features that vary by sequence content.
    """
    h = hashlib.md5(sequence.upper().encode()).hexdigest()
    seed = int(h[:8], 16)
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32) * 0.1


def mock_embedding(sequence: str, dim: int = ESM2_HIDDEN_DIM) -> np.ndarray:
    """
    Generate a deterministic mock embedding when ESM-2 is unavailable.

    Combines amino acid composition (20-dim) with hash-derived features
    to fill the remaining dimensions. The result is reproducible for
    the same input sequence.

    Parameters
    ----------
    sequence : Amino acid sequence string
    dim      : Target embedding dimension (default 480)

    Returns
    -------
    np.ndarray of shape (dim,)
    """
    if not sequence or len(sequence) < 5:
        return np.zeros(dim, dtype=np.float32)

    seq = "".join(c for c in sequence.upper() if c in AA_ALPHABET)
    if len(seq) < 5:
        return np.zeros(dim, dtype=np.float32)

    # First 20 dims: amino acid composition
    comp = _aa_composition(seq)

    # Additional biophysical features (5-dim)
    length_norm = min(len(seq) / 500.0, 1.0)
    charge = (seq.count("K") + seq.count("R") + seq.count("H")
              - seq.count("D") - seq.count("E")) / max(len(seq), 1)
    hydrophobic = sum(seq.count(aa) for aa in "AVILMFYW") / max(len(seq), 1)
    polar = sum(seq.count(aa) for aa in "STNQ") / max(len(seq), 1)
    aromatic = sum(seq.count(aa) for aa in "FWY") / max(len(seq), 1)
    biophys = np.array([length_norm, charge, hydrophobic, polar, aromatic],
                       dtype=np.float32)

    # Hash-derived features fill the remaining dimensions
    n_structured = len(comp) + len(biophys)  # 25
    n_hash = max(0, dim - n_structured)
    hash_feats = _sequence_hash_features(seq, n_hash)

    # Concatenate and pad/truncate to exact dim
    embedding = np.concatenate([comp, biophys, hash_feats])[:dim]
    if len(embedding) < dim:
        embedding = np.pad(embedding, (0, dim - len(embedding)))

    return embedding.astype(np.float32)


def mock_antibody_embedding(
    vh_seq: str,
    vl_seq: str,
    dim: int = ANTIBODY_EMBED_DIM,
) -> np.ndarray:
    """
    Generate mock antibody embedding by concatenating VH + VL mock embeddings.

    Parameters
    ----------
    vh_seq : Heavy chain variable region sequence
    vl_seq : Light chain variable region sequence
    dim    : Target dimension (default 960 = 480 + 480)

    Returns
    -------
    np.ndarray of shape (dim,)
    """
    half = dim // 2
    vh_embed = mock_embedding(vh_seq, half)
    vl_embed = mock_embedding(vl_seq, half)
    return np.concatenate([vh_embed, vl_embed])


# ===========================================================================
# ESM-2 Embedder Class
# ===========================================================================

class ESM2Embedder:
    """
    ESM-2 Protein Language Model embedder with graceful fallback.

    Usage
    -----
    >>> embedder = ESM2Embedder()
    >>> emb = embedder.embed_sequence("EVQLVESGG...")  # (480,)
    >>> ab_emb = embedder.embed_antibody(vh_seq, vl_seq)  # (960,)

    If transformers is not installed, automatically falls back to
    composition-based mock embeddings with a logged warning.
    """

    def __init__(self, model_name: str = "facebook/esm2_t12_35M_UR50D"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._mock_mode = False
        self._warned = False

        if os.environ.get("PROTEPILOT_FORCE_MOCK_EMBEDDING", "").lower() in {"1", "true", "yes"}:
            self._mock_mode = True
            self._warn_once("PROTEPILOT_FORCE_MOCK_EMBEDDING enabled - using mock embeddings")
            return

        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.eval()
            log.info("ESM-2 model loaded: %s", model_name)
        except ImportError:
            self._mock_mode = True
            self._warn_once("transformers not installed — using mock embeddings. "
                           "Install: pip install transformers sentencepiece")
        except Exception as e:
            self._mock_mode = True
            self._warn_once(f"ESM-2 model loading failed ({e}) — using mock embeddings")

    def _warn_once(self, message: str):
        """Log warning only on first call."""
        if not self._warned:
            log.warning(message)
            self._warned = True

    @property
    def is_mock(self) -> bool:
        """Whether the embedder is in mock/fallback mode."""
        return self._mock_mode

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """
        Generate mean-pooled embedding for a single protein sequence.

        Parameters
        ----------
        sequence : Amino acid sequence (any length)

        Returns
        -------
        np.ndarray of shape (480,) — ESM-2 hidden dim (always mean-pooled)
        """
        if self._mock_mode:
            return mock_embedding(sequence, ESM2_HIDDEN_DIM)

        try:
            import torch
            # Clean sequence
            clean_seq = "".join(c for c in sequence.upper() if c in AA_ALPHABET)
            if len(clean_seq) < 5:
                return np.zeros(ESM2_HIDDEN_DIM, dtype=np.float32)

            # Truncate to ESM-2 max length (1024 tokens for t12)
            if len(clean_seq) > 1022:
                clean_seq = clean_seq[:1022]

            inputs = self._tokenizer(clean_seq, return_tensors="pt",
                                     truncation=True, max_length=1024)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Mean-pool over sequence length (excluding BOS/EOS tokens)
            # CRITICAL: Always return mean-pooled (batch_size, 480) not per-token
            hidden_states = outputs.last_hidden_state  # (1, seq_len+2, 480)
            # Remove special tokens (BOS at 0, EOS at -1)
            seq_hidden = hidden_states[0, 1:-1, :]  # (seq_len, 480)
            # Mean-pool across all residues to get fixed (480,) shape
            mean_pooled = seq_hidden.mean(dim=0)  # (480,)

            # Convert to numpy and ensure correct shape
            embedding = mean_pooled.cpu().numpy().astype(np.float32)
            assert embedding.shape == (ESM2_HIDDEN_DIM,), \
                f"Expected shape ({ESM2_HIDDEN_DIM},) but got {embedding.shape}"

            return embedding

        except Exception as e:
            log.warning("ESM-2 inference failed: %s — falling back to mock", e)
            return mock_embedding(sequence, ESM2_HIDDEN_DIM)

    def embed_antibody(self, vh_seq: str, vl_seq: str) -> np.ndarray:
        """
        Generate antibody embedding by concatenating VH + VL embeddings.

        Parameters
        ----------
        vh_seq : Heavy chain (variable region) sequence
        vl_seq : Light chain (variable region) sequence

        Returns
        -------
        np.ndarray of shape (960,) — [VH_embed (480,) | VL_embed (480,)]
            Always mean-pooled to fixed-size vector.
        """
        vh_embed = self.embed_sequence(vh_seq)
        vl_embed = self.embed_sequence(vl_seq)

        # Ensure both embeddings are (480,)
        assert vh_embed.shape == (ESM2_HIDDEN_DIM,), \
            f"VH embed shape mismatch: expected ({ESM2_HIDDEN_DIM},) got {vh_embed.shape}"
        assert vl_embed.shape == (ESM2_HIDDEN_DIM,), \
            f"VL embed shape mismatch: expected ({ESM2_HIDDEN_DIM},) got {vl_embed.shape}"

        result = np.concatenate([vh_embed, vl_embed]).astype(np.float32)
        assert result.shape == (ANTIBODY_EMBED_DIM,), \
            f"Antibody embed shape mismatch: expected ({ANTIBODY_EMBED_DIM},) got {result.shape}"

        return result

    def embed_from_intent(self, intent: Dict[str, Any]) -> np.ndarray:
        """
        Extract embedding from a parsed intent dictionary.

        Attempts to separate VH/VL from chain_analyses. Falls back
        to embedding the combined sequence if chains are not separated.

        Parameters
        ----------
        intent : Parsed intent dict from app.py

        Returns
        -------
        np.ndarray of shape (960,) — antibody embedding (VH 480 + VL 480)
        """
        chain_analyses = intent.get("chain_analyses", [])
        chains = intent.get("chains", [])

        vh_seq = None
        vl_seq = None

        # Try to find HC/LC from chains
        for chain in chains:
            ct = chain.get("chain_type", "").upper()
            seq = chain.get("sequence", "")
            if ct in ("HC", "HEAVY") and seq:
                vh_seq = seq
            elif ct in ("LC", "LIGHT") and seq:
                vl_seq = seq

        # If both chains found, use antibody embedding
        if vh_seq and vl_seq:
            return self.embed_antibody(vh_seq, vl_seq)

        # Fallback: use combined sequence
        combined = intent.get("sequence", "")
        if not combined and chains:
            combined = "".join(c.get("sequence", "") for c in chains)

        if combined:
            # Embed full sequence, then duplicate to fill 960-dim antibody slot
            single_embed = self.embed_sequence(combined)
            # Duplicate to fill both VH and VL slots
            return np.concatenate([single_embed, single_embed]).astype(np.float32)

        # No sequence available: return zeros
        log.warning("No sequence found in intent — returning zero embedding")
        return np.zeros(ANTIBODY_EMBED_DIM, dtype=np.float32)


# ===========================================================================
# Module-level convenience
# ===========================================================================

_CACHED_EMBEDDER: Optional[ESM2Embedder] = None


def get_embedder() -> ESM2Embedder:
    """Get (or create) the cached ESM-2 embedder instance."""
    global _CACHED_EMBEDDER
    if _CACHED_EMBEDDER is None:
        _CACHED_EMBEDDER = ESM2Embedder()
    return _CACHED_EMBEDDER


# ===========================================================================
# Flat convenience API (requested by M28 PLM upgrade)
# ===========================================================================

def get_sequence_embedding(
    sequence: str,
    model_name: str = "facebook/esm2_t12_35M_UR50D",
) -> List[float]:
    """
    Extract a single embedding vector from a protein sequence.

    Uses the cached ESM2Embedder.  Falls back to mock if unavailable.

    Parameters
    ----------
    sequence : Amino acid sequence (one-letter code).
    model_name : Hugging Face model identifier (used only on first init).

    Returns
    -------
    list[float] — 1-D embedding vector (480 dimensions for ESM2-t12).
    """
    embedder = get_embedder()
    arr = embedder.embed_sequence(sequence)
    return arr.tolist()


def get_batch_embeddings(
    sequences: List[str],
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    progress_cb=None,
) -> List[List[float]]:
    """
    Extract embeddings for a list of sequences.

    Parameters
    ----------
    sequences : list[str]
    model_name : Hugging Face model identifier (used only on first init).
    progress_cb : callable(current, total) — optional progress callback.

    Returns
    -------
    list[list[float]] — one embedding vector per sequence.
    """
    embedder = get_embedder()
    results = []
    total = len(sequences)
    for i, seq in enumerate(sequences):
        arr = embedder.embed_sequence(seq)
        results.append(arr.tolist())
        if progress_cb is not None:
            progress_cb(i + 1, total)
    return results


def get_embedding_dim(model_name: str = "facebook/esm2_t12_35M_UR50D") -> int:
    """Return the embedding dimensionality (480 for ESM2-t12, 960 for antibody pairs)."""
    return ESM2_HIDDEN_DIM


# ===========================================================================
# Chunked Batch Processing (M30 — UI unblocking)
# ===========================================================================

DEFAULT_CHUNK_SIZE = 10


def get_chunked_embeddings(
    sequences: List[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    progress_cb=None,
) -> List[List[float]]:
    """
    Extract embeddings for a list of sequences in chunks to prevent UI freezing.

    Processes sequences in batches of `chunk_size`, calling `progress_cb`
    after each chunk completes.  This enables real-time progress bar updates
    in Streamlit without blocking the event loop for the entire batch.

    Parameters
    ----------
    sequences : list[str]
        List of amino acid sequences to embed.
    chunk_size : int
        Number of sequences per processing chunk (default 10).
    model_name : str
        Hugging Face model identifier (used only on first init).
    progress_cb : callable(current: int, total: int, chunk_idx: int) or None
        Progress callback invoked after each chunk.  Receives:
          - current: number of sequences processed so far
          - total: total number of sequences
          - chunk_idx: 0-based chunk index

    Returns
    -------
    list[list[float]] — one embedding vector per input sequence.
    """
    embedder = get_embedder()
    total = len(sequences)
    results: List[List[float]] = []
    chunk_idx = 0

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = sequences[start:end]

        for seq in chunk:
            arr = embedder.embed_sequence(seq)
            results.append(arr.tolist())

        # Report progress after each chunk
        if progress_cb is not None:
            try:
                progress_cb(len(results), total, chunk_idx)
            except TypeError:
                # Fallback for 2-arg callbacks
                progress_cb(len(results), total)

        chunk_idx += 1

    return results


def get_chunked_antibody_embeddings(
    hc_sequences: List[str],
    lc_sequences: List[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress_cb=None,
) -> List[List[float]]:
    """
    Extract antibody (VH+VL) embeddings in chunks.

    Each antibody gets a 960-dim concatenated embedding (VH 480 + VL 480).
    Processes in chunks of `chunk_size` antibodies at a time.

    Parameters
    ----------
    hc_sequences : list[str] — Heavy chain sequences.
    lc_sequences : list[str] — Light chain sequences (same length as hc).
    chunk_size : int — Number of antibodies per chunk.
    progress_cb : callable(current, total, chunk_idx) or None.

    Returns
    -------
    list[list[float]] — one 960-dim embedding per antibody.
    """
    assert len(hc_sequences) == len(lc_sequences), \
        f"HC/LC length mismatch: {len(hc_sequences)} vs {len(lc_sequences)}"

    embedder = get_embedder()
    total = len(hc_sequences)
    results: List[List[float]] = []
    chunk_idx = 0

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)

        for i in range(start, end):
            arr = embedder.embed_antibody(hc_sequences[i], lc_sequences[i])
            results.append(arr.tolist())

        if progress_cb is not None:
            try:
                progress_cb(len(results), total, chunk_idx)
            except TypeError:
                progress_cb(len(results), total)

        chunk_idx += 1

    return results


# ===========================================================================
# __main__: Standalone Test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  ProtePilot — pLM Embedder v1.0 Test")
    print("=" * 60)

    embedder = ESM2Embedder()
    print(f"\nMode: {'MOCK' if embedder.is_mock else 'ESM-2'}")

    # Test single sequence
    test_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTR"
    emb = embedder.embed_sequence(test_seq)
    print(f"\nSingle sequence embedding: shape={emb.shape}, "
          f"mean={emb.mean():.4f}, std={emb.std():.4f}")

    # Test antibody embedding
    vh = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTR"
    vl = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKLLIY"
    ab_emb = embedder.embed_antibody(vh, vl)
    print(f"Antibody embedding: shape={ab_emb.shape}, "
          f"mean={ab_emb.mean():.4f}, std={ab_emb.std():.4f}")

    # Test intent-based
    intent = {
        "sequence": vh + vl,
        "chains": [
            {"chain_type": "HC", "sequence": vh},
            {"chain_type": "LC", "sequence": vl},
        ],
    }
    intent_emb = embedder.embed_from_intent(intent)
    print(f"Intent-based embedding: shape={intent_emb.shape}")

    # Test mock determinism
    emb1 = mock_embedding(test_seq)
    emb2 = mock_embedding(test_seq)
    assert np.allclose(emb1, emb2), "Mock embeddings should be deterministic!"
    print(f"\nMock determinism check: PASS")

    print("\npLM Embedder v1.0 test complete")
