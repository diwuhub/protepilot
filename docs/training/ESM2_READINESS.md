# ESM-2 Integration — Readiness Note

**Status:** Next-stage accelerator (not current priority)
**Date:** 2026-03-23

---

## What is ESM-2?

ESM-2 (Evolutionary Scale Modeling) is Meta AI's protein language model (Lin et al. 2023, Science). It generates per-residue and per-sequence embeddings from amino acid sequences, capturing evolutionary and structural information that hand-crafted features cannot.

Relevant model sizes for biologics:
- `esm2_t6_8M_UR50D` — 8M params, fast, suitable for real-time inference
- `esm2_t12_35M_UR50D` — 35M params, good balance of speed and accuracy
- `esm2_t33_650M_UR50D` — 650M params, highest accuracy, needs GPU

## What's Ready

### 1. Feature Pipeline Integration Point
- `src/training/features.py` has a clean `compute_all_features()` entry point
- Current: 24 hand-crafted biophysical features
- ESM-2 integration: append 320/480/1280-dim embedding to feature vector
- No architectural changes needed — just extend `FEATURE_COLS` in `schema.py`

### 2. Classifier Architecture
- Current XGBoost classifier accepts arbitrary feature vectors
- ESM-2 embeddings can be passed as additional columns in the training CSV
- Fallback: if ESM-2 inference fails, the 24 hand-crafted features still work
- OOD detector: Mahalanobis distance naturally extends to higher dimensions

### 3. Training Data
- Harmonized CSV already contains sequences (hc_sequence, lc_sequence)
- ESM-2 embeddings can be pre-computed during harmonization step
- Caching: embeddings are deterministic — compute once, store as .npy

### 4. Inference Path
- `model_inference.py` → `predict_class()` already takes `sequence` parameter
- On-the-fly ESM-2 embedding during inference (with cache)
- Graceful degradation: if torch/esm not installed, fall back to 24-feature mode

## What's Missing

### 1. ESM-2 Model Wrapper (estimated: 150-200 lines)
```python
# Proposed: src/training/esm_embedder.py
class ESMEmbedder:
    def __init__(self, model_name="esm2_t12_35M_UR50D"):
        self.model, self.alphabet = esm.pretrained.load(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()

    def embed(self, sequence: str) -> np.ndarray:
        """Return mean-pooled embedding vector."""
        ...

    def embed_batch(self, sequences: List[str]) -> np.ndarray:
        """Batch embedding for training data."""
        ...
```

### 2. Feature Schema Extension
- Add `esm_dim_0` through `esm_dim_N` to FEATURE_COLS
- Or: add a single `esm_embedding` column storing serialized vectors
- Decision needed: inline features vs. separate embedding file

### 3. Dependencies
- `torch>=2.0` (CPU or CUDA)
- `fair-esm>=2.0.0` (Meta's ESM package)
- These are heavy — must be optional deps (`[esm]` extra)

### 4. GPU/Performance Infrastructure
- Embedding 1 sequence: ~50ms (GPU) / ~500ms (CPU) with t12
- Embedding full training set (~2000 sequences): ~100s GPU / ~1000s CPU
- Need caching layer to avoid re-computing on every training run
- Consider: pre-compute embeddings in harmonization step, save to `data/esm_cache/`

### 5. Benchmark Validation
- Before/after comparison: 24-feature vs. 24+ESM features
- Expected improvement: 1-3% accuracy on edge cases (bispecific vs fc_fusion, fusion_protein vs engineered_scaffold)
- Must not degrade on canonical_mab / peptide (already ~99%)

## Promotion Criteria

ESM-2 should be promoted from "next-stage" to "active integration" when:

1. **Classifier accuracy plateau confirmed:** Current 24-feature XGBoost reaches ~98.3% accuracy. If accuracy gains stall below 99% on the benchmark holdout panel, ESM-2 embeddings are the next lever.

2. **GPU availability confirmed:** At least one training environment has CUDA available, or CPU inference latency is acceptable for the use case (< 2s per molecule for interactive, < 30min for full training set).

3. **Package independence achieved:** The pharma_classifier and pharma_harmonizer packages (Round C) are fully independent and stable, so ESM-2 can be added as an optional enhancement without destabilizing the core pipeline.

4. **User demand for fine-grained discrimination:** If the CDR-based immunogenicity scoring (Round B) or analytical discrimination is insufficient for specific programs, ESM-2 embeddings provide richer per-residue structural context.

## Estimated Effort

| Task | Lines | Time |
|------|-------|------|
| ESM embedder wrapper | 150-200 | 2-3 hours |
| Schema extension | 30-50 | 30 min |
| Feature pipeline integration | 50-80 | 1 hour |
| Caching layer | 100-150 | 1-2 hours |
| Benchmark comparison | 50-100 | 1 hour |
| Documentation | 50 | 30 min |
| **Total** | **~500 lines** | **~8 hours** |

## References

- Lin et al. 2023. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science 379(6637):1123-1130.
- Rives et al. 2021. "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences." PNAS 118(15).
- ESM GitHub: https://github.com/facebookresearch/esm
