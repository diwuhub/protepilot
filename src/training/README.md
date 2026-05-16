# ProtePilot — Classifier / OOD Training Module

Molecule classification and out-of-distribution detection for the ProtePilot biologics developability platform.

## Overview

This module classifies biologic molecules into 8 therapeutic categories and detects sequences that fall outside the training distribution. It is the platform's **front gate**: every downstream analysis reads the classification result to determine routing, risk weights, and validation strategy.

**Supported classes:** canonical_mab, bispecific, adc, fc_fusion, single_domain, peptide, fusion_protein, engineered_scaffold

## Architecture

```
Raw Data Sources                    Training Pipeline                     Production Inference
─────────────────                   ─────────────────                     ────────────────────
Jain137 (137 mAbs)            ┌──→  data_harmonizer.py  ──→ classifier_data.csv
TheraSAbDab (1287 entries)    │       ↓                        ↓
CoV-AbDab (12346 entries)     │  classifier_trainer.py     ood_trainer.py
SAbDab PDB (914 seqs)         │       ↓                        ↓
DRAMP + ThPD (10000 peptides) │  XGBoost model             Ensemble OOD
Curated CSVs                  │  (24 features, 8 classes)  (Mahalanobis + IForest)
                              │       ↓                        ↓
                              └──  models/MANIFEST.json    models/ood_detector/
                                     ↓
                              model_inference.py  ←──  molecule_classifier.py
                                                         (rule-based + ML)
```

## Quick Start

```bash
# Full pipeline (harmonize → train → ood → benchmark → selftest)
python -m src.training.pipeline

# Individual steps
python -m src.training.pipeline --step harmonize
python -m src.training.pipeline --step train
python -m src.training.pipeline --step ood
python -m src.training.pipeline --step benchmark
python -m src.training.pipeline --step selftest

# Combine steps
python -m src.training.pipeline --step harmonize,train,ood
```

## Module Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Unified CLI entry point, artifact versioning, MANIFEST.json |
| `features.py` | **Canonical** feature computation (single source of truth for all 24 features) |
| `schema.py` | Feature column definitions, split management, validation |
| `data_harmonizer.py` | Merges 7+ data sources into a single training CSV |
| `classifier_trainer.py` | XGBoost training with class-balanced weights (LR/softmax fallback) |
| `ood_trainer.py` | Ensemble OOD: global Mahalanobis + IsolationForest + per-class Mahalanobis |
| `model_inference.py` | Load artifacts and run inference at platform level |
| `benchmark_evaluator.py` | Before/after comparison on fixed holdout panel |
| `feedback_store.py` | Record-only user corrections (append-only CSV, no auto-retrain) |
| `examples/` | Minimal runnable examples: predict, train, features |
| `CHANGELOG.md` | Version history |

## Feature Schema (v1, 24 features)

**Biophysical (12):** seq_length, n_chains, n_unique_chains, pI, mw_kda, gravy, hydrophobicity, deam_sites, ox_sites, cysteine_count, acidic_residues, basic_residues

**Composition (8):** aromatic_frac, pro_gly_frac, cys_frac, deam_density, ox_density, charge_ratio, small_frac, aliphatic_idx

**Chain-level (4):** hc_frac, has_lc, hc_len_norm, lc_len_norm

## Classification Design

The classifier uses a **two-phase approach**:

1. **Rule-based classification** (authoritative): Sequence length, chain count, motif detection, and name hints determine the class. This is the ground truth — the trained model never overrides it.

2. **Trained model second opinion**: XGBoost adjusts confidence scores and flags disagreements. Key discriminators: `n_unique_chains` (21.7% feature importance — critical for bispecific detection), `has_lc` (separates single_domain from canonical_mab).

3. **Ensemble OOD detection**: Three signals with majority vote: global Mahalanobis distance, IsolationForest anomaly score, and per-class Mahalanobis. Flags novel molecules for human review.

## Current Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.983 |
| Test F1 Macro | 0.735 |
| Model | XGBoost (300 trees, depth=6) |
| Training rows | 24,829 (8 classes) |
| OOD Val F1 | 0.518 |

### Per-Class F1
| Class | F1 | Support |
|-------|-----|---------|
| canonical_mab | 0.984 | 12,591 |
| peptide | 0.999 | 10,030 |
| bispecific | 0.989 | 320 |
| single_domain | 0.915 | 1,684 |
| fc_fusion | 0.839 | 107 |
| engineered_scaffold | 1.000 | 15 |
| adc | 0.154 | 58 |
| fusion_protein | 0.000 | 24 |

## Artifact Versioning

Every pipeline run writes `models/MANIFEST.json`:

```json
{
  "pipeline_version": "1.0.0",
  "schema_version": "012225e7f3282c4a",
  "timestamp": "2026-03-21T06:22:43Z",
  "data_checksum": "a934a303...",
  "metrics": {
    "test_accuracy": 0.983,
    "test_f1_macro": 0.735,
    "model_type": "xgboost"
  }
}
```

The `schema_version` is a hash of the feature column list. If features are added or removed, the hash changes and the selftest will flag a mismatch, preventing stale artifacts from being used with incompatible code.

## Feedback System

User corrections are recorded to `data/feedback/feedback.csv` (append-only):

```csv
sequence_hash,predicted_class,corrected_class,confidence_score,timestamp,source,molecule_name
```

Design: record-only, no auto-retrain. Corrections are reviewed manually and selectively incorporated into the next harmonization round.

## Data Sources

| Source | Entries | Classes |
|--------|---------|---------|
| Jain137 | 137 | canonical_mab |
| TheraSAbDab | 1,287 | bispecific, adc, fc_fusion, fusion_protein, etc. |
| CoV-AbDab | 12,346 | canonical_mab, single_domain |
| SAbDab PDB | 914 | single_domain |
| DRAMP | 5,000 | peptide |
| ThPD | 5,000 | peptide |
| Curated (fc_fusion) | 80 | fc_fusion |
| Synthetic | 65 | peptide, single_domain, engineered_scaffold |

## Dependencies

**Required:** numpy, pandas, scikit-learn

**Optional (recommended):** xgboost (falls back to LogisticRegression → numpy softmax if unavailable), biopython (falls back to manual Kyte-Doolittle for pI/MW/GRAVY)

**Platform deps (soft, with fallbacks):** `src.types.MoleculeClass` (enum), `src.platform_config` (constants with defaults)

## Examples

```bash
# Compute features for a sequence
python -m src.training.examples.example_features

# Load model and predict
python -m src.training.examples.example_predict

# Run full training pipeline
python -m src.training.examples.example_train
```

## Known Limitations & Next Steps

- **adc** (F1=0.154) and **fusion_protein** (F1=0.000): insufficient real training data. Priority: source real ADC/fusion protein sequences from ChEMBL, UniProt, or literature.
- **CDR features**: not yet implemented. Will improve discrimination between antibody subtypes.
- **Fragment-level classification**: rule-based heuristic for Fab, scFv, etc. outside the current training package.
- **OOD test F1**: 0.447 — adequate for flagging but not definitive. Consider calibration against real novel submissions.
