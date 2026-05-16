# Retraining All Models — Step-by-Step Guide

## Overview

ProtePilot ships five independently retrainable models:

| # | Model | Script / Entry Point | Output |
|---|-------|----------------------|--------|
| 1 | Unified Multitask (PyTorch) | `scripts/train_unified_model.py` | `models/unified_multitask_best.pt` |
| 2 | XGBoost Developability (7 targets) | `scripts/retrain_devpred_esm2.py` | `models/developability/xgb_*.pkl` |
| 3 | Molecule Classifier | `src/training/classifier_trainer.py` | classifier artefacts |
| 4 | OOD Detector | `src/training/ood_trainer.py` | OOD artefacts |
| 5 | ESM-2 Embedding Cache (not a model per se, but required by 1 & 2) | `scripts/build_esm2_cache.py` | `data/esm2_embeddings_cache.pt` |

- **Total wall-clock time:** ~5-10 minutes on a modern CPU.
- **GPU required:** No. The ESM-2 variant used (`facebook/esm2_t12_35M_UR50D`) fits
  comfortably in CPU memory.
- All commands below assume you are in the **project root** (`ProtePilot/`).

---

## Prerequisites

Install the Python dependencies (a virtual environment is recommended):

```bash
pip install torch xgboost scikit-learn pandas numpy scipy
```

Verify the install:

```bash
python3 -c "import torch, xgboost, sklearn; print('OK')"
```

---

## Step 1 — Rebuild Unified Training Data

Merge all source datasets into a single training file.

```bash
python3 scripts/build_integrated_training_data.py
```

**Inputs:**

- `data/Jain137_Cleaned_Training_Data.csv`
- `data/merged_wetlab_training.csv`
- `data/external/prophet_ab.csv`
- `data/reference/benchmark_sequences.json`

**Output:**

- `data/unified_training_data.csv` (expected: 381 rows)

Spot-check the result:

```bash
head -5 data/unified_training_data.csv
wc -l  data/unified_training_data.csv   # expect 382 (381 + header)
```

---

## Step 2 — Rebuild ESM-2 Embedding Cache

Generate per-sequence embeddings from the ESM-2 t12 model.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/build_esm2_cache.py
```

> **Note:** `KMP_DUPLICATE_LIB_OK=TRUE` silences a harmless OpenMP warning on
> macOS when both PyTorch and NumPy ship their own libiomp.

**Output:**

- `data/esm2_embeddings_cache.pt` (~1.4 MB, tensor shape 381 x 960)

Verify:

```bash
python3 -c "
import torch
cache = torch.load('data/esm2_embeddings_cache.pt', weights_only=False)
print(type(cache), len(cache) if isinstance(cache, dict) else cache.shape)
"
```

---

## Step 3 — Retrain Unified Multitask Model

Train the PyTorch multitask network that predicts multiple endpoints at once.

```bash
PYTHONPATH=. python3 scripts/train_unified_model.py \
    --epochs 100 \
    --lr 0.001 \
    --batch 16 \
    --cache data/esm2_embeddings_cache.pt
```

**Output:**

- `models/unified_multitask_best.pt`

The script prints validation metrics each epoch; final-epoch metrics are the ones
that matter.

---

## Step 4 — Retrain XGBoost Developability Models

Train seven independent XGBoost regressors (one per developability target).

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/retrain_devpred_esm2.py
```

**Input:**

- `data/merged_xgb_training.csv` (370 rows)

**Output:**

- `models/developability/xgb_*.pkl` (7 pickle files)

List the produced artefacts:

```bash
ls -lh models/developability/xgb_*.pkl
```

---

## Step 5 — Retrain Molecule Classifier

Harmonize the classification data and retrain the classifier in one shot:

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from training.data_harmonizer import harmonize
harmonize(data_dir='data', output_path='data/training/classifier_data.csv')
from training.classifier_trainer import train_classifier
train_classifier()
"
```

---

## Step 6 — Retrain OOD Detector

Rebuild the out-of-distribution detector so it reflects the current training
manifold:

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from training.ood_trainer import train_ood_detector
train_ood_detector()
"
```

---

## Step 7 — Verify

Run the full test suite to confirm nothing is broken:

```bash
python3 -m pytest tests/ -v --tb=short
```

All tests should pass. Pay special attention to any test that loads a model
artefact — a shape mismatch or missing file indicates a retraining step was
skipped or failed.

---

## Quick "Retrain Everything" One-Liner

If you just want to run every step back-to-back:

```bash
python3 scripts/build_integrated_training_data.py && \
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/build_esm2_cache.py && \
PYTHONPATH=. python3 scripts/train_unified_model.py --epochs 100 --lr 0.001 --batch 16 --cache data/esm2_embeddings_cache.pt && \
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/retrain_devpred_esm2.py && \
python3 -c "
import sys; sys.path.insert(0, 'src')
from training.data_harmonizer import harmonize
harmonize(data_dir='data', output_path='data/training/classifier_data.csv')
from training.classifier_trainer import train_classifier
train_classifier()
from training.ood_trainer import train_ood_detector
train_ood_detector()
" && \
python3 -m pytest tests/ -v --tb=short
```

---

## Adding New Training Data

To incorporate new molecules into the retraining pipeline:

1. **Append rows** to the appropriate source CSV:
   - For multitask / unified model data: add rows to
     `data/merged_wetlab_training.csv` (must include sequence and target columns).
   - For developability targets: add rows to `data/merged_xgb_training.csv`.

2. **Re-run from Step 1.** The build script will pick up the new rows
   automatically and regenerate `data/unified_training_data.csv`.

3. **Rebuild the embedding cache (Step 2).** New sequences need ESM-2 embeddings
   before any downstream model can use them.

4. **Retrain the relevant model(s)** (Steps 3-6). You only need to retrain
   the models whose training data actually changed — but when in doubt, retrain
   everything.

5. **Run the test suite (Step 7)** to confirm the updated models still pass all
   assertions.

> **Tip:** Keep a changelog entry whenever you add training data so that model
> versions stay traceable.
