# Training Guide — ProtePilot Platform

## Current Status: 训练已接通 (Training Connected)

The platform supports a complete training closed loop: harmonize data → train model → save artifact → load at inference → compare with baseline → selftest validation.

## How to Train

### Step 1: Harmonize Training Data

```bash
python -m src.training.data_harmonizer --output data/training/classifier_data.csv
```

This reads Jain-137 (137 canonical mAbs) + TheraSAbDab (1133 therapeutics) + synthetic peptides/nanobodies/scaffolds, maps them to 8 MoleculeClass labels, computes 11 biophysical features, and outputs a single CSV.

Output: `data/training/classifier_data.csv` (1322 rows, 8 classes, 16 columns).

### Step 2: Create Reproducible Split

```python
from src.training.schema import create_split
import pandas as pd

df = pd.read_csv("data/training/classifier_data.csv")
df_split = create_split(df, seed=42)  # train/val/test/holdout
df_split.to_csv("data/training/classifier_data_split.csv", index=False)
```

Split: train=930, val=193, test=193, holdout=6. Benchmark holdout (trastuzumab, bevacizumab, rituximab, NISTmAb) is never used for training.

### Step 3: Train Classifier

```bash
python -m src.training.classifier_trainer --data data/training/classifier_data.csv --output models/classifier --seed 42
```

This trains a logistic regression on the 11 biophysical features, saves the model artifact (.npz + metadata JSON), and prints metrics including comparison with the rule-based baseline.

### Step 4: Verify Training Results

```bash
# Run benchmark comparison (before/after on 7-molecule panel)
python -m src.training.benchmark_evaluator --artifact-dir models/classifier

# Run selftest (section 20 validates model artifact)
python SelfTest/run_validation.py

# Run pytest training layer
pytest -m "core or governance" -x
```

### Step 5: Use in Platform

After training, the model is automatically loaded by `classify_molecule()` as a "second opinion" that adjusts confidence without overriding the rule-based classification. No manual wiring needed.

## How the Integration Works

`classify_molecule()` uses a two-phase approach:

**Phase 1 — Rule-based:** Chain structure analysis, VH/VL motif detection, sequence length heuristics, name keywords. This is the authoritative classification.

**Phase 2 — Trained model:** Loaded lazily from `models/classifier/`. Provides a probability-based prediction. Integration strategy:

| Scenario | Action |
|----------|--------|
| Rule-based confidence is **High** | Trained model noted in evidence, no change |
| Both **agree** on class | Confidence **boosted** (Low→Medium or Medium→High) |
| They **disagree** | Warning added, rule-based class **retained** |
| No trained model exists | Rule-based only (graceful degradation) |

The trained model never silently overrides the rule-based classification. This is by design — the rule-based classifier uses structural chain information that the trained model cannot see (chain count, VH/VL pairing, assembly structure).

## How to Verify

### Evidence checklist

| Evidence | How to verify |
|----------|--------------|
| Training data exists | `ls data/training/classifier_data.csv` |
| Split is reproducible | `python -c "from src.training.schema import create_split; ..."` with same seed=42 |
| Model artifact exists | `ls models/classifier/classifier_model.npz` |
| Model is loadable | `python -c "from src.training.model_inference import load_classifier; print(load_classifier())"` |
| Inference works | `python -c "from src.training.model_inference import *; print(predict_class(load_classifier(), 'ACDEF'*20))"` |
| Inference is deterministic | Run predict_class 5× with same input → same output |
| Platform uses the model | Run `classify_molecule()` → evidence includes "Trained model" line |
| Selftest validates artifact | Section 20: 12 checks on artifact validity |
| Benchmark comparison | `python -m src.training.benchmark_evaluator` |

### Selftest sections related to training

| Section | What it checks |
|---------|---------------|
| **19** (Platform Alignment) | Registry ↔ enum match, risk weights, factory_reset clears state |
| **20** (Post-Training) | Artifact loadable, metadata schema, deterministic inference, benchmark drift |

## Training Roadmap

### Priority order for future training tasks

| Priority | Task | Why | Data availability |
|----------|------|-----|------------------|
| 1 | **Molecule classifier** | ✅ Done. XGBoost, 24 features, accuracy=0.983 | Jain-137 + TheraSAbDab + synthetic |
| 2 | **OOD detector** | ✅ Done. Ensemble Mahalanobis+IForest, test_f1=0.648 | 11 composition-ratio features, 480 synthetic OOD samples |
| 3 | **Liability weighting** | Core platform value — which liabilities matter most | Jain-137 has experimental Tm, HIC, ACSINS |
| 4 | **Validation recommendation ranking** | Improves report quality, good for rule+ML hybrid | Can bootstrap from existing selftest recommendations |
| 5 | **Developability calibration** | Calibrate composite scores against experimental data | Jain-137 experimental + TheraSAbDab clinical status |
| 6 | **ADA risk** | Currently heuristic-only | Limited public labeled data |
| 7 | **Upstream/downstream** | Requires process-specific data | Not publicly available |
| 8 | **Full developability end-to-end** | Last — needs all sub-models first | Aggregate of all above |

### OOD Detector (completed)

The OOD detector uses an ensemble of Mahalanobis distance + IsolationForest on 11 composition-ratio features (pI, GRAVY, hydrophobicity, aromatic/pro_gly/cys/small fractions, deam/ox densities, charge_ratio, aliphatic_idx). Key design decisions:

- **Composition ratios only**: Raw count features (seq_length, mw_kda, etc.) and chain structure features (hc_len_norm, etc.) are excluded because training data uses VH+VL variable regions while inference uses full-length HC+LC chains. Density/ratio features are invariant to this gap.
- **11 OOD types**: homo-repeats, extreme pI, biased random, cysteine-rich, sub-peptide, tandem repeats, extreme composition, extreme hydrophobic, proline/glycine-rich, extreme chimeric, biased dipeptide (480 total samples).
- **Metrics**: test_f1=0.648, precision=0.635, recall=0.662. Real mAbs (trastuzumab, etc.) are correctly identified as in-distribution.

### Next training task: Liability Weighting (Priority 3)

See `docs/RESEARCH_TRACK.md` for the research track modules that feed into priorities 3-8.

### Improving the classifier

Current limitations and how to address them:
- **Bispecific regression**: Trained model can't see chain count → add `n_unique_chains` as a feature
- **Minority class F1=0**: adc/fc_fusion/fusion_protein have <60 samples → collect more data or use data augmentation
- **Training-inference feature gap**: Training data uses VH+VL variable regions; inference uses full HC+LC. This affects ML model accuracy and OOD detection. Long-term fix: re-harmonize with full-length sequences or add variable region extraction at inference.

### Research Track

Modules not yet production-ready are documented in `docs/RESEARCH_TRACK.md`. These are isolated behind soft imports and do not block production functionality.
