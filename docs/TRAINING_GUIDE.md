# ProtePilot — Training & Retraining Guide

This guide covers every trainable component in the platform, how to run training from the CLI, and what to improve next.

**Last updated:** 2026-03-24 (post ESM-2 integration + Jain2024/Garbinski merge)

---

## Quick Start — Full Pipeline Retrain (~30 seconds)

```bash
cd /path/to/ProtePilot
PYTHONPATH=. python -m src.training.pipeline --harmonize --train --ood --benchmark
```

This runs all four stages in sequence:
1. **Harmonize** — load public databases → unified CSV
2. **Train** — XGBoost classifier on harmonized data
3. **OOD** — out-of-distribution detector (Mahalanobis + IsolationForest)
4. **Benchmark** — compare trained model vs rule-based baseline

---

## Component 1: Molecule Classifier

**Current status:** Production-ready (98.3% accuracy, 8 classes, 12/12 post-training checks)

### What it is
XGBoost model that classifies protein sequences into: canonical_mab, bispecific, adc, fc_fusion, single_domain, peptide, fusion_protein, engineered_scaffold.

### Artifacts
```
models/classifier/
├── classifier_xgboost.json    # Trained model (3.9 MB)
├── classifier_metadata.json   # Training metrics, feature list, timestamp
└── classifier_model.npz       # Scaler/fallback
```

### Training command
```bash
# Step 1: Harmonize data (loads 6+ public sources → one CSV)
PYTHONPATH=. python -c "
from src.training.data_harmonizer import harmonize_all_sources
harmonize_all_sources(output_path='data/training/classifier_data.csv')
"

# Step 2: Train classifier
PYTHONPATH=. python -c "
from src.training.classifier_trainer import train_and_save_classifier
result = train_and_save_classifier(
    data_path='data/training/classifier_data.csv',
    output_dir='models/classifier',
    model_type='xgboost',   # or 'logistic_regression'
)
print(f'Accuracy: {result[\"test_accuracy\"]:.4f}')
print(f'F1 macro: {result[\"test_f1_macro\"]:.4f}')
"
```

### How to improve
- **Add more data:** Place new CSVs in `data/training/` with columns: `name`, `molecule_class`, `hc_sequence`, `lc_sequence` (or `sequence` for single-chain). The harmonizer will auto-detect and process them.
- **Class imbalance:** Currently 50.7% canonical_mab, only 0.1% engineered_scaffold. Adding more bispecific, ADC, Fc-fusion, and scaffold examples would improve macro-F1.
- **Feature engineering:** The 24 biophysical features are in `src/training/features.py`. To add a new feature: add computation in `compute_features()`, add column name to `FEATURE_COLS` in `schema.py`, retrain.

---

## Component 2: OOD Detector

**Current status:** Production-ready (64.8% F1 on synthetic OOD panel)

### What it is
Ensemble of Mahalanobis distance + IsolationForest. Flags sequences that don't look like any known molecule class.

### Artifacts
```
models/ood_detector/
├── ood_detector.npz           # Global mean/covariance/threshold
├── ood_class_*.npz            # Per-class Mahalanobis (8 files)
├── iforest_model.joblib       # IsolationForest ensemble
└── ood_metadata.json          # Thresholds, metrics
```

### Training command
```bash
PYTHONPATH=. python -c "
from src.training.ood_trainer import train_ood_detector
result = train_ood_detector(
    data_path='data/training/classifier_data.csv',
    output_dir='models/ood_detector',
)
print(f'Mahalanobis threshold: {result[\"mahal_threshold\"]:.2f}')
print(f'Test F1: {result[\"test_f1\"]:.3f}')
"
```

### How to improve
- **Real OOD examples:** Currently uses 480 synthetic OOD sequences. Adding real non-therapeutic sequences (e.g., bacterial proteins, random peptides from UniProt) would improve detection.
- **Ensemble voting:** Currently uses union (either detector flags = OOD). Could experiment with intersection (both must flag) for higher precision.

---

## Component 3: Developability Property Predictors

**Current status:** 4/6 targets passing quality gates (up from 3/7)

### What they are
Six XGBoost regressors predicting wet-lab biophysical properties. The training pipeline now automatically selects the best feature set per target from three options: ESM-2 embeddings (640-dim), biophysical features (7-dim), or hybrid (647-dim).

### Current performance

| Target | Pearson r | Status | Best Feature Set | n molecules | Data Sources |
|--------|-----------|--------|-----------------|-------------|--------------|
| FAB Tm (thermal) | 0.773 | PASS | ESM-2 (640d) | 259 | Jain2017 + Jain2024 + Garbinski |
| Titer (expression) | 0.640 | PASS | ESM-2 (640d) | 229 | Jain2017 + Jain2024 + Garbinski |
| HIC RT (hydrophobicity) | 0.550 | PASS | ESM-2 (640d) | 175 | Jain2017 + Garbinski |
| PSR (poly-specificity) | 0.304 | PASS | Biophys (7d) | 137 | Jain2017 |
| AC-SINS (self-interaction) | 0.214 | FAIL | Biophys (7d) | 175 | Jain2017 + Garbinski |
| Stability slope | 0.177 | FAIL | ESM-2 (640d) | 137 | Jain2017 |

**Key improvements from last round:**
- FAB Tm: 0.653 → **0.773** (+18%, ESM-2 embeddings + more data)
- Titer: 0.154 → **0.640** (+315%, ESM-2 + Jain2024/Garbinski data)
- HIC RT: 0.361 → **0.550** (+52%, ESM-2 embeddings)
- PSR: 0.203 → **0.304** (now PASS, biophysical features with more data)

### Artifacts
```
models/developability/
├── xgb_fab_tm.pkl             # FAB thermal stability
├── xgb_hic_rt.pkl             # Hydrophobic interaction chromatography
├── xgb_acsins.pkl             # Self-interaction (AC-SINS)
├── xgb_psr.pkl                # Poly-specificity reagent
├── xgb_smac_rt.pkl            # Size-exclusion / SMAC retention
├── xgb_stability_slope.pkl    # Accelerated stability
├── xgb_titer.pkl              # Expression titer
├── xgb_titer_merged.pkl       # Merged titer model
└── metadata.json              # Training metrics, feature selection, data sources
```

### Training command
```bash
# Retrain developability predictors (uses ESM-2 + biophysical auto-selection)
PYTHONPATH=. python -c "
from src.developability_predictor import retrain_all_targets
results = retrain_all_targets(
    data_path='data/Jain137_Cleaned_Training_Data.csv',
    output_dir='models/developability',
)
for target, metrics in results.items():
    status = 'PASS' if metrics['r'] > 0.25 else 'FAIL'
    print(f'  [{status}] {target}: r={metrics[\"r\"]:.3f}, RMSE={metrics[\"rmse\"]:.3f}')
"
```

### How to improve the 2 remaining FAIL targets

**AC-SINS (r=0.214):** Self-interaction is an inherently noisy assay with high measurement variability. ESM-2 embeddings actually performed *worse* than biophysical features for this target (0.171 vs 0.214), suggesting sequence-level patterns may not capture the colloidal interactions that AC-SINS measures. Options:
- More AC-SINS measurements (currently only 175 molecules)
- Consider this target as "requires experimental measurement" and document it as such
- Structural features (3D surface hydrophobicity patches) might help, but require AlphaFold structures

**Stability slope (r=0.177):** Accelerated stability is measured over weeks/months and reflects complex degradation pathways. Only 137 molecules have this measurement. Options:
- More stability data (this is the scarcest target)
- Published datasets like Mason et al. 2021 may have relevant thermal stability kinetics
- Raybould et al. 2019 SAbDab annotations could supplement

### Adding new training data
```bash
# Place CSV in data/ with columns matching schema:
#   name, Sequence_HC, Sequence_LC, target_columns...

# Register in data_harmonizer.py or pass directly:
PYTHONPATH=. python -c "
from src.training.data_harmonizer import harmonize_custom_csv
harmonize_custom_csv(
    input_path='data/new_dataset.csv',
    output_path='data/training/classifier_data.csv',  # append
    molecule_class='canonical_mab',
)
"
```

---

## Component 4: Benchmark & Validation

### Run the full benchmark suite
```bash
# Quality gate (138 checks — contracts + benchmarks)
PYTHONPATH=. python run_all_checks.py

# Twin engine benchmarks (15 checks — physics model validation)
PYTHONPATH=. python src/twin_benchmark.py

# SelfTest validation (20 sections, 329 checks)
PYTHONPATH=. python SelfTest/run_validation.py

# Package selftests (Level-2 standalone)
python packages/pharma_classifier/selftest.py
python packages/pharma_harmonizer/selftest.py

# Package selftests (Level-1 wrappers)
PYTHONPATH=. python packages/pharma_features/selftest.py
PYTHONPATH=. python packages/pharma_training/selftest.py
PYTHONPATH=. python packages/pharma_assess/selftest.py
PYTHONPATH=. python packages/pharma_engines/selftest.py

# Package drift detection
python scripts/check_package_drift.py -v
```

### Latest benchmark results (2026-03-24, SelfTest v36)

| Suite | Result | Notes |
|-------|--------|-------|
| Pytest | 425/425 | 7.8s, zero failures |
| Quality Gate (run_all_checks.py) | 138/138 | 24.7s, all contract + benchmark checks |
| Twin Benchmarks (twin_benchmark.py) | 15/15 | Upstream, stability, immunogenicity, formulation, COGS |
| Post-Training Selftest | 12/12 | Classifier artifact validity, drift, determinism |
| pharma_classifier (Level-2) | 8/8 | Standalone package selftest |
| pharma_harmonizer (Level-2) | 10/10 | Standalone package selftest |
| Level-1 wrappers | N/A | Require `pip install -e packages/pharma_*` |
| Package Drift | 8/12 drifted | Expected after src/ edits; run `check_package_drift.py` to sync |
| NISTmAb Benchmark | 8/8 | All reference values within range |
| ML vs Heuristic (holdout) | ML wins 4/6 | Blind eval on bevacizumab, rituximab, trastuzumab |

### Post-training validation
After retraining any model, always run:
```bash
# Classifier-specific post-training selftest
PYTHONPATH=. python -c "
from src.training.benchmark_evaluator import post_training_selftest
checks = post_training_selftest('models/classifier')
for name, ok in checks.items():
    print(f'  [{\"PASS\" if ok else \"FAIL\"}] {name}')
print(f'Overall: {sum(checks.values())}/{len(checks)}')
"
```

---

## Component 5: ESM-2 Protein Language Model

**Current status:** Integrated and active in developability predictors

### What it is
ESM-2 (facebook/esm2_t6_8M_UR50D) provides 320-dim per-chain learned embeddings. For antibodies, VH (320) + VL (320) = 640-dim feature vector. The training pipeline tests all three feature sets (ESM-2, biophysical, hybrid) per target and picks the best.

### Architecture
```
ESM-2 Channel:  VH seq → ESM-2 t6_8M → mean-pool → (320,) ─┐
                VL seq → ESM-2 t6_8M → mean-pool → (320,)   ├→ concat (640,)
                                                                ↓
Biophysical:    [pI, MW, GRAVY, hydro, deam, ox, charge] → (7,) ─┐
                                                                    ├→ best-of-3 selection
                                                      ←─────────────
```

### Key files
- `src/esm2_hybrid_encoder.py` — encoder with graceful fallback to mock embeddings
- `src/pLM_embedder.py` — lower-level ESM-2 wrapper
- `src/developability_predictor.py` — uses auto-selection (ESM-2 vs biophys vs hybrid)

### Verify ESM-2 is working
```bash
python -c "
from transformers import AutoModel
AutoModel.from_pretrained('facebook/esm2_t6_8M_UR50D')
print('ESM-2 loaded successfully')
"

# If not installed:
pip install transformers torch --break-system-packages
```

### Impact
ESM-2 embeddings dramatically improved 3 of 6 targets (FAB Tm, Titer, HIC RT), where they outperformed biophysical features alone. For 2 targets (AC-SINS, PSR), biophysical features still perform better — likely because these properties depend on surface/colloidal interactions not captured by sequence-level language models.

---

## Component 6: Continuous Learning (Future)

The feedback infrastructure exists in `src/continuous_learning.py` and `src/training/feedback_store.py`:

```bash
# Record expert correction
PYTHONPATH=. python -c "
from src.training.feedback_store import FeedbackStore
store = FeedbackStore('data/feedback/')
store.record(
    molecule_name='MyMab_001',
    predicted_class='canonical_mab',
    corrected_class='bispecific',
    confidence=0.65,
    user='expert_1',
)
print(f'Total feedback: {store.count()}')
"

# Auto-retrain triggers when feedback count >= 5
```

---

## Training Data Sources (Current)

### Classifier training data (24,829 molecules)

| Source | Molecules | Type | Public? |
|--------|-----------|------|---------|
| CoV-AbDab | 12,346 | COVID antibodies | Yes (free) |
| DRAMP | 5,000 | Antimicrobial peptides | Yes (free) |
| ThPD | 5,000 | Therapeutic peptides | Yes (free) |
| TheraSAbDab | 1,287 | Therapeutic antibody DB | Yes (free) |
| SAbDab | 914 | PDB antibody structures | Yes (free) |
| Jain-137 | 137 | Clinical mAb biophysics | Published (Jain 2017) |
| Curated | 80 | Fc-fusion, scaffold | Curated from literature |
| Synthetic | 65 | Class-balancing | Generated |

### Developability training data (267 molecules with wet-lab measurements)

| Source | Molecules | Targets available | Public? |
|--------|-----------|-------------------|---------|
| Jain et al. 2017 | 137 | FAB Tm, HIC RT, AC-SINS, PSR, stability slope, titer | Published |
| Jain et al. 2024 | ~90 | FAB Tm, titer (updated measurements) | Published |
| Garbinski et al. 2023 | ~40 | FAB Tm, HIC RT, AC-SINS, titer | Published |

---

## Environment Setup

```bash
# Core requirements
pip install -r requirements.txt

# ML training requirements
pip install -r requirements_ml.txt

# ESM-2 protein language model (used by developability predictors)
pip install transformers torch

# Optional: CADET chromatography engine
# Download from: https://github.com/cadet/CADET-Core/releases
# Copy binary to engine/ and libs to lib/
```

---

## Summary: What to Train Next (Priority)

1. **Classifier + OOD** — production-ready, no action needed unless adding new molecule classes
2. **Developability: AC-SINS** — r=0.214 (FAIL). Needs more AC-SINS wet-lab measurements or structural features. Consider documenting as "experimental measurement required"
3. **Developability: Stability slope** — r=0.177 (FAIL). Scarcest target (n=137). Look for published accelerated stability datasets (Mason et al. 2021)
4. **Continuous learning loop** — infrastructure ready, needs real user feedback to activate
5. **Structural features** — AlphaFold-predicted surface patches could help AC-SINS and aggregation prediction, but requires significant new infrastructure
