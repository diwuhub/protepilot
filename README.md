# ProtePilot — Developability Decision Orchestrator

![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)

A decision orchestrator for therapeutic protein developability. Routes a candidate sequence through specialized predictors — aggregation, stability, immunogenicity, manufacturability, process economics — and emits an integrated go/no-go verdict with per-dimension confidence scores. Covers 10 molecule formats from canonical mAbs to bispecifics, nanobodies, ADCs, and engineered scaffolds.

**Platform structure: one integrated system + two independently usable modules.**

| Component | What it does | Status |
|-----------|-------------|--------|
| **ProtePilot Platform** | Integrated workflow: sequence → classification → digital twins → risk report | 138 quality gate checks, all passing |
| **[pharma-classifier](packages/pharma_classifier/)** | Molecule-aware classification + out-of-distribution detection | Standalone Level-2 package, 8/8 selftest |
| **[pharma-harmonizer](packages/pharma_harmonizer/)** | Training data harmonization + ML pipeline | Standalone Level-2 package, 10/10 selftest |

## Key results

| Benchmark | Metric | Value |
|-----------|--------|-------|
| PROPHET-Ab (246 antibodies) | Runtime | 4.5 seconds |
| PROPHET-Ab | Significant correlations | 8/13 (13/13 correct direction) |
| PROPHET-Ab | Best: CE-SDS vs SEC monomer | rho = 0.999 |
| PROPHET-Ab | Shelf life vs Tm | rho = 0.624 |
| XGBoost classifier | Test accuracy | 0.983 |
| IEDB MHC-II validation | ROC AUC | 0.699 |
| Quality gate | Checks passing | 138/138 |
| Test suite | pytest tests | 524+ |
| Molecule formats | Supported | 10 |


## Quick Start

```bash
# Layer 1 — headless pipeline (no UI, no ML, heuristic predictions)
pip install -r requirements-core.txt
python3 SelfTest/run_validation.py        # 18 sections, 524+ checks

# Layer 2 — full Streamlit UI (interactive dashboards, 3D viewer)
pip install -r requirements-analysis.txt
streamlit run app.py

# Layer 3 — ML training (ESM-2, XGBoost, SHAP, model training)
pip install -r requirements-training.txt
```

## Three-Layer Dependency Architecture

The platform uses a layered dependency model so you install only what you need.

### Layer 1 — Core Runtime (`requirements-core.txt`)

The minimal install. Runs the entire analysis pipeline headlessly, generates DOCX/PDF reports, processes bulk CSV batches, and passes all validation tests. All developability predictions use calibrated heuristic scoring (no ML models required).

**What works:** PropertyMapper, DevelopabilityCore, BispecificEngine, ReportAssembler, Bulk Analysis (CSV to ranked candidates), CADET chromatography simulation, SelfTest (18 sections), pytest -m core (477 tests).

**Packages:** numpy, pandas, biopython, plotly, matplotlib, kaleido, python-docx, h5py, requests, pytest.

```bash
pip install -r requirements-core.txt
```

### Layer 2 — Analysis Runtime (`requirements-analysis.txt`)

Adds the Streamlit web application with interactive dashboards, 3D protein visualization, and LLM-powered copilot. Predictions remain heuristic-based at this layer.

**What this adds:** Streamlit UI (app.py), 3D structure viewer (py3Dmol), AI CoPilot and Advisory Board (openai), Pareto frontier visualization, DoE interactive explorer.

**Packages:** Layer 1 + streamlit, py3Dmol, stmol, openai.

```bash
pip install -r requirements-analysis.txt
streamlit run app.py
```

### Layer 3 — Training Runtime (`requirements-training.txt`)

Adds ML model training and inference infrastructure. Replaces heuristic predictions with trained ESM-2 + XGBoost models when available.

**What this adds:** ESM-2 t12 (35M parameter) protein language model embeddings, XGBoost trained developability predictors (real experimental training data), SHAP explainability, AI Training Center (unified_trainer, multitask models), uncertainty quantification (ensemble + MC-dropout).

**Packages:** Layer 2 + torch, xgboost, scikit-learn, joblib, shap, transformers, sentencepiece.

```bash
pip install -r requirements-training.txt
```

> **Note:** torch is ~2 GB. For CPU-only (no GPU): `pip install torch --index-url https://download.pytorch.org/whl/cpu` before installing requirements-training.txt.

### Graceful Degradation

Every optional dependency degrades gracefully. If torch/xgboost aren't installed, the system falls back to calibrated heuristic scoring. If streamlit isn't installed, you can still run bulk analysis and generate reports from the command line. If openai isn't installed, the CoPilot returns mock responses. The platform never crashes due to a missing optional package.

Check your current installation status:

```bash
python3 src/optional_deps.py
```

## Architecture Overview

For a deployment-focused view, including run modes, validation boundaries, and failure modes, see [docs/architecture/DEPLOYMENT_ARCHITECTURE.md](docs/architecture/DEPLOYMENT_ARCHITECTURE.md).

```
app.py                          Streamlit UI (Layer 2)
  |
  +-- src/agents.py             Pipeline orchestrator
  |     +-- src/property_mapper.py       Biophysical feature extraction
  |     +-- src/molecule_classifier.py   10-class molecule classification
  |     +-- src/developability_core.py   5-dimension composite scoring
  |     +-- src/bispecific_engine.py      Species resolution & pI (bispecifics)
  |     +-- src/ml_predictor.py          XGBoost / heuristic prediction (Layer 3)
  |     +-- src/pLM_embedder.py          ESM-2 t12 (35M) embeddings (Layer 3)
  |     +-- src/report_assembler.py      Report generation (DOCX/PDF)
  |
  +-- src/bulk_schema.py        Bulk CSV parsing (10 molecule formats)
  +-- src/bulk_runner.py        Batch execution engine
  +-- src/bulk_summary.py       CSV/JSON/DOCX export & ranking
  |
  +-- src/analytical_qc_twin.py          cIEF, CE-SDS, glycan profiling
  +-- src/stability_twin.py              ICH shelf-life prediction
  +-- src/upstream_twin.py               Fed-batch titer simulation
  +-- src/immunogenicity_twin.py         T-cell epitope & ADA risk
  +-- src/preclinical_twin.py            PK clearance & half-life
  +-- src/formulation_twin.py            pH, excipient, freeze-thaw screening
  +-- src/cadet_engine.py                CADET CEX chromatography simulation
  +-- src/llm_tool_demo.py               Guardrailed LLM tool-calling demo
  +-- src/doe_twin.py                    Design of Experiments optimizer
```

## Supported Molecule Formats

The platform classifies and analyzes 10 molecule types:

| Format | Bulk CSV Columns | Chain Assembly |
|--------|-----------------|----------------|
| Canonical mAb | HC, LC | 2×HC + 2×LC |
| Bispecific (4-chain) | HC1, LC1, HC2, LC2 | HC1+LC1 + HC2+LC2 |
| Bispecific (3-chain) | HC, LC, scfv_arm | HC+LC + scFv |
| Bispecific (2-chain) | chain_a, chain_b | A + B |
| scFv | scfv | 1×scFv |
| Nanobody / VHH | vhh | 1×VHH |
| Fc-Fusion | fc_fusion | 1×fusion |
| Peptide | peptide | 1×peptide |
| ADC | hc, lc | 2×HC + 2×LC + payload |
| Fusion Protein | fusion | 1×fusion |

## Testing

```bash
# Run all core unit tests (Layer 1 environment — safest, fastest)
pytest -m core

# Run bulk analysis tests only
pytest -m bulk

# Run ML tests (Layer 3 environment)
pytest -m ml

# Run the standalone SelfTest validation suite
python3 SelfTest/run_validation.py
```

## LLM Tool-Calling Demo

ProtePilot includes an offline-safe copilot demo. It shows an LLM-style planner calling deterministic ProtePilot tools, then generating a cited answer from tool evidence rather than inventing scientific conclusions.

```bash
# Markdown demo report
python scripts/run_llm_tool_demo.py

# JSON trace for local inspection
python scripts/run_llm_tool_demo.py --json

# Save a demo artifact
python scripts/run_llm_tool_demo.py --out demo_outputs/protepilot_llm_tool_demo.md
```

The default mode is `mock`, so it runs without API keys. Optional OpenAI-backed tool selection is available with `OPENAI_API_KEY`, but execution remains constrained to an allow-list of deterministic tools. See [docs/architecture/LLM_TOOL_CALLING_DEMO.md](docs/architecture/LLM_TOOL_CALLING_DEMO.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full testing guide, code style rules, and PR checklist.

## Monorepo Packages

### Level-2 Standalone Packages (own source code, no monorepo dependency)

These can be installed and used independently — suitable for external integration, CI/CD, or standalone demos.

| Package | Description | Selftest | README |
|---------|-------------|----------|--------|
| [`pharma-classifier`](packages/pharma_classifier/) | Feature computation, OOD detection, molecule classification inference | 8/8 | [README](packages/pharma_classifier/README.md) |
| [`pharma-harmonizer`](packages/pharma_harmonizer/) | Data harmonization, classifier training, benchmark evaluation | 10/10 | [README](packages/pharma_harmonizer/README.md) |

```bash
# Install standalone (no monorepo needed)
pip install packages/pharma_classifier
pip install packages/pharma_harmonizer

# Run standalone selftests
python packages/pharma_classifier/selftest.py
python packages/pharma_harmonizer/selftest.py
```

### Level-1 Wrapper Packages (re-export from monorepo `src/`)

These require the monorepo on `PYTHONPATH`. They provide clean import interfaces for development within the integrated platform.

| Package | Description | Selftest |
|---------|-------------|----------|
| [`pharma-features`](packages/pharma_features/) | Biophysical feature computation (24-feature vector) | 12/12 |
| [`pharma-training`](packages/pharma_training/) | ML training pipeline (XGBoost + OOD) | 10/10 |
| [`pharma-assess`](packages/pharma_assess/) | Developability assessment, report generation, validation planning | 13/13 |
| [`pharma-engines`](packages/pharma_engines/) | Digital twin engines (stability, upstream, immunogenicity, PK, purification, COGS) | 20/20 |

```bash
# Install Level-1 packages (development mode, requires monorepo)
PYTHONPATH=. pip install -e packages/pharma_features -e packages/pharma_training \
                         -e packages/pharma_assess -e packages/pharma_engines --no-deps

# Check for vendored copy drift between Level-2 packages and monorepo
python scripts/check_package_drift.py
```

## Classifier & Training Module

The molecule classifier is the platform's front gate. See [`src/CLASSIFIER_README.md`](src/CLASSIFIER_README.md) for full documentation.

| Component | Metric | Value |
|-----------|--------|-------|
| Rule-based classifier | Selftest | 13/13, Corpus 18/18 |
| XGBoost classifier | Test Accuracy | 0.983 |
| OOD detector | Test F1 | 0.648 |
| Benchmark (fusion mode) | Pass rate | 29/29 + 13/13 contract |

Training docs: [`docs/training/TRAINING_GUIDE.md`](docs/training/TRAINING_GUIDE.md) | Research track: [`docs/training/RESEARCH_TRACK.md`](docs/training/RESEARCH_TRACK.md)

## Quality Gate

```bash
python run_all_checks.py          # Full suite (138 checks)
python run_all_checks.py --fast   # Contracts + audit only
```

| Suite | Checks | Description |
|-------|--------|-------------|
| dependency_audit | 1 | Import layer violations, SSOT consistency |
| twin_contracts | 12 | Digital twin behavioral contracts |
| report_bulk_contracts | 15 | Report + bulk pipeline contracts |
| auxiliary_contracts | 17 | Auxiliary module contracts |
| infra_contracts | 17 | Infrastructure module contracts |
| medium_contracts | 14 | Medium module contracts (bispecific, regulatory, generative, HT, validation) |
| twin_benchmarks | 15 | Twin engine edge-case benchmarks |
| report_bulk_benchmarks | 15 | Report + bulk benchmarks |
| auxiliary_benchmarks | 14 | Auxiliary module benchmarks |
| infra_benchmarks | 8 | Infrastructure module benchmarks |
| medium_benchmarks | 10 | Medium module edge-case benchmarks |
| **Total** | **138** | **11 suites, all passing** |

## Benchmarks

### PROPHET-Ab (246 Clinical-Stage Antibodies)

Validated against the Ginkgo PROPHET-Ab dataset (246 antibodies, 13 experimental assays):

| Prediction | Experimental Assay | Spearman rho | p-value | Significant |
|------------|-------------------|-------------|---------|-------------|
| CE-SDS Intact % | SEC Monomer % | **0.999** | < 1e-171 | Yes |
| shelf_life | Tm1 | **0.624** | 3.5e-22 | Yes |
| k_40c (degradation rate) | Tm1 | **-0.624** | 8.6e-17 | Yes |
| k_5c (storage rate) | SEC Monomer % | **-0.232** | 0.001 | Yes |
| adj_agg_risk | HIC retention | **+0.181** | 3.2e-04 | Yes |
| cIEF main % | Purity | 0.006 / r=0.14 | 0.029 | Yes |
| adj_agg_risk | AC-SINS pH7.4 | -0.098 | 0.033 | Yes |
| ada_risk_score | Polyreactivity (OVA) | **+0.140** | 0.034 | Yes |

**8/13 significant correlations, 13/13 correct direction.**

```bash
python benchmarks/prophet_ab_benchmark.py    # Re-run benchmark
```

### PROPERMAB Comparison (vs Regeneron)

Head-to-head comparison with PROPERMAB (Li et al. mAbs 2025):

| Dimension | PROPERMAB | ProtePilot |
|-----------|----------|------------|
| HIC prediction | Direct (ML, structure-based) | Indirect (hydrophobicity) |
| Viscosity | Direct | Modifier-based |
| Stability (Tm) | No | **Yes** (rho=-0.624) |
| Aggregation (SEC) | Indirect | **Yes** (rho=0.999) |
| Immunogenicity | No | **Yes** (IEDB-calibrated, AUC=0.70) |
| Glycan profile | No | **Yes** (G0F/G1F/G2F) |
| Speed (246 mAbs) | ~30 min | **4 seconds** |

### MHC-II Immunogenicity (IEDB Validation)

Validated against 5,000 IEDB human MHC-II T-cell epitopes:
- **ROC AUC = 0.699** (IEDB-calibrated log-odds matrix)
- Precision@50 = 0.920
- 129 germline V-genes (IMGT GENE-DB)

## ProteLoop Integration

ProtePilot connects to [ProteLoop](https://github.com/dwu-protepilot/proteloop) for autonomous parameter optimization:

```bash
# Export risk flags from ProtePilot assessment
python adapters/proteloop_export.py molecule_result.json > risk_flags.json

# ProteLoop prioritizes optimization loops based on risk
python scripts/protepilot_import.py risk_flags.json
```

High aggregation risk triggers formulation optimization. High immunogenicity triggers peptide mapping. The adapter maps 8 risk dimensions to 14 ProteLoop loops.

## Project Stats

- 98 source modules in `src/` across 4 tiers (A-D maturity)
- 18 test files, 524+ pytest tests across 4 layers (core, governance, bulk, training)
- 20 SelfTest validation sections, 308 substantive checks
- 138 quality gate checks across 11 contract/benchmark suites
- 29 classifier benchmark entries across 3 modes (rule, ML, fusion)
- 10 molecule formats supported in bulk analysis
- 59-field schema alignment contract between single and bulk paths
- 9 molecule classes in central configuration registry
- 6 independently versioned packages in `packages/` (4 Level-1 wrappers + 2 Level-2 standalone)
- 3 benchmark suites (PROPHET-Ab, PROPERMAB comparison, IEDB validation)

## Citation

    @software{wu2026protepilot,
      author = {Wu, Di},
      title  = {ProtePilot: Developability Decision Orchestrator for Therapeutic Proteins},
      year   = {2026},
      url    = {https://github.com/diwuhub/protepilot}
    }

## Known Limitations

ProtePilot is an **educator + prediction tool**, not a production decision system. Key limitations:

- **Training data**: XGBoost models trained on ~370 public antibodies. Predictions reflect ranking correlation (Spearman rho), not absolute values.
- **stability_slope**: Currently unreliable (rho=0.11). Only 137 training samples with compressed value range.
- **Digital twins**: Rule-based heuristics calibrated to public data. Educational tools for exploring developability dimensions.
- **Molecule classification**: VH+VL variable regions only — full HC+LC sequences needed for accurate pI and glycan prediction.
- **ESM-2 t12**: 35M parameter model provides good sequence representations but is not the largest available variant.

Always validate predictions with experimental data before making development decisions. See `docs/USER_GUIDE.md` for score interpretation.
