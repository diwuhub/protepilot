# ProtePilot User Guide

Practical guide for running ProtePilot and interpreting its outputs.

---

## 1. Quick Start

### Prerequisites

- **Python 3.10+** (tested on 3.10 and 3.11)
- A working `pip` installation

### Installation

```bash
# Clone the repository
git clone <repo-url> && cd ProtePilot

# Install dependencies (three-layer system)
pip install -r requirements-core.txt    # Core: pandas, numpy, biopython, streamlit
pip install -r requirements-ml.txt      # ML layer: torch, fair-esm, xgboost, scikit-learn
pip install -r requirements.txt         # Full: adds CADET, optional analytics
```

The three-layer dependency system lets you run lighter configurations when you
do not need the full ML stack. Core is enough for analytical QC modules.
ML adds ESM-2 embeddings and XGBoost predictions. Full includes everything.

### Launch

```bash
streamlit run app.py
```

The app opens in your browser. Paste or upload a protein sequence, select the
molecule type, and click **Run Analysis**.

---

## 2. Understanding Your Results

### Risk Score System

Every module produces a score between 0 and 1. The traffic-light system:

| Range       | Color  | Meaning                                      |
|-------------|--------|----------------------------------------------|
| < 0.30      | Green  | **Low Risk** -- Safe operating range          |
| 0.30 - 0.60 | Yellow | **Medium Risk** -- Needs attention            |
| >= 0.60     | Red    | **High Risk** -- Investigate further          |

Scores are relative rankings within the training population, not absolute
experimental measurements. A score of 0.45 means the molecule sits in the
middle of the risk distribution, not that it has a 45% chance of failure.

### Module Outputs

#### 2.1 Developability Score

A composite score aggregating three axes of developability concern:

- **Aggregation risk** -- propensity for self-association
- **Stability** -- thermal and colloidal stability indicators
- **Viscosity** -- predicted high-concentration behavior

The score is derived from sequence-level features (charge patches, hydrophobic
surface area, net charge at pH 7.4) combined with ESM-2 learned embeddings.
A low score indicates a molecule that is likely straightforward to develop at
manufacturing scale.

#### 2.2 Immunogenicity Risk

Predicts the likelihood of anti-drug antibody (ADA) response in patients.

- **MHC-II epitope scanning**: Uses the IEDB scoring matrix to identify
  peptide segments that bind to common HLA-DR alleles.
- **Humanization scoring**: Compares the variable region against 129 IMGT
  human germline sequences. Higher similarity to germline reduces
  immunogenicity risk.
- **ADA score**: 0-1 output. Scores above 0.60 flag sequences with multiple
  predicted T-cell epitopes or low humanness.

#### 2.3 Stability Prediction

Estimates shelf life using Arrhenius kinetics extrapolation.

**Inputs:**
- Melting temperature (Tm)
- Sequence hydrophobicity
- Excipient conditions (pH, surfactant, buffer)

**Outputs:**
- **Shelf life (months)** -- predicted time to 5% monomer loss at 5 degrees C
- **k_5C** -- degradation rate constant at refrigerated storage
- **k_40C** -- degradation rate constant at accelerated (stress) conditions

The ratio k_40C / k_5C gives a sense of how temperature-sensitive the
molecule is. A large ratio means accelerated studies will overpredict
real-world degradation.

#### 2.4 Analytical QC

Predicts the analytical profile you would see in a real QC lab:

- **cIEF charge variants**: Predicted distribution of acidic, main, and basic
  peaks (reported as percentages). A high acidic fraction may indicate
  deamidation or glycation liabilities.
- **Glycan profile**: Predicted distribution of major N-linked glycoforms at
  the N297 site -- G0F (afucosylated high-mannose), G1F (mono-galactosylated),
  and G2F (fully galactosylated). Only meaningful for molecules with an Fc
  domain.
- **CE-SDS purity**: Predicted percent purity under reducing and non-reducing
  conditions.

#### 2.5 XGBoost Predictions

Seven property predictions trained on 370 real antibody molecules with ESM-2
t12 embeddings as features:

| Target           | What It Measures                                    |
|------------------|-----------------------------------------------------|
| `fab_tm`         | Fab melting temperature (thermal stability)          |
| `hic_rt`         | HIC retention time (hydrophobicity / self-interaction)|
| `acsins`         | AC-SINS score (self-interaction by nanoparticle spectroscopy)|
| `psr`            | Polyspecificity reagent score (off-target binding)   |
| `stability_slope`| Monomer loss rate over time                          |
| `titer`          | Expression titer in CHO cells                        |
| `smac_rt`        | Standup monoclonal antibody chromatography RT         |

Each prediction includes a confidence interval. Wider intervals mean the
molecule is further from the training distribution -- treat those predictions
with more caution.

---

## 3. Supported Molecule Types

| Molecule Type          | Training Examples | Notes                          |
|------------------------|------------------:|--------------------------------|
| `canonical_mab`        |            12,591 | Best-supported category        |
| `peptide`              |            10,030 | Sequence-only (no Fc modules)  |
| `single_domain`        |             1,684 | VHH / nanobodies               |
| `bispecific`           |               320 | Dual-target formats            |
| `adc`                  |               208 | Antibody-drug conjugates       |
| `fusion_protein`       |               124 | Non-Fc fusions                 |
| `fc_fusion`            |               107 | Fc-containing fusions          |
| `engineered_scaffold`  |                15 | Lowest data -- predictions are rough |

Predictions are most reliable for `canonical_mab` and `peptide` where
training data is abundant. For categories with fewer than 200 examples,
treat outputs as directional guidance rather than precise predictions.

---

## 4. Limitations and Honest Disclaimers

These are real constraints you should keep in mind:

1. **Small XGBoost training set.** The seven-target XGBoost model was trained
   on 370 antibody rows. This is a useful dataset but not a large one.
   Predictions will be less reliable for molecules that differ substantially
   from the training population.

2. **ESM-2 model size.** ProtePilot uses ESM-2 t12 (35M parameters). This is
   a capable protein language model, but it is not the largest available
   (ESM-2 t33 has 650M parameters). Embeddings may miss subtle structural
   features that a larger model would capture.

3. **VH-only pI is not intact mAb pI.** If you provide only the variable
   heavy chain sequence, the predicted isoelectric point reflects VH alone.
   Accurate intact-mAb pI requires the full heavy chain + light chain
   sequence.

4. **Relative rankings, not absolute values.** Scores indicate where a
   molecule falls within the training distribution. They do not directly
   correspond to experimental assay readouts.

5. **Glycan predictions require an Fc domain.** The glycan profile module
   models N297 glycosylation. It is not applicable to peptides, single-domain
   antibodies, or other formats lacking an Fc region.

6. **Stability shelf-life is extrapolated.** The Arrhenius model extrapolates
   from predicted degradation kinetics. Real-time stability studies remain
   necessary for regulatory submissions.

7. **cIEF calibration is narrow.** Charge variant predictions are calibrated
   against 20 reference mAbs, all IgG1 subclass. Accuracy may decrease for
   IgG2, IgG4, bispecifics, or non-antibody formats.

---

## 5. Retraining Models

To retrain the XGBoost models on new data or update ESM-2 embeddings, see
the step-by-step instructions in:

- **[docs/TRAINING_GUIDE.md](TRAINING_GUIDE.md)** -- full retraining workflow
- **[docs/training/ESM2_READINESS.md](training/ESM2_READINESS.md)** -- ESM-2 embedding generation
- **[docs/training/DATA_SOURCES.md](training/DATA_SOURCES.md)** -- training data provenance

---

## 6. Troubleshooting

### Import errors on startup

```
ModuleNotFoundError: No module named 'esm'
```

You need the ML layer. Run `pip install -r requirements-ml.txt`.

### CADET engine not found

```
CADETProcessError: Could not find CADET simulator
```

CADET is an optional dependency for chromatography simulation. Install via
the full requirements file or `pip install cadet-process`. On macOS you may
also need to install the CADET simulator binary separately.

### ESM-2 model loading is slow or fails

First load of ESM-2 downloads weights (~140 MB for t12). If you are behind a
firewall, pre-download the weights and set the `TORCH_HOME` environment
variable to point to the cache directory.

Out-of-memory errors on CPU are rare with t12 but can happen with very long
sequences (>1500 residues). Try truncating to the variable region.

### `types.py` collision (resolved)

Earlier versions included a file named `types.py` at the project root, which
shadowed Python's built-in `types` module and caused cryptic import failures.
This has been fixed by renaming the file. If you are on an older checkout,
pull the latest version or rename `types.py` to `molecule_types.py`.

### Streamlit port already in use

```bash
streamlit run app.py --server.port 8502
```

### Predictions seem unreasonable

- Verify you selected the correct molecule type.
- Check that the sequence is valid (standard amino acid letters, no headers).
- For multi-chain molecules, ensure chains are separated correctly.
- Review the confidence intervals -- wide intervals signal extrapolation.

---

## Further Reading

- [docs/PUBLIC_READY_BOUNDARY.md](PUBLIC_READY_BOUNDARY.md) -- what is and is not ready for external use
- [docs/training/RESEARCH_TRACK.md](training/RESEARCH_TRACK.md) -- ongoing research directions
- [docs/architecture/](architecture/) -- system design documentation
