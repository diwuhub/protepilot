# ProtePilot Module Specifications

> Version 1.0 · 2026-03-20
> This document defines the service boundaries, input/output contracts, and
> dependency graphs for the platform's core analysis modules.

---

## 1. Molecule Classifier + OOD Detector

**Files:** `src/molecule_classifier.py`, `src/ood_baseline.py`, `src/types.py`

### Purpose
Front-gate classification: determines the biologic format (canonical mAb,
bispecific, Fc-fusion, ADC, single-domain, peptide, fusion protein,
engineered scaffold, or unknown). Also detects out-of-distribution (OOD)
inputs that fall outside the training data manifold.

### Input Contract
```python
classify_molecule(
    sequence: str,                           # concatenated amino acid sequence
    chains: Optional[List[Dict]],            # [{"sequence": str, "chain_type": str, "copy_number": int}]
    name: str = "",                          # molecule name (used for keyword hints)
    user_hint: Optional[str] = None,         # explicit class override
    assembly_chains: Optional[List[Dict]] = None,  # pre-assembled chain info (bulk path)
)
```
- `chain_type` values: "HC", "LC", "scFv", "VHH", "scFv_Arm", "Fc", etc.
- Minimum sequence length: `MIN_SEQUENCE_LENGTH` (10 aa) from `platform_config`
- HC detection threshold: `MIN_HC_LENGTH` (200 aa) from `platform_config`

### Output Contract
```python
@dataclass
class ClassificationResult:
    molecule_class: MoleculeClass     # enum from src.types
    confidence: str                   # "High" / "Medium" / "Low"
    confidence_score: float           # 0.0–1.0
    evidence: List[str]               # human-readable reasoning
    warnings: List[str]               # advisory messages
    chain_types: List[str]            # detected chain types
    n_chains: int                     # total chain count (with copy_number)
    n_unique_chains: int              # structurally distinct chains
    risk_weights: Dict[str, float]    # class-specific analysis weights
```

### Dependencies
- `src.types.MoleculeClass` (shared enum)
- `src.platform_config` (chain detection thresholds)
- `src.ood_baseline` (OOD scoring, optional)
- No external ML packages required (rule-based primary)

### Key Constants
| Constant | Value | Source |
|----------|-------|--------|
| MIN_SEQUENCE_LENGTH | 10 | platform_config |
| MIN_HC_LENGTH | 200 | platform_config |
| MIN_CHAIN_CLUSTER_LENGTH | 80 | platform_config |
| HC_IDENTITY_THRESHOLD | 0.85 | platform_config |

### Self-Test
Built-in `__main__` validation (85%+ accuracy gate on test panel).
Post-training validation via `SelfTest/run_validation.py` (12 checks).

---

## 2. Training Workbench (Harmonizer + Trainer + Benchmark)

**Files:** `src/training/data_harmonizer.py`, `src/training/classifier_trainer.py`,
`src/training/benchmark_evaluator.py`, `src/training/model_inference.py`,
`src/training/schema.py`

### Purpose
End-to-end ML pipeline: harmonize external data → train classifier →
benchmark against rule-based → produce deployable model artifact.

### Input Contract (Harmonizer)
```python
harmonize_datasets(
    jain_path: str,            # Jain et al. 2017 CSV
    therasabdab_path: str,     # TheraSAbDab dump
    peptide_path: str,         # SATPdb peptide CSV
    output_path: str,          # target harmonized CSV
)
```
- External CSV schemas are handled internally (column mapping)
- Minimum sequence length: `MIN_SEQUENCE_LENGTH` from `platform_config`

### Output Contract (Model Artifact)
```
models/classifier/
├── classifier_model.npz     # weights (coef, intercept, mean, std)
└── classifier_metadata.json # {model_type, classes, feature_cols, label_to_idx, test_accuracy, ...}
```

### Output Contract (Benchmark)
```python
@dataclass
class BenchmarkReport:
    results: List[Dict]              # per-panel-entry results
    rule_based_accuracy: float       # rule-based classifier accuracy
    trained_accuracy: float          # trained model accuracy
    class_changes: List[Dict]        # [{name, from, to, expected}]
```

### Dependencies
- `src.molecule_classifier.classify_molecule` (benchmark only)
- `src.types.MoleculeClass` (class enum validation)
- `src.platform_config` (MIN_TEST_ACCURACY, MAX_BENCHMARK_DRIFT, CONFIDENCE thresholds)
- `numpy`, `pandas`, `scikit-learn` (training only)

### Validation Gates
| Gate | Threshold | Source |
|------|-----------|--------|
| MIN_TEST_ACCURACY | 0.50 | platform_config |
| MAX_BENCHMARK_DRIFT | 4 changes | platform_config |
| MAX_ACCURACY_DEGRADATION | 0.05 | platform_config |

### Self-Test
Benchmark evaluator runs full panel + validation checks (12 sub-checks).
Integrated into `SelfTest/run_validation.py` Test #20.

---

## 3. Analytical Twin (PTM / Liability / Mass Characterization)

**Files:** `src/analytical_twin.py`

### Purpose
In-silico analytical characterization: intact mass prediction, peptide
mapping with tryptic digest, PTM liability scanning (deamidation,
oxidation, glycosylation, isomerization), and liability density assessment.

### Input Contract
```python
run_analytical_characterization(
    sequence: str,                           # amino acid sequence
    chains: Optional[List[Dict]] = None,     # multi-chain assembly info
    molecule_class: str = "canonical_mab",   # from classifier
    protein_name: str = "",
    glycoform_profile: str = "standard_cho",
)
```
- Chain dicts: `{"sequence": str, "chain_type": str, "copy_number": int, "name": str}`
- `chain_type` uses "HC"/"LC" convention (normalized internally)

### Output Contract
```python
@dataclass
class AnalyticalResult:
    status: str                          # "success" / "error"
    protein_name: str
    sequence_length: int
    intact_mass: Dict[str, Any]          # {theoretical_mass, observed_mass_range, glycoform_contributions}
    peptide_map: List[Dict[str, Any]]    # [{peptide, start, end, mass, liabilities: [...]}]
    liability_density: Dict[str, Any]    # {total_motifs, density_per_1000, risk_level, per_type_density}
    n_glycosylation_sites: int
    assembly_mode: str                   # "stoichiometric" / "single_chain"
    chains_used: List[Dict[str, Any]]
    summary: Dict[str, Any]              # {total_peptides, liability_peptides, coverage_pct, ...}
    message: str
```
All output objects have `.to_dict()` for backward compatibility.

### Dependencies
- `src.types.MoleculeClass` (via molecule_classifier re-export, for format-specific logic)
- No external ML packages (pure computational chemistry)
- Optional: `Bio.SeqUtils.ProtParam` for enhanced pI/MW (graceful fallback)

### Key Constants (module-specific)
| Constant | Value | Notes |
|----------|-------|-------|
| NIST_MASSES | dict | NIST amino acid monoisotopic masses |
| LIABILITY_MOTIFS | 9 patterns | NG, NS, NT, M, W, NxST, DP, DG, DS |
| GLYCOFORM_PROFILES | 4 profiles | standard_cho, high_mannose, afucosylated, highly_sialylated |
| density_high | 80/1000 res | High liability density threshold |
| density_medium | 50/1000 res | Medium threshold |

### Self-Test
Built-in `__main__` validation with NISTmAb reference.
Integrated into `SelfTest/run_validation.py` (end-to-end + cross-section tests).

---

## Dependency Graph (Simplified)

```
src/types.py (MoleculeClass)
    ↑
    ├── src/molecule_classifier.py (re-exports + classify_molecule)
    │       ↑
    │       ├── src/developability_core.py
    │       ├── src/bulk_runner.py / bulk_schema.py
    │       └── src/training/benchmark_evaluator.py
    │
    ├── src/analytical_twin.py
    ├── src/feature_registry.py
    └── src/training/schema.py

src/platform_config.py (thresholds)
    ↑
    ├── src/molecule_classifier.py
    ├── src/training/model_inference.py
    ├── src/training/benchmark_evaluator.py
    └── src/training/data_harmonizer.py
```
