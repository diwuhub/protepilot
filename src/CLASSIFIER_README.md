# ProtePilot — Molecule Classifier

The platform's **front gate**: every downstream analysis reads the classification result to determine routing, risk weights, and validation strategy.

## What It Does

Classifies biologic molecules into 9 categories based on sequence structure, chain architecture, motif detection, and name hints. The classification drives feature selection, model routing, risk weight profiles, and recommended validation assays.

**Supported classes:** canonical_mab, bispecific, fc_fusion, adc, single_domain, peptide, fusion_protein, engineered_scaffold, unknown

## 3-Phase Fusion Strategy

```
Phase 1: Rule-Based (authoritative)
    Sequence length → chain count → motif detection → name hints
    This is the ground truth — Phases 2 and 3 never override it.

Phase 2: Trained Model Second Opinion (XGBoost, 24 features)
    High rule confidence → skip model
    Both agree → boost confidence
    Disagree → flag warning, keep rule-based

Phase 3: OOD Detection (Mahalanobis + IForest ensemble)
    If OOD: cap confidence ≤ 0.30, add warning
    Skips peptide and single_domain (inherently out-of-distribution)

User Override: absolute priority, feedback recorded on disagreement
```

## Files

| File | Purpose |
|------|---------|
| `molecule_classifier.py` | Core classifier: rule-based + ML + OOD fusion, CLI |
| `classification_contract.py` | Formal contract: input/output schema, behavioral guarantees |
| `classifier_benchmark.py` | Expanded benchmark: 28+ test cases, contract compliance |
| `types.py` | `MoleculeClass` enum (shared across platform) |

## Quick Start

```bash
# Classify a sequence
python -m src.molecule_classifier --sequence "ACDEFGHIKLM"

# Classify with chain info
python -m src.molecule_classifier --sequence "EVQL..." \
    --chains '[{"chain_type":"HC","sequence":"EVQL..."},{"chain_type":"LC","sequence":"DIQM..."}]'

# User override
python -m src.molecule_classifier --sequence "ACDEFGHIKLM" --hint adc

# JSON output
python -m src.molecule_classifier --sequence "ACDEFGHIKLM" --json

# Run selftest (13 checks, all 8 classes + contract)
python -m src.molecule_classifier --selftest

# Run full benchmark (27 cases + 13 contract compliance)
python -m src.classifier_benchmark

# Run benchmark in specific mode
python -m src.classifier_benchmark --mode rule     # Phase 1 only
python -m src.classifier_benchmark --mode ml       # Phase 1 + 2
python -m src.classifier_benchmark --mode fusion   # Full pipeline (default)

# Run benchmark for a single class
python -m src.classifier_benchmark --class adc

# Run validation corpus
python -m src.molecule_classifier --validate
```

## Classification Result Schema

```python
ClassificationResult:
    molecule_class: MoleculeClass   # One of 9 enum values
    confidence: str                 # "High", "Medium", "Low"
    confidence_score: float         # 0.0–1.0
    evidence: List[str]             # Why this classification
    warnings: List[str]             # Caveats and disagreements
    n_chains: int                   # Number of input chains
    n_unique_chains: int            # Chains with <85% identity
    chain_types: List[str]          # ["HC", "LC", ...]
    chain_lengths: List[int]        # Per-chain lengths
    user_override: Optional[str]    # If user explicitly selected
```

## Risk Weight Profiles

Each class has a distinct risk weight profile (sums to 1.0) that the Developability Core Layer uses for composite scoring:

| Class | Primary Risk | Secondary |
|-------|-------------|-----------|
| canonical_mab | aggregation (0.30) | stability (0.25) |
| bispecific | aggregation (0.25) | stability (0.20) + species_purity (0.15) |
| fc_fusion | aggregation (0.30) | expression (0.25) |
| adc | stability (0.25) | aggregation (0.20) + conjugation (0.15) |
| single_domain | aggregation (0.35) | stability (0.30) |
| peptide | stability (0.40) | immunogenicity (0.35) |

## Contract & Behavioral Guarantees

`classification_contract.py` defines 10 testable guarantees:

1. **peptide_under_80aa**: Single chain <80 aa → peptide, High confidence
2. **bispecific_two_distinct_hc**: Two HCs with <85% identity → bispecific
3. **canonical_mab_hc_lc_motifs**: HC + LC + ≥3 motifs → canonical_mab
4. **fc_no_cl_means_fusion**: Fc present but no CL → fc_fusion
5. **user_hint_absolute**: Valid user_hint always wins
6. **ood_never_overrides_class**: OOD caps confidence only
7. **risk_weights_complete**: All classes have valid weights
8. **output_schema_valid**: to_dict() always passes validation
9. **empty_input_returns_unknown**: Empty → unknown/Low
10. **validation_corpus_accuracy**: Built-in corpus ≥85%

## Dependencies

**Required:** None (pure Python, stdlib only)

**Optional (soft):**
- `src.training.model_inference` — Phase 2 trained model (falls back to rule-based only)
- `src.training.ood_trainer` — Phase 3 OOD detection (falls back to skip)
- `src.training.feedback_store` — Feedback recording (fire-and-forget)
- `src.training.features` — Feature computation for ML paths
- `src.platform_config` — Threshold constants (has defaults)
- `src.types` — MoleculeClass enum (re-exported for compatibility)
