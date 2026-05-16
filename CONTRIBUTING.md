# Contributing to ProtePilot (ProtePilot Platform)

## Environment Setup

The project uses a three-layer dependency model. Install only what you need.

### Layer 1 — Core Runtime

Runs the headless analysis pipeline, bulk CSV processing, report generation, and all validation tests. All predictions are heuristic-based (no ML models required).

```bash
pip install -r requirements-core.txt
```

Packages: numpy, pandas, biopython, plotly, matplotlib, kaleido, python-docx, h5py, requests, pytest.

### Layer 2 — Analysis Runtime

Adds the Streamlit UI, 3D protein visualization, and LLM copilot. Predictions remain heuristic-based at this layer.

```bash
pip install -r requirements-analysis.txt
```

Adds: streamlit, py3Dmol, stmol, openai.

### Layer 3 — Training Runtime

Adds ML training infrastructure: ESM-2 embeddings, XGBoost models, SHAP explainability.

```bash
pip install -r requirements-training.txt
```

Adds: torch, xgboost, scikit-learn, joblib, shap, transformers, sentencepiece.

### Check Your Environment

```bash
python3 src/optional_deps.py
```

This prints which layers are installed and which packages are missing.

---

## Running Tests

Tests are organized into layers matching the three dependency tiers.

### Core tests (Layer 1 — safest, fastest)

```bash
pytest -m core
```

Runs all unit tests that do not require torch, sklearn, or streamlit. Covers the pipeline, bulk analysis, bispecific engine, report assembler, OOD baseline, and property mapper.

### Bulk analysis tests only

```bash
pytest -m bulk
```

Runs the 52 tests covering bulk_schema, bulk_runner, and bulk_summary. Safe in a Layer 1 environment.

### ML tests (Layer 3)

```bash
pytest -m ml
```

Tests that import torch or sklearn. Automatically skipped if those packages are not installed.

### Integration tests (Layer 2+)

```bash
pytest -m integration
```

End-to-end tests that require streamlit. Automatically skipped in Layer 1 environments.

### Full suite

```bash
pytest -m "core or ml or integration"
```

### Excluding slow tests

```bash
pytest -m "core and not slow"
```

---

## SelfTest Suite

The standalone validation runner at `SelfTest/run_validation.py` does not require pytest and can be run directly:

```bash
python3 SelfTest/run_validation.py
```

It executes 19 test sections (308 substantive checks) and is the canonical pre-commit gate for numerical correctness and cross-module consistency. Safe in a Layer 1 environment.

| Section | Coverage |
|---------|----------|
| 1 – Property computation | MW, pI, GRAVY, deamidation, oxidation |
| 2 – Molecule classifier | All 10 molecule classes |
| 3 – Developability scoring | 5-dimension composite score |
| 4 – Bispecific engine | pI estimation, species peaks, Rs |
| 5 – Report assembly | ReportObject field completeness |
| 6 – Cross-path alignment | Single vs. bulk numeric identity |
| 7 – Bulk schema | CSV parse, chain building, stoichiometry |
| 8 – Bulk runner | BulkRowResult, error isolation |
| 9 – Bulk summary | Export, rank, display stats |
| 10 – OOD detection | Baseline thresholds, z-score detection |
| 11 – Formulation twin | pH, excipient, freeze-thaw screening |
| 12 – Immunogenicity twin | T-cell epitope scanning, risk flags |
| 13 – Preclinical twin | Toxicity flags, clearance prediction |
| 14 – Property mapper | Intent dict completeness |
| 15 – ML predictor (heuristic) | Rule-based fallback accuracy |
| 16 – Biophysical features | Feature registry completeness |
| 17 – Grade canonicality | "Low Risk" / "Medium Risk" / "High Risk" enforcement |
| 18 – Schema alignment | Bulk/single field-map integrity, 59 fields |
| 19 – Platform alignment | Cross-module consistency, gap detection, determinism, color audit |

---

## Pytest Marks Reference

| Mark | Description | Minimum Layer |
|------|-------------|---------------|
| `core` | No torch/sklearn/streamlit | Layer 1 |
| `bulk` | Bulk analysis pipeline tests | Layer 1 |
| `governance` | Schema alignment and grade canonicality | Layer 1 |
| `ml` | Requires torch + sklearn | Layer 3 |
| `integration` | Requires streamlit | Layer 2 |
| `slow` | Tests taking > 10 seconds | Any |

---

## Optional Dependency Handling

All optional packages degrade gracefully. When adding new code that uses an optional package:

1. Use lazy imports inside functions (not at module top level) for ML/UI packages.
2. Wrap the import in try/except and provide a fallback.
3. Use `from src.optional_deps import available, require` for clear error messages:
   ```python
   from src.optional_deps import require
   require("torch")  # raises ImportError with install instructions
   ```
4. Document which layer the dependency belongs to in the function docstring.

---

## Context Isolation Guarantees

The platform enforces strict state isolation to prevent cross-analysis pollution:

**Single-molecule analysis:** Each analysis creates a fresh `ReportContext` (frozen immutably after population). All section builders read from this frozen context. No module-level mutable state in the analysis path. No `st.session_state` references in backend `src/` analysis modules.

**Bulk row analysis:** Each CSV row creates its own independent `BulkRowResult`. No shared mutable cache between rows. Comprehensive twin modules (analytical, QC, PK, etc.) are isolated per-row with independent try/except.

**Training state:** Training functions create new model instances (atomic replacement, not in-place mutation). `factory_reset()` clears ALL cached state including the JAIN137 calibration flag. Training objects never mutate inference/report state.

**Export chain:** All export paths (CSV, JSON, DOCX, UI summary) read from the same canonical `ReportObject` or `BulkBatchResult`. No export function calls scoring functions directly — all values are pre-computed. `ReportContext.to_dict()` uses `asdict()` for deep copy safety.

These guarantees are enforced by selftest section 19 (Platform Alignment) checks 17–21.

---

## Code Style

- Python 3.10+. Type hints encouraged on all public functions.
- Docstrings: use the existing `"""One-line summary.\n\nDetail paragraph."""` style.
- No bare `except:` — always catch specific exceptions or `Exception`.
- Grade strings: always use `"Low Risk"` / `"Medium Risk"` / `"High Risk"` in output-facing code. Internal scoring uses bare `"Low"` / `"Medium"` / `"High"`; `grade_to_risk_label()` in `report_schema.py` is the canonical converter.
- Colors: use design tokens from `src/ui_colors.py` (`T.PASS`, `T.CAUTION`, `T.FAIL`, etc.) — never raw hex values.

---

## Pull Request Checklist

Before opening a PR, verify:

1. `python3 -m py_compile src/<changed_file>.py` — no syntax errors.
2. `pytest -m core -x` — all core tests pass.
3. `python3 SelfTest/run_validation.py` — 20 sections, 325 checks (section 20 may SKIP if no trained model).
4. No new raw hex color values introduced in app.py (use `T.*` tokens).
5. Any new molecule class added to `MOLECULE_RECOMMENDATION_SUFFIX` in `report_schema.py` and handled in the `_is_non_canonical` branches of `_generate_recommendation()` in `report_assembler.py`.
6. Any new bulk/single output field documented in `src/bulk_single_schema_alignment.py` FIELD_MAP.
7. Any new external dependency added to the correct requirements layer file.
8. Optional dependencies imported lazily (inside functions) with try/except fallback.
