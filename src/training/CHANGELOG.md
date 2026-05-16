# Changelog

## v1.0.0 (2026-03-21)

### Added
- Unified pipeline CLI: `python -m src.training.pipeline --step harmonize,train,ood,benchmark,selftest`
- `features.py`: Canonical single-source feature computation (24 features)
- Artifact versioning: `feature_schema_version` + `pipeline_version` in all metadata
- `MANIFEST.json`: Pipeline-level version tracking with data checksums
- `feedback_store.py`: Record-only user correction mechanism (append-only CSV)
- 9-point selftest suite: schema, artifacts, features alignment, n_unique_chains propagation, e2e
- XGBoost classifier with class-balanced sample weights (300 trees, depth=6)
- Ensemble OOD detector: global Mahalanobis + IsolationForest + per-class Mahalanobis
- 24-feature schema: 12 biophysical + 8 composition + 4 chain-level
- Curated Fc-fusion training data (80 entries with IgG1 Fc sequences)

### Fixed
- `_apply_ood_detection()` missing `n_unique_chains` parameter (critical: OOD used wrong features for bispecifics)
- `_FEATURE_COLS` duplication across modules → consolidated to `schema.FEATURE_COLS`
- Format mapping priority bugs in `_map_format()` (nanobody/Fc-fusion/fusion_protein ordering)

### Performance
- Test Accuracy: 0.983
- Test F1 Macro: 0.735
- fc_fusion F1: 0.839 (up from 0.000)
- OOD Test F1: 0.473
- Training data: 24,829 rows, 8 classes
