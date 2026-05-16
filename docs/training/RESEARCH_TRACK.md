# Research Track Modules

> Modules in this file are **not production-blocking**. They may have
> incomplete contracts, missing selftest coverage, or depend on external
> models (ESM-2, large MLP, etc.) that are not yet validated at
> production quality.

## Purpose

The research track isolates experimental functionality from the
production classifier + training pipeline. All research modules are
accessed through **soft (deferred) imports** — the production path never
hard-depends on them, and `ImportError` is always handled gracefully.

## Research Modules

| Module | Status | Why research | Path to production |
|--------|--------|-------------|-------------------|
| `developability_predictor.py` | Prototype | No contract, experimental scoring model | Needs contract + benchmark + calibration against Jain-137 |
| `ml_predictor.py` | Prototype | Alternative ML classifier approach | May merge into classifier_trainer if XGBoost ceiling is hit |
| `esm2_encoder.py` | Stub | Requires ESM-2 model download (~2 GB) | Needs infrastructure for model hosting |
| `unified_trainer.py` | Experimental | Multi-task training across modules | Blocked on sub-model maturity |
| `uncertainty_engine.py` | Experimental | MC-dropout / ensemble uncertainty | Needs calibration study |
| `structural_twin.py` | Placeholder | 3D structure prediction integration | Requires AlphaFold/ESMFold API access |

## Integration Rules

1. **Soft imports only**: Research modules must be imported inside
   `try/except ImportError` blocks or inside function bodies (deferred).

2. **No production dependency**: No production module (`molecule_classifier`,
   `analytical_twin`, `developability_core`, `bulk_runner`) may hard-import
   a research module at module level.

3. **Graceful degradation**: If a research module fails to load, the
   production path must continue without it. Feature flags or
   `platform_config` toggles may gate research features.

4. **Promotion criteria**: A research module is promoted to production
   when it has:
   - A formal contract (`classification_contract.py` style)
   - A selftest with ≥85% pass rate
   - A benchmark with ≥3 edge cases
   - No hard dependencies on unavailable external models
   - Reviewed by at least one domain expert

## Relationship to Training Roadmap

The TRAINING_GUIDE.md roadmap (priorities 3-8) describes future
training tasks, some of which will be implemented in research modules
before promotion:

| Training Priority | Research Module | Status |
|-------------------|----------------|--------|
| 3. Liability weighting | `developability_predictor` | Prototype |
| 5. Developability calibration | `developability_predictor` | Not started |
| 6. ADA risk | (new module needed) | Not started |
| 7. Upstream/downstream | (new module needed) | Not started |
| 8. Full end-to-end | `unified_trainer` | Blocked |
