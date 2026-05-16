"""
src/training/ — ProtePilot Training Infrastructure
====================================================
Modular training system for biologics ML models.

Modules:
    pipeline             — Unified entry point: harmonize → train → ood → benchmark → selftest
    data_harmonizer      — Build unified training CSV from Jain137 + TheraSAbDab + curated
    classifier_trainer   — Train molecule classifier (XGBoost with LR/softmax fallback)
    ood_trainer          — Train ensemble OOD detector (Mahalanobis + IsolationForest)
    model_inference      — Load trained artifacts and run inference
    benchmark_evaluator  — Before/after comparison on fixed benchmark panel
    feedback_store       — Record-only user corrections (no auto-retrain)
    schema               — Feature column definitions, split management, validation
"""
