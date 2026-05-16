"""
pipeline.py — Classifier / OOD Unified Pipeline Module
=======================================================
Single entry point for the ProtePilot molecule classification subsystem.

This module wraps harmonize → train → ood → benchmark into a reproducible,
versioned pipeline with a fixed I/O contract.  It is designed to be the
**first independently repo-able module** in the platform.

I/O Schema (v1)
----------------
Input:  raw data files in data/external/ + curated CSVs
Output: versioned artifact bundle in models/ with:
        - classifier_xgboost.json   (or classifier_model.npz for LR/softmax)
        - classifier_model.npz      (mean/std scaler)
        - classifier_metadata.json  (feature_cols, label_to_idx, metrics, version)
        - ood_detector.npz          (global Mahalanobis)
        - iforest_model.joblib      (IsolationForest)
        - ood_class_*.npz           (per-class Mahalanobis)
        - ood_metadata.json         (thresholds, metrics, version)
        - MANIFEST.json             (pipeline version, git hash, timestamp, step checksums)

Artifact Versioning
-------------------
Each pipeline run writes a MANIFEST.json at the artifact root that records:
  - pipeline_version (semver, bumped manually)
  - schema_version   (feature column hash — changes when features are added/removed)
  - git_hash         (HEAD at training time, if available)
  - timestamp        (ISO-8601 UTC)
  - steps_run        (which steps were executed)
  - data_checksum    (SHA-256 of harmonized CSV)
  - metrics          (test accuracy, F1 macro, OOD F1)

Usage:
    # Full pipeline
    python -m src.training.pipeline

    # Individual steps
    python -m src.training.pipeline --step harmonize
    python -m src.training.pipeline --step train
    python -m src.training.pipeline --step ood
    python -m src.training.pipeline --step benchmark

    # Selftest (no training, just validates artifacts + inference)
    python -m src.training.pipeline --step selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("ProtePilot.Training.Pipeline")

# ── Constants ────────────────────────────────────────────────────────────

PIPELINE_VERSION = "1.0.0"

# Paths (relative to project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
TRAINING_CSV = os.path.join(DATA_DIR, "training", "classifier_data.csv")
CLASSIFIER_DIR = os.path.join(_PROJECT_ROOT, "models", "classifier")
OOD_DIR = os.path.join(_PROJECT_ROOT, "models", "ood_detector")
MANIFEST_PATH = os.path.join(_PROJECT_ROOT, "models", "MANIFEST.json")


# ── Utilities ────────────────────────────────────────────────────────────

def _git_hash() -> str:
    """Get current HEAD short hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=_PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _file_sha256(path: str) -> str:
    """SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _schema_hash() -> str:
    """Hash of the feature column list — changes when features are added/removed."""
    from src.training.schema import FEATURE_COLS
    return hashlib.sha256(",".join(FEATURE_COLS).encode()).hexdigest()[:16]


# ── Step: Harmonize ──────────────────────────────────────────────────────

def step_harmonize(verbose: bool = True) -> Dict[str, Any]:
    """Run data harmonization → produce classifier_data.csv."""
    from src.training.data_harmonizer import harmonize
    log.info("Step: HARMONIZE")
    t0 = time.time()

    records = harmonize(DATA_DIR)

    import pandas as pd
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(TRAINING_CSV), exist_ok=True)
    df.to_csv(TRAINING_CSV, index=False)

    elapsed = time.time() - t0
    n_rows = len(df)
    class_dist = df["molecule_class"].value_counts().to_dict() if "molecule_class" in df.columns else {}
    data_hash = _file_sha256(TRAINING_CSV)

    result = {
        "step": "harmonize",
        "n_rows": n_rows,
        "class_distribution": class_dist,
        "output_path": TRAINING_CSV,
        "data_checksum": data_hash,
        "elapsed_s": round(elapsed, 2),
    }
    if verbose:
        log.info("  Harmonized %d rows in %.1fs", n_rows, elapsed)
        for cls, cnt in sorted(class_dist.items(), key=lambda x: -x[1]):
            log.info("    %-25s %d", cls, cnt)
    return result


# ── Step: Train Classifier ───────────────────────────────────────────────

def step_train(verbose: bool = True) -> Dict[str, Any]:
    """Train molecule classifier → save artifacts."""
    from src.training.classifier_trainer import train_classifier
    log.info("Step: TRAIN CLASSIFIER")
    t0 = time.time()

    result = train_classifier(TRAINING_CSV, artifact_dir=CLASSIFIER_DIR)
    elapsed = time.time() - t0

    out = {
        "step": "train",
        "model_type": getattr(result, "model_type", "unknown"),
        "test_accuracy": getattr(result, "test_accuracy", None),
        "test_f1_macro": getattr(result, "test_f1_macro", None),
        "per_class_f1": getattr(result, "per_class_f1", {}),
        "elapsed_s": round(elapsed, 2),
    }
    if verbose:
        log.info("  Model: %s | Acc=%.3f | F1-macro=%.3f | %.1fs",
                 out["model_type"], out["test_accuracy"] or 0,
                 out["test_f1_macro"] or 0, elapsed)
    return out


# ── Step: OOD Detector ───────────────────────────────────────────────────

def step_ood(verbose: bool = True) -> Dict[str, Any]:
    """Train OOD detector → save artifacts."""
    from src.training.ood_trainer import train_ood_detector
    log.info("Step: TRAIN OOD DETECTOR")
    t0 = time.time()

    result = train_ood_detector(TRAINING_CSV, artifact_dir=OOD_DIR)
    elapsed = time.time() - t0

    out = {
        "step": "ood",
        "detector_type": "ensemble_mahalanobis_iforest",
        "val_f1": getattr(result, "val_f1", None),
        "test_f1": getattr(result, "test_f1", None),
        "elapsed_s": round(elapsed, 2),
    }
    if verbose:
        log.info("  OOD: %s | val_f1=%.3f | test_f1=%.3f | %.1fs",
                 out["detector_type"], out["val_f1"] or 0,
                 out["test_f1"] or 0, elapsed)
    return out


# ── Step: Benchmark ──────────────────────────────────────────────────────

def step_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """Run benchmark on holdout panel."""
    from src.training.benchmark_evaluator import run_benchmark
    log.info("Step: BENCHMARK")
    t0 = time.time()

    report = run_benchmark()
    elapsed = time.time() - t0

    out = {
        "step": "benchmark",
        "n_molecules": report.panel_size,
        "rule_based_accuracy": report.rule_based_accuracy,
        "trained_accuracy": report.trained_accuracy,
        "class_changes": len(report.class_changes),
        "elapsed_s": round(elapsed, 2),
    }
    if verbose:
        log.info("  Benchmarked %d molecules in %.1fs (rule=%.2f, trained=%.2f)",
                 report.panel_size, elapsed,
                 report.rule_based_accuracy, report.trained_accuracy)
    return out


# ── Step: Selftest ───────────────────────────────────────────────────────

def step_selftest(verbose: bool = True) -> Dict[str, Any]:
    """
    Validate artifacts exist and inference produces sane results.
    Does NOT retrain — reads existing artifacts only.
    """
    log.info("Step: SELFTEST")
    errors = []

    # 1. Schema selftest
    try:
        from src.training.schema import _selftest as schema_test
        schema_test()
    except Exception as e:
        errors.append(f"schema selftest: {e}")

    # 2. Classifier artifact exists and loads
    try:
        from src.training.model_inference import load_classifier, predict_class
        clf = load_classifier(CLASSIFIER_DIR)
        if clf is None:
            errors.append("No classifier artifact found")
        else:
            # Smoke-test inference
            pred = predict_class(
                clf,
                sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYSSSWYWGQGTLVTVSS" * 3,
                n_chains=2, n_unique_chains=2,
                hc_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYSSSWYWGQGTLVTVSS",
                lc_sequence="DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKRLIYAASSLQSGVPSRFSGSGSGTEFTLTISSLQPEDIATYYC",
            )
            if not pred or "molecule_class" not in pred:
                errors.append("predict_class returned invalid output")
            else:
                log.info("  Classifier inference OK: %s (p=%.2f)",
                         pred["molecule_class"], pred.get("probability", 0))
    except Exception as e:
        errors.append(f"classifier selftest: {e}")

    # 3. OOD detector artifact exists and loads
    try:
        from src.training.ood_trainer import load_ood_detector, predict_ood
        from src.training.features import compute_all_features
        det = load_ood_detector(OOD_DIR)
        if det is None:
            errors.append("No OOD detector artifact found")
        else:
            # Build feature dict using the canonical features.py
            test_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYSSSWYWGQGTLVTVSS"
            test_features = compute_all_features(
                test_seq, n_chains=2, n_unique_chains=2,
                hc_sequence=test_seq, lc_sequence="",
            )
            ood_result = predict_ood(
                features=test_features,
                detector=det,
                predicted_class="canonical_mab",
            )
            if ood_result is None:
                errors.append("predict_ood returned None")
            else:
                log.info("  OOD inference OK: is_ood=%s", ood_result.get("is_ood"))
    except Exception as e:
        errors.append(f"ood selftest: {e}")

    # 4. Feedback store selftest
    try:
        from src.training.feedback_store import _selftest as fb_test
        fb_test()
    except Exception as e:
        errors.append(f"feedback_store selftest: {e}")

    # 5. Manifest check
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        current_schema = _schema_hash()
        manifest_schema = manifest.get("schema_version", "")
        if current_schema != manifest_schema:
            errors.append(
                f"Schema version mismatch: code={current_schema}, "
                f"manifest={manifest_schema}. Re-train required."
            )
        else:
            log.info("  Manifest schema version matches: %s", current_schema)
    else:
        log.info("  No MANIFEST.json found (will be created on next full run)")

    # 6. Canonical features.py: selftest + verify all 24 cols present
    try:
        from src.training.features import compute_all_features, _selftest as feat_selftest
        from src.training.schema import FEATURE_COLS
        feat_selftest()
        # Verify output keys match schema exactly
        test_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYSSSWYWGQGTLVTVSS"
        feats = compute_all_features(test_seq, n_chains=2, n_unique_chains=2,
                                     hc_sequence=test_seq,
                                     lc_sequence="DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKRLIYAASSLQSGVPSRFSGSGSGTEFTLTISSLQPEDIATYYC")
        missing = set(FEATURE_COLS) - set(feats.keys())
        if missing:
            errors.append(f"features.py output missing cols: {missing}")
        else:
            log.info("  features.py canonical: 24/24 features, selftest PASS")
    except Exception as e:
        errors.append(f"features.py selftest: {e}")

    # 7. Feature column + version consistency: schema vs all artifacts
    try:
        from src.training.schema import FEATURE_COLS
        current_schema = _schema_hash()
        for label, dir_path in [("classifier", CLASSIFIER_DIR), ("ood", OOD_DIR)]:
            meta_file = os.path.join(dir_path, f"{label}_metadata.json" if label != "ood" else "ood_metadata.json")
            if not os.path.exists(meta_file):
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            # Check feature cols
            if label == "classifier" and meta.get("feature_cols") != list(FEATURE_COLS):
                errors.append(f"{label}: feature_cols mismatch ({len(meta.get('feature_cols', []))} vs {len(FEATURE_COLS)})")
            # Check feature schema version
            art_schema = meta.get("feature_schema_version", "")
            if art_schema and art_schema != current_schema:
                errors.append(f"{label}: feature_schema_version mismatch (artifact={art_schema}, code={current_schema})")
            elif not art_schema:
                log.info("  %s: no feature_schema_version in artifact (pre-versioning)", label)
            else:
                log.info("  %s artifact version OK: schema=%s, pipeline=%s",
                         label, art_schema, meta.get("pipeline_version", "?"))
    except Exception as e:
        errors.append(f"artifact version check: {e}")

    # 8. n_unique_chains propagation: verify classify_molecule passes it to both
    #    trained model AND OOD detector (guards against the critical bug we fixed)
    try:
        import inspect
        from src.molecule_classifier import _apply_ood_detection
        sig = inspect.signature(_apply_ood_detection)
        if "n_unique_chains" not in sig.parameters:
            errors.append(
                "_apply_ood_detection() missing n_unique_chains parameter — "
                "OOD detector will use wrong features for bispecifics"
            )
        else:
            log.info("  n_unique_chains propagation OK: _apply_ood_detection has parameter")
    except Exception as e:
        errors.append(f"n_unique_chains check: {e}")

    # 9. End-to-end classify_molecule smoke test (peptide + canonical paths)
    try:
        from src.molecule_classifier import classify_molecule
        # Peptide path
        r1 = classify_molecule(sequence="ACDEFGHIKLM" * 2)
        assert r1.molecule_class.value == "peptide", f"Expected peptide, got {r1.molecule_class.value}"
        # User override → feedback path
        r2 = classify_molecule(sequence="ACDEFGHIKLM" * 2, user_hint="engineered_scaffold")
        assert r2.user_override == "engineered_scaffold", "User override not set"
        log.info("  End-to-end classify_molecule OK (2 paths tested)")
    except Exception as e:
        errors.append(f"classify_molecule e2e: {e}")

    result = {
        "step": "selftest",
        "passed": len(errors) == 0,
        "errors": errors,
    }
    if errors:
        for err in errors:
            log.error("  FAIL: %s", err)
    else:
        log.info("  ALL SELFTESTS PASSED")

    return result


# ── Manifest Writer ──────────────────────────────────────────────────────

def write_manifest(step_results: List[Dict[str, Any]]) -> str:
    """
    Write MANIFEST.json summarizing the pipeline run.

    Returns the manifest path.
    """
    # Collect metrics from step results
    metrics = {}
    data_checksum = ""
    for sr in step_results:
        step_name = sr.get("step", "unknown")
        if step_name == "harmonize":
            data_checksum = sr.get("data_checksum", "")
        elif step_name == "train":
            metrics["test_accuracy"] = sr.get("test_accuracy")
            metrics["test_f1_macro"] = sr.get("test_f1_macro")
            metrics["model_type"] = sr.get("model_type")
        elif step_name == "ood":
            metrics["ood_val_f1"] = sr.get("val_f1")
            metrics["ood_test_f1"] = sr.get("test_f1")

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": _schema_hash(),
        "git_hash": _git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps_run": [sr.get("step") for sr in step_results],
        "data_checksum": data_checksum,
        "metrics": metrics,
        "feature_count": len(_get_feature_cols()),
        "artifacts": {
            "classifier_dir": CLASSIFIER_DIR,
            "ood_dir": OOD_DIR,
            "training_csv": TRAINING_CSV,
        },
    }

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest written: %s", MANIFEST_PATH)
    return MANIFEST_PATH


def _get_feature_cols():
    from src.training.schema import FEATURE_COLS
    return FEATURE_COLS


# ── Full Pipeline ────────────────────────────────────────────────────────

def run_pipeline(
    steps: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full training pipeline (or selected steps).

    Parameters
    ----------
    steps : list of str, optional
        Subset of ["harmonize", "train", "ood", "benchmark", "selftest"].
        Defaults to all steps in order.
    verbose : bool
        Print progress to log.

    Returns
    -------
    dict with keys: steps, results, manifest_path, total_elapsed_s
    """
    all_steps = ["harmonize", "train", "ood", "benchmark", "selftest"]
    if steps is None:
        steps = all_steps

    step_fn = {
        "harmonize": step_harmonize,
        "train": step_train,
        "ood": step_ood,
        "benchmark": step_benchmark,
        "selftest": step_selftest,
    }

    t0 = time.time()
    results = []
    for step_name in steps:
        fn = step_fn.get(step_name)
        if fn is None:
            log.warning("Unknown step '%s', skipping", step_name)
            continue
        try:
            r = fn(verbose=verbose)
            results.append(r)
        except Exception as e:
            log.error("Step '%s' FAILED: %s", step_name, e)
            results.append({"step": step_name, "error": str(e)})

    total = time.time() - t0

    # Write manifest if we ran training steps
    manifest_path = None
    training_steps = {"harmonize", "train", "ood"}
    if training_steps & set(steps):
        manifest_path = write_manifest(results)

    out = {
        "steps": steps,
        "results": results,
        "manifest_path": manifest_path,
        "total_elapsed_s": round(total, 2),
    }

    if verbose:
        log.info("Pipeline complete: %d steps in %.1fs", len(steps), total)
        selftest_results = [r for r in results if r.get("step") == "selftest"]
        if selftest_results and not selftest_results[0].get("passed", True):
            log.warning("SELFTEST FAILURES: %s", selftest_results[0].get("errors"))

    return out


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ProtePilot — Classifier/OOD Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.training.pipeline                          # Full pipeline
  python -m src.training.pipeline --step harmonize,train   # Only harmonize + train
  python -m src.training.pipeline --step selftest          # Validate artifacts
  python -m src.training.pipeline --step benchmark         # Run benchmark only
        """,
    )
    parser.add_argument(
        "--step", type=str, default=None,
        help="Comma-separated list of steps: harmonize,train,ood,benchmark,selftest",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()
    steps = args.step.split(",") if args.step else None
    verbose = not args.quiet

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(name)s | %(message)s",
    )

    result = run_pipeline(steps=steps, verbose=verbose)

    # Print final summary
    print("\n" + "=" * 60)
    print(f"Pipeline v{PIPELINE_VERSION}  |  schema={_schema_hash()}")
    print(f"Steps: {', '.join(result['steps'])}")
    print(f"Total: {result['total_elapsed_s']}s")
    if result.get("manifest_path"):
        print(f"Manifest: {result['manifest_path']}")

    # Check for failures
    for r in result.get("results", []):
        if r.get("error"):
            print(f"  FAIL [{r['step']}]: {r['error']}")
        elif r.get("step") == "selftest" and not r.get("passed", True):
            print(f"  SELFTEST FAILURES: {r.get('errors')}")

    print("=" * 60)

    # Exit with error code if selftest failed
    for r in result.get("results", []):
        if r.get("step") == "selftest" and not r.get("passed", True):
            sys.exit(1)
        if r.get("error"):
            sys.exit(1)


if __name__ == "__main__":
    main()
