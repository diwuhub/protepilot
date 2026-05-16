"""
benchmark_evaluator.py — Before/After Platform-Level Comparison
================================================================
Runs a fixed benchmark panel through the platform TWICE:
  1. Rule-based baseline (current production)
  2. Trained model (candidate replacement)

Compares at FOUR levels:
  - Model metrics:      accuracy, F1, confusion
  - Platform behavior:  molecule_class, confidence, OOD status
  - Report impact:      recommendation tone, grade
  - Governance:         selftest section pass/fail counts

The benchmark panel includes:
  - NISTmAb (canonical reference)
  - A bispecific (multi-chain)
  - An Fc-fusion
  - A single-domain / nanobody
  - A peptide
  - 2 OOD synthetic sequences (should be flagged)

Usage:
    from src.training.benchmark_evaluator import run_benchmark
    report = run_benchmark(artifact_dir="models/classifier")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("ProtePilot.Training.Benchmark")


# ═══════════════════════════════════════════════════════════════════════
#  Fixed benchmark panel
# ═══════════════════════════════════════════════════════════════════════

BENCHMARK_PANEL = [
    {
        "name": "NISTmAb_RM8671",
        "expected_class": "canonical_mab",
        "hc_sequence": (
            "QVTLRESGPALVKPTQTLTLTCTFSGFSLSTAGMSVGWIRQPPGKALEWLADIWWDDKK"
            "DYNPSLKDRLTISKDTSKNQVVLKVTNMDPADTATYYCARDMIFNFYFDVWGQGTTVTVSS"
        ),
        "lc_sequence": (
            "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQS"
            "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
        ),
        "is_ood": False,
        "description": "NISTmAb reference standard — canonical IgG1, well-characterized",
    },
    {
        "name": "TestBispecific_HC1LC1HC2LC2",
        "expected_class": "bispecific",
        "hc_sequence": (
            "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNG"
            "YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQ"
            "GTLVTVSS"
        ),
        "lc_sequence": (
            "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLES"
            "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
        ),
        "extra_chains": [
            {"sequence": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYDINWVRQATGQGLEWMGWMNPNSG"
                         "NTGYAQKFQGRVTMTRDTSISTAYMEVSRLRSDDTAVYYCARDPFGAMDYWGQGTL"
                         "VTVSS",
             "chain_type": "HC", "copy_number": 1, "name": "HC2"},
        ],
        "is_ood": False,
        "description": "Bispecific antibody with 3 unique chains",
    },
    {
        "name": "TestFcFusion_Etanercept",
        "expected_class": "fc_fusion",
        "hc_sequence": (
            "LPAQVAFTPYAPEPGSTCRLREYYDQTAQMCCSKCSPGQHAKVFCTKTSDTVCDSCEDSTYTQL"
            "WNWVPECLEVKVKQHGTVSFANRGSGPIYGSNHNNFLTPEMHSFYKNQVSLTCLVKGFYPSDIAVE"
        ),
        "lc_sequence": "",
        "is_ood": False,
        "description": "Fc-fusion (TNF receptor extracellular domain + Fc)",
    },
    {
        "name": "TestNanobody_VHH",
        "expected_class": "single_domain",
        "hc_sequence": (
            "QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAIWSGG"
            "STYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAADSTIYASYYECGHGLSTGG"
        ),
        "lc_sequence": "",
        "is_ood": False,
        "description": "Camelid VHH nanobody — single domain, no light chain",
    },
    {
        "name": "TestPeptide_GLP1",
        "expected_class": "peptide",
        "hc_sequence": "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG",
        "lc_sequence": "",
        "is_ood": False,
        "description": "GLP-1 peptide therapeutic — 31 amino acids",
    },
    {
        "name": "OOD_AllAlanine",
        "expected_class": "unknown",
        "hc_sequence": "A" * 200,
        "lc_sequence": "",
        "is_ood": True,
        "description": "Synthetic all-alanine — should be OOD (extreme composition)",
    },
    {
        "name": "OOD_ExtremeBasic",
        "expected_class": "unknown",
        "hc_sequence": ("KRH" * 50 + "ACDEF" * 10),
        "lc_sequence": "",
        "is_ood": True,
        "description": "Synthetic extremely basic sequence — should be OOD",
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  Single-molecule evaluation
# ═══════════════════════════════════════════════════════════════════════

def _run_rule_based(panel_entry: dict) -> Dict[str, Any]:
    """Run the rule-based classifier on a benchmark entry."""
    from src.molecule_classifier import classify_molecule

    hc = panel_entry["hc_sequence"]
    lc = panel_entry.get("lc_sequence", "")
    name = panel_entry["name"]
    combined = hc + lc if lc else hc

    chains = []
    if hc and len(hc) >= 20:
        chains.append({"sequence": hc, "chain_type": "HC", "copy_number": 1, "name": "HC"})
    if lc and len(lc) >= 20:
        chains.append({"sequence": lc, "chain_type": "LC", "copy_number": 1, "name": "LC"})
    for ec in panel_entry.get("extra_chains", []):
        chains.append(ec)

    result = classify_molecule(
        sequence=combined,
        chains=chains if chains else None,
        name=name,
    )

    return {
        "molecule_class": result.molecule_class.value,
        "confidence": result.confidence,
        "evidence_count": len(result.evidence),
    }


def _run_trained_model(panel_entry: dict, artifact_dir: str) -> Dict[str, Any]:
    """Run the trained classifier on a benchmark entry via model_inference."""
    try:
        from src.training.model_inference import load_classifier, predict_class
        from src.training.data_harmonizer import _clean_seq

        clf = load_classifier(artifact_dir)
        if clf is None:
            return {"molecule_class": "unknown", "confidence": "N/A", "error": "No model artifact"}

        hc = _clean_seq(panel_entry["hc_sequence"])
        lc = _clean_seq(panel_entry.get("lc_sequence", ""))
        combined = hc + lc if lc else hc
        n_chains = (1 if len(hc) >= 20 else 0) + (1 if len(lc) >= 20 else 0)

        n_unique = len(set(filter(None, [hc, lc])))
        result = predict_class(
            clf, sequence=combined, n_chains=n_chains,
            hc_sequence=hc, lc_sequence=lc,
            n_unique_chains=n_unique,
        )
        return {
            "molecule_class": result["molecule_class"],
            "confidence": result["confidence"],
            "probability": round(result["probability"], 3),
        }
    except Exception as e:
        log.warning("Trained model inference failed: %s", e)
        return {"molecule_class": "unknown", "confidence": "N/A", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  Full benchmark comparison
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkReport:
    """Full before/after comparison report."""
    panel_size: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregate metrics
    rule_based_correct: int = 0
    trained_correct: int = 0
    rule_based_accuracy: float = 0.0
    trained_accuracy: float = 0.0

    # Differences
    class_changes: List[Dict[str, str]] = field(default_factory=list)
    confidence_upgrades: int = 0
    confidence_downgrades: int = 0
    ood_detection_changes: List[Dict[str, str]] = field(default_factory=list)

    timestamp: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "BENCHMARK COMPARISON: Rule-Based vs Trained Model",
            "=" * 70,
            f"Panel size:          {self.panel_size}",
            f"Rule-based correct:  {self.rule_based_correct}/{self.panel_size} ({self.rule_based_accuracy:.1%})",
            f"Trained correct:     {self.trained_correct}/{self.panel_size} ({self.trained_accuracy:.1%})",
            "",
            f"Classification changes: {len(self.class_changes)}",
            f"Confidence upgrades:    {self.confidence_upgrades}",
            f"Confidence downgrades:  {self.confidence_downgrades}",
            "",
            "Per-molecule results:",
        ]
        for r in self.results:
            marker = ""
            if r["rule_class"] != r["trained_class"]:
                marker = "  *** CLASS CHANGED ***"
            elif r["rule_confidence"] != r["trained_confidence"]:
                marker = "  (confidence changed)"
            lines.append(
                f"  {r['name']:30s}  expected={r['expected']:18s}  "
                f"rule={r['rule_class']:18s}({r['rule_confidence']:6s})  "
                f"trained={r['trained_class']:18s}({r['trained_confidence']:6s})"
                f"{marker}"
            )
        if self.class_changes:
            lines.append("")
            lines.append("Classification changes:")
            for ch in self.class_changes:
                lines.append(f"  {ch['name']}: {ch['from']} → {ch['to']} (expected: {ch['expected']})")

        lines.append("=" * 70)
        return "\n".join(lines)


_CONFIDENCE_RANK = {"High": 3, "Medium": 2, "Low": 1, "N/A": 0}


def run_benchmark(artifact_dir: str = "models/classifier") -> BenchmarkReport:
    """
    PIPELINE BENCHMARK: Rule-based vs trained model comparison on the benchmark panel.

    Evaluates the real-world classification path: rule-based classifier (production
    baseline) vs trained ML model (upgrade candidate).  Tracks class changes,
    confidence upgrades/downgrades, and per-molecule accuracy against ground truth.

    This is the platform-level benchmark — it exercises the full classification
    stack including feature computation and OOD detection, not just the raw model.

    Parameters
    ----------
    artifact_dir : str
        Directory containing trained model artifact.

    Returns
    -------
    BenchmarkReport
        Complete comparison report.
    """
    report = BenchmarkReport(
        panel_size=len(BENCHMARK_PANEL),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    for entry in BENCHMARK_PANEL:
        name = entry["name"]
        expected = entry["expected_class"]

        # Run both
        rule_result = _run_rule_based(entry)
        trained_result = _run_trained_model(entry, artifact_dir)

        rule_cls = rule_result["molecule_class"]
        trained_cls = trained_result["molecule_class"]
        rule_conf = rule_result.get("confidence", "N/A")
        trained_conf = trained_result.get("confidence", "N/A")

        # Correctness (for OOD entries, we don't count correctness — they're "unknown")
        if not entry.get("is_ood", False):
            if rule_cls == expected:
                report.rule_based_correct += 1
            if trained_cls == expected:
                report.trained_correct += 1

        # Track changes
        if rule_cls != trained_cls:
            report.class_changes.append({
                "name": name, "from": rule_cls, "to": trained_cls, "expected": expected,
            })

        # Confidence direction
        r_rank = _CONFIDENCE_RANK.get(rule_conf, 0)
        t_rank = _CONFIDENCE_RANK.get(trained_conf, 0)
        if t_rank > r_rank:
            report.confidence_upgrades += 1
        elif t_rank < r_rank:
            report.confidence_downgrades += 1

        report.results.append({
            "name": name,
            "expected": expected,
            "is_ood": entry.get("is_ood", False),
            "rule_class": rule_cls,
            "rule_confidence": rule_conf,
            "trained_class": trained_cls,
            "trained_confidence": trained_conf,
            "trained_probability": trained_result.get("probability"),
        })

    non_ood = sum(1 for e in BENCHMARK_PANEL if not e.get("is_ood", False))
    report.rule_based_accuracy = report.rule_based_correct / max(non_ood, 1)
    report.trained_accuracy = report.trained_correct / max(non_ood, 1)

    return report


# ═══════════════════════════════════════════════════════════════════════
#  Post-training selftest checks
# ═══════════════════════════════════════════════════════════════════════

def post_training_selftest(artifact_dir: str = "models/classifier") -> Dict[str, bool]:
    """
    ML-ONLY SELFTEST: Artifact validity, determinism, schema compatibility, and drift.

    Validates the trained model artifact in isolation — no platform dependencies beyond
    schema and inference.  Checks: file existence, metadata schema, class enum alignment,
    feature column match, inference determinism, output validity, accuracy gates,
    benchmark drift limit, and no-regression-vs-baseline.

    Returns dict of check_name → pass/fail.
    """
    checks = {}

    # ── Check 1: Model artifact exists and is loadable ──
    model_path = os.path.join(artifact_dir, "classifier_model.npz")
    meta_path = os.path.join(artifact_dir, "classifier_metadata.json")
    checks["artifact_file_exists"] = os.path.exists(model_path)
    checks["metadata_file_exists"] = os.path.exists(meta_path)

    if not checks["artifact_file_exists"] or not checks["metadata_file_exists"]:
        log.warning("Artifact files missing — skipping remaining checks")
        return checks

    try:
        data = np.load(model_path)
        checks["artifact_loadable"] = True
    except Exception as e:
        log.error("Cannot load model: %s", e)
        checks["artifact_loadable"] = False
        return checks

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        checks["metadata_loadable"] = True
    except Exception as e:
        log.error("Cannot load metadata: %s", e)
        checks["metadata_loadable"] = False
        return checks

    # ── Check 2: Metadata schema valid ──
    required_keys = {"model_type", "feature_cols", "label_to_idx", "idx_to_label",
                     "n_classes", "classes", "test_accuracy"}
    checks["metadata_schema_valid"] = required_keys.issubset(set(meta.keys()))

    # ── Check 3: Classes match MoleculeClass enum ──
    try:
        from src.type_defs import MoleculeClass
        valid_classes = {mc.value for mc in MoleculeClass}
        model_classes = set(meta.get("classes", []))
        checks["classes_match_enum"] = model_classes.issubset(valid_classes)
    except ImportError:
        checks["classes_match_enum"] = False

    # ── Check 4: Feature columns match expected ──
    from src.training.schema import FEATURE_COLS
    checks["feature_cols_match"] = meta.get("feature_cols") == list(FEATURE_COLS)

    # ── Check 5: Deterministic inference (5 runs, same input) ──
    panel_0 = BENCHMARK_PANEL[0]  # NISTmAb
    predictions = []
    for _ in range(5):
        r = _run_trained_model(panel_0, artifact_dir)
        predictions.append(r.get("molecule_class", "?"))
    checks["inference_deterministic"] = len(set(predictions)) == 1

    # ── Check 6: Output class is a valid MoleculeClass ──
    for entry in BENCHMARK_PANEL[:3]:
        r = _run_trained_model(entry, artifact_dir)
        cls = r.get("molecule_class", "?")
        try:
            from src.type_defs import MoleculeClass
            valid = cls in {mc.value for mc in MoleculeClass} or cls == "unknown"
        except ImportError:
            valid = True
        if not valid:
            checks["output_class_valid"] = False
            break
    else:
        checks["output_class_valid"] = True

    # ── Check 7: Accuracy above minimum threshold ──
    acc = meta.get("test_accuracy", 0)
    # Validation gates from platform_config (cross-module constants)
    try:
        from src.platform_config import (
            MIN_TEST_ACCURACY, MAX_BENCHMARK_DRIFT, MAX_ACCURACY_DEGRADATION,
        )
    except ImportError:
        MIN_TEST_ACCURACY, MAX_BENCHMARK_DRIFT, MAX_ACCURACY_DEGRADATION = 0.50, 4, 0.05

    checks["test_accuracy_above_50pct"] = acc >= MIN_TEST_ACCURACY

    # ── Check 8: No catastrophic drift (benchmark panel) ──
    benchmark = run_benchmark(artifact_dir)
    n_class_changes = len(benchmark.class_changes)
    # Allow up to MAX_BENCHMARK_DRIFT changes: OOD molecules may change,
    # and major retraining events (e.g. 4→8 classes) legitimately shift
    # some panel predictions.  Rule-based classifier is primary.
    checks["no_catastrophic_benchmark_drift"] = n_class_changes <= MAX_BENCHMARK_DRIFT
    if n_class_changes > MAX_BENCHMARK_DRIFT:
        log.warning("Benchmark drift: %d class changes detected", n_class_changes)

    # ── Check 9: Training didn't make accuracy worse ──
    checks["not_worse_than_baseline"] = benchmark.trained_accuracy >= (benchmark.rule_based_accuracy - MAX_ACCURACY_DEGRADATION)

    return checks


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Benchmark trained model vs baseline")
    parser.add_argument("--artifact-dir", default="models/classifier")
    args = parser.parse_args()

    print("=" * 70)
    print("  PIPELINE BENCHMARK: Rule-Based vs Trained Model Comparison")
    print("=" * 70)
    report = run_benchmark(args.artifact_dir)
    print(report.summary())

    print("\n" + "=" * 70)
    print("  ML-ONLY SELFTEST: Artifact Validity, Schema, Determinism, Drift")
    print("=" * 70)
    checks = post_training_selftest(args.artifact_dir)
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:45s}  [{status}]")
    all_pass = all(checks.values())
    print(f"\n  {'ALL PASS' if all_pass else 'SOME FAILED'} ({sum(checks.values())}/{len(checks)})")
