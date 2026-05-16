#!/usr/bin/env python3
"""
Example: Load trained classifier and predict molecule class.

Usage:
    python -m src.training.examples.example_predict
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.training.model_inference import load_classifier, predict_class
from src.training.features import compute_all_features

# 1. Load trained model
clf = load_classifier("models/classifier")
if clf is None:
    print("No trained classifier found. Run: python -m src.training.pipeline --step train")
    sys.exit(1)

print(f"Loaded: {clf.model_type}, {len(clf.classes)} classes")
print(f"Features: {len(clf.feature_cols)} ({clf.metadata.get('feature_schema_version', 'unversioned')})")
print()

# 2. Predict on example sequences
examples = [
    {
        "name": "Trastuzumab-like (canonical mAb)",
        "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYSSSWYWGQGTLVTVSS" * 3,
        "n_chains": 2, "n_unique_chains": 2,
        "hc": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYSSSWYWGQGTLVTVSS",
        "lc": "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKRLIYAASSLQSGVPSRFSGSGSGTEFTLTISSLQPEDIATYYC",
    },
    {
        "name": "Short peptide",
        "sequence": "ACDEFGHIKLMNPQRST",
        "n_chains": 1, "n_unique_chains": 1,
        "hc": "", "lc": "",
    },
    {
        "name": "Nanobody-like (single domain)",
        "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYSSSWYWGQGTLVTVSS",
        "n_chains": 1, "n_unique_chains": 1,
        "hc": "", "lc": "",
    },
]

for ex in examples:
    result = predict_class(
        clf, sequence=ex["sequence"],
        n_chains=ex["n_chains"], n_unique_chains=ex["n_unique_chains"],
        hc_sequence=ex["hc"], lc_sequence=ex["lc"],
    )
    print(f"{ex['name']:40s} → {result['molecule_class']:20s} "
          f"(conf={result['confidence']}, p={result['probability']:.3f})")

print("\nDone.")
