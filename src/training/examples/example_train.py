#!/usr/bin/env python3
"""
Example: Run full training pipeline from harmonization to selftest.

Usage:
    python -m src.training.examples.example_train
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from src.training.pipeline import run_pipeline, PIPELINE_VERSION

print(f"ProtePilot Training Pipeline v{PIPELINE_VERSION}")
print("=" * 50)

# Run full pipeline
result = run_pipeline(
    steps=["harmonize", "train", "ood", "selftest"],
    verbose=True,
)

# Print results
print(f"\nTotal time: {result['total_elapsed_s']}s")
if result.get("manifest_path"):
    print(f"Manifest: {result['manifest_path']}")

for r in result.get("results", []):
    if r.get("step") == "selftest":
        if r.get("passed"):
            print("\nSELFTEST: ALL PASSED")
        else:
            print(f"\nSELFTEST FAILURES: {r.get('errors')}")
