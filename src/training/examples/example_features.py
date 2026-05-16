#!/usr/bin/env python3
"""
Example: Compute biophysical features for a sequence.

Usage:
    python -m src.training.examples.example_features
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.training.features import compute_all_features
from src.training.schema import FEATURE_COLS

# Example: Trastuzumab heavy chain (truncated)
hc = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYSSSWYWGQGTLVTVSS"
lc = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKRLIYAASSLQSGVPSRFSGSGSGTEFTLTISSLQPEDIATYYC"

features = compute_all_features(
    sequence=hc,
    n_chains=2, n_unique_chains=2,
    hc_sequence=hc, lc_sequence=lc,
)

print("Feature computation example (trastuzumab-like)")
print("=" * 50)
for col in FEATURE_COLS:
    val = features.get(col, "MISSING")
    print(f"  {col:25s} = {val}")
print(f"\nTotal features: {len(FEATURE_COLS)}")
