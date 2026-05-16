"""
Example: Train the molecule classifier from harmonized data.

Usage:
    PYTHONPATH=/path/to/ProtePilot python examples/train_classifier.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.classifier_trainer import train_classifier
from src.training.model_inference import load_classifier, predict_class

# Train
print("Training classifier...")
result = train_classifier()
print(f"  Model:    {result.model_type}")
print(f"  Classes:  {result.n_classes}")
print(f"  Accuracy: {result.test_accuracy:.4f}")
print(f"  F1 macro: {result.test_f1_macro:.4f}")
print(f"  Artifact: {result.artifact_path}")

# Load and predict
clf = load_classifier()
pred = predict_class(clf, sequence="EVQLVESGGGLVQ" * 30, n_chains=2)
print(f"\n  Prediction: {pred['molecule_class']} ({pred['confidence']}, p={pred['probability']:.3f})")
