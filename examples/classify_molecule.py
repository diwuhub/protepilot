"""
Example: Classify a molecule using the ProtePilot classifier.

Usage:
    PYTHONPATH=/path/to/ProtePilot python examples/classify_molecule.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.molecule_classifier import classify_molecule

# Canonical mAb (HC + LC)
result = classify_molecule(
    sequence="EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
    chains=[
        {"sequence": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS", "chain_type": "HC"},
        {"sequence": "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK", "chain_type": "LC"},
    ],
)

print(f"Class:      {result.molecule_class.value}")
print(f"Display:    {result.molecule_class.display_name}")
print(f"Confidence: {result.confidence} ({result.confidence_score:.2f})")
print(f"Evidence:   {result.evidence}")
if result.warnings:
    print(f"Warnings:   {result.warnings}")

# Short peptide
result2 = classify_molecule(sequence="HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG")
print(f"\nPeptide:    {result2.molecule_class.value} ({result2.confidence})")

# Empty input
result3 = classify_molecule(sequence="")
print(f"Empty:      {result3.molecule_class.value} ({result3.confidence})")
