"""
Example: Run the 7-molecule benchmark panel to compare rule-based vs trained classifier.

Usage:
    PYTHONPATH=/path/to/ProtePilot python examples/run_benchmark.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.benchmark_evaluator import run_benchmark

report = run_benchmark()

print(f"{'Molecule':<32} {'Expected':<16} {'Rule-Based':<16} {'Trained':<16}")
print("-" * 80)
for r in report.results:
    print(f"{r['name']:<32} {r['expected']:<16} {r['rule_class']:<16} {r['trained_class']:<16}")

print(f"\nRule-based accuracy: {report.rule_based_accuracy:.0%}")
print(f"Trained accuracy:   {report.trained_accuracy:.0%}")

if report.class_changes:
    print(f"\nClass changes ({len(report.class_changes)}):")
    for ch in report.class_changes:
        print(f"  {ch['name']}: {ch['from']} -> {ch['to']} (expected: {ch['expected']})")
