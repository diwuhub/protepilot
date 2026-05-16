"""
ood_baseline.py  ·  ProtePilot — Dynamic OOD Baseline Calculator
===================================================================
Computes Out-of-Distribution reference statistics from actual training
data, replacing the hardcoded IGG_REFERENCE_STATS in
developability_predictor.py.

When no training data is available, falls back to the default IgG
reference statistics (backward compatible).

Usage:
    from src.ood_baseline import OODBaselineCalculator

    calc = OODBaselineCalculator()
    baseline = calc.compute_from_training_data(harmonized_df)
    # baseline = {"length": {"mean": 480, "std": 120}, ...}
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("ProtePilot.OODBaseline")


# ── Default IgG reference (same as developability_predictor.py) ──
DEFAULT_IGG_REFERENCE = {
    "gravy":            {"mean": -0.40, "std": 0.15, "min": -0.8, "max": 0.1},
    "cys_count_per_100": {"mean": 3.5, "std": 0.8, "min": 1.5, "max": 6.0},
    "length":           {"mean": 450, "std": 100, "min": 100, "max": 1400},
    "pI":               {"mean": 8.2, "std": 0.8, "min": 5.5, "max": 10.0},
    "MW_kDa":           {"mean": 148, "std": 30, "min": 12, "max": 200},
}

# ── Class-specific reference baselines ──
CLASS_SPECIFIC_BASELINES: Dict[str, Dict[str, Dict[str, float]]] = {
    "canonical_mab": DEFAULT_IGG_REFERENCE,
    "bispecific": {
        "gravy":            {"mean": -0.38, "std": 0.16, "min": -0.8, "max": 0.1},
        "cys_count_per_100": {"mean": 3.4, "std": 0.9, "min": 1.5, "max": 6.0},
        "length":           {"mean": 470, "std": 120, "min": 200, "max": 1500},
        "pI":               {"mean": 8.0, "std": 1.0, "min": 5.5, "max": 10.0},
        "MW_kDa":           {"mean": 150, "std": 35, "min": 20, "max": 220},
    },
    "fc_fusion": {
        "gravy":            {"mean": -0.35, "std": 0.20, "min": -1.0, "max": 0.2},
        "cys_count_per_100": {"mean": 2.8, "std": 1.0, "min": 0.5, "max": 6.0},
        "length":           {"mean": 400, "std": 150, "min": 150, "max": 1200},
        "pI":               {"mean": 7.5, "std": 1.2, "min": 4.5, "max": 10.0},
        "MW_kDa":           {"mean": 120, "std": 50, "min": 15, "max": 200},
    },
    "adc": DEFAULT_IGG_REFERENCE,  # Same as mAb (antibody backbone)
    "single_domain": {
        "gravy":            {"mean": -0.45, "std": 0.20, "min": -1.0, "max": 0.2},
        "cys_count_per_100": {"mean": 1.8, "std": 0.6, "min": 0.5, "max": 4.0},
        "length":           {"mean": 130, "std": 30, "min": 80, "max": 250},
        "pI":               {"mean": 7.0, "std": 1.5, "min": 4.0, "max": 10.0},
        "MW_kDa":           {"mean": 15, "std": 5, "min": 8, "max": 30},
    },
    "peptide": {
        "gravy":            {"mean": -0.20, "std": 0.50, "min": -2.0, "max": 2.0},
        "cys_count_per_100": {"mean": 1.0, "std": 2.0, "min": 0.0, "max": 10.0},
        "length":           {"mean": 35, "std": 20, "min": 5, "max": 80},
        "pI":               {"mean": 7.0, "std": 2.5, "min": 3.0, "max": 12.0},
        "MW_kDa":           {"mean": 4, "std": 3, "min": 0.5, "max": 10},
    },
    "fusion_protein": {
        "gravy":            {"mean": -0.30, "std": 0.25, "min": -1.0, "max": 0.3},
        "cys_count_per_100": {"mean": 2.5, "std": 1.2, "min": 0.0, "max": 6.0},
        "length":           {"mean": 500, "std": 200, "min": 100, "max": 1500},
        "pI":               {"mean": 7.0, "std": 1.5, "min": 4.0, "max": 10.0},
        "MW_kDa":           {"mean": 60, "std": 40, "min": 10, "max": 200},
    },
    "engineered_scaffold": {
        "gravy":            {"mean": -0.30, "std": 0.30, "min": -1.5, "max": 0.5},
        "cys_count_per_100": {"mean": 1.5, "std": 1.5, "min": 0.0, "max": 8.0},
        "length":           {"mean": 170, "std": 60, "min": 50, "max": 400},
        "pI":               {"mean": 6.5, "std": 2.0, "min": 3.0, "max": 11.0},
        "MW_kDa":           {"mean": 20, "std": 10, "min": 5, "max": 50},
    },
    "unknown": DEFAULT_IGG_REFERENCE,
}


# v32.1: Consolidated — import canonical GRAVY from feature_registry.
from src.feature_registry import _compute_gravy  # noqa: E402


def _estimate_pi(seq: str) -> float:
    if not seq:
        return 7.0
    s = seq.upper()
    n = len(s)
    basic = s.count("K") + s.count("R") + s.count("H")
    acidic = s.count("D") + s.count("E")
    return 7.0 + 0.5 * (basic - acidic) / max(n, 1)


class OODBaselineCalculator:
    """
    Compute and manage OOD reference statistics from training data.

    Falls back to DEFAULT_IGG_REFERENCE when no training data is available.
    """

    def __init__(self):
        self.computed_stats: Optional[Dict[str, Dict[str, float]]] = None
        self.n_samples: int = 0
        self.source: str = "default"   # "default" | "computed"
        self.computed_at: Optional[str] = None

    def compute_from_training_data(
        self,
        harmonized_df,
        sequence_column: str = "Combined_Sequence",
        min_samples: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute OOD reference statistics from harmonized training data.

        Parameters
        ----------
        harmonized_df : pd.DataFrame with at least a sequence column
        sequence_column : column name containing sequences
        min_samples : minimum samples needed to override defaults

        Returns
        -------
        Dict with keys: length, gravy, pI, MW_kDa, cys_count_per_100
        """
        import pandas as pd

        if harmonized_df is None or len(harmonized_df) < min_samples:
            log.warning(
                f"Insufficient training data ({len(harmonized_df) if harmonized_df is not None else 0} samples). "
                f"Using default IgG reference stats."
            )
            return dict(DEFAULT_IGG_REFERENCE)

        sequences = harmonized_df[sequence_column].dropna().tolist()
        sequences = [s for s in sequences if isinstance(s, str) and len(s) >= 20]

        if len(sequences) < min_samples:
            log.warning(f"Only {len(sequences)} valid sequences. Using default baseline.")
            return dict(DEFAULT_IGG_REFERENCE)

        # Compute distributions
        lengths = [len(s) for s in sequences]
        gravys = [_compute_gravy(s) for s in sequences]
        pis = [_estimate_pi(s) for s in sequences]
        mws = [len(s) * 0.11 for s in sequences]  # kDa estimate
        cys_per_100 = [s.upper().count("C") * 100.0 / max(len(s), 1) for s in sequences]

        def _stats(values):
            arr = np.array(values)
            return {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(max(float(np.std(arr)), 1e-4), 4),  # Floor at 1e-4
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
            }

        self.computed_stats = {
            "length": _stats(lengths),
            "gravy": _stats(gravys),
            "pI": _stats(pis),
            "MW_kDa": _stats(mws),
            "cys_count_per_100": _stats(cys_per_100),
        }
        self.n_samples = len(sequences)
        self.source = "computed"
        self.computed_at = datetime.now().isoformat()

        log.info(
            f"OOD baseline computed from {self.n_samples} sequences. "
            f"Length: {self.computed_stats['length']['mean']:.0f} ± {self.computed_stats['length']['std']:.0f}"
        )

        return self.computed_stats

    def get_active_baseline(self) -> Dict[str, Dict[str, float]]:
        """Return computed baseline if available, else defaults."""
        if self.computed_stats is not None:
            return self.computed_stats
        return dict(DEFAULT_IGG_REFERENCE)

    def get_baseline_for_class(self, molecule_class: str) -> Dict[str, Dict[str, float]]:
        """Return class-specific baseline, falling back to computed or default IgG."""
        # If we have computed stats from actual training data, prefer those
        if self.computed_stats is not None:
            return self.computed_stats
        # Otherwise use class-specific reference
        return dict(CLASS_SPECIFIC_BASELINES.get(molecule_class, DEFAULT_IGG_REFERENCE))

    def get_sequence_length_stats(self) -> Dict[str, float]:
        """Get length stats from the active baseline."""
        baseline = self.get_active_baseline()
        return baseline.get("length", DEFAULT_IGG_REFERENCE["length"])

    def get_baseline_info(self) -> Dict[str, Any]:
        """Summary for UI display."""
        baseline = self.get_active_baseline()
        return {
            "source": self.source,
            "n_samples": self.n_samples,
            "computed_at": self.computed_at,
            "length_mean": baseline["length"]["mean"],
            "length_std": baseline["length"]["std"],
            "is_default": self.source == "default",
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for session_state storage."""
        return {
            "computed_stats": self.computed_stats,
            "n_samples": self.n_samples,
            "source": self.source,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OODBaselineCalculator":
        """Restore from session_state."""
        calc = cls()
        calc.computed_stats = d.get("computed_stats")
        calc.n_samples = d.get("n_samples", 0)
        calc.source = d.get("source", "default")
        calc.computed_at = d.get("computed_at")
        return calc


def compute_composite_ood_score(
    z_scores: List[float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute a composite OOD score from individual metric z-scores.

    Uses root-mean-square of z-scores (Mahalanobis-like for diagonal covariance).
    Returns a score where:
        < 1.5  → clearly in-distribution
        1.5–2.5 → borderline
        > 2.5  → out-of-distribution
    """
    if not z_scores:
        return 0.0
    arr = np.array(z_scores)
    rms = float(np.sqrt(np.mean(arr ** 2)))
    return round(rms, 3)


def _selftest() -> bool:
    """Smoke test for OOD baseline."""
    calc = OODBaselineCalculator()

    # Default baseline should work
    baseline = calc.get_active_baseline()
    assert "length" in baseline, "Missing 'length' in baseline"
    assert "gravy" in baseline, "Missing 'gravy' in baseline"

    # Class-specific baselines should exist for all classes
    for cls_name in CLASS_SPECIFIC_BASELINES:
        b = calc.get_baseline_for_class(cls_name)
        assert "length" in b, f"Missing 'length' for {cls_name}"
        assert "pI" in b, f"Missing 'pI' for {cls_name}"

    # Peptide baseline should have much shorter length than IgG
    pep_len = CLASS_SPECIFIC_BASELINES["peptide"]["length"]["mean"]
    igg_len = DEFAULT_IGG_REFERENCE["length"]["mean"]
    assert pep_len < igg_len / 5, f"Peptide length {pep_len} should be << IgG length {igg_len}"

    # Composite score
    score = compute_composite_ood_score([1.0, 1.5, 0.5])
    assert 0.5 < score < 2.0, f"Unexpected composite score {score}"

    log.info("OODBaseline selftest PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
    print("All ood_baseline tests passed.")
