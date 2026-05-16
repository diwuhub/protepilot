"""
multitask_adapter.py  ·  ProtePilot — Unified Integration
===========================================================
Backward-Compatible Adapter for Unified Multi-Task Model

Wraps the UnifiedMultiTaskModel to produce outputs in the exact same
format as the original 4 independent predictors:
  - ChromatographyMLP.predict_single()  → predict_chromatography()
  - DevelopabilityPredictor.predict()    → predict_developability()
  - WetLabPredictor.predict_single()     → predict_wetlab()
  - PotencyPredictor.predict_single()    → predict_potency()

This allows existing ProtePilot tools/pipelines (agents.py) to switch
to the unified model without any API changes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

log = logging.getLogger("ProtePilot.MultiTaskAdapter")


class MultiTaskAdapter:
    """
    Adapter that wraps UnifiedMultiTaskModel and exposes prediction
    methods compatible with ProtePilot's existing tool APIs.

    Parameters
    ----------
    model_path : str
        Path to saved UnifiedMultiTaskModel state dict.
    device : str
        Inference device.
    """

    def __init__(
        self,
        model_path: str = "models/unified_multitask_best.pt",
        device: str = "cpu",
    ):
        self.device = device
        self.model = None
        self._model_path = model_path
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        """Lazily load the unified model."""
        if self._loaded:
            return self.model is not None

        self._loaded = True
        if not os.path.exists(self._model_path):
            log.warning(f"Unified model not found at {self._model_path}")
            return False

        try:
            from esm2_hybrid_encoder import ESM2HybridEncoder
            from unified_multitask_model import UnifiedMultiTaskModel

            state = torch.load(self._model_path, map_location=self.device, weights_only=False)
            # Auto-detect task heads from checkpoint keys to avoid mismatch
            ckpt_tasks = sorted({k.split(".")[1] for k in state if k.startswith("heads.")})
            encoder = ESM2HybridEncoder()
            self.model = UnifiedMultiTaskModel(
                encoder=encoder,
                tasks=ckpt_tasks if ckpt_tasks else None,
            )
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            log.info(f"Loaded unified model from {self._model_path}")
            return True
        except Exception as e:
            log.error(f"Failed to load unified model: {e}")
            self.model = None
            return False

    @property
    def is_available(self) -> bool:
        """Check if the unified model is loaded and ready."""
        return self._ensure_loaded()

    def _predict_all_raw(
        self,
        hc_seq: str,
        lc_seq: str,
        biophys: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Run unified model and return all 8 predictions as floats.
        """
        if not self._ensure_loaded():
            raise RuntimeError("Unified model not available")

        biophys_tensor = None
        if biophys is not None:
            from unified_dataset import BIOPHYS_COLUMNS
            vals = [biophys.get(c, 0.0) for c in BIOPHYS_COLUMNS]
            biophys_tensor = torch.tensor([vals], dtype=torch.float32)

        return self.model.predict_numpy([hc_seq], [lc_seq], biophys_tensor)

    # ------------------------------------------------------------------
    # API-compatible prediction methods
    # ------------------------------------------------------------------

    def predict_chromatography(
        self,
        pI: float,
        mw: float,
        deam_sites: int = 1,
        ox_sites: int = 1,
        acidic_residues: int = 40,
        basic_residues: int = 50,
        hydrophobicity: float = 0.35,
        sequence: Optional[str] = None,
        hc_sequence: Optional[str] = None,
        lc_sequence: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Drop-in replacement for ChromatographyMLP.predict_single().

        Returns
        -------
        dict matching {"ka": float, "nu": float, "estimated_rt_min": float}
        """
        hc = hc_sequence or sequence or "EVQLVESGGGLVQPGGSLRLSCAAS"
        lc = lc_sequence or sequence or "DIQMTQSPSSLSASVGDRVTITC"

        biophys = {
            "pI": pI, "MW_kDa": mw,
            "deam_sites": float(deam_sites), "ox_sites": float(ox_sites),
            "acidic_residues": float(acidic_residues),
            "basic_residues": float(basic_residues),
            "hydrophobicity_gravy": hydrophobicity,
        }

        preds = self._predict_all_raw(hc, lc, biophys)

        ka = preds.get("ka", 1.0)
        nu = preds.get("nu", 3.0)

        # Estimated RT (same heuristic as ChromatographyMLP)
        estimated_rt = 10.0 + nu * 1.5 + (1.0 / max(ka, 0.01)) * 2.0

        return {
            "ka": ka,
            "nu": nu,
            "estimated_rt_min": estimated_rt,
        }

    def predict_developability(
        self,
        hc_sequence: Optional[str] = None,
        lc_sequence: Optional[str] = None,
        sequence: Optional[str] = None,
        biophys: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Drop-in replacement for DevelopabilityPredictor.predict().

        Returns
        -------
        dict matching {"agg_risk": float, "stability": float, "viscosity_risk": float}
        """
        hc = hc_sequence or sequence or ""
        lc = lc_sequence or sequence or ""
        preds = self._predict_all_raw(hc, lc, biophys)

        return {
            "agg_risk": preds.get("aggregation_risk", 0.5),
            "stability": preds.get("stability", 0.5),
            "viscosity_risk": preds.get("viscosity_risk", 0.5),
        }

    def predict_wetlab(
        self,
        hc_sequence: Optional[str] = None,
        lc_sequence: Optional[str] = None,
        sequence: Optional[str] = None,
        biophys: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Drop-in replacement for WetLabPredictor.predict_single().

        Returns
        -------
        dict matching {"Exp_Aggregation_Percent": float, "Exp_Tm_MeltingTemp": float}
        """
        hc = hc_sequence or sequence or ""
        lc = lc_sequence or sequence or ""
        preds = self._predict_all_raw(hc, lc, biophys)

        # Map unified task outputs to WetLab format
        agg_risk = preds.get("aggregation_risk", 0.5)
        tm = preds.get("tm", 65.0)

        return {
            "Exp_Aggregation_Percent": agg_risk * 100.0,  # Convert [0,1] to %
            "Exp_Tm_MeltingTemp": tm,
        }

    def predict_potency(
        self,
        hc_sequence: Optional[str] = None,
        lc_sequence: Optional[str] = None,
        sequence: Optional[str] = None,
        biophys: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Drop-in replacement for PotencyPredictor.predict_single().

        Returns
        -------
        dict matching {"Predicted_Potency_Score": float}
        """
        hc = hc_sequence or sequence or ""
        lc = lc_sequence or sequence or ""
        preds = self._predict_all_raw(hc, lc, biophys)

        return {
            "Predicted_Potency_Score": preds.get("potency", 0.5),
        }

    def predict_all(
        self,
        hc_sequence: str,
        lc_sequence: str,
        biophys: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Return all 8 unified predictions in a single call.

        This is the new unified API — not a replacement for any
        existing predictor, but a superset of all.

        Returns
        -------
        dict with all 8 task predictions + metadata
        """
        preds = self._predict_all_raw(hc_sequence, lc_sequence, biophys)

        # Compute composite developability score
        agg = preds.get("aggregation_risk", 0.5)
        stab = preds.get("stability", 0.5)
        visc = preds.get("viscosity_risk", 0.5)
        dev_score = 0.40 * agg + 0.30 * (1.0 - stab) + 0.30 * visc

        if dev_score < 0.3:
            grade = "Low Risk"
        elif dev_score < 0.6:
            grade = "Medium Risk"
        else:
            grade = "High Risk"

        return {
            "predictions": preds,
            "developability_score": dev_score,
            "developability_grade": grade,
            "ml_override": {
                "ka": preds.get("ka", 1.0),
                "nu": preds.get("nu", 3.0),
            },
            "model": "UnifiedMultiTaskModel",
            "model_path": self._model_path,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_ADAPTER: Optional[MultiTaskAdapter] = None


def get_adapter(model_path: str = "models/unified_multitask_best.pt") -> MultiTaskAdapter:
    """Get or create the global MultiTaskAdapter singleton."""
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = MultiTaskAdapter(model_path=model_path)
    return _ADAPTER
