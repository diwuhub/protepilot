"""
unified_multitask_model.py  ·  ProtePilot — Unified Integration
===========================================================
8-Task Unified Multi-Task Model

Replaces 4 independent predictors (ChromatographyMLP, WetLabPredictor,
DevelopabilityXGB, PotencyPredictor) with a single shared-backbone model.

Architecture
------------------------------------------------------------
ESM2HybridEncoder (256-dim)
        ↓
Shared MLP: 256 → 128 → ReLU → Dropout → 128 → 64 → ReLU
        ↓
Dynamic Task Heads: {task: Linear(64, 1) for each task}
        ↓
Output: Dict[task_name, Tensor(B,)]

Tasks (8 total)
------------------------------------------------------------
  ka               - SMA adsorption rate (for CADET)
  nu               - SMA characteristic charge (for CADET)
  tm               - Melting temperature (°C)
  aggregation_risk - Aggregation risk [0, 1]
  stability        - Thermal stability [0, 1]
  viscosity_risk   - Viscosity risk [0, 1]
  hydrophobicity   - GRAVY-based hydrophobicity
  potency          - Binding potency score [0, 1]

Origin: Architecture from Biologics AI MultiTaskModel; tasks from ProtePilot.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.unified_trainer import TASK_TARGET_STATS

log = logging.getLogger("ProtePilot.UnifiedMultiTaskModel")

# ---------------------------------------------------------------------------
# Default task configuration
# ---------------------------------------------------------------------------
UNIFIED_TASKS = [
    "ka",                 # SMA adsorption rate → drives CADET
    "nu",                 # SMA characteristic charge → drives CADET
    "tm",                 # Melting temperature (°C)
    "aggregation_risk",   # Aggregation risk [0, 1]
    "stability",          # Thermal stability [0, 1]
    "viscosity_risk",     # Viscosity risk [0, 1]
    "hydrophobicity",     # GRAVY hydrophobicity
    "potency",            # Binding potency [0, 1]
]

# Tasks that should be clamped to [0, 1]
BOUNDED_TASKS = {"aggregation_risk", "stability", "viscosity_risk", "potency"}

# Tasks that need denormalization in inference (unbounded, normalized during training)
UNBOUNDED_TASKS = {"ka", "nu", "tm"}

# Task descriptions (for reporting / SHAP)
TASK_DESCRIPTIONS = {
    "ka": "SMA adsorption rate constant (m³/mol/s)",
    "nu": "SMA characteristic charge number",
    "tm": "Melting temperature (°C)",
    "aggregation_risk": "Aggregation risk score [0-1]",
    "stability": "Thermal stability score [0-1]",
    "viscosity_risk": "Viscosity risk score [0-1]",
    "hydrophobicity": "GRAVY hydrophobicity index",
    "potency": "Binding potency score [0-1]",
}


# ---------------------------------------------------------------------------
# Unified Multi-Task Model
# ---------------------------------------------------------------------------
class UnifiedMultiTaskModel(nn.Module):
    """
    Shared-backbone multi-task model for antibody property prediction.

    Parameters
    ----------
    encoder : nn.Module
        Backbone encoder (e.g., ESM2HybridEncoder) that outputs
        a fixed-size embedding vector.
    encoder_dim : int
        Dimensionality of the encoder output.
    tasks : list of str
        Task names. Each gets an independent prediction head.
    shared_hidden : int
        Hidden dimension in the shared MLP layers.
    head_hidden : int
        Hidden dimension before each task's output layer.
    dropout : float
        Dropout rate in the shared MLP.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 256,
        tasks: Optional[List[str]] = None,
        shared_hidden: int = 128,
        head_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.task_names = tasks or UNIFIED_TASKS

        # Shared representation layers
        self.shared = nn.Sequential(
            nn.Linear(encoder_dim, shared_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden, head_hidden),
            nn.ReLU(),
        )

        # Dynamic task heads — one per task
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(head_hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            for task in self.task_names
        })

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def num_tasks(self) -> int:
        return len(self.task_names)

    def forward(
        self,
        hc_seqs: List[str],
        lc_seqs: List[str],
        biophys_features: Optional[Tensor] = None,
        cached_esm2_emb: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through encoder → shared MLP → task heads.

        Parameters
        ----------
        hc_seqs : List[str]
            Heavy chain sequences.
        lc_seqs : List[str]
            Light chain sequences.
        biophys_features : Tensor, shape ``(batch, 7)``, optional
            Biophysical features.
        cached_esm2_emb : Tensor, shape ``(batch, 960)``, optional
            Pre-computed ESM-2 embeddings. Skips ESM-2 inference if provided.

        Returns
        -------
        Dict[str, Tensor]
            Mapping from task name → prediction tensor of shape ``(batch,)``.
        """
        # Encode (pass cached embeddings through if available)
        h = self.encoder(hc_seqs, lc_seqs, biophys_features,
                         cached_esm2_emb=cached_esm2_emb)  # (B, encoder_dim)

        # Shared representation
        shared_h = self.shared(h)  # (B, head_hidden)

        # Task-specific predictions
        outputs = {}
        for task_name, head in self.heads.items():
            out = head(shared_h).squeeze(-1)  # (B,)

            # Clamp bounded tasks to [0, 1]
            if task_name in BOUNDED_TASKS:
                out = torch.sigmoid(out)

            outputs[task_name] = out

        return outputs

    def predict_numpy(
        self,
        hc_seqs: List[str],
        lc_seqs: List[str],
        biophys_features: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single-sample prediction returning plain Python floats.

        Convenience method for inference in ProtePilot tools/pipelines.
        Denormalizes unbounded tasks (ka, nu, tm) using TASK_TARGET_STATS.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(hc_seqs, lc_seqs, biophys_features)

        # Convert to dict with denormalization for unbounded tasks
        result = {}
        for k, v in outputs.items():
            val = v.item() if v.numel() == 1 else v.cpu().numpy().tolist()

            # Denormalize unbounded tasks using TASK_TARGET_STATS
            if k in UNBOUNDED_TASKS and k in TASK_TARGET_STATS:
                stats = TASK_TARGET_STATS[k]
                val = val * stats["std"] + stats["mean"]

            result[k] = val

        return result

    def get_task_info(self) -> Dict[str, str]:
        """Return task names with descriptions."""
        return {t: TASK_DESCRIPTIONS.get(t, t) for t in self.task_names}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"tasks={self.task_names}, "
            f"num_tasks={self.num_tasks}, "
            f"encoder={self.encoder.__class__.__name__})"
        )
