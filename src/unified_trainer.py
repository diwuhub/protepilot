"""
unified_trainer.py  ·  ProtePilot — Unified Integration
===========================================================
Multi-Task Trainer with Masked Loss and Early Stopping

Supports training the UnifiedMultiTaskModel with:
  - Masked loss: only computes gradients for tasks with valid labels
  - Task-level loss weighting
  - Early stopping on validation loss
  - Best model checkpointing
  - Per-task metrics tracking

Origin: Extends Biologics AI Trainer with ProtePilot-specific enhancements.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

log = logging.getLogger("ProtePilot.UnifiedTrainer")


# ---------------------------------------------------------------------------
# Task normalization (target_range) and weights
# ---------------------------------------------------------------------------
# Approximate target ranges for each task — used to normalize MSE loss
# so that all tasks contribute equally regardless of their natural scale.
TASK_TARGET_STATS = {
    "ka":               {"mean": 30.0,  "std": 20.0},   # range ~0.1-100
    "nu":               {"mean": 4.5,   "std": 2.0},    # range ~1-20
    "tm":               {"mean": 68.0,  "std": 8.0},    # range ~40-95
    "aggregation_risk": {"mean": 0.2,   "std": 0.15},   # range 0-1
    "stability":        {"mean": 0.8,   "std": 0.15},   # range 0-1
    "viscosity_risk":   {"mean": 0.2,   "std": 0.15},   # range 0-1
    "hydrophobicity":   {"mean": 0.35,  "std": 0.1},    # range 0-1
    "potency":          {"mean": 0.7,   "std": 0.2},    # range 0-1
}

DEFAULT_TASK_WEIGHTS = {
    "ka": 1.0,
    "nu": 1.0,
    "tm": 1.0,
    "aggregation_risk": 1.0,
    "stability": 1.0,
    "viscosity_risk": 1.0,
    "hydrophobicity": 0.5,   # Lower weight — easy to predict
    "potency": 1.0,
}


# ---------------------------------------------------------------------------
# Unified Trainer
# ---------------------------------------------------------------------------
class UnifiedTrainer:
    """
    Multi-task trainer for UnifiedMultiTaskModel.

    Parameters
    ----------
    model : nn.Module
        The UnifiedMultiTaskModel to train.
    train_loader : DataLoader
        Training data loader (using unified_collate_fn).
    val_loader : DataLoader
        Validation data loader.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    task_weights : dict, optional
        Per-task loss weights. Default: equal weights.
    device : str
        Device for training.
    patience : int
        Early stopping patience (epochs without improvement).
    save_dir : str
        Directory for model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        task_weights: Optional[Dict[str, float]] = None,
        device: str = "cpu",
        patience: int = 10,
        save_dir: str = "models",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.save_dir = save_dir

        self.task_weights = task_weights or DEFAULT_TASK_WEIGHTS
        self.criterion = nn.MSELoss(reduction="none")  # Per-element loss

        # Training state
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.counter = 0
        self.history: List[Dict[str, Any]] = []

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        labels: Dict[str, Tensor],
        masks: Dict[str, Tensor],
    ) -> Optional[Tensor]:
        """
        Compute masked multi-task loss.

        Only tasks with valid labels (mask=1) contribute to the loss.
        Each task's loss is weighted by task_weights.

        Returns None if no valid labels exist in the batch.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        valid_tasks = 0

        for task_name in outputs:
            if task_name not in labels or task_name not in masks:
                continue

            task_mask = masks[task_name].to(self.device)
            n_valid = task_mask.sum()

            if n_valid < 1:
                continue

            pred = outputs[task_name].to(self.device)
            target = labels[task_name].to(self.device)

            # Per-element MSE, masked
            loss_elements = self.criterion(pred, target)  # (B,)
            masked_loss = (loss_elements * task_mask).sum() / n_valid

            # Normalize by task target variance so all tasks contribute equally
            # regardless of their natural scale (e.g., ka ~0-100 vs risk ~0-1)
            stats = TASK_TARGET_STATS.get(task_name)
            if stats is not None:
                masked_loss = masked_loss / (stats["std"] ** 2 + 1e-8)

            # Apply task weight
            weight = self.task_weights.get(task_name, 1.0)
            total_loss = total_loss + masked_loss * weight
            valid_tasks += 1

        if valid_tasks == 0:
            return None

        return total_loss / valid_tasks

    def _run_epoch(self, loader: DataLoader, training: bool = True) -> Dict[str, float]:
        """Run one epoch (training or validation)."""
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        n_batches = 0
        task_losses = {}
        task_counts = {}

        ctx = torch.no_grad() if not training else _nullcontext()
        with ctx:
            for batch in loader:
                hc_seqs, lc_seqs, labels, masks, biophys, cached_emb = batch

                # Move tensors to device
                biophys_dev = biophys.to(self.device) if biophys is not None else None
                cached_emb_dev = cached_emb.to(self.device) if cached_emb is not None else None
                labels_dev = {k: v.to(self.device) for k, v in labels.items()}
                masks_dev = {k: v.to(self.device) for k, v in masks.items()}

                outputs = self.model(hc_seqs, lc_seqs, biophys_dev,
                                     cached_esm2_emb=cached_emb_dev)
                loss = self.compute_loss(outputs, labels_dev, masks_dev)

                if loss is not None:
                    if training:
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                    # Track per-task losses (normalized)
                    for task_name in outputs:
                        if task_name in masks_dev and masks_dev[task_name].sum() > 0:
                            mask = masks_dev[task_name]
                            pred = outputs[task_name]
                            target = labels_dev[task_name]
                            t_loss = ((pred - target) ** 2 * mask).sum() / mask.sum()
                            stats = TASK_TARGET_STATS.get(task_name)
                            if stats is not None:
                                t_loss = t_loss / (stats["std"] ** 2 + 1e-8)
                            task_losses[task_name] = task_losses.get(task_name, 0.0) + t_loss.item()
                            task_counts[task_name] = task_counts.get(task_name, 0) + 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_task_losses = {
            t: task_losses[t] / task_counts[t]
            for t in task_losses
        }

        return {"loss": avg_loss, "task_losses": avg_task_losses, "n_batches": n_batches}

    def train(
        self,
        epochs: int = 100,
        save_best: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model with early stopping.

        Parameters
        ----------
        epochs : int
            Maximum number of training epochs.
        save_best : bool
            Whether to save the best model checkpoint.
        verbose : bool
            Print progress per epoch.

        Returns
        -------
        dict
            Training summary with history, best metrics, and model path.
        """
        log.info(f"Starting training: {epochs} epochs, patience={self.patience}")
        start_time = time.time()

        for epoch in range(epochs):
            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_task_losses": train_metrics["task_losses"],
                "val_task_losses": val_metrics["task_losses"],
            }
            self.history.append(epoch_record)

            if verbose:
                task_str = ", ".join(
                    f"{t}={val_metrics['task_losses'].get(t, 0):.4f}"
                    for t in sorted(val_metrics["task_losses"])
                )
                log.info(
                    f"Epoch {epoch:3d} | "
                    f"Train: {train_metrics['loss']:.4f} | "
                    f"Val: {val_metrics['loss']:.4f} | "
                    f"Tasks: [{task_str}]"
                )

            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.counter = 0

                if save_best:
                    save_path = f"{self.save_dir}/unified_multitask_best.pt"
                    torch.save(self.model.state_dict(), save_path)
                    log.info(f"  Saved best model (val_loss={self.best_val_loss:.6f})")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    log.info(f"Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Save final model
        final_path = f"{self.save_dir}/unified_multitask_latest.pt"
        torch.save(self.model.state_dict(), final_path)

        elapsed = time.time() - start_time

        summary = {
            "status": "success",
            "epochs_trained": len(self.history),
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.history[-1]["train_loss"] if self.history else None,
            "final_val_loss": self.history[-1]["val_loss"] if self.history else None,
            "elapsed_seconds": elapsed,
            "model_path_best": f"{self.save_dir}/unified_multitask_best.pt",
            "model_path_latest": final_path,
            "history": self.history,
        }

        log.info(
            f"Training complete: {summary['epochs_trained']} epochs, "
            f"best_val={self.best_val_loss:.6f}, time={elapsed:.1f}s"
        )

        return summary


# ---------------------------------------------------------------------------
# Context manager fallback for Python < 3.7
# ---------------------------------------------------------------------------
class _nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
