"""
unified_dataset.py  ·  ProtePilot — Unified Integration
===========================================================
Unified Antibody Dataset with Masked Multi-Task Labels

Supports:
  - HC / LC separate sequence columns
  - 8 task labels (any can be missing → auto-masked)
  - 7 biophysical feature columns (optional, derived from sequence if missing)
  - Compatible with Jain-137 format and ProtePilot calibration results

Origin: Extends Biologics AI AntibodyDataset mask mechanism.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

log = logging.getLogger("ProtePilot.UnifiedDataset")

# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------
# Task label columns (order must match UNIFIED_TASKS in unified_multitask_model.py)
TASK_COLUMNS = [
    "ka", "nu", "tm", "aggregation_risk",
    "stability", "viscosity_risk", "hydrophobicity", "potency",
]

# Biophysical feature columns
BIOPHYS_COLUMNS = [
    "pI", "MW_kDa", "deam_sites", "ox_sites",
    "acidic_residues", "basic_residues", "hydrophobicity_gravy",
]

# Sequence columns
HC_COL = "hc_sequence"
LC_COL = "lc_sequence"
SINGLE_SEQ_COL = "sequence"  # Fallback: use same sequence for HC and LC


# ---------------------------------------------------------------------------
# Simple biophysical feature extraction (when not in CSV)
# ---------------------------------------------------------------------------
def _extract_biophys_from_sequence(seq: str) -> Dict[str, float]:
    """
    Compute basic biophysical features from an amino acid sequence.
    Returns approximate values suitable for model input.
    """
    seq = seq.upper()
    n = max(len(seq), 1)

    # Approximate amino acid properties
    acidic = seq.count("D") + seq.count("E")
    basic = seq.count("K") + seq.count("R") + seq.count("H")

    # Crude pI approximation
    net_charge_ph7 = basic - acidic
    pI = 7.0 + net_charge_ph7 * 0.05  # Rough linear approx
    pI = max(4.0, min(12.0, pI))

    # MW approximation (average AA weight ~110 Da)
    mw_kda = n * 0.110

    # Deamidation / oxidation sites (simplified)
    deam_sites = seq.count("N")  # Simplified: count Asn
    ox_sites = seq.count("M") + seq.count("W")  # Met + Trp

    # GRAVY hydrophobicity (Kyte-Doolittle scale, normalized)
    kd_scale = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }
    gravy = sum(kd_scale.get(aa, 0) for aa in seq) / n
    # Normalize to ~[0, 1]
    hydro_norm = (gravy + 2.0) / 4.0
    hydro_norm = max(0.0, min(1.0, hydro_norm))

    return {
        "pI": pI,
        "MW_kDa": mw_kda,
        "deam_sites": float(deam_sites),
        "ox_sites": float(ox_sites),
        "acidic_residues": float(acidic),
        "basic_residues": float(basic),
        "hydrophobicity_gravy": hydro_norm,
    }


# ---------------------------------------------------------------------------
# Unified Dataset
# ---------------------------------------------------------------------------
class UnifiedAntibodyDataset(Dataset):
    """
    PyTorch dataset for multi-task antibody property prediction.

    Reads a CSV with optional columns for:
      - ``hc_sequence``, ``lc_sequence`` (or ``sequence`` as fallback)
      - Task labels: ``ka``, ``nu``, ``tm``, ``aggregation_risk``,
        ``stability``, ``viscosity_risk``, ``hydrophobicity``, ``potency``
      - Biophysical features: ``pI``, ``MW_kDa``, ``deam_sites``,
        ``ox_sites``, ``acidic_residues``, ``basic_residues``,
        ``hydrophobicity_gravy``

    Missing task labels are automatically masked (mask=0) so the loss
    function ignores them during training.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    tasks : list of str, optional
        Subset of tasks to load. Default: all 8 tasks.
    compute_biophys : bool
        If True and biophysical columns are missing, compute them
        from the sequence. Default True.
    """

    def __init__(
        self,
        csv_path: str,
        tasks: Optional[List[str]] = None,
        compute_biophys: bool = True,
        embedding_cache_path: Optional[str] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.tasks = tasks or TASK_COLUMNS
        self.compute_biophys = compute_biophys

        # Load pre-computed ESM-2 embeddings if available
        self.cached_embeddings = None
        if embedding_cache_path is not None:
            import torch as _torch
            cache = _torch.load(embedding_cache_path, map_location="cpu", weights_only=False)
            self.cached_embeddings = cache["embeddings"]  # (N, 960)
            assert len(self.cached_embeddings) == len(self.df), \
                f"Cache size {len(self.cached_embeddings)} != CSV size {len(self.df)}"
            log.info(f"Loaded cached ESM-2 embeddings: {self.cached_embeddings.shape}")

        # Detect sequence columns
        self.has_dual_seq = HC_COL in self.df.columns and LC_COL in self.df.columns
        if not self.has_dual_seq and SINGLE_SEQ_COL not in self.df.columns:
            raise ValueError(
                f"CSV must have either ({HC_COL}, {LC_COL}) or ({SINGLE_SEQ_COL}) columns"
            )

        # Compute biophysical features if needed
        self.has_biophys = all(c in self.df.columns for c in BIOPHYS_COLUMNS)
        if not self.has_biophys and compute_biophys:
            log.info("Biophysical columns not found; computing from sequences")
            self._compute_biophys_columns()

        log.info(
            f"Loaded {len(self.df)} samples, "
            f"tasks={self.tasks}, "
            f"dual_seq={self.has_dual_seq}, "
            f"biophys={'csv' if self.has_biophys else 'computed'}"
        )

    def _compute_biophys_columns(self) -> None:
        """Derive biophysical features from sequences."""
        seq_col = HC_COL if self.has_dual_seq else SINGLE_SEQ_COL
        feats = self.df[seq_col].apply(
            lambda s: _extract_biophys_from_sequence(str(s))
        )
        feats_df = pd.DataFrame(feats.tolist())
        for col in BIOPHYS_COLUMNS:
            if col not in self.df.columns and col in feats_df.columns:
                self.df[col] = feats_df[col]
        self.has_biophys = True

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, str, Dict[str, Tensor], Dict[str, Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Returns
        -------
        hc_seq : str
        lc_seq : str
        labels : Dict[str, Tensor]
            Task labels (0.0 if missing).
        mask : Dict[str, Tensor]
            1.0 if label present, 0.0 if missing.
        biophys : Tensor or None
            (7,) biophysical feature vector, or None.
        cached_emb : Tensor or None
            (960,) pre-computed ESM-2 embedding, or None.
        """
        row = self.df.iloc[idx]

        # Sequences
        if self.has_dual_seq:
            hc_seq = str(row[HC_COL])
            lc_seq = str(row[LC_COL])
        else:
            seq = str(row[SINGLE_SEQ_COL])
            hc_seq = seq
            lc_seq = seq

        # Task labels + masks
        labels = {}
        mask = {}
        for task in self.tasks:
            if task in self.df.columns:
                value = row[task]
                if pd.isna(value):
                    labels[task] = torch.tensor(0.0, dtype=torch.float32)
                    mask[task] = torch.tensor(0.0, dtype=torch.float32)
                else:
                    labels[task] = torch.tensor(float(value), dtype=torch.float32)
                    mask[task] = torch.tensor(1.0, dtype=torch.float32)
            else:
                labels[task] = torch.tensor(0.0, dtype=torch.float32)
                mask[task] = torch.tensor(0.0, dtype=torch.float32)

        # Biophysical features
        biophys = None
        if self.has_biophys:
            biophys_vals = []
            for col in BIOPHYS_COLUMNS:
                val = row.get(col, 0.0)
                biophys_vals.append(0.0 if pd.isna(val) else float(val))
            biophys = torch.tensor(biophys_vals, dtype=torch.float32)

        # Cached ESM-2 embedding
        cached_emb = None
        if self.cached_embeddings is not None:
            cached_emb = self.cached_embeddings[idx]

        return hc_seq, lc_seq, labels, mask, biophys, cached_emb


def unified_collate_fn(batch):
    """
    Custom collate function for UnifiedAntibodyDataset.

    Returns
    -------
    hc_seqs : List[str]
    lc_seqs : List[str]
    labels : Dict[str, Tensor(B,)]
    masks : Dict[str, Tensor(B,)]
    biophys : Tensor(B, 7) or None
    cached_emb : Tensor(B, 960) or None
    """
    hc_seqs, lc_seqs, labels_list, masks_list, biophys_list, emb_list = zip(*batch)

    hc_seqs = list(hc_seqs)
    lc_seqs = list(lc_seqs)

    # Stack labels and masks
    task_names = labels_list[0].keys()
    labels = {t: torch.stack([l[t] for l in labels_list]) for t in task_names}
    masks = {t: torch.stack([m[t] for m in masks_list]) for t in task_names}

    # Stack biophys features
    if biophys_list[0] is not None:
        biophys = torch.stack(biophys_list)
    else:
        biophys = None

    # Stack cached ESM-2 embeddings
    if emb_list[0] is not None:
        cached_emb = torch.stack(emb_list)
    else:
        cached_emb = None

    return hc_seqs, lc_seqs, labels, masks, biophys, cached_emb
