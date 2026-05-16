"""
data_curator.py  ·  ProtePilot — STEP 2
===========================================================
Deterministic Data Curation Pipeline for Antibody Datasets

Version   : 1.0
Depends   : numpy, pandas, biopython (optional), pLM_embedder

Pipeline
------------------------------------------------------------
  1. Load CSV (Jain-137, TheraSAbDab, or generic)
  2. Detect & validate schema
  3. Dual-chain fusion: VH + (G4S)3 linker + VL → single Fv
  4. ESM-2 embedding of fused Fv (480-dim, CPU)
  5. Biophysical feature extraction (7-dim)
  6. Assemble feature matrix (487-dim) + target matrix
  7. Optionally save to .npz

All operations are 100% deterministic (seed-controlled).
No LLM inference. Pure Python.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ProtePilot.DataCurator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G4S3_LINKER = "GGGGSGGGGSGGGGS"          # (G4S)3 — 15 amino acids
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
ESM2_DIM = 480                            # ESM-2 t12_35M mean-pooled output
BIOPHYS_DIM = 7
FEATURE_DIM = ESM2_DIM + BIOPHYS_DIM     # 327

BIOPHYS_NAMES = [
    "pI", "MW_kDa", "deam_sites", "ox_sites",
    "acidic_residues", "basic_residues", "hydrophobicity",
]

FEATURE_NAMES = [f"esm2_{i}" for i in range(ESM2_DIM)] + BIOPHYS_NAMES

# Jain-137 target column mapping (CSV header → internal name)
JAIN137_TARGET_MAP = {
    "Fab Tm by DSF (°C)":       "Fab_Tm",
    "HIC Retention Time (Min)a": "HIC_RT",
    "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average": "AC_SINS",
    "Poly-Specificity Reagent (PSR) SMP Score (0-1)": "PSR",
    "Slope for Accelerated Stability": "Stability_Slope",
}

# Kyte-Doolittle hydrophobicity scale
_KD_SCALE = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_sequence(seq: str) -> str:
    """Remove non-standard amino acid characters, upper-case."""
    return "".join(c for c in seq.upper() if c in STANDARD_AA)


# v32.1: Consolidated — import canonical implementations from feature_registry.
from src.feature_registry import (
    _compute_gravy,
    _count_deamidation_sites as _count_deamidation_sites_full,
    _count_oxidation_sites as _count_oxidation_sites_full,
)

def _count_deamidation_sites(seq: str) -> int:
    """Thin wrapper: returns count only (feature_registry returns Tuple)."""
    return _count_deamidation_sites_full(seq)[0]

def _count_oxidation_sites(seq: str) -> int:
    """Thin wrapper: returns count only (feature_registry returns Tuple)."""
    return _count_oxidation_sites_full(seq)[0]


def _compute_biophysical_biopython(seq: str) -> np.ndarray:
    """Compute 7 biophysical features using Biopython ProteinAnalysis."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    pa = ProteinAnalysis(seq)
    pI = pa.isoelectric_point()
    mw_kda = pa.molecular_weight() / 1000.0
    gravy = pa.gravy()
    deam = _count_deamidation_sites(seq)
    ox = _count_oxidation_sites(seq)
    acidic = seq.count("D") + seq.count("E")
    basic = seq.count("K") + seq.count("R") + seq.count("H")
    # Normalize GRAVY to [0,1] for the feature vector
    hydro_norm = max(0.0, min(1.0, (gravy + 2.0) / 4.0))
    return np.array([pI, mw_kda, deam, ox, acidic, basic, hydro_norm],
                    dtype=np.float32)


def _compute_biophysical_heuristic(seq: str) -> np.ndarray:
    """Fallback biophysical features without Biopython."""
    n = len(seq)
    acidic = seq.count("D") + seq.count("E")
    basic = seq.count("K") + seq.count("R") + seq.count("H")
    # Crude pI estimate from net charge
    net = basic - acidic
    pI = 7.0 + 0.5 * net / max(n, 1) * 100
    pI = max(4.0, min(11.0, pI))
    mw_kda = n * 0.110  # average aa MW ~110 Da
    deam = _count_deamidation_sites(seq)
    ox = _count_oxidation_sites(seq)
    gravy = _compute_gravy(seq)
    hydro_norm = max(0.0, min(1.0, (gravy + 2.0) / 4.0))
    return np.array([pI, mw_kda, deam, ox, acidic, basic, hydro_norm],
                    dtype=np.float32)


def compute_biophysical(seq: str) -> np.ndarray:
    """
    Compute 7-dim biophysical feature vector from sequence.

    Features: [pI, MW_kDa, deam_sites, ox_sites, acidic, basic, hydro_norm]
    Falls back to heuristic if Biopython unavailable.
    """
    try:
        return _compute_biophysical_biopython(seq)
    except ImportError:
        log.warning("Biopython unavailable — using heuristic biophysical features")
        return _compute_biophysical_heuristic(seq)
    except Exception as e:
        log.warning("Biopython failed (%s) — using heuristic fallback", e)
        return _compute_biophysical_heuristic(seq)


# ---------------------------------------------------------------------------
# Schema Detection
# ---------------------------------------------------------------------------

def _detect_schema(columns: List[str]) -> str:
    """Detect dataset schema from column names."""
    col_lower = {c.lower().strip() for c in columns}
    # Jain-137: has "VH", "VL", "Fab Tm"
    if "vh" in col_lower and "vl" in col_lower:
        return "jain137"
    # TheraSAbDab: has "HeavySequence", "LightSequence"
    if any("heavysequence" in c for c in col_lower):
        return "therasabdab"
    # Generic: look for common sequence column names
    if any(k in col_lower for k in ("hc_sequence", "lc_sequence", "sequence_hc")):
        return "generic"
    return "unknown"


def _get_vh_vl(row: dict, schema: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract VH and VL sequences from a row based on schema."""
    if schema == "jain137":
        return row.get("VH"), row.get("VL")
    elif schema == "therasabdab":
        return row.get("HeavySequence"), row.get("LightSequence")
    else:
        # Try common aliases
        vh = row.get("VH") or row.get("hc_sequence") or row.get("Sequence_HC")
        vl = row.get("VL") or row.get("lc_sequence") or row.get("Sequence_LC")
        return vh, vl


# ---------------------------------------------------------------------------
# DataCurator
# ---------------------------------------------------------------------------

class DataCurator:
    """
    Deterministic data curation pipeline for antibody datasets.

    Loads CSV, fuses VH+VL chains with (G4S)3 linker, computes ESM-2
    embeddings (480-dim) and biophysical features (7-dim), producing
    a standardized feature matrix (N, 487) + target matrix.

    Parameters
    ----------
    dataset_path : Path to CSV file (Jain-137, TheraSAbDab, or generic)
    seed         : Random seed for reproducibility (default 42)
    linker       : Peptide linker for chain fusion (default G4S×3)
    """

    def __init__(self, dataset_path: str, seed: int = 42,
                 linker: str = G4S3_LINKER):
        self.dataset_path = os.path.abspath(dataset_path)
        self.seed = seed
        self.linker = linker
        self._embedder = None
        self._curated = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_dataset(self) -> Dict[str, Any]:
        """
        Load CSV and detect schema.

        Returns
        -------
        dict with keys: status, rows (list of dicts), schema, n_rows, columns
        """
        import csv

        if not os.path.exists(self.dataset_path):
            return {"status": "error",
                    "message": f"File not found: {self.dataset_path}"}

        rows = []
        with open(self.dataset_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            for row in reader:
                # Strip whitespace from values
                rows.append({k: v.strip() if isinstance(v, str) else v
                             for k, v in row.items()})

        schema = _detect_schema(columns)
        log.info("Loaded %d rows from %s (schema: %s)",
                 len(rows), os.path.basename(self.dataset_path), schema)

        return {
            "status": "success",
            "rows": rows,
            "schema": schema,
            "n_rows": len(rows),
            "columns": columns,
        }

    # ------------------------------------------------------------------
    # Chain Fusion
    # ------------------------------------------------------------------

    def fuse_chains(self, vh: str, vl: str) -> str:
        """
        Fuse heavy and light chains via (G4S)3 linker.

        Returns: VH + GGGGSGGGGSGGGGS + VL (single Fv sequence)
        """
        vh_clean = _clean_sequence(vh)
        vl_clean = _clean_sequence(vl)
        if not vh_clean or not vl_clean:
            raise ValueError(
                f"Empty chain after cleaning: VH={len(vh_clean)}aa, "
                f"VL={len(vl_clean)}aa"
            )
        return vh_clean + self.linker + vl_clean

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-load ESM-2 embedder (singleton)."""
        if self._embedder is None:
            try:
                import sys
                src_dir = os.path.dirname(os.path.abspath(__file__))
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)
                from pLM_embedder import get_embedder
                self._embedder = get_embedder()
            except ImportError:
                log.warning("pLM_embedder unavailable — using minimal mock")
                self._embedder = None
        return self._embedder

    def embed_fused(self, fused_seq: str) -> np.ndarray:
        """
        Embed fused Fv sequence via ESM-2 (480-dim).

        Uses pLM_embedder.embed_sequence() with CPU fallback to mock
        embeddings if transformers is unavailable.

        Returns: np.ndarray shape (480,), dtype float32
        """
        embedder = self._get_embedder()
        if embedder is not None:
            emb = embedder.embed_sequence(fused_seq)
            if emb.shape[0] != ESM2_DIM:
                raise ValueError(
                    f"Embedding dim mismatch: expected {ESM2_DIM}, "
                    f"got {emb.shape[0]}"
                )
            return emb.astype(np.float32)
        else:
            # Minimal mock: AA composition + hash features
            from pLM_embedder import mock_embedding
            return mock_embedding(fused_seq, ESM2_DIM).astype(np.float32)

    # ------------------------------------------------------------------
    # Target Extraction
    # ------------------------------------------------------------------

    def extract_targets(self, row: dict, schema: str) -> Dict[str, float]:
        """
        Extract training targets from a CSV row.

        Returns dict of {target_name: float_value} for non-null targets.
        """
        targets = {}

        if schema == "jain137":
            for csv_col, target_name in JAIN137_TARGET_MAP.items():
                val = row.get(csv_col)
                if val is not None and val != "" and val != "N/A":
                    try:
                        targets[target_name] = float(val)
                    except (ValueError, TypeError):
                        pass

        elif schema == "therasabdab":
            # TheraSAbDab has limited numeric targets
            # Use computed properties instead
            pass

        return targets

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    def curate(self) -> Dict[str, Any]:
        """
        Full curation pipeline.

        Returns
        -------
        dict with:
          - status: "success" or "error"
          - X: np.ndarray (n_samples, 327) feature matrix
          - y: np.ndarray (n_samples, n_targets) target matrix
          - target_names: list of target names
          - feature_names: list of 327 feature names
          - sample_names: list of molecule names
          - n_samples, n_features: int
          - metadata: dict
        """
        if self._curated is not None:
            return self._curated

        t0 = time.time()

        # 1. Load
        loaded = self.load_dataset()
        if loaded["status"] != "success":
            return loaded

        rows = loaded["rows"]
        schema = loaded["schema"]

        # 2. Process each row
        embeddings = []
        biophys_list = []
        target_rows = []
        sample_names = []
        skipped = 0

        for i, row in enumerate(rows):
            vh, vl = _get_vh_vl(row, schema)
            if not vh or not vl:
                skipped += 1
                continue

            # 2a. Fuse chains
            try:
                fused = self.fuse_chains(vh, vl)
            except ValueError as e:
                log.warning("Row %d: chain fusion failed: %s", i, e)
                skipped += 1
                continue

            # 2b. ESM-2 embedding (480-dim)
            emb = self.embed_fused(fused)
            embeddings.append(emb)

            # 2c. Biophysical features (7-dim) — from fused sequence
            biophys = compute_biophysical(fused)
            biophys_list.append(biophys)

            # 2d. Targets
            targets = self.extract_targets(row, schema)
            target_rows.append(targets)

            # 2e. Name
            name = (row.get("Name") or row.get("Therapeutic")
                    or row.get("name") or f"sample_{i}")
            sample_names.append(name)

        if not embeddings:
            return {"status": "error",
                    "message": "No valid samples after curation"}

        # 3. Assemble feature matrix
        X_emb = np.stack(embeddings)           # (N, 480)
        X_bio = np.stack(biophys_list)         # (N, 7)
        X = np.hstack([X_emb, X_bio])         # (N, 487)
        assert X.shape[1] == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {X.shape[1]}"

        # 4. Assemble target matrix
        # Collect all target names that appear at least once
        all_target_names = []
        if schema == "jain137":
            all_target_names = list(JAIN137_TARGET_MAP.values())
        else:
            # Infer from data
            seen = set()
            for t in target_rows:
                seen.update(t.keys())
            all_target_names = sorted(seen)

        # Filter to targets with >50% coverage
        n = len(target_rows)
        valid_targets = []
        for tname in all_target_names:
            count = sum(1 for t in target_rows if tname in t)
            if count > n * 0.5:
                valid_targets.append(tname)
            else:
                log.info("Target '%s' has only %d/%d coverage (%.0f%%) — skipping",
                         tname, count, n, 100 * count / n)

        if not valid_targets:
            return {"status": "error",
                    "message": "No targets with >50% coverage found"}

        # Build y matrix (NaN for missing values)
        y = np.full((n, len(valid_targets)), np.nan, dtype=np.float32)
        for i, t in enumerate(target_rows):
            for j, tname in enumerate(valid_targets):
                if tname in t:
                    y[i, j] = t[tname]

        # Remove rows with all-NaN targets
        valid_mask = ~np.all(np.isnan(y), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        sample_names = [s for s, m in zip(sample_names, valid_mask) if m]

        # Impute remaining NaN with column median
        for j in range(y.shape[1]):
            col = y[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any() and not nan_mask.all():
                median_val = np.nanmedian(col)
                y[nan_mask, j] = median_val
                log.info("Imputed %d NaN values for '%s' with median=%.3f",
                         nan_mask.sum(), valid_targets[j], median_val)

        elapsed = time.time() - t0

        self._curated = {
            "status": "success",
            "X": X,
            "y": y,
            "target_names": valid_targets,
            "feature_names": FEATURE_NAMES,
            "sample_names": sample_names,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_targets": len(valid_targets),
            "metadata": {
                "dataset_path": self.dataset_path,
                "schema": schema,
                "seed": self.seed,
                "linker": self.linker,
                "linker_length": len(self.linker),
                "n_skipped": skipped,
                "n_total_rows": len(rows),
                "curation_time_s": round(elapsed, 2),
                "embedding_dim": ESM2_DIM,
                "biophys_dim": BIOPHYS_DIM,
            },
        }

        log.info(
            "Curation complete: %d samples, %d features, %d targets, "
            "%.1fs (skipped %d rows)",
            X.shape[0], X.shape[1], len(valid_targets), elapsed, skipped,
        )
        return self._curated

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_features(self, output_path: str) -> str:
        """
        Save curated features to .npz file.

        Returns absolute path to saved file.
        """
        result = self.curate()
        if result["status"] != "success":
            raise RuntimeError(f"Curation failed: {result.get('message')}")

        abs_path = os.path.abspath(output_path)
        np.savez_compressed(
            abs_path,
            X=result["X"],
            y=result["y"],
            target_names=np.array(result["target_names"], dtype=object),
            feature_names=np.array(result["feature_names"], dtype=object),
            sample_names=np.array(result["sample_names"], dtype=object),
        )
        size_kb = os.path.getsize(abs_path) / 1024
        log.info("Features saved: %s (%.1f KB)", abs_path, size_kb)
        return abs_path


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  DataCurator v1.0 — Self-Test")
    print("=" * 60)

    # Find Jain-137 CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "Jain137_Cleaned_Training_Data.csv")

    if not os.path.exists(csv_path):
        print(f"  SKIP: Jain-137 CSV not found at {csv_path}")
        import sys
        sys.exit(0)

    curator = DataCurator(csv_path, seed=42)

    # Test 1: Load
    print("\n[1/5] Loading dataset...")
    loaded = curator.load_dataset()
    assert loaded["status"] == "success", f"Load failed: {loaded}"
    assert loaded["schema"] == "jain137", f"Schema: {loaded['schema']}"
    print(f"  ✓ Loaded {loaded['n_rows']} rows (schema: {loaded['schema']})")

    # Test 2: Chain fusion
    print("\n[2/5] Testing chain fusion...")
    row0 = loaded["rows"][0]
    fused = curator.fuse_chains(row0["VH"], row0["VL"])
    expected_len = len(_clean_sequence(row0["VH"])) + len(G4S3_LINKER) + len(_clean_sequence(row0["VL"]))
    assert len(fused) == expected_len, f"Fused length {len(fused)} != {expected_len}"
    assert G4S3_LINKER in fused, "Linker not found in fused sequence"
    print(f"  ✓ Fused: {len(fused)}aa (VH={len(_clean_sequence(row0['VH']))}"
          f" + linker={len(G4S3_LINKER)} + VL={len(_clean_sequence(row0['VL']))})")

    # Test 3: Biophysical
    print("\n[3/5] Testing biophysical features...")
    biophys = compute_biophysical(fused)
    assert biophys.shape == (7,), f"Shape: {biophys.shape}"
    assert 4.0 < biophys[0] < 12.0, f"pI out of range: {biophys[0]}"
    assert biophys[1] > 0, f"MW <= 0: {biophys[1]}"
    print(f"  ✓ pI={biophys[0]:.2f}, MW={biophys[1]:.1f} kDa, "
          f"deam={biophys[2]:.0f}, ox={biophys[3]:.0f}, hydro={biophys[6]:.3f}")

    # Test 4: Full curation
    print("\n[4/5] Running full curation pipeline...")
    result = curator.curate()
    assert result["status"] == "success", f"Curation failed: {result}"
    X = result["X"]
    y = result["y"]
    assert X.shape[1] == FEATURE_DIM, f"X dim: {X.shape[1]} != {FEATURE_DIM}"
    assert len(result["target_names"]) == y.shape[1]
    assert np.isfinite(X).all(), "X contains non-finite values"
    assert np.isfinite(y).all(), "y contains non-finite values"
    print(f"  ✓ X: {X.shape}, y: {y.shape}")
    print(f"    Targets: {result['target_names']}")
    print(f"    Curation time: {result['metadata']['curation_time_s']}s")

    # Test 5: Determinism
    print("\n[5/5] Testing determinism...")
    curator2 = DataCurator(csv_path, seed=42)
    result2 = curator2.curate()
    assert np.allclose(result["X"], result2["X"]), "X not deterministic!"
    assert np.allclose(result["y"], result2["y"]), "y not deterministic!"
    print("  ✓ Deterministic: two runs produce identical features")

    print(f"\nSelf-test: 5/5 passed")
