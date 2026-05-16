"""
ht_screening.py  ·  ProtePilot — Milestone 16
===========================================================
High-Throughput Virtual Screening Engine

Version   : 1.0 (Early Discovery HT Screening)
Author    : Di (ProtePilot)
Depends   : numpy, biopython (optional), ml_predictor (v5.0),
             pLM_embedder (M8), data_pipeline (v2.0)

Purpose
------------------------------------------------------------
Bulk processing module that accepts CSV files containing hundreds
to thousands of candidate antibody sequences from Early Discovery.
For each candidate, the engine:

  1. Extracts ESM-2 pLM embeddings (or mock fallback)
  2. Computes Biopython-derived biophysical features (7-dim)
  3. Runs XGBoost developability model → Developability_Score
  4. Runs XGBoost potency model → Predicted_Potency_Score
  5. Classifies into Magic Quadrant (Top-Right = best)
  6. Ranks and filters candidates

Input CSV Schema (Discovery):
  - Candidate_ID       : Unique identifier for each candidate
  - Sequence_HC        : Heavy chain amino acid sequence
  - Sequence_LC        : Light chain amino acid sequence (optional)
  - Exp_ELISA_OD       : Optional ELISA OD binding signal
  - Exp_Kd_nM          : Optional SPR/BLI Kd in nM

Output:
  - Developability_Score (0-1, higher = more developable)
  - Predicted_Potency_Score (0-1, higher = more potent)
  - Quadrant classification (Star, Developable, Potent, Risky)
  - Rank within quadrant
  - Full feature details for downstream analysis

Architecture
------------------------------------------------------------
    CSV Upload (Candidate_ID, Seq_HC, Seq_LC)
          │
          ▼
    ┌─────────────────────────────────┐
    │  Per-candidate processing loop  │
    │  ┌───────────────┐              │
    │  │ ESM-2 Embed   │──→ (960,)   │
    │  └───────────────┘              │
    │  ┌───────────────┐              │
    │  │ Biopython     │──→ (7,)     │
    │  └───────────────┘              │
    │  ┌───────────────┐              │
    │  │ XGB Develop.  │──→ Score    │
    │  └───────────────┘              │
    │  ┌───────────────┐              │
    │  │ XGB Potency   │──→ Score    │
    │  └───────────────┘              │
    └─────────────────────────────────┘
          │
          ▼
    Magic Quadrant Classification
    ┌───────────┬───────────┐
    │ Developable│   STAR    │
    │  (low P,  │  (high D, │
    │   high D) │   high P) │
    ├───────────┼───────────┤
    │   RISKY   │  Potent   │
    │  (low D,  │  (low D,  │
    │   low P)  │   high P) │
    └───────────┴───────────┘
      low D ◄──── Developability ────► high D
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ProtePilot.HTScreening")


# ===========================================================================
# Constants
# ===========================================================================

DISCOVERY_CSV_COLUMNS = [
    "candidate_id", "sequence_hc", "sequence_lc",
    "exp_elisa_od", "exp_kd_nm",
]

# Column alias mapping for flexible CSV parsing
COLUMN_ALIASES: Dict[str, List[str]] = {
    "candidate_id": ["candidate_id", "id", "name", "sample_id", "clone_id", "antibody_id"],
    "sequence_hc": ["sequence_hc", "hc_sequence", "heavy_chain", "vh", "vh_sequence", "hc"],
    "sequence_lc": ["sequence_lc", "lc_sequence", "light_chain", "vl", "vl_sequence", "lc"],
    "exp_elisa_od": ["exp_elisa_od", "elisa_od", "elisa", "binding_od", "od450"],
    "exp_kd_nm": ["exp_kd_nm", "kd_nm", "kd", "affinity_kd", "kd_value"],
}

# Quadrant thresholds (configurable)
DEFAULT_DEV_THRESHOLD = 0.5
DEFAULT_POTENCY_THRESHOLD = 0.5

# Quadrant names
QUADRANT_STAR = "Star"              # High Developability + High Potency
QUADRANT_DEVELOPABLE = "Developable"  # High Developability + Low Potency
QUADRANT_POTENT = "Potent"          # Low Developability + High Potency
QUADRANT_RISKY = "Risky"            # Low Developability + Low Potency


# ===========================================================================
# Feature Extraction Helpers
# ===========================================================================

def _extract_biophysical_features(sequence: str) -> np.ndarray:
    """
    Extract 7-dim biophysical feature vector from amino acid sequence.

    Returns: np.ndarray of shape (7,) — same features as ChromatographyMLP input:
        [pI, MW_kDa, deam_sites, ox_sites, acidic_residues, basic_residues, hydrophobicity]
    """
    seq = re.sub(r'[^A-Z]', '', sequence.upper())
    if len(seq) < 10:
        return np.array([8.0, 25.0, 0, 0, 0, 0, 0.35], dtype=np.float32)

    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        pa = ProteinAnalysis(seq)
        pI = pa.isoelectric_point()
        mw_kda = pa.molecular_weight() / 1000.0
        gravy = pa.gravy()
    except Exception:
        pI = 8.0
        mw_kda = len(seq) * 0.11  # ~110 Da per residue
        gravy = -0.3

    hydrophobicity = max(0.0, min(1.0, (gravy + 2.0) / 4.0))
    deam_sites = len(re.findall(r"N[GS]", seq))
    ox_sites = seq.count("M") + seq.count("W")
    acidic = seq.count("D") + seq.count("E")
    basic = seq.count("K") + seq.count("R") + seq.count("H")

    return np.array([pI, mw_kda, deam_sites, ox_sites, acidic, basic, hydrophobicity],
                    dtype=np.float32)


def _compute_developability_score(features: np.ndarray) -> float:
    """
    Compute heuristic developability score from biophysical features.

    Score 0-1 where 1 = most developable. Used as fallback when XGBoost
    model is not trained.

    Factors (weighted):
      - pI in optimal range (6-9): good
      - Low hydrophobicity: reduces aggregation risk
      - Low liability count: fewer PTM hotspots
      - MW in typical mAb range (140-160 kDa): typical
    """
    pI, mw_kda, deam, ox, acidic, basic, hydro = features

    # pI penalty: optimal near 8.0 for IEX
    pI_score = max(0.0, 1.0 - abs(pI - 8.0) / 4.0)

    # Hydrophobicity: lower is better for developability
    hydro_score = max(0.0, 1.0 - hydro)

    # Liability density: penalize high deamidation + oxidation per residue
    total_liabilities = deam + ox
    liability_score = max(0.0, 1.0 - total_liabilities / 20.0)

    # MW: typical mAb ~150 kDa gets best score
    mw_score = max(0.0, 1.0 - abs(mw_kda - 150.0) / 200.0)

    # Weighted composite
    score = (0.30 * pI_score +
             0.30 * hydro_score +
             0.25 * liability_score +
             0.15 * mw_score)

    return max(0.0, min(1.0, score))


def _compute_potency_heuristic(features: np.ndarray, sequence: str = "") -> float:
    """
    Compute heuristic potency score from biophysical features.

    Score 0-1 where 1 = highest predicted potency. Used as fallback
    when XGBoost potency model is not trained.

    This is a coarse proxy — real potency requires CDR-level analysis.
    """
    pI, mw_kda, deam, ox, acidic, basic, hydro = features

    # Moderate hydrophobicity in CDRs is sometimes associated with better binding
    hydro_score = 1.0 - abs(hydro - 0.45) / 0.5  # optimal ~0.45

    # Charge balance: moderate charge diversity is good for binding
    charge_balance = min(acidic, basic) / max(max(acidic, basic), 1)
    charge_score = charge_balance  # 0-1, 1 means balanced

    # Fewer liabilities → more stable → better functional binding
    liability_score = max(0.0, 1.0 - (deam + ox) / 25.0)

    # Sequence length contribution (longer CDRs sometimes correlate with better binding)
    if sequence:
        seq_len = len(re.sub(r'[^A-Z]', '', sequence.upper()))
        length_score = min(1.0, seq_len / 1200.0)  # saturates at ~1200aa (typical HC+LC)
    else:
        length_score = 0.5

    score = (0.35 * hydro_score +
             0.25 * charge_score +
             0.25 * liability_score +
             0.15 * length_score)

    return max(0.0, min(1.0, score))


# ===========================================================================
# CSV Parsing for Discovery Format
# ===========================================================================

def parse_discovery_csv(file_content: bytes | str) -> Dict[str, Any]:
    """
    Parse an Early Discovery CSV file with candidate sequences.

    Accepts flexible column names via COLUMN_ALIASES.

    Parameters
    ----------
    file_content : Raw CSV content (bytes or string)

    Returns
    -------
    dict with keys:
        - status: "success" or "error"
        - candidates: List[dict] with standardized keys
        - n_candidates: int
        - has_elisa: bool
        - has_kd: bool
        - warnings: List[str]
    """
    try:
        import pandas as pd
    except ImportError:
        return {"status": "error", "message": "pandas required for CSV parsing"}

    try:
        if isinstance(file_content, bytes):
            import io
            df = pd.read_csv(io.BytesIO(file_content))
        else:
            import io
            df = pd.read_csv(io.StringIO(file_content))
    except Exception as e:
        return {"status": "error", "message": f"CSV parse error: {e}"}

    if len(df) == 0:
        return {"status": "error", "message": "CSV file is empty"}

    # Normalize column names
    col_lower = {c: c.lower().strip().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=col_lower)

    # Map aliases to standard names
    col_map = {}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                col_map[alias] = standard
                break

    df = df.rename(columns=col_map)

    warnings = []

    # Validate required columns
    if "sequence_hc" not in df.columns:
        return {"status": "error",
                "message": "CSV must contain a heavy chain sequence column "
                           "(e.g., 'Sequence_HC', 'HC_Sequence', 'VH')"}

    # Generate candidate IDs if missing
    if "candidate_id" not in df.columns:
        df["candidate_id"] = [f"Candidate_{i+1:04d}" for i in range(len(df))]
        warnings.append("No Candidate_ID column found — auto-generated IDs")

    has_lc = "sequence_lc" in df.columns
    has_elisa = "exp_elisa_od" in df.columns
    has_kd = "exp_kd_nm" in df.columns

    if not has_lc:
        warnings.append("No light chain column found — screening HC only")

    # Build candidate list
    candidates = []
    skipped = 0
    for _, row in df.iterrows():
        hc = str(row.get("sequence_hc", "")).strip()
        if len(re.sub(r'[^A-Za-z]', '', hc)) < 20:
            skipped += 1
            continue

        candidate = {
            "candidate_id": str(row.get("candidate_id", "")),
            "sequence_hc": hc.upper(),
            "sequence_lc": str(row.get("sequence_lc", "")).strip().upper() if has_lc else "",
            "exp_elisa_od": float(row["exp_elisa_od"]) if has_elisa and not pd.isna(row.get("exp_elisa_od")) else None,
            "exp_kd_nm": float(row["exp_kd_nm"]) if has_kd and not pd.isna(row.get("exp_kd_nm")) else None,
        }
        candidates.append(candidate)

    if skipped > 0:
        warnings.append(f"Skipped {skipped} rows with invalid/short HC sequences (<20 aa)")

    return {
        "status": "success",
        "candidates": candidates,
        "n_candidates": len(candidates),
        "has_elisa": has_elisa,
        "has_kd": has_kd,
        "has_lc": has_lc,
        "warnings": warnings,
    }


# ===========================================================================
# Mock Discovery Dataset Generator
# ===========================================================================

def generate_mock_discovery_csv(n_candidates: int = 200, seed: int = 42) -> str:
    """
    Generate a mock Early Discovery CSV with realistic antibody candidates.

    Creates synthetic HC + LC sequences with varied biophysical properties,
    along with mock ELISA OD and Kd values.

    Parameters
    ----------
    n_candidates : Number of candidates to generate (default 200)
    seed : Random seed for reproducibility

    Returns
    -------
    CSV string content
    """
    rng = np.random.RandomState(seed)

    # IgG1 framework fragments for realistic sequences
    framework_hc = [
        "EVQLVESGGGLVQPGGSLRLSCAAS",  # FR1
        "WVRQAPGKGLEWVS",              # FR2
        "RFTISRDNSKNTLYLQMNSLRAEDTAVYYC",  # FR3
        "WGQGTLVTVSS",                 # FR4
    ]
    framework_lc = [
        "DIQMTQSPSSLSASVGDRVTITC",    # FR1
        "WYQQKPGKAPKLLIY",              # FR2
        "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC",  # FR3
        "FGQGTKVEIK",                   # FR4
    ]

    # CH1/CL constant regions (abbreviated)
    ch_const = "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
    cl_const = "RTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"

    aa_pool = "ACDEFGHIKLMNPQRSTVWY"
    lines = ["Candidate_ID,Sequence_HC,Sequence_LC,Exp_ELISA_OD,Exp_Kd_nM"]

    for i in range(n_candidates):
        # Generate CDRs with some variety
        cdr_h1_len = rng.randint(5, 12)
        cdr_h2_len = rng.randint(10, 20)
        cdr_h3_len = rng.randint(8, 25)
        cdr_l1_len = rng.randint(6, 14)
        cdr_l2_len = rng.randint(5, 9)
        cdr_l3_len = rng.randint(7, 12)

        cdr_h1 = "".join(rng.choice(list(aa_pool)) for _ in range(cdr_h1_len))
        cdr_h2 = "".join(rng.choice(list(aa_pool)) for _ in range(cdr_h2_len))
        cdr_h3 = "".join(rng.choice(list(aa_pool)) for _ in range(cdr_h3_len))
        cdr_l1 = "".join(rng.choice(list(aa_pool)) for _ in range(cdr_l1_len))
        cdr_l2 = "".join(rng.choice(list(aa_pool)) for _ in range(cdr_l2_len))
        cdr_l3 = "".join(rng.choice(list(aa_pool)) for _ in range(cdr_l3_len))

        hc_seq = framework_hc[0] + cdr_h1 + framework_hc[1] + cdr_h2 + framework_hc[2] + cdr_h3 + framework_hc[3] + ch_const
        lc_seq = framework_lc[0] + cdr_l1 + framework_lc[1] + cdr_l2 + framework_lc[2] + cdr_l3 + framework_lc[3] + cl_const

        # Introduce mutations for diversity
        n_mutations = rng.randint(0, 8)
        hc_list = list(hc_seq)
        for _ in range(n_mutations):
            pos = rng.randint(0, len(hc_list) - 1)
            hc_list[pos] = rng.choice(list(aa_pool))
        hc_seq = "".join(hc_list)

        # Generate correlated potency/developability
        # Potency: CDR hydrophobicity and diversity drive binding
        cdr_combined = cdr_h1 + cdr_h2 + cdr_h3
        hydrophobic_aa = set("VILFWM")
        cdr_hydro_frac = sum(1 for aa in cdr_combined if aa in hydrophobic_aa) / max(len(cdr_combined), 1)

        # Base potency from CDR properties
        base_potency = 0.3 + 0.4 * cdr_hydro_frac + rng.normal(0, 0.15)

        # Developability inversely correlated with extreme hydrophobicity
        base_dev = 0.7 - 0.3 * cdr_hydro_frac + rng.normal(0, 0.12)

        # ELISA OD: correlated with potency (0.1-3.5 range)
        elisa_od = max(0.05, min(3.5, base_potency * 2.5 + rng.normal(0, 0.3)))

        # Kd in nM: inversely correlated with potency (lower Kd = better binding)
        kd_nm = max(0.01, min(500.0, (1.0 - base_potency) * 200 + rng.exponential(20)))

        name = f"mAb-{i+1:04d}"
        lines.append(f"{name},{hc_seq},{lc_seq},{elisa_od:.2f},{kd_nm:.2f}")

    return "\n".join(lines)


# ===========================================================================
# Core Screening Engine
# ===========================================================================

class HTScreeningEngine:
    """
    High-Throughput Virtual Screening Engine for antibody candidates.

    Processes bulk candidate sequences through ESM-2 embeddings,
    biophysical feature extraction, and XGBoost-based developability
    and potency prediction.
    """

    def __init__(
        self,
        dev_threshold: float = DEFAULT_DEV_THRESHOLD,
        potency_threshold: float = DEFAULT_POTENCY_THRESHOLD,
    ):
        self.dev_threshold = dev_threshold
        self.potency_threshold = potency_threshold
        self._embedder = None
        self._dev_predictor_available = False
        self._potency_predictor_available = False
        self._results: List[Dict[str, Any]] = []
        self._screening_time: float = 0.0

    def _init_embedder(self):
        """Lazy-load ESM-2 embedder."""
        if self._embedder is None:
            try:
                from src.pLM_embedder import ESM2Embedder
                self._embedder = ESM2Embedder()
                log.info("ESM-2 embedder initialized for HT screening")
            except Exception as e:
                log.warning("ESM-2 embedder unavailable: %s (using mock embeddings)", e)
                self._embedder = None

    def _check_ml_models(self):
        """Check availability of trained ML models."""
        try:
            from src.ml_predictor import get_wetlab_model
            model = get_wetlab_model()
            self._dev_predictor_available = model is not None and model.trained
        except Exception:
            self._dev_predictor_available = False

        try:
            from src.ml_predictor import get_potency_model
            model = get_potency_model()
            self._potency_predictor_available = model is not None and model.trained
        except Exception:
            self._potency_predictor_available = False

    def screen_candidates(
        self,
        candidates: List[Dict[str, Any]],
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Screen a list of candidates for developability and potency.

        Parameters
        ----------
        candidates : List of dicts with candidate_id, sequence_hc, sequence_lc
        progress_callback : Optional callable(fraction, message) for UI updates

        Returns
        -------
        dict with:
            - status: "success"
            - results: List[dict] per candidate with scores
            - summary: Overall statistics
            - screening_time_sec: Processing duration
            - models_used: Which ML models were active
        """
        t0 = time.time()

        self._init_embedder()
        self._check_ml_models()

        results = []
        n = len(candidates)

        for i, cand in enumerate(candidates):
            if progress_callback and i % max(1, n // 20) == 0:
                progress_callback(i / n, f"Screening {i+1}/{n}: {cand.get('candidate_id', '?')}")

            result = self._screen_single(cand)
            results.append(result)

        if progress_callback:
            progress_callback(1.0, f"Screening complete: {n} candidates processed")

        self._results = results
        self._screening_time = time.time() - t0

        # Classify into quadrants
        self._classify_quadrants(results)

        # Compute summary
        summary = self._compute_summary(results)

        return {
            "status": "success",
            "results": results,
            "summary": summary,
            "screening_time_sec": round(self._screening_time, 2),
            "n_candidates": n,
            "models_used": {
                "esm2_embedder": self._embedder is not None,
                "xgb_developability": self._dev_predictor_available,
                "xgb_potency": self._potency_predictor_available,
            },
        }

    def _screen_single(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single candidate through the full screening pipeline."""
        cid = candidate.get("candidate_id", "unknown")
        hc = candidate.get("sequence_hc", "")
        lc = candidate.get("sequence_lc", "")
        combined = hc + lc if lc else hc

        # Step 1: Extract biophysical features
        features = _extract_biophysical_features(combined)

        # Step 2: ESM-2 embedding
        embedding = None
        if self._embedder is not None:
            try:
                if lc:
                    embedding = self._embedder.embed_antibody(hc, lc)
                else:
                    embedding = self._embedder.embed_sequence(hc)
            except Exception as e:
                log.debug("Embedding failed for %s: %s", cid, e)

        if embedding is None:
            # Mock fallback
            try:
                from src.pLM_embedder import mock_embedding
                embedding = mock_embedding(combined)
            except Exception:
                from src.pLM_embedder import ANTIBODY_EMBED_DIM
                embedding = np.random.RandomState(hash(combined) % (2**31)).randn(ANTIBODY_EMBED_DIM).astype(np.float32) * 0.1

        # Step 3: Developability score (unified with Dashboard)
        dev_score = self._predict_developability(features, embedding)

        # Step 3b: Get per-dimension risk breakdown if available
        dev_dimensions = {}
        try:
            from src.developability_core import assess_developability
            _feats = {
                "pI": float(features[0]), "mw_kda": float(features[1]),
                "deam_sites": int(features[2]), "ox_sites": int(features[3]),
                "acidic_residues": int(features[4]), "basic_residues": int(features[5]),
                "hydrophobicity": float(features[6]),
            }
            _assessment = assess_developability(biophysical_features=_feats)
            for dim in _assessment.dimensions:
                dev_dimensions[dim.name] = {
                    "score": round(dim.score, 3),
                    "grade": dim.grade,
                }
        except Exception:
            pass

        # Step 4: Potency score
        potency_score = self._predict_potency(features, embedding, combined, candidate)

        # Step 5: Build result
        result = {
            "candidate_id": cid,
            "sequence_hc_len": len(re.sub(r'[^A-Z]', '', hc.upper())),
            "sequence_lc_len": len(re.sub(r'[^A-Z]', '', lc.upper())) if lc else 0,
            "pI": round(float(features[0]), 2),
            "mw_kda": round(float(features[1]), 2),
            "deam_sites": int(features[2]),
            "ox_sites": int(features[3]),
            "hydrophobicity": round(float(features[6]), 3),
            "developability_score": round(dev_score, 4),
            "potency_score": round(potency_score, 4),
            "dev_model": "Unified" if dev_dimensions else ("XGBoost" if self._dev_predictor_available else "Heuristic"),
            "potency_model": "XGBoost" if self._potency_predictor_available else "Heuristic",
            "dev_dimensions": dev_dimensions,
            "quadrant": "",  # Filled in _classify_quadrants
        }

        # Attach experimental data if available
        if candidate.get("exp_elisa_od") is not None:
            result["exp_elisa_od"] = round(candidate["exp_elisa_od"], 3)
        if candidate.get("exp_kd_nm") is not None:
            result["exp_kd_nm"] = round(candidate["exp_kd_nm"], 3)

        return result

    def _predict_developability(self, features: np.ndarray, embedding: np.ndarray) -> float:
        """
        Predict developability score — unified with Dashboard's assess_developability().

        v2.0: Calls developability_core.assess_developability() for multi-dimensional
        risk assessment using the same grade thresholds and evidence tiers.
        Falls back to heuristic only if the full assessment fails.
        """
        pI, mw_kda, deam, ox, acidic, basic, hydro = features

        # Try unified assessment via developability_core (same as Dashboard)
        try:
            from src.developability_core import assess_developability
            feats = {
                "pI": float(pI),
                "mw_kda": float(mw_kda),
                "deam_sites": int(deam),
                "ox_sites": int(ox),
                "acidic_residues": int(acidic),
                "basic_residues": int(basic),
                "hydrophobicity": float(hydro),
            }
            # Build dev_predictions from XGBoost if available
            dev_preds = {}
            if self._dev_predictor_available:
                try:
                    from src.ml_predictor import get_wetlab_model
                    model = get_wetlab_model()
                    raw_preds = model.predict_single(features)
                    agg_pct = raw_preds.get("Exp_Aggregation_Percent", 5.0)
                    tm = raw_preds.get("Exp_Tm_MeltingTemp", 70.0)
                    dev_preds["agg_risk"] = max(0.0, min(1.0, agg_pct / 30.0))
                    dev_preds["stability"] = max(0.0, min(1.0, (tm - 55.0) / 30.0))
                    dev_preds["viscosity_risk"] = 0.15  # default when not predicted
                except Exception:
                    pass

            assessment = assess_developability(
                biophysical_features=feats,
                dev_predictions=dev_preds if dev_preds else None,
            )
            # Return 1 - composite_score because assess_developability returns
            # RISK score (high = bad), but HT screening wants DEVELOPABILITY
            # score (high = good)
            return max(0.0, min(1.0, 1.0 - assessment.composite_score))
        except Exception as e:
            log.debug("Unified developability assessment failed: %s — using heuristic", e)

        return _compute_developability_score(features)

    def _predict_potency(
        self,
        features: np.ndarray,
        embedding: np.ndarray,
        sequence: str,
        candidate: Dict[str, Any],
    ) -> float:
        """Predict potency score using XGBoost model or heuristic fallback."""
        if self._potency_predictor_available:
            try:
                from src.ml_predictor import get_potency_model
                model = get_potency_model()
                pred = model.predict_single(features)
                potency_raw = pred.get("Predicted_Potency_Score", 0.5)
                return max(0.0, min(1.0, potency_raw))
            except Exception as e:
                log.debug("XGBoost potency failed: %s — using heuristic", e)

        return _compute_potency_heuristic(features, sequence)

    def _classify_quadrants(self, results: List[Dict[str, Any]]) -> None:
        """Assign Magic Quadrant classification to each result."""
        for r in results:
            dev = r["developability_score"]
            pot = r["potency_score"]

            if dev >= self.dev_threshold and pot >= self.potency_threshold:
                r["quadrant"] = QUADRANT_STAR
            elif dev >= self.dev_threshold and pot < self.potency_threshold:
                r["quadrant"] = QUADRANT_DEVELOPABLE
            elif dev < self.dev_threshold and pot >= self.potency_threshold:
                r["quadrant"] = QUADRANT_POTENT
            else:
                r["quadrant"] = QUADRANT_RISKY

    def _compute_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics from screening results."""
        n = len(results)
        if n == 0:
            return {"n_total": 0}

        dev_scores = [r["developability_score"] for r in results]
        pot_scores = [r["potency_score"] for r in results]

        quadrant_counts = {
            QUADRANT_STAR: sum(1 for r in results if r["quadrant"] == QUADRANT_STAR),
            QUADRANT_DEVELOPABLE: sum(1 for r in results if r["quadrant"] == QUADRANT_DEVELOPABLE),
            QUADRANT_POTENT: sum(1 for r in results if r["quadrant"] == QUADRANT_POTENT),
            QUADRANT_RISKY: sum(1 for r in results if r["quadrant"] == QUADRANT_RISKY),
        }

        # Top candidates: Star quadrant, sorted by combined score
        stars = [r for r in results if r["quadrant"] == QUADRANT_STAR]
        stars_sorted = sorted(stars,
                              key=lambda x: x["developability_score"] + x["potency_score"],
                              reverse=True)

        return {
            "n_total": n,
            "quadrant_counts": quadrant_counts,
            "dev_score_mean": round(float(np.mean(dev_scores)), 4),
            "dev_score_std": round(float(np.std(dev_scores)), 4),
            "potency_score_mean": round(float(np.mean(pot_scores)), 4),
            "potency_score_std": round(float(np.std(pot_scores)), 4),
            "top_5_stars": [
                {"candidate_id": s["candidate_id"],
                 "dev": s["developability_score"],
                 "potency": s["potency_score"]}
                for s in stars_sorted[:5]
            ],
            "star_rate_pct": round(100.0 * quadrant_counts[QUADRANT_STAR] / max(n, 1), 1),
        }

    def get_results_as_csv(self) -> str:
        """Export screening results as CSV string."""
        if not self._results:
            return ""

        # Define column order
        cols = [
            "candidate_id", "quadrant", "developability_score", "potency_score",
            "pI", "mw_kda", "deam_sites", "ox_sites", "hydrophobicity",
            "sequence_hc_len", "sequence_lc_len", "dev_model", "potency_model",
        ]

        # Add optional columns
        has_elisa = any("exp_elisa_od" in r for r in self._results)
        has_kd = any("exp_kd_nm" in r for r in self._results)
        if has_elisa:
            cols.append("exp_elisa_od")
        if has_kd:
            cols.append("exp_kd_nm")

        lines = [",".join(cols)]
        for r in self._results:
            values = []
            for c in cols:
                v = r.get(c, "")
                values.append(str(v))
            lines.append(",".join(values))

        return "\n".join(lines)

    def get_star_candidates(self) -> List[Dict[str, Any]]:
        """Return candidates in the Star quadrant (top-right), sorted by combined score."""
        stars = [r for r in self._results if r.get("quadrant") == QUADRANT_STAR]
        return sorted(stars,
                      key=lambda x: x["developability_score"] + x["potency_score"],
                      reverse=True)


# ===========================================================================
# Convenience Functions
# ===========================================================================

def run_ht_screening(
    csv_content: bytes | str,
    dev_threshold: float = DEFAULT_DEV_THRESHOLD,
    potency_threshold: float = DEFAULT_POTENCY_THRESHOLD,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    End-to-end convenience function: parse CSV → screen → return results.

    Parameters
    ----------
    csv_content : Raw CSV file content
    dev_threshold : Developability score threshold for quadrant classification
    potency_threshold : Potency score threshold for quadrant classification
    progress_callback : Optional callable(fraction, message)

    Returns
    -------
    dict with screening results, summary, and CSV export
    """
    # Parse CSV
    parsed = parse_discovery_csv(csv_content)
    if parsed["status"] != "success":
        return parsed

    # Screen
    engine = HTScreeningEngine(
        dev_threshold=dev_threshold,
        potency_threshold=potency_threshold,
    )
    result = engine.screen_candidates(
        parsed["candidates"],
        progress_callback=progress_callback,
    )

    # Attach CSV export
    result["csv_export"] = engine.get_results_as_csv()
    result["star_candidates"] = engine.get_star_candidates()
    result["parse_warnings"] = parsed.get("warnings", [])

    return result
