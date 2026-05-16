"""
uncertainty_engine.py — Milestone 26
=====================================================================
Uncertainty Quantification + Active Learning Query Strategy

Provides:
    1. MC Dropout uncertainty for PyTorch ChromatographyMLP
    2. Ensemble variance for XGBoost WetLabPredictor
    3. Combined uncertainty scoring across all models
    4. Active Learning: Expected Improvement selection strategy
    5. Virtual mutant library generation (in-silico scanning)

The core idea: if a model predicts high Titer but has HIGH variance,
that molecule is a "High-Value Data Candidate" — the wet-lab experiment
that would most improve the model.

Version : 1.0 (Active Learning & DoE Automation — M26)
Author  : Di (ProtePilot)
"""

from __future__ import annotations

import logging
import time
import re
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# =========================================================================
# 1. Data Classes
# =========================================================================

@dataclass
class PredictionWithUncertainty:
    """Single prediction bundled with uncertainty metrics."""
    name: str
    features: np.ndarray                # 7-dim feature vector
    sequence: Optional[str] = None

    # Chromatography MLP predictions (MC Dropout)
    ka_mean: float = 0.0
    ka_std: float = 0.0
    nu_mean: float = 0.0
    nu_std: float = 0.0
    mlp_uncertainty: float = 0.0        # combined ka+nu variance

    # WetLab XGBoost predictions (Ensemble Variance)
    agg_mean: float = 0.0               # Aggregation %
    agg_std: float = 0.0
    tm_mean: float = 0.0                # Tm (°C)
    tm_std: float = 0.0
    xgb_uncertainty: float = 0.0

    # Combined
    combined_uncertainty: float = 0.0   # weighted mean of mlp + xgb
    predicted_performance: float = 0.0  # composite "goodness" score
    expected_improvement: float = 0.0   # acquisition function value
    is_high_value_candidate: bool = False

    # Metadata
    uncertainty_source: str = "empirical"


@dataclass
class ActiveLearningResult:
    """Result of an active learning query cycle."""
    candidate_pool_size: int
    n_selected: int
    selected: List[PredictionWithUncertainty]
    selection_method: str           # "expected_improvement"
    best_ei: float
    mean_ei: float
    wall_time_s: float
    summary: str = ""


# =========================================================================
# 2. MC Dropout Uncertainty for PyTorch MLP
# =========================================================================

def mc_dropout_predict(
    model,
    X: np.ndarray,
    n_forward: int = 30,
) -> Dict[str, np.ndarray]:
    """
    Monte Carlo Dropout inference for the ChromatographyMLP.

    Enables dropout at inference time and runs n_forward stochastic
    passes to estimate prediction mean and variance.

    Parameters
    ----------
    model      : ChromatographyMLP instance (with Dropout layers)
    X          : (n_samples, 7) feature matrix (already scaled or raw)
    n_forward  : Number of stochastic forward passes

    Returns
    -------
    dict with: mean (n, 2), std (n, 2), all_preds (n_forward, n, 2)
    """
    try:
        import torch
    except ImportError:
        log.warning("PyTorch not available; using empirical uncertainty")
        return _empirical_mlp_uncertainty(X)

    # Ensure 2D input
    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_samples = X.shape[0]

    # Scale features using the model's scaler
    if hasattr(model, 'scaler') and model.scaler.fitted:
        X_scaled = model.scaler.transform(X)
    else:
        X_scaled = X.copy()

    X_tensor = torch.FloatTensor(X_scaled).to(model.device)

    # Enable dropout for MC inference
    net = model.model if hasattr(model, 'model') else model.net
    net.train()  # enable dropout

    all_preds = np.zeros((n_forward, n_samples, 2))

    with torch.no_grad():
        for i in range(n_forward):
            out = net(X_tensor).cpu().numpy()
            # Ensure 2D output
            if out.ndim == 1:
                out = out.reshape(1, -1)
            # Handle models with != 2 outputs
            if out.shape[1] >= 2:
                all_preds[i] = out[:, :2]
            else:
                all_preds[i, :, 0] = out[:, 0]
                all_preds[i, :, 1] = out[:, 0] * 0.7  # dummy second output

    # Restore eval mode
    net.eval()

    mean = np.mean(all_preds, axis=0)  # (n, 2)
    std = np.std(all_preds, axis=0)    # (n, 2)

    return {
        "mean": mean,
        "std": std,
        "all_preds": all_preds,
        "n_forward": n_forward,
        "method": "mc_dropout",
    }


def _empirical_mlp_uncertainty(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Fallback: empirical uncertainty proportional to feature deviation."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n = X.shape[0]

    # Heuristic: higher pI deviation and hydrophobicity → more uncertainty
    pI = X[:, 0] if X.shape[1] > 0 else np.full(n, 8.0)
    hydro = X[:, 6] if X.shape[1] > 6 else np.full(n, 0.35)

    # v7.3: Standard-range ka (typical range 1-5)
    base_ka = 2.5 + 0.3 * (pI - 7.0)
    base_nu = 3.5 + 0.3 * (pI - 7.0)

    ka_std = 0.5 + 0.3 * np.abs(hydro - 0.35)   # v7.3: scaled for standard ka
    nu_std = 0.25 + 0.15 * np.abs(pI - 8.0)

    mean = np.column_stack([base_ka, base_nu])
    std = np.column_stack([ka_std, nu_std])

    return {
        "mean": mean,
        "std": std,
        "all_preds": None,
        "n_forward": 0,
        "method": "empirical_fallback",
    }


# =========================================================================
# 3. Ensemble Variance for XGBoost
# =========================================================================

def xgboost_ensemble_uncertainty(
    predictor,
    X: np.ndarray,
    n_submodels: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Estimate uncertainty from XGBoost using tree-based sub-ensemble variance.

    Splits the full boosting ensemble into n_submodels sub-ensembles
    (using ntree_limit) and measures prediction variance.

    Parameters
    ----------
    predictor  : WetLabPredictor instance with trained .models list
    X          : (n_samples, 7) feature matrix

    Returns
    -------
    dict with: mean (n, n_targets), std (n, n_targets)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n_samples = X.shape[0]

    if not hasattr(predictor, 'models') or not predictor.models or not predictor.trained:
        return _empirical_xgb_uncertainty(X, n_targets=2)

    try:
        import xgboost as xgb
    except ImportError:
        return _empirical_xgb_uncertainty(X, n_targets=len(predictor.models))

    # Scale features
    if hasattr(predictor, 'scaler') and predictor.scaler.fitted:
        X_scaled = predictor.scaler.transform(X)
    else:
        X_scaled = X.copy()

    n_targets = len(predictor.models)
    all_preds = []  # shape will be (n_submodels, n_samples, n_targets)

    for sub_i in range(n_submodels):
        preds_per_target = []
        for model in predictor.models:
            n_trees = model.n_estimators if hasattr(model, 'n_estimators') else 100
            # Use a random subset fraction of trees for sub-ensemble
            frac = 0.5 + 0.5 * random.random()  # 50-100% of trees
            ntree_limit = max(1, int(n_trees * frac))
            try:
                dmat = xgb.DMatrix(X_scaled)
                pred = model.get_booster().predict(dmat, iteration_range=(0, ntree_limit))
            except Exception:
                try:
                    pred = model.predict(X_scaled)
                except Exception:
                    pred = np.full(n_samples, 0.0)
            preds_per_target.append(pred)
        all_preds.append(np.column_stack(preds_per_target))

    all_preds = np.array(all_preds)  # (n_submodels, n_samples, n_targets)
    mean = np.mean(all_preds, axis=0)
    std = np.std(all_preds, axis=0)

    return {
        "mean": mean,
        "std": std,
        "n_submodels": n_submodels,
        "method": "xgb_ensemble_variance",
    }


def _empirical_xgb_uncertainty(
    X: np.ndarray, n_targets: int = 2,
) -> Dict[str, np.ndarray]:
    """Fallback: empirical uncertainty for XGBoost predictions."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n = X.shape[0]

    # Heuristic: aggregation ~5%, Tm ~70°C with uncertainty
    pI = X[:, 0] if X.shape[1] > 0 else np.full(n, 8.0)
    hydro = X[:, 6] if X.shape[1] > 6 else np.full(n, 0.35)

    agg_mean = 3.0 + 5.0 * hydro + 0.5 * np.abs(pI - 8.5)
    tm_mean = 72.0 - 5.0 * hydro - 0.5 * np.abs(pI - 8.5)
    agg_std = 1.5 + 2.0 * hydro
    tm_std = 2.0 + 1.5 * np.abs(pI - 8.5)

    mean = np.column_stack([agg_mean, tm_mean])
    std = np.column_stack([agg_std, tm_std])

    if n_targets > 2:
        # Pad extra targets
        extra_mean = np.full((n, n_targets - 2), 0.5)
        extra_std = np.full((n, n_targets - 2), 0.2)
        mean = np.column_stack([mean, extra_mean])
        std = np.column_stack([std, extra_std])

    return {
        "mean": mean,
        "std": std,
        "n_submodels": 0,
        "method": "empirical_fallback",
    }


# =========================================================================
# 4. Combined Uncertainty Scoring
# =========================================================================

def score_uncertainty(
    mlp_result: Dict[str, np.ndarray],
    xgb_result: Dict[str, np.ndarray],
    weight_mlp: float = 0.4,
    weight_xgb: float = 0.6,
) -> np.ndarray:
    """
    Compute combined normalized uncertainty score for each sample.

    Combines the coefficient of variation from both model families
    into a single [0, 1] score where 1 = highest uncertainty.

    Parameters
    ----------
    mlp_result : Output of mc_dropout_predict()
    xgb_result : Output of xgboost_ensemble_uncertainty()

    Returns
    -------
    (n_samples,) array of combined uncertainty scores in [0, 1]
    """
    # MLP uncertainty: mean CoV across ka and nu
    mlp_mean = mlp_result["mean"]
    mlp_std = mlp_result["std"]
    mlp_cov = np.mean(mlp_std / (np.abs(mlp_mean) + 1e-8), axis=1)

    # XGB uncertainty: mean CoV across targets
    xgb_mean = xgb_result["mean"]
    xgb_std = xgb_result["std"]
    xgb_cov = np.mean(xgb_std / (np.abs(xgb_mean) + 1e-8), axis=1)

    # Normalize to [0, 1] via sigmoid-like transform
    mlp_norm = 1.0 - np.exp(-3.0 * mlp_cov)
    xgb_norm = 1.0 - np.exp(-3.0 * xgb_cov)

    combined = weight_mlp * mlp_norm + weight_xgb * xgb_norm
    return np.clip(combined, 0.0, 1.0)


def score_predicted_performance(
    xgb_result: Dict[str, np.ndarray],
    mlp_result: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute a composite performance score from predictions.

    Higher is better: low aggregation + high Tm + good chromatography.

    Returns
    -------
    (n_samples,) array of performance scores in [0, 1]
    """
    xgb_mean = xgb_result["mean"]
    mlp_mean = mlp_result["mean"]

    # Aggregation: lower is better (invert and normalize to 0-1)
    agg = xgb_mean[:, 0] if xgb_mean.shape[1] > 0 else np.full(len(xgb_mean), 5.0)
    agg_score = np.clip(1.0 - agg / 20.0, 0.0, 1.0)

    # Tm: higher is better (normalize 50-90°C range to 0-1)
    tm = xgb_mean[:, 1] if xgb_mean.shape[1] > 1 else np.full(len(xgb_mean), 70.0)
    tm_score = np.clip((tm - 50.0) / 40.0, 0.0, 1.0)

    # Chromatography: moderate ka is best (not too tight, not too loose)
    ka = mlp_mean[:, 0] if mlp_mean.shape[1] > 0 else np.full(len(mlp_mean), 1.5)
    ka_score = np.clip(1.0 - np.abs(ka - 1.5) / 2.0, 0.0, 1.0)

    # Weighted composite
    performance = 0.4 * agg_score + 0.35 * tm_score + 0.25 * ka_score
    return np.clip(performance, 0.0, 1.0)


# =========================================================================
# 5. Expected Improvement Acquisition Function
# =========================================================================

def expected_improvement(
    predicted_performance: np.ndarray,
    uncertainty: np.ndarray,
    current_best: float = 0.7,
    exploration_weight: float = 0.5,
) -> np.ndarray:
    """
    Expected Improvement (EI) acquisition function for Active Learning.

    Balances exploitation (high predicted performance) with exploration
    (high uncertainty) to select the most informative experiments.

    EI(x) = alpha * (mu(x) - f_best) + (1 - alpha) * sigma(x)

    Parameters
    ----------
    predicted_performance : (n,) performance scores
    uncertainty           : (n,) uncertainty scores
    current_best          : Performance of the best known molecule
    exploration_weight    : alpha=0 => pure exploitation, alpha=1 => pure exploration

    Returns
    -------
    (n,) array of EI scores (higher = more valuable to test)
    """
    exploitation = np.maximum(predicted_performance - current_best, 0.0)
    exploration = uncertainty

    ei = (1.0 - exploration_weight) * exploitation + exploration_weight * exploration

    # Bonus for high-performance + high-uncertainty (the sweet spot)
    synergy_bonus = predicted_performance * uncertainty * 0.3
    ei += synergy_bonus

    return ei


# =========================================================================
# 6. Virtual Mutant Library Generation
# =========================================================================

# Common antibody framework amino acids for mutations
_HYDROPHOBIC = list("AILMFVW")
_CHARGED_POS = list("KRH")
_CHARGED_NEG = list("DE")
_POLAR = list("NQSTY")
_ALL_AA = list("ACDEFGHIKLMNPQRSTVWY")


def generate_virtual_mutant_library(
    parent_sequence: str,
    n_mutants: int = 1000,
    max_mutations_per_variant: int = 3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate a virtual mutant library by in-silico random mutagenesis.

    Creates n_mutants single/double/triple point mutants of the parent
    sequence. Each mutant gets a unique name and the mutation descriptor.

    Parameters
    ----------
    parent_sequence     : Wild-type amino acid sequence
    n_mutants           : Number of virtual mutants to generate
    max_mutations_per_variant : Max substitutions per variant (1-3)
    seed                : Random seed

    Returns
    -------
    List of dicts: {name, sequence, mutations, n_mutations}
    """
    rng = random.Random(seed)
    seq = parent_sequence.upper().replace(" ", "").replace("\n", "")
    n = len(seq)

    if n < 10:
        log.warning("Parent sequence too short (%d aa) for mutagenesis", n)
        return []

    # Exclude first/last 5 residues (signal peptide / C-terminal) from mutation
    mutable_range = list(range(5, n - 5))

    library = []
    seen = set()

    for i in range(n_mutants):
        n_muts = rng.randint(1, min(max_mutations_per_variant, 3))
        positions = sorted(rng.sample(mutable_range, min(n_muts, len(mutable_range))))

        mut_seq = list(seq)
        mutations = []

        for pos in positions:
            wt_aa = seq[pos]
            # Choose a different amino acid
            candidates = [aa for aa in _ALL_AA if aa != wt_aa]
            new_aa = rng.choice(candidates)
            mut_seq[pos] = new_aa
            mutations.append(f"{wt_aa}{pos+1}{new_aa}")  # 1-indexed

        mut_seq_str = "".join(mut_seq)
        mut_key = "|".join(mutations)

        if mut_key in seen:
            continue
        seen.add(mut_key)

        library.append({
            "name": f"Mut_{i+1:04d}_{'_'.join(mutations)}",
            "sequence": mut_seq_str,
            "mutations": mutations,
            "n_mutations": len(mutations),
        })

    log.info("Generated virtual mutant library: %d variants from %d-aa parent",
             len(library), n)
    return library


def extract_features_for_mutants(
    mutants: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract 7-dim feature vectors for a list of mutant sequences.

    Returns
    -------
    X       : (n, 7) feature matrix
    names   : list of mutant names
    """
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        _HAS_BIO = True
    except ImportError:
        _HAS_BIO = False

    X_list = []
    names = []

    for mut in mutants:
        seq = mut["sequence"]
        seq_clean = re.sub(r'[^A-Z]', '', seq.upper())
        if len(seq_clean) < 20:
            continue

        if _HAS_BIO:
            try:
                pa = ProteinAnalysis(seq_clean)
                pI = pa.isoelectric_point()
                mw = pa.molecular_weight() / 1000.0
                gravy = pa.gravy()
            except Exception:
                pI, mw, gravy = 8.0, len(seq_clean) * 0.11, -0.3
        else:
            pI = 8.0 + 0.01 * (seq_clean.count("K") + seq_clean.count("R")
                                - seq_clean.count("D") - seq_clean.count("E"))
            mw = len(seq_clean) * 0.11
            gravy = -0.3

        hydro = max(0.0, min(1.0, (gravy + 2.0) / 4.0))
        deam = len(re.findall(r"N[GS]", seq_clean))
        ox = seq_clean.count("M") + seq_clean.count("W")
        acidic = seq_clean.count("D") + seq_clean.count("E")
        basic = seq_clean.count("K") + seq_clean.count("R") + seq_clean.count("H")

        X_list.append([pI, mw, float(deam), float(ox),
                       float(acidic), float(basic), hydro])
        names.append(mut["name"])

    if not X_list:
        return np.zeros((0, 7)), []

    return np.array(X_list, dtype=np.float32), names


# =========================================================================
# 7. Active Learning Query Strategy
# =========================================================================

def suggest_next_experiments(
    candidate_pool: List[Dict[str, Any]],
    n: int = 10,
    current_best_performance: float = 0.7,
    exploration_weight: float = 0.5,
    mlp_model=None,
    xgb_model=None,
    n_mc_forward: int = 30,
) -> ActiveLearningResult:
    """
    Select the top-N most informative experiments from a candidate pool.

    This is the core Active Learning function. It:
      1. Extracts features for all candidates
      2. Runs MC Dropout inference on the MLP model
      3. Runs Ensemble Variance on the XGBoost model
      4. Computes Expected Improvement (EI) for each candidate
      5. Returns the top-N candidates ranked by EI

    Parameters
    ----------
    candidate_pool     : List of {name, sequence, mutations, ...} dicts
    n                  : Number of experiments to select
    current_best_performance : Best known performance score (for EI baseline)
    exploration_weight : Balance exploitation vs exploration (0-1)
    mlp_model          : Trained ChromatographyMLP (optional)
    xgb_model          : Trained WetLabPredictor (optional)
    n_mc_forward       : MC Dropout forward passes

    Returns
    -------
    ActiveLearningResult with selected candidates
    """
    t0 = time.time()
    pool_size = len(candidate_pool)
    log.info("Active learning: scoring %d candidates, selecting top %d", pool_size, n)

    # Step 1: Extract features
    X, names = extract_features_for_mutants(candidate_pool)
    if len(X) == 0:
        return ActiveLearningResult(
            candidate_pool_size=pool_size,
            n_selected=0,
            selected=[],
            selection_method="expected_improvement",
            best_ei=0.0,
            mean_ei=0.0,
            wall_time_s=time.time() - t0,
            summary="No valid candidates in pool",
        )

    # Step 2: MLP uncertainty
    if mlp_model is not None and hasattr(mlp_model, 'trained') and mlp_model.trained:
        mlp_result = mc_dropout_predict(mlp_model, X, n_forward=n_mc_forward)
    else:
        mlp_result = _empirical_mlp_uncertainty(X)

    # Step 3: XGBoost uncertainty
    if xgb_model is not None and hasattr(xgb_model, 'trained') and xgb_model.trained:
        xgb_result = xgboost_ensemble_uncertainty(xgb_model, X)
    else:
        xgb_result = _empirical_xgb_uncertainty(X)

    # Step 4: Score
    uncertainty = score_uncertainty(mlp_result, xgb_result)
    performance = score_predicted_performance(xgb_result, mlp_result)
    ei = expected_improvement(
        performance, uncertainty,
        current_best=current_best_performance,
        exploration_weight=exploration_weight,
    )

    # Step 5: Rank and select top-N
    top_indices = np.argsort(ei)[::-1][:n]

    selected = []
    for idx in top_indices:
        idx = int(idx)
        candidate = candidate_pool[idx] if idx < len(candidate_pool) else {}

        pred = PredictionWithUncertainty(
            name=names[idx] if idx < len(names) else f"candidate_{idx}",
            features=X[idx],
            sequence=candidate.get("sequence"),
            ka_mean=float(mlp_result["mean"][idx, 0]),
            ka_std=float(mlp_result["std"][idx, 0]),
            nu_mean=float(mlp_result["mean"][idx, 1]),
            nu_std=float(mlp_result["std"][idx, 1]),
            mlp_uncertainty=float(1.0 - np.exp(-3.0 * np.mean(
                mlp_result["std"][idx] / (np.abs(mlp_result["mean"][idx]) + 1e-8)))),
            agg_mean=float(xgb_result["mean"][idx, 0]),
            agg_std=float(xgb_result["std"][idx, 0]),
            tm_mean=float(xgb_result["mean"][idx, 1]) if xgb_result["mean"].shape[1] > 1 else 70.0,
            tm_std=float(xgb_result["std"][idx, 1]) if xgb_result["std"].shape[1] > 1 else 2.0,
            xgb_uncertainty=float(1.0 - np.exp(-3.0 * np.mean(
                xgb_result["std"][idx] / (np.abs(xgb_result["mean"][idx]) + 1e-8)))),
            combined_uncertainty=float(uncertainty[idx]),
            predicted_performance=float(performance[idx]),
            expected_improvement=float(ei[idx]),
            is_high_value_candidate=(float(performance[idx]) > 0.5 and float(uncertainty[idx]) > 0.3),
            uncertainty_source=mlp_result.get("method", "empirical"),
        )
        selected.append(pred)

    wall = time.time() - t0
    best_ei = float(ei[top_indices[0]]) if len(top_indices) > 0 else 0.0
    mean_ei = float(np.mean(ei))

    summary_lines = [
        f"Active Learning Query — {pool_size} candidates → {len(selected)} selected",
        f"  Method: Expected Improvement (exploration={exploration_weight:.1f})",
        f"  Best EI: {best_ei:.4f} | Mean EI: {mean_ei:.4f}",
        f"  High-value candidates: {sum(1 for s in selected if s.is_high_value_candidate)}",
        f"  Uncertainty source: {mlp_result.get('method', 'N/A')} + {xgb_result.get('method', 'N/A')}",
        f"  Wall time: {wall:.2f}s",
    ]

    result = ActiveLearningResult(
        candidate_pool_size=pool_size,
        n_selected=len(selected),
        selected=selected,
        selection_method="expected_improvement",
        best_ei=best_ei,
        mean_ei=mean_ei,
        wall_time_s=round(wall, 3),
        summary="\n".join(summary_lines),
    )

    log.info(result.summary)
    return result


# =========================================================================
# 8. Convenience: Full Active Learning Pipeline
# =========================================================================

def run_active_learning_cycle(
    parent_sequence: str,
    n_virtual_mutants: int = 1000,
    n_select: int = 10,
    exploration_weight: float = 0.5,
    mlp_model=None,
    xgb_model=None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    End-to-end active learning cycle:
      1. Generate virtual mutant library from parent sequence
      2. Extract features and predict with uncertainty
      3. Select top-N experiments via Expected Improvement

    Parameters
    ----------
    parent_sequence   : Wild-type antibody sequence
    n_virtual_mutants : Size of virtual mutant library
    n_select          : Number of experiments to recommend
    exploration_weight: EI exploration parameter (0=exploit, 1=explore)
    mlp_model         : Trained ChromatographyMLP (optional)
    xgb_model         : Trained WetLabPredictor (optional)
    seed              : Random seed

    Returns
    -------
    dict with: mutant_library, al_result, summary
    """
    t0 = time.time()

    # Generate mutant library
    library = generate_virtual_mutant_library(
        parent_sequence=parent_sequence,
        n_mutants=n_virtual_mutants,
        seed=seed,
    )

    if not library:
        return {
            "status": "error",
            "message": "Could not generate virtual mutants (sequence too short?)",
            "mutant_library": [],
            "al_result": None,
        }

    # Run active learning selection
    al_result = suggest_next_experiments(
        candidate_pool=library,
        n=n_select,
        exploration_weight=exploration_weight,
        mlp_model=mlp_model,
        xgb_model=xgb_model,
    )

    return {
        "status": "success",
        "mutant_library_size": len(library),
        "mutant_library": library,
        "al_result": al_result,
        "wall_time_s": round(time.time() - t0, 3),
    }


# =========================================================================
# Self-Test
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYA"
          "DSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS")

    print("=" * 60)
    print("uncertainty_engine.py — Self-Test")
    print("=" * 60)

    # Test 1: Virtual mutant library
    lib = generate_virtual_mutant_library(VH, n_mutants=200, seed=42)
    assert len(lib) > 100, f"Expected >100 mutants, got {len(lib)}"
    assert "sequence" in lib[0]
    assert "mutations" in lib[0]
    print(f"  [1/6] Virtual mutant library: {len(lib)} variants ✅")

    # Test 2: Feature extraction
    X, names = extract_features_for_mutants(lib[:50])
    assert X.shape == (50, 7), f"Expected (50,7), got {X.shape}"
    assert len(names) == 50
    print(f"  [2/6] Feature extraction: {X.shape} ✅")

    # Test 3: Empirical MLP uncertainty
    mlp_unc = _empirical_mlp_uncertainty(X)
    assert mlp_unc["mean"].shape == (50, 2)
    assert mlp_unc["std"].shape == (50, 2)
    print(f"  [3/6] MLP uncertainty (empirical): mean={mlp_unc['mean'].mean():.3f} ✅")

    # Test 4: Empirical XGB uncertainty
    xgb_unc = _empirical_xgb_uncertainty(X)
    assert xgb_unc["mean"].shape == (50, 2)
    print(f"  [4/6] XGB uncertainty (empirical): mean={xgb_unc['mean'].mean():.3f} ✅")

    # Test 5: Expected Improvement
    unc_scores = score_uncertainty(mlp_unc, xgb_unc)
    perf_scores = score_predicted_performance(xgb_unc, mlp_unc)
    ei_scores = expected_improvement(perf_scores, unc_scores)
    assert len(ei_scores) == 50
    assert np.all(ei_scores >= 0)
    print(f"  [5/6] Expected Improvement: min={ei_scores.min():.4f}, "
          f"max={ei_scores.max():.4f}, mean={ei_scores.mean():.4f} ✅")

    # Test 6: Full active learning cycle
    result = run_active_learning_cycle(
        parent_sequence=VH,
        n_virtual_mutants=500,
        n_select=10,
        exploration_weight=0.5,
    )
    assert result["status"] == "success"
    al = result["al_result"]
    assert al.n_selected == 10
    assert len(al.selected) == 10
    assert al.selected[0].expected_improvement >= al.selected[-1].expected_improvement
    high_val = sum(1 for s in al.selected if s.is_high_value_candidate)
    print(f"  [6/6] Full AL cycle: {al.candidate_pool_size} → {al.n_selected} selected, "
          f"{high_val} high-value ✅")

    print()
    print(al.summary)
    print()
    print(f"Self-test: 6/6 passed")
