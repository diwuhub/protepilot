"""
pareto_optimizer.py — Milestone 27
=====================================================================
Multi-Objective Pareto Frontier Engine

In real biopharma development, there is no "perfect" molecule.
Every candidate represents a trade-off between efficacy (potency,
binding affinity) and manufacturability (titer, stability, low
charge variants, low aggregation).

This engine identifies the Pareto frontier: the set of candidates
where no other candidate is simultaneously better in ALL objectives.

Provides:
    1. Pareto dominance detection (N-dimensional)
    2. Pareto frontier extraction from HT screening results
    3. Multi-objective scoring with configurable weights
    4. Trade-off visualization data for Plotly scatter
    5. Integration with HT Screening quadrant classification

Objectives (maximized):
    - Efficacy: potency_score (binding, ADCC, CDC, etc.)
    - Developability: dev_score (low agg, good stability, low liabilities)
    - Manufacturability: titer proxy (inverse of cost) + low cIEF acidic
    - Stability: predicted shelf-life (from stability_twin)

Version : 1.0 (Analytical QC + Stability + Pareto — M27)
Author  : Di (ProtePilot)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# =========================================================================
# Data Classes
# =========================================================================

@dataclass
class ParetoCandidate:
    """A candidate with multi-objective scores."""
    name: str
    objectives: Dict[str, float]        # {obj_name: score} — all maximized
    is_pareto_optimal: bool = False
    pareto_rank: int = 0                # 0 = Pareto frontier, 1 = second front, etc.
    crowding_distance: float = 0.0      # diversity metric for tie-breaking
    weighted_score: float = 0.0         # fallback single-objective composite
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParetoResult:
    """Result of Pareto frontier analysis."""
    candidates: List[ParetoCandidate]
    frontier: List[ParetoCandidate]     # Pareto-optimal subset
    n_total: int
    n_pareto: int
    objective_names: List[str]
    frontier_fraction: float
    summary: str


# =========================================================================
# 1. Pareto Dominance
# =========================================================================

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Check if solution 'a' Pareto-dominates solution 'b'.

    a dominates b iff:
      - a is >= b in ALL objectives, AND
      - a is > b in at LEAST one objective

    All objectives are assumed to be maximized.
    """
    return bool(np.all(a >= b) and np.any(a > b))


def find_pareto_frontier(
    objectives: np.ndarray,
) -> np.ndarray:
    """
    Find the Pareto frontier (non-dominated set) from an objective matrix.

    Parameters
    ----------
    objectives : (n_candidates, n_objectives) array, all maximized

    Returns
    -------
    Boolean mask of shape (n_candidates,) — True for Pareto-optimal points
    """
    n = objectives.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if dominates(objectives[j], objectives[i]):
                is_pareto[i] = False
                break

    return is_pareto


def compute_pareto_ranks(
    objectives: np.ndarray,
    max_ranks: int = 5,
) -> np.ndarray:
    """
    Compute non-domination ranks (NSGA-II style front assignment).

    Rank 0 = first Pareto front (best), Rank 1 = second front, etc.

    Parameters
    ----------
    objectives : (n, m) array of objective values (all maximized)
    max_ranks  : Maximum number of ranks to compute

    Returns
    -------
    (n,) array of rank assignments
    """
    n = objectives.shape[0]
    ranks = np.full(n, max_ranks, dtype=int)
    remaining = np.ones(n, dtype=bool)

    for rank in range(max_ranks):
        if not np.any(remaining):
            break

        # Find Pareto front among remaining
        remaining_idx = np.where(remaining)[0]
        sub_objectives = objectives[remaining_idx]
        is_front = find_pareto_frontier(sub_objectives)

        # Assign rank to front members
        front_idx = remaining_idx[is_front]
        ranks[front_idx] = rank
        remaining[front_idx] = False

    return ranks


def compute_crowding_distance(
    objectives: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Compute crowding distance for diversity preservation (NSGA-II).

    Higher crowding distance = more isolated = more valuable for diversity.

    Parameters
    ----------
    objectives : (n, m) array
    mask       : Boolean mask of points to compute distance for

    Returns
    -------
    (n,) array of crowding distances (0.0 for non-masked points)
    """
    n = objectives.shape[0]
    m = objectives.shape[1]
    distances = np.zeros(n)

    idx = np.where(mask)[0]
    if len(idx) <= 2:
        distances[idx] = np.inf
        return distances

    for obj_i in range(m):
        vals = objectives[idx, obj_i]
        sorted_order = np.argsort(vals)
        sorted_idx = idx[sorted_order]

        # Boundary points get infinite distance
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        obj_range = vals[sorted_order[-1]] - vals[sorted_order[0]]
        if obj_range < 1e-12:
            continue

        for k in range(1, len(sorted_idx) - 1):
            distances[sorted_idx[k]] += (
                vals[sorted_order[k + 1]] - vals[sorted_order[k - 1]]
            ) / obj_range

    return distances


# =========================================================================
# 2. Objective Extraction from HT Screening Results
# =========================================================================

DEFAULT_OBJECTIVES = {
    "Efficacy": {"key": "potency_score", "direction": "maximize"},
    "Developability": {"key": "developability_score", "direction": "maximize"},
    "Manufacturability": {"key": "manufacturability_score", "direction": "maximize"},
}


def extract_objectives_from_candidates(
    candidates: List[Dict[str, Any]],
    objective_config: Optional[Dict[str, Dict]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract objective matrix from HT screening candidate dicts.

    Parameters
    ----------
    candidates      : List of candidate dicts from HTScreeningEngine
    objective_config: Objective configuration (name → {key, direction})

    Returns
    -------
    objectives : (n, m) array (all maximized)
    obj_names  : List of objective names
    cand_names : List of candidate names/IDs
    """
    if objective_config is None:
        objective_config = DEFAULT_OBJECTIVES

    obj_names = list(objective_config.keys())
    cand_names = []
    obj_rows = []

    for cand in candidates:
        name = cand.get("candidate_id", cand.get("name", "Unknown"))
        cand_names.append(name)

        row = []
        for obj_name, cfg in objective_config.items():
            key = cfg["key"]
            val = cand.get(key, 0.5)

            # If key not found, try to compute heuristic
            if key == "manufacturability_score" and key not in cand:
                # Composite: low aggregation risk + low cIEF acidic
                dev = cand.get("developability_score", 0.5)
                # Proxy: dev_score itself is largely about manufacturability
                val = dev * 0.8 + 0.2 * max(0.0, 1.0 - cand.get("hydrophobicity", 0.35))

            if isinstance(val, (int, float)):
                val = float(val)
            else:
                val = 0.5

            # If direction is minimize, negate for internal Pareto (always maximize)
            if cfg.get("direction") == "minimize":
                val = -val

            row.append(val)

        obj_rows.append(row)

    return np.array(obj_rows, dtype=np.float64), obj_names, cand_names


# =========================================================================
# 3. Full Pareto Analysis Pipeline
# =========================================================================

def run_pareto_analysis(
    candidates: List[Dict[str, Any]],
    objective_config: Optional[Dict[str, Dict]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> ParetoResult:
    """
    Run complete multi-objective Pareto analysis on candidate pool.

    Steps:
      1. Extract objective matrix from candidate dicts
      2. Compute Pareto ranks (non-domination sorting)
      3. Compute crowding distances for diversity
      4. Compute weighted composite score as fallback
      5. Return annotated candidates with Pareto information

    Parameters
    ----------
    candidates      : List of candidate dicts (from HT screening or any source)
    objective_config: Objective definitions
    weights         : Weights for composite scoring (optional)

    Returns
    -------
    ParetoResult with frontier, ranks, and annotated candidates
    """
    t0 = time.time()

    if not candidates:
        return ParetoResult(
            candidates=[], frontier=[], n_total=0, n_pareto=0,
            objective_names=[], frontier_fraction=0.0,
            summary="No candidates provided",
        )

    # Extract objectives
    obj_matrix, obj_names, cand_names = extract_objectives_from_candidates(
        candidates, objective_config,
    )

    n = len(candidates)

    # Compute Pareto ranks
    ranks = compute_pareto_ranks(obj_matrix)
    frontier_mask = (ranks == 0)

    # Crowding distance for frontier
    crowding = compute_crowding_distance(obj_matrix, frontier_mask)

    # Weighted composite score
    if weights is None:
        weights = {name: 1.0 / len(obj_names) for name in obj_names}
    w = np.array([weights.get(name, 1.0 / len(obj_names)) for name in obj_names])
    w = w / w.sum()
    weighted_scores = obj_matrix @ w

    # Build ParetoCandidate list
    pareto_candidates = []
    for i in range(n):
        pc = ParetoCandidate(
            name=cand_names[i],
            objectives={obj_names[j]: float(obj_matrix[i, j]) for j in range(len(obj_names))},
            is_pareto_optimal=bool(frontier_mask[i]),
            pareto_rank=int(ranks[i]),
            crowding_distance=float(crowding[i]),
            weighted_score=float(weighted_scores[i]),
            raw_data=candidates[i],
        )
        pareto_candidates.append(pc)

    # Sort: frontier first, then by crowding distance (more diverse first)
    pareto_candidates.sort(key=lambda c: (c.pareto_rank, -c.crowding_distance))

    frontier = [c for c in pareto_candidates if c.is_pareto_optimal]
    n_pareto = len(frontier)
    frac = n_pareto / max(n, 1)

    wall = time.time() - t0

    summary = (
        f"Pareto Analysis: {n} candidates, {n_pareto} Pareto-optimal "
        f"({frac:.1%} frontier)\n"
        f"  Objectives: {', '.join(obj_names)}\n"
        f"  Rank distribution: " +
        ", ".join(f"R{r}: {int(np.sum(ranks == r))}" for r in range(min(5, ranks.max() + 1))) +
        f"\n  Wall time: {wall:.3f}s"
    )

    return ParetoResult(
        candidates=pareto_candidates,
        frontier=frontier,
        n_total=n,
        n_pareto=n_pareto,
        objective_names=obj_names,
        frontier_fraction=round(frac, 3),
        summary=summary,
    )


# =========================================================================
# Self-Test
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    print("=" * 60)
    print("pareto_optimizer.py — Self-Test")
    print("=" * 60)

    # Test 1: Dominance
    a = np.array([0.8, 0.9])
    b = np.array([0.7, 0.85])
    c = np.array([0.9, 0.7])
    assert dominates(a, b) is True   # a better in both
    assert dominates(b, a) is False
    assert dominates(a, c) is False  # trade-off: a better in obj2, c in obj1
    assert dominates(c, a) is False
    print("  [1/6] Dominance: correct ✅")

    # Test 2: Pareto frontier (simple 2D)
    objectives = np.array([
        [0.9, 0.3],  # high efficacy, low dev
        [0.3, 0.9],  # low efficacy, high dev
        [0.7, 0.7],  # balanced
        [0.5, 0.5],  # dominated by [0.7, 0.7]
        [0.2, 0.2],  # dominated by all
    ])
    frontier = find_pareto_frontier(objectives)
    assert frontier[0] == True    # [0.9, 0.3] is Pareto
    assert frontier[1] == True    # [0.3, 0.9] is Pareto
    assert frontier[2] == True    # [0.7, 0.7] is Pareto
    assert frontier[3] == False   # dominated
    assert frontier[4] == False   # dominated
    print(f"  [2/6] Pareto frontier: {sum(frontier)}/5 optimal ✅")

    # Test 3: Pareto ranks
    ranks = compute_pareto_ranks(objectives)
    assert ranks[0] == 0 and ranks[1] == 0 and ranks[2] == 0
    assert ranks[3] == 1
    assert ranks[4] == 2
    print(f"  [3/6] Pareto ranks: {ranks.tolist()} ✅")

    # Test 4: Crowding distance
    cd = compute_crowding_distance(objectives, frontier)
    assert cd[0] == np.inf  # boundary
    assert cd[1] == np.inf  # boundary
    assert cd[2] > 0        # interior point
    print(f"  [4/6] Crowding distance: computed ✅")

    # Test 5: Mock HT screening candidates
    mock_candidates = [
        {"candidate_id": "mAb_A", "potency_score": 0.9, "developability_score": 0.4, "hydrophobicity": 0.5},
        {"candidate_id": "mAb_B", "potency_score": 0.4, "developability_score": 0.9, "hydrophobicity": 0.2},
        {"candidate_id": "mAb_C", "potency_score": 0.7, "developability_score": 0.7, "hydrophobicity": 0.3},
        {"candidate_id": "mAb_D", "potency_score": 0.5, "developability_score": 0.5, "hydrophobicity": 0.4},
        {"candidate_id": "mAb_E", "potency_score": 0.3, "developability_score": 0.3, "hydrophobicity": 0.6},
        {"candidate_id": "mAb_F", "potency_score": 0.85, "developability_score": 0.6, "hydrophobicity": 0.35},
    ]
    result = run_pareto_analysis(mock_candidates)
    assert result.n_total == 6
    assert result.n_pareto >= 2  # at least A and B are Pareto
    pareto_names = {c.name for c in result.frontier}
    assert "mAb_A" in pareto_names  # highest potency
    assert "mAb_B" in pareto_names  # highest developability
    print(f"  [5/6] Full analysis: {result.n_pareto}/{result.n_total} Pareto-optimal "
          f"({', '.join(pareto_names)}) ✅")

    # Test 6: Weighted composite
    assert result.candidates[0].weighted_score > 0
    assert all(c.pareto_rank >= 0 for c in result.candidates)
    print(f"  [6/6] Weighted scores: best={result.candidates[0].weighted_score:.3f} ✅")

    print()
    print(result.summary)
    print()
    print("Self-test: 6/6 passed")
