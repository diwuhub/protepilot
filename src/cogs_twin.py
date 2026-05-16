"""
src/cogs_twin.py — Cost of Goods Sold (COGS) Digital Twin
==========================================================
ProtePilot — Milestone 21 · Version 1.0

Calculates commercial manufacturing cost ($/gram purified API) for
monoclonal antibody production.  Connects upstream titer, downstream
yield, and user-defined cost inputs to produce a full cost breakdown.

Cost Model
----------
  Batch Output (g) = Titer (g/L) * Volume (L) * Downstream Yield (0-1)

  Upstream Cost     = Media_cost_per_L * Volume + Seed_train_fixed
  Downstream Cost   = Resin_cost + Buffer/Consumables + Column packing
  Overhead          = QC_testing + Facility + Labor + Utilities

  COGS ($/g) = Total_batch_cost / Batch_output_g
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# ===========================================================================
# 1. Data Classes
# ===========================================================================

@dataclass
class COGSInputs:
    """User-configurable manufacturing cost inputs."""
    # Upstream
    bioreactor_volume_L: float = 2000.0
    media_cost_per_L: float = 45.0          # $/L (chemically defined media)
    seed_train_cost: float = 15000.0        # $ fixed per batch

    # Downstream
    protein_a_resin_cost: float = 50000.0   # $ per batch (resin lifecycle amortized)
    buffer_consumables: float = 12000.0     # $ per batch
    column_packing_cost: float = 8000.0     # $ per batch
    uf_df_cost: float = 5000.0              # $ UF/DF step

    # Overhead (fixed per batch)
    # v7.3.2: Calibrated to industry benchmarks (BioPhorum 2023 COGS survey,
    # Kelley 2009 "Industrialization of mAb production technology").
    # Previous: QC=$25K, facility=$40K, labor=$30K → unrealistic $37/g.
    # Revised to reflect GMP QC panel ($65K includes release testing,
    # stability pulls, environmental monitoring), commercial-scale facility
    # depreciation ($95K amortized over 15-year life), and fully burdened
    # labor for 2000L process ($55K per batch).
    qc_testing: float = 65000.0             # $ per batch (GMP QC panel)
    facility_cost: float = 95000.0          # $ per batch (depreciation + utilities)
    labor_cost: float = 55000.0             # $ per batch (fully burdened)
    utilities_cost: float = 8000.0          # $ per batch

    # Process parameters (from upstream/downstream twins)
    titer_g_per_L: float = 5.0              # from upstream simulation
    downstream_yield: float = 0.70          # from DoE optimization (0-1)


@dataclass
class COGSResult:
    """Full COGS calculation result."""
    # Batch output
    batch_output_g: float
    batch_output_kg: float

    # Cost components
    upstream_cost: float
    downstream_cost: float
    overhead_cost: float
    total_batch_cost: float

    # Per-gram metrics
    cogs_per_gram: float
    upstream_per_gram: float
    downstream_per_gram: float
    overhead_per_gram: float

    # Breakdown percentages
    upstream_pct: float
    downstream_pct: float
    overhead_pct: float

    # Flags
    is_commercial: bool            # True if < $150/g
    cost_rating: str               # "Excellent" / "Good" / "Marginal" / "Non-viable"

    # Inputs used
    inputs: COGSInputs


# ===========================================================================
# 2. COGS Calculator
# ===========================================================================

def calculate_cogs(
    inputs: COGSInputs,
    molecule_class: Optional[str] = None,
) -> COGSResult:
    """
    Calculate the full Cost of Goods Sold for a batch.

    Parameters
    ----------
    inputs         : COGSInputs with all cost parameters and process metrics
    molecule_class : Optional molecule type. Adjusts default purification and
                     scale assumptions for non-mAb formats:
                     - peptide: no Protein A, smaller bioreactor (200L), HPLC purification
                     - single_domain: no Protein A, smaller scale (500L), IEX-only capture
                     - adc: add conjugation step cost ($25K)

    Returns
    -------
    COGSResult with cost breakdown and per-gram COGS
    """
    # -- Molecule-class-aware cost adjustments --
    # These override ONLY defaults that haven't been user-customized.
    _protein_a_cost = inputs.protein_a_resin_cost
    _bioreactor_vol = inputs.bioreactor_volume_L
    _extra_ds_cost = 0.0  # additional downstream costs (e.g., conjugation)

    # QC/overhead scale factor: smaller processes have proportionally lower overhead
    _qc_scale = 1.0
    _facility_scale = 1.0
    _labor_scale = 1.0

    if molecule_class == "peptide":
        # Peptides: no Protein A capture (use RP-HPLC instead), smaller scale
        # Synthetic peptide or E.coli fermentation → 200L scale
        if inputs.protein_a_resin_cost == 50000.0:  # still at default
            _protein_a_cost = 15000.0  # RP-HPLC resin is cheaper
        if inputs.bioreactor_volume_L == 2000.0:  # still at default
            _bioreactor_vol = 200.0  # synthetic/E.coli scale
        # Peptide QC is simpler: RP-HPLC identity + purity, no glycan/charge
        # variant panels. Facility and labor scale with bioreactor volume.
        # Literature: Lax 2014 "Cost of goods in peptide manufacturing"
        _qc_scale = 0.40       # simpler release panel (~$26K vs $65K)
        _facility_scale = 0.30  # 200L vs 2000L footprint
        _labor_scale = 0.40     # fewer unit operations
    elif molecule_class == "single_domain":
        # Nanobodies/VHH: no Protein A, IEX capture is cheaper
        if inputs.protein_a_resin_cost == 50000.0:
            _protein_a_cost = 20000.0  # IEX resin lifecycle
        if inputs.bioreactor_volume_L == 2000.0:
            _bioreactor_vol = 500.0  # smaller CHO/E.coli scale
        # Moderate reduction: still needs proper protein QC but smaller scale
        _qc_scale = 0.70
        _facility_scale = 0.50
        _labor_scale = 0.60
    elif molecule_class == "adc":
        # ADCs: mAb purification + conjugation chemistry step
        _extra_ds_cost = 25000.0  # conjugation reagents + QC per batch
        # ADC QC is MORE complex (DAR, free drug, potency) — no reduction

    # -- Batch output --
    batch_output_g = inputs.titer_g_per_L * _bioreactor_vol * inputs.downstream_yield
    batch_output_g = max(batch_output_g, 0.001)  # avoid division by zero
    batch_output_kg = batch_output_g / 1000.0

    # -- Upstream costs --
    upstream_cost = (
        inputs.media_cost_per_L * _bioreactor_vol
        + inputs.seed_train_cost
    )

    # -- Downstream costs --
    downstream_cost = (
        _protein_a_cost
        + inputs.buffer_consumables
        + inputs.column_packing_cost
        + inputs.uf_df_cost
        + _extra_ds_cost
    )

    # -- Overhead costs (molecule-class scaled) --
    overhead_cost = (
        inputs.qc_testing * _qc_scale
        + inputs.facility_cost * _facility_scale
        + inputs.labor_cost * _labor_scale
        + inputs.utilities_cost
    )

    # -- Totals --
    total_batch_cost = upstream_cost + downstream_cost + overhead_cost

    # -- Per-gram --
    cogs_per_gram = total_batch_cost / batch_output_g
    upstream_per_gram = upstream_cost / batch_output_g
    downstream_per_gram = downstream_cost / batch_output_g
    overhead_per_gram = overhead_cost / batch_output_g

    # -- Percentages --
    upstream_pct = (upstream_cost / total_batch_cost * 100) if total_batch_cost > 0 else 0
    downstream_pct = (downstream_cost / total_batch_cost * 100) if total_batch_cost > 0 else 0
    overhead_pct = (overhead_cost / total_batch_cost * 100) if total_batch_cost > 0 else 0

    # -- Cost rating --
    if cogs_per_gram < 50:
        cost_rating = "Excellent"
    elif cogs_per_gram < 100:
        cost_rating = "Good"
    elif cogs_per_gram < 150:
        cost_rating = "Marginal"
    else:
        cost_rating = "Non-viable"

    is_commercial = cogs_per_gram < 150.0

    log.info(
        "COGS: %.0f g output (%.1f g/L * %.0f L * %.0f%% yield) = $%.2f/g (%s)",
        batch_output_g, inputs.titer_g_per_L, inputs.bioreactor_volume_L,
        inputs.downstream_yield * 100, cogs_per_gram, cost_rating,
    )

    return COGSResult(
        batch_output_g=round(batch_output_g, 1),
        batch_output_kg=round(batch_output_kg, 3),
        upstream_cost=round(upstream_cost, 2),
        downstream_cost=round(downstream_cost, 2),
        overhead_cost=round(overhead_cost, 2),
        total_batch_cost=round(total_batch_cost, 2),
        cogs_per_gram=round(cogs_per_gram, 2),
        upstream_per_gram=round(upstream_per_gram, 2),
        downstream_per_gram=round(downstream_per_gram, 2),
        overhead_per_gram=round(overhead_per_gram, 2),
        upstream_pct=round(upstream_pct, 1),
        downstream_pct=round(downstream_pct, 1),
        overhead_pct=round(overhead_pct, 1),
        is_commercial=is_commercial,
        cost_rating=cost_rating,
        inputs=inputs,
    )


# ===========================================================================
# 3. High-Level API
# ===========================================================================

def run_cogs_analysis(
    titer_g_per_L: float = 5.0,
    downstream_yield: float = 0.70,
    bioreactor_volume_L: float = 2000.0,
    media_cost_per_L: float = 45.0,
    protein_a_resin_cost: float = 50000.0,
) -> COGSResult:
    """
    One-call API for COGS calculation.

    Parameters
    ----------
    titer_g_per_L        : From upstream bioreactor simulation
    downstream_yield     : From DoE purification optimization (0-1)
    bioreactor_volume_L  : Manufacturing scale
    media_cost_per_L     : Cell culture media cost
    protein_a_resin_cost : Protein A chromatography resin (amortized per batch)

    Returns
    -------
    COGSResult with full cost breakdown
    """
    inputs = COGSInputs(
        titer_g_per_L=titer_g_per_L,
        downstream_yield=downstream_yield,
        bioreactor_volume_L=bioreactor_volume_L,
        media_cost_per_L=media_cost_per_L,
        protein_a_resin_cost=protein_a_resin_cost,
    )
    return calculate_cogs(inputs)


def cogs_to_dict(result: COGSResult) -> Dict[str, Any]:
    """Serialize for session state."""
    return {
        "batch_output_g": result.batch_output_g,
        "batch_output_kg": result.batch_output_kg,
        "cogs_per_gram": result.cogs_per_gram,
        "total_batch_cost": result.total_batch_cost,
        "upstream_cost": result.upstream_cost,
        "downstream_cost": result.downstream_cost,
        "overhead_cost": result.overhead_cost,
        "upstream_pct": result.upstream_pct,
        "downstream_pct": result.downstream_pct,
        "overhead_pct": result.overhead_pct,
        "cost_rating": result.cost_rating,
        "is_commercial": result.is_commercial,
        "titer_g_per_L": result.inputs.titer_g_per_L,
        "downstream_yield": result.inputs.downstream_yield,
        "bioreactor_volume_L": result.inputs.bioreactor_volume_L,
    }


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    passed = 0

    # Test 1: Basic calculation
    r = run_cogs_analysis(titer_g_per_L=5.0, downstream_yield=0.70)
    assert r.batch_output_g > 0
    assert r.cogs_per_gram > 0
    print(f"Test 1 PASS: {r.batch_output_g:.0f}g output, ${r.cogs_per_gram:.2f}/g ({r.cost_rating})")
    passed += 1

    # Test 2: Higher titer = lower COGS
    r_high = run_cogs_analysis(titer_g_per_L=10.0, downstream_yield=0.85)
    r_low = run_cogs_analysis(titer_g_per_L=2.0, downstream_yield=0.50)
    assert r_high.cogs_per_gram < r_low.cogs_per_gram
    print(f"Test 2 PASS: high_titer=${r_high.cogs_per_gram:.2f}/g < low_titer=${r_low.cogs_per_gram:.2f}/g")
    passed += 1

    # Test 3: Percentages sum to ~100
    total_pct = r.upstream_pct + r.downstream_pct + r.overhead_pct
    assert 99.5 < total_pct < 100.5
    print(f"Test 3 PASS: cost split = {r.upstream_pct:.1f}% + {r.downstream_pct:.1f}% + {r.overhead_pct:.1f}%")
    passed += 1

    # Test 4: Serialization
    d = cogs_to_dict(r)
    assert "cogs_per_gram" in d and "cost_rating" in d
    print(f"Test 4 PASS: serialization OK")
    passed += 1

    # Test 5: Non-viable detection
    r_bad = run_cogs_analysis(titer_g_per_L=0.5, downstream_yield=0.30)
    assert not r_bad.is_commercial
    assert r_bad.cost_rating == "Non-viable"
    print(f"Test 5 PASS: low titer+yield = ${r_bad.cogs_per_gram:.2f}/g (Non-viable)")
    passed += 1

    print(f"\n{'='*50}")
    print(f"cogs_twin self-test: {passed}/5 passed")
    sys.exit(0 if passed == 5 else 1)
