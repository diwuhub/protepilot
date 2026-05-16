"""
src/scaleup_twin.py — Tech Transfer & Scale-Up Digital Twin
============================================================
ProtePilot — Milestone 23 · Version 1.0

Simulates the bioreactor scale-up from 2L bench-scale to 2000L
manufacturing scale, calculating engineering parameters under two
standard scale-up strategies:

    1. Constant P/V (Power per unit Volume)
    2. Constant kLa (Volumetric oxygen mass transfer coefficient)

Science Background
------------------
Scale-up is one of the highest-risk steps in biomanufacturing.
Key challenges:

  - Tip speed increases with impeller diameter → shear stress on cells
  - kLa depends on power input and superficial gas velocity
  - Mixing time increases non-linearly → nutrient gradients
  - CHO cells are sensitive to shear (tip speed >1.5 m/s → damage)

Correlations Used
-----------------
  P = Np · ρ · N³ · D_i⁵          (Power number correlation)
  P/V = const  →  N₂ = N₁ · (V₁/V₂)^(2/3) · (D₁/D₂)^(5/3)

  kLa = C · (P/V)^α · (v_s)^β     (van't Riet correlation)
  Typical: α ≈ 0.4, β ≈ 0.5

  Tip Speed = π · D_i · N

  Shear Rate (γ) ~ N · D_i / (T - D_i)  (Kolmogorov-based estimate)

Outputs
-------
  - Large-scale RPM, tip speed, P/V at both strategies
  - Shear stress assessment (safe / warning / danger)
  - Expected titer scaling factor
  - Engineering recommendation (constant P/V vs constant kLa)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ===========================================================================
# 1. Constants — Standard Bioreactor Geometry
# ===========================================================================

# Typical stirred-tank bioreactor geometric ratios
# H/T = 2.0 (height/diameter), D_i/T = 0.33 (impeller/tank diameter)
# Np = 1.5 (Rushton turbine power number for aerated system)

_NP = 1.5           # Power number (Rushton turbine, aerated)
_RHO = 1010.0       # Culture density (kg/m³, close to water)
_PI = math.pi

# CHO cell shear sensitivity thresholds
_TIP_SPEED_SAFE = 1.0      # m/s — generally safe for CHO
_TIP_SPEED_WARNING = 1.5   # m/s — caution zone
_TIP_SPEED_DANGER = 2.0    # m/s — cell damage likely

# kLa van't Riet correlation parameters (air-water, Rushton)
_KLA_ALPHA = 0.4    # P/V exponent
_KLA_BETA = 0.5     # superficial gas velocity exponent
_KLA_C = 0.026      # empirical constant (for kLa in 1/s)


# ===========================================================================
# 2. Data Classes
# ===========================================================================

@dataclass
class BioreactorGeometry:
    """Physical dimensions of a stirred-tank bioreactor."""
    volume_L: float           # working volume (L)
    tank_diameter_m: float    # internal diameter (m)
    impeller_diameter_m: float  # impeller diameter (m)
    liquid_height_m: float    # liquid height (m)
    n_impellers: int = 1      # number of impellers (large scale may have 2-3)

    @classmethod
    def from_volume(cls, volume_L: float) -> "BioreactorGeometry":
        """Estimate bioreactor geometry from working volume using standard ratios."""
        vol_m3 = volume_L / 1000.0
        # H/T = 2.0 for standard STR
        # V = π/4 · T² · H = π/4 · T² · 2T = π/2 · T³
        tank_diam = (vol_m3 * 2.0 / _PI) ** (1.0 / 3.0)
        liquid_h = 2.0 * tank_diam
        imp_diam = 0.33 * tank_diam
        n_imp = 1 if volume_L <= 200 else (2 if volume_L <= 5000 else 3)
        return cls(
            volume_L=volume_L,
            tank_diameter_m=round(tank_diam, 4),
            impeller_diameter_m=round(imp_diam, 4),
            liquid_height_m=round(liquid_h, 4),
            n_impellers=n_imp,
        )


@dataclass
class ScaleUpResult:
    """Results from a tech transfer / scale-up simulation."""
    # Geometry
    small_scale: BioreactorGeometry
    large_scale: BioreactorGeometry

    # Small-scale operating conditions
    small_rpm: float
    small_tip_speed: float           # m/s
    small_pv: float                  # W/m³ (P/V)
    small_kla: float                 # 1/h
    small_titer: float               # g/L from upstream simulation

    # Constant P/V strategy
    pv_rpm: float                    # large-scale RPM
    pv_tip_speed: float              # m/s
    pv_kla: float                    # 1/h
    pv_shear_status: str             # "Safe", "Warning", "Danger"

    # Constant kLa strategy
    kla_rpm: float
    kla_tip_speed: float
    kla_pv: float                    # W/m³
    kla_shear_status: str

    # Recommendations
    recommended_strategy: str        # "Constant P/V" or "Constant kLa"
    predicted_titer_large: float     # expected titer at large scale
    titer_scaling_factor: float      # fraction of bench titer retained
    mixing_time_small: float         # seconds
    mixing_time_large: float         # seconds

    # Warnings
    warnings: List[str]
    summary: str


# ===========================================================================
# 3. Engineering Calculations
# ===========================================================================

def _calc_power(rpm: float, impeller_d: float, n_impellers: int = 1) -> float:
    """Calculate impeller power (W) using P = Np · ρ · N³ · D⁵."""
    N = rpm / 60.0  # rev/s
    return _NP * _RHO * (N ** 3) * (impeller_d ** 5) * n_impellers


def _calc_pv(rpm: float, geom: BioreactorGeometry) -> float:
    """Calculate P/V (W/m³)."""
    power = _calc_power(rpm, geom.impeller_diameter_m, geom.n_impellers)
    vol_m3 = geom.volume_L / 1000.0
    return power / vol_m3 if vol_m3 > 0 else 0.0


def _calc_tip_speed(rpm: float, impeller_d: float) -> float:
    """Calculate impeller tip speed (m/s)."""
    return _PI * impeller_d * (rpm / 60.0)


def _calc_kla(pv: float, vvm: float = 0.02, vol_m3: float = 0.002) -> float:
    """
    Estimate kLa (1/h) using van't Riet correlation.

    kLa = C · (P/V)^α · (v_s)^β × 3600

    v_s (superficial gas velocity) = vvm · V / A_cross (m/s)
    """
    # Superficial velocity estimate
    tank_d = (vol_m3 * 2.0 / _PI) ** (1.0 / 3.0)
    area = _PI / 4.0 * tank_d ** 2
    v_s = (vvm * vol_m3 / 60.0) / area if area > 0 else 0.005  # m/s

    kla_per_s = _KLA_C * (max(0.1, pv) ** _KLA_ALPHA) * (max(0.001, v_s) ** _KLA_BETA)
    return kla_per_s * 3600.0  # convert to 1/h


def _calc_mixing_time(rpm: float, geom: BioreactorGeometry) -> float:
    """
    Estimate mixing time (seconds) using Norwood-Metzner correlation.

    t_mix ∝ (T² · H) / (N · D_i²) for turbulent flow
    """
    N = rpm / 60.0  # rev/s
    if N <= 0:
        return 999.0
    T = geom.tank_diameter_m
    H = geom.liquid_height_m
    Di = geom.impeller_diameter_m
    # Empirical: t_mix = 5.9 · (T/Di)^2.3 · (H/T)^0.5 / N
    t_mix = 5.9 * (T / Di) ** 2.3 * (H / T) ** 0.5 / N
    return max(1.0, t_mix)


def _assess_shear(tip_speed: float) -> str:
    """Assess CHO cell shear stress risk from tip speed."""
    if tip_speed <= _TIP_SPEED_SAFE:
        return "Safe"
    elif tip_speed <= _TIP_SPEED_WARNING:
        return "Warning"
    else:
        return "Danger"


# ===========================================================================
# 4. Scale-Up RPM Calculation
# ===========================================================================

def _rpm_constant_pv(
    small_rpm: float,
    small_geom: BioreactorGeometry,
    large_geom: BioreactorGeometry,
) -> float:
    """
    Calculate large-scale RPM maintaining constant P/V.

    P/V = Np · ρ · N³ · D_i⁵ / V = const
    → N₂ = N₁ · (D_i1/D_i2)^(5/3) · (V₂/V₁)^(1/3)
    """
    N1 = small_rpm / 60.0
    Di_ratio = small_geom.impeller_diameter_m / large_geom.impeller_diameter_m
    n_imp_ratio = small_geom.n_impellers / large_geom.n_impellers
    N2 = N1 * (Di_ratio ** (5.0 / 3.0)) * (
        (large_geom.volume_L / small_geom.volume_L) ** (1.0 / 3.0)
    ) * (n_imp_ratio ** (1.0 / 3.0))
    return max(10.0, N2 * 60.0)  # convert back to RPM


def _rpm_constant_kla(
    small_rpm: float,
    small_geom: BioreactorGeometry,
    large_geom: BioreactorGeometry,
    target_kla: float,
) -> float:
    """
    Find large-scale RPM that achieves the same kLa as small scale.

    Iterative approach: sweep RPM to match target kLa.
    """
    vol_m3 = large_geom.volume_L / 1000.0
    best_rpm = 50.0
    best_diff = float('inf')

    for rpm in np.arange(10, 300, 0.5):
        pv = _calc_pv(rpm, large_geom)
        kla = _calc_kla(pv, vvm=0.02, vol_m3=vol_m3)
        diff = abs(kla - target_kla)
        if diff < best_diff:
            best_diff = diff
            best_rpm = rpm

    return float(best_rpm)


# ===========================================================================
# 5. Main Tech Transfer Simulation
# ===========================================================================

def run_scaleup_simulation(
    small_volume_L: float = 2.0,
    large_volume_L: float = 2000.0,
    small_rpm: float = 200.0,
    bench_titer: float = 5.0,
    vvm: float = 0.02,
) -> ScaleUpResult:
    """
    Simulate tech transfer from bench to manufacturing scale.

    Parameters
    ----------
    small_volume_L : Bench-scale working volume (L)
    large_volume_L : Manufacturing-scale working volume (L)
    small_rpm      : Bench-scale agitation speed (RPM)
    bench_titer    : Bench-scale harvest titer from upstream twin (g/L)
    vvm            : Volume of gas per Volume of liquid per Minute

    Returns
    -------
    ScaleUpResult with engineering parameters and recommendations
    """
    log.info(f"Scale-up simulation: {small_volume_L}L → {large_volume_L}L")

    # ---- Geometry ----
    small_geom = BioreactorGeometry.from_volume(small_volume_L)
    large_geom = BioreactorGeometry.from_volume(large_volume_L)

    # ---- Small-scale parameters ----
    small_tip = _calc_tip_speed(small_rpm, small_geom.impeller_diameter_m)
    small_pv = _calc_pv(small_rpm, small_geom)
    small_kla = _calc_kla(small_pv, vvm, small_geom.volume_L / 1000.0)
    small_mix = _calc_mixing_time(small_rpm, small_geom)

    # ---- Strategy 1: Constant P/V ----
    pv_rpm = _rpm_constant_pv(small_rpm, small_geom, large_geom)
    pv_tip = _calc_tip_speed(pv_rpm, large_geom.impeller_diameter_m)
    pv_pv = _calc_pv(pv_rpm, large_geom)
    pv_kla = _calc_kla(pv_pv, vvm, large_geom.volume_L / 1000.0)
    pv_shear = _assess_shear(pv_tip)

    # ---- Strategy 2: Constant kLa ----
    kla_rpm = _rpm_constant_kla(small_rpm, small_geom, large_geom, small_kla)
    kla_tip = _calc_tip_speed(kla_rpm, large_geom.impeller_diameter_m)
    kla_pv = _calc_pv(kla_rpm, large_geom)
    kla_shear = _assess_shear(kla_tip)

    # ---- Mixing times ----
    pv_mix = _calc_mixing_time(pv_rpm, large_geom)
    kla_mix = _calc_mixing_time(kla_rpm, large_geom)
    large_mix = min(pv_mix, kla_mix)

    # ---- Titer prediction ----
    # Titer at large scale is typically 85–95% of bench scale
    # due to heterogeneity, suboptimal mixing, and shear effects
    warnings = []

    # Base scaling factor
    titer_factor = 0.92  # typical 8% loss

    # Shear penalty
    best_shear = pv_shear if pv_tip <= kla_tip else kla_shear
    best_tip = min(pv_tip, kla_tip)
    if best_tip > _TIP_SPEED_DANGER:
        titer_factor *= 0.70  # 30% titer loss from cell death
        warnings.append(
            f"CRITICAL: Tip speed {best_tip:.2f} m/s exceeds CHO shear "
            f"threshold ({_TIP_SPEED_DANGER} m/s). Expect significant cell death "
            f"and titer drop. Consider reducing RPM or using low-shear impeller."
        )
    elif best_tip > _TIP_SPEED_WARNING:
        titer_factor *= 0.85  # 15% penalty
        warnings.append(
            f"WARNING: Tip speed {best_tip:.2f} m/s is in the caution zone "
            f"({_TIP_SPEED_WARNING}–{_TIP_SPEED_DANGER} m/s). Some CHO cell "
            f"damage possible. Monitor viability closely at manufacturing scale."
        )

    # Mixing time penalty (>120s → nutrient gradients → lower titer)
    if large_mix > 120:
        mix_penalty = min(0.15, (large_mix - 120) / 1000.0)
        titer_factor *= (1.0 - mix_penalty)
        warnings.append(
            f"WARNING: Mixing time {large_mix:.0f}s exceeds 120s. Nutrient "
            f"gradients may reduce productivity."
        )

    predicted_titer = bench_titer * titer_factor

    # ---- Recommendation ----
    # Prefer the strategy with lower shear risk
    if pv_shear == "Danger" and kla_shear != "Danger":
        rec = "Constant kLa"
    elif kla_shear == "Danger" and pv_shear != "Danger":
        rec = "Constant P/V"
    elif pv_tip <= kla_tip:
        rec = "Constant P/V"
    else:
        rec = "Constant kLa"

    # ---- Summary ----
    summary_lines = [
        f"Tech Transfer Simulation: {small_volume_L:.0f}L → {large_volume_L:.0f}L",
        f"",
        f"Bench Scale ({small_volume_L:.0f}L):",
        f"  RPM={small_rpm:.0f}, Tip Speed={small_tip:.3f} m/s, "
        f"P/V={small_pv:.1f} W/m³, kLa={small_kla:.1f} 1/h, "
        f"Mixing={small_mix:.1f}s",
        f"",
        f"Constant P/V Strategy ({large_volume_L:.0f}L):",
        f"  RPM={pv_rpm:.1f}, Tip Speed={pv_tip:.3f} m/s, "
        f"P/V={pv_pv:.1f} W/m³, kLa={pv_kla:.1f} 1/h — Shear: {pv_shear}",
        f"",
        f"Constant kLa Strategy ({large_volume_L:.0f}L):",
        f"  RPM={kla_rpm:.1f}, Tip Speed={kla_tip:.3f} m/s, "
        f"P/V={kla_pv:.1f} W/m³ — Shear: {kla_shear}",
        f"",
        f"Recommended: {rec}",
        f"Predicted titer at {large_volume_L:.0f}L: {predicted_titer:.2f} g/L "
        f"({titer_factor:.0%} of bench)",
    ]
    if warnings:
        summary_lines.append("")
        for w in warnings:
            summary_lines.append(f"  ⚠ {w}")

    return ScaleUpResult(
        small_scale=small_geom,
        large_scale=large_geom,
        small_rpm=small_rpm,
        small_tip_speed=round(small_tip, 4),
        small_pv=round(small_pv, 2),
        small_kla=round(small_kla, 2),
        small_titer=bench_titer,
        pv_rpm=round(pv_rpm, 1),
        pv_tip_speed=round(pv_tip, 4),
        pv_kla=round(pv_kla, 2),
        pv_shear_status=pv_shear,
        kla_rpm=round(kla_rpm, 1),
        kla_tip_speed=round(kla_tip, 4),
        kla_pv=round(kla_pv, 2),
        kla_shear_status=kla_shear,
        recommended_strategy=rec,
        predicted_titer_large=round(predicted_titer, 3),
        titer_scaling_factor=round(titer_factor, 3),
        mixing_time_small=round(small_mix, 1),
        mixing_time_large=round(large_mix, 1),
        warnings=warnings,
        summary="\n".join(summary_lines),
    )


# ===========================================================================
# 6. Self-Test
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("scaleup_twin.py — Self-Test")
    print("=" * 60)

    passed = 0
    total = 6

    # Test 1: Basic 2L → 2000L
    result = run_scaleup_simulation(2.0, 2000.0, 200, 5.5)
    assert result.pv_rpm > 0 and result.kla_rpm > 0
    print(f"  [1/6] Basic scale-up: 2L→2000L OK ✅")
    passed += 1

    # Test 2: Geometry calculations
    g2 = BioreactorGeometry.from_volume(2.0)
    g2000 = BioreactorGeometry.from_volume(2000.0)
    assert g2000.tank_diameter_m > g2.tank_diameter_m
    assert g2000.impeller_diameter_m > g2.impeller_diameter_m
    print(f"  [2/6] Geometry: 2L tank={g2.tank_diameter_m:.3f}m, "
          f"2000L tank={g2000.tank_diameter_m:.3f}m ✅")
    passed += 1

    # Test 3: Tip speed increases with scale
    assert result.pv_tip_speed > result.small_tip_speed
    print(f"  [3/6] Tip speed: small={result.small_tip_speed:.3f} → "
          f"P/V={result.pv_tip_speed:.3f} m/s ✅")
    passed += 1

    # Test 4: Titer scaling
    assert 0.5 < result.titer_scaling_factor <= 1.0
    print(f"  [4/6] Titer scaling: {result.titer_scaling_factor:.0%} → "
          f"{result.predicted_titer_large:.2f} g/L ✅")
    passed += 1

    # Test 5: Recommendation is valid
    assert result.recommended_strategy in ("Constant P/V", "Constant kLa")
    print(f"  [5/6] Recommended: {result.recommended_strategy} ✅")
    passed += 1

    # Test 6: Shear assessment
    assert result.pv_shear_status in ("Safe", "Warning", "Danger")
    assert result.kla_shear_status in ("Safe", "Warning", "Danger")
    print(f"  [6/6] Shear: P/V={result.pv_shear_status}, "
          f"kLa={result.kla_shear_status} ✅")
    passed += 1

    print(f"\nSummary:\n{result.summary}")
    print(f"\n{'=' * 60}")
    print(f"Self-test: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
