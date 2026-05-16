"""
PropertyMapper.py  ·  ProtePilot
===========================================================
Protein Physical Properties -> CADET SMA Parameter Mapper

Version   : 7.2 (CADET-Scale ka Recalibration + ML-First + Bispecific)

Physical Model
------------------------------------------------------------
SMA (Steric Mass Action) model core parameters:

  nu  (nu)      Characteristic charge number  <- related to protein net charge / pI
  ka           Adsorption rate constant       <- DECOUPLED from nu, flat baseline
  kd           Desorption rate constant       <- fixed at 1000.0 (CADET SMA standard)
  Lambda       Ionic capacity                 <- column property, fixed
  sigma        Steric shielding factor        <- related to protein MW (size)

Architecture (v7.0 — ML-First Override + Bispecific)
------------------------------------------------------------
  Priority chain:
    1. ML-predicted ka/nu (from PyTorch MLP) — PRIMARY source
    2. Static thermodynamic formulas         — FALLBACK only

  v7.0 Bispecific Species Mapping:
    - map_bispecific_species(chain_a, chain_b) builds AA/AB/BB species
    - Each species gets independent SMA parameters
    - ML override applies to AB (target) species only

  When ml_override dict is provided with {"ka": float, "nu": float},
  those values replace the static empirical calculations entirely.
  The PropertyMapper then applies only the structural corrections
  (sigma, lambda, variant offsets) on top of the ML predictions.

  This enables the neural network to "steer" the system toward
  realistic 15-20 min elution windows while the deterministic
  formulas remain as a safety net.

Calibration Rationale (v5.0 baseline, preserved as fallback)
------------------------------------------------------------
  - nu_base: 2.5 (mAb ~2-3 CEX binding sites at pH 6-7)
  - ka_base: 1.2 (lower adsorption rate -> weaker initial capture)
  - Combined: Keq ~0.0013, yielding ~15-20 min elution under
    15 mM/min gradient (50->500 mM, 0.25m column)

  Process conditions: L=0.25m, epsilon=0.37, v=5.75e-4 m/s
  Gradient: 50 -> 500 mM NaCl over 1800s (30 min) at 15 mM/min

References
------------------------------------------------------------
  Brooks & Cramer (1992) AIChE J. 38(12):1969-1978
  Yamamoto et al. (1987) J. Chromatogr. 409:101-113
  Du et al. (2012) mAbs 4(5):578-585 — charge variant analysis
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import copy

log = logging.getLogger("PropertyMapper")


# ---------------------------------------------------------------------------
# Input: Protein Physical Properties
# ---------------------------------------------------------------------------

@dataclass
class ProteinProperties:
    """
    Measurable or predictable physicochemical properties of a protein.

    Attributes
    ----------
    name           : Protein / molecule name
    sequence       : Amino acid sequence (single-letter code, optional)
    pI             : Isoelectric point (pH units), mAb typical range 7-9
    MW_kDa         : Molecular weight (kDa), mAb typical ~150
    hydrophobicity : Hydrophobicity score in [0, 1], 0=hydrophilic, 1=highly hydrophobic
    pH_working     : Chromatography working pH (buffer pH)
    ptm_profile    : Post-translational modification profile
                     {"deamidation_sites": int, "oxidation_sites": int}
    notes          : Optional notes
    """
    name:          str
    pI:            float
    MW_kDa:        float
    hydrophobicity: float                       # normalized to [0, 1]
    pH_working:    float = 7.0
    sequence:      Optional[str]  = None
    gravy_score:   Optional[float] = None       # Raw GRAVY (Kyte-Doolittle), typically -0.8 to +0.2 for mAbs
    ptm_profile:   Optional[Dict[str, int]] = None
    notes:         Optional[str]  = None

    def __post_init__(self):
        if not (0.0 <= self.hydrophobicity <= 1.0):
            raise ValueError(
                f"hydrophobicity={self.hydrophobicity} must be in [0, 1]"
            )
        if not (0.0 < self.pI < 14.0):
            raise ValueError(f"pI={self.pI} out of valid range (0-14)")
        if self.MW_kDa <= 0:
            raise ValueError(f"MW_kDa={self.MW_kDa} must be > 0")
        if self.ptm_profile is not None:
            for key in ("deamidation_sites", "oxidation_sites"):
                if key in self.ptm_profile:
                    val = self.ptm_profile[key]
                    if not isinstance(val, (int, float)) or val < 0:
                        raise ValueError(
                            f"ptm_profile['{key}']={val} must be non-negative"
                        )

    def net_charge_factor(self) -> float:
        """Absolute deviation between working pH and pI."""
        return abs(self.pH_working - self.pI)

    def __repr__(self) -> str:
        ptm_str = f", ptm={self.ptm_profile}" if self.ptm_profile else ""
        seq_str = f", seq={len(self.sequence)}aa" if self.sequence else ""
        return (
            f"ProteinProperties({self.name!r}, pI={self.pI}, "
            f"MW={self.MW_kDa} kDa, h={self.hydrophobicity:.2f}, "
            f"pH={self.pH_working}{ptm_str}{seq_str})"
        )


# ---------------------------------------------------------------------------
# Mapper Configuration
# ---------------------------------------------------------------------------

@dataclass
class MapperConfig:
    """
    Tunable hyperparameters for PropertyMapper.

    v5.0 Principled Recalibration (preserved as FALLBACK):
      - nu_hardcoded_base lowered from 3.5 to 2.5
      - ka_flat_base lowered from 2.0 to 1.2
      - These serve as safety-net defaults when ML override is unavailable

    v6.0 ML-First Override:
      - When ml_override is provided, static nu/ka formulas are bypassed entirely
      - Only sigma, lambda, and variant offsets use the mechanistic path
    """
    # -- nu mapping (FALLBACK only in v6.0) ------------------------------------
    # v7.1 Recalibrated: k_nu_offset raised 0.15 -> 0.30 so that molecules
    # spanning pI 7.4-9.4 (delta_pH ~2 units) produce delta_nu ~0.6 in the
    # main peak — enough for Rs >1.2 between charge variants under standard
    # 30-min salt gradients on 25 cm columns.
    nu_hardcoded_base: float = 2.5     # Physical: mAb ~2-3 CEX interaction sites
    k_nu_offset:       float = 0.30    # pI-dependent offset: nu += 0.30 * |pH - pI|  [v7.1: was 0.15]
    nu_offset_max:     float = 1.5     # Max offset from pI contribution
    nu_min:            float = 2.0     # Hard lower bound
    nu_max:            float = 6.0     # Hard upper bound

    # -- ka mapping (FALLBACK only in v6.0) ------------------------------------
    # v7.3 Standard-Range Recalibration: ka in standard SMA Keq-units (1-5).
    # Rs > 1.2 is achieved through WIDE VARIANT DELTAS (nu ±0.4, ka 0.75x/1.25x)
    # rather than inflated ka values.  With ka~3, kd=1000, Keq~0.003,
    # Lambda=1200, F=1.7:
    #   c_elution = 1200 * (1.7 * 0.003)^(1/3) ≈ 206 mM → RT ≈ 17.6 min
    # Acidic variant (ka*0.75, nu-0.4) → RT ~13 min
    # Basic variant  (ka*1.25, nu+0.3) → RT ~21 min → Rs > 1.5
    ka_flat_base:      float = 3.0     # Standard SMA base ka  [v7.3: reverted from 65.0]
    alpha_h:           float = 0.8     # Hydrophobicity contribution (legacy): ka *= (1 + 0.8*h)
    beta_gravy:        float = 2.0     # GRAVY exponential coefficient: ka = base * exp(beta * GRAVY)
    ka_min:            float = 0.3     # ka lower bound  [v7.3: reverted from 5.0]
    ka_max:            float = 8.0     # ka upper bound  [v7.3: reverted from 200.0]

    # -- kd -------------------------------------------------------------------
    kd_fixed:   float = 1000.0   # Fixed kd = 1000.0 (CADET SMA standard)

    # -- sigma mapping --------------------------------------------------------
    sigma_base: float = 8.0      # Reference steric factor (reduced from 10)
    MW_ref_kDa: float = 150.0    # Reference molecular weight (mAb ~150 kDa)

    # -- lambda (column constant) ---------------------------------------------
    lambda_:    float = 1200.0   # Ionic capacity [mol/m^3_solid]

    # -- PTM offset parameters ------------------------------------------------
    # v7.1 Recalibrated: widened nu offsets for realistic charge variant
    # separation on modern CEX columns.
    #   Acidic variants (deamidation → Asp/isoAsp, net charge -1 per site)
    #   carry more negative charge → weaker cation-exchange retention →
    #   elute earlier.  Typical ΔtR ~1-3 min in 30-min gradients (Rs >1.2).
    #   Basic variants (C-term Lys, succinimide) carry more positive charge →
    #   stronger retention → elute later.
    deamidation_nu_offset:   float = -0.40   # nu shift per deam site  [v7.1: was -0.25]
    deamidation_ka_factor:   float = 0.75    # ka multiplier per deam  [v7.1: was 0.85]
    oxidation_nu_offset:     float = +0.30   # nu shift per ox site    [v7.1: was +0.15]
    oxidation_ka_factor:     float = 1.25    # ka multiplier per ox    [v7.1: was 1.15]

    sigma_acidic_factor:     float = 1.02
    sigma_basic_factor:      float = 0.98

    default_deamidation_sites: int = 1
    default_oxidation_sites:   int = 1

    c_fractions: Tuple = (0.12, 0.80, 0.08)


# ---------------------------------------------------------------------------
# Main Mapper
# ---------------------------------------------------------------------------

class PropertyMapper:
    """
    Maps protein physical properties to CADET SMA parameters.

    v6.0 ML-First Dynamic Override Architecture:
      - Primary: Accept ML-predicted ka/nu and use them directly
      - Fallback: Static thermodynamic formulas (v5.0) when no ML override

    Usage::

        mapper = PropertyMapper()

        # ML-First mode (recommended):
        params = mapper.map(protein, ml_override={"ka": 1.45, "nu": 2.65})

        # Fallback mode (no ML available):
        params = mapper.map(protein)
    """

    def __init__(self, config: Optional[MapperConfig] = None):
        self.config = config or MapperConfig()

    def map(
        self,
        protein: ProteinProperties,
        ml_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Map protein properties to SMA parameter dictionary.

        Parameters
        ----------
        protein      : Protein physical properties
        ml_override  : Optional ML-predicted overrides {"ka": float, "nu": float}.
                       When provided, these REPLACE the static nu/ka formulas.
                       sigma, kd, and lambda are still computed mechanistically.

        Returns
        -------
        dict : {"nu", "ka", "kd", "sigma", "lambda_"}
        """
        cfg = self.config

        # ==============================================================
        # nu and ka: ML OVERRIDE (primary) vs STATIC FALLBACK
        # ==============================================================
        if ml_override is not None and "ka" in ml_override and "nu" in ml_override:
            # --- ML-FIRST PATH: quality-gated ---
            ml_ka = float(ml_override["ka"])
            ml_nu = float(ml_override["nu"])

            # v7.3.2: Quality gate — reject ML if R² < threshold or
            # if the model's own validation metric (passed via ml_override)
            # indicates poor fit.
            _ml_r2 = ml_override.get("val_r2")          # optional validation R²
            _ml_rejected = False
            _ML_R2_THRESHOLD = 0.5  # model must explain >50% of variance

            if _ml_r2 is not None and _ml_r2 < _ML_R2_THRESHOLD:
                log.warning(
                    "  [ML REJECTED] R²=%.4f < threshold %.2f. "
                    "Model performs worse than baseline; reverting to static physics.",
                    _ml_r2, _ML_R2_THRESHOLD,
                )
                _ml_rejected = True

            # Sanity: reject if ka or nu are at extreme clamp boundaries
            # (indicates the model is extrapolating beyond training domain)
            if not _ml_rejected:
                if ml_ka <= cfg.ka_min * 1.01 or ml_ka >= cfg.ka_max * 0.99:
                    log.warning(
                        "  [ML REJECTED] ka=%.4f at boundary [%.2f, %.2f]. "
                        "Model is extrapolating; reverting to static physics.",
                        ml_ka, cfg.ka_min, cfg.ka_max,
                    )
                    _ml_rejected = True
                elif ml_nu <= cfg.nu_min * 1.01 or ml_nu >= cfg.nu_max * 0.99:
                    log.warning(
                        "  [ML REJECTED] nu=%.3f at boundary [%.2f, %.2f]. "
                        "Model is extrapolating; reverting to static physics.",
                        ml_nu, cfg.nu_min, cfg.nu_max,
                    )
                    _ml_rejected = True

            if not _ml_rejected:
                # Apply safety clamps (physical constraints still enforced)
                nu = float(max(cfg.nu_min, min(cfg.nu_max, ml_nu)))
                ka = float(max(cfg.ka_min, min(cfg.ka_max, ml_ka)))

                log.info("  [ML OVERRIDE] nu = %.3f (raw=%.3f), ka = %.4f (raw=%.4f)",
                         nu, ml_nu, ka, ml_ka)
                log.info("  [ML OVERRIDE] Keq = ka/kd = %.4f / %.1f = %.6f",
                         ka, cfg.kd_fixed, ka / cfg.kd_fixed)
                source = "ml_override"
            else:
                # Fall through to static path below
                ml_override = None

        if ml_override is None:
            # --- STATIC FALLBACK: v5.0 empirical formulas ---
            charge_dist = protein.net_charge_factor()
            nu_offset = min(cfg.k_nu_offset * charge_dist, cfg.nu_offset_max)
            nu_raw = cfg.nu_hardcoded_base + nu_offset
            nu = float(max(cfg.nu_min, min(cfg.nu_max, nu_raw)))

            # ka: Prefer GRAVY-based exponential (thermodynamic) when available,
            # fall back to legacy normalized-hydrophobicity formula otherwise.
            if protein.gravy_score is not None:
                # Thermodynamic model: ΔG_bind ∝ hydrophobicity, ka ∝ exp(-ΔG/RT)
                # ka = base * exp(beta * GRAVY)
                # Typical mAb GRAVY: -0.5 to 0 → ka range ~1.1 to 3.0
                gravy = protein.gravy_score
                ka_raw = cfg.ka_flat_base * math.exp(cfg.beta_gravy * gravy)
                ka = float(max(cfg.ka_min, min(cfg.ka_max, ka_raw)))
                log.info("  [STATIC FALLBACK] nu = %.2f + min(%.2f * %.2f, %.1f) = %.3f",
                         cfg.nu_hardcoded_base, cfg.k_nu_offset, charge_dist,
                         cfg.nu_offset_max, nu)
                log.info("  [STATIC FALLBACK] ka = %.2f * exp(%.2f * %.3f) = %.4f  [GRAVY-derived]",
                         cfg.ka_flat_base, cfg.beta_gravy, gravy, ka)
                source = "static_v5_gravy"
            else:
                h = protein.hydrophobicity
                ka_raw = cfg.ka_flat_base * (1.0 + cfg.alpha_h * h)
                ka = float(max(cfg.ka_min, min(cfg.ka_max, ka_raw)))
                log.info("  [STATIC FALLBACK] nu = %.2f + min(%.2f * %.2f, %.1f) = %.3f",
                         cfg.nu_hardcoded_base, cfg.k_nu_offset, charge_dist,
                         cfg.nu_offset_max, nu)
                log.info("  [STATIC FALLBACK] ka = %.2f * (1 + %.2f * %.2f) = %.4f  [legacy normalized]",
                         cfg.ka_flat_base, cfg.alpha_h, h, ka)
                source = "static_v5"

            log.info("  [STATIC FALLBACK] Keq = %.4f / %.1f = %.6f",
                     ka, cfg.kd_fixed, ka / cfg.kd_fixed)

        # ==============================================================
        # Structural parameters: always mechanistic (not overridden by ML)
        # ==============================================================
        kd = cfg.kd_fixed
        size_ratio = protein.MW_kDa / cfg.MW_ref_kDa
        sigma = cfg.sigma_base * (size_ratio ** (1.0 / 3.0))
        sigma = float(max(1.0, sigma))
        lambda_ = cfg.lambda_

        # Intermediate mapping trace (v7.3.1 — sequence→process transparency)
        _keq = ka / kd
        _mapping_trace = {
            "input_pI": round(protein.pI, 3),
            "input_MW_kDa": round(protein.MW_kDa, 2),
            "input_GRAVY": round(protein.gravy_score, 4) if protein.gravy_score is not None else None,
            "input_hydrophobicity": round(protein.hydrophobicity, 4) if protein.hydrophobicity else None,
            "derived_charge_factor": round(protein.net_charge_factor(), 4),
            "derived_Keq": round(_keq, 6),
            "mapping_source": source,
            "explanation": (
                f"pI={protein.pI:.2f} → net charge factor → nu={nu:.3f}; "
                f"{'GRAVY=' + str(round(protein.gravy_score, 3)) if protein.gravy_score is not None else 'hydro=' + str(protein.hydrophobicity)} → ka={ka:.4f}; "
                f"MW={protein.MW_kDa:.1f} → sigma={sigma:.3f}; "
                f"Keq=ka/kd={_keq:.6f}"
            ),
        }

        params = {
            "nu":      round(nu,      3),
            "ka":      round(ka,      4),
            "kd":      round(kd,      3),
            "sigma":   round(sigma,   3),
            "lambda_": round(lambda_, 1),
            "source":  source,
            "mapping_trace": _mapping_trace,
        }

        log.info("PropertyMapper [%s] -> nu=%.3f ka=%.4f kd=%.1f sigma=%.3f "
                 "lambda=%.0f Keq=%.6f (source=%s)",
                 protein.name, nu, ka, kd, sigma, lambda_, ka / kd, source)
        return params

    def _calculate_variant_offsets(
        self,
        main_nu:    float,
        main_ka:    float,
        main_sigma: float,
        ptm_profile: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate three-variant SMA parameters from PTM profile."""
        cfg = self.config

        if ptm_profile is not None:
            deam_sites = ptm_profile.get("deamidation_sites", cfg.default_deamidation_sites)
            ox_sites = ptm_profile.get("oxidation_sites", cfg.default_oxidation_sites)
        else:
            deam_sites = cfg.default_deamidation_sites
            ox_sites   = cfg.default_oxidation_sites

        nu_main  = main_nu
        ka_main  = main_ka
        sig_main = main_sigma

        # Acidic variant (deamidation -> weaker retention)
        nu_acidic  = max(cfg.nu_min, nu_main + cfg.deamidation_nu_offset * deam_sites)
        ka_acidic  = max(cfg.ka_min, ka_main * (cfg.deamidation_ka_factor ** deam_sites))
        sig_acidic = main_sigma * cfg.sigma_acidic_factor

        # Basic variant (oxidation -> stronger retention)
        nu_basic  = min(cfg.nu_max, nu_main + cfg.oxidation_nu_offset * ox_sites)
        ka_basic  = min(cfg.ka_max, ka_main * (cfg.oxidation_ka_factor ** ox_sites))
        sig_basic = main_sigma * cfg.sigma_basic_factor

        log.info("  Variants: deam=%d ox=%d", deam_sites, ox_sites)
        log.info("  Acidic: nu=%.3f ka=%.4f Keq=%.6f", nu_acidic, ka_acidic, ka_acidic/cfg.kd_fixed)
        log.info("  Main:   nu=%.3f ka=%.4f Keq=%.6f", nu_main, ka_main, ka_main/cfg.kd_fixed)
        log.info("  Basic:  nu=%.3f ka=%.4f Keq=%.6f", nu_basic, ka_basic, ka_basic/cfg.kd_fixed)

        return {
            "acidic": {"ka": round(ka_acidic, 6), "nu": round(nu_acidic, 3), "sigma": round(sig_acidic, 3)},
            "main":   {"ka": round(ka_main, 6),   "nu": round(nu_main, 3),   "sigma": round(sig_main, 3)},
            "basic":  {"ka": round(ka_basic, 6),   "nu": round(nu_basic, 3),  "sigma": round(sig_basic, 3)},
        }

    def map_variants(
        self,
        protein: ProteinProperties,
        ml_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Map protein + PTM to multi-component SMA dict.

        Parameters
        ----------
        protein      : Protein physical properties
        ml_override  : Optional ML-predicted overrides {"ka": float, "nu": float}.
                       These override the static baseline for the MAIN variant;
                       acidic/basic offsets are still applied mechanistically.

        Returns
        -------
        dict : Three-variant parameters with kd, lambda, fractions, and source info
        """
        cfg = self.config
        base_params = self.map(protein, ml_override=ml_override)

        variants = self._calculate_variant_offsets(
            main_nu    = base_params["nu"],
            main_ka    = base_params["ka"],
            main_sigma = base_params["sigma"],
            ptm_profile = protein.ptm_profile,
        )

        source = base_params.get("source", "static_v5")

        return {
            "acidic":      variants["acidic"],
            "main":        variants["main"],
            "basic":       variants["basic"],
            "kd":          cfg.kd_fixed,
            "lambda_":     cfg.lambda_,
            "c_fractions": cfg.c_fractions,
            "source":      source,
        }

    def map_to_variant_params(self, protein: ProteinProperties, ml_override=None):
        """Return a VariantParams object for CadetEngine."""
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from cadet_engine import VariantParams
        except ImportError as e:
            raise ImportError(f"Cannot import VariantParams: {e}")
        return VariantParams.from_dict(self.map_variants(protein, ml_override=ml_override))

    def map_to_protein_params(self, protein: ProteinProperties, ml_override=None):
        """Return a ProteinParams object for CadetEngine (legacy)."""
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from cadet_engine import ProteinParams
        except ImportError as e:
            raise ImportError(f"Cannot import ProteinParams: {e}")
        d = self.map(protein, ml_override=ml_override)
        return ProteinParams(ka=d["ka"], kd=d["kd"], lambda_=d["lambda_"], nu=d["nu"], sigma=d["sigma"])

    # -- v7.0: Bispecific Species Mapping ------------------------------------

    def map_bispecific_species(
        self,
        species_dict: Dict[str, Dict[str, Any]],
        ml_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Map bispecific assembly species (AA, AB, BB) to SMA parameters.

        Each species is mapped independently through the standard map() method.
        ml_override applies only to the target AB species.

        Parameters
        ----------
        species_dict : Dict with keys "AA", "AB", "BB", each containing:
            - pI: float
            - mw_kda: float
            - hydrophobicity: float
            - deam_sites: int (optional)
            - ox_sites: int (optional)
            - display_name: str (optional)
            - sequence: str (optional)
        ml_override : Optional ML override {ka, nu} for AB species only

        Returns
        -------
        dict : {
            "AA": {nu, ka, kd, sigma, lambda_, source},
            "AB": {nu, ka, kd, sigma, lambda_, source},
            "BB": {nu, ka, kd, sigma, lambda_, source},
        }
        """
        result = {}
        for key in ("AA", "AB", "BB"):
            sp = species_dict.get(key, {})
            protein = ProteinProperties(
                name=sp.get("display_name", key),
                pI=sp.get("pI", 8.0),
                MW_kDa=sp.get("mw_kda", 150.0),
                hydrophobicity=sp.get("hydrophobicity", 0.35),
                pH_working=sp.get("pH_working", 7.0),
                sequence=sp.get("sequence") if len(sp.get("sequence", "")) < 5000 else None,
                ptm_profile={
                    "deamidation_sites": sp.get("deam_sites", 1),
                    "oxidation_sites": sp.get("ox_sites", 1),
                },
            )
            # ML override only for target species (AB)
            override = ml_override if key == "AB" else None
            params = self.map(protein, ml_override=override)
            result[key] = params

        log.info("Bispecific species mapped: AA(nu=%.3f) AB(nu=%.3f) BB(nu=%.3f)",
                 result["AA"]["nu"], result["AB"]["nu"], result["BB"]["nu"])
        return result

    def sensitivity_analysis(self, protein: ProteinProperties, param: str, delta: float = 0.10) -> Dict[str, float]:
        """Single-parameter sensitivity analysis."""
        base = self.map(protein)
        p_plus  = copy.copy(protein)
        p_minus = copy.copy(protein)
        val = getattr(protein, param)
        setattr(p_plus, param, val * (1 + delta))
        setattr(p_minus, param, val * (1 - delta))
        for p in (p_plus, p_minus):
            try: p.__post_init__()
            except ValueError: pass
        return {"param": param, "delta": delta, "base": base, "plus": self.map(p_plus), "minus": self.map(p_minus)}

    def explain(
        self,
        protein: ProteinProperties,
        ml_override: Optional[Dict[str, float]] = None,
    ) -> str:
        """Return a human-readable explanation of the mapping."""
        cfg = self.config
        d   = self.map(protein, ml_override=ml_override)
        charge_dist = protein.net_charge_factor()
        nu_offset = min(cfg.k_nu_offset * charge_dist, cfg.nu_offset_max)
        vd = self.map_variants(protein, ml_override=ml_override)
        source = d.get("source", "static_v5")

        lines = [
            "=" * 60,
            f"  PropertyMapper v7.0 Report — {protein.name}",
            "=" * 60,
            f"  Input: pI={protein.pI:.2f}  MW={protein.MW_kDa:.1f} kDa  "
            f"h={protein.hydrophobicity:.3f}  pH={protein.pH_working:.2f}",
            f"  |pH - pI| = {charge_dist:.2f}",
            f"  Source: {source.upper()}",
        ]
        if protein.sequence:
            lines.append(f"  Sequence: {len(protein.sequence)} aa")
        if protein.ptm_profile:
            lines.append(f"  PTM: {protein.ptm_profile}")

        if source == "ml_override" and ml_override:
            lines += [
                f"",
                f"  v6.0 ML-First Override (Neural Network Steering):",
                f"    ML predicted: ka={ml_override['ka']:.4f}, nu={ml_override['nu']:.3f}",
                f"    After clamp:  ka={d['ka']:.4f}, nu={d['nu']:.3f}",
                f"    Keq = {d['ka']:.4f}/{d['kd']:.0f} = {d['ka']/d['kd']:.6f}",
                f"    sigma = {d['sigma']:.3f}  lambda = {d['lambda_']:.0f}",
                f"",
                f"    Static fallback would have been:",
                f"      nu = {cfg.nu_hardcoded_base} + min({cfg.k_nu_offset}*{charge_dist:.2f}, "
                f"{cfg.nu_offset_max}) = {cfg.nu_hardcoded_base + min(cfg.k_nu_offset * charge_dist, cfg.nu_offset_max):.3f}",
                f"      ka = {cfg.ka_flat_base} * (1 + {cfg.alpha_h}*{protein.hydrophobicity:.2f}) = "
                f"{cfg.ka_flat_base * (1 + cfg.alpha_h * protein.hydrophobicity):.4f}",
            ]
        else:
            lines += [
                f"",
                f"  v5.0 Static Fallback Mapping:",
                f"    nu = {cfg.nu_hardcoded_base} + min({cfg.k_nu_offset}*{charge_dist:.2f}, {cfg.nu_offset_max}) = {d['nu']:.3f}",
            ]
            if protein.gravy_score is not None:
                lines.append(
                    f"    ka = {cfg.ka_flat_base} * exp({cfg.beta_gravy}*{protein.gravy_score:.3f}) = "
                    f"{d['ka']:.4f}  [GRAVY-derived, thermodynamic]"
                )
            else:
                lines.append(
                    f"    ka = {cfg.ka_flat_base} * (1 + {cfg.alpha_h}*{protein.hydrophobicity:.2f}) = "
                    f"{d['ka']:.4f}  [legacy normalized h]"
                )
            lines += [
                f"    Keq = {d['ka']:.4f}/{d['kd']:.0f} = {d['ka']/d['kd']:.6f}",
                f"    sigma = {d['sigma']:.3f}  lambda = {d['lambda_']:.0f}",
            ]

        lines += [
            f"",
            f"  Three-Variant Parameters:",
        ]
        for var_name in ("acidic", "main", "basic"):
            v = vd[var_name]
            lines.append(f"    {var_name:7s}: nu={v['nu']:.3f}  ka={v['ka']:.6f}  "
                         f"sigma={v['sigma']:.3f}  Keq={v['ka']/cfg.kd_fixed:.6f}")
        lines.append(f"    c_fractions = {vd['c_fractions']}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    mapper = PropertyMapper()

    print("=" * 60)
    print("  PropertyMapper v7.0 — ML-First + Bispecific + GRAVY-ka")
    print("=" * 60)

    # -- Static fallback test (legacy, no GRAVY) --
    print("\n--- Static Fallback Mode (legacy, no GRAVY) ---")
    test_proteins = [
        ("Standard mAb (pI 8.5, h=0.35, pH=7.0)", 8.5, 0.35, 7.0),
        ("Low pI mAb (pI 7.4, h=0.35, pH=7.0)",   7.4, 0.35, 7.0),
        ("High pI mAb (pI 9.0, h=0.35, pH=7.0)",  9.0, 0.35, 7.0),
        ("Hydrophobic (pI 8.5, h=0.60, pH=7.0)",   8.5, 0.60, 7.0),
        ("Hydrophilic (pI 8.5, h=0.15, pH=7.0)",   8.5, 0.15, 7.0),
    ]
    for label, pi, h, ph in test_proteins:
        p = ProteinProperties(name=label, pI=pi, MW_kDa=150.0, hydrophobicity=h, pH_working=ph)
        d = mapper.map(p)
        print(f"  {label:45s}: nu={d['nu']:.3f}  ka={d['ka']:.4f}  Keq={d['ka']/d['kd']:.6f}  src={d['source']}")

    # -- GRAVY-derived ka test (thermodynamic model) --
    print("\n--- GRAVY-Derived ka (Thermodynamic Model) ---")
    gravy_test_molecules = [
        ("adalimumab",   8.72, 150.0, -0.471, 7.0),
        ("trastuzumab",  8.45, 148.0, -0.415, 7.0),
        ("bevacizumab",  8.26, 149.0, -0.500, 7.0),
        ("rituximab",    9.04, 147.0, -0.380, 7.0),
        ("NISTmAb",      8.31, 148.0, -0.320, 7.0),
        ("Hydrophobic_mAb", 8.50, 150.0, +0.05, 7.0),
    ]
    ka_values = []
    for label, pi, mw, gravy, ph in gravy_test_molecules:
        h_norm = max(0.0, min(1.0, (gravy + 2.0) / 4.0))
        p = ProteinProperties(name=label, pI=pi, MW_kDa=mw,
                              hydrophobicity=h_norm, pH_working=ph,
                              gravy_score=gravy)
        d = mapper.map(p)
        ka_values.append(d['ka'])
        print(f"  {label:20s} GRAVY={gravy:+.3f}  →  ka={d['ka']:.4f}  nu={d['nu']:.3f}  src={d['source']}")

    # Verify GRAVY-derived ka values are distinct
    assert len(set(round(k, 4) for k in ka_values)) == len(ka_values), \
        f"GRAVY-derived ka values should all be distinct, got: {ka_values}"
    # Verify ordering: more hydrophobic (higher GRAVY) → higher ka
    gravy_ka_pairs = list(zip([g for _, _, _, g, _ in gravy_test_molecules], ka_values))
    for i in range(len(gravy_ka_pairs)):
        for j in range(i + 1, len(gravy_ka_pairs)):
            if gravy_ka_pairs[i][0] < gravy_ka_pairs[j][0]:
                assert gravy_ka_pairs[i][1] < gravy_ka_pairs[j][1], \
                    f"Higher GRAVY should give higher ka: {gravy_ka_pairs[i]} vs {gravy_ka_pairs[j]}"
    print("  ✓ All GRAVY-derived ka values are distinct and monotonically ordered")

    # -- ML Override test --
    print("\n--- ML-First Override Mode ---")
    mab = ProteinProperties(name="mAb_ML_Test", pI=8.45, MW_kDa=148.0, hydrophobicity=0.35,
                            pH_working=7.0, ptm_profile={"deamidation_sites": 1, "oxidation_sites": 1})
    ml_pred = {"ka": 1.42, "nu": 2.58}
    d_ml = mapper.map(mab, ml_override=ml_pred)
    d_fb = mapper.map(mab)
    print(f"  ML override:    nu={d_ml['nu']:.3f}  ka={d_ml['ka']:.4f}  src={d_ml['source']}")
    print(f"  Static fallback: nu={d_fb['nu']:.3f}  ka={d_fb['ka']:.4f}  src={d_fb['source']}")
    print(f"  Delta:          nu={d_ml['nu'] - d_fb['nu']:+.3f}  ka={d_ml['ka'] - d_fb['ka']:+.4f}")

    # -- Variant test with ML override --
    vp_ml = mapper.map_variants(mab, ml_override=ml_pred)
    vp_fb = mapper.map_variants(mab)
    print(f"\n  Variants (ML override):")
    for vn in ("acidic", "main", "basic"):
        print(f"    {vn:7s}: nu={vp_ml[vn]['nu']:.3f}  ka={vp_ml[vn]['ka']:.6f}")
    print(f"  Variants (static fallback):")
    for vn in ("acidic", "main", "basic"):
        print(f"    {vn:7s}: nu={vp_fb[vn]['nu']:.3f}  ka={vp_fb[vn]['ka']:.6f}")

    print(f"\n  Source (ML): {vp_ml['source']}")
    print(f"  Source (FB): {vp_fb['source']}")

    # Full report
    print("\n" + mapper.explain(mab, ml_override=ml_pred))
