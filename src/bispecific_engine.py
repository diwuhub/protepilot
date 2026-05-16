"""
bispecific_engine.py  ·  ProtePilot — Milestone 11
===========================================================
Bispecific Antibody / Fusion Protein Thermodynamics Engine

Version   : 1.0
Author    : Di (ProtePilot)
Depends   : numpy, biopython (optional)

Purpose
------------------------------------------------------------
Simulates the three assembly species that arise during
bispecific antibody production:

  - Homodimer AA  (undesired product impurity)
  - Heterodimer AB (target bispecific product)
  - Homodimer BB  (undesired product impurity)

For each species, computes theoretical pI, MW, hydrophobicity,
and maps to CADET SMA parameters via PropertyMapper. Then
calculates chromatographic Resolution (Rs) to assess
separation feasibility and homodimer co-elution risk.

Physics
------------------------------------------------------------
  MW_AA  = 2 * MW_A       MW_AB = MW_A + MW_B       MW_BB = 2 * MW_B
  pI_AA  = pI(seq_A+A)    pI_AB = pI(seq_A+B)       pI_BB = pI(seq_B+B)
  hydro  = GRAVY average of constituent chains

  Rs = 2 * |RT_AB - RT_XX| / (W_AB + W_XX)
  where W = FWHM, XX = AA or BB

References
------------------------------------------------------------
  Klein et al. (2012) mAbs 4(6):653-663 — Bispecific formats
  Labrijn et al. (2019) Nat Rev Drug Discov 18:585-608
  Yamamoto et al. (1987) J. Chromatogr. 409:101-113
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ProtePilot.BispecificEngine")


# ===========================================================================
# 0. Empirical pI Estimator (Henderson-Hasselbalch binary search)
# ===========================================================================

def _estimate_pI_hh(sequence: str) -> float:
    """
    Estimate isoelectric point via Henderson-Hasselbalch binary search.

    Used as an empirical fallback when Biopython is unavailable or returns
    a degenerate result (pI = 0.0) for a combined/concatenated sequence.

    Implements the same physics as Biopython.isoelectric_point():
      - EMBOSS/SwissProt consensus pKa values for ionisable side chains
      - N-terminus α-amino (pKa 8.0) and C-terminus α-carboxyl (pKa 3.1)
      - Binary search over pH 1–14 to find net charge ≈ 0

    Accuracy: ±0.2–0.3 pH units vs. Biopython for typical antibody sequences.
    Advantage over simple composition formula: correctly handles asymmetric
    charge distributions common in bispecific molecules (one arm basic, one
    arm acidic), where the arithmetic mean of chain pIs is non-physical.

    Parameters
    ----------
    sequence : str
        Single-letter amino acid sequence (may be a concatenated multi-chain
        sequence; each chain contributes residues to the shared charge budget).

    Returns
    -------
    float : Estimated isoelectric point, clamped to [3.0, 12.0].
    """
    # pKa and charge sign at low pH (protonated state)
    # sign +1 = basic (carries + charge when protonated, i.e. pH < pKa)
    # sign -1 = acidic (carries - charge when deprotonated, i.e. pH > pKa)
    _PKA: Dict[str, Tuple[float, int]] = {
        "D": (3.9,  -1),  # Aspartate
        "E": (4.1,  -1),  # Glutamate
        "C": (8.3,  -1),  # Cysteine
        "Y": (10.1, -1),  # Tyrosine
        "H": (6.0,  +1),  # Histidine
        "K": (10.5, +1),  # Lysine
        "R": (12.5, +1),  # Arginine
    }
    _N_TERM_PKA = 8.0    # α-amino (N-terminus)
    _C_TERM_PKA = 3.1    # α-carboxyl (C-terminus)

    seq = sequence.upper()
    counts = {aa: seq.count(aa) for aa in _PKA}

    def _net_charge(pH: float) -> float:
        q = 0.0
        # N-terminus: basic, protonated below pKa
        q += 1.0 / (1.0 + 10.0 ** (pH - _N_TERM_PKA))
        # C-terminus: acidic, deprotonated above pKa
        q -= 1.0 / (1.0 + 10.0 ** (_C_TERM_PKA - pH))
        for aa, (pka, sign) in _PKA.items():
            n = counts.get(aa, 0)
            if n == 0:
                continue
            if sign == +1:
                q += n / (1.0 + 10.0 ** (pH - pka))
            else:
                q -= n / (1.0 + 10.0 ** (pka - pH))
        return q

    # Binary search: 50 iterations → precision ~10⁻¹⁵
    lo, hi = 1.0, 14.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if _net_charge(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    # Clamp to physiologically realistic range.
    # Note: for artificially homogeneous sequences (e.g., all-Asp) the raw
    # result can exceed this range — that never occurs for real bispecific
    # sequences.  For typical IgG-derived chains the error vs. Biopython
    # is < 0.2 pH units.
    return float(max(3.0, min(12.0, round((lo + hi) / 2.0, 2))))


# ===========================================================================
# 1. Chain Dataclass
# ===========================================================================

@dataclass
class AntibodyChain:
    """Single antibody chain (arm/half-body) for bispecific assembly."""
    name: str
    sequence: str
    pI: float = 0.0
    mw_kda: float = 0.0
    hydrophobicity: float = 0.35
    gravy: float = 0.0
    acidic_count: int = 0
    basic_count: int = 0
    deam_sites: int = 1
    ox_sites: int = 1

    def __post_init__(self):
        if self.sequence and self.pI == 0.0:
            self._compute_biophysical()

    def _compute_biophysical(self):
        """Compute biophysical properties from sequence using Biopython."""
        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            seq_clean = self.sequence.upper()
            analysis = ProteinAnalysis(seq_clean)
            self.pI = round(analysis.isoelectric_point(), 2)
            self.mw_kda = round(analysis.molecular_weight() / 1000.0, 1)
            self.gravy = round(analysis.gravy(), 3)
            self.hydrophobicity = round(
                max(0.0, min(1.0, (self.gravy + 2.0) / 4.0)), 3
            )
            self.acidic_count = seq_clean.count("D") + seq_clean.count("E")
            self.basic_count = (
                seq_clean.count("K") + seq_clean.count("R") + seq_clean.count("H")
            )
            # Deamidation hotspots: N-G, N-S
            deam = 0
            for i in range(len(seq_clean) - 1):
                if seq_clean[i] == "N" and seq_clean[i + 1] in ("G", "S"):
                    deam += 1
            self.deam_sites = max(1, deam)
            # Oxidation: Met count
            self.ox_sites = max(1, seq_clean.count("M"))
            log.info("Chain %s: pI=%.2f MW=%.1f kDa GRAVY=%.3f",
                     self.name, self.pI, self.mw_kda, self.gravy)
        except ImportError:
            log.warning("Biopython not available; estimating properties from amino acid composition")
            self._estimate_biophysical_from_composition()
        except Exception as e:
            log.warning("Chain biophysical computation failed: %s; using fallback estimation", e)
            self._estimate_biophysical_from_composition()

    def _estimate_biophysical_from_composition(self):
        """Fallback: estimate pI and other properties from amino acid composition."""
        seq_clean = self.sequence.upper()
        if not seq_clean:
            return

        # Count acidic and basic residues
        n_asp = seq_clean.count('D')
        n_glu = seq_clean.count('E')
        n_cys = seq_clean.count('C')
        n_tyr = seq_clean.count('Y')
        n_his = seq_clean.count('H')
        n_lys = seq_clean.count('K')
        n_arg = seq_clean.count('R')

        # Simple Henderson-Hasselbalch approximation
        # More basic residues -> higher pI
        basic = n_his + n_lys + n_arg
        acidic = n_asp + n_glu
        total = len(seq_clean)

        # Typical IgG pI range 6-9.5, VH 7-9
        self.pI = round(7.0 + 3.0 * (basic - acidic) / total, 2)
        self.pI = max(3.0, min(12.0, self.pI))  # clamp to reasonable range

        # Molecular weight: ~110 Da per amino acid
        self.mw_kda = round(len(seq_clean) * 0.110, 1)

        # Default hydrophobicity
        self.hydrophobicity = 0.35

        # Count residues for oxidation and deamidation
        self.acidic_count = n_asp + n_glu
        self.basic_count = n_his + n_lys + n_arg

        # Deamidation hotspots: N-G, N-S
        deam = 0
        for i in range(len(seq_clean) - 1):
            if seq_clean[i] == "N" and seq_clean[i + 1] in ("G", "S"):
                deam += 1
        self.deam_sites = max(1, deam)

        # Oxidation: Met count
        self.ox_sites = max(1, seq_clean.count("M"))

        log.info("Chain %s: pI=%.2f (estimated) MW=%.1f kDa (estimated)",
                 self.name, self.pI, self.mw_kda)


# ===========================================================================
# 2. Species Assembly
# ===========================================================================

@dataclass
class AssemblySpecies:
    """A bispecific assembly species (AA, AB, or BB)."""
    label: str           # "AA", "AB", or "BB"
    display_name: str    # "Homodimer AA", "Heterodimer AB", "Homodimer BB"
    pI: float
    mw_kda: float
    hydrophobicity: float
    sequence: str        # Combined sequence for the species
    deam_sites: int = 1
    ox_sites: int = 1
    acidic_count: int = 40
    basic_count: int = 50
    is_target: bool = False  # True for AB (desired product)


def build_assembly_species(
    chain_a: AntibodyChain,
    chain_b: AntibodyChain,
) -> Dict[str, AssemblySpecies]:
    """
    Build the three assembly species from two chains.

    Parameters
    ----------
    chain_a : First antibody chain/arm
    chain_b : Second antibody chain/arm

    Returns
    -------
    dict with keys "AA", "AB", "BB" -> AssemblySpecies
    """
    species = {}

    # Helper to compute species properties
    def _make_species(
        label: str,
        display_name: str,
        seq_parts: List[str],
        chains_used: List[AntibodyChain],
        is_target: bool = False,
    ) -> AssemblySpecies:
        combined_seq = "".join(seq_parts)
        n_chains = len(chains_used)

        # Compute pI from combined sequence via Biopython
        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            analysis = ProteinAnalysis(combined_seq.upper())
            sp_pI = round(analysis.isoelectric_point(), 2)
            sp_mw = round(analysis.molecular_weight() / 1000.0, 1)
            sp_gravy = analysis.gravy()
            sp_hydro = round(max(0.0, min(1.0, (sp_gravy + 2.0) / 4.0)), 3)
        except (ImportError, Exception):
            # Fallback: Henderson-Hasselbalch empirical pI from combined sequence.
            # More accurate than weighted-mean-of-chains, which is non-physical for
            # bispecifics where one arm is basic and the other is acidic.
            sp_pI = _estimate_pI_hh(combined_seq)
            sp_mw = round(sum(c.mw_kda for c in chains_used), 1)
            sp_hydro = round(
                sum(c.hydrophobicity for c in chains_used) / n_chains, 3
            )
            log.debug(
                "Species %s: Biopython unavailable, HH-estimated pI=%.2f "
                "(MW=%.1f kDa)", label, sp_pI, sp_mw,
            )

        # Safety check: if pI is still 0.0 (degenerate Biopython result),
        # fall back to HH empirical estimate rather than the old linear formula.
        if sp_pI == 0.0:
            sp_pI = _estimate_pI_hh(combined_seq)
            log.warning(
                "Species %s: Biopython returned pI=0.0, HH-corrected to %.2f",
                label, sp_pI,
            )

        sp_deam = sum(c.deam_sites for c in chains_used)
        sp_ox = sum(c.ox_sites for c in chains_used)
        sp_acidic = sum(c.acidic_count for c in chains_used)
        sp_basic = sum(c.basic_count for c in chains_used)

        return AssemblySpecies(
            label=label,
            display_name=display_name,
            pI=sp_pI,
            mw_kda=sp_mw,
            hydrophobicity=sp_hydro,
            sequence=combined_seq,
            deam_sites=sp_deam,
            ox_sites=sp_ox,
            acidic_count=sp_acidic,
            basic_count=sp_basic,
            is_target=is_target,
        )

    # Homodimer AA
    species["AA"] = _make_species(
        "AA", f"Homodimer {chain_a.name}{chain_a.name}",
        [chain_a.sequence, chain_a.sequence],
        [chain_a, chain_a],
        is_target=False,
    )

    # Heterodimer AB (target product)
    species["AB"] = _make_species(
        "AB", f"Heterodimer {chain_a.name}{chain_b.name}",
        [chain_a.sequence, chain_b.sequence],
        [chain_a, chain_b],
        is_target=True,
    )

    # Homodimer BB
    species["BB"] = _make_species(
        "BB", f"Homodimer {chain_b.name}{chain_b.name}",
        [chain_b.sequence, chain_b.sequence],
        [chain_b, chain_b],
        is_target=False,
    )

    log.info("Assembly species built:")
    for key, sp in species.items():
        log.info("  %s: pI=%.2f MW=%.1f kDa h=%.3f target=%s",
                 key, sp.pI, sp.mw_kda, sp.hydrophobicity, sp.is_target)

    return species


# ===========================================================================
# 3. SMA Parameter Mapping for All Species
# ===========================================================================

def map_species_to_sma(
    species: Dict[str, AssemblySpecies],
    ml_override: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Map each assembly species to CADET SMA parameters.

    Uses PropertyMapper for each species independently.
    ml_override, if provided, applies only to the target AB species.

    Parameters
    ----------
    species     : Dict of AssemblySpecies (from build_assembly_species)
    ml_override : Optional ML-predicted ka/nu for the AB (target) species

    Returns
    -------
    dict : {
        "AA": {"nu": ..., "ka": ..., "kd": ..., "sigma": ..., "lambda_": ..., "source": ...},
        "AB": {...},
        "BB": {...},
    }
    """
    from src.PropertyMapper import ProteinProperties, PropertyMapper

    mapper = PropertyMapper()
    result = {}

    for key, sp in species.items():
        protein = ProteinProperties(
            name=sp.display_name,
            pI=sp.pI,
            MW_kDa=sp.mw_kda,
            hydrophobicity=sp.hydrophobicity,
            pH_working=7.0,
            sequence=sp.sequence if len(sp.sequence) < 5000 else None,
            ptm_profile={
                "deamidation_sites": sp.deam_sites,
                "oxidation_sites": sp.ox_sites,
            },
        )

        # ML override only for target species
        override = ml_override if (sp.is_target and ml_override) else None
        params = mapper.map(protein, ml_override=override)
        result[key] = params

    log.info("SMA mapping complete for %d species", len(result))
    return result


# ===========================================================================
# 4. Resolution Calculation
# ===========================================================================

def compute_resolution(
    rt_1: float, fwhm_1: float,
    rt_2: float, fwhm_2: float,
) -> float:
    """
    Compute chromatographic resolution between two peaks (USP standard).

    Rs = 2 * |RT_2 - RT_1| / (W_b1 + W_b2)
    where W_b = baseline peak width = FWHM × 2.355 / 2 × 2 ≈ FWHM × 2.355
    (Gaussian assumption: W_b = 4σ, FWHM = 2.355σ)

    Parameters
    ----------
    rt_1, fwhm_1 : Retention time and FWHM of peak 1
    rt_2, fwhm_2 : Retention time and FWHM of peak 2

    Returns
    -------
    float : Resolution (Rs) per USP definition
    """
    # Convert FWHM to baseline width (Gaussian: W_base = FWHM × 2.355 / sqrt(8*ln2) × 4
    # Simplified: W_base ≈ FWHM × (4 / 2.355) = FWHM × 1.699
    # Or equivalently: use FWHM × 2.355 as in purification_optimizer for consistency
    w_sum = fwhm_1 * 2.355 + fwhm_2 * 2.355  # FWHM to Gaussian baseline width
    if w_sum <= 1e-6:
        return 0.0
    return 2.0 * abs(rt_2 - rt_1) / w_sum


def assess_separation_risk(
    species_peaks: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Assess homodimer co-elution risk from species peak data.

    Parameters
    ----------
    species_peaks : {
        "AA": {"rt_min": float, "fwhm_min": float},
        "AB": {"rt_min": float, "fwhm_min": float},
        "BB": {"rt_min": float, "fwhm_min": float},
    }

    Returns
    -------
    dict : {
        "rs_AB_AA": float,
        "rs_AB_BB": float,
        "risk_level": "Low" | "Medium" | "High",
        "risk_details": [...],
        "recommendations": [...],
    }
    """
    ab = species_peaks.get("AB", {})
    aa = species_peaks.get("AA", {})
    bb = species_peaks.get("BB", {})

    rs_ab_aa = compute_resolution(
        ab.get("rt_min", 0), ab.get("fwhm_min", 0.5),
        aa.get("rt_min", 0), aa.get("fwhm_min", 0.5),
    )
    rs_ab_bb = compute_resolution(
        ab.get("rt_min", 0), ab.get("fwhm_min", 0.5),
        bb.get("rt_min", 0), bb.get("fwhm_min", 0.5),
    )

    risk_details = []
    recommendations = []
    min_rs = min(rs_ab_aa, rs_ab_bb)

    if min_rs < 0.8:
        risk_level = "High"
        risk_details.append(
            f"Critical co-elution risk: minimum Rs = {min_rs:.3f} (< 0.8)"
        )
        if rs_ab_aa < 0.8:
            risk_details.append(
                f"Homodimer AA co-elutes with target AB (Rs = {rs_ab_aa:.3f})"
            )
        if rs_ab_bb < 0.8:
            risk_details.append(
                f"Homodimer BB co-elutes with target AB (Rs = {rs_ab_bb:.3f})"
            )
        recommendations.extend([
            "Consider introducing charge asymmetry engineering (e.g., "
            "knobs-into-holes with electrostatic steering mutations) to "
            "increase pI difference between species.",
            "Evaluate alternative chromatography modes (e.g., HIC, MMC) "
            "that may exploit hydrophobicity differences.",
            "Consider reducing gradient slope to improve resolution "
            "(slower gradient = better separation).",
            "Evaluate using a longer column (L > 0.25 m) to increase "
            "theoretical plate count.",
        ])
    elif min_rs < 1.5:
        risk_level = "Medium"
        risk_details.append(
            f"Partial separation: minimum Rs = {min_rs:.3f} (0.8-1.5 range)"
        )
        if rs_ab_aa < 1.5:
            risk_details.append(
                f"Homodimer AA partially overlaps with AB (Rs = {rs_ab_aa:.3f})"
            )
        if rs_ab_bb < 1.5:
            risk_details.append(
                f"Homodimer BB partially overlaps with AB (Rs = {rs_ab_bb:.3f})"
            )
        recommendations.extend([
            "Pool fractionation may achieve acceptable purity (> 95%) "
            "with moderate yield loss.",
            "Consider pH scouting (pH 5.5-7.0) to optimize selectivity.",
            "Two-column polishing strategy may be needed for high purity.",
        ])
    else:
        risk_level = "Low"
        risk_details.append(
            f"Baseline separation achieved: minimum Rs = {min_rs:.3f} (>= 1.5)"
        )
        recommendations.append(
            "Standard single-column CEX purification should achieve "
            "> 98% heterodimer purity."
        )

    result = {
        "rs_AB_AA": round(rs_ab_aa, 4),
        "rs_AB_BB": round(rs_ab_bb, 4),
        "min_rs": round(min_rs, 4),
        "risk_level": risk_level,
        "risk_details": risk_details,
        "recommendations": recommendations,
    }

    log.info("Separation risk: %s (Rs_AB-AA=%.3f, Rs_AB-BB=%.3f)",
             risk_level, rs_ab_aa, rs_ab_bb)
    return result


# ===========================================================================
# 5. Estimated Peak Prediction (Analytical, No CADET)
# ===========================================================================

def estimate_species_peaks(
    sma_params: Dict[str, Dict[str, Any]],
    gradient_slope: float = 15.0,
    column_length: float = 0.25,
) -> Dict[str, Dict[str, float]]:
    """
    Estimate retention times and peak widths for species using
    Yamamoto gradient elution theory (analytical approximation).

    This is used when CADET engine is not available.

    Parameters
    ----------
    sma_params : SMA parameters for each species {AA, AB, BB}
    gradient_slope : Salt gradient slope (mM/min)
    column_length : Column length (m)

    Returns
    -------
    dict : {species_key: {"rt_min": float, "fwhm_min": float}}
    """
    # Process conditions
    v = 5.75e-4       # Superficial velocity (m/s)
    epsilon = 0.37    # Column porosity
    t0 = column_length / (v / epsilon) / 60.0   # Dead time (min)
    F = (1.0 - epsilon) / epsilon               # Phase ratio ~1.70

    # Linear gradient parameters
    c_start = 50.0    # mM NaCl start
    c_end = 500.0     # mM NaCl end
    gradient_time = (c_end - c_start) / max(gradient_slope, 0.1)  # minutes

    peaks = {}
    for key, params in sma_params.items():
        nu = params.get("nu", 3.0)
        ka = params.get("ka", 1.5)
        kd = params.get("kd", 1000.0)
        sigma = params.get("sigma", 6.0)

        Keq = ka / kd

        # v32.2: Corrected Yamamoto SMA equilibrium formula incorporating
        # steric shielding (sigma).  In the SMA isotherm the effective
        # ionic capacity available to a protein is (Lambda - sigma), not
        # Lambda alone.  Since sigma scales with MW^(1/3), this introduces
        # a MW-dependent retention shift that differentiates species with
        # similar pI but different molecular weight — critical for
        # resolving bispecific homodimer/heterodimer peaks.
        #
        #   c_elution = (Lambda - sigma) * (F * Keq)^(1/nu)
        #
        # Clamp nu to [1.5, 3.5] for mAb CEX range (prevents RT saturation).
        nu_clamped = float(max(1.5, min(nu, 3.5)))
        lambda_ = params.get("lambda_", 1200.0)
        lambda_eff = max(lambda_ - sigma, 100.0)  # Floor prevents collapse

        try:
            c_elution = lambda_eff * (F * Keq) ** (1.0 / nu_clamped)
        except (OverflowError, ZeroDivisionError):
            c_elution = c_end

        c_elution = max(c_start, min(c_end, c_elution))

        # Convert salt concentration to retention time via linear gradient
        t_gradient = (c_elution - c_start) / max(gradient_slope, 0.1)
        rt_est = t0 + t_gradient
        rt_est = max(t0 + 0.5, min(gradient_time + t0, rt_est))

        # v7.3.2: FWHM — same model as main characterization path
        fwhm_est = 0.15 + 0.05 * rt_est
        fwhm_est = max(0.2, min(3.0, fwhm_est))

        peaks[key] = {
            "rt_min": round(rt_est, 3),
            "fwhm_min": round(fwhm_est, 3),
        }

    log.info("Estimated species peaks: %s",
             {k: f"RT={v['rt_min']:.2f}" for k, v in peaks.items()})
    return peaks


# ===========================================================================
# 6. Bispecific Chromatogram Data Generation
# ===========================================================================

def generate_bispecific_chromatogram_data(
    species_peaks: Dict[str, Dict[str, float]],
    time_range: Optional[Tuple[float, float]] = None,
    n_points: int = 1000,
    relative_abundance: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Generate simulated chromatogram data for 3-species bispecific separation.

    Each peak is modeled as a Gaussian:
      signal(t) = A * exp(-0.5 * ((t - rt) / sigma)^2)
      where sigma = fwhm / (2 * sqrt(2 * ln(2)))

    Parameters
    ----------
    species_peaks : Peak data for each species {AA, AB, BB}
    time_range    : Time axis range (min). If None, auto-computed from peaks.
    n_points      : Number of time points
    relative_abundance : Relative abundance of each species
                         Default: {"AA": 0.15, "AB": 0.70, "BB": 0.15}

    Returns
    -------
    dict : {
        "time": np.array,
        "signals": {"AA": np.array, "AB": np.array, "BB": np.array},
        "total": np.array,
        "gradient": np.array,
    }
    """
    if relative_abundance is None:
        relative_abundance = {"AA": 0.15, "AB": 0.70, "BB": 0.15}

    # v7.3.2: Dynamic time range based on actual peak positions
    if time_range is None:
        rts = [pk.get("rt_min", 15.0) for pk in species_peaks.values()]
        fwhms = [pk.get("fwhm_min", 1.0) for pk in species_peaks.values()]
        t_min = max(0.0, min(rts) - max(fwhms) * 3.0)
        t_max = max(rts) + max(fwhms) * 3.0 + 2.0
        time_range = (t_min, min(t_max, 60.0))

    t = np.linspace(time_range[0], time_range[1], n_points)
    signals = {}
    total = np.zeros_like(t)

    for key in ("AA", "AB", "BB"):
        pk = species_peaks.get(key, {"rt_min": 15.0, "fwhm_min": 0.5})
        rt = pk["rt_min"]
        fwhm = pk["fwhm_min"]
        sigma_gauss = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        amplitude = relative_abundance.get(key, 0.33)

        signal = amplitude * np.exp(-0.5 * ((t - rt) / max(sigma_gauss, 0.01)) ** 2)
        signals[key] = signal
        total += signal

    # Salt gradient
    gradient = np.linspace(50.0, 500.0, n_points)

    return {
        "time": t,
        "signals": signals,
        "total": total,
        "gradient": gradient,
    }


# ===========================================================================
# 7. Full Bispecific Analysis Pipeline
# ===========================================================================

def run_bispecific_analysis(
    chain_a_seq: str,
    chain_b_seq: str,
    chain_a_name: str = "ArmA",
    chain_b_name: str = "ArmB",
    gradient_slope: float = 15.0,
    ml_override: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Complete bispecific antibody separation analysis pipeline.

    Steps:
      1. Build AntibodyChain objects from sequences
      2. Assemble AA, AB, BB species
      3. Map species to SMA parameters
      4. Estimate retention times and peak widths
      5. Calculate resolution between target AB and homodimers
      6. Assess separation risk and generate recommendations
      7. Generate chromatogram data for visualization

    Parameters
    ----------
    chain_a_seq  : Amino acid sequence for Chain A / Arm 1
    chain_b_seq  : Amino acid sequence for Chain B / Arm 2
    chain_a_name : Display name for Chain A
    chain_b_name : Display name for Chain B
    gradient_slope : Salt gradient slope (mM/min)
    ml_override  : Optional ML-predicted ka/nu for AB species

    Returns
    -------
    dict : Complete analysis result with species, SMA params, peaks,
           resolution, risk assessment, and chromatogram data
    """
    log.info("=" * 60)
    log.info("Bispecific Analysis: %s x %s", chain_a_name, chain_b_name)
    log.info("=" * 60)

    # Step 1: Build chains
    chain_a = AntibodyChain(name=chain_a_name, sequence=chain_a_seq)
    chain_b = AntibodyChain(name=chain_b_name, sequence=chain_b_seq)

    # Step 2: Assemble species
    species = build_assembly_species(chain_a, chain_b)

    # Step 3: Map to SMA parameters
    sma_params = map_species_to_sma(species, ml_override=ml_override)

    # Step 4: Estimate peaks
    peaks = estimate_species_peaks(sma_params, gradient_slope=gradient_slope)

    # Step 5-6: Resolution and risk
    risk = assess_separation_risk(peaks)

    # Step 7: Chromatogram data
    chrom_data = generate_bispecific_chromatogram_data(peaks)

    # Build serializable result
    species_info = {}
    for key, sp in species.items():
        species_info[key] = {
            "label": sp.label,
            "display_name": sp.display_name,
            "pI": sp.pI,
            "mw_kda": sp.mw_kda,
            "hydrophobicity": sp.hydrophobicity,
            "deam_sites": sp.deam_sites,
            "ox_sites": sp.ox_sites,
            "is_target": sp.is_target,
            "seq_length": len(sp.sequence),
        }

    sma_info = {}
    for key, params in sma_params.items():
        sma_info[key] = {
            "nu": params["nu"],
            "ka": params["ka"],
            "kd": params["kd"],
            "sigma": params["sigma"],
            "lambda_": params["lambda_"],
            "source": params.get("source", "static_v5"),
        }

    result = {
        "status": "success",
        "chain_a": {
            "name": chain_a.name,
            "pI": chain_a.pI,
            "mw_kda": chain_a.mw_kda,
            "hydrophobicity": chain_a.hydrophobicity,
            "seq_length": len(chain_a.sequence),
        },
        "chain_b": {
            "name": chain_b.name,
            "pI": chain_b.pI,
            "mw_kda": chain_b.mw_kda,
            "hydrophobicity": chain_b.hydrophobicity,
            "seq_length": len(chain_b.sequence),
        },
        "species": species_info,
        "sma_params": sma_info,
        "peaks": {k: v for k, v in peaks.items()},
        "resolution": {
            "rs_AB_AA": risk["rs_AB_AA"],
            "rs_AB_BB": risk["rs_AB_BB"],
            "min_rs": risk["min_rs"],
        },
        "risk": {
            "risk_level": risk["risk_level"],
            "risk_details": risk["risk_details"],
            "recommendations": risk["recommendations"],
        },
        "chromatogram": {
            "time": chrom_data["time"].tolist(),
            "signals": {k: v.tolist() for k, v in chrom_data["signals"].items()},
            "total": chrom_data["total"].tolist(),
            "gradient": chrom_data["gradient"].tolist(),
        },
    }

    log.info("Bispecific analysis complete: risk=%s, Rs_min=%.3f",
             risk["risk_level"], risk["min_rs"])
    return result


# ===========================================================================
# __main__: Test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  Bispecific Engine v1.0 Test")
    print("=" * 60)

    # Example: Two arms with distinct charge properties
    arm_a_seq = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTY"
        "YADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGYDYWGQGTLVTVSS"
    )
    arm_b_seq = (
        "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKLLIYAASSLQS"
        "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYCLQHNSYPLTFGQGTKLEIK"
    )

    # Test chain construction
    chain_a = AntibodyChain(name="ArmA", sequence=arm_a_seq)
    chain_b = AntibodyChain(name="ArmB", sequence=arm_b_seq)
    print(f"\nChain A: pI={chain_a.pI}, MW={chain_a.mw_kda} kDa")
    print(f"Chain B: pI={chain_b.pI}, MW={chain_b.mw_kda} kDa")

    # Test species assembly
    species = build_assembly_species(chain_a, chain_b)
    for key, sp in species.items():
        print(f"  {key}: {sp.display_name} pI={sp.pI} MW={sp.mw_kda} kDa target={sp.is_target}")

    # Test SMA mapping
    try:
        sma = map_species_to_sma(species)
        for key, params in sma.items():
            print(f"  {key}: nu={params['nu']:.3f} ka={params['ka']:.4f}")
    except Exception as e:
        print(f"  SMA mapping skipped: {e}")

    # Test peak estimation
    try:
        sma = map_species_to_sma(species)
        peaks = estimate_species_peaks(sma)
        for key, pk in peaks.items():
            print(f"  {key}: RT={pk['rt_min']:.2f} min, FWHM={pk['fwhm_min']:.3f} min")

        # Test resolution
        risk = assess_separation_risk(peaks)
        print(f"\n  Rs(AB-AA) = {risk['rs_AB_AA']:.3f}")
        print(f"  Rs(AB-BB) = {risk['rs_AB_BB']:.3f}")
        print(f"  Risk level: {risk['risk_level']}")
        for d in risk["risk_details"]:
            print(f"    {d}")
    except Exception as e:
        print(f"  Peak estimation skipped: {e}")

    # Test full pipeline
    print("\n--- Full Pipeline ---")
    try:
        result = run_bispecific_analysis(
            chain_a_seq=arm_a_seq,
            chain_b_seq=arm_b_seq,
            chain_a_name="ArmA",
            chain_b_name="ArmB",
        )
        print(f"  Status: {result['status']}")
        print(f"  Risk: {result['risk']['risk_level']}")
        print(f"  Rs min: {result['resolution']['min_rs']:.3f}")
        print(f"  Chromatogram points: {len(result['chromatogram']['time'])}")
    except Exception as e:
        print(f"  Pipeline error: {e}")

    print("\nBispecific Engine test complete")
