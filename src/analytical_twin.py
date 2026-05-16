"""
analytical_twin.py  ·  ProtePilot — Milestone 13
===========================================================
Analytical Digital Twin: In-Silico Mass Spectrometry + Glycan Profiling

Version   : 3.0 (Preclinical & Post-Translational Twin)
Author    : Di (ProtePilot)
Depends   : numpy, biopython (optional), pandas (optional)

Purpose
------------------------------------------------------------
Provides in-silico analytical characterization for therapeutic
proteins with TRUE molecular assembly (stoichiometry-aware):

  1. Stoichiometric Intact Mass Calculation
     - Per-chain mass x copy number assembly
     - IgG1 tetramer: (Mass_HC * 2) + (Mass_LC * 2)
     - Disulfide bond correction (-2.016 Da per bond)
     - Common N-glycoform mass additions (G0F, G1F, G2F, etc.)
     - Bispecific: HC-Knob(1) + HC-Hole(1) + LC(2)

  2. In-Silico Tryptic Digest (Peptide Mapping)
     - Trypsin cleavage: after K or R, except when followed by P
     - Source Chain column identifies peptide origin (HC, LC, etc.)
     - Multi-charge m/z prediction (+2, +3)
     - Liability motif annotation per peptide

  3. Liability Density (Normalized Risk)
     - Motifs per 1000 assembled residues
     - Comparable across molecule sizes (mAb vs scFv vs fusion)

  4. Super-Sequence Construction
     - build_super_sequence() concatenates chains * copy numbers
     - Used for global pI, MW, and developability scoring

  5. [M13] Host Cell Glycoform Profiles
     - Standard CHO (G0F/G1F), High-Mannose, Highly Sialylated
     - Profile-specific glycoform mass distributions
     - Sialylation pI shift for physics feedback loop

References
------------------------------------------------------------
  NIST Amino Acid Masses: https://www.nist.gov/
  Zhang et al. (2009) Anal Chem 81:8354 — mAb characterization
  Liu & May (2012) mAbs 4(1):17-23 — N-glycan profiling
  Goetze et al. (2011) Glycobiology 21:949 — Glycan impact on PK
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("ProtePilot.AnalyticalTwin")


# ===========================================================================
# 0. Output Schema — Typed Result Objects
# ===========================================================================
# These dataclasses define the contract between the Analytical module and
# all consumers (report_assembler, bulk_runner, app.py).  The module still
# returns dicts via to_dict() for backward compatibility, but the typed
# objects are available for future callers that want IDE autocomplete and
# type safety.

@dataclass
class LiabilityDensityResult:
    """Liability motif density across all chains."""
    total_residues: int = 0
    total_motifs: int = 0
    density_per_1000: float = 0.0
    per_type_density: Dict[str, float] = field(default_factory=dict)
    per_type_counts: Dict[str, int] = field(default_factory=dict)
    risk_level: str = "Low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_residues": self.total_residues,
            "total_motifs": self.total_motifs,
            "density_per_1000": self.density_per_1000,
            "per_type_density": dict(self.per_type_density),
            "per_type_counts": dict(self.per_type_counts),
            "risk_level": self.risk_level,
        }


@dataclass
class ChainInfo:
    """Assembly chain metadata."""
    name: str = ""
    chain_type: str = "unknown"
    copy_number: int = 1
    length: int = 0


@dataclass
class AnalyticalSummary:
    """Summary statistics for an MS characterization run."""
    total_peptides: int = 0
    peptides_with_liabilities: int = 0
    liability_peptides: int = 0
    unique_liability_types: List[str] = field(default_factory=list)
    coverage_pct: float = 0.0
    sequence_coverage_pct: float = 0.0


@dataclass
class AnalyticalResult:
    """Top-level result from run_ms_characterization().

    This is the canonical output contract for the Analytical / PTM /
    Liability Engine module.  All fields are JSON-serializable.
    """
    status: str = "success"
    protein_name: str = ""
    sequence_length: int = 0
    intact_mass: Dict[str, Any] = field(default_factory=dict)
    peptide_map: List[Dict[str, Any]] = field(default_factory=list)
    liability_density: Dict[str, Any] = field(default_factory=dict)
    n_glycosylation_sites: int = 0
    assembly_mode: str = "stoichiometric"
    chains_used: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    # Error-only field
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Backward-compatible dict export (matches legacy return format)."""
        d: Dict[str, Any] = {
            "status": self.status,
            "protein_name": self.protein_name,
            "sequence_length": self.sequence_length,
            "intact_mass": self.intact_mass,
            "peptide_map": self.peptide_map,
            "liability_density": self.liability_density,
            "n_glycosylation_sites": self.n_glycosylation_sites,
            "assembly_mode": self.assembly_mode,
            "chains_used": self.chains_used,
            "summary": self.summary,
            "data": {
                "intact_mass": self.intact_mass,
                "peptide_map": self.peptide_map,
                "liability_density": self.liability_density,
                "summary": self.summary,
            },
        }
        if self.message:
            d["message"] = self.message
        return d


# ===========================================================================
# 1. Amino Acid Mass Data
# ===========================================================================

# Average residue masses (Da) — standard amino acids
AMINO_ACID_AVG_MASS: Dict[str, float] = {
    "A":  71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886,
    "C": 103.1388, "E": 129.1155, "Q": 128.1307, "G":  57.0519,
    "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
    "M": 131.1926, "F": 147.1766, "P":  97.1167, "S":  87.0782,
    "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V":  99.1326,
}

# Monoisotopic residue masses (Da)
AMINO_ACID_MONO_MASS: Dict[str, float] = {
    "A":  71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G":  57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P":  97.05276, "S":  87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V":  99.06841,
}

# Water mass (added once to get full peptide/protein mass)
WATER_MASS_AVG = 18.0153
WATER_MASS_MONO = 18.01056

# Proton mass for m/z calculations
PROTON_MASS = 1.00728

# Disulfide bond mass correction: loss of 2 H per bond
DISULFIDE_CORRECTION = -2.01565  # Da per disulfide bond


# ===========================================================================
# 2. Common mAb N-Glycoforms
# ===========================================================================

GLYCOFORM_MASSES: Dict[str, Dict[str, Any]] = {
    "G0F": {
        "name": "G0F (core-fucosylated, no galactose)",
        "mass_da": 1444.53,
        "abundance": "major",
        "description": "Most common mAb glycoform; GlcNAc2Man3GlcNAc2Fuc1",
    },
    "G1F": {
        "name": "G1F (one galactose)",
        "mass_da": 1606.58,
        "abundance": "major",
        "description": "One galactose added; GlcNAc2Man3GlcNAc2Fuc1Gal1",
    },
    "G2F": {
        "name": "G2F (two galactose)",
        "mass_da": 1768.64,
        "abundance": "minor",
        "description": "Fully galactosylated; GlcNAc2Man3GlcNAc2Fuc1Gal2",
    },
    "G0": {
        "name": "G0 (no fucose, no galactose)",
        "mass_da": 1298.47,
        "abundance": "minor",
        "description": "Afucosylated core; GlcNAc2Man3GlcNAc2",
    },
    "Man5": {
        "name": "Man5 (high-mannose)",
        "mass_da": 1234.43,
        "abundance": "minor",
        "description": "High-mannose species; Man5GlcNAc2",
    },
    "G1F_SA": {
        "name": "G1F+SA (sialylated)",
        "mass_da": 1897.69,
        "abundance": "trace",
        "description": "Sialylated G1F; GlcNAc2Man3GlcNAc2Fuc1Gal1NeuAc1",
    },
    "G2F_2SA": {
        "name": "G2F+2SA (di-sialylated)",
        "mass_da": 2351.13,
        "abundance": "trace",
        "description": "Di-sialylated G2F; GlcNAc2Man3GlcNAc2Fuc1Gal2NeuAc2",
    },
    "Man8": {
        "name": "Man8 (high-mannose)",
        "mass_da": 1720.60,
        "abundance": "trace",
        "description": "High-mannose species; Man8GlcNAc2",
    },
    "Man9": {
        "name": "Man9 (high-mannose)",
        "mass_da": 1882.66,
        "abundance": "trace",
        "description": "High-mannose species; Man9GlcNAc2",
    },
}


# ===========================================================================
# 2B. Host Cell Glycoform Profiles (M13: Preclinical Twin)
# ===========================================================================

# Each profile defines the expected glycoform distribution for a given
# expression system or cell line variant. Used by the UI dropdown and
# by the PK predictor for half-life estimation.

HOST_CELL_GLYCOFORM_PROFILES: Dict[str, Dict[str, Any]] = {
    "standard_cho": {
        "name": "Standard CHO (G0F/G1F)",
        "description": "Normal CHO-K1 / CHO-DG44 glycosylation; dominant G0F/G1F species",
        "glycoforms": [
            {"key": "G0F",  "abundance_pct": 45.0, "label": "major"},
            {"key": "G1F",  "abundance_pct": 30.0, "label": "major"},
            {"key": "G2F",  "abundance_pct": 12.0, "label": "minor"},
            {"key": "G0",   "abundance_pct":  5.0, "label": "minor"},
            {"key": "Man5", "abundance_pct":  3.0, "label": "trace"},
            {"key": "G1F_SA", "abundance_pct": 2.0, "label": "trace"},
        ],
        "pi_shift": 0.0,
        "dominant_mass_key": "G0F",
    },
    "high_mannose": {
        "name": "High-Mannose (Man5/Man8/Man9)",
        "description": "Elevated high-mannose species; common with kifunensine treatment or early harvest",
        "glycoforms": [
            {"key": "Man5", "abundance_pct": 40.0, "label": "major"},
            {"key": "Man8", "abundance_pct": 25.0, "label": "major"},
            {"key": "Man9", "abundance_pct": 15.0, "label": "minor"},
            {"key": "G0F",  "abundance_pct": 10.0, "label": "minor"},
            {"key": "G1F",  "abundance_pct":  5.0, "label": "trace"},
            {"key": "G0",   "abundance_pct":  5.0, "label": "trace"},
        ],
        "pi_shift": 0.0,
        "dominant_mass_key": "Man5",
    },
    "afucosylated": {
        "name": "Afucosylated (G0, enhanced ADCC)",
        "description": "Afucosylated profile; Potelligent / FUT8 knockout for enhanced ADCC",
        "glycoforms": [
            {"key": "G0",   "abundance_pct": 50.0, "label": "major"},
            {"key": "G1F",  "abundance_pct": 15.0, "label": "minor"},
            {"key": "G0F",  "abundance_pct": 12.0, "label": "minor"},
            {"key": "G2F",  "abundance_pct":  8.0, "label": "minor"},
            {"key": "Man5", "abundance_pct": 10.0, "label": "minor"},
            {"key": "G1F_SA", "abundance_pct": 5.0, "label": "trace"},
        ],
        "pi_shift": 0.0,
        "dominant_mass_key": "G0",
    },
    "highly_sialylated": {
        "name": "Highly Sialylated (G2F+2SA)",
        "description": "Maximized sialylation via ST6GAL1 overexpression or CMP-NeuAc supplementation. Adds negative charges, lowering pI.",
        "glycoforms": [
            {"key": "G1F_SA",  "abundance_pct": 35.0, "label": "major"},
            {"key": "G2F_2SA", "abundance_pct": 25.0, "label": "major"},
            {"key": "G2F",     "abundance_pct": 15.0, "label": "minor"},
            {"key": "G1F",     "abundance_pct": 12.0, "label": "minor"},
            {"key": "G0F",     "abundance_pct":  8.0, "label": "minor"},
            {"key": "Man5",    "abundance_pct":  5.0, "label": "trace"},
        ],
        "pi_shift": -0.3,  # Sialic acid negative charges lower pI
        "dominant_mass_key": "G1F_SA",
    },
}


def get_glycoform_profile_for_ms(
    profile_key: str = "standard_cho",
    n_glycosylation_sites: int = 2,
    base_mass: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Generate glycoform mass variants for a given host cell profile.

    Parameters
    ----------
    profile_key           : Key from HOST_CELL_GLYCOFORM_PROFILES
    n_glycosylation_sites : Number of N-glycosylation sites (2 for mAb Fc)
    base_mass             : Disulfide-corrected protein mass (Da)

    Returns
    -------
    list of dicts: [{name, glycan_key, per_site_da, total_glycan_da,
                     intact_mass_da, abundance_pct, label, n_sites}]
    """
    profile = HOST_CELL_GLYCOFORM_PROFILES.get(profile_key)
    if not profile:
        profile = HOST_CELL_GLYCOFORM_PROFILES["standard_cho"]

    result = []
    for gf_entry in profile["glycoforms"]:
        gf_key = gf_entry["key"]
        gf_info = GLYCOFORM_MASSES.get(gf_key)
        if not gf_info:
            continue

        per_site = gf_info["mass_da"]
        total_glycan = per_site * n_glycosylation_sites
        intact = base_mass + total_glycan if base_mass > 0 else 0.0

        result.append({
            "name": gf_info["name"],
            "glycan_key": gf_key,
            "per_site_da": per_site,
            "total_glycan_da": round(total_glycan, 2),
            "intact_mass_da": round(intact, 2) if intact > 0 else 0.0,
            "abundance_pct": gf_entry["abundance_pct"],
            "abundance": gf_entry["label"],
            "n_sites": n_glycosylation_sites,
        })

    return result


def get_glycoform_pi_shift(profile_key: str = "standard_cho") -> float:
    """Return the pI shift for a given glycoform profile.

    Note: This function reads from HOST_CELL_GLYCOFORM_PROFILES
    (analytical characterization context). A same-name function exists in
    preclinical_twin.py reading from GLYCOFORM_PK_MODIFIERS (PK context).
    The two dicts encode different assumptions — kept intentionally separate.
    """
    profile = HOST_CELL_GLYCOFORM_PROFILES.get(profile_key, {})
    return profile.get("pi_shift", 0.0)


# ===========================================================================
# 3. Intact Mass Calculation
# ===========================================================================

def calculate_sequence_mass(
    sequence: str,
    mass_type: str = "average",
) -> float:
    """
    Calculate the theoretical mass of a protein sequence.

    Parameters
    ----------
    sequence  : Amino acid sequence (single-letter code)
    mass_type : "average" or "monoisotopic"

    Returns
    -------
    float : Mass in Daltons (includes water for intact protein)
    """
    seq = sequence.upper()
    mass_table = AMINO_ACID_AVG_MASS if mass_type == "average" else AMINO_ACID_MONO_MASS
    water = WATER_MASS_AVG if mass_type == "average" else WATER_MASS_MONO

    total = water  # N-terminal H + C-terminal OH
    for aa in seq:
        total += mass_table.get(aa, 0.0)

    return round(total, 4)


def _resolve_is_mab(molecule_class: Optional[str] = None, is_mab: Optional[bool] = None) -> bool:
    """Resolve is_mab from molecule_class, with backward-compatible fallback."""
    if molecule_class is not None:
        try:
            from src.molecule_classifier import MoleculeClass
            mc = MoleculeClass(molecule_class)
            return mc.is_mab_like
        except (ValueError, ImportError):
            pass
    if is_mab is not None:
        return is_mab
    return True  # legacy default


def estimate_disulfide_bonds(sequence: str, is_mab: bool = True,
                             molecule_class: Optional[str] = None) -> int:
    """
    Estimate the number of disulfide bonds in a protein.

    For mAb-like molecules: typically 16 (4 inter-chain + 12 intra-chain for IgG1).
    For single chains: floor(Cys_count / 2).

    Parameters
    ----------
    sequence        : Amino acid sequence
    is_mab          : Whether this is a full mAb (legacy, use molecule_class instead)
    molecule_class  : MoleculeClass string (preferred over is_mab)

    Returns
    -------
    int : Estimated number of disulfide bonds
    """
    _mab = _resolve_is_mab(molecule_class, is_mab)
    cys_count = sequence.upper().count("C")

    # Try class-specific disulfide count
    if molecule_class:
        try:
            from src.molecule_classifier import MoleculeClass
            mc = MoleculeClass(molecule_class)
            expected = mc.expected_disulfide_bonds
            if expected > 0 and cys_count >= expected:
                return expected
        except (ValueError, ImportError):
            pass

    if _mab and cys_count >= 20:
        return 16  # Standard IgG1: 4 inter-chain + 12 intra-chain
    elif _mab and cys_count >= 10:
        return cys_count // 2
    else:
        return max(0, cys_count // 2)


def calculate_intact_mass(
    sequence: str,
    n_disulfide_bonds: Optional[int] = None,
    is_mab: bool = True,
    include_glycoforms: bool = True,
    n_glycosylation_sites: int = 2,
    molecule_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate the intact mass of a protein including modifications.

    For mAbs, computes:
      - Bare protein mass (no modifications)
      - Disulfide-corrected mass
      - Glycoform masses (G0F, G1F, G2F, etc.)

    Parameters
    ----------
    sequence             : Full amino acid sequence
    n_disulfide_bonds    : Override for disulfide bond count
    is_mab               : Whether this is a full mAb (legacy, prefer molecule_class)
    include_glycoforms   : Whether to add glycoform mass predictions
    n_glycosylation_sites: Number of N-glycosylation sites (default 2 for mAb)
    molecule_class       : MoleculeClass string (preferred over is_mab)

    Returns
    -------
    dict : {
        "bare_mass_da": float,
        "disulfide_corrected_da": float,
        "n_disulfide_bonds": int,
        "glycoforms": [{name, mass_da, abundance}],
        "sequence_length": int,
        "formula_summary": str,
    }
    """
    seq = sequence.upper()
    bare_mass = calculate_sequence_mass(seq, mass_type="average")

    if n_disulfide_bonds is None:
        n_ss = estimate_disulfide_bonds(seq, is_mab=is_mab, molecule_class=molecule_class)
    else:
        n_ss = n_disulfide_bonds

    ss_corrected = bare_mass + (n_ss * DISULFIDE_CORRECTION)

    result = {
        "bare_mass_da": round(bare_mass, 2),
        "disulfide_corrected_da": round(ss_corrected, 2),
        "n_disulfide_bonds": n_ss,
        "sequence_length": len(seq),
        "glycoforms": [],
        "formula_summary": (
            f"MW = {bare_mass:.2f} Da (bare) "
            f"- {abs(n_ss * DISULFIDE_CORRECTION):.2f} Da ({n_ss} S-S bonds) "
            f"= {ss_corrected:.2f} Da"
        ),
    }

    if include_glycoforms and n_glycosylation_sites > 0:
        for glyform_key, glyform_info in GLYCOFORM_MASSES.items():
            # Each glycosylation site gets one glycan
            total_glycan_mass = glyform_info["mass_da"] * n_glycosylation_sites
            glyco_mass = ss_corrected + total_glycan_mass

            result["glycoforms"].append({
                "name": glyform_info["name"],
                "glycan_key": glyform_key,
                "per_site_da": glyform_info["mass_da"],
                "total_glycan_da": round(total_glycan_mass, 2),
                "intact_mass_da": round(glyco_mass, 2),
                "abundance": glyform_info["abundance"],
                "n_sites": n_glycosylation_sites,
            })

    log.info("Intact mass: bare=%.2f, SS-corrected=%.2f, glycoforms=%d",
             bare_mass, ss_corrected, len(result["glycoforms"]))
    return result


# ===========================================================================
# 3B. Super-Sequence Construction (M12: True Molecular Assembly)
# ===========================================================================

def build_super_sequence(
    chains: List[Dict[str, Any]],
) -> str:
    """
    Build a super-sequence by concatenating chains multiplied by copy numbers.

    Each chain dict must have:
      - "sequence": str (amino acid sequence)
      - "copy_number": int (stoichiometric multiplier, default 1)

    Example for standard IgG1:
      chains = [
          {"sequence": HC_seq, "copy_number": 2, "name": "HC"},
          {"sequence": LC_seq, "copy_number": 2, "name": "LC"},
      ]
      => super_sequence = HC_seq + HC_seq + LC_seq + LC_seq

    Parameters
    ----------
    chains : List of chain dicts with sequence and copy_number

    Returns
    -------
    str : Concatenated super-sequence
    """
    parts = []
    for chain in chains:
        seq = chain.get("sequence", "").upper()
        copies = max(1, int(chain.get("copy_number", 1)))
        for _ in range(copies):
            parts.append(seq)
    return "".join(parts)


def calculate_stoichiometric_intact_mass(
    chains: List[Dict[str, Any]],
    n_disulfide_bonds: Optional[int] = None,
    is_mab: bool = True,
    include_glycoforms: bool = True,
    n_glycosylation_sites: Optional[int] = None,
    glycoform_profile: Optional[str] = None,
    molecule_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate the intact mass of a fully assembled multi-chain protein.

    Uses true stoichiometry: Mass = SUM(chain_mass_i * copy_number_i)
    instead of single concatenated sequence mass.

    For IgG1: (Mass_HC * 2) + (Mass_LC * 2) - 32 Da (16 disulfide bonds)

    Parameters
    ----------
    chains : List of chain dicts, each with:
        - "sequence": str
        - "copy_number": int (default 1)
        - "name": str (optional, for labeling)
        - "chain_type": str (optional, "HC", "LC", etc.)
    n_disulfide_bonds : Override (default: auto-estimate)
    is_mab            : Whether this is a full mAb (legacy, prefer molecule_class)
    include_glycoforms: Whether to add glycoform mass predictions
    n_glycosylation_sites : Override for glyco site count
    molecule_class    : MoleculeClass string (preferred over is_mab)

    Returns
    -------
    dict : {
        "bare_mass_da": float (assembled, no modifications),
        "disulfide_corrected_da": float,
        "n_disulfide_bonds": int,
        "glycoforms": list,
        "per_chain_masses": list of per-chain mass info,
        "total_assembled_length": int,
        "stoichiometry_summary": str,
    }
    """
    per_chain = []
    total_bare_mass = 0.0
    total_length = 0
    total_cys = 0
    stoich_parts = []

    for chain in chains:
        seq = chain.get("sequence", "").upper()
        copies = max(1, int(chain.get("copy_number", 1)))
        chain_name = chain.get("name", "Chain")
        chain_type = chain.get("chain_type", "unknown")

        chain_mass = calculate_sequence_mass(seq, mass_type="average")
        chain_cys = seq.count("C")

        per_chain.append({
            "name": chain_name,
            "chain_type": chain_type,
            "sequence_length": len(seq),
            "copy_number": copies,
            "chain_mass_da": round(chain_mass, 2),
            "total_mass_da": round(chain_mass * copies, 2),
            "cys_count": chain_cys,
        })

        total_bare_mass += chain_mass * copies
        total_length += len(seq) * copies
        total_cys += chain_cys * copies
        stoich_parts.append(f"{chain_name}(x{copies})")

    # Disulfide bonds — prefer molecule_class if available
    _mab = _resolve_is_mab(molecule_class, is_mab)
    if n_disulfide_bonds is not None:
        n_ss = n_disulfide_bonds
    elif molecule_class:
        try:
            from src.molecule_classifier import MoleculeClass
            mc = MoleculeClass(molecule_class)
            expected = mc.expected_disulfide_bonds
            if expected > 0 and total_cys >= expected:
                n_ss = expected
            else:
                n_ss = max(0, total_cys // 2)
        except (ValueError, ImportError):
            n_ss = 16 if _mab and total_cys >= 20 else max(0, total_cys // 2)
    elif _mab and total_cys >= 20:
        n_ss = 16  # Standard IgG1: 4 inter-chain + 12 intra-chain
    else:
        n_ss = max(0, total_cys // 2)

    ss_corrected = total_bare_mass + (n_ss * DISULFIDE_CORRECTION)

    # Glycosylation sites
    if n_glycosylation_sites is not None:
        n_glyco = n_glycosylation_sites
    else:
        # Count NxS/T motifs in super-sequence — do NOT assume sites exist
        import re as _re
        super_seq = build_super_sequence(chains)
        n_glyco = len(_re.compile(r"N[^P][ST]").findall(super_seq))

    result = {
        "bare_mass_da": round(total_bare_mass, 2),
        "disulfide_corrected_da": round(ss_corrected, 2),
        "n_disulfide_bonds": n_ss,
        "per_chain_masses": per_chain,
        "total_assembled_length": total_length,
        "sequence_length": total_length,
        "stoichiometry_summary": " + ".join(stoich_parts),
        "glycoforms": [],
        "formula_summary": (
            f"MW = {total_bare_mass:.2f} Da (bare assembled) "
            f"- {abs(n_ss * DISULFIDE_CORRECTION):.2f} Da ({n_ss} S-S bonds) "
            f"= {ss_corrected:.2f} Da"
        ),
    }

    if include_glycoforms and n_glyco > 0:
        # M13: Use host cell glycoform profile if specified
        if glycoform_profile and glycoform_profile in HOST_CELL_GLYCOFORM_PROFILES:
            result["glycoforms"] = get_glycoform_profile_for_ms(
                profile_key=glycoform_profile,
                n_glycosylation_sites=n_glyco,
                base_mass=ss_corrected,
            )
            result["glycoform_profile"] = glycoform_profile
            result["glycoform_pi_shift"] = get_glycoform_pi_shift(glycoform_profile)
        else:
            # Legacy: iterate all known glycoforms
            for glyform_key, glyform_info in GLYCOFORM_MASSES.items():
                total_glycan_mass = glyform_info["mass_da"] * n_glyco
                glyco_mass = ss_corrected + total_glycan_mass
                result["glycoforms"].append({
                    "name": glyform_info["name"],
                    "glycan_key": glyform_key,
                    "per_site_da": glyform_info["mass_da"],
                    "total_glycan_da": round(total_glycan_mass, 2),
                    "intact_mass_da": round(glyco_mass, 2),
                    "abundance": glyform_info["abundance"],
                    "n_sites": n_glyco,
                })

    log.info("Stoichiometric intact mass: %s = %.2f Da (SS-corrected), "
             "%d chains assembled, %d total residues",
             result["stoichiometry_summary"], ss_corrected,
             len(chains), total_length)
    return result


# ===========================================================================
# 3C. Liability Density (M12: Normalized Risk)
# ===========================================================================

def calculate_liability_density(
    chains: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate liability density: motifs per 1000 assembled residues.

    This normalizes risk so that a 150 kDa mAb and a 25 kDa scFv
    can be compared fairly. An IgG1 with 10 oxidation sites across
    1340 residues is different from a nanobody with 5 across 130 residues.

    Parameters
    ----------
    chains : List of chain dicts with "sequence" and "copy_number"

    Returns
    -------
    dict : {
        "total_residues": int,
        "total_motifs": int,
        "density_per_1000": float,
        "per_type_density": {motif_type: density_per_1000},
        "per_type_counts": {motif_type: absolute_count},
        "risk_level": "Low" | "Medium" | "High",
    }
    """
    super_seq = build_super_sequence(chains)
    total_len = len(super_seq)

    if total_len == 0:
        return {
            "total_residues": 0, "total_motifs": 0,
            "density_per_1000": 0.0,
            "per_type_density": {}, "per_type_counts": {},
            "risk_level": "Low",
        }

    per_type_counts: Dict[str, int] = {}
    total_motifs = 0

    for motif_name, pattern in LIABILITY_MOTIFS.items():
        count = len(list(pattern.finditer(super_seq)))
        per_type_counts[motif_name] = count
        total_motifs += count

    density = (total_motifs / total_len) * 1000.0

    per_type_density = {}
    for motif_name, count in per_type_counts.items():
        per_type_density[motif_name] = round((count / total_len) * 1000.0, 2)

    # Risk classification based on total density
    if density > 80:
        risk_level = "High"
    elif density > 50:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "total_residues": total_len,
        "total_motifs": total_motifs,
        "density_per_1000": round(density, 2),
        "per_type_density": per_type_density,
        "per_type_counts": per_type_counts,
        "risk_level": risk_level,
    }


# ===========================================================================
# 4. In-Silico Tryptic Digest (Peptide Mapping)
# ===========================================================================

# Liability motif patterns
LIABILITY_MOTIFS = {
    "Deamidation (NG)": re.compile(r"NG"),
    "Deamidation (NS)": re.compile(r"NS"),
    "Deamidation (NT)": re.compile(r"NT"),
    "Oxidation (Met)": re.compile(r"M"),
    "Oxidation (Trp)": re.compile(r"W"),
    "N-Glycosylation": re.compile(r"N[^P][ST]"),
    "Asp-Pro Clip": re.compile(r"DP"),
    "Isomerization (DG)": re.compile(r"DG"),
    "Isomerization (DS)": re.compile(r"DS"),
}


def tryptic_digest(
    sequence: str,
    missed_cleavages: int = 0,
    min_length: int = 4,
    max_length: int = 60,
    source_chain: str = "",
) -> List[Dict[str, Any]]:
    """
    Simulate trypsin digestion of a protein sequence.

    Trypsin cleaves after K (Lys) or R (Arg), except when
    followed by P (Pro) — the "KR/P rule".

    Parameters
    ----------
    sequence         : Amino acid sequence
    missed_cleavages : Number of allowed missed cleavages (0, 1, or 2)
    min_length       : Minimum peptide length to include
    max_length       : Maximum peptide length to include
    source_chain     : Label identifying the source chain (e.g. "HC", "LC")

    Returns
    -------
    list of dicts, each containing:
        - index: peptide number
        - sequence: peptide amino acid sequence
        - start: start position (1-indexed)
        - end: end position (1-indexed)
        - length: peptide length
        - mass_avg: average mass (Da)
        - mass_mono: monoisotopic mass (Da)
        - mz_2: m/z at charge +2
        - mz_3: m/z at charge +3
        - missed_cleavages: number of missed cleavage sites
        - source_chain: source chain label (M12)
        - liabilities: list of liability annotations
        - has_liability: bool
    """
    seq = sequence.upper()
    n = len(seq)

    # Find cleavage sites (positions AFTER K or R, except before P)
    cleavage_sites = []
    for i in range(n - 1):
        if seq[i] in ("K", "R"):
            if seq[i + 1] != "P":
                cleavage_sites.append(i + 1)
    cleavage_sites.append(n)  # C-terminal end

    # Generate zero-missed-cleavage peptides
    base_peptides = []
    start = 0
    for site in cleavage_sites:
        pep_seq = seq[start:site]
        if pep_seq:
            base_peptides.append((start, site, pep_seq))
        start = site

    # Generate peptides with missed cleavages
    all_peptides = []
    for mc in range(missed_cleavages + 1):
        for i in range(len(base_peptides) - mc):
            # Merge mc+1 consecutive base peptides
            merged_start = base_peptides[i][0]
            merged_end = base_peptides[i + mc][1]
            merged_seq = seq[merged_start:merged_end]

            if min_length <= len(merged_seq) <= max_length:
                all_peptides.append({
                    "start": merged_start,
                    "end": merged_end,
                    "sequence": merged_seq,
                    "missed_cleavages": mc,
                })

    # Calculate masses and annotate
    result = []
    for idx, pep in enumerate(all_peptides):
        pep_seq = pep["sequence"]
        mass_avg = calculate_sequence_mass(pep_seq, mass_type="average")
        mass_mono = calculate_sequence_mass(pep_seq, mass_type="monoisotopic")

        # m/z at different charge states: (M + z*H) / z
        mz_2 = (mass_mono + 2 * PROTON_MASS) / 2.0
        mz_3 = (mass_mono + 3 * PROTON_MASS) / 3.0

        # Annotate liabilities
        liabilities = []
        for motif_name, pattern in LIABILITY_MOTIFS.items():
            matches = list(pattern.finditer(pep_seq))
            for match in matches:
                abs_pos = pep["start"] + match.start()
                liabilities.append({
                    "motif": motif_name,
                    "local_pos": match.start() + 1,  # 1-indexed within peptide
                    "abs_pos": abs_pos + 1,           # 1-indexed in full protein
                    "residues": match.group(),
                })

        result.append({
            "index": idx + 1,
            "sequence": pep_seq,
            "start": pep["start"] + 1,  # 1-indexed
            "end": pep["end"],
            "length": len(pep_seq),
            "mass_avg_da": round(mass_avg, 4),
            "mass_mono_da": round(mass_mono, 4),
            "mz_2": round(mz_2, 4),
            "mz_3": round(mz_3, 4),
            "missed_cleavages": pep["missed_cleavages"],
            "source_chain": source_chain,
            "liabilities": liabilities,
            "has_liability": len(liabilities) > 0,
            "n_liabilities": len(liabilities),
            "liability_summary": ", ".join(
                sorted(set(l["motif"] for l in liabilities))
            ) if liabilities else "",
        })

    log.info("Tryptic digest: %d peptides (mc=%d), %d with liabilities",
             len(result), missed_cleavages,
             sum(1 for p in result if p["has_liability"]))
    return result


# ===========================================================================
# 5. Convenience: Full MS Characterization
# ===========================================================================

def run_ms_characterization(
    sequence: str = "",
    protein_name: str = "Protein",
    is_mab: bool = True,
    missed_cleavages: int = 1,
    chains: Optional[List[Dict[str, Any]]] = None,
    glycoform_profile: Optional[str] = None,
    molecule_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete in-silico MS characterization.

    M12: When chains are provided, uses true stoichiometric assembly:
      - Intact mass = SUM(chain_mass * copy_number)
      - Peptide map includes 'source_chain' column per peptide
      - Liability density (motifs per 1000 residues) for normalized risk
      - Only unique chains are digested (no duplicate peptide entries)

    Parameters
    ----------
    sequence          : Full amino acid sequence (legacy single-chain mode)
    protein_name      : Protein name for labeling
    is_mab            : Whether this is a full mAb (legacy, prefer molecule_class)
    missed_cleavages  : Missed cleavages for tryptic digest
    chains            : M12 multi-chain assembly (overrides sequence param)
                        Each chain: {"sequence", "copy_number", "name", "chain_type"}
    molecule_class    : MoleculeClass string (preferred over is_mab)

    Returns
    -------
    dict : {
        "status": "success" | "error",
        "data" or top-level keys: {
            "protein_name": str,
            "intact_mass": intact mass result dict,
            "peptide_map": list of peptide dicts (with source_chain),
            "liability_density": dict (M12),
            "summary": {...},
        }
    }
    """
    # -- M12: Multi-chain stoichiometric mode --------------------------------
    if chains and len(chains) > 0:
        # Validate chains
        valid_chains = []
        for ch in chains:
            seq = ch.get("sequence", "").upper()
            if len(seq) >= 10:
                valid_chains.append({
                    "sequence": seq,
                    "copy_number": max(1, int(ch.get("copy_number", 1))),
                    "name": ch.get("name", "Chain"),
                    "chain_type": ch.get("chain_type", "unknown"),
                })
        if not valid_chains:
            return {"status": "error", "message": "No valid chains (min 10 aa)"}

        # Stoichiometric intact mass (M13: glycoform profile support)
        intact = calculate_stoichiometric_intact_mass(
            valid_chains,
            is_mab=is_mab,
            glycoform_profile=glycoform_profile,
            molecule_class=molecule_class,
        )

        # Per-chain tryptic digest (unique chains only, labeled)
        all_peptides = []
        peptide_index = 1
        for ch in valid_chains:
            chain_label = f"{ch['chain_type']}({ch['name']})" if ch['chain_type'] != 'unknown' else ch['name']
            chain_peps = tryptic_digest(
                ch["sequence"],
                missed_cleavages=missed_cleavages,
                source_chain=chain_label,
            )
            # Re-index globally
            for pep in chain_peps:
                pep["index"] = peptide_index
                peptide_index += 1
            all_peptides.extend(chain_peps)

        # Liability density (M12)
        liab_density = calculate_liability_density(valid_chains)

        # Summary
        liab_peptides = [p for p in all_peptides if p["has_liability"]]
        all_liab_types = set()
        for p in liab_peptides:
            for lb in p["liabilities"]:
                all_liab_types.add(lb["motif"])

        super_seq = build_super_sequence(valid_chains)
        covered = set()
        # Coverage per unique chain
        for ch in valid_chains:
            for p in all_peptides:
                if p.get("source_chain", "").startswith(ch.get("chain_type", "?")):
                    for pos in range(p["start"] - 1, p["end"]):
                        covered.add(pos)
        # Approximate coverage as fraction of longest unique chain
        unique_len = sum(len(ch["sequence"]) for ch in valid_chains)
        coverage_pct = round(100.0 * len(covered) / unique_len, 1) if unique_len else 0.0

        result = {
            "status": "success",
            "protein_name": protein_name,
            "sequence_length": intact["total_assembled_length"],
            "intact_mass": intact,
            "peptide_map": all_peptides,
            "liability_density": liab_density,
            "n_glycosylation_sites": 2 if _resolve_is_mab(molecule_class, is_mab) else 0,
            "assembly_mode": "stoichiometric",
            "chains_used": [
                {"name": ch["name"], "chain_type": ch["chain_type"],
                 "copy_number": ch["copy_number"], "length": len(ch["sequence"])}
                for ch in valid_chains
            ],
            "summary": {
                "total_peptides": len(all_peptides),
                "peptides_with_liabilities": len(liab_peptides),
                "liability_peptides": len(liab_peptides),
                "unique_liability_types": sorted(all_liab_types),
                "coverage_pct": coverage_pct,
                "sequence_coverage_pct": coverage_pct,
            },
            "data": {},  # compatibility
        }
        # Also place under "data" for compatibility with app.py rendering
        result["data"] = {
            "intact_mass": result["intact_mass"],
            "peptide_map": result["peptide_map"],
            "liability_density": result["liability_density"],
            "summary": result["summary"],
        }

        log.info("MS characterization (stoichiometric) for %s: %d unique peptides, "
                 "liability density=%.1f/1000 residues",
                 protein_name, len(all_peptides), liab_density["density_per_1000"])
        return result

    # -- Legacy single-sequence mode ----------------------------------------
    seq = sequence.upper()
    if len(seq) < 10:
        return {"status": "error", "message": "Sequence too short (min 10 aa)"}

    # Count N-glycosylation sites via NxS/T motif (x != P)
    import re as _re
    _nglyco_motif = _re.compile(r"N[^P][ST]")
    n_glyco_sites = len(_nglyco_motif.findall(seq))
    # If zero motifs found, molecule is aglycosylated — do NOT hallucinate glycans
    _is_aglycosylated = (n_glyco_sites == 0)

    # Intact mass
    intact = calculate_intact_mass(
        seq,
        is_mab=is_mab,
        include_glycoforms=(not _is_aglycosylated),
        n_glycosylation_sites=n_glyco_sites,
        molecule_class=molecule_class,
    )

    # Tryptic digest
    peptides = tryptic_digest(
        seq,
        missed_cleavages=missed_cleavages,
        source_chain="Single",
    )

    # Liability density for single sequence
    single_chains = [{"sequence": seq, "copy_number": 1}]
    liab_density = calculate_liability_density(single_chains)

    # Summary statistics
    liab_peptides = [p for p in peptides if p["has_liability"]]
    all_liab_types = set()
    for p in liab_peptides:
        for lb in p["liabilities"]:
            all_liab_types.add(lb["motif"])

    # Sequence coverage
    covered = set()
    for p in peptides:
        for pos in range(p["start"] - 1, p["end"]):
            covered.add(pos)
    coverage_pct = round(100.0 * len(covered) / len(seq), 1) if seq else 0.0

    result = {
        "status": "success",
        "protein_name": protein_name,
        "sequence_length": len(seq),
        "intact_mass": intact,
        "peptide_map": peptides,
        "liability_density": liab_density,
        "n_glycosylation_sites": n_glyco_sites,
        "is_aglycosylated": _is_aglycosylated,
        "assembly_mode": "single_sequence",
        "summary": {
            "total_peptides": len(peptides),
            "peptides_with_liabilities": len(liab_peptides),
            "liability_peptides": len(liab_peptides),
            "unique_liability_types": sorted(all_liab_types),
            "coverage_pct": coverage_pct,
            "sequence_coverage_pct": coverage_pct,
        },
        "data": {},
    }
    result["data"] = {
        "intact_mass": result["intact_mass"],
        "peptide_map": result["peptide_map"],
        "liability_density": result["liability_density"],
        "summary": result["summary"],
    }

    log.info("MS characterization complete for %s: %d peptides, %.1f%% coverage",
             protein_name, len(peptides), coverage_pct)
    return result


def peptide_map_to_dataframe(peptides: List[Dict[str, Any]]):
    """
    Convert peptide map list to a pandas DataFrame.

    Returns None if pandas is not available.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    rows = []
    for p in peptides:
        row = {
            "#": p["index"],
            "Source Chain": p.get("source_chain", ""),
            "Sequence": p["sequence"],
            "Start": p["start"],
            "End": p["end"],
            "Length": p["length"],
            "Mass (avg)": p["mass_avg_da"],
            "Mass (mono)": p["mass_mono_da"],
            "m/z (+2)": p["mz_2"],
            "m/z (+3)": p["mz_3"],
            "MC": p["missed_cleavages"],
            "PTM Liabilities": p["liability_summary"],
            "# Liab": p["n_liabilities"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_ptm_hotspot_table(peptides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a consolidated PTM hotspot summary from tryptic peptide map.

    v2.0: Groups all liabilities by motif type with positions, risk severity,
    and recommended monitoring strategy.

    Returns list of dicts:
        [{"motif": "Deamidation (NG)", "count": 3, "positions": [45, 102, 298],
          "severity": "High", "monitoring": "cIEF charge variants, peptide map LC-MS"}, ...]
    """
    # Liability risk classification and monitoring recommendations
    PTM_RISK_MAP = {
        "Deamidation (NG)": {"severity": "High", "monitoring": "cIEF (acidic shift), peptide map LC-MS/MS"},
        "Deamidation (NS)": {"severity": "Medium", "monitoring": "cIEF (acidic shift), peptide map LC-MS/MS"},
        "Deamidation (NT)": {"severity": "Low", "monitoring": "Peptide map LC-MS/MS (slow kinetics)"},
        "Oxidation (Met)":  {"severity": "High", "monitoring": "RP-HPLC, peptide map; forced degradation (H₂O₂)"},
        "Oxidation (Trp)":  {"severity": "Medium", "monitoring": "UV spec, peptide map; light stress study"},
        "N-Glycosylation":  {"severity": "Medium", "monitoring": "HILIC-FLD, CE-LIF, LC-MS glycopeptide"},
        "Asp-Pro Clip":     {"severity": "High", "monitoring": "CE-SDS (fragment peaks), peptide map"},
        "Isomerization (DG)": {"severity": "Medium", "monitoring": "cIEF (acidic shift), RP-HPLC"},
        "Isomerization (DS)": {"severity": "Low", "monitoring": "Peptide map LC-MS/MS"},
    }

    # Collect all liability positions by motif type
    from collections import defaultdict
    motif_positions: Dict[str, List[int]] = defaultdict(list)

    for p in peptides:
        for liab in p.get("liabilities", []):
            motif_name = liab["motif"]
            # Convert peptide-local positions to protein-level positions
            for local_pos in liab.get("position_in_peptide", []):
                protein_pos = p["start"] + local_pos
                motif_positions[motif_name].append(protein_pos)

    # Build table rows
    table = []
    for motif_name in sorted(motif_positions.keys()):
        positions = sorted(set(motif_positions[motif_name]))
        risk_info = PTM_RISK_MAP.get(motif_name, {"severity": "Unknown", "monitoring": "TBD"})
        table.append({
            "motif": motif_name,
            "count": len(positions),
            "positions": positions,
            "positions_str": ", ".join(str(p) for p in positions[:10]) + ("..." if len(positions) > 10 else ""),
            "severity": risk_info["severity"],
            "monitoring": risk_info["monitoring"],
        })

    # Sort: High severity first
    severity_order = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
    table.sort(key=lambda x: (severity_order.get(x["severity"], 3), -x["count"]))

    return table


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
    print("  Analytical Twin v3.0 — Molecular Assembly + Glycan Profiles Test")
    print("=" * 60)

    # Test sequences
    hc_seq = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTY"
        "YADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGYDYWGQGTLVTVSSAST"
        "KGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGL"
        "YSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGP"
        "SVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYN"
        "STYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDE"
        "LTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRW"
        "QQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
    )
    lc_seq = (
        "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKLLIYAASSLQSGVP"
        "SRFSGSGSGTDFTLTISSLQPEDFATYYCLQHNSYPLTFGQGTRLEIKRTVAAPSVFIFP"
        "PSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTL"
        "TLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    )

    # -- M12: Stoichiometric assembly test (standard IgG1: 2xHC + 2xLC)
    print("\n--- M12: Stoichiometric Intact Mass (2xHC + 2xLC) ---")
    mab_chains = [
        {"sequence": hc_seq, "copy_number": 2, "name": "HC", "chain_type": "HC"},
        {"sequence": lc_seq, "copy_number": 2, "name": "LC", "chain_type": "LC"},
    ]
    intact = calculate_stoichiometric_intact_mass(mab_chains)
    print(f"  Assembly: {intact['stoichiometry_summary']}")
    print(f"  Total residues: {intact['total_assembled_length']}")
    print(f"  Bare mass: {intact['bare_mass_da']:.2f} Da ({intact['bare_mass_da']/1000:.1f} kDa)")
    print(f"  SS-corrected: {intact['disulfide_corrected_da']:.2f} Da")
    print(f"  Per chain:")
    for pc in intact["per_chain_masses"]:
        print(f"    {pc['name']}: {pc['chain_mass_da']:.2f} Da x{pc['copy_number']} = {pc['total_mass_da']:.2f} Da")
    if intact["glycoforms"]:
        print(f"  Glycoforms (top 3):")
        for gf in intact["glycoforms"][:3]:
            print(f"    {gf['name']}: {gf['intact_mass_da']:.2f} Da ({gf['abundance']})")

    # -- M12: Super-sequence
    print("\n--- M12: Super-Sequence ---")
    super_seq = build_super_sequence(mab_chains)
    print(f"  Length: {len(super_seq)} aa (expected: {len(hc_seq)*2 + len(lc_seq)*2})")

    # -- M12: Liability density
    print("\n--- M12: Liability Density ---")
    liab_d = calculate_liability_density(mab_chains)
    print(f"  Total residues: {liab_d['total_residues']}")
    print(f"  Total motifs: {liab_d['total_motifs']}")
    print(f"  Density: {liab_d['density_per_1000']:.1f} / 1000 residues")
    print(f"  Risk level: {liab_d['risk_level']}")
    for mtype, dens in sorted(liab_d["per_type_density"].items()):
        if dens > 0:
            print(f"    {mtype}: {dens:.1f}/1k ({liab_d['per_type_counts'][mtype]} total)")

    # -- M12: Full MS with chains
    print("\n--- M12: Full MS Characterization (stoichiometric) ---")
    ms = run_ms_characterization(
        protein_name="Adalimumab_IgG1",
        is_mab=True,
        chains=mab_chains,
    )
    print(f"  Status: {ms['status']}")
    print(f"  Assembly: {ms.get('assembly_mode', 'N/A')}")
    print(f"  Peptides: {ms['summary']['total_peptides']}")
    print(f"  With liabilities: {ms['summary']['peptides_with_liabilities']}")

    # Check source chain labels
    chain_labels = set(p.get("source_chain", "") for p in ms["peptide_map"])
    print(f"  Source chains in map: {chain_labels}")

    # DataFrame
    df = peptide_map_to_dataframe(ms["peptide_map"])
    if df is not None:
        print(f"  DataFrame: {len(df)} rows x {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        assert "Source Chain" in df.columns, "Missing Source Chain column!"

    # -- M13: Glycoform profile tests
    print("\n--- M13: Host Cell Glycoform Profiles ---")
    for profile_key, profile_info in HOST_CELL_GLYCOFORM_PROFILES.items():
        pi_shift = get_glycoform_pi_shift(profile_key)
        gf_list = get_glycoform_profile_for_ms(profile_key, n_glycosylation_sites=2, base_mass=143000.0)
        print(f"  {profile_info['name']}: pI shift={pi_shift}, glycoforms={len(gf_list)}")
        for gf in gf_list[:2]:
            print(f"    {gf['name']}: {gf['intact_mass_da']:,.1f} Da ({gf['abundance_pct']}%)")

    print("\n--- M13: Stoichiometric Mass with Glycoform Profile ---")
    for pkey in ("standard_cho", "highly_sialylated", "high_mannose"):
        intact_p = calculate_stoichiometric_intact_mass(
            mab_chains, is_mab=True, glycoform_profile=pkey,
        )
        top_gf = intact_p["glycoforms"][0] if intact_p["glycoforms"] else {}
        print(f"  {pkey}: top glycoform={top_gf.get('name', 'N/A')}, "
              f"mass={top_gf.get('intact_mass_da', 0):,.1f} Da, "
              f"pI shift={intact_p.get('glycoform_pi_shift', 0)}")

    print("\nAnalytical Twin v3.0 test complete")
