"""
feature_registry.py  ·  ProtePilot — Centralized Feature Store
=================================================================
Phase 1B: Single source of truth for all biophysical feature computation.

Every module in the platform should read features from this registry
instead of computing them independently. This eliminates:
- Duplicate computation (deamidation counted in 3+ places)
- Inconsistent algorithms (oxidation: M-only vs M+W in different modules)
- Scattered normalization (hydrophobicity: two different scales)

Architecture
------------
FeatureRegistry takes a sequence (+ optional chain info and molecule class)
and returns a FeatureSet — a structured, frozen object containing every
computed feature with its metadata (value, unit, normalization, provenance).

Each feature has a schema entry defining:
- name          : Canonical identifier used across all modules
- compute_fn    : The single authoritative computation function
- unit          : Physical unit (e.g., "kDa", "count", "score")
- normalization : How to normalize for ML input (if applicable)
- applicable_to : Which molecule classes this feature applies to
- in_ml_vector  : Whether this feature feeds the ML model
- in_report     : Whether this feature appears in reports

References
----------
- Kyte & Doolittle, J. Mol. Biol., 1982: Hydropathy scale
- Pace et al., Protein Science, 2009: pKa values for pI calculation
- Stanton et al., PNAS, 2017: mAb physicochemical properties
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger("ProtePilot.FeatureRegistry")


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

# Kyte-Doolittle hydropathy scale (canonical reference)
_KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Average amino acid mass (Daltons)
_AA_AVG_MASS_DA = 110.0

# Amino acid molecular weights (monoisotopic, for precise MW)
_AA_MASSES = {
    "A": 71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P": 97.05276, "S": 87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841,
}
_WATER_MASS = 18.01056  # Mass of water molecule for peptide bond accounting

# pKa values for pI calculation (Lehninger)
_PKA = {
    "N_term": 9.69, "C_term": 2.34,
    "D": 3.65, "E": 4.25, "C": 8.18,
    "Y": 10.07, "H": 6.00, "K": 10.54, "R": 12.48,
}

# Chou-Fasman beta-sheet propensity scale
_BETA_SHEET_PROPENSITY = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}


# ═══════════════════════════════════════════════════════════════════════
#  Feature Value Container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FeatureValue:
    """
    Single computed feature with full provenance.

    Attributes
    ----------
    name : str
        Canonical feature identifier (e.g., "pI", "deam_sites").
    value : Any
        The computed value (float, int, list, etc.).
    unit : str
        Physical unit (e.g., "kDa", "count", "", "score [0-1]").
    method : str
        Brief description of computation method.
    normalized : float or None
        ML-ready normalized value (0-1 range), if applicable.
    in_ml_vector : bool
        Whether this feature feeds the ML prediction model.
    """
    name: str = ""
    value: Any = None
    unit: str = ""
    method: str = ""
    normalized: Optional[float] = None
    in_ml_vector: bool = False


# ═══════════════════════════════════════════════════════════════════════
#  Feature Set — complete computed feature bundle
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FeatureSet:
    """
    All computed features for a single molecule.

    Access by name:  fs.get("pI").value
    Get ML vector:   fs.ml_vector()
    Get report dict: fs.report_dict()
    """
    features: Dict[str, FeatureValue] = field(default_factory=dict)
    sequence: str = ""
    molecule_class: str = "unknown"
    provenance: str = "feature_registry_v1"

    def get(self, name: str) -> Optional[FeatureValue]:
        """Get a feature by canonical name."""
        return self.features.get(name)

    def value(self, name: str, default: Any = None) -> Any:
        """Get a feature's raw value, with fallback."""
        fv = self.features.get(name)
        return fv.value if fv is not None else default

    def ml_vector(self) -> List[float]:
        """
        Build the ML feature vector from all features flagged in_ml_vector.

        Returns features in a fixed canonical order to ensure consistency
        across training and inference.
        """
        # Fixed order for ML vector — must match training data order
        ml_feature_order = [
            "pI", "mw_kda", "deam_sites", "ox_sites",
            "acidic_residues", "basic_residues", "hydrophobicity",
            # New features added in Phase 1B
            "beta_sheet_propensity", "cdr_hydrophobicity",
            "asp_isomerization_sites", "pyroglutamate_risk",
            "n_glycosylation_sites",
        ]
        vec = []
        for name in ml_feature_order:
            fv = self.features.get(name)
            if fv and fv.in_ml_vector and fv.normalized is not None:
                vec.append(fv.normalized)
            elif fv and fv.in_ml_vector and isinstance(fv.value, (int, float)):
                vec.append(float(fv.value))
        return vec

    def biophysical_7dim(self) -> List[float]:
        """
        Legacy 7-dim biophysical vector for backward compatibility.

        Order: [pI, MW_kDa, deam_sites, ox_sites, acidic_res, basic_res, hydrophobicity]
        This matches the existing developability_predictor.py BIOPHYS_NAMES.
        """
        names = ["pI", "mw_kda", "deam_sites", "ox_sites",
                 "acidic_residues", "basic_residues", "hydrophobicity"]
        return [float(self.value(n, 0.0)) for n in names]

    def report_dict(self) -> Dict[str, Any]:
        """Feature summary for report generation."""
        out = {}
        for name, fv in self.features.items():
            out[name] = {
                "value": fv.value,
                "unit": fv.unit,
                "method": fv.method,
            }
        return out

    def liability_summary(self) -> Dict[str, Any]:
        """Sequence liability summary for characterization panels."""
        return {
            "deamidation_sites": self.value("deam_sites", 0),
            "deamidation_motifs": self.value("deam_motifs", []),
            "oxidation_sites": self.value("ox_sites", 0),
            "oxidation_positions": self.value("ox_positions", []),
            "asp_isomerization_sites": self.value("asp_isomerization_sites", 0),
            "asp_isomerization_motifs": self.value("asp_isomerization_motifs", []),
            "n_glycosylation_sites": self.value("n_glycosylation_sites", 0),
            "n_glycosylation_motifs": self.value("n_glyco_motifs", []),
            "dp_clipping_sites": self.value("dp_sites", 0),
            "pyroglutamate_risk": self.value("pyroglutamate_risk", 0),
            "cysteine_count": self.value("cysteine_count", 0),
            "free_cysteine_risk": self.value("cysteine_count", 0) % 2 != 0,
        }

    def features_for_class(self) -> Dict[str, FeatureValue]:
        """Return only features applicable to this molecule's class.

        Class-aware filtering:
        - CDR hydrophobicity only applies to antibody-like molecules
        - Pyroglutamate risk is universal
        - N-glycosylation relevant for Fc-bearing molecules
        """
        from src.molecule_classifier import MoleculeClass
        try:
            mc = MoleculeClass(self.molecule_class)
        except ValueError:
            return dict(self.features)

        exclude = set()
        # CDR features only meaningful for antibody-like scaffolds
        if not mc.is_mab_like and mc != MoleculeClass.FC_FUSION:
            exclude.add("cdr_hydrophobicity")
        # N-glycosylation sites mostly relevant for Fc-bearing molecules
        # (still computed but flagged as not-in-ML for non-Fc)
        return {k: v for k, v in self.features.items() if k not in exclude}

    def ml_vector_for_class(self) -> List[float]:
        """Build ML feature vector filtered by molecule class.

        Falls back to ml_vector() for backward compatibility, but excludes
        features not applicable to the current molecule_class.
        """
        applicable = self.features_for_class()
        ml_feature_order = [
            "pI", "mw_kda", "deam_sites", "ox_sites",
            "acidic_residues", "basic_residues", "hydrophobicity",
            "beta_sheet_propensity", "cdr_hydrophobicity",
            "asp_isomerization_sites", "pyroglutamate_risk",
            "n_glycosylation_sites",
        ]
        vec = []
        for name in ml_feature_order:
            fv = applicable.get(name)
            if fv and fv.in_ml_vector and fv.normalized is not None:
                vec.append(fv.normalized)
            elif fv and fv.in_ml_vector and isinstance(fv.value, (int, float)):
                vec.append(float(fv.value))
        return vec

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization."""
        return {
            "molecule_class": self.molecule_class,
            "provenance": self.provenance,
            "features": {k: {"value": v.value, "unit": v.unit, "method": v.method}
                         for k, v in self.features.items()},
        }


# ═══════════════════════════════════════════════════════════════════════
#  Core Computation Functions (the single authoritative implementations)
# ═══════════════════════════════════════════════════════════════════════

def _compute_pI(seq: str) -> float:
    """
    Isoelectric point via bisection on Henderson-Hasselbalch.

    Primary: Biopython ProteinAnalysis (if available).
    Fallback: Manual bisection using Lehninger pKa values.
    """
    if not seq:
        return 7.0
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        return round(ProteinAnalysis(seq).isoelectric_point(), 2)
    except Exception:
        pass

    # Manual bisection fallback
    s = seq.upper()
    aa_counts = {aa: s.count(aa) for aa in _PKA if aa not in ("N_term", "C_term")}

    def _charge_at_ph(pH: float) -> float:
        charge = 1.0 / (1.0 + 10 ** (pH - _PKA["N_term"]))  # N-terminus (+)
        charge -= 1.0 / (1.0 + 10 ** (_PKA["C_term"] - pH))  # C-terminus (-)
        for aa, pka in _PKA.items():
            if aa in ("N_term", "C_term"):
                continue
            n = aa_counts.get(aa, 0)
            if n == 0:
                continue
            if aa in ("D", "E", "C", "Y"):  # Acidic
                charge -= n / (1.0 + 10 ** (pka - pH))
            else:  # Basic (H, K, R)
                charge += n / (1.0 + 10 ** (pH - pka))
        return charge

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _charge_at_ph(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.001:
            break
    return round((lo + hi) / 2.0, 2)


def _compute_mw_kda(seq: str) -> float:
    """
    Molecular weight in kDa from amino acid composition.

    Uses monoisotopic masses when available, falls back to average mass.
    """
    if not seq:
        return 0.0
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        return round(ProteinAnalysis(seq).molecular_weight() / 1000.0, 3)
    except Exception:
        pass
    s = seq.upper()
    mw_da = sum(_AA_MASSES.get(aa, _AA_AVG_MASS_DA) for aa in s)
    mw_da += _WATER_MASS  # Terminal water
    return round(mw_da / 1000.0, 3)


def _compute_gravy(seq: str) -> float:
    """
    GRAVY score (Kyte-Doolittle grand average of hydropathy).

    This is the single authoritative GRAVY computation.
    All other modules must use this function.
    """
    if not seq:
        return 0.0
    s = seq.upper()
    total = sum(_KD_SCALE.get(aa, 0.0) for aa in s)
    return round(total / max(len(s), 1), 4)


def _compute_hydrophobicity(gravy: float) -> float:
    """
    Normalized hydrophobicity [0, 1] from GRAVY score.

    Uses the canonical normalization:
        hydro = (GRAVY + 2.0) / 4.0, clamped to [0, 1]

    This is the ONLY normalization used platform-wide.
    The alternative (GRAVY + 4.5) / 9.0 is deprecated.
    """
    return max(0.0, min(1.0, (gravy + 2.0) / 4.0))


def _compute_charge_at_ph(seq: str, pH: float = 7.4) -> float:
    """Net charge at a given pH using Henderson-Hasselbalch."""
    if not seq:
        return 0.0
    s = seq.upper()
    charge = 1.0 / (1.0 + 10 ** (pH - _PKA["N_term"]))
    charge -= 1.0 / (1.0 + 10 ** (_PKA["C_term"] - pH))
    for aa, pka in _PKA.items():
        if aa in ("N_term", "C_term"):
            continue
        n = s.count(aa)
        if n == 0:
            continue
        if aa in ("D", "E", "C", "Y"):
            charge -= n / (1.0 + 10 ** (pka - pH))
        else:
            charge += n / (1.0 + 10 ** (pH - pka))
    return round(charge, 2)


def _count_deamidation_sites(seq: str) -> Tuple[int, List[Dict]]:
    """
    Count deamidation hotspots: N followed by G or S (NG, NS motifs).

    These are the highest-risk deamidation sites. N followed by other
    residues (except P) can also deamidate but at lower rates.
    """
    s = seq.upper()
    motifs = []
    for m in re.finditer(r"N[GS]", s):
        motifs.append({"position": m.start(), "motif": m.group()})
    return len(motifs), motifs


def _count_oxidation_sites(seq: str) -> Tuple[int, List[int]]:
    """
    Count oxidation-susceptible residues: Met (M) and Trp (W).

    Both Met and Trp are susceptible to oxidation under stress.
    This is the canonical definition used platform-wide.
    (Previous inconsistency: some modules counted M only.)
    """
    s = seq.upper()
    positions = [i for i, aa in enumerate(s) if aa in ("M", "W")]
    return len(positions), positions


def _count_asp_isomerization_sites(seq: str) -> Tuple[int, List[Dict]]:
    """
    Count Asp isomerization hotspots: DG, DS motifs.

    Asp followed by Gly or Ser undergoes isomerization to iso-Asp,
    which can affect potency and charge heterogeneity.
    """
    s = seq.upper()
    motifs = []
    for m in re.finditer(r"D[GS]", s):
        motifs.append({"position": m.start(), "motif": m.group()})
    return len(motifs), motifs


def _count_n_glycosylation_sites(seq: str) -> Tuple[int, List[Dict]]:
    """
    Count N-glycosylation sequons: N-X-S/T where X != P.

    The canonical N-linked glycosylation motif. Only Asn in this
    context is glycosylated. Pro at X blocks glycosylation.
    """
    s = seq.upper()
    motifs = []
    for i in range(len(s) - 2):
        if s[i] == "N" and s[i + 1] != "P" and s[i + 2] in ("S", "T"):
            motifs.append({"position": i, "motif": s[i:i + 3]})
    return len(motifs), motifs


def _count_dp_clipping_sites(seq: str) -> int:
    """Count Asp-Pro (DP) acid-labile clipping sites."""
    return len(re.findall(r"DP", seq.upper()))


def _assess_pyroglutamate_risk(seq: str) -> int:
    """
    Pyroglutamate formation risk from N-terminal Gln (Q) or Glu (E).

    N-terminal Q cyclizes to pyroglutamate almost quantitatively.
    N-terminal E can also cyclize but at lower rates.
    Returns: 0 (no risk), 1 (E, moderate), 2 (Q, high)
    """
    if not seq:
        return 0
    first_aa = seq.upper()[0]
    if first_aa == "Q":
        return 2  # High risk
    elif first_aa == "E":
        return 1  # Moderate risk
    return 0


def _compute_beta_sheet_propensity(seq: str) -> float:
    """
    Average Chou-Fasman beta-sheet propensity score.

    Higher values indicate greater tendency to form beta-sheet structures,
    which correlates with aggregation risk (beta-sheet-mediated aggregation
    is a major degradation pathway for biologics).

    Reference: Chou & Fasman, Biochemistry, 1974
    """
    if not seq:
        return 1.0
    s = seq.upper()
    total = sum(_BETA_SHEET_PROPENSITY.get(aa, 1.0) for aa in s)
    return round(total / max(len(s), 1), 4)


def _compute_cdr_hydrophobicity(
    seq: str,
    chain_type: str = "HC",
) -> Optional[float]:
    """
    Average GRAVY of CDR regions (heuristic identification).

    CDR hydrophobicity is one of the strongest predictors of
    aggregation propensity in antibodies. High CDR hydrophobicity
    (> 0.0 GRAVY) is a significant risk flag.

    Returns None if CDRs cannot be identified.
    """
    s = seq.upper()
    if len(s) < 100:
        return None

    # Normalize chain_type: accept "Heavy"→"HC", "Light"→"LC" (training path compat)
    _ct = chain_type.upper().replace(" ", "")
    if _ct in ("HEAVY",) or _ct.startswith("HEAVY"):
        chain_type = "HC"
    elif _ct in ("LIGHT",) or _ct.startswith("LIGHT"):
        chain_type = "LC"

    # Heuristic CDR extraction based on conserved Cys positions
    cys_positions = [i for i, aa in enumerate(s) if aa == "C"]
    if len(cys_positions) < 2:
        return None

    cdr_residues = []
    if chain_type == "HC" and len(s) > 120:
        # CDR-H1: ~positions 26-35, CDR-H2: ~50-65, CDR-H3: ~95-102
        # Use Cys anchors: CDR-H1 near Cys1+4 to +15
        c1 = cys_positions[0]
        cdr_residues.extend(list(s[c1 + 4: c1 + 16]))
        # CDR-H2: look for WVRQ/WYRQ/WIRQ anchor
        for trp_pos in (i for i, aa in enumerate(s) if aa == "W"):
            if trp_pos > c1 and trp_pos < len(s) - 30:
                cdr_residues.extend(list(s[trp_pos + 15: trp_pos + 30]))
                break
        # CDR-H3: between Cys2 and end of V-region
        if len(cys_positions) >= 2:
            c2 = cys_positions[1]
            cdr_residues.extend(list(s[c2 + 3: c2 + 15]))
    elif chain_type == "LC" and len(s) > 90:
        c1 = cys_positions[0]
        cdr_residues.extend(list(s[c1 + 1: c1 + 18]))
        cdr_residues.extend(list(s[49:57] if len(s) > 57 else []))
        if len(cys_positions) >= 2:
            c2 = cys_positions[1]
            cdr_residues.extend(list(s[c2 + 1: c2 + 11]))

    if not cdr_residues:
        return None

    total = sum(_KD_SCALE.get(aa, 0.0) for aa in cdr_residues)
    return round(total / max(len(cdr_residues), 1), 4)


# ═══════════════════════════════════════════════════════════════════════
#  The Registry: compute_all_features
# ═══════════════════════════════════════════════════════════════════════

def compute_features(
    sequence: str,
    molecule_class: str = "unknown",
    chains: Optional[List[Dict[str, Any]]] = None,
    glycan_mass_per_site_da: float = 2400.0,
    n_glycan_sites_per_hc: int = 2,
) -> FeatureSet:
    """
    Compute all biophysical features for a molecule — THE single entry point.

    Every module in the platform should call this function instead of
    computing features independently. The result is cached in the
    workspace/session state and shared across all modules.

    Parameters
    ----------
    sequence : str
        Full amino acid sequence (concatenated if multi-chain).
    molecule_class : str
        Classification from MoleculeClassifier (e.g., "canonical_mab").
    chains : list of dict, optional
        Individual chain info: [{sequence, chain_type, copy_number, name}]
    glycan_mass_per_site_da : float
        Glycan mass per N-glycan site (Da). Default: 2400 (standard CHO).
    n_glycan_sites_per_hc : int
        N-glycan sites per heavy chain. Default: 2 (N297 + potential Fab site).

    Returns
    -------
    FeatureSet
        Complete feature bundle with raw values, normalized values,
        and ML vector generation capability.
    """
    fs = FeatureSet(sequence=sequence, molecule_class=molecule_class)
    seq = (sequence or "").strip().upper()
    seq_len = len(seq)

    if seq_len == 0:
        return fs

    # ── Core biophysical features (the original 7) ────────────────

    # For multi-chain molecules (e.g., IgG = 2×HC + 2×LC), compute pI on the
    # stoichiometrically assembled sequence. pI depends on ALL ionizable residues
    # in the intact molecule, so a single-chain calculation is physically wrong.
    # Reference: Pace et al., Protein Sci. 2009; typical IgG pI ~8.0-9.5.
    if chains and len(chains) > 0:
        assembled_seq = ""
        for ch in chains:
            ch_seq = ch.get("sequence", "")
            ch_copy = ch.get("copy_number", 1)
            assembled_seq += ch_seq.upper() * ch_copy
        pI_seq = assembled_seq if assembled_seq else seq
    else:
        pI_seq = seq
    pI = _compute_pI(pI_seq)
    fs.features["pI"] = FeatureValue(
        name="pI", value=pI, unit="",
        method="Henderson-Hasselbalch bisection on stoichiometric assembly (Biopython primary, Lehninger pKa fallback)",
        normalized=pI / 14.0,  # Normalized to [0, 1] over pH range
        in_ml_vector=True,
    )
    fs.features["pI_per_chain"] = FeatureValue(
        name="pI_per_chain", value=_compute_pI(seq), unit="",
        method="Single-chain pI (for reference only)",
        in_ml_vector=False,
    )

    mw = _compute_mw_kda(seq)
    # For multi-chain assemblies, compute stoichiometric MW
    stoich_mw = mw
    if chains:
        total_da = 0.0
        n_hc = 0
        for ch in chains:
            ch_seq = ch.get("sequence", "")
            ch_copy = ch.get("copy_number", 1)
            ch_mw = _compute_mw_kda(ch_seq) * 1000.0  # Back to Da
            total_da += ch_mw * ch_copy
            ct_upper = ch.get("chain_type", "").upper()
            if ct_upper in ("HC", "HEAVY") or ct_upper.startswith("HEAVY"):
                n_hc += ch_copy
        # Add glycan mass for Fc-containing molecules
        from src.molecule_classifier import MoleculeClass
        mc = MoleculeClass(molecule_class) if molecule_class != "unknown" else MoleculeClass.UNKNOWN
        if mc.expects_glycosylation and n_hc > 0:
            total_da += n_hc * n_glycan_sites_per_hc * glycan_mass_per_site_da
        if total_da > 0:
            stoich_mw = round(total_da / 1000.0, 3)

    fs.features["mw_kda"] = FeatureValue(
        name="mw_kda", value=stoich_mw, unit="kDa",
        method="Monoisotopic residue masses + stoichiometric assembly + glycan correction",
        normalized=min(stoich_mw / 250.0, 1.0),  # Normalized assuming max ~250 kDa
        in_ml_vector=True,
    )
    fs.features["mw_kda_per_chain"] = FeatureValue(
        name="mw_kda_per_chain", value=mw, unit="kDa",
        method="Single-chain MW (no stoichiometry)",
        in_ml_vector=False,
    )

    gravy = _compute_gravy(seq)
    fs.features["gravy"] = FeatureValue(
        name="gravy", value=gravy, unit="",
        method="Kyte-Doolittle grand average of hydropathy",
        in_ml_vector=False,  # hydrophobicity (normalized) is used instead
    )

    hydro = _compute_hydrophobicity(gravy)
    fs.features["hydrophobicity"] = FeatureValue(
        name="hydrophobicity", value=hydro, unit="score [0-1]",
        method="Normalized GRAVY: (GRAVY + 2.0) / 4.0, clamped [0, 1]",
        normalized=hydro,
        in_ml_vector=True,
    )

    # Motif-based features: compute per-chain when available to avoid
    # cross-boundary false positives (e.g., chain1 ends 'N' + chain2
    # starts 'G' would create a spurious NG deamidation motif).
    if chains and len(chains) > 0:
        deam_count = 0
        deam_motifs = []
        ox_count = 0
        ox_positions = []
        _offset = 0
        for ch in chains:
            ch_seq = ch.get("sequence", "")
            ch_copy = ch.get("copy_number", 1)
            _dc, _dm = _count_deamidation_sites(ch_seq)
            deam_count += _dc * ch_copy
            for m in _dm:
                deam_motifs.append({"position": m["position"] + _offset, "motif": m["motif"]})
            _oc, _op = _count_oxidation_sites(ch_seq)
            ox_count += _oc * ch_copy
            ox_positions.extend([p + _offset for p in _op])
            _offset += len(ch_seq) * ch_copy
    else:
        deam_count, deam_motifs = _count_deamidation_sites(seq)
        ox_count, ox_positions = _count_oxidation_sites(seq)

    fs.features["deam_sites"] = FeatureValue(
        name="deam_sites", value=deam_count, unit="count",
        method="Regex count of N[GS] motifs (per-chain to avoid boundary artifacts)",
        normalized=min(deam_count / 10.0, 1.0),
        in_ml_vector=True,
    )
    fs.features["deam_motifs"] = FeatureValue(
        name="deam_motifs", value=deam_motifs, unit="list",
        method="Positional NG/NS motif annotation",
        in_ml_vector=False,
    )

    fs.features["ox_sites"] = FeatureValue(
        name="ox_sites", value=ox_count, unit="count",
        method="Count of Met (M) + Trp (W) residues (per-chain with copy_number)",
        normalized=min(ox_count / 20.0, 1.0),
        in_ml_vector=True,
    )
    fs.features["ox_positions"] = FeatureValue(
        name="ox_positions", value=ox_positions, unit="list",
        method="Positions of Met and Trp residues",
        in_ml_vector=False,
    )

    acidic = seq.count("D") + seq.count("E")
    fs.features["acidic_residues"] = FeatureValue(
        name="acidic_residues", value=acidic, unit="count",
        method="Sum of Asp (D) + Glu (E) residues",
        normalized=min(acidic / 100.0, 1.0),
        in_ml_vector=True,
    )

    basic = seq.count("K") + seq.count("R") + seq.count("H")
    fs.features["basic_residues"] = FeatureValue(
        name="basic_residues", value=basic, unit="count",
        method="Sum of Lys (K) + Arg (R) + His (H) residues",
        normalized=min(basic / 100.0, 1.0),
        in_ml_vector=True,
    )

    # ── New features (Phase 1B additions) ─────────────────────────

    charge = _compute_charge_at_ph(seq, 7.4)
    fs.features["net_charge_pH7.4"] = FeatureValue(
        name="net_charge_pH7.4", value=charge, unit="e",
        method="Henderson-Hasselbalch net charge at pH 7.4",
        in_ml_vector=False,
    )

    cys_count = seq.count("C")
    fs.features["cysteine_count"] = FeatureValue(
        name="cysteine_count", value=cys_count, unit="count",
        method="Count of Cys (C) residues for disulfide bond inventory",
        in_ml_vector=False,
    )

    beta_prop = _compute_beta_sheet_propensity(seq)
    fs.features["beta_sheet_propensity"] = FeatureValue(
        name="beta_sheet_propensity", value=beta_prop, unit="score",
        method="Chou-Fasman average beta-sheet propensity (>1.1 = beta-prone, aggregation risk)",
        normalized=(beta_prop - 0.5) / 1.5,  # Normalize ~[0.5, 2.0] → [0, 1]
        in_ml_vector=True,
    )

    # Asp isomerization, N-glycosylation, DP clipping: per-chain to avoid
    # cross-boundary false positives (same rationale as deam/ox above).
    if chains and len(chains) > 0:
        iso_count = 0
        iso_motifs = []
        nglyco_count = 0
        nglyco_motifs = []
        dp_count = 0
        pyro = 0
        _offset = 0
        for ch in chains:
            ch_seq = ch.get("sequence", "")
            ch_copy = ch.get("copy_number", 1)
            _ic, _im = _count_asp_isomerization_sites(ch_seq)
            iso_count += _ic * ch_copy
            iso_motifs.extend([{"position": m["position"] + _offset, "motif": m["motif"]} for m in _im])
            _nc, _nm = _count_n_glycosylation_sites(ch_seq)
            nglyco_count += _nc * ch_copy
            nglyco_motifs.extend([{"position": m["position"] + _offset, "motif": m["motif"]} for m in _nm])
            dp_count += _count_dp_clipping_sites(ch_seq) * ch_copy
            # Pyroglutamate: check each chain's N-terminus, take max risk
            pyro = max(pyro, _assess_pyroglutamate_risk(ch_seq))
            _offset += len(ch_seq) * ch_copy
    else:
        iso_count, iso_motifs = _count_asp_isomerization_sites(seq)
        nglyco_count, nglyco_motifs = _count_n_glycosylation_sites(seq)
        dp_count = _count_dp_clipping_sites(seq)
        pyro = _assess_pyroglutamate_risk(seq)

    fs.features["asp_isomerization_sites"] = FeatureValue(
        name="asp_isomerization_sites", value=iso_count, unit="count",
        method="Regex count of D[GS] motifs (per-chain to avoid boundary artifacts)",
        normalized=min(iso_count / 8.0, 1.0),
        in_ml_vector=True,
    )
    fs.features["asp_isomerization_motifs"] = FeatureValue(
        name="asp_isomerization_motifs", value=iso_motifs, unit="list",
        method="Positional DG/DS motif annotation",
        in_ml_vector=False,
    )

    pyro_method = (
        "Per-chain N-terminal Q/E pyroglutamate risk (max across chains)"
        if chains else "N-terminal Q → pyroGlu (risk=2), E → pyroGlu (risk=1), other (risk=0)"
    )
    fs.features["pyroglutamate_risk"] = FeatureValue(
        name="pyroglutamate_risk", value=pyro, unit="score [0-2]",
        method=pyro_method,
        normalized=pyro / 2.0,
        in_ml_vector=True,
    )

    fs.features["n_glycosylation_sites"] = FeatureValue(
        name="n_glycosylation_sites", value=nglyco_count, unit="count",
        method="Count of N-X-S/T sequons (per-chain to avoid boundary artifacts)",
        normalized=min(nglyco_count / 6.0, 1.0),
        in_ml_vector=True,
    )
    fs.features["n_glyco_motifs"] = FeatureValue(
        name="n_glyco_motifs", value=nglyco_motifs, unit="list",
        method="Positional NxS/T motif annotation",
        in_ml_vector=False,
    )

    fs.features["dp_sites"] = FeatureValue(
        name="dp_sites", value=dp_count, unit="count",
        method="Count of Asp-Pro (DP) acid-labile clipping sites (per-chain with copy_number)",
        in_ml_vector=False,
    )

    # For multi-chain assemblies, compute stoichiometric seq_length
    # (sum of chain_length × copy_number) to match MW calculation.
    stoich_seq_len = seq_len
    if chains:
        _asm_len = sum(
            len(ch.get("sequence", "")) * ch.get("copy_number", 1)
            for ch in chains
        )
        if _asm_len > 0:
            stoich_seq_len = _asm_len
    fs.features["seq_length"] = FeatureValue(
        name="seq_length", value=stoich_seq_len, unit="aa",
        method="Total residue count (stoichiometric assembly with copy_number)",
        normalized=min(stoich_seq_len / 1400.0, 1.0),
        in_ml_vector=False,
    )

    # ── CDR hydrophobicity (antibody-specific) ────────────────────
    # Compute per-chain CDR hydrophobicity if chains are available
    cdr_hydro_values = []
    if chains:
        for ch in chains:
            ch_seq = ch.get("sequence", "").upper()
            ch_type = ch.get("chain_type", "HC")
            cdr_h = _compute_cdr_hydrophobicity(ch_seq, ch_type)
            if cdr_h is not None:
                cdr_hydro_values.append(cdr_h)

    if not cdr_hydro_values and seq_len > 100:
        # Fallback: try on full sequence assuming HC
        cdr_h = _compute_cdr_hydrophobicity(seq, "HC")
        if cdr_h is not None:
            cdr_hydro_values.append(cdr_h)

    cdr_hydro_avg = (
        round(sum(cdr_hydro_values) / len(cdr_hydro_values), 4)
        if cdr_hydro_values else None
    )
    fs.features["cdr_hydrophobicity"] = FeatureValue(
        name="cdr_hydrophobicity", value=cdr_hydro_avg, unit="GRAVY",
        method="Average Kyte-Doolittle GRAVY of heuristically identified CDR regions",
        normalized=_compute_hydrophobicity(cdr_hydro_avg) if cdr_hydro_avg is not None else 0.5,
        in_ml_vector=True,
    )

    # ── Extinction coefficient ────────────────────────────────────
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        ec = ProteinAnalysis(seq).molar_extinction_coefficient()
        fs.features["extinction_coeff_reduced"] = FeatureValue(
            name="extinction_coeff_reduced", value=ec[0], unit="M⁻¹cm⁻¹",
            method="Biopython: Trp + Tyr absorbance at 280 nm (reduced)",
            in_ml_vector=False,
        )
        fs.features["extinction_coeff_cystines"] = FeatureValue(
            name="extinction_coeff_cystines", value=ec[1], unit="M⁻¹cm⁻¹",
            method="Biopython: Trp + Tyr + disulfide absorbance at 280 nm",
            in_ml_vector=False,
        )
    except Exception:
        pass

    log.debug(
        "FeatureRegistry: computed %d features for %s (%d aa, class=%s)",
        len(fs.features), molecule_class, seq_len, molecule_class,
    )
    return fs


# ═══════════════════════════════════════════════════════════════════════
#  SelfTest
# ═══════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    """Smoke test for feature computation consistency."""
    # NISTmAb VH sequence fragment
    test_seq = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARI"
        "YPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGD"
        "GFYAMDYWGQGTLVTVSS"
    )

    fs = compute_features(test_seq, molecule_class="canonical_mab")

    # Verify core features exist and are reasonable
    assert fs.value("pI") is not None, "pI not computed"
    assert 4.0 < fs.value("pI") < 12.0, f"pI out of range: {fs.value('pI')}"

    assert fs.value("mw_kda") is not None, "MW not computed"
    assert fs.value("mw_kda") > 0, f"MW not positive: {fs.value('mw_kda')}"

    assert fs.value("hydrophobicity") is not None, "Hydrophobicity not computed"
    assert 0.0 <= fs.value("hydrophobicity") <= 1.0, f"Hydro out of range: {fs.value('hydrophobicity')}"

    assert fs.value("deam_sites") is not None, "Deamidation sites not computed"
    assert fs.value("ox_sites") is not None, "Oxidation sites not computed"
    assert fs.value("beta_sheet_propensity") is not None, "Beta-sheet propensity not computed"

    # Verify legacy 7-dim vector
    vec7 = fs.biophysical_7dim()
    assert len(vec7) == 7, f"Legacy vector wrong length: {len(vec7)}"

    # Verify ML vector (should be > 7 with new features)
    ml_vec = fs.ml_vector()
    assert len(ml_vec) >= 7, f"ML vector too short: {len(ml_vec)}"

    # Verify liability summary
    liab = fs.liability_summary()
    assert "deamidation_sites" in liab
    assert "oxidation_sites" in liab
    assert "asp_isomerization_sites" in liab

    # Verify GRAVY consistency
    gravy = fs.value("gravy")
    hydro = fs.value("hydrophobicity")
    expected_hydro = max(0.0, min(1.0, (gravy + 2.0) / 4.0))
    assert abs(hydro - expected_hydro) < 0.001, "Hydrophobicity/GRAVY inconsistency"

    log.info("FeatureRegistry selftest PASSED")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
    print("All feature_registry tests passed.")
