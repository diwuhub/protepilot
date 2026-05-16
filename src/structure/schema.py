"""Structure analyzer schema.

Every dataclass here is frozen where feasible; mutation is explicit via
`dataclasses.replace`. The goal is that a `StructureMetrics` object can
be round-tripped to JSON for caching without type drift.

Provenance
----------
Every metric carries `source` (where the structure came from) and
`confidence` (a 0-1 quality score — for predicted structures this maps
from pLDDT/RMSD estimates; for experimental X-ray it is 1.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StructureSource(str, Enum):
    EXPERIMENTAL_PDB = "experimental_pdb"   # downloaded crystal structure
    IGFOLD = "igfold"                        # antibody-specialized predictor
    ESMFOLD = "esmfold"                      # general-purpose PLM folder
    SABPRED = "sabpred"                      # OPIG SAbPred suite (ABodyBuilder2)
    USER_SUPPLIED = "user_supplied"          # a PDB path the caller passed in
    UNAVAILABLE = "unavailable"              # couldn't produce a structure


@dataclass(frozen=True)
class StructureInput:
    """The provenance-carrying input to the analyzer."""

    pdb_path: str                            # absolute path on disk
    source: StructureSource                  # where it came from
    confidence: float = 1.0                  # 0..1, 1.0 for experimental
    vh_chain_id: str = "H"                   # PDB chain letter for VH
    vl_chain_id: str = "L"                   # PDB chain letter for VL
    sequence_hash: str = ""                  # optional content-addressed key
    pdb_id: str = ""                         # e.g. "1N8Z" for experimental
    notes: str = ""


@dataclass(frozen=True)
class ParatopeProfile:
    """Paratope characterization from CDR residues with SASA > exposure_cutoff."""

    cdr_residues: list[int]                  # 0-indexed positions within VH+VL joint
    chain_of_residue: list[str]              # parallel array: "H" or "L"
    total_surface_area_a2: float             # Å²  summed SASA of paratope residues
    hydrophobic_fraction: float              # fraction of paratope that's hydrophobic
    charge_breakdown: dict[str, int]         # {"positive": n, "negative": n, "neutral": n}
    flatness_proxy: float                    # shape-complementarity heuristic, 0..1


@dataclass(frozen=True)
class InterfaceAnalysis:
    """VH-VL domain-domain interface summary."""

    buried_surface_area_a2: float            # Å² buried at the VH-VL interface
    packing_density: float                   # 0..1; higher = tighter
    n_hbonds_across_interface: int           # inter-chain H-bonds (heuristic)
    vh_residues_at_interface: list[int]      # VH positions with SASA drop ≥ cutoff
    vl_residues_at_interface: list[int]
    notes: str = ""


@dataclass(frozen=True)
class AggregationPatch:
    """Spatial cluster of exposed hydrophobic residues (aggregation-prone region)."""

    center_residue: int                      # 0-indexed joint sequence position
    member_residues: list[int]
    total_hydrophobic_sasa_a2: float
    patch_score: float                       # 0..1 risk score (size × exposure × hydro)
    chain_of_residues: list[str]


@dataclass
class StructureMetrics:
    """Aggregated per-antibody structural metrics for downstream consumption."""

    input: StructureInput

    # Per-residue primary measurements (parallel arrays).
    vh_sasa: list[float] = field(default_factory=list)   # Å² per VH residue
    vl_sasa: list[float] = field(default_factory=list)
    vh_resnames: list[str] = field(default_factory=list)  # single-letter AAs
    vl_resnames: list[str] = field(default_factory=list)

    # Derived summaries.
    paratope: ParatopeProfile | None = None
    interface: InterfaceAnalysis | None = None
    aggregation_patches: list[AggregationPatch] = field(default_factory=list)

    # Structure-based TAP guideline metrics (Phase 3 upgrade of Phase 0 proxies).
    tap_psh: float = float("nan")            # Patches of Surface Hydrophobicity
    tap_ppc: float = float("nan")            # Patches of Positive Charge
    tap_pnc: float = float("nan")            # Patches of Negative Charge
    tap_sfvcsp: float = float("nan")         # Structural Fv Charge Symmetry Parameter
    tap_total_cdr_length: int = 0
    tap_risk_flag_count: int = 0
    # Full Lauer 2012 DI = -SFvCSP + β·SAP — populated when SASA is computed.
    di_full: float = float("nan")
    di_sap: float = float("nan")             # Spatial Aggregation Propensity proxy

    # Summary fields that downstream mutation-report consumers use.
    structure_risk_score: float = 0.0        # 0..1, composite risk
    available: bool = True
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializable dict — compatible with json.dump."""
        import dataclasses

        def _norm(v):
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                return {k: _norm(getattr(v, k)) for k in v.__dataclass_fields__}
            if isinstance(v, list):
                return [_norm(x) for x in v]
            if isinstance(v, dict):
                return {k: _norm(x) for k, x in v.items()}
            if isinstance(v, StructureSource):
                return v.value
            return v
        return {k: _norm(getattr(self, k)) for k in self.__dataclass_fields__}
