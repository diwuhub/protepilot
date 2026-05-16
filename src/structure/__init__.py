"""Structure analyzer for antibody developability features.

Consumes existing antibody PDB files (e.g. SAbDab downloads) and produces:
    - per-residue SASA + paratope identification
    - VH-VL interface buried surface area (BSA)
    - aggregation-prone hydrophobic patches (spatial clustering)
    - structure-based TAP / DI metric upgrades (replacing the Phase 0
      sequence-proxy versions)

When no PDB is on disk, a pluggable predictor (IgFold / SAbPred / ESMFold)
fills the gap; results are cached content-addressed at
    runs/structures/{sequence_hash}/

No MD. No ΔΔG. No claim of experimental-accuracy structure for CDR-H3.
"""

from src.structure.schema import (
    AggregationPatch,
    InterfaceAnalysis,
    ParatopeProfile,
    StructureInput,
    StructureMetrics,
)

__all__ = [
    "AggregationPatch",
    "InterfaceAnalysis",
    "ParatopeProfile",
    "StructureInput",
    "StructureMetrics",
]
