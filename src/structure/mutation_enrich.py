"""Enrich Phase 2 mutation reports with Phase 3 structure-derived annotations.

Given a list of `ScoredMutation` (Phase 2 output) and a `StructureMetrics`
(Phase 3 output), annotate each mutation with:
    - sasa_at_position     — per-residue SASA of the WT residue
    - is_at_interface      — True if the WT residue sits at the VH-VL interface
    - is_in_paratope       — True if the WT residue is part of the paratope
    - in_aggregation_patch — True if WT residue is a member of any patch

Also emits a parent-level `structure_risk_score` delta when called with a
mutant StructureMetrics (optional — requires actually rebuilding the
structure, which is expensive; deferred to Phase 4 integration).

Additive: does NOT modify `ScoredMutation`; sticks new fields into `extras`.
"""

from __future__ import annotations

from typing import Iterable

from src.mutation.schema import AntibodyChain, ScoredMutation
from src.structure.schema import StructureMetrics


def _residue_in_patch(joint_idx: int, metrics: StructureMetrics) -> bool:
    for p in metrics.aggregation_patches:
        if joint_idx in p.member_residues:
            return True
    return False


def _residue_at_interface(
    mutation: ScoredMutation, metrics: StructureMetrics
) -> bool:
    iface = metrics.interface
    if iface is None:
        return False
    # Interface residues are stored as PDB resnums (1-indexed in 1N8Z);
    # our ScoredMutation positions are 0-indexed within VH/VL. We do a
    # simple +1 offset here, which is correct for most antibody PDBs
    # that don't renumber. Known limitation: structures with non-standard
    # numbering (IMGT, Kabat) need proper alignment in Phase 4+.
    pdb_resnum = mutation.candidate.position + 1
    if mutation.candidate.chain is AntibodyChain.VH:
        return pdb_resnum in iface.vh_residues_at_interface
    return pdb_resnum in iface.vl_residues_at_interface


def _residue_in_paratope(
    mutation: ScoredMutation, metrics: StructureMetrics
) -> bool:
    if metrics.paratope is None:
        return False
    joint_idx = mutation.candidate.position
    if mutation.candidate.chain is AntibodyChain.VL:
        joint_idx += len(metrics.vh_resnames)
    return joint_idx in metrics.paratope.cdr_residues


def _sasa_at_position(
    mutation: ScoredMutation, metrics: StructureMetrics
) -> float:
    pos = mutation.candidate.position
    if mutation.candidate.chain is AntibodyChain.VH:
        if 0 <= pos < len(metrics.vh_sasa):
            return metrics.vh_sasa[pos]
    else:
        if 0 <= pos < len(metrics.vl_sasa):
            return metrics.vl_sasa[pos]
    return float("nan")


# ---------------------------------------------------------------------------


def enrich_mutations(
    scored: Iterable[ScoredMutation],
    metrics: StructureMetrics,
) -> list[ScoredMutation]:
    """Add structure-derived fields to ScoredMutation.extras; return list.

    Each scored mutation gets the following extras (all in-place, return
    is the same list object for convenience):
        sasa_at_position   (float Å²)
        is_at_interface    (bool)
        is_in_paratope     (bool)
        in_aggregation_patch (bool)
        structure_source   (string)
        structure_confidence (float)
        parent_structure_risk_score (float)  — copied for every row

    If the StructureMetrics is unavailable, every extra is set to a
    sentinel (NaN / False) and a note is appended to the rationale.
    """
    scored_list = list(scored)
    available = metrics.available

    for s in scored_list:
        if not available:
            s.extras.update({
                "sasa_at_position": float("nan"),
                "is_at_interface": False,
                "is_in_paratope": False,
                "in_aggregation_patch": False,
                "structure_source": metrics.input.source.value,
                "structure_confidence": metrics.input.confidence,
                "parent_structure_risk_score": float("nan"),
            })
            s.rationale = f"{s.rationale} | structure unavailable".strip(" |")
            continue

        joint_idx = s.candidate.position
        if s.candidate.chain is AntibodyChain.VL:
            joint_idx += len(metrics.vh_resnames)

        sasa = _sasa_at_position(s, metrics)
        at_iface = _residue_at_interface(s, metrics)
        in_para = _residue_in_paratope(s, metrics)
        in_patch = _residue_in_patch(joint_idx, metrics)

        s.extras.update({
            "sasa_at_position": float(sasa),
            "is_at_interface": bool(at_iface),
            "is_in_paratope": bool(in_para),
            "in_aggregation_patch": bool(in_patch),
            "structure_source": metrics.input.source.value,
            "structure_confidence": float(metrics.input.confidence),
            "parent_structure_risk_score": float(metrics.structure_risk_score),
        })
        # Gentle rationale for mutations at risky positions.
        risk_flags = []
        if in_patch:
            risk_flags.append("aggregation patch")
        if at_iface:
            risk_flags.append("VH-VL interface")
        if in_para:
            risk_flags.append("paratope")
        if risk_flags:
            s.rationale = f"{s.rationale} | at {' + '.join(risk_flags)}"

    return scored_list
