"""Tests for src/structure/mutation_enrich.py."""

import math

import pytest

from src.mutation.schema import AntibodyChain, MutationCandidate, ScoredMutation
from src.structure.mutation_enrich import enrich_mutations
from src.structure.schema import (
    AggregationPatch,
    InterfaceAnalysis,
    ParatopeProfile,
    StructureInput,
    StructureMetrics,
    StructureSource,
)


def _fake_metrics(available: bool = True) -> StructureMetrics:
    return StructureMetrics(
        input=StructureInput(
            pdb_path="/tmp/x.pdb",
            source=StructureSource.USER_SUPPLIED if available else StructureSource.UNAVAILABLE,
            confidence=0.9 if available else 0.0,
        ),
        vh_sasa=[10.0, 40.0, 80.0, 5.0, 120.0],   # positions 0..4
        vl_sasa=[50.0, 5.0, 60.0],                # positions 0..2
        vh_resnames=["A", "V", "L", "P", "F"],
        vl_resnames=["G", "K", "Y"],
        paratope=ParatopeProfile(
            cdr_residues=[2, 7],  # VH pos 2 and VL pos 2 (joint idx 5+2=7)
            chain_of_residue=["H", "L"],
            total_surface_area_a2=140.0,
            hydrophobic_fraction=0.5,
            charge_breakdown={"positive": 0, "negative": 0, "neutral": 2},
            flatness_proxy=0.8,
        ),
        interface=InterfaceAnalysis(
            buried_surface_area_a2=1800.0,
            packing_density=0.5,
            n_hbonds_across_interface=3,
            vh_residues_at_interface=[3, 4],  # PDB resnums (position 0 + 1)
            vl_residues_at_interface=[2],
        ),
        aggregation_patches=[
            AggregationPatch(
                center_residue=2,
                member_residues=[1, 2, 7],  # VH pos 1,2 and VL pos 2 (joint 7)
                total_hydrophobic_sasa_a2=210.0,
                patch_score=0.3,
                chain_of_residues=["H", "H", "L"],
            ),
        ],
        structure_risk_score=0.42,
        available=available,
    )


def _cand(chain, pos, wt, mt="A", region="framework"):
    return MutationCandidate(
        chain=chain, position=pos, wildtype_aa=wt, mutant_aa=mt, region=region,
    )


def _scored(chain, pos, wt, mt="A", region="framework", llr=0.5):
    c = _cand(chain, pos, wt, mt, region)
    return ScoredMutation(candidate=c, llr=llr, wildtype_prob=0.1, mutant_prob=0.3)


class TestEnrichAvailable:
    def test_sasa_at_position_vh(self):
        s = _scored(AntibodyChain.VH, 2, "L")
        enrich_mutations([s], _fake_metrics())
        assert s.extras["sasa_at_position"] == 80.0

    def test_sasa_at_position_vl(self):
        s = _scored(AntibodyChain.VL, 2, "Y")
        enrich_mutations([s], _fake_metrics())
        assert s.extras["sasa_at_position"] == 60.0

    def test_in_aggregation_patch_vh(self):
        # VH pos 2 is in the patch (joint idx 2).
        s = _scored(AntibodyChain.VH, 2, "L")
        enrich_mutations([s], _fake_metrics())
        assert s.extras["in_aggregation_patch"] is True

    def test_in_aggregation_patch_vl_via_joint_index(self):
        # VL pos 2 has joint idx 5+2=7, which is in the patch.
        s = _scored(AntibodyChain.VL, 2, "Y")
        enrich_mutations([s], _fake_metrics())
        assert s.extras["in_aggregation_patch"] is True

    def test_at_interface(self):
        # VH pos 3 corresponds to PDB resnum 4 → in vh interface list.
        s = _scored(AntibodyChain.VH, 3, "P")
        enrich_mutations([s], _fake_metrics())
        assert s.extras["is_at_interface"] is True

    def test_in_paratope_vl(self):
        # VL pos 2 → joint idx 7 → in paratope list.
        s = _scored(AntibodyChain.VL, 2, "Y")
        enrich_mutations([s], _fake_metrics())
        assert s.extras["is_in_paratope"] is True

    def test_rationale_flags_patch(self):
        s = _scored(AntibodyChain.VH, 2, "L")
        enrich_mutations([s], _fake_metrics())
        assert "aggregation patch" in s.rationale

    def test_structure_provenance_fields(self):
        s = _scored(AntibodyChain.VH, 0, "A", mt="G")
        enrich_mutations([s], _fake_metrics())
        assert s.extras["structure_source"] == "user_supplied"
        assert s.extras["structure_confidence"] == 0.9
        assert s.extras["parent_structure_risk_score"] == 0.42


class TestEnrichUnavailable:
    def test_sentinels_and_rationale(self):
        s = _scored(AntibodyChain.VH, 0, "A", mt="G")
        enrich_mutations([s], _fake_metrics(available=False))
        assert math.isnan(s.extras["sasa_at_position"])
        assert s.extras["is_at_interface"] is False
        assert s.extras["is_in_paratope"] is False
        assert s.extras["in_aggregation_patch"] is False
        assert "structure unavailable" in s.rationale
