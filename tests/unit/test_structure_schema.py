"""Tests for src/structure/schema.py."""

import json

import pytest

from src.structure.schema import (
    AggregationPatch,
    InterfaceAnalysis,
    ParatopeProfile,
    StructureInput,
    StructureMetrics,
    StructureSource,
)


class TestStructureInput:
    def test_defaults(self):
        s = StructureInput(pdb_path="/tmp/x.pdb", source=StructureSource.EXPERIMENTAL_PDB)
        assert s.confidence == 1.0
        assert s.vh_chain_id == "H"
        assert s.vl_chain_id == "L"


class TestStructureMetrics:
    def test_empty_dict_roundtrip(self):
        m = StructureMetrics(
            input=StructureInput(pdb_path="/tmp/x.pdb", source=StructureSource.UNAVAILABLE),
            available=False,
        )
        d = m.to_dict()
        assert d["available"] is False
        assert d["input"]["source"] == "unavailable"
        # json-serializable:
        s = json.dumps(d)
        assert "unavailable" in s

    def test_full_dict_roundtrip(self):
        para = ParatopeProfile(
            cdr_residues=[1, 2, 3],
            chain_of_residue=["H", "H", "L"],
            total_surface_area_a2=123.4,
            hydrophobic_fraction=0.5,
            charge_breakdown={"positive": 1, "negative": 2, "neutral": 0},
            flatness_proxy=0.7,
        )
        iface = InterfaceAnalysis(
            buried_surface_area_a2=1800.0,
            packing_density=0.6,
            n_hbonds_across_interface=4,
            vh_residues_at_interface=[44, 45, 47],
            vl_residues_at_interface=[36, 38],
        )
        patch = AggregationPatch(
            center_residue=50,
            member_residues=[48, 50, 52],
            total_hydrophobic_sasa_a2=180.0,
            patch_score=0.4,
            chain_of_residues=["H", "H", "H"],
        )
        m = StructureMetrics(
            input=StructureInput(pdb_path="/x", source=StructureSource.EXPERIMENTAL_PDB),
            vh_sasa=[10.0, 20.0, 30.0],
            vl_sasa=[5.0, 15.0],
            vh_resnames=["A", "C", "D"],
            vl_resnames=["E", "F"],
            paratope=para,
            interface=iface,
            aggregation_patches=[patch],
            tap_psh=1.0, tap_ppc=2.0, tap_pnc=1.0,
            tap_sfvcsp=0.3, tap_total_cdr_length=50,
            tap_risk_flag_count=1,
            di_full=-0.2, di_sap=2.0,
            structure_risk_score=0.3,
        )
        d = m.to_dict()
        s = json.dumps(d)
        restored = json.loads(s)
        assert restored["paratope"]["total_surface_area_a2"] == 123.4
        assert restored["interface"]["buried_surface_area_a2"] == 1800.0
        assert restored["aggregation_patches"][0]["patch_score"] == 0.4


class TestEnums:
    def test_all_structure_sources(self):
        expected = {"experimental_pdb", "igfold", "esmfold", "sabpred", "user_supplied", "unavailable"}
        assert {s.value for s in StructureSource} == expected
