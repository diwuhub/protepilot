"""Tests for src/structure/analyzer.py — both synthetic and real Trastuzumab."""

from pathlib import Path

import pytest

from src.structure.analyzer import DEFAULT_EXPOSURE_CUTOFF, analyze_structure
from src.structure.schema import StructureInput, StructureSource

# --- synthetic PDB: two short chains (H and L) ---------------------------
SYNTHETIC_PDB = """\
ATOM      1  N   ALA H   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA H   1      11.000  10.500  10.500  1.00 20.00           C
ATOM      3  C   ALA H   1      12.000  11.000  10.000  1.00 20.00           C
ATOM      4  O   ALA H   1      12.500  11.500  10.500  1.00 20.00           O
ATOM      5  CB  ALA H   1      11.300  11.100  11.200  1.00 20.00           C
ATOM      6  N   VAL H   2      13.000  11.500  10.000  1.00 20.00           N
ATOM      7  CA  VAL H   2      14.000  12.000  10.500  1.00 20.00           C
ATOM      8  C   VAL H   2      15.000  12.500  10.000  1.00 20.00           C
ATOM      9  O   VAL H   2      15.500  13.000  10.500  1.00 20.00           O
ATOM     10  CB  VAL H   2      14.300  12.600  11.200  1.00 20.00           C
ATOM     11  N   LEU H   3      16.000  13.000  10.000  1.00 20.00           N
ATOM     12  CA  LEU H   3      17.000  13.500  10.500  1.00 20.00           C
ATOM     13  C   LEU H   3      18.000  14.000  10.000  1.00 20.00           C
ATOM     14  O   LEU H   3      18.500  14.500  10.500  1.00 20.00           O
ATOM     15  CB  LEU H   3      17.300  14.100  11.200  1.00 20.00           C
ATOM     16  N   GLY L   1      10.000  20.000  10.000  1.00 20.00           N
ATOM     17  CA  GLY L   1      11.000  20.500  10.500  1.00 20.00           C
ATOM     18  C   GLY L   1      12.000  21.000  10.000  1.00 20.00           C
ATOM     19  O   GLY L   1      12.500  21.500  10.500  1.00 20.00           O
ATOM     20  N   PHE L   2      13.000  21.500  10.000  1.00 20.00           N
ATOM     21  CA  PHE L   2      14.000  22.000  10.500  1.00 20.00           C
ATOM     22  C   PHE L   2      15.000  22.500  10.000  1.00 20.00           C
ATOM     23  O   PHE L   2      15.500  23.000  10.500  1.00 20.00           O
ATOM     24  CB  PHE L   2      14.300  22.600  11.200  1.00 20.00           C
ATOM     25  N   THR L   3      16.000  23.000  10.000  1.00 20.00           N
ATOM     26  CA  THR L   3      17.000  23.500  10.500  1.00 20.00           C
ATOM     27  C   THR L   3      18.000  24.000  10.000  1.00 20.00           C
ATOM     28  O   THR L   3      18.500  24.500  10.500  1.00 20.00           O
END
"""

TRASTUZUMAB_PDB_PATH = Path("/Users/di/Projects/ProtePilot/data/structures/1n8z.pdb")


class TestSyntheticPdb:
    @pytest.fixture
    def synthetic_input(self, tmp_path):
        p = tmp_path / "tiny.pdb"
        p.write_text(SYNTHETIC_PDB)
        return StructureInput(
            pdb_path=str(p), source=StructureSource.USER_SUPPLIED,
            vh_chain_id="H", vl_chain_id="L",
        )

    def test_produces_metrics(self, synthetic_input):
        m = analyze_structure(synthetic_input)
        assert m.available
        assert m.vh_resnames == ["A", "V", "L"]
        assert m.vl_resnames == ["G", "F", "T"]

    def test_sasa_nonzero(self, synthetic_input):
        m = analyze_structure(synthetic_input)
        assert all(s > 0 for s in m.vh_sasa)
        assert all(s > 0 for s in m.vl_sasa)

    def test_interface_computed(self, synthetic_input):
        m = analyze_structure(synthetic_input)
        # Two chains too far apart → BSA near zero, but not None.
        assert m.interface is not None
        assert m.interface.buried_surface_area_a2 >= 0

    def test_structure_risk_score_in_range(self, synthetic_input):
        m = analyze_structure(synthetic_input)
        assert 0.0 <= m.structure_risk_score <= 1.0


class TestMissingPdb:
    def test_unavailable_input_returns_unavailable(self):
        inp = StructureInput(pdb_path="", source=StructureSource.UNAVAILABLE, notes="test")
        m = analyze_structure(inp)
        assert not m.available
        assert "test" in m.notes

    def test_missing_file_returns_unavailable(self, tmp_path):
        inp = StructureInput(
            pdb_path=str(tmp_path / "nope.pdb"),
            source=StructureSource.USER_SUPPLIED,
        )
        m = analyze_structure(inp)
        assert not m.available

    def test_parse_failure_returns_unavailable(self, tmp_path):
        p = tmp_path / "garbage.pdb"
        p.write_text("this is not a pdb at all")
        inp = StructureInput(pdb_path=str(p), source=StructureSource.USER_SUPPLIED)
        m = analyze_structure(inp)
        # Some BioPython versions accept anything; check either unavailable
        # or an empty vh_resnames result.
        if m.available:
            assert m.vh_resnames == []
        else:
            assert "parse" in m.notes.lower() or "no VH chain" in m.notes


@pytest.mark.skipif(
    not TRASTUZUMAB_PDB_PATH.exists(),
    reason=f"Trastuzumab 1N8Z fixture not at {TRASTUZUMAB_PDB_PATH}",
)
class TestTrastuzumab1N8Z:
    """Integration: real Trastuzumab Fab structure. Covers milestone checks."""

    @pytest.fixture
    def t_input(self):
        return StructureInput(
            pdb_path=str(TRASTUZUMAB_PDB_PATH),
            source=StructureSource.EXPERIMENTAL_PDB,
            vh_chain_id="B", vl_chain_id="A",
            pdb_id="1N8Z",
            confidence=1.0,
        )

    def test_parses_and_yields_both_chains(self, t_input):
        m = analyze_structure(t_input)
        assert m.available
        assert len(m.vh_resnames) > 100  # full Fab VH
        assert len(m.vl_resnames) > 100

    def test_vh_vl_bsa_in_expected_range(self, t_input):
        """Literature VH-VL BSA is 1500-2000 Å². 1N8Z includes Fab constant
        domains so we allow a bit more headroom (up to 2500)."""
        m = analyze_structure(t_input)
        assert m.interface is not None
        bsa = m.interface.buried_surface_area_a2
        assert 1500 <= bsa <= 2500, f"BSA {bsa:.0f} outside [1500, 2500]"

    def test_paratope_nonempty(self, t_input):
        m = analyze_structure(t_input)
        assert m.paratope is not None
        assert len(m.paratope.cdr_residues) > 5
        assert m.paratope.total_surface_area_a2 > 500  # Å²

    def test_structural_tap_all_populated(self, t_input):
        m = analyze_structure(t_input)
        import math
        for field in ("tap_psh", "tap_ppc", "tap_pnc", "tap_sfvcsp", "di_full", "di_sap"):
            val = getattr(m, field)
            assert math.isfinite(val), f"{field} is NaN/Inf"
        assert m.tap_total_cdr_length >= 0

    def test_aggregation_patches_nonzero(self, t_input):
        m = analyze_structure(t_input)
        assert len(m.aggregation_patches) >= 1

    def test_structure_risk_score_bounded(self, t_input):
        m = analyze_structure(t_input)
        assert 0.0 <= m.structure_risk_score <= 1.0
