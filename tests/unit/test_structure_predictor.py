"""Tests for src/structure/predictor.py — sequence hashing, caching, unavailable path."""

from pathlib import Path

import pytest

from src.structure.predictor import (
    cached_pdb,
    predict_structure,
    register_user_pdb,
    sequence_pair_hash,
)
from src.structure.schema import StructureSource


SIMPLE_PDB = """\
ATOM      1  N   ALA H   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA H   1      11.000  10.500  10.500  1.00 20.00           C
ATOM      3  C   ALA H   1      12.000  11.000  10.000  1.00 20.00           C
ATOM      4  O   ALA H   1      12.500  11.500  10.500  1.00 20.00           O
END
"""


class TestHash:
    def test_stable(self):
        h1 = sequence_pair_hash("ACDEF", "GHIKL")
        h2 = sequence_pair_hash("ACDEF", "GHIKL")
        assert h1 == h2

    def test_case_insensitive(self):
        assert sequence_pair_hash("acdef", "GHIKL") == sequence_pair_hash("ACDEF", "ghikl")

    def test_different_pairs(self):
        assert sequence_pair_hash("A", "B") != sequence_pair_hash("B", "A")


class TestUnavailable:
    def test_no_predictors_returns_unavailable(self, tmp_path):
        # Real runs on this environment have IgFold/ESMFold/SAbPred absent,
        # so predict_structure should return an UNAVAILABLE input.
        res = predict_structure(
            vh="EVQLV", vl="DIQMT", cache_root=tmp_path, prefer_cached=False,
        )
        assert res.source is StructureSource.UNAVAILABLE
        assert res.pdb_path == ""
        assert "no predictor" in res.notes or "wrapper not yet" in res.notes


class TestRegisterUserPdb:
    def test_copies_to_cache(self, tmp_path):
        src = tmp_path / "input.pdb"
        src.write_text(SIMPLE_PDB)
        res = register_user_pdb(
            vh="ABC", vl="DEF", pdb_path=src,
            cache_root=tmp_path / "cache",
            source=StructureSource.USER_SUPPLIED,
        )
        cached = Path(res.pdb_path)
        assert cached.exists()
        assert cached.name == "user.pdb"
        assert cached.parent.name == sequence_pair_hash("ABC", "DEF")

    def test_experimental_labeled_differently(self, tmp_path):
        src = tmp_path / "x.pdb"
        src.write_text(SIMPLE_PDB)
        res = register_user_pdb(
            vh="ABC", vl="DEF", pdb_path=src,
            cache_root=tmp_path / "cache",
            source=StructureSource.EXPERIMENTAL_PDB,
            pdb_id="1N8Z",
        )
        assert Path(res.pdb_path).name == "experimental.pdb"
        assert res.pdb_id == "1N8Z"

    def test_cached_pdb_lookup(self, tmp_path):
        src = tmp_path / "x.pdb"
        src.write_text(SIMPLE_PDB)
        register_user_pdb(
            vh="AB", vl="CD", pdb_path=src, cache_root=tmp_path / "cache",
            source=StructureSource.EXPERIMENTAL_PDB,
        )
        found = cached_pdb("AB", "CD", cache_root=tmp_path / "cache")
        assert found is not None
        assert found.name == "experimental.pdb"

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            register_user_pdb(
                vh="A", vl="B", pdb_path=tmp_path / "nope.pdb",
                cache_root=tmp_path / "cache",
            )
