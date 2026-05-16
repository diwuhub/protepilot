"""Tests for src/mutation/enumerate.py."""

import pytest

from src.mutation.enumerate import cdr_only, enumerate_single_mutations
from src.mutation.schema import (
    CDR_REGIONS,
    FRAMEWORK_REGIONS,
    STANDARD_AA,
    AntibodyChain,
)

# Trastuzumab — previously used in Phase 0 and Phase 1 tests.
TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


class TestEnumerate:
    def test_count_vh_only(self):
        cands = enumerate_single_mutations(TRASTUZUMAB_VH)
        # len(VH) * 19 non-identity AAs
        assert len(cands) == len(TRASTUZUMAB_VH) * 19

    def test_count_vh_plus_vl(self):
        cands = enumerate_single_mutations(TRASTUZUMAB_VH, TRASTUZUMAB_VL)
        assert len(cands) == (len(TRASTUZUMAB_VH) + len(TRASTUZUMAB_VL)) * 19

    def test_no_identity_mutations(self):
        cands = enumerate_single_mutations(TRASTUZUMAB_VH, TRASTUZUMAB_VL)
        assert all(c.wildtype_aa != c.mutant_aa for c in cands)

    def test_all_standard_aas_used(self):
        cands = enumerate_single_mutations(TRASTUZUMAB_VH)
        used_mt = {c.mutant_aa for c in cands}
        assert used_mt == set(STANDARD_AA)

    def test_allowed_aas_restricts_mutants(self):
        cands = enumerate_single_mutations(TRASTUZUMAB_VH, allowed_aas="AG")
        used_mt = {c.mutant_aa for c in cands}
        assert used_mt <= set("AG")

    def test_vh_positions_are_within_range(self):
        cands = enumerate_single_mutations(TRASTUZUMAB_VH, TRASTUZUMAB_VL)
        for c in cands:
            if c.chain is AntibodyChain.VH:
                assert 0 <= c.position < len(TRASTUZUMAB_VH)
            else:
                assert 0 <= c.position < len(TRASTUZUMAB_VL)

    def test_wildtype_matches_parent(self):
        cands = enumerate_single_mutations(TRASTUZUMAB_VH, TRASTUZUMAB_VL)
        for c in cands:
            parent = TRASTUZUMAB_VH if c.chain is AntibodyChain.VH else TRASTUZUMAB_VL
            assert parent[c.position] == c.wildtype_aa

    def test_cdr_extraction_labels_at_least_h3(self):
        cands = enumerate_single_mutations(TRASTUZUMAB_VH, TRASTUZUMAB_VL)
        h3 = [c for c in cands if c.region == "cdr_h3"]
        # H3 should have >= 1 residue × 19 mutants = 19+ candidates.
        assert len(h3) >= 19

    def test_exclude_framework_drops_framework_candidates(self):
        cands = enumerate_single_mutations(TRASTUZUMAB_VH, TRASTUZUMAB_VL, exclude_framework=True)
        for c in cands:
            assert c.region in CDR_REGIONS

    def test_restrict_to_positions(self):
        cands = enumerate_single_mutations(
            TRASTUZUMAB_VH,
            TRASTUZUMAB_VL,
            restrict_to_positions={AntibodyChain.VH: [27]},
        )
        vh_positions = {c.position for c in cands if c.chain is AntibodyChain.VH}
        assert vh_positions == {27}
        # VL positions should still produce all candidates.
        vl_count = sum(1 for c in cands if c.chain is AntibodyChain.VL)
        assert vl_count == len(TRASTUZUMAB_VL) * 19


class TestCdrOnly:
    def test_all_regions_are_cdr(self):
        cands = cdr_only(TRASTUZUMAB_VH, TRASTUZUMAB_VL)
        for c in cands:
            assert c.region in CDR_REGIONS


class TestErrors:
    def test_empty_vh_raises(self):
        with pytest.raises(ValueError, match="VH sequence is required"):
            enumerate_single_mutations("")

    def test_non_standard_residue_raises(self):
        with pytest.raises(ValueError, match="non-standard residue"):
            enumerate_single_mutations("EVQLVBX")

    def test_bad_allowed_aas_raises(self):
        with pytest.raises(ValueError, match="allowed_aas"):
            enumerate_single_mutations(TRASTUZUMAB_VH, allowed_aas="AZ")
