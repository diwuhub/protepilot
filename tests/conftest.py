"""
ProtePilot Test Configuration
=============================
Shared fixtures and skip helpers for the pytest test suite.

Four-layer architecture:
  Layer 1 — core:       No torch/sklearn/streamlit — core analysis pipeline
  Layer 2 — governance: Grade canonicality, determinism, schema alignment, evidence
  Layer 3 — bulk:       Bulk CSV processing, row isolation, bulk-vs-single parity
  Layer 4 — training:   ML models (requires torch/sklearn), uncertainty, reproducibility

Environment helpers:
  @requires_torch      — skip if torch not installed
  @requires_sklearn    — skip if sklearn not installed
  @requires_streamlit  — skip if streamlit not installed
"""
import sys
import os
import pytest

# Ensure src/ is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ═══════════════════════════════════════════════════════════════════════
#  Skip helpers — use on tests that need optional packages
# ═══════════════════════════════════════════════════════════════════════

def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _has_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def _has_streamlit():
    try:
        import streamlit  # noqa: F401
        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(not _has_torch(), reason="torch not installed (Layer 3)")
requires_sklearn = pytest.mark.skipif(not _has_sklearn(), reason="sklearn not installed (Layer 3)")
requires_streamlit = pytest.mark.skipif(not _has_streamlit(), reason="streamlit not installed (Layer 2)")


# ═══════════════════════════════════════════════════════════════════════
#  Shared sequences — used across multiple test layers
# ═══════════════════════════════════════════════════════════════════════

HC_SEQ = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNG"
    "YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQ"
    "GTLVTVSS"
)

LC_SEQ = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLES"
    "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)

PEPTIDE_SEQ = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"

VHH_SEQ = "QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWS"


# ═══════════════════════════════════════════════════════════════════════
#  Layer 1: Core fixtures — single-molecule pipeline
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_mab_intent():
    """Minimal canonical mAb intent dict for testing."""
    return {
        "name": "Trastuzumab-test",
        "format": "canonical_mab",
        "hc_sequence": HC_SEQ,
        "lc_sequence": LC_SEQ,
        "is_mab": True,
        "glycoform": "standard_cho",
    }


@pytest.fixture
def sample_bispecific_intent():
    """Minimal bispecific intent dict for testing."""
    return {
        "name": "BispecTest",
        "format": "bispecific",
        "hc_sequence": HC_SEQ,
        "lc_sequence": LC_SEQ,
        "hc2_sequence": (
            "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYDINWVRQATGQGLEWMGWMNPNSG"
            "NTGYAQKFQGRVTMTRDTSISTAYMEVSRLRSDDTAVYYCARDPFGAMDYWGQGTL"
            "VTVSS"
        ),
        "lc2_sequence": (
            "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQS"
            "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
        ),
        "is_mab": True,
        "glycoform": "standard_cho",
    }


@pytest.fixture
def sample_peptide_intent():
    """Minimal peptide intent dict for testing."""
    return {
        "name": "GLP1-test",
        "format": "peptide",
        "hc_sequence": PEPTIDE_SEQ,
        "is_mab": False,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Layer 3: Bulk fixtures — CSV data and parse results
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def mab_csv_3rows():
    """3-row canonical mAb CSV for bulk tests."""
    return (
        "name,HC,LC\n"
        f"Trastuzumab_v1,{HC_SEQ},{LC_SEQ}\n"
        f"Adalimumab_v1,{HC_SEQ}ADDD,{LC_SEQ}KKKK\n"
        f"Candidate_3,{HC_SEQ}EEE,{LC_SEQ}RRR\n"
    )


@pytest.fixture
def mab_csv_with_bad_row():
    """CSV with one good row and one invalid-AA row."""
    return (
        "name,HC,LC\n"
        f"Good_mAb,{HC_SEQ},{LC_SEQ}\n"
        f"Bad_mAb,{HC_SEQ}123XZ,{LC_SEQ}\n"
        f"Also_Good,{HC_SEQ}AAA,{LC_SEQ}GGG\n"
    )
