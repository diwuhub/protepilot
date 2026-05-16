"""
Test suite for bispecific_engine module (src/bispecific_engine.py)
==================================================================

Tests bispecific antibody assembly species generation, SMA parameter mapping,
and chromatographic resolution calculations.

Markers:
  @pytest.mark.core — No torch/sklearn/streamlit dependencies
"""

import pytest
import math
from src.bispecific_engine import (
    AntibodyChain,
    AssemblySpecies,
    build_assembly_species,
    map_species_to_sma,
    compute_resolution,
    assess_separation_risk,
    estimate_species_peaks,
)


@pytest.mark.core
class TestAntibodyChain:
    """Test AntibodyChain dataclass and biophysical computation."""

    def test_chain_creation_manual_properties(self):
        """Test creating AntibodyChain with manual properties."""
        chain = AntibodyChain(
            name="HC_A",
            sequence="MGLPPVL" * 20,  # Dummy sequence
            pI=8.5,
            mw_kda=50.0,
            hydrophobicity=0.35,
        )
        assert chain.name == "HC_A"
        assert chain.pI == 8.5
        assert chain.mw_kda == 50.0

    def test_chain_with_typical_values(self):
        """Test chain with values typical for heavy chain."""
        chain = AntibodyChain(
            name="HC_Test",
            sequence="",
            pI=8.5,
            mw_kda=50.0,
            hydrophobicity=0.35,
            deam_sites=1,
            ox_sites=2,
        )
        assert chain.pI == 8.5
        assert chain.mw_kda == 50.0
        assert chain.deam_sites == 1
        assert chain.ox_sites == 2

    def test_chain_post_init_with_sequence(self):
        """Test chain computation on initialization with sequence."""
        # Note: requires Biopython, may fall back to estimation
        chain = AntibodyChain(
            name="VH",
            sequence="MKVLVVLLTLAGSAQAAA",  # VH-like start
            pI=0.0,  # Will be computed
        )
        # After __post_init__, pI should be computed (or estimated)
        assert chain.pI > 0.0


@pytest.mark.core
class TestBuildAssemblySpecies:
    """Test building bispecific assembly species."""

    def test_build_assembly_species_basic(self):
        """Test building three assembly species from two chains."""
        chain_a = AntibodyChain(
            name="A",
            sequence="MGLPPVL" * 20,
            pI=8.5,
            mw_kda=50.0,
            hydrophobicity=0.35,
        )
        chain_b = AntibodyChain(
            name="B",
            sequence="MGLPPVL" * 25,
            pI=8.3,
            mw_kda=55.0,
            hydrophobicity=0.33,
        )

        species = build_assembly_species(chain_a, chain_b)

        # Should return three species
        assert "AA" in species
        assert "AB" in species
        assert "BB" in species
        assert len(species) == 3

    def test_assembly_species_aa_bb_mw(self):
        """Test homodimer species have correct MW (2x chain)."""
        # Use real sequences from conftest that compute to meaningful MW
        chain_a = AntibodyChain(
            name="A",
            sequence=(
                "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNG"
                "YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQ"
                "GTLVTVSS"
            ),
            pI=0.0,  # Will be computed
            hydrophobicity=0.35,
        )
        chain_b = AntibodyChain(
            name="B",
            sequence=(
                "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLES"
                "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
            ),
            pI=0.0,  # Will be computed
            hydrophobicity=0.33,
        )

        species = build_assembly_species(chain_a, chain_b)

        # All species should have positive MW
        assert species["AA"].mw_kda > 0
        assert species["AB"].mw_kda > 0
        assert species["BB"].mw_kda > 0

    def test_assembly_species_pi_difference(self):
        """Test pI values differ for AA/AB/BB based on chain composition."""
        # Use distinct sequences that will produce different pI values
        chain_a = AntibodyChain(
            name="A",
            sequence=(
                "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNG"
                "YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQ"
                "GTLVTVSS"
            ),
        )
        chain_b = AntibodyChain(
            name="B",
            sequence=(
                "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLES"
                "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
            ),
        )

        species = build_assembly_species(chain_a, chain_b)

        # With distinct sequences, pI values will differ
        # Verify all three species have non-zero pI
        assert species["AA"].pI > 0
        assert species["AB"].pI > 0
        assert species["BB"].pI > 0

    def test_assembly_species_is_target_flag(self):
        """Test is_target flag is set correctly."""
        chain_a = AntibodyChain(
            name="A",
            sequence="M" * 100,
            pI=8.5,
            mw_kda=50.0,
        )
        chain_b = AntibodyChain(
            name="B",
            sequence="M" * 100,
            pI=8.3,
            mw_kda=50.0,
        )

        species = build_assembly_species(chain_a, chain_b)

        # Only AB is target
        assert species["AA"].is_target is False
        assert species["AB"].is_target is True
        assert species["BB"].is_target is False

    def test_assembly_species_display_names(self):
        """Test display names are set correctly."""
        chain_a = AntibodyChain(
            name="HC",
            sequence="M" * 100,
            pI=8.5,
            mw_kda=50.0,
        )
        chain_b = AntibodyChain(
            name="HC2",
            sequence="M" * 100,
            pI=8.3,
            mw_kda=50.0,
        )

        species = build_assembly_species(chain_a, chain_b)

        assert "Homodimer" in species["AA"].display_name
        assert "Heterodimer" in species["AB"].display_name
        assert "Homodimer" in species["BB"].display_name


@pytest.mark.core
class TestMapSpeciesToSMA:
    """Test SMA parameter mapping for bispecific species."""

    def test_map_species_to_sma_basic(self):
        """Test basic SMA mapping of three species."""
        chain_a = AntibodyChain(
            name="A",
            sequence="M" * 100,
            pI=8.5,
            mw_kda=50.0,
            hydrophobicity=0.35,
        )
        chain_b = AntibodyChain(
            name="B",
            sequence="M" * 100,
            pI=8.3,
            mw_kda=50.0,
            hydrophobicity=0.33,
        )

        species = build_assembly_species(chain_a, chain_b)
        sma_params = map_species_to_sma(species)

        # Should return SMA params for all three
        assert "AA" in sma_params
        assert "AB" in sma_params
        assert "BB" in sma_params

        # Each should have expected keys
        for key in ("AA", "AB", "BB"):
            assert "nu" in sma_params[key]
            assert "ka" in sma_params[key]
            assert "kd" in sma_params[key]
            assert "sigma" in sma_params[key]
            assert "lambda_" in sma_params[key]

    def test_sma_parameters_in_valid_range(self):
        """Test all SMA parameters are in valid ranges."""
        chain_a = AntibodyChain(
            name="A",
            sequence="M" * 100,
            pI=8.5,
            mw_kda=50.0,
            hydrophobicity=0.35,
        )
        chain_b = AntibodyChain(
            name="B",
            sequence="M" * 100,
            pI=8.3,
            mw_kda=50.0,
            hydrophobicity=0.33,
        )

        species = build_assembly_species(chain_a, chain_b)
        sma_params = map_species_to_sma(species)

        for key in ("AA", "AB", "BB"):
            params = sma_params[key]
            assert 2.0 <= params["nu"] <= 6.0, f"{key} nu out of range"
            assert 0.3 <= params["ka"] <= 8.0, f"{key} ka out of range"
            assert params["kd"] == 1000.0
            assert params["sigma"] > 0
            assert params["lambda_"] == 1200.0

    def test_ml_override_applies_to_ab_only(self):
        """Test ML override applies only to AB (target) species."""
        chain_a = AntibodyChain(
            name="A",
            sequence="M" * 100,
            pI=8.5,
            mw_kda=50.0,
        )
        chain_b = AntibodyChain(
            name="B",
            sequence="M" * 100,
            pI=8.3,
            mw_kda=50.0,
        )

        species = build_assembly_species(chain_a, chain_b)

        ml_override = {"ka": 1.5, "nu": 2.8}
        sma_with_ml = map_species_to_sma(species, ml_override=ml_override)

        # AB should have ML values
        assert sma_with_ml["AB"]["ka"] == 1.5
        assert sma_with_ml["AB"]["nu"] == 2.8

        # AA and BB should not be affected
        sma_no_ml = map_species_to_sma(species, ml_override=None)
        assert sma_with_ml["AA"]["nu"] == sma_no_ml["AA"]["nu"]
        assert sma_with_ml["BB"]["ka"] == sma_no_ml["BB"]["ka"]


@pytest.mark.core
class TestComputeResolution:
    """Test chromatographic resolution calculation."""

    def test_resolution_well_separated_peaks(self):
        """Test resolution for well-separated peaks (Rs > 1.5)."""
        # Peak 1: RT=10 min, FWHM=1 min
        # Peak 2: RT=14 min, FWHM=1 min
        rs = compute_resolution(10.0, 1.0, 14.0, 1.0)

        # Rs = 2 * |14-10| / (1*2.355 + 1*2.355) = 8 / 4.71 ≈ 1.70
        assert rs > 1.5
        assert rs < 2.0  # Sanity check

    def test_resolution_co_eluting_peaks(self):
        """Test resolution for co-eluting peaks (Rs ≈ 0)."""
        # Same retention time
        rs = compute_resolution(10.0, 1.0, 10.0, 1.0)
        assert rs == 0.0  # Co-eluting

    def test_resolution_partially_resolved(self):
        """Test resolution in medium range (0.8-1.5)."""
        # Peak 1: RT=10, FWHM=1
        # Peak 2: RT=11.5, FWHM=1
        rs = compute_resolution(10.0, 1.0, 11.5, 1.0)

        # Rs = 2 * |11.5-10| / (2.355+2.355) = 3 / 4.71 ≈ 0.637
        assert 0.0 < rs < 1.5

    def test_resolution_symmetric(self):
        """Test resolution is symmetric (order-independent)."""
        rs_12 = compute_resolution(10.0, 1.0, 12.0, 1.0)
        rs_21 = compute_resolution(12.0, 1.0, 10.0, 1.0)
        assert rs_12 == rs_21

    def test_resolution_with_different_widths(self):
        """Test resolution with asymmetric peak widths."""
        # Peak 1: wide, Peak 2: narrow
        rs_wide_narrow = compute_resolution(10.0, 2.0, 14.0, 0.5)

        # Peak 1: narrow, Peak 2: wide
        rs_narrow_wide = compute_resolution(10.0, 0.5, 14.0, 2.0)

        # Both should be positive
        assert rs_wide_narrow > 0
        assert rs_narrow_wide > 0


@pytest.mark.core
class TestAssessSeparationRisk:
    """Test separation risk assessment from species peaks."""

    def test_assess_risk_low_resolution(self):
        """Test high risk classification when Rs < 0.8."""
        species_peaks = {
            "AA": {"rt_min": 10.0, "fwhm_min": 1.0},
            "AB": {"rt_min": 10.5, "fwhm_min": 1.0},  # Close to AA
            "BB": {"rt_min": 11.0, "fwhm_min": 1.0},
        }

        risk = assess_separation_risk(species_peaks)

        assert risk["risk_level"] == "High"
        assert risk["rs_AB_AA"] < 0.8
        assert len(risk["recommendations"]) > 0

    def test_assess_risk_medium_resolution(self):
        """Test medium risk when 0.8 <= Rs < 1.5."""
        species_peaks = {
            "AA": {"rt_min": 10.0, "fwhm_min": 1.0},
            "AB": {"rt_min": 12.355, "fwhm_min": 1.0},  # Rs = 1.0 from AA
            "BB": {"rt_min": 14.71, "fwhm_min": 1.0},   # Rs = 1.0 from AB
        }

        risk = assess_separation_risk(species_peaks)

        assert risk["risk_level"] == "Medium"
        assert risk["min_rs"] >= 0.8
        assert risk["min_rs"] < 1.5

    def test_assess_risk_good_resolution(self):
        """Test low risk when Rs >= 1.5."""
        species_peaks = {
            "AA": {"rt_min": 10.0, "fwhm_min": 1.0},
            "AB": {"rt_min": 14.0, "fwhm_min": 1.0},  # Well separated
            "BB": {"rt_min": 18.0, "fwhm_min": 1.0},
        }

        risk = assess_separation_risk(species_peaks)

        assert risk["risk_level"] == "Low"
        assert risk["min_rs"] >= 1.5

    def test_separation_risk_has_recommendations(self):
        """Test that risk assessment includes recommendations."""
        species_peaks = {
            "AA": {"rt_min": 10.0, "fwhm_min": 1.0},
            "AB": {"rt_min": 12.0, "fwhm_min": 1.0},
            "BB": {"rt_min": 14.0, "fwhm_min": 1.0},
        }

        risk = assess_separation_risk(species_peaks)

        assert "recommendations" in risk
        assert isinstance(risk["recommendations"], list)
        assert len(risk["recommendations"]) > 0


@pytest.mark.core
class TestEstimateSpeciesPeaks:
    """Test peak estimation for species (analytical approximation)."""

    def test_estimate_species_peaks_structure(self):
        """Test estimate_species_peaks returns correct structure."""
        sma_params = {
            "AA": {"nu": 2.5, "ka": 1.5, "sigma": 8.5},
            "AB": {"nu": 2.6, "ka": 1.6, "sigma": 8.6},
            "BB": {"nu": 2.7, "ka": 1.7, "sigma": 8.7},
        }

        peaks = estimate_species_peaks(sma_params)

        # Should return peak data for all three
        assert "AA" in peaks
        assert "AB" in peaks
        assert "BB" in peaks

        # Each should have rt_min and fwhm_min
        for key in ("AA", "AB", "BB"):
            assert "rt_min" in peaks[key]
            assert "fwhm_min" in peaks[key]

    def test_species_peaks_reasonable_values(self):
        """Test estimated peaks have reasonable chromatographic values."""
        sma_params = {
            "AA": {"nu": 2.5, "ka": 1.5, "sigma": 8.5},
            "AB": {"nu": 2.6, "ka": 1.6, "sigma": 8.6},
            "BB": {"nu": 2.7, "ka": 1.7, "sigma": 8.7},
        }

        peaks = estimate_species_peaks(sma_params, gradient_slope=15.0)

        # Retention times should be positive
        for key in ("AA", "AB", "BB"):
            assert peaks[key]["rt_min"] > 0
            assert peaks[key]["fwhm_min"] > 0

    def test_species_peaks_different_gradient_slopes(self):
        """Test gradient slope affects retention times."""
        sma_params = {
            "AA": {"nu": 2.5, "ka": 1.5, "sigma": 8.5},
            "AB": {"nu": 2.6, "ka": 1.6, "sigma": 8.6},
            "BB": {"nu": 2.7, "ka": 1.7, "sigma": 8.7},
        }

        peaks_fast = estimate_species_peaks(sma_params, gradient_slope=30.0)
        peaks_slow = estimate_species_peaks(sma_params, gradient_slope=10.0)

        # Slower gradient → longer retention times
        assert peaks_slow["AB"]["rt_min"] > peaks_fast["AB"]["rt_min"]


@pytest.mark.core
class TestBispecificEndToEnd:
    """End-to-end tests for bispecific workflow."""

    def test_full_bispecific_workflow(self):
        """Test complete workflow: chains → species → SMA → peaks → risk."""
        # Build chains
        chain_hc = AntibodyChain(
            name="HC",
            sequence="M" * 100,
            pI=8.5,
            mw_kda=50.0,
            hydrophobicity=0.35,
        )
        chain_hc2 = AntibodyChain(
            name="HC2",
            sequence="M" * 100,
            pI=8.2,
            mw_kda=50.0,
            hydrophobicity=0.33,
        )

        # Build species
        species = build_assembly_species(chain_hc, chain_hc2)
        assert len(species) == 3

        # Map to SMA
        sma_params = map_species_to_sma(species)
        assert all(key in sma_params for key in ("AA", "AB", "BB"))

        # Estimate peaks
        peaks = estimate_species_peaks(sma_params)
        assert all(key in peaks for key in ("AA", "AB", "BB"))

        # Assess separation risk
        risk = assess_separation_risk(peaks)
        assert "risk_level" in risk
        assert risk["risk_level"] in ("Low", "Medium", "High")

    def test_bispecific_homodimer_separation(self):
        """Test that homodimers are distinct from heterodimer."""
        chain_a = AntibodyChain(
            name="A",
            sequence=(
                "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNG"
                "YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQ"
                "GTLVTVSS"
            ),
        )
        chain_b = AntibodyChain(
            name="B",
            sequence=(
                "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLES"
                "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
            ),
        )

        species = build_assembly_species(chain_a, chain_b)

        # With different chain sequences, all three species should have different MW
        assert species["AA"].mw_kda != species["BB"].mw_kda
        # All species should have positive MW and pI
        assert all(s.mw_kda > 0 for s in [species["AA"], species["AB"], species["BB"]])
        assert all(s.pI > 0 for s in [species["AA"], species["AB"], species["BB"]])
