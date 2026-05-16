"""
Test suite for PropertyMapper module (src/PropertyMapper.py)
============================================================

Tests the core mapping functionality from protein physical properties
to CADET SMA parameters. Includes tests for pI/MW/GRAVY calculations,
ML override logic, and edge cases.

Markers:
  @pytest.mark.core — No torch/sklearn/streamlit dependencies
"""

import pytest
import math
from src.PropertyMapper import (
    ProteinProperties,
    PropertyMapper,
    MapperConfig,
)


@pytest.mark.core
class TestProteinProperties:
    """Test ProteinProperties dataclass validation and utilities."""

    def test_valid_protein_creation(self):
        """Test creation of valid ProteinProperties object."""
        protein = ProteinProperties(
            name="TestmAb",
            pI=8.45,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )
        assert protein.name == "TestmAb"
        assert protein.pI == 8.45
        assert protein.MW_kDa == 150.0
        assert protein.hydrophobicity == 0.35

    def test_hydrophobicity_validation(self):
        """Test hydrophobicity must be in [0, 1]."""
        with pytest.raises(ValueError, match="hydrophobicity"):
            ProteinProperties(
                name="Bad",
                pI=8.0,
                MW_kDa=150.0,
                hydrophobicity=1.5,  # Out of range
            )

    def test_pi_validation(self):
        """Test pI must be in valid range (0-14)."""
        with pytest.raises(ValueError, match="pI"):
            ProteinProperties(
                name="Bad",
                pI=15.0,  # Out of range
                MW_kDa=150.0,
                hydrophobicity=0.35,
            )

    def test_mw_validation(self):
        """Test MW_kDa must be positive."""
        with pytest.raises(ValueError, match="MW_kDa"):
            ProteinProperties(
                name="Bad",
                pI=8.0,
                MW_kDa=-10.0,  # Negative
                hydrophobicity=0.35,
            )

    def test_net_charge_factor(self):
        """Test net charge factor calculation."""
        protein = ProteinProperties(
            name="Test",
            pI=8.5,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )
        charge_factor = protein.net_charge_factor()
        assert charge_factor == abs(7.0 - 8.5)
        assert charge_factor == 1.5

    def test_ptm_profile_validation(self):
        """Test PTM profile validation."""
        # Valid PTM profile
        protein = ProteinProperties(
            name="Test",
            pI=8.0,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            ptm_profile={"deamidation_sites": 2, "oxidation_sites": 1},
        )
        assert protein.ptm_profile["deamidation_sites"] == 2

        # Invalid PTM profile (negative value)
        with pytest.raises(ValueError, match="ptm_profile"):
            ProteinProperties(
                name="Bad",
                pI=8.0,
                MW_kDa=150.0,
                hydrophobicity=0.35,
                ptm_profile={"deamidation_sites": -1},
            )


@pytest.mark.core
class TestPropertyMapperBasic:
    """Test basic PropertyMapper functionality."""

    def test_mapper_initialization(self):
        """Test PropertyMapper initialization with default config."""
        mapper = PropertyMapper()
        assert mapper.config is not None
        assert mapper.config.nu_hardcoded_base == 2.5
        assert mapper.config.ka_flat_base == 3.0

    def test_mapper_custom_config(self):
        """Test PropertyMapper initialization with custom config."""
        config = MapperConfig(nu_hardcoded_base=2.0)
        mapper = PropertyMapper(config=config)
        assert mapper.config.nu_hardcoded_base == 2.0

    def test_map_canonical_mab(self):
        """Test mapping a canonical mAb with known properties."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name="NISTmAb",
            pI=8.31,
            MW_kDa=148.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )
        params = mapper.map(protein)

        # Verify output structure
        assert "nu" in params
        assert "ka" in params
        assert "kd" in params
        assert "sigma" in params
        assert "lambda_" in params
        assert "source" in params

        # Verify value ranges
        assert 2.0 <= params["nu"] <= 6.0  # nu bounds
        assert 0.3 <= params["ka"] <= 8.0  # ka bounds
        assert params["kd"] == 1000.0  # Fixed kd
        assert params["sigma"] > 0
        assert params["lambda_"] == 1200.0  # Fixed lambda

    def test_nu_calculation_from_pI(self):
        """Test nu calculation depends on pI difference from pH."""
        mapper = PropertyMapper()

        # Low pI (far from pH 7)
        protein_low_pi = ProteinProperties(
            name="LowpI",
            pI=7.4,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )
        params_low = mapper.map(protein_low_pi)

        # High pI (far from pH 7)
        protein_high_pi = ProteinProperties(
            name="HighpI",
            pI=9.0,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )
        params_high = mapper.map(protein_high_pi)

        # Higher pI → larger |pH - pI| → higher nu
        assert params_high["nu"] > params_low["nu"]

    def test_sigma_calculation_from_mw(self):
        """Test sigma (steric factor) increases with MW."""
        mapper = PropertyMapper()

        # Small protein
        protein_small = ProteinProperties(
            name="Small",
            pI=8.0,
            MW_kDa=50.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )
        params_small = mapper.map(protein_small)

        # Large protein
        protein_large = ProteinProperties(
            name="Large",
            pI=8.0,
            MW_kDa=250.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )
        params_large = mapper.map(protein_large)

        # Larger MW → larger sigma
        assert params_large["sigma"] > params_small["sigma"]


@pytest.mark.core
class TestPropertyMapperGRAVY:
    """Test GRAVY-based ka calculation."""

    def test_gravy_derived_ka(self):
        """Test ka is derived from GRAVY score when available."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name="GRAVYtest",
            pI=8.45,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
            gravy_score=-0.415,  # Trastuzumab-like
        )
        params = mapper.map(protein)

        # Should use GRAVY model
        assert params["source"] == "static_v5_gravy"

        # ka should be computed via exp(beta * GRAVY)
        expected_ka = mapper.config.ka_flat_base * math.exp(
            mapper.config.beta_gravy * (-0.415)
        )
        expected_ka = max(
            mapper.config.ka_min,
            min(mapper.config.ka_max, expected_ka),
        )
        assert abs(params["ka"] - expected_ka) < 0.0001

    def test_gravy_monotonicity(self):
        """Test that higher GRAVY → higher ka (thermodynamic model)."""
        mapper = PropertyMapper()

        gravy_scores = [-0.5, -0.3, -0.1, 0.0]
        ka_values = []

        for gravy in gravy_scores:
            protein = ProteinProperties(
                name=f"GRAVY{gravy}",
                pI=8.0,
                MW_kDa=150.0,
                hydrophobicity=0.35,
                pH_working=7.0,
                gravy_score=gravy,
            )
            params = mapper.map(protein)
            ka_values.append(params["ka"])

        # More hydrophobic (higher GRAVY) → higher ka
        for i in range(len(ka_values) - 1):
            assert ka_values[i] < ka_values[i + 1]


@pytest.mark.core
class TestPropertyMapperMLOverride:
    """Test ML override mode."""

    def test_ml_override_acceptance(self):
        """Test ML override is accepted and replaces static calculation."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name="MLtest",
            pI=8.45,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )

        ml_override = {"ka": 1.5, "nu": 2.8}
        params = mapper.map(protein, ml_override=ml_override)

        # Should use ML override
        assert params["source"] == "ml_override"
        assert params["ka"] == 1.5
        assert params["nu"] == 2.8

    def test_ml_override_quality_gate(self):
        """Test ML override is rejected if R² below threshold."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name="BadML",
            pI=8.45,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )

        # Provide bad ML with very low R²
        ml_override = {"ka": 1.5, "nu": 2.8, "val_r2": 0.3}  # Below 0.5 threshold
        params = mapper.map(protein, ml_override=ml_override)

        # Should fall back to static
        assert params["source"] == "static_v5"

    def test_ml_override_boundary_rejection(self):
        """Test ML override is rejected if at parameter boundaries."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name="BoundaryML",
            pI=8.45,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )

        # ML at ka boundary
        ml_override = {"ka": 0.3, "nu": 2.8}  # ka near minimum
        params = mapper.map(protein, ml_override=ml_override)

        # Should fall back to static
        assert params["source"] == "static_v5"


@pytest.mark.core
class TestPropertyMapperEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_sequence(self):
        """Test mapping with empty sequence (should not crash)."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name="NoSeq",
            pI=8.0,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
            sequence="",
        )
        params = mapper.map(protein)
        assert params["nu"] > 0
        assert params["ka"] > 0

    def test_very_short_peptide(self):
        """Test mapping a very short peptide sequence."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name="Peptide",
            pI=7.5,
            MW_kDa=3.0,  # Small peptide
            hydrophobicity=0.5,
            pH_working=7.0,
            sequence="MGLPPVL",
        )
        params = mapper.map(protein)

        # Should produce valid parameters
        assert params["nu"] > 0
        assert params["ka"] > 0
        assert params["sigma"] > 0

    def test_extreme_hydrophobicity_boundaries(self):
        """Test with maximum and minimum hydrophobicity values."""
        mapper = PropertyMapper()

        # Maximum hydrophobicity
        protein_hydro = ProteinProperties(
            name="Hydrophobic",
            pI=8.0,
            MW_kDa=150.0,
            hydrophobicity=1.0,
            pH_working=7.0,
        )
        params_hydro = mapper.map(protein_hydro)

        # Minimum hydrophobicity
        protein_hydrophilic = ProteinProperties(
            name="Hydrophilic",
            pI=8.0,
            MW_kDa=150.0,
            hydrophobicity=0.0,
            pH_working=7.0,
        )
        params_hydro_phil = mapper.map(protein_hydrophilic)

        # Both should produce valid outputs
        assert 0.3 <= params_hydro["ka"] <= 8.0
        assert 0.3 <= params_hydro_phil["ka"] <= 8.0
        # Hydrophobic > hydrophilic
        assert params_hydro["ka"] > params_hydro_phil["ka"]


@pytest.mark.core
class TestPropertyMapperVariants:
    """Test multi-variant mapping (PTM variants)."""

    def test_map_variants_structure(self):
        """Test map_variants returns correct structure."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name="VariantTest",
            pI=8.45,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
            ptm_profile={"deamidation_sites": 2, "oxidation_sites": 1},
        )
        vp = mapper.map_variants(protein)

        # Check structure
        assert "acidic" in vp
        assert "main" in vp
        assert "basic" in vp
        assert "kd" in vp
        assert "lambda_" in vp
        assert "c_fractions" in vp
        assert "source" in vp

    def test_variants_distinct_parameters(self):
        """Test that acidic/main/basic variants have distinct parameters."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name="VariantDistinct",
            pI=8.45,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
            ptm_profile={"deamidation_sites": 1, "oxidation_sites": 1},
        )
        vp = mapper.map_variants(protein)

        # All three should be different
        acidic_nu = vp["acidic"]["nu"]
        main_nu = vp["main"]["nu"]
        basic_nu = vp["basic"]["nu"]

        assert acidic_nu < main_nu  # Deamidation reduces nu (more acidic)
        assert main_nu < basic_nu   # Oxidation increases nu (more basic)

    def test_variant_resolution_capability(self):
        """Test variants produce delta_nu suitable for separation."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name="ResolutionTest",
            pI=8.45,
            MW_kDa=150.0,
            hydrophobicity=0.35,
            pH_working=7.0,
            ptm_profile={"deamidation_sites": 1, "oxidation_sites": 1},
        )
        vp = mapper.map_variants(protein)

        # Delta nu between variants should be >0.3 for Rs > 1.2
        delta_nu = vp["basic"]["nu"] - vp["acidic"]["nu"]
        assert delta_nu > 0.3, f"Delta nu {delta_nu} too small for resolution"


@pytest.mark.core
class TestPropertyMapperBispecific:
    """Test bispecific species mapping."""

    def test_map_bispecific_species(self):
        """Test mapping of bispecific assembly species."""
        mapper = PropertyMapper()

        species_dict = {
            "AA": {
                "pI": 8.5,
                "mw_kda": 150.0,
                "hydrophobicity": 0.35,
                "display_name": "Homodimer AA",
            },
            "AB": {
                "pI": 8.4,
                "mw_kda": 155.0,
                "hydrophobicity": 0.33,
                "display_name": "Heterodimer AB",
            },
            "BB": {
                "pI": 8.3,
                "mw_kda": 160.0,
                "hydrophobicity": 0.31,
                "display_name": "Homodimer BB",
            },
        }

        params = mapper.map_bispecific_species(species_dict)

        # Check all species mapped
        assert "AA" in params
        assert "AB" in params
        assert "BB" in params

        # Each should have SMA parameters
        for key in ("AA", "AB", "BB"):
            assert "nu" in params[key]
            assert "ka" in params[key]

    def test_bispecific_species_distinct_nu(self):
        """Test bispecific species have distinct nu values based on pI."""
        mapper = PropertyMapper()

        species_dict = {
            "AA": {"pI": 8.2, "mw_kda": 150.0, "hydrophobicity": 0.35},
            "AB": {"pI": 8.5, "mw_kda": 155.0, "hydrophobicity": 0.35},
            "BB": {"pI": 8.8, "mw_kda": 160.0, "hydrophobicity": 0.35},
        }

        params = mapper.map_bispecific_species(species_dict)

        # AA (lower pI) → lower nu
        # AB (medium pI) → medium nu
        # BB (higher pI) → higher nu
        assert params["AA"]["nu"] < params["AB"]["nu"]
        assert params["AB"]["nu"] < params["BB"]["nu"]


@pytest.mark.core
class TestPropertyMapperWithSampleIntents:
    """Test PropertyMapper with sample fixture intents."""

    def test_map_sample_mab(self, sample_mab_intent):
        """Test mapping with sample mAb from fixture."""
        mapper = PropertyMapper()

        # For fixture-based testing, we need actual biophysical properties
        # Create a representative mAb protein
        protein = ProteinProperties(
            name=sample_mab_intent.get("name", "Trastuzumab-test"),
            pI=8.45,  # Typical mAb
            MW_kDa=148.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )
        params = mapper.map(protein)

        # Basic sanity checks
        assert params["nu"] >= 2.0
        assert params["ka"] >= 0.3
        assert params["kd"] == 1000.0

    def test_map_sample_bispecific(self, sample_bispecific_intent):
        """Test with sample bispecific intent."""
        mapper = PropertyMapper()

        # Create representative bispecific protein
        protein = ProteinProperties(
            name=sample_bispecific_intent.get("name", "BispecTest"),
            pI=8.4,
            MW_kDa=155.0,
            hydrophobicity=0.33,
            pH_working=7.0,
        )
        params = mapper.map(protein)

        assert params["nu"] > 0
        assert params["ka"] > 0
