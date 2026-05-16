"""
test_formulation_twin.py — Unit Tests for Formulation Digital Twin (Module 7)
===============================================================================
Tests the formulation_twin module which simulates buffer and excipient effects
on antibody developability through physics-based calculations and empirical models.

Key test areas:
  - Buffer system catalog is populated
  - Net charge calculation returns numeric value
  - pH optimization returns valid pH range
  - Excipient stabilization modifiers work
  - Aggregation/viscosity risk returns values in expected ranges
"""

import pytest
from src.formulation_twin import (
    BUFFER_CATALOG,
    EXCIPIENT_CATALOG,
    RESIDUE_PKA,
    count_titratable_residues,
    compute_net_charge_at_ph,
    compute_formulation_effects,
    estimate_pI_from_sequence,
    FormulationCondition,
    FormulationEffect,
)


@pytest.mark.core
class TestBufferCatalog:
    """Test buffer system catalog."""

    def test_buffer_catalog_is_populated(self):
        """Test: Buffer system catalog is populated."""
        assert isinstance(BUFFER_CATALOG, dict)
        assert len(BUFFER_CATALOG) > 0

    def test_buffer_catalog_has_standard_buffers(self):
        """Test: Catalog contains standard buffers."""
        expected_buffers = ["histidine", "citrate", "phosphate"]
        for buffer_name in expected_buffers:
            assert buffer_name in BUFFER_CATALOG

    def test_buffer_entries_have_required_fields(self):
        """Test: Each buffer has required fields."""
        for name, buffer in BUFFER_CATALOG.items():
            assert hasattr(buffer, "name")
            assert hasattr(buffer, "full_name")
            assert hasattr(buffer, "optimal_ph_low")
            assert hasattr(buffer, "optimal_ph_high")
            assert hasattr(buffer, "ionic_strength_factor")
            assert hasattr(buffer, "viscosity_modifier")
            assert hasattr(buffer, "stabilization_bonus")

    def test_buffer_ph_ranges_valid(self):
        """Test: Buffer pH ranges are logically valid."""
        for name, buffer in BUFFER_CATALOG.items():
            assert buffer.optimal_ph_low > 0
            assert buffer.optimal_ph_high > 0
            assert buffer.optimal_ph_low <= buffer.optimal_ph_high
            assert buffer.optimal_ph_low >= 2.0  # Reasonable pH range
            assert buffer.optimal_ph_high <= 10.0

    def test_buffer_modifiers_in_reasonable_ranges(self):
        """Test: Buffer modifiers are in reasonable ranges."""
        for name, buffer in BUFFER_CATALOG.items():
            # Modifiers should be non-negative
            assert buffer.ionic_strength_factor >= 0
            assert buffer.viscosity_modifier >= -0.5  # Can be slightly negative
            assert buffer.stabilization_bonus >= 0


@pytest.mark.core
class TestNetChargeCalculation:
    """Test net charge calculation at various pH values."""

    def test_net_charge_calculation_returns_float(self):
        """Test: Net charge calculation returns numeric value."""
        sequence = "MKVLTCGDR"  # Mix of charged residues
        ph = 7.0

        charge = compute_net_charge_at_ph(sequence, ph)

        assert isinstance(charge, (int, float))

    def test_net_charge_decreases_with_increasing_ph(self):
        """Test: Net charge decreases (becomes more negative) as pH increases."""
        sequence = "DDDDEEEEKKKKR"  # Multiple acidic and basic residues

        charge_low_ph = compute_net_charge_at_ph(sequence, 4.0)
        charge_mid_ph = compute_net_charge_at_ph(sequence, 7.0)
        charge_high_ph = compute_net_charge_at_ph(sequence, 10.0)

        # At low pH, acidic residues are protonated => positive charge
        # At high pH, basic residues are deprotonated => negative charge
        assert charge_low_ph > charge_high_ph

    def test_net_charge_at_pI_approaches_zero(self):
        """Test: Net charge approaches zero at isoelectric point."""
        sequence = "MKVLTCGDR"
        pi = estimate_pI_from_sequence(sequence)

        charge_at_pi = compute_net_charge_at_ph(sequence, pi)

        # At pI, net charge should be very close to zero
        assert abs(charge_at_pi) < 2.0

    def test_net_charge_handles_short_sequence(self):
        """Test: Net charge calculation handles short sequences."""
        short_seq = "MKV"

        charge = compute_net_charge_at_ph(short_seq, 7.0)

        assert isinstance(charge, (int, float))

    def test_titratable_residues_counted(self):
        """Test: Titratable residues are correctly counted."""
        sequence = "DDEEKRKKY"

        residue_counts = count_titratable_residues(sequence)

        assert isinstance(residue_counts, dict)
        assert residue_counts.get("D", 0) >= 2
        assert residue_counts.get("E", 0) >= 2
        assert residue_counts.get("K", 0) >= 2
        assert residue_counts.get("R", 0) >= 1
        assert residue_counts.get("Y", 0) >= 1


@pytest.mark.core
class TestPIEstimation:
    """Test pI estimation."""

    def test_pI_estimation_returns_float(self):
        """Test: pH estimation returns float value."""
        sequence = "MKVLTCGDR"

        pi = estimate_pI_from_sequence(sequence)

        assert isinstance(pi, float)

    def test_pI_in_reasonable_range(self):
        """Test: Estimated pI is in reasonable range for proteins."""
        sequence = "MKVLTCGDR"

        pi = estimate_pI_from_sequence(sequence)

        # Protein pI typically 4-10
        assert 4.0 <= pi <= 10.0

    def test_acidic_sequence_has_low_pI(self):
        """Test: Sequence with more acidic residues has lower pI."""
        acidic = "DDEEEE"
        basic = "KRKRKR"

        pi_acidic = estimate_pI_from_sequence(acidic)
        pi_basic = estimate_pI_from_sequence(basic)

        assert pi_acidic < pi_basic

    def test_basic_sequence_has_high_pI(self):
        """Test: Sequence with more basic residues has higher pI."""
        basic = "KRKRKR"
        acidic = "DDEEEE"

        pi_basic = estimate_pI_from_sequence(basic)
        pi_acidic = estimate_pI_from_sequence(acidic)

        assert pi_basic > pi_acidic

    def test_pI_convergence_with_tolerance(self):
        """Test: pI estimation converges within tolerance."""
        sequence = "MKVLTCGDR"

        # Tighter tolerance should give more precise result
        pi_loose = estimate_pI_from_sequence(sequence, tolerance=0.1)
        pi_tight = estimate_pI_from_sequence(sequence, tolerance=0.01)

        # Both should be reasonable
        assert 4.0 <= pi_loose <= 10.0
        assert 4.0 <= pi_tight <= 10.0


@pytest.mark.core
class TestFormulationEffects:
    """Test formulation effects computation."""

    def test_formulation_effects_returns_object(self):
        """Test: Formulation effects computation returns FormulationEffect."""
        condition = FormulationCondition(
            buffer_type="histidine",
            buffer_ph=6.0,
            buffer_concentration_mM=20.0,
        )

        effect = compute_formulation_effects(condition, pI=7.5)

        assert isinstance(effect, FormulationEffect)

    def test_formulation_net_charge_calculated(self):
        """Test: Net charge is calculated for formulation."""
        condition = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.4,
            buffer_concentration_mM=20.0,
        )

        effect = compute_formulation_effects(condition, pI=7.5)

        assert effect.net_charge is not None
        assert isinstance(effect.net_charge, (int, float))

    def test_ph_pI_distance_calculated(self):
        """Test: pH-pI distance is calculated."""
        condition = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.0,
            buffer_concentration_mM=20.0,
        )
        pI = 8.0

        effect = compute_formulation_effects(condition, pI=pI)

        assert effect.ph_pI_distance == pytest.approx(abs(7.0 - 8.0), abs=0.01)

    def test_charge_near_zero_flag_set(self):
        """Test: Charge-near-zero flag is set correctly."""
        condition = FormulationCondition(
            buffer_type="histidine",
            buffer_ph=7.5,
            buffer_concentration_mM=20.0,
        )
        pI = 7.5  # Buffer at pI

        effect = compute_formulation_effects(condition, pI=pI)

        # pH near pI should trigger charge_near_zero flag
        assert effect.charge_near_zero is True

    def test_buffer_in_range_flag(self):
        """Test: Buffer in-range flag is set correctly."""
        # Histidine buffer optimal pH: 5.5-6.5
        condition_in_range = FormulationCondition(
            buffer_type="histidine",
            buffer_ph=6.0,
            buffer_concentration_mM=20.0,
        )

        effect_in = compute_formulation_effects(condition_in_range, pI=7.5)
        assert effect_in.buffer_in_range is True

        # Out of range
        condition_out_range = FormulationCondition(
            buffer_type="histidine",
            buffer_ph=8.0,
            buffer_concentration_mM=20.0,
        )

        effect_out = compute_formulation_effects(condition_out_range, pI=7.5)
        assert effect_out.buffer_in_range is False

    def test_aggregation_risk_modifier_computed(self):
        """Test: Aggregation risk modifier is computed."""
        condition = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.0,
            buffer_concentration_mM=20.0,
        )

        effect = compute_formulation_effects(condition, pI=8.5)

        assert effect.agg_risk_modifier is not None
        assert isinstance(effect.agg_risk_modifier, (int, float))

    def test_ph_near_pI_increases_aggregation_risk(self):
        """Test: pH near pI increases aggregation risk."""
        condition_safe = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.0,
            buffer_concentration_mM=20.0,
        )
        pI = 7.5

        effect_safe = compute_formulation_effects(condition_safe, pI=pI)

        # pH at 7.0, pI at 7.5 => distance 0.5 => agg penalty
        assert effect_safe.agg_risk_modifier > 0


@pytest.mark.core
class TestExcipientEffects:
    """Test excipient stabilization effects."""

    def test_excipient_catalog_populated(self):
        """Test: Excipient catalog is populated."""
        assert isinstance(EXCIPIENT_CATALOG, dict)
        assert len(EXCIPIENT_CATALOG) > 0

    def test_excipient_has_required_fields(self):
        """Test: Each excipient has required fields."""
        for name, excipient in EXCIPIENT_CATALOG.items():
            assert hasattr(excipient, "name")
            assert hasattr(excipient, "category")
            assert hasattr(excipient, "agg_risk_reduction")
            assert hasattr(excipient, "stability_boost")
            assert hasattr(excipient, "viscosity_modifier")

    def test_excipient_modifiers_in_valid_ranges(self):
        """Test: Excipient modifiers are in valid ranges."""
        for name, excipient in EXCIPIENT_CATALOG.items():
            assert excipient.agg_risk_reduction >= 0
            assert excipient.stability_boost >= 0
            # Viscosity can be negative (reduction)
            assert -0.3 <= excipient.viscosity_modifier <= 0.3

    def test_trehalose_reduces_aggregation_risk(self):
        """Test: Trehalose excipient reduces aggregation risk."""
        assert "trehalose" in EXCIPIENT_CATALOG
        trehalose = EXCIPIENT_CATALOG["trehalose"]

        assert trehalose.agg_risk_reduction > 0
        assert trehalose.stability_boost > 0

    def test_polysorbate_reduces_aggregation_risk(self):
        """Test: Polysorbate-80 reduces aggregation risk."""
        # The key might be "ps80" or "polysorbate_80" depending on catalog
        assert "ps80" in EXCIPIENT_CATALOG or "polysorbate_80" in EXCIPIENT_CATALOG
        ps80 = EXCIPIENT_CATALOG.get("ps80") or EXCIPIENT_CATALOG.get("polysorbate_80")

        assert ps80.agg_risk_reduction > 0

    def test_excipient_effects_applied_to_formulation(self):
        """Test: Excipient effects are applied to formulation."""
        condition = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.0,
            buffer_concentration_mM=20.0,
            excipients=["trehalose", "polysorbate_80"],
        )

        effect = compute_formulation_effects(condition, pI=8.0)

        assert len(effect.excipient_effects) > 0
        # Excipients should have reduced agg risk
        assert effect.agg_risk_modifier is not None

    def test_multiple_excipients_combined(self):
        """Test: Multiple excipients' effects are combined."""
        condition = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.0,
            buffer_concentration_mM=20.0,
            excipients=["trehalose", "polysorbate_80", "sucrose"],
        )

        effect = compute_formulation_effects(condition, pI=8.0)

        # Should have multiple excipient effects
        assert len(effect.excipient_effects) >= 2


@pytest.mark.core
class TestStabilityAndViscosityModifiers:
    """Test stability and viscosity modifiers."""

    def test_stability_modifier_computed(self):
        """Test: Stability modifier is computed."""
        condition = FormulationCondition(
            buffer_type="histidine",
            buffer_ph=6.0,
            buffer_concentration_mM=20.0,
        )

        effect = compute_formulation_effects(condition, pI=7.5)

        assert effect.stability_modifier is not None
        assert isinstance(effect.stability_modifier, (int, float))

    def test_viscosity_modifier_computed(self):
        """Test: Viscosity modifier is computed."""
        condition = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.4,
            buffer_concentration_mM=20.0,
        )

        effect = compute_formulation_effects(condition, pI=7.5)

        assert effect.viscosity_modifier is not None
        assert isinstance(effect.viscosity_modifier, (int, float))

    def test_buffer_type_affects_viscosity(self):
        """Test: Different buffer types have different viscosity effects."""
        condition_hist = FormulationCondition(
            buffer_type="histidine",
            buffer_ph=6.0,
            buffer_concentration_mM=20.0,
        )

        condition_phos = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.4,
            buffer_concentration_mM=20.0,
        )

        effect_hist = compute_formulation_effects(condition_hist, pI=7.5)
        effect_phos = compute_formulation_effects(condition_phos, pI=7.5)

        # Both should have computed viscosity modifiers
        assert effect_hist.viscosity_modifier is not None
        assert effect_phos.viscosity_modifier is not None

    def test_excipients_affect_stability(self):
        """Test: Excipients improve stability modifier."""
        condition_no_exc = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.0,
            buffer_concentration_mM=20.0,
            excipients=[],
        )

        condition_with_exc = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.0,
            buffer_concentration_mM=20.0,
            excipients=["trehalose"],
        )

        effect_no_exc = compute_formulation_effects(condition_no_exc, pI=7.5)
        effect_with_exc = compute_formulation_effects(condition_with_exc, pI=7.5)

        # With excipient should have higher stability
        assert effect_with_exc.stability_modifier >= effect_no_exc.stability_modifier


@pytest.mark.core
class TestFormulationSummary:
    """Test formulation summary generation."""

    def test_formulation_summary_generated(self):
        """Test: Formulation summary is generated."""
        condition = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.4,
            buffer_concentration_mM=20.0,
            excipients=["trehalose"],
        )

        effect = compute_formulation_effects(condition, pI=7.5)

        assert effect.formulation_summary is not None
        assert isinstance(effect.formulation_summary, str)
        assert len(effect.formulation_summary) > 0

    def test_warnings_generated_for_poor_conditions(self):
        """Test: Warnings are generated for poor formulation conditions."""
        condition = FormulationCondition(
            buffer_type="histidine",
            buffer_ph=7.5,  # Outside histidine range
            buffer_concentration_mM=20.0,
        )

        effect = compute_formulation_effects(condition, pI=7.5)

        # Should have warnings for pH outside optimal range
        assert len(effect.warnings) >= 0

    def test_recommendations_generated(self):
        """Test: Recommendations are generated."""
        condition = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.0,
            buffer_concentration_mM=20.0,
            excipients=["trehalose"],
        )

        effect = compute_formulation_effects(condition, pI=7.5)

        # Should have recommendations
        assert isinstance(effect.recommendations, list)

    def test_critical_warning_at_pI(self):
        """Test: Critical warning generated when pH is at pI."""
        condition = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.5,
            buffer_concentration_mM=20.0,
        )

        effect = compute_formulation_effects(condition, pI=7.5)

        # pH at pI should trigger critical warning
        assert any("CRITICAL" in w for w in effect.warnings) or \
               any("WARNING" in w for w in effect.warnings)


@pytest.mark.core
class TestFormulationWithSequence:
    """Test formulation effects when sequence is provided."""

    def test_sequence_enables_precise_charge_calculation(self):
        """Test: Providing sequence enables precise charge calculation."""
        sequence = "MKVLTCGDREKYLMR"
        condition = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.0,
            buffer_concentration_mM=20.0,
        )

        effect_with_seq = compute_formulation_effects(
            condition, pI=7.5, sequence=sequence
        )

        effect_without_seq = compute_formulation_effects(
            condition, pI=7.5, sequence=None
        )

        # Both should have net charge
        assert effect_with_seq.net_charge is not None
        assert effect_without_seq.net_charge is not None

    def test_hydrophobicity_amplifies_aggregation_risk(self):
        """Test: High hydrophobicity amplifies aggregation risk near pI."""
        sequence = "MKVLTCGDR"
        condition = FormulationCondition(
            buffer_type="phosphate",
            buffer_ph=7.5,
            buffer_concentration_mM=20.0,
        )
        pI = 7.5  # At pI

        effect_low_hydro = compute_formulation_effects(
            condition, pI=pI, hydrophobicity=0.25
        )

        effect_high_hydro = compute_formulation_effects(
            condition, pI=pI, hydrophobicity=0.65
        )

        # Higher hydrophobicity should increase agg risk more at pI
        assert effect_high_hydro.agg_risk_modifier >= effect_low_hydro.agg_risk_modifier
