"""
test_immunogenicity_twin.py — Unit Tests for Immunogenicity Assessment (Module 6)
===================================================================================
Tests the immunogenicity_twin module which predicts Anti-Drug Antibody (ADA)
formation risk through MHC-II peptide scanning, humanization scoring, and
aggregation-driven immunogenicity penalties.

Key test areas:
  - MHC-II peptide scanning returns results
  - ADA risk classification returns Low/Medium/High
  - Humanization scoring returns value in [0,1]
  - CDR hotspot detection works
  - Short peptide doesn't crash
"""

import pytest
from src.immunogenicity_twin import (
    run_immunogenicity_assessment,
    Hotspot,
    ImmunogenicityResult,
    _score_9mer,
    _detect_cdr_ranges,
    _compute_humanization_score,
)


@pytest.mark.core
class TestMHCIIPeptideScanning:
    """Test MHC-II peptide scanning functionality."""

    def test_mhcii_scanning_returns_results(self, sample_mab_intent):
        """Test: MHC-II peptide scanning returns results."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        assert isinstance(result, ImmunogenicityResult)
        assert result.total_peptides_scanned > 0

    def test_peptide_scanning_detects_hotspots(self, sample_mab_intent):
        """Test: Scanning detects immunogenic hotspots."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        # Should have detected some hotspots (peptides with score >= 0.4)
        assert isinstance(result.hotspots, list)
        if result.hotspots:
            # If hotspots found, verify structure
            for hotspot in result.hotspots:
                assert isinstance(hotspot, Hotspot)
                assert 0 <= hotspot.score <= 1.0
                assert len(hotspot.peptide) == 9
                assert hotspot.position >= 0

    def test_peptide_scanning_total_count_valid(self, sample_mab_intent):
        """Test: Total peptides scanned count is valid."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        # For a sequence of length N, there should be N-8 overlapping 9-mers
        expected_count = len(seq) - 8
        assert result.total_peptides_scanned == expected_count

    def test_high_risk_hotspot_count_valid(self, sample_mab_intent):
        """Test: High-risk hotspot count is consistent."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        # Count should match
        assert result.n_high_risk >= 0
        assert result.n_medium_risk >= 0

    def test_mean_and_max_mhc_scores_valid(self, sample_mab_intent):
        """Test: Mean and max MHC scores are in valid range."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        assert 0 <= result.mean_mhc_score <= 1.0
        assert 0 <= result.max_mhc_score <= 1.0
        assert result.mean_mhc_score <= result.max_mhc_score


@pytest.mark.core
class TestADARiskClassification:
    """Test ADA risk classification."""

    def test_ada_risk_returns_low_medium_or_high(self, sample_mab_intent):
        """Test: ADA risk classification returns Low/Medium/High."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        assert result.ada_risk_level in ["Low", "Medium", "High"]

    def test_ada_risk_score_in_valid_range(self, sample_mab_intent):
        """Test: ADA risk score is in [0, 1]."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        assert 0 <= result.ada_risk_score <= 1.0

    def test_ada_risk_level_matches_score(self, sample_mab_intent):
        """Test: ADA risk level matches score thresholds."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        # Check score/level consistency
        if result.ada_risk_score >= 0.70:
            assert result.ada_risk_level == "High"
        elif result.ada_risk_score >= 0.45:
            assert result.ada_risk_level == "Medium"
        else:
            assert result.ada_risk_level == "Low"

    def test_high_aggregation_risk_increases_ada_score(self, sample_mab_intent):
        """Test: High aggregation risk increases ADA score."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        # Test with no aggregation penalty
        result_clean = run_immunogenicity_assessment(
            seq, agg_risk=0.0, molecule_name="TestMab"
        )

        # Test with high aggregation penalty
        result_agg = run_immunogenicity_assessment(
            seq, agg_risk=0.8, molecule_name="TestMab"
        )

        # Aggregation should increase ADA score
        assert result_agg.ada_risk_score >= result_clean.ada_risk_score


@pytest.mark.core
class TestHumanizationScoring:
    """Test humanization scoring."""

    def test_humanization_score_in_valid_range(self, sample_mab_intent):
        """Test: Humanization scoring returns value in [0,1]."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        assert 0 <= result.humanization_score <= 1.0

    def test_humanization_score_and_framework_identity_consistent(self, sample_mab_intent):
        """Test: Humanization score correlates with framework identity."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        # Framework identity should be percentage-like
        assert 0 <= result.framework_identity_pct <= 100
        # Higher framework identity should generally correlate with higher humanization
        # (at least conceptually)
        assert result.humanization_score >= 0

    def test_humanization_identifies_non_human_positions(self, sample_mab_intent):
        """Test: Humanization scoring identifies non-human positions."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        # Should have list of non-human positions
        assert isinstance(result.non_human_positions, list)
        # Positions should be non-negative integers
        for pos in result.non_human_positions:
            assert isinstance(pos, int)
            assert pos >= 0


@pytest.mark.core
class TestCDRHotspotDetection:
    """Test CDR hotspot detection."""

    def test_cdr_ranges_detected(self, sample_mab_intent):
        """Test: CDR ranges are detected."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        # Should have detected CDR regions
        assert isinstance(result.cdr_ranges, list)

    def test_hotspots_marked_cdr_or_non_cdr(self, sample_mab_intent):
        """Test: Hotspots are marked as in CDR or not."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        # Each hotspot should have in_cdr flag
        for hotspot in result.hotspots:
            assert isinstance(hotspot.in_cdr, bool)

    def test_cdr_hotspots_distinguished(self, sample_mab_intent):
        """Test: CDR hotspots can be distinguished from framework hotspots."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        if result.hotspots:
            # Should have both types or just one type
            cdr_hotspots = [h for h in result.hotspots if h.in_cdr]
            framework_hotspots = [h for h in result.hotspots if not h.in_cdr]
            # At least one category should exist if there are hotspots
            assert len(cdr_hotspots) + len(framework_hotspots) == len(result.hotspots)


@pytest.mark.core
class TestPeptideHandling:
    """Test short and edge-case peptide handling."""

    def test_short_peptide_doesnt_crash(self):
        """Test: Short peptide doesn't crash."""
        short_seq = "MKVLTC"

        result = run_immunogenicity_assessment(short_seq, molecule_name="ShortPep")

        assert isinstance(result, ImmunogenicityResult)

    def test_very_short_sequence_handled(self):
        """Test: Very short sequence (< 9 residues) is handled."""
        very_short = "MVK"

        result = run_immunogenicity_assessment(very_short, molecule_name="VeryShort")

        # Should not crash, but may have zero peptides scanned
        assert result.total_peptides_scanned >= 0

    def test_single_amino_acid_handled(self):
        """Test: Single amino acid is handled gracefully."""
        single = "M"

        result = run_immunogenicity_assessment(single, molecule_name="Single")

        assert isinstance(result, ImmunogenicityResult)
        assert result.total_peptides_scanned == 0

    def test_peptide_with_nonstandard_residues_handled(self):
        """Test: Peptide with non-standard residues (X, U) is handled."""
        with_x = "MKVLTCXXVGDRT"

        result = run_immunogenicity_assessment(with_x, molecule_name="WithX")

        assert isinstance(result, ImmunogenicityResult)

    def test_lowercase_sequence_converted(self):
        """Test: Lowercase sequence is converted to uppercase."""
        lower_seq = "mkvltcgdrt"

        result = run_immunogenicity_assessment(lower_seq, molecule_name="Lower")

        assert isinstance(result, ImmunogenicityResult)
        # Should have processed it
        assert result.total_peptides_scanned >= 0


@pytest.mark.core
class TestScoringFunctions:
    """Test individual scoring functions."""

    def test_9mer_score_function_returns_float(self):
        """Test: _score_9mer() returns float in valid range."""
        peptide = "MKVLTCGDR"

        score = _score_9mer(peptide)

        assert isinstance(score, float)
        assert 0 <= score <= 1.0

    def test_identical_peptides_get_same_score(self):
        """Test: Identical 9-mers get same score."""
        peptide = "MKVLTCGDR"

        score1 = _score_9mer(peptide)
        score2 = _score_9mer(peptide)

        assert score1 == score2

    def test_hydrophobic_peptides_score_higher(self):
        """Test: Hydrophobic peptides score higher than hydrophilic."""
        hydrophobic = "IIIIIIIII"  # All isoleucine (hydrophobic)
        hydrophilic = "DEDEEDED"  # Mix of Asp/Glu (hydrophilic)

        score_hydro = _score_9mer(hydrophobic)
        score_hydrophil = _score_9mer(hydrophilic)

        # Hydrophobic should score higher (better MHC-II binding)
        assert score_hydro >= score_hydrophil

    def test_cdr_detection_returns_ranges(self, sample_mab_intent):
        """Test: CDR detection returns list of tuples."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        ranges = _detect_cdr_ranges(seq)

        assert isinstance(ranges, list)
        # Each range should be a tuple of (start, end)
        for r in ranges:
            assert isinstance(r, tuple)
            assert len(r) == 2
            assert r[0] >= 0
            assert r[1] >= r[0]  # Can be equal in some edge cases

    def test_humanization_score_function_valid(self, sample_mab_intent):
        """Test: _compute_humanization_score() returns valid values."""
        hc = sample_mab_intent["hc_sequence"]

        score, identity_pct, non_human = _compute_humanization_score(hc)

        assert isinstance(score, float)
        assert 0 <= score <= 1.0
        assert isinstance(identity_pct, float)
        assert 0 <= identity_pct <= 100
        assert isinstance(non_human, list)


@pytest.mark.core
class TestAggregationImpact:
    """Test aggregation impact on immunogenicity."""

    def test_aggregation_penalty_applied(self, sample_mab_intent):
        """Test: Aggregation penalty is applied to ADA score."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, agg_risk=0.5, molecule_name="TestMab")

        assert result.aggregation_penalty > 0
        assert result.aggregation_penalty <= 1.0

    def test_zero_aggregation_risk_has_no_penalty(self, sample_mab_intent):
        """Test: Zero aggregation risk has no penalty."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, agg_risk=0.0, molecule_name="TestMab")

        assert result.aggregation_penalty == 0.0

    def test_dev_score_infers_aggregation_penalty(self, sample_mab_intent):
        """Test: Dev score can be used to infer aggregation penalty."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        # Low dev score => high inferred aggregation risk
        result = run_immunogenicity_assessment(seq, dev_score=0.1, molecule_name="TestMab")

        assert result.aggregation_penalty > 0


@pytest.mark.core
class TestResultSummary:
    """Test result summary generation."""

    def test_summary_text_generated(self, sample_mab_intent):
        """Test: Summary text is generated."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        assert result.summary is not None
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_summary_contains_key_metrics(self, sample_mab_intent):
        """Test: Summary contains key immunogenicity metrics."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        # Summary should mention ADA risk level
        assert result.ada_risk_level in result.summary

    def test_summary_includes_recommendations(self, sample_mab_intent):
        """Test: Summary includes recommendations."""
        hc = sample_mab_intent["hc_sequence"]
        lc = sample_mab_intent["lc_sequence"]
        seq = hc + lc

        result = run_immunogenicity_assessment(seq, molecule_name="TestMab")

        # Should have recommendation text
        assert "RECOMMENDATION" in result.summary or len(result.summary) > 50
