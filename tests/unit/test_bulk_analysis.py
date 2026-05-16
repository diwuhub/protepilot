"""
test_bulk_analysis.py  ·  B11 — Comprehensive tests for bulk analysis modules.

Tests bulk_schema (B7), bulk_runner (B8), and bulk_summary (B9).
"""

import csv
import io
import json
import pytest
from unittest.mock import patch, MagicMock

# All tests in this file exercise the bulk analysis pipeline modules
# (bulk_schema, bulk_runner, bulk_summary) and run without torch/sklearn.
pytestmark = [pytest.mark.bulk, pytest.mark.core]

from src.bulk_schema import (
    BATCH_TYPES, BatchTypeSpec, BulkRow, BulkParseResult,
    generate_csv_template, parse_bulk_csv, row_to_intent,
    _clean_sequence, _validate_sequence, _build_chains_and_assembly,
)
from src.bulk_runner import (
    BulkRowResult, BulkBatchResult, run_bulk_analysis, _extract_results,
)
from src.bulk_summary import (
    export_summary_csv, export_summary_json,
    generate_display_stats, rank_candidates,
)
import src.agents as _agents_mod


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

# Realistic HC/LC sequences (truncated but valid)
HC_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRY"
LC_SEQ = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVP"

PEPTIDE_SEQ = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"

VHH_SEQ = "QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWS"


@pytest.fixture
def mab_csv():
    return (
        "name,HC,LC\n"
        f"Trastuzumab_v1,{HC_SEQ},{LC_SEQ}\n"
        f"Adalimumab_v1,{HC_SEQ}ADDD,{LC_SEQ}KKKK\n"
        f"Candidate_3,{HC_SEQ}EEE,{LC_SEQ}RRR\n"
    )


@pytest.fixture
def peptide_csv():
    return f"name,peptide\nSema,{PEPTIDE_SEQ}\nGLP1,HAEGTFTSDVSSYLEG\n"


@pytest.fixture
def bispecific_csv():
    return (
        "name,HC1,LC1,HC2,LC2\n"
        f"Bsp_1,{HC_SEQ},{LC_SEQ},{HC_SEQ}GGG,{LC_SEQ}AAA\n"
    )

@pytest.fixture
def bispecific_3chain_csv():
    return (
        "name,HC,LC,scfv_arm\n"
        f"BiTE_1,{HC_SEQ},{LC_SEQ},{VHH_SEQ}\n"
    )


@pytest.fixture
def bad_csv_missing_col():
    return "name,sequence\nBad,ACDEFG\n"


@pytest.fixture
def bad_csv_invalid_aa():
    return f"name,HC,LC\nBadAA,{HC_SEQ}123XZ,{LC_SEQ}\n"


# ═══════════════════════════════════════════════════════════════════════
#  B7: bulk_schema tests
# ═══════════════════════════════════════════════════════════════════════

class TestBatchTypes:
    """Test that all batch types are correctly registered."""

    def test_all_types_present(self):
        expected = {"canonical_mab", "bispecific_4chain", "bispecific_3chain",
                    "bispecific_2chain", "scfv", "nanobody",
                    "fc_fusion", "peptide", "adc", "fusion_protein"}
        assert set(BATCH_TYPES.keys()) == expected

    def test_each_type_has_required_fields(self):
        for key, spec in BATCH_TYPES.items():
            assert spec.display_name, f"{key} missing display_name"
            assert spec.molecule_class, f"{key} missing molecule_class"
            assert len(spec.required_columns) > 0, f"{key} has no required_columns"
            assert spec.assembly_description, f"{key} missing assembly_description"

    def test_canonical_mab_spec(self):
        spec = BATCH_TYPES["canonical_mab"]
        assert spec.required_columns == ("hc", "lc")
        assert spec.chain_count == 2
        assert "2×HC" in spec.assembly_description

    def test_bispecific_4chain_spec(self):
        spec = BATCH_TYPES["bispecific_4chain"]
        assert spec.required_columns == ("hc1", "lc1", "hc2", "lc2")
        assert spec.chain_count == 4

    def test_bispecific_3chain_spec(self):
        spec = BATCH_TYPES["bispecific_3chain"]
        assert spec.required_columns == ("hc", "lc", "scfv_arm")
        assert spec.chain_count == 3

    def test_bispecific_2chain_spec(self):
        spec = BATCH_TYPES["bispecific_2chain"]
        assert spec.required_columns == ("chain_a", "chain_b")
        assert spec.chain_count == 2


class TestCSVTemplate:
    """Test CSV template generation."""

    def test_template_has_header(self):
        tmpl = generate_csv_template("canonical_mab")
        reader = csv.reader(io.StringIO(tmpl))
        header = next(reader)
        assert "name" in header
        assert "hc" in [h.lower() for h in header]
        assert "lc" in [h.lower() for h in header]

    def test_template_has_example_row(self):
        tmpl = generate_csv_template("peptide")
        lines = tmpl.strip().split("\n")
        assert len(lines) >= 2  # header + example

    def test_all_types_generate_templates(self):
        for key in BATCH_TYPES:
            tmpl = generate_csv_template(key)
            assert "name" in tmpl

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            generate_csv_template("nonexistent_type")


class TestSequenceValidation:
    """Test sequence cleaning and validation helpers."""

    def test_clean_sequence(self):
        assert _clean_sequence("  EVQ LVE  ") == "EVQLVE"
        assert _clean_sequence("E1V2Q3") == "EVQ"
        assert _clean_sequence("") == ""

    def test_validate_good_sequence(self):
        assert _validate_sequence("ACDEFGHIKLMNPQRSTVWY", "hc", 0) is None

    def test_validate_empty(self):
        err = _validate_sequence("", "hc", 0)
        assert err and "empty" in err.lower()

    def test_validate_too_short(self):
        err = _validate_sequence("AC", "hc", 0, min_len=5)
        assert err and "too short" in err.lower()

    def test_validate_invalid_aa(self):
        err = _validate_sequence("ACDEFXZ", "hc", 0)
        assert err and "invalid" in err.lower()


class TestCSVParsing:
    """Test CSV parsing and validation."""

    def test_parse_mab_csv(self, mab_csv):
        result = parse_bulk_csv(mab_csv, "canonical_mab")
        assert result.is_ok
        assert result.n_total == 3
        assert result.n_valid == 3

    def test_parse_peptide_csv(self, peptide_csv):
        result = parse_bulk_csv(peptide_csv, "peptide")
        assert result.is_ok
        assert result.n_valid == 2

    def test_parse_bispecific_4chain_csv(self, bispecific_csv):
        result = parse_bulk_csv(bispecific_csv, "bispecific_4chain")
        assert result.is_ok
        assert result.n_valid == 1
        row = result.valid_rows[0]
        assert len(row.sequences) == 4
        assert len(row.assembly_chains) == 4

    def test_parse_bispecific_3chain_csv(self, bispecific_3chain_csv):
        result = parse_bulk_csv(bispecific_3chain_csv, "bispecific_3chain")
        assert result.is_ok
        assert result.n_valid == 1
        row = result.valid_rows[0]
        assert len(row.sequences) == 3
        assert len(row.assembly_chains) == 3

    def test_missing_column_error(self, bad_csv_missing_col):
        result = parse_bulk_csv(bad_csv_missing_col, "canonical_mab")
        assert not result.is_ok
        assert "Missing required" in result.errors[0]

    def test_invalid_aa_skipped(self, bad_csv_invalid_aa):
        result = parse_bulk_csv(bad_csv_invalid_aa, "canonical_mab")
        # Row should have error
        assert result.n_total == 1
        assert result.n_valid == 0

    def test_unknown_batch_type(self):
        result = parse_bulk_csv("name,x\nA,B\n", "nonexistent")
        assert not result.is_ok

    def test_empty_csv(self):
        result = parse_bulk_csv("name,hc,lc\n", "canonical_mab")
        assert not result.is_ok  # no data rows

    def test_max_rows_limit(self):
        rows = "name,peptide\n"
        for i in range(100):
            rows += f"pep_{i},{PEPTIDE_SEQ}\n"
        result = parse_bulk_csv(rows, "peptide", max_rows=10)
        assert result.n_total == 10
        assert len(result.warnings) > 0

    def test_auto_name_if_missing(self):
        csv_text = f"name,peptide\n,{PEPTIDE_SEQ}\n"
        result = parse_bulk_csv(csv_text, "peptide")
        assert result.rows[0].name == "Molecule_1"


class TestChainsAndAssembly:
    """Test chain and assembly structure building."""

    def test_mab_assembly_stoichiometry(self, mab_csv):
        result = parse_bulk_csv(mab_csv, "canonical_mab")
        row = result.valid_rows[0]
        hc_asm = [c for c in row.assembly_chains if c["type"] in ("HC", "Heavy")]
        lc_asm = [c for c in row.assembly_chains if c["type"] in ("LC", "Light")]
        assert len(hc_asm) == 1 and hc_asm[0]["copy_number"] == 2
        assert len(lc_asm) == 1 and lc_asm[0]["copy_number"] == 2

    def test_peptide_single_chain(self, peptide_csv):
        result = parse_bulk_csv(peptide_csv, "peptide")
        row = result.valid_rows[0]
        assert len(row.chains) == 1
        assert row.chains[0]["type"] == "Peptide"

    def test_combined_sequence(self, mab_csv):
        result = parse_bulk_csv(mab_csv, "canonical_mab")
        row = result.valid_rows[0]
        # Stoichiometric assembly: 2×HC + 2×LC for canonical_mab
        assert row.combined_sequence == HC_SEQ * 2 + LC_SEQ * 2

    def test_bispecific_3chain_lc_stoichiometry(self, bispecific_3chain_csv):
        """Bispecific 3-chain: LC should be copy_number=2 (shared across both arms)."""
        result = parse_bulk_csv(bispecific_3chain_csv, "bispecific_3chain")
        row = result.valid_rows[0]
        lc_asm = [c for c in row.assembly_chains if c["type"] in ("LC", "Light")]
        assert len(lc_asm) == 1, "Expected exactly one LC chain entry"
        assert lc_asm[0]["copy_number"] == 2, (
            f"LC copy_number should be 2 for 3-chain bispecific, got {lc_asm[0]['copy_number']}"
        )
        # Combined sequence should have LC×2
        expected = HC_SEQ + LC_SEQ * 2 + VHH_SEQ
        assert row.combined_sequence == expected

    def test_bispecific_4chain_all_copy_1(self, bispecific_csv):
        """Bispecific 4-chain: all chains copy_number=1 (distinct arms)."""
        result = parse_bulk_csv(bispecific_csv, "bispecific_4chain")
        row = result.valid_rows[0]
        for ch in row.assembly_chains:
            assert ch["copy_number"] == 1, (
                f"Chain {ch['type']} should have copy_number=1, got {ch['copy_number']}"
            )

    def test_chain_length_warning_logged(self, mab_csv, caplog):
        """Short test chains should trigger advisory warning (never block)."""
        import logging
        with caplog.at_level(logging.WARNING, logger="ProtePilot.BulkSchema"):
            result = parse_bulk_csv(mab_csv, "canonical_mab")
        assert result.is_ok, "Short chains should NOT block analysis"
        assert result.n_valid > 0, "Rows must still be valid"
        assert any("truncated" in m or "incomplete" in m for m in caplog.messages), (
            "Expected chain-length advisory warning for 60aa test sequences"
        )

    def test_glycan_mw_correction_multichain(self):
        """MW should include glycan correction for Fc-containing molecules."""
        from src.feature_registry import compute_features
        # Use realistic-length HC fragment (has N-glycosylation motifs)
        hc = HC_SEQ * 7  # ~420 aa (close to real HC)
        lc = LC_SEQ * 4  # ~240 aa
        fs = compute_features(
            sequence=hc + lc,
            molecule_class="canonical_mab",
            chains=[
                {"sequence": hc, "chain_type": "HC", "copy_number": 2},
                {"sequence": lc, "chain_type": "LC", "copy_number": 2},
            ],
        )
        mw = fs.value("mw_kda")
        mw_per_chain = fs.value("mw_kda_per_chain")
        assert mw is not None and mw > 0
        # Stoichiometric MW should be > single-chain MW (assembly + glycan)
        assert mw > mw_per_chain, (
            f"Stoichiometric MW ({mw:.1f}) should exceed per-chain MW ({mw_per_chain:.1f})"
        )


class TestRowToIntent:
    """Test conversion of BulkRow to pipeline-compatible intent dict."""

    def test_intent_has_required_keys(self, mab_csv):
        result = parse_bulk_csv(mab_csv, "canonical_mab")
        row = result.valid_rows[0]
        spec = result.batch_type

        # Mock biophysical_features at the import target inside row_to_intent
        with patch.dict("sys.modules", {"src.biophysical_features": MagicMock(
                compute_biophysical_features=MagicMock(side_effect=Exception("mocked")))}):
            intent = row_to_intent(row, spec)

        assert "name" in intent
        assert "pI" in intent
        assert "mw" in intent
        assert "sequence" in intent
        assert "chains" in intent
        assert "assembly_chains" in intent
        assert intent["source"] == "bulk_csv"
        assert intent["molecule_class"] == "canonical_mab"

    def test_intent_mw_from_stoichiometry(self, mab_csv):
        result = parse_bulk_csv(mab_csv, "canonical_mab")
        row = result.valid_rows[0]
        spec = result.batch_type

        intent = row_to_intent(row, spec)

        # MW should be stoichiometric (HC×2 + LC×2) computed via
        # feature_registry.compute_features() which uses monoisotopic
        # residue masses + glycan correction for canonical_mab (~2.4 kDa
        # per Fc glycan site).  For these short test sequences the result
        # is approximately 35 kDa.  We verify it is positive, in the right
        # order of magnitude, and greater than the naive per-residue minimum.
        n_residues = len(HC_SEQ) * 2 + len(LC_SEQ) * 2
        naive_min = n_residues * 0.100  # absolute floor
        assert intent["mw"] > naive_min, (
            f"MW {intent['mw']:.2f} below naive minimum {naive_min:.1f}"
        )
        assert intent["mw"] < 200.0, (
            f"MW {intent['mw']:.2f} unreasonably large"
        )


# ═══════════════════════════════════════════════════════════════════════
#  B8: bulk_runner tests
# ═══════════════════════════════════════════════════════════════════════

class TestBulkRowResult:
    """Test BulkRowResult data class."""

    def test_summary_dict(self):
        r = BulkRowResult(
            row_index=0, name="Test", status="success",
            developability_score=0.45, developability_grade="Medium",
            agg_risk=0.3, stability=0.5, viscosity_risk=0.2,
            molecule_class="canonical_mab", wall_time=1.5,
        )
        d = r.to_summary_dict()
        assert d["name"] == "Test"
        assert d["dev_score"] == 0.45
        assert d["status"] == "success"

    def test_summary_dict_error(self):
        r = BulkRowResult(
            row_index=0, name="Fail", status="error",
            error_message="Pipeline failed",
        )
        d = r.to_summary_dict()
        assert d["error"] == "Pipeline failed"
        assert d["dev_score"] == ""


class TestBulkBatchResult:
    """Test BulkBatchResult aggregate stats."""

    def _make_batch(self):
        br = BulkBatchResult(batch_type="canonical_mab", molecule_class="canonical_mab")
        br.results = [
            BulkRowResult(0, "A", "success", developability_score=0.2, developability_grade="Low"),
            BulkRowResult(1, "B", "success", developability_score=0.5, developability_grade="Medium"),
            BulkRowResult(2, "C", "success", developability_score=0.8, developability_grade="High"),
            BulkRowResult(3, "D", "error", error_message="fail"),
            BulkRowResult(4, "E", "skipped", error_message="bad seq"),
        ]
        br.wall_time_total = 5.0
        return br

    def test_counts(self):
        br = self._make_batch()
        assert br.n_total == 5
        assert br.n_success == 3
        assert br.n_error == 1
        assert br.n_skipped == 1
        assert abs(br.success_rate - 0.6) < 0.01

    def test_summary_stats(self):
        br = self._make_batch()
        stats = br.summary_stats()
        assert stats["n_scored"] == 3
        assert abs(stats["mean_score"] - 0.5) < 0.01
        assert stats["n_low_risk"] == 1
        assert stats["n_high_risk"] == 1


class TestExtractResults:
    """Test _extract_results helper."""

    def test_extracts_dev_score(self):
        raw = {
            "developability": {
                "score": {"score": 0.42, "grade": "Medium"},
                "predictions": {
                    "agg_risk": 0.3,
                    "stability": 0.6,
                    "viscosity_risk": 0.1,
                },
            }
        }
        ex = _extract_results(raw)
        assert ex["developability_score"] == 0.42
        assert ex["developability_grade"] == "Medium"
        assert ex["agg_risk"] == 0.3

    def test_missing_developability(self):
        ex = _extract_results({"status": "success"})
        assert "developability_score" not in ex


class TestRunBulkAnalysis:
    """Test the bulk runner with mocked pipeline."""

    def _make_mock_intent(self, name="Test", row_index=0):
        """Create a mock intent dict."""
        return {
            "name": name, "pI": 7.5, "mw": 50.0, "hydrophobicity": 0.35,
            "pH_working": 7.0, "deam_sites": 3, "ox_sites": 2,
            "cysteine_count": 5, "gradient_slope": 15.0,
            "source": "bulk_csv", "sequence": HC_SEQ,
            "seq_length": len(HC_SEQ), "gravy": -0.3,
            "chains": [], "assembly_chains": [],
            "molecule_class": "canonical_mab",
            "bulk_row_index": row_index, "bulk_metadata": {},
        }

    def test_run_with_mock_pipeline(self, mab_csv):
        parse_result = parse_bulk_csv(mab_csv, "canonical_mab")

        mock_pipeline_result = {
            "status": "success",
            "developability": {
                "score": {"score": 0.35, "grade": "Medium"},
                "predictions": {
                    "agg_risk": 0.25,
                    "stability": 0.55,
                    "viscosity_risk": 0.15,
                },
            },
            "wall_time_total": 0.5,
        }

        # Mock both row_to_intent (avoids biophysical import) and PharmaAgentManager
        mock_manager = MagicMock()
        mock_manager.run_developability_pipeline.return_value = mock_pipeline_result

        with patch("src.bulk_runner.row_to_intent", side_effect=lambda r, s: self._make_mock_intent(r.name, r.row_index)), \
             patch.object(_agents_mod, "PharmaAgentManager", return_value=mock_manager):

            batch_result = run_bulk_analysis(parse_result)

        assert batch_result.n_total == 3
        assert batch_result.n_success == 3
        assert batch_result.results[0].developability_score == 0.35

    def test_run_with_row_error(self):
        """Test that rows with parse errors are skipped."""
        csv_text = f"name,HC,LC\nGood,{HC_SEQ},{LC_SEQ}\nBad,AB,{LC_SEQ}\n"
        parse_result = parse_bulk_csv(csv_text, "canonical_mab")

        mock_result = {
            "status": "success",
            "developability": {
                "score": {"score": 0.4, "grade": "Medium"},
                "predictions": {"agg_risk": 0.2, "stability": 0.5, "viscosity_risk": 0.1},
            },
        }

        mock_manager = MagicMock()
        mock_manager.run_developability_pipeline.return_value = mock_result

        with patch("src.bulk_runner.row_to_intent", side_effect=lambda r, s: self._make_mock_intent(r.name, r.row_index)), \
             patch.object(_agents_mod, "PharmaAgentManager", return_value=mock_manager):

            batch_result = run_bulk_analysis(parse_result)

        assert batch_result.n_skipped == 1
        assert batch_result.n_success == 1

    def test_run_with_pipeline_exception(self, mab_csv):
        """Test error isolation: one row exception doesn't stop batch."""
        parse_result = parse_bulk_csv(mab_csv, "canonical_mab")

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated failure")
            return {
                "status": "success",
                "developability": {
                    "score": {"score": 0.3, "grade": "Low"},
                    "predictions": {"agg_risk": 0.1, "stability": 0.7, "viscosity_risk": 0.05},
                },
            }

        mock_manager = MagicMock()
        mock_manager.run_developability_pipeline.side_effect = side_effect

        with patch("src.bulk_runner.row_to_intent", side_effect=lambda r, s: self._make_mock_intent(r.name, r.row_index)), \
             patch.object(_agents_mod, "PharmaAgentManager", return_value=mock_manager):

            batch_result = run_bulk_analysis(parse_result)

        assert batch_result.n_success == 2
        assert batch_result.n_error == 1
        err_row = [r for r in batch_result.results if r.status == "error"][0]
        assert "RuntimeError" in err_row.error_message

    def test_progress_callback(self, mab_csv):
        parse_result = parse_bulk_csv(mab_csv, "canonical_mab")
        calls = []

        mock_result = {
            "status": "success",
            "developability": {
                "score": {"score": 0.3, "grade": "Low"},
                "predictions": {"agg_risk": 0.1, "stability": 0.7, "viscosity_risk": 0.05},
            },
        }

        mock_manager = MagicMock()
        mock_manager.run_developability_pipeline.return_value = mock_result

        with patch("src.bulk_runner.row_to_intent", side_effect=lambda r, s: self._make_mock_intent(r.name, r.row_index)), \
             patch.object(_agents_mod, "PharmaAgentManager", return_value=mock_manager):

            run_bulk_analysis(parse_result, progress_callback=lambda c, t, n: calls.append((c, t, n)))

        assert len(calls) == 4  # 3 rows + 1 "Complete"
        assert calls[-1][2] == "Complete"


# ═══════════════════════════════════════════════════════════════════════
#  B9: bulk_summary tests
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_batch_result():
    br = BulkBatchResult(
        batch_type="canonical_mab",
        molecule_class="canonical_mab",
        started_at="2026-03-14 10:00:00",
        finished_at="2026-03-14 10:01:00",
        wall_time_total=60.0,
    )
    # Use canonical "Low Risk" / "Medium Risk" / "High Risk" grade strings
    # (as produced by grade_to_risk_label() in production — P1 fix).
    # composite_dev_grade is set as primary; developability_grade as fallback.
    br.results = [
        BulkRowResult(0, "mAb_A", "success",
                      composite_dev_score=0.15, composite_dev_grade="Low Risk",
                      developability_score=0.15, developability_grade="Low Risk",
                      agg_risk=0.1, stability=0.8,
                      viscosity_risk=0.05, molecule_class="canonical_mab"),
        BulkRowResult(1, "mAb_B", "success",
                      composite_dev_score=0.55, composite_dev_grade="Medium Risk",
                      developability_score=0.55, developability_grade="Medium Risk",
                      agg_risk=0.4, stability=0.5,
                      viscosity_risk=0.3, molecule_class="canonical_mab"),
        BulkRowResult(2, "mAb_C", "success",
                      composite_dev_score=0.85, composite_dev_grade="High Risk",
                      developability_score=0.85, developability_grade="High Risk",
                      agg_risk=0.7, stability=0.2,
                      viscosity_risk=0.6, molecule_class="canonical_mab",
                      ood_flag=True, ood_details="High z-score for pI"),
        BulkRowResult(3, "mAb_D", "error", error_message="Pipeline failed",
                      molecule_class="canonical_mab"),
    ]
    return br


class TestExportCSV:
    def test_csv_has_header_and_rows(self, sample_batch_result):
        csv_str = export_summary_csv(sample_batch_result)
        lines = csv_str.strip().split("\n")
        assert len(lines) == 5  # header + 4 rows

    def test_csv_parseable(self, sample_batch_result):
        csv_str = export_summary_csv(sample_batch_result)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 4
        assert rows[0]["name"] == "mAb_A"
        assert rows[0]["dev_score"] == "0.15"


class TestExportJSON:
    def test_json_valid(self, sample_batch_result):
        json_str = export_summary_json(sample_batch_result)
        data = json.loads(json_str)
        assert data["report_type"] == "ProtePilot_Bulk_Analysis"
        assert data["batch_info"]["n_total"] == 4
        assert data["batch_info"]["n_success"] == 3

    def test_json_has_statistics(self, sample_batch_result):
        data = json.loads(export_summary_json(sample_batch_result))
        assert "statistics" in data
        assert data["statistics"]["n_scored"] == 3


class TestDisplayStats:
    def test_overview(self, sample_batch_result):
        stats = generate_display_stats(sample_batch_result)
        assert stats["overview"]["n_total"] == 4
        assert stats["overview"]["n_success"] == 3

    def test_score_stats(self, sample_batch_result):
        stats = generate_display_stats(sample_batch_result)
        ss = stats["score_stats"]
        assert ss is not None
        assert ss["min"] == 0.15
        assert ss["max"] == 0.85

    def test_grade_distribution(self, sample_batch_result):
        stats = generate_display_stats(sample_batch_result)
        gd = stats["grade_distribution"]
        assert gd["Low"] == 1
        assert gd["Medium"] == 1
        assert gd["High"] == 1

    def test_top_candidates(self, sample_batch_result):
        stats = generate_display_stats(sample_batch_result)
        top = stats["top_candidates"]
        assert len(top) >= 1
        assert top[0]["name"] == "mAb_A"  # lowest score = best

    def test_flagged_ood(self, sample_batch_result):
        stats = generate_display_stats(sample_batch_result)
        assert len(stats["flagged"]) == 1
        assert stats["flagged"][0]["name"] == "mAb_C"


class TestRankCandidates:
    def test_rank_by_dev_score(self, sample_batch_result):
        ranked = rank_candidates(sample_batch_result, sort_by="dev_score")
        assert ranked[0]["name"] == "mAb_A"
        assert ranked[0]["rank"] == 1

    def test_rank_by_agg_risk(self, sample_batch_result):
        ranked = rank_candidates(sample_batch_result, sort_by="agg_risk")
        assert ranked[0]["name"] == "mAb_A"  # lowest agg_risk

    def test_rank_excludes_errors(self, sample_batch_result):
        ranked = rank_candidates(sample_batch_result)
        names = [r["name"] for r in ranked]
        assert "mAb_D" not in names  # error row excluded
