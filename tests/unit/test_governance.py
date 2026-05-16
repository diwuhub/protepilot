"""
test_governance.py  ·  Layer 2 — Governance Tests
=================================================
Validates cross-cutting quality properties that span the entire platform:
  - Grade canonicality: display grades always use "X Risk" format
  - Determinism: same input → same output across repeated runs
  - Molecule-aware recommendations: lead sentences are format-specific
  - Evidence completeness: risk dimensions have populated evidence
  - Cross-section consistency: no section contradicts another
  - Schema alignment: FIELD_MAP covers all BulkRowResult fields

These tests catch regressions that do NOT break scoring but silently degrade
output quality, report integrity, or regulatory compliance.
"""

import pytest

pytestmark = [pytest.mark.governance, pytest.mark.core]


# ═══════════════════════════════════════════════════════════════════════
#  Grade Canonicality
# ═══════════════════════════════════════════════════════════════════════

class TestGradeCanonicalityEnforcement:
    """Verify all output-facing grades use 'Low Risk' / 'Medium Risk' / 'High Risk'."""

    def test_grade_to_risk_label_converts_bare_grades(self):
        from src.report_schema import grade_to_risk_label
        assert grade_to_risk_label("Low") == "Low Risk"
        assert grade_to_risk_label("Medium") == "Medium Risk"
        assert grade_to_risk_label("High") == "High Risk"

    def test_grade_to_risk_label_is_idempotent(self):
        """Applying the label converter twice produces the same result."""
        from src.report_schema import grade_to_risk_label
        for raw in ("Low", "Medium", "High"):
            canonical = grade_to_risk_label(raw)
            assert grade_to_risk_label(canonical) == canonical, (
                f"grade_to_risk_label is not idempotent: "
                f"'{raw}' → '{canonical}' → '{grade_to_risk_label(canonical)}'"
            )

    def test_grade_from_score_returns_bare_internal_grades(self):
        """Internal scoring uses bare grades; display uses canonical."""
        from src.report_schema import grade_from_score
        assert grade_from_score(0.10) == "Low"
        assert grade_from_score(0.35) == "Medium"
        assert grade_from_score(0.70) == "High"

    def test_composite_grade_is_canonical_after_assess(self):
        """assess_developability() produces canonical 'X Risk' composite_grade."""
        from src.developability_core import assess_developability
        result = assess_developability(
            molecule_name="GradeTest",
            molecule_class="canonical_mab",
            dev_predictions={"agg_risk": 0.1, "stability": 0.8, "viscosity_risk": 0.05},
        )
        assert result.composite_grade in ("Low Risk", "Medium Risk", "High Risk"), (
            f"composite_grade should be canonical, got '{result.composite_grade}'"
        )

    def test_bulk_grade_validator_catches_bare_strings(self):
        """The bulk grade validator flags bare 'Low'/'Medium'/'High'."""
        from src.bulk_single_schema_alignment import validate_grade_strings

        class _BareResult:
            composite_dev_grade = "Low"
            developability_grade = "Medium"

        violations = validate_grade_strings(_BareResult())
        assert len(violations) >= 2, "Should flag both bare grade fields"

    def test_bulk_grade_validator_accepts_canonical(self):
        from src.bulk_single_schema_alignment import validate_grade_strings

        class _Good:
            composite_dev_grade = "Low Risk"
            developability_grade = "Low Risk"

        assert len(validate_grade_strings(_Good())) == 0


# ═══════════════════════════════════════════════════════════════════════
#  Determinism — same input → same output
# ═══════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Verify that the pipeline produces identical results across multiple runs."""

    def test_developability_score_is_deterministic(self):
        """Same sequence → identical 5-dim composite score, 10 times."""
        from src.developability_core import assess_developability
        seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNG" * 3

        results = []
        for _ in range(10):
            r = assess_developability(
                molecule_name="DeterminismTest",
                molecule_class="canonical_mab",
                dev_predictions={"agg_risk": 0.15, "stability": 0.75, "viscosity_risk": 0.1},
            )
            results.append(r.composite_score)

        assert len(set(results)) == 1, (
            f"Composite score should be identical across 10 runs, got {set(results)}"
        )

    def test_recommendation_text_is_deterministic(self):
        """Same input → identical recommendation text, 5 times."""
        from src.report_assembler import assemble_report

        intent = {
            "name": "DetTest", "format": "canonical_mab", "is_mab": True,
            "hc_sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVAR" * 2,
            "lc_sequence": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIY" * 2,
            "glycoform": "standard_cho",
        }
        cache = {
            "developability": {
                "status": "success",
                "data": {
                    "score": {"score": 0.35, "grade": "Low Risk"},
                    "predictions": {"agg_risk": 0.15, "stability": 0.7, "viscosity_risk": 0.1},
                },
            }
        }

        recs = []
        for _ in range(5):
            report = assemble_report(intent, cache)
            recs.append(report.executive_summary.recommendation)

        assert len(set(recs)) == 1, "Recommendation text should be deterministic"

    def test_property_mapper_is_deterministic(self):
        """Same sequence → identical biophysical features."""
        from src.feature_registry import compute_features
        seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH" * 4

        features_list = [compute_features(seq) for _ in range(5)]
        # FeatureSet may be a dataclass — convert to dict for comparison
        dicts = []
        for f in features_list:
            d = f.__dict__ if hasattr(f, "__dict__") else (f._asdict() if hasattr(f, "_asdict") else dict(f))
            dicts.append(d)

        for key in dicts[0]:
            values = [d[key] for d in dicts]
            assert len(set(str(v) for v in values)) == 1, (
                f"Feature '{key}' not deterministic: {values}"
            )


# ═══════════════════════════════════════════════════════════════════════
#  Molecule-Aware Recommendations
# ═══════════════════════════════════════════════════════════════════════

class TestMoleculeAwareRecommendations:
    """Verify recommendation text mentions the molecule format, not generic mAb text."""

    def _get_recommendation(self, mol_cls, score, is_mab=False):
        """Helper: call _generate_recommendation directly with a mock ReportContext."""
        from src.report_assembler import _generate_recommendation
        from src.report_schema import ReportContext

        ctx = ReportContext(
            molecule_name="MolAwareTest",
            molecule_class=mol_cls,
            overall_score=score,
            overall_grade="Low Risk" if score < 0.25 else ("Medium Risk" if score < 0.55 else "High Risk"),
            prediction_mode="rule_based",
            is_ood=False,
        )
        _rec, detail = _generate_recommendation(ctx)
        return detail

    def test_canonical_mab_low_risk_says_standard_cmc(self):
        detail = self._get_recommendation("canonical_mab", 0.15, is_mab=True)
        assert "Standard CMC" in detail or "standard" in detail.lower()

    def test_canonical_mab_does_not_say_nanobody(self):
        detail = self._get_recommendation("canonical_mab", 0.15, is_mab=True)
        assert "nanobody" not in detail.lower()
        assert "VHH" not in detail

    def test_single_domain_mentions_nanobody_or_vhh(self):
        detail = self._get_recommendation("single_domain", 0.15)
        has_nanobody = "nanobody" in detail.lower() or "VHH" in detail
        has_single_domain = "single-domain" in detail.lower() or "single domain" in detail.lower()
        assert has_nanobody or has_single_domain, (
            f"single_domain recommendation should mention nanobody/VHH, got: {detail[:200]}"
        )

    def test_peptide_mentions_protease_or_peptide(self):
        detail = self._get_recommendation("peptide", 0.15)
        has_peptide = "peptide" in detail.lower()
        has_protease = "protease" in detail.lower()
        assert has_peptide or has_protease, (
            f"peptide recommendation should mention peptide/protease, got: {detail[:200]}"
        )

    def test_bispecific_mentions_species_or_dual(self):
        detail = self._get_recommendation("bispecific", 0.15)
        has_species = "species" in detail.lower()
        has_dual = "dual" in detail.lower() or "bispecific" in detail.lower()
        assert has_species or has_dual, (
            f"bispecific recommendation should mention species/dual-target, got: {detail[:200]}"
        )

    def test_high_risk_non_canonical_has_format_language(self):
        """High risk + non-canonical should NOT use generic mAb wording."""
        detail = self._get_recommendation("single_domain", 0.70)
        assert "Standard CMC" not in detail, (
            "High-risk single_domain should not mention 'Standard CMC'"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Evidence Completeness
# ═══════════════════════════════════════════════════════════════════════

class TestEvidenceCompleteness:
    """Verify that risk assessments include supporting evidence, not empty fields."""

    def test_composite_assessment_has_dimensions(self):
        """assess_developability produces at least 3 risk dimensions."""
        from src.developability_core import assess_developability
        seq = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVAR" * 4
        result = assess_developability(
            molecule_name="EvidTest",
            molecule_class="canonical_mab",
            dev_predictions={"agg_risk": 0.3, "stability": 0.5, "viscosity_risk": 0.2},
            upstream_results={"titer_g_l": 3.0},
            ada_results={"risk_level": "Medium", "risk_score": 0.5},
        )
        assert len(result.dimensions) >= 3, (
            f"Expected ≥3 risk dimensions, got {len(result.dimensions)}"
        )
        for dim in result.dimensions:
            assert dim.name, "Dimension name should not be empty"
            assert dim.grade in ("Low", "Medium", "High", "Unknown"), (
                f"Dimension grade invalid: {dim.grade}"
            )

    def test_report_sections_not_empty(self):
        """A report from valid input should have non-empty executive summary fields."""
        from src.report_assembler import assemble_report

        intent = {
            "name": "EvidReport", "format": "canonical_mab", "is_mab": True,
            "hc_sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVAR" * 3,
            "lc_sequence": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIY" * 3,
            "glycoform": "standard_cho",
        }
        cache = {
            "developability": {
                "status": "success",
                "data": {
                    "score": {"score": 0.30, "grade": "Medium Risk"},
                    "predictions": {"agg_risk": 0.2, "stability": 0.6, "viscosity_risk": 0.15},
                },
            }
        }
        report = assemble_report(intent, cache)

        assert report.executive_summary is not None
        assert report.executive_summary.recommendation is not None
        assert len(report.executive_summary.recommendation) > 5
        assert report.executive_summary.recommendation_detail is not None
        assert len(report.executive_summary.recommendation_detail) > 20
        assert report.molecule_overview is not None


# ═══════════════════════════════════════════════════════════════════════
#  Schema Alignment (FIELD_MAP completeness)
# ═══════════════════════════════════════════════════════════════════════

class TestSchemaAlignment:
    """Verify the FIELD_MAP covers all BulkRowResult output fields."""

    def test_field_map_covers_all_bulkrowresult_output_fields(self):
        """Every field in to_summary_dict() should appear in FIELD_MAP."""
        from src.bulk_runner import BulkRowResult
        from src.bulk_single_schema_alignment import FIELD_MAP

        # Get all bulk field names from FIELD_MAP
        mapped_bulk_fields = set()
        for _cat, _single, bulk_field, _notes in FIELD_MAP:
            if bulk_field != "(none)":
                mapped_bulk_fields.add(bulk_field)

        # Get all BulkRowResult dataclass field names (excluding raw/internal storage)
        import dataclasses
        all_fields = {f.name for f in dataclasses.fields(BulkRowResult)}
        # These are raw pipeline storage, not mapped to schema
        internal_only = {"raw_result", "raw_comprehensive", "intent"}

        output_fields = all_fields - internal_only
        unmapped = output_fields - mapped_bulk_fields

        assert len(unmapped) == 0, (
            f"BulkRowResult fields not in FIELD_MAP: {unmapped}"
        )

    def test_field_map_no_duplicate_bulk_fields(self):
        """No bulk field name appears twice in FIELD_MAP."""
        from src.bulk_single_schema_alignment import FIELD_MAP

        seen = set()
        dupes = []
        for _cat, _single, bulk_field, _notes in FIELD_MAP:
            if bulk_field != "(none)":
                if bulk_field in seen:
                    dupes.append(bulk_field)
                seen.add(bulk_field)

        assert len(dupes) == 0, f"Duplicate bulk fields in FIELD_MAP: {dupes}"

    def test_field_map_has_minimum_entries(self):
        from src.bulk_single_schema_alignment import FIELD_MAP
        assert len(FIELD_MAP) >= 50, f"FIELD_MAP too small: {len(FIELD_MAP)} entries"

    def test_csv_columns_match_summary_dict_keys(self):
        """_CSV_COLUMNS in bulk_summary must cover all to_summary_dict() keys."""
        from src.bulk_summary import _CSV_COLUMNS
        from src.bulk_runner import BulkRowResult

        r = BulkRowResult(0, "Test", "success")
        d = r.to_summary_dict()
        csv_set = set(_CSV_COLUMNS)
        dict_set = set(d.keys())

        missing_in_csv = dict_set - csv_set
        assert len(missing_in_csv) == 0, (
            f"to_summary_dict() keys not in _CSV_COLUMNS: {missing_in_csv}"
        )

    def test_extract_bulk_core_returns_all_canonical_keys(self):
        """extract_bulk_core() produces all expected canonical keys."""
        from src.bulk_single_schema_alignment import extract_bulk_core

        class _FakeRow:
            composite_dev_score = 0.25
            developability_score = 0.20
            composite_dev_grade = "Low Risk"
            developability_grade = "Low Risk"
            agg_risk = 0.1
            stability = 0.8
            viscosity_risk = 0.05
            mw_kda = 145.0
            pI = 8.5
            gravy = -0.3
            hydrophobicity = 0.35
            seq_length = 1200
            deam_sites = 7
            ox_sites = 5
            acidic_residues = 140
            basic_residues = 160
            cysteine_count = 16
            ood_flag = False
            ood_details = None
            cief_main_pct = 80.0
            cief_acidic_pct = 10.0
            cief_basic_pct = 10.0
            ce_sds_purity_pct = 97.0
            intact_mass_da = 145000.0
            half_life_days = 21.0
            predicted_titer_g_L = 3.5
            ada_risk_category = "Low"
            molecule_class = "canonical_mab"

        core = extract_bulk_core(_FakeRow())

        expected_keys = {
            "overall_score", "base_risk_score", "overall_grade",
            "agg_risk", "stability", "viscosity_risk",
            "molecular_weight_kda", "isoelectric_point", "gravy_score",
            "hydrophobicity", "sequence_length", "deam_sites", "ox_sites",
            "acidic_residues", "basic_residues", "cysteine_count",
            "is_ood", "ood_reason",
            "cief_main_pct", "cief_acidic_pct", "cief_basic_pct",
            "cesds_intact_pct", "ms_intact_mass_da",
            "half_life_days", "final_titer_g_l", "ada_risk_level",
            "molecule_class",
        }
        assert expected_keys.issubset(core.keys()), (
            f"Missing keys in extract_bulk_core: {expected_keys - core.keys()}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Molecule Registry Completeness
# ═══════════════════════════════════════════════════════════════════════

class TestMoleculeRegistryCompleteness:
    """Verify the central molecule_registry covers all known molecule classes."""

    def test_registry_covers_all_classifier_classes(self):
        """Every MoleculeClass enum value has a registry entry."""
        from src.molecule_classifier import MoleculeClass
        from src.molecule_registry import MOLECULE_REGISTRY

        enum_values = {mc.value for mc in MoleculeClass}
        registry_keys = set(MOLECULE_REGISTRY.keys())
        missing = enum_values - registry_keys
        assert len(missing) == 0, (
            f"MoleculeClass values not in MOLECULE_REGISTRY: {missing}"
        )

    def test_registry_risk_weights_sum_to_one(self):
        """Risk weights for every class sum to ~1.0."""
        from src.molecule_registry import MOLECULE_REGISTRY

        for cls_name, cfg in MOLECULE_REGISTRY.items():
            total = sum(cfg["risk_weights"].values())
            assert 0.95 <= total <= 1.05, (
                f"{cls_name}: risk weights sum to {total:.3f}, expected ~1.0"
            )

    def test_registry_has_standard_five_dimensions(self):
        """Every class has at least the 5 standard dimensions."""
        from src.molecule_registry import MOLECULE_REGISTRY

        standard = {"aggregation", "stability", "viscosity", "expression", "immunogenicity"}
        for cls_name, cfg in MOLECULE_REGISTRY.items():
            dims = set(cfg["risk_weights"].keys())
            missing = standard - dims
            assert not missing, f"{cls_name}: missing standard dimensions {missing}"

    def test_registry_recommendation_suffix_matches_report_schema(self):
        """Registry suffixes should match MOLECULE_RECOMMENDATION_SUFFIX in report_schema."""
        from src.molecule_registry import MOLECULE_REGISTRY
        from src.report_schema import MOLECULE_RECOMMENDATION_SUFFIX

        for cls_name in MOLECULE_REGISTRY:
            if cls_name in MOLECULE_RECOMMENDATION_SUFFIX:
                registry_suffix = MOLECULE_REGISTRY[cls_name]["recommendation_suffix"]
                schema_suffix = MOLECULE_RECOMMENDATION_SUFFIX[cls_name]
                assert registry_suffix == schema_suffix, (
                    f"{cls_name}: registry suffix differs from report_schema.\n"
                    f"  Registry: {registry_suffix[:80]}...\n"
                    f"  Schema:   {schema_suffix[:80]}..."
                )

    def test_registry_selftest_passes(self):
        """The registry's own selftest should pass."""
        from src.molecule_registry import _selftest
        assert _selftest() is True

    def test_confidence_defaults_cover_all_weight_keys(self):
        """confidence_defaults should have entries for every risk weight dimension."""
        from src.molecule_registry import MOLECULE_REGISTRY

        for cls_name, cfg in MOLECULE_REGISTRY.items():
            for dim in cfg["risk_weights"]:
                assert dim in cfg["confidence_defaults"], (
                    f"{cls_name}: no confidence_default for '{dim}'"
                )

    def test_extra_dimensions_appear_in_risk_weights(self):
        """Extra dimensions (e.g., species_purity) must also appear in risk_weights."""
        from src.molecule_registry import MOLECULE_REGISTRY

        for cls_name, cfg in MOLECULE_REGISTRY.items():
            for dim_cfg in cfg["extra_dimensions"]:
                assert dim_cfg["name"] in cfg["risk_weights"], (
                    f"{cls_name}: extra dimension '{dim_cfg['name']}' not in risk_weights"
                )
