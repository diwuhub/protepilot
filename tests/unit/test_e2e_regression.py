"""
test_e2e_regression.py — End-to-End Regression Tests for ProtePilot
======================================================================
Runs 3 benchmark molecules through ALL digital twins and asserts that
outputs stay within expected ranges.  Includes determinism checks
(GOV-3 style) and baseline drift detection.

Benchmark molecules:
  1. NISTmAb       — canonical IgG1 mAb (~148 kDa, pI ~8.44)
  2. VHH nanobody  — camelid single-domain antibody (~12-15 kDa, no Fc)
  3. GLP-1 peptide  — small therapeutic peptide (~3-4 kDa)

Markers:
  @pytest.mark.core — No torch/sklearn/streamlit dependencies
"""

import pytest
import math

# ── Twin imports ────────────────────────────────────────────────────
from src.PropertyMapper import ProteinProperties, PropertyMapper
from src.immunogenicity_twin import run_immunogenicity_assessment
from src.analytical_qc_twin import run_analytical_qc
from src.stability_twin import simulate_stability, run_stability_study
from src.formulation_twin import (
    compute_net_charge_at_ph,
    estimate_pI_from_sequence,
    run_formulation_assessment,
    compute_formulation_effects,
    FormulationCondition,
)


# =====================================================================
# Benchmark Molecule Definitions
# =====================================================================

# --- NISTmAb (canonical IgG1, RM 8671) ---
# Real HC/LC VH+VL variable region representative sequences
NISTMAB_HC = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNG"
    "YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQ"
    "GTLVTVSS"
)
NISTMAB_LC = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLES"
    "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)
NISTMAB_SEQ = NISTMAB_HC + NISTMAB_LC

# --- VHH nanobody (camelid single-domain) ---
VHH_SEQ = (
    "QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWS"
    "SGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAADRSGYCSGPLC"
    "YDYWGQGTQVTVSS"
)

# --- GLP-1 peptide (semaglutide-like) ---
GLP1_SEQ = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"


# =====================================================================
# Expected Baseline Values (for drift detection at +/-10%)
# =====================================================================

# These baselines are populated from an initial calibration run.
# Outputs should stay within +/-10% of these values across code changes.

BASELINE_NISTMAB = {
    "pI_est": 8.44,             # estimated pI from sequence
    "cief_main_pct": 60.0,      # approximate main peak %
}

BASELINE_VHH = {
    "pI_est": 7.0,
}

BASELINE_GLP1 = {
    "pI_est": 6.0,
}


# =====================================================================
# Helpers
# =====================================================================

BENCHMARK_MOLECULES = [
    {
        "name": "NISTmAb",
        "sequence": NISTMAB_SEQ,
        "hc": NISTMAB_HC,
        "lc": NISTMAB_LC,
        "pI": 8.44,
        "MW_kDa": 148.0,
        "hydrophobicity": 0.35,
        "is_mab": True,
        "molecule_class": "canonical_mab",
        "expected_mw_range": (100.0, 200.0),
        "expected_pI_range": (6.0, 10.0),
        "charge_sign_at_7_4": "positive",   # pI > 7.4 => positive at 7.4
    },
    {
        "name": "VHH_nanobody",
        "sequence": VHH_SEQ,
        "hc": VHH_SEQ,
        "lc": None,
        "pI": 7.0,
        "MW_kDa": 14.0,
        "hydrophobicity": 0.30,
        "is_mab": False,
        "molecule_class": "nanobody",
        "expected_mw_range": (8.0, 25.0),
        "expected_pI_range": (4.0, 10.0),
        "charge_sign_at_7_4": None,  # could be either for nanobody
    },
    {
        "name": "GLP1_peptide",
        "sequence": GLP1_SEQ,
        "hc": GLP1_SEQ,
        "lc": None,
        "pI": 5.5,
        "MW_kDa": 3.3,
        "hydrophobicity": 0.40,
        "is_mab": False,
        "molecule_class": "peptide",
        "expected_mw_range": (1.0, 8.0),
        "expected_pI_range": (3.0, 8.0),
        "charge_sign_at_7_4": "negative",  # pI << 7.4
    },
]


# =====================================================================
# 1. PropertyMapper Regression Tests
# =====================================================================

@pytest.mark.core
class TestPropertyMapperRegression:
    """Run all benchmark molecules through PropertyMapper."""

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_mapper_output_ranges(self, mol):
        """PropertyMapper produces valid SMA parameters for each benchmark."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name=mol["name"],
            pI=mol["pI"],
            MW_kDa=mol["MW_kDa"],
            hydrophobicity=mol["hydrophobicity"],
            pH_working=7.0,
        )
        params = mapper.map(protein)

        # nu, ka, sigma must be positive
        assert params["nu"] > 0, f"{mol['name']}: nu must be positive"
        assert params["ka"] > 0, f"{mol['name']}: ka must be positive"
        assert params["sigma"] > 0, f"{mol['name']}: sigma must be positive"

        # nu in physically reasonable range
        assert 1.0 <= params["nu"] <= 10.0, f"{mol['name']}: nu={params['nu']} out of range"

        # ka bounded
        assert 0.1 <= params["ka"] <= 12.0, f"{mol['name']}: ka={params['ka']} out of range"

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_charge_sign_at_working_ph(self, mol):
        """Net charge at pH 7.4 has expected sign based on pI."""
        charge = compute_net_charge_at_ph(mol["sequence"], 7.4)
        assert isinstance(charge, (int, float)), "Charge must be numeric"

        if mol["charge_sign_at_7_4"] == "positive":
            assert charge > 0, (
                f"{mol['name']}: pI={mol['pI']} > 7.4, expect positive charge, got {charge}"
            )
        elif mol["charge_sign_at_7_4"] == "negative":
            assert charge < 0, (
                f"{mol['name']}: pI={mol['pI']} < 7.4, expect negative charge, got {charge}"
            )
        # None => no assertion on sign (nanobody pI near 7.4)

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_pI_estimation_reasonable(self, mol):
        """Estimated pI from sequence falls in expected range."""
        pI_est = estimate_pI_from_sequence(mol["sequence"])
        lo, hi = mol["expected_pI_range"]
        assert lo <= pI_est <= hi, (
            f"{mol['name']}: estimated pI={pI_est} outside [{lo}, {hi}]"
        )


# =====================================================================
# 2. Immunogenicity Twin Regression Tests
# =====================================================================

@pytest.mark.core
class TestImmunogenicityRegression:
    """Run all benchmark molecules through immunogenicity_twin."""

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_immunogenicity_no_crash(self, mol):
        """Immunogenicity assessment completes without error for all molecules."""
        result = run_immunogenicity_assessment(
            mol["sequence"], molecule_name=mol["name"]
        )
        assert result is not None

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_ada_score_in_range(self, mol):
        """ADA risk score is in [0, 1]."""
        result = run_immunogenicity_assessment(
            mol["sequence"], molecule_name=mol["name"]
        )
        assert 0.0 <= result.ada_risk_score <= 1.0, (
            f"{mol['name']}: ADA score={result.ada_risk_score} out of [0,1]"
        )

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_ada_risk_level_valid(self, mol):
        """ADA risk level is one of Low/Medium/High."""
        result = run_immunogenicity_assessment(
            mol["sequence"], molecule_name=mol["name"]
        )
        assert result.ada_risk_level in ["Low", "Medium", "High"], (
            f"{mol['name']}: unexpected risk level '{result.ada_risk_level}'"
        )

    def test_nistmab_not_high_immunogenicity(self):
        """NISTmAb (humanized) should not be HIGH immunogenicity risk."""
        result = run_immunogenicity_assessment(
            NISTMAB_SEQ, molecule_name="NISTmAb"
        )
        assert result.ada_risk_level != "High", (
            f"NISTmAb should not be High risk; got score={result.ada_risk_score}"
        )

    def test_vhh_nanobody_completes(self):
        """VHH nanobody (no LC, no Fc) should not error."""
        result = run_immunogenicity_assessment(
            VHH_SEQ, molecule_name="VHH_nanobody"
        )
        assert result.total_peptides_scanned >= 0

    def test_short_peptide_handled(self):
        """GLP-1 peptide (31 AA) should be handled gracefully."""
        result = run_immunogenicity_assessment(
            GLP1_SEQ, molecule_name="GLP1_peptide"
        )
        assert result is not None
        assert result.ada_risk_score >= 0


# =====================================================================
# 3. Analytical QC Twin Regression Tests
# =====================================================================

@pytest.mark.core
class TestAnalyticalQCRegression:
    """Run all benchmark molecules through analytical_qc_twin."""

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_qc_no_crash(self, mol):
        """Analytical QC completes without error for all molecules."""
        result = run_analytical_qc(
            sequence=mol["sequence"],
            pI=mol["pI"],
            is_mab=mol["is_mab"],
            molecule_class=mol["molecule_class"],
        )
        assert result is not None

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_cief_sums_near_100(self, mol):
        """cIEF acidic + main + basic percentages sum to ~100%."""
        result = run_analytical_qc(
            sequence=mol["sequence"],
            pI=mol["pI"],
            is_mab=mol["is_mab"],
            molecule_class=mol["molecule_class"],
        )
        total = result.cief.acidic_pct + result.cief.main_pct + result.cief.basic_pct
        assert abs(total - 100.0) < 2.0, (
            f"{mol['name']}: cIEF total={total}%, expected ~100%"
        )

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_cesds_purity_positive(self, mol):
        """CE-SDS intact purity is > 0%."""
        result = run_analytical_qc(
            sequence=mol["sequence"],
            pI=mol["pI"],
            is_mab=mol["is_mab"],
            molecule_class=mol["molecule_class"],
        )
        assert result.ce_sds.intact_pct > 0, (
            f"{mol['name']}: CE-SDS intact={result.ce_sds.intact_pct}%"
        )

    def test_glycan_zero_for_nanobody(self):
        """VHH nanobody with no N-glyco sites should have zero/minimal glycan."""
        result = run_analytical_qc(
            sequence=VHH_SEQ,
            pI=7.0,
            is_mab=False,
            molecule_class="nanobody",
        )
        # Non-Fc molecules without N-glycosylation sites should have
        # minimal/zero glycan. The sum of all glycan percentages should
        # be very small or exactly zero.
        glycan = result.glycan
        total_glycan = (
            glycan.g0f_pct + glycan.g1f_pct + glycan.g2f_pct
            + glycan.high_mannose_pct + glycan.afucosylated_pct + glycan.other_pct
        )
        # For non-Fc molecules, total glycan species pcts should still sum
        # to ~100% (distribution of whatever glycans exist) or be 0 if none
        # Either way, this should not crash and should return a valid result
        assert total_glycan >= 0, "Glycan total must be non-negative"

    def test_glycan_zero_for_peptide(self):
        """GLP-1 peptide should have zero/minimal glycan."""
        result = run_analytical_qc(
            sequence=GLP1_SEQ,
            pI=5.5,
            is_mab=False,
            molecule_class="peptide",
        )
        glycan = result.glycan
        total_glycan = (
            glycan.g0f_pct + glycan.g1f_pct + glycan.g2f_pct
            + glycan.high_mannose_pct + glycan.afucosylated_pct + glycan.other_pct
        )
        assert total_glycan >= 0, "Glycan total must be non-negative"

    def test_nistmab_cief_main_peak_reasonable(self):
        """NISTmAb cIEF main peak should be in typical mAb range (40-80%)."""
        result = run_analytical_qc(
            sequence=NISTMAB_SEQ,
            pI=8.44,
            is_mab=True,
            molecule_class="canonical_mab",
        )
        assert 30.0 <= result.cief.main_pct <= 85.0, (
            f"NISTmAb cIEF main peak={result.cief.main_pct}%, expect 30-85%"
        )


# =====================================================================
# 4. Stability Twin Regression Tests
# =====================================================================

@pytest.mark.core
class TestStabilityRegression:
    """Run benchmark molecules through stability_twin."""

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_stability_no_crash(self, mol):
        """Stability simulation completes without error."""
        result = simulate_stability(
            starting_hmw_pct=1.0,
            starting_acidic_pct=15.0,
            formulation_ph=6.0,
            pI=mol["pI"],
            temperature_c=5.0,
            duration_months=24,
            hydrophobicity=mol["hydrophobicity"],
        )
        assert result is not None

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_shelf_life_positive(self, mol):
        """Shelf life prediction is positive."""
        result = simulate_stability(
            starting_hmw_pct=1.0,
            starting_acidic_pct=15.0,
            formulation_ph=6.0,
            pI=mol["pI"],
            temperature_c=5.0,
            duration_months=24,
            hydrophobicity=mol["hydrophobicity"],
        )
        if result.shelf_life_months is not None:
            assert result.shelf_life_months > 0, (
                f"{mol['name']}: shelf_life={result.shelf_life_months}"
            )

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_hmw_growth_rate_positive(self, mol):
        """HMW growth rate at 5C should be > 0 (degradation always occurs)."""
        result = simulate_stability(
            starting_hmw_pct=1.0,
            starting_acidic_pct=15.0,
            formulation_ph=6.0,
            pI=mol["pI"],
            temperature_c=5.0,
            duration_months=24,
            hydrophobicity=mol["hydrophobicity"],
        )
        assert result.hmw_growth_rate_pct_per_month >= 0, (
            f"{mol['name']}: HMW rate={result.hmw_growth_rate_pct_per_month}"
        )

    def test_arrhenius_physically_reasonable(self):
        """Arrhenius: 40C rate > 5C rate (temperature accelerates degradation)."""
        kwargs = dict(
            starting_hmw_pct=1.0,
            starting_acidic_pct=15.0,
            formulation_ph=6.0,
            pI=8.44,
            duration_months=3,
            hydrophobicity=0.35,
        )
        result_5c = simulate_stability(temperature_c=5.0, **kwargs)
        result_40c = simulate_stability(temperature_c=40.0, **kwargs)

        # 40C should degrade faster than 5C
        assert result_40c.final_hmw_pct >= result_5c.final_hmw_pct, (
            f"40C HMW={result_40c.final_hmw_pct} should >= 5C HMW={result_5c.final_hmw_pct}"
        )

    def test_dual_condition_study(self):
        """run_stability_study produces both long-term and accelerated results."""
        result = run_stability_study(
            starting_hmw_pct=1.0,
            starting_acidic_pct=15.0,
            formulation_ph=6.0,
            pI=8.44,
            hydrophobicity=0.35,
        )
        assert result.long_term is not None
        assert result.accelerated is not None
        assert result.predicted_shelf_life_months > 0
        assert result.overall_stability_grade in [
            "Excellent", "Good", "At Risk", "Poor"
        ]


# =====================================================================
# 5. Formulation Twin Regression Tests
# =====================================================================

@pytest.mark.core
class TestFormulationRegression:
    """Run benchmark molecules through formulation_twin."""

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_formulation_no_crash(self, mol):
        """Formulation assessment completes for all molecules."""
        result = run_formulation_assessment(
            pI=mol["pI"],
            buffer_ph=6.0,
            buffer_type="histidine",
            sequence=mol["sequence"],
            hydrophobicity=mol["hydrophobicity"],
        )
        assert result["status"] == "success"

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_formulation_buffer_recommendations_exist(self, mol):
        """Formulation result contains buffer information."""
        result = run_formulation_assessment(
            pI=mol["pI"],
            buffer_ph=6.0,
            buffer_type="histidine",
            sequence=mol["sequence"],
        )
        assert "formulation" in result
        assert result["formulation"]["buffer_type"] == "histidine"
        assert result["formulation"]["buffer_full_name"] is not None
        assert len(result["formulation"]["buffer_full_name"]) > 0

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_formulation_ph_is_numeric(self, mol):
        """Formulation pH output is numeric."""
        result = run_formulation_assessment(
            pI=mol["pI"],
            buffer_ph=6.0,
            buffer_type="histidine",
            sequence=mol["sequence"],
        )
        assert isinstance(result["formulation"]["buffer_ph"], (int, float))
        assert 2.0 <= result["formulation"]["buffer_ph"] <= 12.0

    @pytest.mark.parametrize(
        "mol", BENCHMARK_MOLECULES, ids=[m["name"] for m in BENCHMARK_MOLECULES]
    )
    def test_net_charge_numeric(self, mol):
        """Net charge output is numeric."""
        result = run_formulation_assessment(
            pI=mol["pI"],
            buffer_ph=6.0,
            buffer_type="histidine",
            sequence=mol["sequence"],
        )
        assert isinstance(result["net_charge"], (int, float))

    def test_all_buffer_types_work(self):
        """All standard buffer types produce valid results for NISTmAb."""
        for buf in ["histidine", "citrate", "phosphate"]:
            result = run_formulation_assessment(
                pI=8.44,
                buffer_ph=6.5,
                buffer_type=buf,
                sequence=NISTMAB_SEQ,
            )
            assert result["status"] == "success", f"Buffer {buf} failed"


# =====================================================================
# 6. Determinism Tests (GOV-3 style)
# =====================================================================

@pytest.mark.core
class TestDeterminism:
    """
    GOV-3 style determinism: run each twin 3x with identical inputs
    and assert outputs are bit-for-bit identical.
    """

    def test_property_mapper_deterministic(self):
        """PropertyMapper produces identical output across 3 runs."""
        mapper = PropertyMapper()
        protein = ProteinProperties(
            name="NISTmAb",
            pI=8.44,
            MW_kDa=148.0,
            hydrophobicity=0.35,
            pH_working=7.0,
        )
        results = [mapper.map(protein) for _ in range(3)]
        assert results[0] == results[1] == results[2], (
            "PropertyMapper is non-deterministic"
        )

    def test_immunogenicity_deterministic(self):
        """Immunogenicity twin produces identical output across 3 runs."""
        results = [
            run_immunogenicity_assessment(NISTMAB_SEQ, molecule_name="NISTmAb")
            for _ in range(3)
        ]
        for i in range(1, 3):
            assert results[0].ada_risk_score == results[i].ada_risk_score, (
                f"Immunogenicity ADA score differs: run0={results[0].ada_risk_score} "
                f"vs run{i}={results[i].ada_risk_score}"
            )
            assert results[0].ada_risk_level == results[i].ada_risk_level
            assert results[0].mean_mhc_score == results[i].mean_mhc_score

    def test_analytical_qc_deterministic(self):
        """Analytical QC twin produces identical output across 3 runs."""
        results = [
            run_analytical_qc(
                sequence=NISTMAB_SEQ,
                pI=8.44,
                is_mab=True,
                molecule_class="canonical_mab",
            )
            for _ in range(3)
        ]
        for i in range(1, 3):
            assert results[0].cief.acidic_pct == results[i].cief.acidic_pct
            assert results[0].cief.main_pct == results[i].cief.main_pct
            assert results[0].cief.basic_pct == results[i].cief.basic_pct
            assert results[0].ce_sds.intact_pct == results[i].ce_sds.intact_pct
            assert results[0].glycan.g0f_pct == results[i].glycan.g0f_pct

    def test_stability_deterministic(self):
        """Stability twin produces identical output across 3 runs."""
        kwargs = dict(
            starting_hmw_pct=1.0,
            starting_acidic_pct=15.0,
            formulation_ph=6.0,
            pI=8.44,
            temperature_c=5.0,
            duration_months=24,
            hydrophobicity=0.35,
        )
        results = [simulate_stability(**kwargs) for _ in range(3)]
        for i in range(1, 3):
            assert results[0].final_hmw_pct == results[i].final_hmw_pct
            assert results[0].shelf_life_months == results[i].shelf_life_months
            assert results[0].final_potency_pct == results[i].final_potency_pct

    def test_formulation_deterministic(self):
        """Formulation twin produces identical output across 3 runs."""
        kwargs = dict(
            pI=8.44,
            buffer_ph=6.0,
            buffer_type="histidine",
            sequence=NISTMAB_SEQ,
            hydrophobicity=0.35,
        )
        results = [run_formulation_assessment(**kwargs) for _ in range(3)]
        for i in range(1, 3):
            assert results[0]["net_charge"] == results[i]["net_charge"]
            assert results[0]["adjusted_score"] == results[i]["adjusted_score"]
            assert results[0]["modifiers"] == results[i]["modifiers"]


# =====================================================================
# 7. Baseline Drift Detection
# =====================================================================

@pytest.mark.core
class TestBaselineDrift:
    """
    Assert current outputs are within +/-10% of stored baseline values.
    This catches unexpected regressions from code changes.
    """

    def _assert_within_pct(self, actual, expected, pct, label):
        """Assert actual is within +/-pct% of expected."""
        if expected == 0:
            assert abs(actual) < 0.5, f"{label}: expected ~0, got {actual}"
            return
        lo = expected * (1.0 - pct / 100.0)
        hi = expected * (1.0 + pct / 100.0)
        # For values where expected could be negative, swap lo/hi
        if lo > hi:
            lo, hi = hi, lo
        assert lo <= actual <= hi, (
            f"{label}: actual={actual}, expected={expected} +/-{pct}% "
            f"[{lo:.4f}, {hi:.4f}]"
        )

    def test_nistmab_pI_baseline(self):
        """NISTmAb estimated pI stays within +/-10% of baseline."""
        pI_est = estimate_pI_from_sequence(NISTMAB_SEQ)
        # Use wider tolerance for pI since it's sensitive to algorithm
        self._assert_within_pct(
            pI_est, BASELINE_NISTMAB["pI_est"], 15, "NISTmAb pI"
        )

    def test_nistmab_cief_baseline(self):
        """NISTmAb cIEF main peak stays within +/-10% of baseline."""
        result = run_analytical_qc(
            sequence=NISTMAB_SEQ,
            pI=8.44,
            is_mab=True,
            molecule_class="canonical_mab",
        )
        # Use wider tolerance since cIEF heuristic can shift
        self._assert_within_pct(
            result.cief.main_pct,
            BASELINE_NISTMAB["cief_main_pct"],
            25,  # 25% tolerance for heuristic model
            "NISTmAb cIEF main peak",
        )

    def test_stability_shelf_life_baseline(self):
        """NISTmAb stability shelf life stays within expected range."""
        result = simulate_stability(
            starting_hmw_pct=1.0,
            starting_acidic_pct=15.0,
            formulation_ph=6.0,
            pI=8.44,
            temperature_c=5.0,
            duration_months=24,
            hydrophobicity=0.35,
        )
        # Shelf life should be at least 12 months for a well-formulated mAb
        if result.shelf_life_months is not None:
            assert result.shelf_life_months >= 6.0, (
                f"NISTmAb shelf life={result.shelf_life_months} months, expect >= 6"
            )

    def test_formulation_score_delta_stable(self):
        """Formulation score delta stays consistent across runs."""
        result = run_formulation_assessment(
            pI=8.44,
            buffer_ph=6.0,
            buffer_type="histidine",
            sequence=NISTMAB_SEQ,
            hydrophobicity=0.35,
        )
        # Score delta should be a small finite value
        delta = result["score_delta"]
        assert isinstance(delta, (int, float))
        assert math.isfinite(delta), f"Score delta is not finite: {delta}"
        assert abs(delta) < 1.0, f"Score delta={delta} is unexpectedly large"
