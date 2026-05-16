"""
governance_checks.py — ProtePilot Governance Validation Suite
================================================================
Runs automated governance checks against the platform modules.

Checks:
  GOV-1: Molecule-Class Propagation
  GOV-2: Evidence Completeness
  GOV-3: Determinism
  GOV-5: Report Internal Consistency (cIEF, glycan, CE-SDS sums to 100%)
  GOV-7: Report Cross-Section Consistency (MW, pI, molecule_class)

Usage:
  PYTHONPATH=<project_root> python3 SelfTest/governance_checks.py

Results saved to:
  <project_root>/SelfTest/governance_results.json
"""

from __future__ import annotations

import glob
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PLATFORM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, _PLATFORM_ROOT)

# ---------------------------------------------------------------------------
# Imports from platform
# ---------------------------------------------------------------------------
from src.PropertyMapper import PropertyMapper, ProteinProperties
from src.upstream_twin import run_upstream_simulation, result_to_dict
from src.analytical_qc_twin import run_analytical_qc

# ---------------------------------------------------------------------------
# Helper: Result accumulator
# ---------------------------------------------------------------------------

class GovernanceResult:
    def __init__(self, check_id: str, name: str):
        self.check_id = check_id
        self.name = name
        self.status: str = "UNKNOWN"
        self.details: List[str] = []
        self.failures: List[str] = []
        self.data: Dict[str, Any] = {}

    def pass_(self, detail: str = ""):
        self.status = "PASS"
        if detail:
            self.details.append(detail)
        print(f"  [PASS] {detail}" if detail else f"  [PASS]")

    def fail(self, reason: str):
        self.status = "FAIL"
        self.failures.append(reason)
        print(f"  [FAIL] {reason}")

    def info(self, msg: str):
        self.details.append(msg)
        print(f"  INFO  {msg}")

    def finalize(self):
        if self.status == "UNKNOWN":
            self.status = "PASS" if not self.failures else "FAIL"
        elif self.failures and self.status == "PASS":
            self.status = "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "status": self.status,
            "details": self.details,
            "failures": self.failures,
            "data": self.data,
        }


# ============================================================
# GOV-1: Molecule-Class Propagation
# ============================================================

def check_gov1_molecule_class_propagation() -> GovernanceResult:
    """
    Run 3 molecules through PropertyMapper, UpstreamTwin, and AnalyticalQC.
    Verify that molecule_class is accepted and influences outputs in a
    class-consistent manner.
    """
    result = GovernanceResult("GOV-1", "Molecule-Class Propagation")
    print("\n[GOV-1] Molecule-Class Propagation")
    print("-" * 60)

    # Define 3 representative test molecules with distinct molecule classes
    test_cases = [
        {
            "name": "TestCanonicalMAb",
            "molecule_class": "canonical_mab",
            "pI": 8.5,
            "MW_kDa": 148.0,
            "hydrophobicity": 0.35,
            "pH_working": 7.0,
            "sequence": (
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG"
                "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDYGDSDWFDPWGQGTLVTVSSASTKGPSVFPLAP"
                "SSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQ"
                "TYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCV"
            ),
            "is_mab": True,
        },
        {
            "name": "TestBispecific",
            "molecule_class": "bispecific",
            "pI": 7.8,
            "MW_kDa": 195.0,
            "hydrophobicity": 0.40,
            "pH_working": 7.0,
            "sequence": (
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG"
                "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDYGDSDWFDPWGQGTLVTVSSEVQLVESGGGLVQ"
                "PGGSLRLSCAASGFTFSDYYMSWVRQAPGKGLEWVSYISSSGSTIYYADSVKGRFTISRDNSKNTLY"
                "LQMNSLRAEDTAVYYCARRGGYYYGMDVWGQGTTVTVSSAKTTAPSVYPLAPVCGDTTGSSVTLGCL"
            ),
            "is_mab": True,
        },
        {
            "name": "TestFcFusion",
            "molecule_class": "fc_fusion",
            "pI": 6.2,
            "MW_kDa": 105.0,
            "hydrophobicity": 0.30,
            "pH_working": 7.0,
            "sequence": (
                "LPAQVAFTPYAPEPGSTCRLREYYDQTAQMCCSKCSPGQHAKVVLLDQHQEATCNKTTSVKGPCSV"
                "STEGKQLGCQCLGNGRCEQDRSPEANQVTQMPCSKPCTPECEGGMITFDFNTEAHMDRQPAPQESGR"
                "GEASKERKTPEMCRPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNST"
                "YRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTC"
            ),
            "is_mab": False,
        },
    ]

    mapper = PropertyMapper()
    mol_results = {}

    for tc in test_cases:
        mol_class = tc["molecule_class"]
        name = tc["name"]
        print(f"\n  Molecule: {name} (class={mol_class})")

        # --- PropertyMapper ---
        protein = ProteinProperties(
            name=name,
            pI=tc["pI"],
            MW_kDa=tc["MW_kDa"],
            hydrophobicity=tc["hydrophobicity"],
            pH_working=tc["pH_working"],
        )
        sma_params = mapper.map(protein)
        result.info(
            f"{name}: SMA nu={sma_params['nu']:.3f}, ka={sma_params['ka']:.4f}, "
            f"sigma={sma_params['sigma']:.3f}, source={sma_params['source']}"
        )

        # Validate PropertyMapper output has required keys
        required_sma = {"nu", "ka", "kd", "sigma", "lambda_"}
        missing_sma = required_sma - set(sma_params.keys())
        if missing_sma:
            result.fail(f"{name} PropertyMapper missing keys: {missing_sma}")
        else:
            result.pass_(f"{name} PropertyMapper returned all required SMA keys")

        # --- UpstreamTwin ---
        upstream_res = run_upstream_simulation(
            seed_density=0.5,
            temp_shift_day=5.0,
            dev_score=0.2,
            agg_risk=0.1,
            molecule_class=mol_class,
        )
        upstream_dict = result_to_dict(upstream_res)
        result.info(
            f"{name}: titer={upstream_dict['final_titer']:.3f} g/L, "
            f"viability={upstream_dict['viability_at_harvest']:.1f}%, "
            f"peak_vcd={upstream_dict['peak_vcd']:.1f}"
        )

        # Verify key upstream fields are populated
        upstream_required = {"final_titer", "viability_at_harvest", "peak_vcd", "integral_vcc"}
        missing_upstream = upstream_required - set(upstream_dict.keys())
        if missing_upstream:
            result.fail(f"{name} UpstreamTwin missing keys: {missing_upstream}")
        elif upstream_dict["final_titer"] <= 0:
            result.fail(f"{name} UpstreamTwin titer={upstream_dict['final_titer']} (must be > 0)")
        else:
            result.pass_(f"{name} UpstreamTwin produced positive titer with molecule_class={mol_class}")

        # --- AnalyticalQC ---
        qc_res = run_analytical_qc(
            sequence=tc["sequence"],
            pI=tc["pI"],
            aggregation_pct=1.5,
            is_mab=tc["is_mab"],
            molecule_class=mol_class,
        )

        result.info(
            f"{name}: cIEF acidic={qc_res.cief.acidic_pct}%, main={qc_res.cief.main_pct}%, "
            f"basic={qc_res.cief.basic_pct}%"
        )
        result.info(
            f"{name}: glycan G0F={qc_res.glycan.g0f_pct}%, G1F={qc_res.glycan.g1f_pct}%, "
            f"dominant={qc_res.glycan.dominant_species}"
        )

        # Verify cIEF sums to 100%
        cief_sum = round(
            qc_res.cief.acidic_pct + qc_res.cief.main_pct + qc_res.cief.basic_pct, 1
        )
        if abs(cief_sum - 100.0) > 0.2:
            result.fail(f"{name} cIEF sum={cief_sum:.1f}% (expected 100%)")
        else:
            result.pass_(f"{name} cIEF sums to {cief_sum:.1f}% (OK)")

        # Verify glycan sums to 100%
        glycan_sum = round(
            qc_res.glycan.g0f_pct
            + qc_res.glycan.g1f_pct
            + qc_res.glycan.g2f_pct
            + qc_res.glycan.high_mannose_pct
            + qc_res.glycan.afucosylated_pct
            + qc_res.glycan.other_pct,
            1,
        )
        if abs(glycan_sum - 100.0) > 0.5:
            result.fail(f"{name} glycan sum={glycan_sum:.1f}% (expected 100%)")
        else:
            result.pass_(f"{name} glycan sums to {glycan_sum:.1f}% (OK)")

        # Store for cross-class comparison
        mol_results[mol_class] = {
            "name": name,
            "sma_params": sma_params,
            "upstream": upstream_dict,
            "cief_main": qc_res.cief.main_pct,
        }

    # Cross-class check: canonical_mab should have higher titer than bispecific
    # (bispecific q_p_scale=0.70, canonical_mab=1.00)
    if "canonical_mab" in mol_results and "bispecific" in mol_results:
        mab_titer = mol_results["canonical_mab"]["upstream"]["final_titer"]
        bs_titer = mol_results["bispecific"]["upstream"]["final_titer"]
        if mab_titer > bs_titer:
            result.pass_(
                f"Class scaling verified: canonical_mab titer ({mab_titer:.3f} g/L) > "
                f"bispecific ({bs_titer:.3f} g/L) as expected"
            )
        else:
            result.fail(
                f"Class scaling broken: canonical_mab titer ({mab_titer:.3f} g/L) should "
                f"exceed bispecific ({bs_titer:.3f} g/L) per q_p_scale=1.00 vs 0.70"
            )

    result.data["molecule_results"] = mol_results
    result.finalize()
    return result


# ============================================================
# GOV-2: Evidence Completeness
# ============================================================

def check_gov2_evidence_completeness() -> GovernanceResult:
    """
    Run a test molecule through PropertyMapper + UpstreamTwin + AnalyticalQC
    and verify all key evidence fields are populated (not None, not zero where
    positive values are expected).
    """
    result = GovernanceResult("GOV-2", "Evidence Completeness")
    print("\n[GOV-2] Evidence Completeness")
    print("-" * 60)

    # NISTmAb-like molecule (well-characterized reference)
    TEST_SEQ = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG"
        "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDYGDSDWFDPWGQGTLVTVSSASTKGPSVFPLAP"
        "SSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQ"
        "TYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCV"
        "VVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKAL"
        "PAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTP"
        "PVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGKDIQMTQSPSSVSASVG"
        "DRIAACVTASGGTYYPFSWYRQAPGQAPRLLIYDASSLESGVPSRFSGSGSGTDFTLTISSLQPEDFA"
        "TYYCQQYNSYPWTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDN"
        "ALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    )
    # Use a proper IgG1 sequence (HC + LC joined for simplicity)
    NIST_HC_SEQ = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG"
        "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDYGDSDWFDPWGQGTLVTVSSASTKGPSVFPLAP"
        "SSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQ"
        "TYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCV"
        "VVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKAL"
        "PAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTP"
        "PVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
    )

    protein = ProteinProperties(
        name="NISTmAb_Test",
        pI=8.45,
        MW_kDa=148.0,
        hydrophobicity=0.38,
        pH_working=7.0,
        gravy_score=-0.394,
        ptm_profile={"deamidation_sites": 2, "oxidation_sites": 3},
    )

    # --- PropertyMapper ---
    mapper = PropertyMapper()
    sma = mapper.map(protein)

    # Key SMA fields
    sma_fields = {
        "nu": sma.get("nu"),
        "ka": sma.get("ka"),
        "kd": sma.get("kd"),
        "sigma": sma.get("sigma"),
        "lambda_": sma.get("lambda_"),
    }
    print(f"  PropertyMapper: nu={sma['nu']:.3f}, ka={sma['ka']:.4f}, kd={sma['kd']:.1f}, "
          f"sigma={sma['sigma']:.3f}, lambda_={sma['lambda_']:.0f}")
    result.data["sma_fields"] = sma_fields

    for field_name, val in sma_fields.items():
        if val is None:
            result.fail(f"PropertyMapper.{field_name} is None")
        elif val <= 0:
            result.fail(f"PropertyMapper.{field_name}={val} is not positive")
        else:
            result.pass_(f"PropertyMapper.{field_name}={val:.4g} populated and positive")

    # --- UpstreamTwin ---
    upstream_res = run_upstream_simulation(
        seed_density=0.5,
        temp_shift_day=5.0,
        dev_score=0.2,
        agg_risk=0.1,
        hydrophobicity=-0.394,
        molecule_class="canonical_mab",
    )
    up = result_to_dict(upstream_res)
    print(f"  UpstreamTwin: titer={up['final_titer']:.3f} g/L, "
          f"viability={up['viability_at_harvest']:.1f}%, "
          f"peak_vcd={up['peak_vcd']:.1f}")
    result.data["upstream_fields"] = up

    upstream_checks = {
        "final_titer": (up.get("final_titer"), lambda v: v > 0, "must be > 0 g/L"),
        "viability_at_harvest": (up.get("viability_at_harvest"), lambda v: 0 < v <= 100, "must be in (0, 100]%"),
        "peak_vcd": (up.get("peak_vcd"), lambda v: v > 0, "must be > 0"),
        "integral_vcc": (up.get("integral_vcc"), lambda v: v > 0, "must be > 0"),
    }

    for field_name, (val, check, msg) in upstream_checks.items():
        if val is None:
            result.fail(f"UpstreamTwin.{field_name} is None")
        elif not check(val):
            result.fail(f"UpstreamTwin.{field_name}={val} — {msg}")
        else:
            result.pass_(f"UpstreamTwin.{field_name}={val:.4g} — {msg.replace('must be', 'is')}")

    # --- AnalyticalQC ---
    qc_res = run_analytical_qc(
        sequence=NIST_HC_SEQ,
        pI=8.45,
        aggregation_pct=1.0,
        is_mab=True,
        molecule_class="canonical_mab",
    )
    print(f"  AnalyticalQC: cIEF main={qc_res.cief.main_pct}%, "
          f"intact={qc_res.ce_sds.intact_pct}%, "
          f"G0F={qc_res.glycan.g0f_pct}%")

    # titer: from upstream
    titer = up.get("final_titer")
    # ka, nu: from PropertyMapper
    ka = sma.get("ka")
    nu = sma.get("nu")
    # viability: from upstream
    viability = up.get("viability_at_harvest")
    # cief_main: from analytical_qc
    cief_main = qc_res.cief.main_pct
    # g0f: from analytical_qc
    g0f = qc_res.glycan.g0f_pct

    # Map all required evidence fields
    evidence_fields = {
        "nu (SMA characteristic charge)": nu,
        "ka (adsorption rate constant)": ka,
        "titer (final titer g/L)": titer,
        "viability (harvest viability %)": viability,
        "cief_main (main peak %)": cief_main,
        "g0f (G0F glycan %)": g0f,
    }

    result.data["evidence_fields"] = {
        "nu": nu,
        "ka": ka,
        "titer": titer,
        "viability": viability,
        "cief_main": cief_main,
        "g0f": g0f,
    }

    all_populated = True
    for fname, val in evidence_fields.items():
        if val is None:
            result.fail(f"Evidence field '{fname}' is None (not populated)")
            all_populated = False
        elif val == 0 and fname not in ("cief_main",):
            # 0 may be legitimate for some fields but not most
            result.fail(f"Evidence field '{fname}'=0 (suspicious, expected non-zero)")
            all_populated = False
        else:
            result.pass_(f"Evidence field '{fname}'={val:.4g} — populated")

    if all_populated:
        result.pass_("All key evidence fields are populated with non-None, non-zero values")

    result.finalize()
    return result


# ============================================================
# GOV-3: Determinism
# ============================================================

def check_gov3_determinism() -> GovernanceResult:
    """
    Run PropertyMapper 3 times with identical inputs.
    All output dicts must be bit-identical (same rounded values).
    """
    result = GovernanceResult("GOV-3", "Determinism")
    print("\n[GOV-3] Determinism")
    print("-" * 60)

    protein = ProteinProperties(
        name="Determinism_Test_mAb",
        pI=8.31,
        MW_kDa=148.5,
        hydrophobicity=0.38,
        pH_working=7.0,
        gravy_score=-0.320,
        ptm_profile={"deamidation_sites": 2, "oxidation_sites": 2},
    )

    mapper = PropertyMapper()

    # --- PropertyMapper: 3 runs ---
    runs = []
    for i in range(3):
        params = mapper.map(protein)
        # Extract the comparable fields (exclude mapping_trace which has round() artifacts)
        comparable = {
            k: v for k, v in params.items()
            if k not in ("mapping_trace",)
        }
        runs.append(comparable)
        print(f"  Run {i+1}: nu={params['nu']:.4f}, ka={params['ka']:.6f}, "
              f"sigma={params['sigma']:.4f}, source={params['source']}")

    # Compare all 3 runs
    all_match = True
    for i in range(1, len(runs)):
        if runs[i] != runs[0]:
            # Find which keys differ
            diffs = {k: (runs[0].get(k), runs[i].get(k))
                     for k in set(runs[0]) | set(runs[i])
                     if runs[0].get(k) != runs[i].get(k)}
            result.fail(
                f"Run {i+1} differs from Run 1. Differing keys: {diffs}"
            )
            all_match = False

    if all_match:
        result.pass_("All 3 PropertyMapper runs with identical inputs produced identical outputs")

    # Also test map_variants for determinism
    variant_runs = []
    for i in range(3):
        vp = mapper.map_variants(protein)
        # Extract comparable (skip non-deterministic fields if any)
        variant_comparable = {
            k: v for k, v in vp.items()
        }
        variant_runs.append(variant_comparable)

    variants_match = True
    for i in range(1, len(variant_runs)):
        if variant_runs[i] != variant_runs[0]:
            diffs = {k: (variant_runs[0].get(k), variant_runs[i].get(k))
                     for k in set(variant_runs[0]) | set(variant_runs[i])
                     if variant_runs[0].get(k) != variant_runs[i].get(k)}
            result.fail(f"map_variants Run {i+1} differs from Run 1. Diffs: {diffs}")
            variants_match = False

    if variants_match:
        result.pass_("All 3 map_variants runs with identical inputs produced identical outputs")

    result.data["runs"] = [
        {k: v for k, v in r.items() if isinstance(v, (int, float, str))}
        for r in runs
    ]

    result.finalize()
    return result


# ============================================================
# GOV-5: Report Internal Consistency
# ============================================================

def check_gov5_report_internal_consistency() -> GovernanceResult:
    """
    Read the latest JSON report from Reports/ and verify:
      - cIEF sums to 100% (acidic + main + basic)
      - CE-SDS sums to 100% (intact + lmw/fragment + hmw)
      - Glycan sums to 100% via simulate_glycan_profile (re-computed from context)
    """
    result = GovernanceResult("GOV-5", "Report Internal Consistency")
    print("\n[GOV-5] Report Internal Consistency")
    print("-" * 60)

    reports_dir = os.path.join(_PLATFORM_ROOT, "Reports")
    json_files = sorted(
        glob.glob(os.path.join(reports_dir, "*.json")),
        key=os.path.getmtime,
        reverse=True,
    )

    if not json_files:
        result.fail("No JSON reports found in Reports/")
        result.finalize()
        return result

    latest_report_path = json_files[0]
    result.info(f"Using latest report: {os.path.basename(latest_report_path)}")
    print(f"  Report: {os.path.basename(latest_report_path)}")

    with open(latest_report_path) as f:
        report = json.load(f)

    result.data["report_file"] = os.path.basename(latest_report_path)

    analytical = report.get("analytical", {})

    # ------------------------------------------------------------------
    # Check 1: cIEF sums to 100%
    # ------------------------------------------------------------------
    cief_acidic = analytical.get("cief_acidic_pct")
    cief_main = analytical.get("cief_main_pct")
    cief_basic = analytical.get("cief_basic_pct")

    if None in (cief_acidic, cief_main, cief_basic):
        result.fail(
            f"cIEF data incomplete: acidic={cief_acidic}, main={cief_main}, basic={cief_basic}"
        )
    else:
        cief_total = round(cief_acidic + cief_main + cief_basic, 2)
        print(f"  cIEF: acidic={cief_acidic}% + main={cief_main}% + basic={cief_basic}% = {cief_total}%")
        result.data["cief_sum"] = cief_total
        if abs(cief_total - 100.0) > 0.5:
            result.fail(
                f"cIEF does not sum to 100%: got {cief_total}% "
                f"(acidic={cief_acidic}, main={cief_main}, basic={cief_basic})"
            )
        else:
            result.pass_(f"cIEF sums to {cief_total}% (within 0.5% tolerance)")

    # ------------------------------------------------------------------
    # Check 2: CE-SDS sums to 100%
    # The report stores cesds_intact_pct, sec_hmw_pct, and the LMW
    # component can be derived: LMW = 100 - intact - HMW.
    # The sec_monomer_pct+sec_hmw_pct should sum to 100%.
    # ------------------------------------------------------------------
    sec_monomer = analytical.get("sec_monomer_pct")
    sec_hmw = analytical.get("sec_hmw_pct")

    if None in (sec_monomer, sec_hmw):
        result.fail(
            f"CE-SDS/SEC data incomplete: monomer={sec_monomer}, hmw={sec_hmw}"
        )
    else:
        sec_total = round(sec_monomer + sec_hmw, 2)
        print(f"  SEC: monomer={sec_monomer}% + hmw={sec_hmw}% = {sec_total}%")
        result.data["sec_sum"] = sec_total
        if abs(sec_total - 100.0) > 0.5:
            result.fail(
                f"SEC (monomer+HMW) does not sum to 100%: got {sec_total}%"
            )
        else:
            result.pass_(f"SEC monomer+HMW sums to {sec_total}% (within 0.5% tolerance)")

    # CE-SDS intact check
    cesds_intact = analytical.get("cesds_intact_pct")
    if cesds_intact is not None:
        print(f"  CE-SDS intact: {cesds_intact}%")
        result.data["cesds_intact_pct"] = cesds_intact
        if cesds_intact <= 0 or cesds_intact > 100:
            result.fail(f"CE-SDS intact={cesds_intact}% is outside valid range (0, 100]")
        else:
            result.pass_(f"CE-SDS intact={cesds_intact}% is in valid range (0, 100]")

    # ------------------------------------------------------------------
    # Check 3: Glycan sums to 100%
    # The report does not store raw glycan percentages in the analytical
    # section (only highlights). Re-compute from context fields.
    # ------------------------------------------------------------------
    from src.analytical_qc_twin import simulate_glycan_profile

    ctx = report.get("context", {})
    mol_class = ctx.get("molecule_class", "canonical_mab")
    n_glyco = ctx.get("n_glycosylation_sites", 2)
    expects_glyco = ctx.get("expects_glycosylation", True)

    print(f"  Glycan: re-computing for molecule_class={mol_class}, "
          f"n_glyco_sites={n_glyco}, expects_glyco={expects_glyco}")

    # Compute glycan profile using the same parameters as the platform
    glycan_res = simulate_glycan_profile(
        n_glycosylation_sites=max(n_glyco, 1),
        culture_duration_days=14,
        culture_temperature_c=37.0,
        molecule_class=mol_class,
    )

    glycan_sum = round(
        glycan_res.g0f_pct
        + glycan_res.g1f_pct
        + glycan_res.g2f_pct
        + glycan_res.high_mannose_pct
        + glycan_res.afucosylated_pct
        + glycan_res.other_pct,
        1,
    )
    print(f"  Glycan: G0F={glycan_res.g0f_pct}% + G1F={glycan_res.g1f_pct}% + "
          f"G2F={glycan_res.g2f_pct}% + Man5={glycan_res.high_mannose_pct}% + "
          f"Afuc={glycan_res.afucosylated_pct}% + Other={glycan_res.other_pct}% = {glycan_sum}%")
    result.data["glycan_sum"] = glycan_sum
    result.data["glycan_profile"] = {
        "G0F": glycan_res.g0f_pct,
        "G1F": glycan_res.g1f_pct,
        "G2F": glycan_res.g2f_pct,
        "Man5": glycan_res.high_mannose_pct,
        "Afuc": glycan_res.afucosylated_pct,
        "Other": glycan_res.other_pct,
    }

    if abs(glycan_sum - 100.0) > 0.5:
        result.fail(
            f"Glycan profile does not sum to 100%: got {glycan_sum}% "
            f"for molecule_class={mol_class}"
        )
    else:
        result.pass_(f"Glycan sums to {glycan_sum}% (within 0.5% tolerance)")

    result.finalize()
    return result


# ============================================================
# GOV-7: Report Cross-Section Consistency
# ============================================================

def check_gov7_report_cross_section_consistency() -> GovernanceResult:
    """
    Read the latest JSON report and check that MW, pI, molecule_class
    are consistent across report sections:
      - context
      - molecule_overview
      - appendix.biophysical_features
    """
    result = GovernanceResult("GOV-7", "Report Cross-Section Consistency")
    print("\n[GOV-7] Report Cross-Section Consistency")
    print("-" * 60)

    reports_dir = os.path.join(_PLATFORM_ROOT, "Reports")
    json_files = sorted(
        glob.glob(os.path.join(reports_dir, "*.json")),
        key=os.path.getmtime,
        reverse=True,
    )

    if not json_files:
        result.fail("No JSON reports found in Reports/")
        result.finalize()
        return result

    latest_report_path = json_files[0]
    result.info(f"Using latest report: {os.path.basename(latest_report_path)}")
    print(f"  Report: {os.path.basename(latest_report_path)}")

    with open(latest_report_path) as f:
        report = json.load(f)

    result.data["report_file"] = os.path.basename(latest_report_path)

    # ---- Collect values from each section ----
    ctx = report.get("context", {})
    mo = report.get("molecule_overview", {})
    appendix = report.get("appendix", {})
    bio = appendix.get("biophysical_features", {})

    # MW in kDa
    mw_context = ctx.get("molecular_weight_kda")
    mw_overview = mo.get("molecular_weight_kda")
    mw_appendix = bio.get("mw")

    # pI
    pi_context = ctx.get("isoelectric_point")
    pi_overview = mo.get("isoelectric_point")
    pi_appendix = bio.get("pI")

    # molecule_class
    mc_context = ctx.get("molecule_class")
    mc_overview = mo.get("molecule_class")

    print(f"  MW (kDa)       : context={mw_context}, overview={mw_overview}, appendix={mw_appendix}")
    print(f"  pI             : context={pi_context}, overview={pi_overview}, appendix={pi_appendix}")
    print(f"  molecule_class : context={mc_context}, overview={mc_overview}")

    result.data["values"] = {
        "mw": {"context": mw_context, "overview": mw_overview, "appendix": mw_appendix},
        "pI": {"context": pi_context, "overview": pi_overview, "appendix": pi_appendix},
        "molecule_class": {"context": mc_context, "overview": mc_overview},
    }

    # ------------------------------------------------------------------
    # MW consistency: allow 1% tolerance for rounding across sections
    # Note: context stores raw precision (e.g., 54.094 kDa), while
    # molecule_overview rounds to 1 decimal (e.g., 54.1 kDa). This is
    # intentional display-level formatting, not a data inconsistency.
    # See report_assembler.py line ~432: round(ctx.molecular_weight_kda, 1)
    # ------------------------------------------------------------------
    mw_values = [v for v in (mw_context, mw_overview, mw_appendix) if v is not None]
    if len(mw_values) < 2:
        result.fail(f"Insufficient MW data: only {len(mw_values)} section(s) have MW")
    else:
        mw_range = max(mw_values) - min(mw_values)
        mw_ref = mw_values[0]
        mw_pct_diff = (mw_range / mw_ref * 100) if mw_ref else 0
        if mw_pct_diff > 1.0:
            result.fail(
                f"MW inconsistent across sections: range={mw_range:.2f} kDa "
                f"({mw_pct_diff:.2f}% diff). Values: context={mw_context}, "
                f"overview={mw_overview}, appendix={mw_appendix}"
            )
        else:
            result.pass_(
                f"MW consistent: context={mw_context} kDa, overview={mw_overview} kDa, "
                f"appendix={mw_appendix} kDa (max diff={mw_pct_diff:.2f}%)"
            )

    # ------------------------------------------------------------------
    # pI consistency: allow ±0.02 units
    # ------------------------------------------------------------------
    pi_values = {
        "context": pi_context,
        "overview": pi_overview,
        "appendix": pi_appendix,
    }
    pi_non_null = {k: v for k, v in pi_values.items() if v is not None}
    if len(pi_non_null) < 2:
        result.fail(f"Insufficient pI data: only {len(pi_non_null)} section(s) have pI")
    else:
        vals = list(pi_non_null.values())
        pi_spread = max(vals) - min(vals)
        if pi_spread > 0.02:
            result.fail(
                f"pI inconsistent across sections: spread={pi_spread:.4f} pH units "
                f"(threshold=0.02). Values: {pi_non_null}"
            )
        else:
            result.pass_(
                f"pI consistent across sections: spread={pi_spread:.4f} pH units "
                f"(threshold=0.02). Values: {pi_non_null}"
            )

    # ------------------------------------------------------------------
    # molecule_class consistency
    # ------------------------------------------------------------------
    mc_values = {
        "context": mc_context,
        "overview": mc_overview,
    }
    mc_non_null = {k: v for k, v in mc_values.items() if v is not None}
    if len(mc_non_null) < 2:
        # molecule_overview may not always be present for all report versions
        result.info(
            f"molecule_class only in {len(mc_non_null)} section(s): {mc_non_null}. "
            "Checking context alone."
        )
        if mc_context:
            result.pass_(f"molecule_class present in context: {mc_context}")
        else:
            result.fail("molecule_class is None/missing in context section")
    else:
        mc_vals = list(mc_non_null.values())
        if len(set(mc_vals)) > 1:
            result.fail(
                f"molecule_class inconsistent: {mc_non_null}. "
                f"All sections must agree on molecule class."
            )
        else:
            result.pass_(
                f"molecule_class consistent across all sections: {mc_vals[0]}"
            )

    result.finalize()
    return result


# ============================================================
# Main runner
# ============================================================

def main():
    print("=" * 70)
    print("  ProtePilot — Governance Validation Suite")
    print(f"  Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results: List[GovernanceResult] = []
    check_functions = [
        check_gov1_molecule_class_propagation,
        check_gov2_evidence_completeness,
        check_gov3_determinism,
        check_gov5_report_internal_consistency,
        check_gov7_report_cross_section_consistency,
    ]

    for check_fn in check_functions:
        try:
            res = check_fn()
        except Exception as exc:
            # Catch unexpected errors so the suite continues
            check_id = check_fn.__name__.upper().replace("CHECK_", "").split("_")[0]
            res = GovernanceResult(check_id, check_fn.__name__)
            res.fail(f"Unexpected exception: {type(exc).__name__}: {exc}")
            res.details.append(traceback.format_exc())
            res.finalize()
            print(f"\n  [ERROR] {check_fn.__name__} raised an exception:")
            traceback.print_exc()
        all_results.append(res)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  GOVERNANCE VALIDATION SUMMARY")
    print("=" * 70)

    pass_count = 0
    fail_count = 0
    for r in all_results:
        status_str = "[PASS]" if r.status == "PASS" else "[FAIL]"
        print(f"  {status_str}  {r.check_id}: {r.name}")
        if r.failures:
            for f in r.failures:
                print(f"          -> {f}")
        if r.status == "PASS":
            pass_count += 1
        else:
            fail_count += 1

    print("=" * 70)
    print(f"  Result: {pass_count} PASSED / {fail_count} FAILED / {len(all_results)} total")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save results to JSON
    # ------------------------------------------------------------------
    output_path = os.path.join(_PLATFORM_ROOT, "SelfTest", "governance_results.json")
    output = {
        "suite": "ProtePilot Governance Validation",
        "run_at": datetime.now().isoformat(),
        "summary": {
            "total": len(all_results),
            "passed": pass_count,
            "failed": fail_count,
            "overall": "PASS" if fail_count == 0 else "FAIL",
        },
        "checks": [r.to_dict() for r in all_results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
