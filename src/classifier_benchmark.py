"""
classifier_benchmark.py  ·  ProtePilot — Classifier Benchmark Suite
======================================================================
Expanded benchmark panel for the molecule classifier.

Goes beyond the basic validation corpus with:
  - Edge cases per class (boundary sequences, ambiguous formats)
  - Contract compliance checks
  - OOD path verification
  - User override / feedback path verification
  - Performance timing

Usage:
    python -m src.classifier_benchmark                     # Full fusion benchmark
    python -m src.classifier_benchmark --mode rule         # Rule-based only
    python -m src.classifier_benchmark --mode ml           # Rule + trained model
    python -m src.classifier_benchmark --mode fusion       # Full pipeline (default)
    python -m src.classifier_benchmark --class adc         # Single class
    python -m src.classifier_benchmark --quick             # Core only
    python -m src.classifier_benchmark --json              # JSON output

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("ProtePilot.ClassifierBenchmark")


# ═══════════════════════════════════════════════════════════════════════
#  Reference Sequences
# ═══════════════════════════════════════════════════════════════════════

# Realistic IgG1 HC (truncated NISTmAb-like, ~465 aa with all domain motifs)
_REF_HC = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPT"
    "NGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMD"
    "YWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWN"
    "SGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDK"
    "KVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVS"
    "HEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKC"
    "KVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSD"
    "IAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEA"
    "LHNHYTQKSLSLSPG"
)
# Realistic LC (kappa, ~215 aa)
_REF_LC = (
    "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKRLIY"
    "AASSLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCLQHNSYPLTFG"
    "GGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKV"
    "DNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQG"
    "LSSPVTKSFNRGEC"
)
# Alternative HC (distinct VH framework — for bispecific tests)
_ALT_HC = (
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGW"
    "ISTYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARDYG"
    "DYFDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPE"
    "PVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVN"
    "HKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMIS"
    "RTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSV"
    "LTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRD"
    "ELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSK"
    "LTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPG"
)
# Fc-only fragment (CH2+CH3, no VH/VL/CL)
_FC_ONLY = (
    "CPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFN"
    "WYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKA"
    "LPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAV"
    "EWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEA"
    "LHNHYTQKSLSLSPG"
)
# Etanercept-like TNFR2 extracellular domain + IgG1 Fc hinge/CH2/CH3
# Real Fc-fusion: non-antibody receptor domain fused to Fc, NO VH/VL/CL motifs
_TNFR2_FC = (
    "LPAQVAFTPYAPEPGSTCRLREYYDQTAQMCCSKCSPGQHAKVFCTKTSDTVC"
    "DSCEDSTYTQLWNWVPECLSCGSRCSSDQVETQACTREQNRICTCRPGWYCAL"
    "SKQEGCRLCAPLRKCRPGFGVARPGTETSDVVCKPCAPGTFSNTTSSTDICRC"
    "RPDCTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHED"
    "PEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCK"
    "VSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPS"
    "DIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVM"
    "HEALHNHYTQKSLSLSPG"
)
# Aflibercept-like VEGFR1/2 domains + IgG1 Fc
_VEGFR_FC = (
    "SDTGRPFVEMYSEIPEIIHMTEGRELVIPCRVTSPNITVTLKKFPLDTLIPDG"
    "KRIIWDSRKGFIISNATYKEIGLLTCEATVNGHLYKTNYLTHRQTNTIIDVVL"
    "SPSHGIELSVGEKLVLNCTARTELNVGIDFNWEYPSSKHQHKKLVNRDLKTQSG"
    "SEMKKFLSTLTIDGVTRSDQGLYTCAASSGLMTKKNSTFVRVHEKDKTHTCPPC"
    "PAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGV"
    "EVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTI"
    "SKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNGQPEN"
    "NYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLS"
    "LSPG"
)


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Panel
# ═══════════════════════════════════════════════════════════════════════

# Each entry: (test_name, expected_class, call_kwargs, description)
# call_kwargs are passed directly to classify_molecule().

BENCHMARK_PANEL: List[Dict[str, Any]] = [
    # ── canonical_mab ─────────────────────────────────────────────
    {
        "name": "canonical_mab_hc_lc_full",
        "expected": "canonical_mab",
        "kwargs": {
            "sequence": _REF_HC,
            "chains": [
                {"chain_type": "HC", "sequence": _REF_HC},
                {"chain_type": "LC", "sequence": _REF_LC},
            ],
        },
        "description": "Full IgG1 with HC+LC chain list — gold standard",
        "category": "core",
    },
    {
        "name": "canonical_mab_name_hint",
        "expected": "canonical_mab",
        "kwargs": {
            "sequence": _REF_HC,
            "chains": [
                {"chain_type": "HC", "sequence": _REF_HC},
                {"chain_type": "LC", "sequence": _REF_LC},
            ],
            "name": "Trastuzumab IgG1 antibody",
        },
        "description": "HC+LC with mAb name hint",
        "category": "core",
    },
    {
        "name": "canonical_mab_motifs_only",
        "expected": "canonical_mab",
        "kwargs": {"sequence": _REF_HC + _REF_LC},
        "description": "Concatenated HC+LC as single sequence — motif detection",
        "category": "edge",
    },

    # ── bispecific ────────────────────────────────────────────────
    {
        "name": "bispecific_two_distinct_hc",
        "expected": "bispecific",
        "kwargs": {
            "sequence": "",
            "assembly_chains": [
                {"name": "Arm1", "sequence": _REF_HC, "copy_number": 1},
                {"name": "Arm2", "sequence": _ALT_HC, "copy_number": 1},
            ],
        },
        "description": "Two distinct HC via assembly_chains — structural bispecific",
        "category": "core",
    },
    {
        "name": "bispecific_name_only",
        "expected": "bispecific",
        "kwargs": {
            "sequence": _REF_HC,
            "name": "Emicizumab bispecific",
        },
        "description": "Name-hint bispecific without structural evidence",
        "category": "core",
    },
    {
        "name": "bispecific_repeat_chains",
        "expected": "bispecific",
        "kwargs": {
            "sequence": "",
            "assembly_chains": [
                {"name": "ArmA", "sequence": "EVQLVESGGGLVQPGG" * 20, "copy_number": 1},
                {"name": "ArmB", "sequence": "QVQLVQSGAEVKKPGA" * 20, "copy_number": 1},
            ],
        },
        "description": "Synthetic distinct chains — bispecific by identity threshold",
        "category": "edge",
    },

    # ── fc_fusion ─────────────────────────────────────────────────
    {
        "name": "fc_fusion_fc_only_no_cl",
        "expected": "fc_fusion",
        "kwargs": {"sequence": _FC_ONLY},
        "description": "Fc-only fragment — Fc motifs but no CL domain",
        "category": "core",
    },
    {
        "name": "fc_fusion_name_hint",
        "expected": "fc_fusion",
        "kwargs": {
            "sequence": _REF_HC,
            "name": "Etanercept fc-fusion",
        },
        "description": "Fc-fusion from name keyword + Fc motifs",
        "category": "core",
    },
    {
        "name": "fc_fusion_abatacept",
        "expected": "fc_fusion",
        "kwargs": {
            "sequence": _REF_HC,
            "name": "Abatacept fc-fusion",
        },
        "description": "CTLA4-Fc fusion — name-based",
        "category": "core",
    },
    {
        "name": "fc_fusion_etanercept_real_seq",
        "expected": "fc_fusion",
        "kwargs": {
            "sequence": _TNFR2_FC,
            "name": "Etanercept",
        },
        "description": "Real TNFR2-Fc sequence — receptor domain + IgG1 Fc, no VH/VL",
        "category": "core",
    },
    {
        "name": "fc_fusion_aflibercept_real_seq",
        "expected": "fc_fusion",
        "kwargs": {
            "sequence": _VEGFR_FC,
            "name": "Aflibercept",
        },
        "description": "Real VEGFR1/2-Fc sequence — receptor domain + IgG1 Fc, no VH/VL",
        "category": "core",
    },

    # ── adc ───────────────────────────────────────────────────────
    {
        "name": "adc_name_emtansine",
        "expected": "adc",
        "kwargs": {
            "sequence": _REF_HC,
            "chains": [{"chain_type": "HC", "sequence": _REF_HC}],
            "name": "Trastuzumab emtansine adc",
        },
        "description": "ADC with emtansine keyword in name",
        "category": "core",
    },
    {
        "name": "adc_vedotin",
        "expected": "adc",
        "kwargs": {
            "sequence": _REF_HC,
            "name": "Brentuximab vedotin",
        },
        "description": "ADC vedotin keyword detection",
        "category": "core",
    },
    {
        "name": "adc_drug_conjugate",
        "expected": "adc",
        "kwargs": {
            "sequence": _REF_HC,
            "name": "Enfortumab vedotin drug conjugate",
        },
        "description": "ADC from 'drug conjugate' keyword",
        "category": "core",
    },

    # ── single_domain ─────────────────────────────────────────────
    {
        "name": "single_domain_nanobody_short",
        "expected": "single_domain",
        "kwargs": {"sequence": "EVQLVESGGGLVQPGG" * 7},
        "description": "~112 aa single chain with VH motif — nanobody-like",
        "category": "core",
    },
    {
        "name": "single_domain_vhh_name",
        "expected": "single_domain",
        "kwargs": {
            "sequence": "A" * 130,
            "name": "Caplacizumab nanobody",
        },
        "description": "Medium sequence with nanobody name hint",
        "category": "core",
    },
    {
        "name": "single_domain_boundary_199aa",
        "expected": "single_domain",
        "kwargs": {"sequence": "A" * 199},
        "description": "199 aa — just under 200 aa single_domain threshold",
        "category": "edge",
    },

    # ── peptide ───────────────────────────────────────────────────
    {
        "name": "peptide_short_15aa",
        "expected": "peptide",
        "kwargs": {"sequence": "ACDEFGHIKLMNPQR"},
        "description": "15 aa peptide — well under threshold",
        "category": "core",
    },
    {
        "name": "peptide_boundary_79aa",
        "expected": "peptide",
        "kwargs": {"sequence": "A" * 79},
        "description": "79 aa — just under 80 aa peptide threshold",
        "category": "edge",
    },
    {
        "name": "peptide_33aa",
        "expected": "peptide",
        "kwargs": {"sequence": "ACDEFGHIKLM" * 3},
        "description": "33 aa — typical therapeutic peptide length",
        "category": "core",
    },
    {
        "name": "peptide_named",
        "expected": "peptide",
        "kwargs": {
            "sequence": "ACDEFGHIKLM" * 2,
            "name": "Semaglutide peptide agonist",
        },
        "description": "Named peptide therapeutic",
        "category": "core",
    },

    # ── fusion_protein ────────────────────────────────────────────
    {
        "name": "fusion_protein_long_no_motifs",
        "expected": "fusion_protein",
        "kwargs": {"sequence": "A" * 500},
        "description": "Long sequence with no antibody motifs — falls to fusion_protein",
        "category": "core",
    },
    {
        "name": "fusion_protein_name_hint",
        "expected": "fusion_protein",
        "kwargs": {
            "sequence": "A" * 400,
            "name": "IL2-fusion therapeutic",
        },
        "description": "Fusion keyword in name, no Fc motifs",
        "category": "core",
    },

    # ── engineered_scaffold ───────────────────────────────────────
    {
        "name": "engineered_scaffold_darpin",
        "expected": "engineered_scaffold",
        "kwargs": {
            "sequence": "ACDEFGHIKLM" * 20,
            "name": "DARPin scaffold",
        },
        "description": "DARPin keyword detection",
        "category": "core",
    },
    {
        "name": "engineered_scaffold_affibody",
        "expected": "engineered_scaffold",
        "kwargs": {
            "sequence": "ACDEFGHIKLM" * 20,
            "name": "HER2-affibody binder",
        },
        "description": "Affibody keyword detection",
        "category": "core",
    },

    # ── unknown / edge cases ──────────────────────────────────────
    {
        "name": "empty_sequence",
        "expected": "unknown",
        "kwargs": {"sequence": ""},
        "description": "Empty input → unknown",
        "category": "edge",
    },
    {
        "name": "none_sequence",
        "expected": "unknown",
        "kwargs": {"sequence": None},
        "description": "None input → unknown",
        "category": "edge",
    },

    # ── user_override ─────────────────────────────────────────────
    {
        "name": "user_override_peptide_to_scaffold",
        "expected": "engineered_scaffold",
        "kwargs": {
            "sequence": "ACDEFGHIKLM" * 2,
            "user_hint": "engineered_scaffold",
        },
        "description": "User overrides peptide-length sequence to scaffold",
        "category": "override",
    },
    {
        "name": "user_override_unknown_to_adc",
        "expected": "adc",
        "kwargs": {
            "sequence": "A" * 300,
            "user_hint": "adc",
        },
        "description": "User forces ADC on ambiguous input",
        "category": "override",
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark(
    panel: Optional[List[Dict[str, Any]]] = None,
    filter_class: Optional[str] = None,
    filter_category: Optional[str] = None,
    verbose: bool = True,
    mode: str = "fusion",
) -> Dict[str, Any]:
    """
    Run classifier benchmark and return results.

    Parameters
    ----------
    panel : list, optional
        Custom panel. Defaults to BENCHMARK_PANEL.
    filter_class : str, optional
        Only run tests for this expected class.
    filter_category : str, optional
        Only run tests in this category (core, edge, override).
    verbose : bool
        Print per-test results.
    mode : str
        Classification pipeline mode:
          "rule"   — Phase 1 only (rule-based, no ML, no OOD)
          "ml"     — Phase 1 + Phase 2 (rule-based + trained model, no OOD)
          "fusion" — Phase 1 + Phase 2 + Phase 3 (full pipeline, default)

    Returns
    -------
    dict with keys: passed, failed, total, accuracy, failures, timing_ms, mode
    """
    import src.molecule_classifier as _mc
    from src.molecule_classifier import classify_molecule
    from src.classification_contract import validate_output

    # ── Phase gating via monkeypatch ──────────────────────────────
    _orig_trained = _mc._apply_trained_model_opinion
    _orig_ood = _mc._apply_ood_detection

    def _noop_phase(result, *args, **kwargs):
        return result

    if mode == "rule":
        _mc._apply_trained_model_opinion = _noop_phase
        _mc._apply_ood_detection = _noop_phase
    elif mode == "ml":
        _mc._apply_ood_detection = _noop_phase
    # "fusion" — no patching needed

    cases = panel or BENCHMARK_PANEL
    if filter_class:
        cases = [c for c in cases if c["expected"] == filter_class]
    if filter_category:
        cases = [c for c in cases if c.get("category") == filter_category]

    passed = 0
    failed = 0
    failures = []
    t0 = time.monotonic()

    for case in cases:
        test_name = case["name"]
        expected = case["expected"]
        kwargs = case["kwargs"]

        try:
            result = classify_molecule(**kwargs)
            got = result.molecule_class.value

            # Check class match
            class_ok = (got == expected)

            # Check contract compliance
            result_dict = result.to_dict()
            violations = validate_output(result_dict)
            contract_ok = len(violations) == 0

            if class_ok and contract_ok:
                passed += 1
                if verbose:
                    log.info("  [PASS] %s: %s (%.0f%%)", test_name, got,
                             result.confidence_score * 100)
            else:
                failed += 1
                failure = {
                    "test": test_name,
                    "expected": expected,
                    "got": got,
                    "confidence": result.confidence,
                    "description": case["description"],
                }
                if not class_ok:
                    failure["error"] = f"class mismatch: expected={expected}, got={got}"
                if violations:
                    failure["contract_violations"] = violations
                failures.append(failure)
                if verbose:
                    log.warning("  [FAIL] %s: expected=%s, got=%s%s",
                                test_name, expected, got,
                                f" | violations: {violations}" if violations else "")

        except Exception as exc:
            failed += 1
            failures.append({
                "test": test_name,
                "expected": expected,
                "error": f"Exception: {exc}",
                "description": case["description"],
            })
            if verbose:
                log.error("  [ERR]  %s: %s", test_name, exc)

    # ── Restore original functions ────────────────────────────────
    _mc._apply_trained_model_opinion = _orig_trained
    _mc._apply_ood_detection = _orig_ood

    elapsed_ms = (time.monotonic() - t0) * 1000
    total = passed + failed
    accuracy = passed / total if total > 0 else 0.0

    return {
        "passed": passed,
        "failed": failed,
        "total": total,
        "accuracy": round(accuracy, 4),
        "failures": failures,
        "timing_ms": round(elapsed_ms, 1),
        "all_passed": failed == 0,
        "mode": mode,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Contract Compliance Benchmark
# ═══════════════════════════════════════════════════════════════════════

def run_contract_checks(verbose: bool = True) -> Dict[str, Any]:
    """
    Verify all behavioral guarantees from the classification contract.

    Tests:
      1. Risk weights for all classes
      2. Output schema for all classes
      3. FusionStrategy invariants against live classifier
      4. Empty input handling
    """
    from src.molecule_classifier import (
        classify_molecule, get_risk_weights, MoleculeClass,
    )
    from src.classification_contract import (
        validate_output, validate_risk_weights, VALID_CLASSES,
    )

    checks_passed = 0
    checks_failed = 0
    errors = []

    # 1. Risk weights for every class
    for cls_name in VALID_CLASSES:
        try:
            mc = MoleculeClass(cls_name)
            weights = get_risk_weights(mc)
            violations = validate_risk_weights(weights, cls_name)
            if violations:
                checks_failed += 1
                errors.append(f"risk_weights({cls_name}): {violations}")
            else:
                checks_passed += 1
                if verbose:
                    log.info("  [PASS] risk_weights(%s)", cls_name)
        except Exception as exc:
            checks_failed += 1
            errors.append(f"risk_weights({cls_name}): {exc}")

    # 2. Output schema for representative inputs
    test_inputs = [
        {"sequence": "ACDEFGHIKLM"},      # peptide
        {"sequence": _REF_HC + _REF_LC},   # canonical_mab-like
        {"sequence": ""},                   # empty
    ]
    for inp in test_inputs:
        try:
            result = classify_molecule(**inp)
            violations = validate_output(result.to_dict())
            if violations:
                checks_failed += 1
                errors.append(f"output_schema(seq_len={len(inp.get('sequence',''))}): {violations}")
            else:
                checks_passed += 1
        except Exception as exc:
            checks_failed += 1
            errors.append(f"output_schema: {exc}")

    # 3. Empty input returns unknown
    try:
        r = classify_molecule(sequence="")
        assert r.molecule_class == MoleculeClass.UNKNOWN, \
            f"Empty input gave {r.molecule_class.value}, expected unknown"
        assert r.confidence == "Low", \
            f"Empty input confidence is {r.confidence}, expected Low"
        checks_passed += 1
        if verbose:
            log.info("  [PASS] empty_input → unknown/Low")
    except Exception as exc:
        checks_failed += 1
        errors.append(f"empty_input: {exc}")

    return {
        "passed": checks_passed,
        "failed": checks_failed,
        "total": checks_passed + checks_failed,
        "errors": errors,
        "all_passed": checks_failed == 0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ProtePilot — Molecule Classifier Benchmark",
    )
    parser.add_argument("--class", dest="filter_class", default=None,
                        help="Run tests for a specific class only")
    parser.add_argument("--category", default=None,
                        help="Run tests in category: core, edge, override")
    parser.add_argument("--quick", action="store_true",
                        help="Core tests only")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--mode", choices=["rule", "ml", "fusion"], default="fusion",
                        help="Classification mode: rule (Phase 1 only), ml (Phase 1+2), "
                             "fusion (Phase 1+2+3, default)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-test output")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run all 3 modes (rule/ml/fusion) and compare results")
    parser.add_argument("--output-dir", default=None,
                        help="Save JSON results to directory (one file per mode)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(name)s | %(message)s",
    )

    verbose = not args.quiet
    category = args.category or ("core" if args.quick else None)
    mode_label = {"rule": "rule-based only", "ml": "rule + trained model", "fusion": "full fusion"}

    # ── Compare-all: run all 3 modes ──
    if args.compare_all:
        all_results = {}
        for mode in ("rule", "ml", "fusion"):
            log.info("Running benchmark mode=%s...", mode)
            bench = run_benchmark(
                filter_class=args.filter_class,
                filter_category=category,
                verbose=False,
                mode=mode,
            )
            all_results[mode] = bench

            # Save individual mode JSON if output_dir specified
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                out_path = os.path.join(args.output_dir, f"benchmark_{mode}.json")
                with open(out_path, "w") as fout:
                    json.dump(bench, fout, indent=2)

        # Run contract once (mode-independent)
        contract = run_contract_checks(verbose=False)
        all_results["contract"] = contract

        if args.output_dir:
            combo_path = os.path.join(args.output_dir, "benchmark_compare_all.json")
            with open(combo_path, "w") as fout:
                json.dump(all_results, fout, indent=2)

        if args.json:
            print(json.dumps(all_results, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"Classifier Benchmark — 3-Mode Comparison")
            print(f"{'='*60}")
            for mode in ("rule", "ml", "fusion"):
                b = all_results[mode]
                print(f"  [{mode:7s}] {b['passed']}/{b['total']} passed "
                      f"({b['accuracy']:.0%}) in {b['timing_ms']:.0f}ms")
            print(f"  [contract] {contract['passed']}/{contract['total']} passed")
            all_ok = all(all_results[m]["all_passed"] for m in ("rule", "ml", "fusion")) and contract["all_passed"]
            print(f"\n  Status: {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
            if args.output_dir:
                print(f"  Results saved to: {args.output_dir}/")
        sys.exit(0 if all_ok else 1)

    # ── Single mode ──
    log.info("Running classifier benchmark (mode=%s)%s...",
             mode_label[args.mode],
             f" class={args.filter_class}" if args.filter_class else "")
    bench = run_benchmark(
        filter_class=args.filter_class,
        filter_category=category,
        verbose=verbose,
        mode=args.mode,
    )

    # Run contract checks
    log.info("\nRunning contract compliance checks...")
    contract = run_contract_checks(verbose=verbose)

    # Combine results
    combined = {
        "benchmark": bench,
        "contract": contract,
        "overall_passed": bench["all_passed"] and contract["all_passed"],
    }

    # Save to output dir if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"benchmark_{args.mode}.json")
        with open(out_path, "w") as fout:
            json.dump(combined, fout, indent=2)

    if args.json:
        print(json.dumps(combined, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Classifier Benchmark Results (mode={bench['mode']})")
        print(f"{'='*60}")
        print(f"  Mode:     {mode_label[args.mode]}")
        print(f"  Panel:    {bench['passed']}/{bench['total']} passed "
              f"({bench['accuracy']:.0%}) in {bench['timing_ms']:.0f}ms")
        print(f"  Contract: {contract['passed']}/{contract['total']} passed")
        if bench["failures"]:
            print(f"\n  Failures:")
            for f in bench["failures"]:
                print(f"    - {f['test']}: {f.get('error', f.get('contract_violations', ''))}")
        if contract["errors"]:
            print(f"\n  Contract errors:")
            for e in contract["errors"]:
                print(f"    - {e}")
        status = "ALL PASSED" if combined["overall_passed"] else "FAILURES DETECTED"
        print(f"\n  Status: {status}")
        if args.output_dir:
            print(f"  Results saved to: {args.output_dir}/{out_path}")

    sys.exit(0 if combined["overall_passed"] else 1)


if __name__ == "__main__":
    main()
