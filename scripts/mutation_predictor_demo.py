"""End-to-end Phase 2 demo: Trastuzumab CDR mutation ranking.

Runs the full Phase 2 pipeline on Trastuzumab and writes:
    reports/phase2/trastuzumab_top20_cdr.csv
    reports/phase2/trastuzumab_full_ranked.csv
    reports/phase2/trastuzumab_summary.json

Default device is CPU. On Apple Silicon pass --device mps for ~5-10× speedup.

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/mutation_predictor_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.mutation.masked_marginal import MaskedMarginalConfig, MaskedMarginalScorer  # noqa: E402
from src.mutation.pipeline import rank_mutations  # noqa: E402
from src.mutation.report import save_report, summarize, top_n  # noqa: E402


TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    ap.add_argument("--cdr-only", action="store_true", default=True,
                    help="Restrict to CDR positions (default).")
    ap.add_argument("--all-positions", action="store_true",
                    help="Include framework positions.")
    ap.add_argument("--out-dir", type=Path, default=_PROJECT_ROOT / "reports" / "phase2")
    ap.add_argument("--skip-guardrails", action="store_true",
                    help="Score only; don't filter by TAP/DI/CamSol.")
    args = ap.parse_args(argv)

    cdr_only = not args.all_positions

    print("=" * 60)
    print("Phase 2 mutation-predictor demo — Trastuzumab")
    print("=" * 60)
    print(f"VH length: {len(TRASTUZUMAB_VH)}")
    print(f"VL length: {len(TRASTUZUMAB_VL)}")
    print(f"CDR only:  {cdr_only}")
    print(f"Device:    {args.device}")
    print(f"Guardrails: {'off' if args.skip_guardrails else 'on'}")
    print()

    scorer = MaskedMarginalScorer(
        config=MaskedMarginalConfig(device=args.device),
    )

    t0 = time.time()
    scored = rank_mutations(
        vh=TRASTUZUMAB_VH,
        vl=TRASTUZUMAB_VL,
        cdr_only=cdr_only,
        scorer=scorer,
        apply_developability_guardrails=not args.skip_guardrails,
    )
    elapsed = time.time() - t0

    print(f"Scored {len(scored)} candidates in {elapsed:.1f}s.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Full ranked table.
    full_path = save_report(scored, args.out_dir / "trastuzumab_full_ranked.csv", top_n_rows=10_000)
    # Top 20 passing (default) table.
    df_top = top_n(scored, n=20, passing_only=not args.skip_guardrails)
    top_path = args.out_dir / "trastuzumab_top20_cdr.csv"
    df_top.to_csv(top_path, index=False)

    summary = summarize(scored)
    summary["n_candidates"] = len(scored)
    summary["elapsed_seconds"] = round(elapsed, 2)
    summary["device"] = args.device
    summary["cdr_only"] = cdr_only
    summary["guardrails"] = not args.skip_guardrails
    (args.out_dir / "trastuzumab_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nTop 20 ranked mutations (passes_guardrails=True):")
    print(df_top.to_string(
        index=False,
        columns=["mutation_label", "region", "llr", "wildtype_prob", "mutant_prob",
                 "tap_risk_flag_delta", "di_seq_proxy_delta", "camsol_intrinsic_mean_delta",
                 "passes_guardrails"],
        float_format=lambda x: f"{x:+.3f}",
    ))
    print(f"\nWritten: {top_path}")
    print(f"         {full_path}")
    print(f"         {args.out_dir / 'trastuzumab_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
