"""Phase 3 end-to-end demo: structural analysis + structure-aware mutation ranking.

Pulls the Trastuzumab 1N8Z structure (downloaded in Phase 3), runs the
full structure analyzer, and enriches the Phase 2 mutation ranking with
per-residue SASA, interface / paratope / patch flags, and structure
risk.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/phase3_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.mutation.masked_marginal import MaskedMarginalConfig, MaskedMarginalScorer  # noqa: E402
from src.mutation.pipeline import rank_mutations  # noqa: E402
from src.mutation.report import scored_to_dataframe, top_n  # noqa: E402
from src.structure.analyzer import analyze_structure  # noqa: E402
from src.structure.mutation_enrich import enrich_mutations  # noqa: E402
from src.structure.predictor import register_user_pdb  # noqa: E402
from src.structure.schema import StructureSource  # noqa: E402


TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)

DEFAULT_PDB = _PROJECT_ROOT / "data" / "structures" / "1n8z.pdb"
CACHE_ROOT = _PROJECT_ROOT / "runs" / "structures"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", type=Path, default=DEFAULT_PDB,
                    help="Path to Trastuzumab PDB (default: data/structures/1n8z.pdb)")
    ap.add_argument("--vh-chain", default="B")
    ap.add_argument("--vl-chain", default="A")
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=_PROJECT_ROOT / "reports" / "phase3")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("Phase 3 demo — structural analysis + structure-aware ranking")
    print("=" * 62)

    # 1. Structure analysis -------------------------------------------------
    if not args.pdb.exists():
        raise SystemExit(
            f"PDB not found: {args.pdb}. Run Phase 3 setup to download 1N8Z."
        )

    print(f"\n[1/3] Registering structure: {args.pdb}")
    inp = register_user_pdb(
        vh=TRASTUZUMAB_VH, vl=TRASTUZUMAB_VL, pdb_path=args.pdb,
        cache_root=CACHE_ROOT, source=StructureSource.EXPERIMENTAL_PDB,
        confidence=1.0, pdb_id="1N8Z",
    )
    inp = replace(inp, vh_chain_id=args.vh_chain, vl_chain_id=args.vl_chain)

    t0 = time.time()
    metrics = analyze_structure(inp)
    elapsed_struct = time.time() - t0
    print(f"      analyze_structure done in {elapsed_struct:.1f}s")

    print("\n[structure summary]")
    print(f"  Available:          {metrics.available}")
    print(f"  VH length:          {len(metrics.vh_resnames)}")
    print(f"  VL length:          {len(metrics.vl_resnames)}")
    if metrics.interface:
        print(f"  VH-VL BSA:          {metrics.interface.buried_surface_area_a2:.0f} Å²")
    if metrics.paratope:
        print(f"  Paratope residues:  {len(metrics.paratope.cdr_residues)}")
        print(f"  Paratope SASA:      {metrics.paratope.total_surface_area_a2:.0f} Å²")
    print(f"  Aggregation patches: {len(metrics.aggregation_patches)}")
    print(f"  TAP PSH/PPC/PNC:    {metrics.tap_psh:.0f}/{metrics.tap_ppc:.0f}/{metrics.tap_pnc:.0f}")
    print(f"  TAP risk flags:     {metrics.tap_risk_flag_count}")
    print(f"  DI full (SAP-based): {metrics.di_full:+.2f}")
    print(f"  Structure risk:     {metrics.structure_risk_score:.2f}")

    (args.out_dir / "trastuzumab_structure_metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2)
    )

    # 2. Phase 2 mutation ranking -----------------------------------------
    print("\n[2/3] Ranking CDR mutations (ESM-2 masked-marginal + guardrails)")
    scorer = MaskedMarginalScorer(config=MaskedMarginalConfig(device=args.device))
    t0 = time.time()
    scored = rank_mutations(
        vh=TRASTUZUMAB_VH, vl=TRASTUZUMAB_VL, cdr_only=True,
        scorer=scorer, apply_developability_guardrails=True,
    )
    elapsed_mut = time.time() - t0
    print(f"      scored {len(scored)} candidates in {elapsed_mut:.1f}s")

    # 3. Structure enrichment --------------------------------------------
    print("\n[3/3] Enriching mutations with structure context")
    enriched = enrich_mutations(scored, metrics)
    # Count how many top-LLR mutations hit a risky position.
    top = top_n(enriched, n=args.top_n, passing_only=True)
    in_patch = int(top["in_aggregation_patch"].sum()) if "in_aggregation_patch" in top else 0
    at_iface = int(top["is_at_interface"].sum()) if "is_at_interface" in top else 0
    in_para = int(top["is_in_paratope"].sum()) if "is_in_paratope" in top else 0
    print(f"      of top-{args.top_n}: "
          f"{in_patch} in agg patch, {at_iface} at VH-VL iface, {in_para} in paratope")

    out_csv = args.out_dir / "trastuzumab_top20_structural.csv"
    cols = ["mutation_label", "region", "llr",
            "sasa_at_position", "is_at_interface", "is_in_paratope",
            "in_aggregation_patch", "structure_confidence",
            "passes_guardrails", "rationale"]
    # top is a DataFrame; extras are in the underlying ScoredMutation objects.
    # Rebuild the DataFrame so extras are columns, not buried in a dict.
    import pandas as pd
    rows = []
    for s in enriched:
        row = s.to_row()
        row.update(s.extras or {})
        rows.append(row)
    full_df = pd.DataFrame(rows).sort_values(
        by=["composite_score", "mutation_label"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    head = full_df[full_df["passes_guardrails"]].head(args.top_n)
    have_cols = [c for c in cols if c in head.columns]
    head[have_cols].to_csv(out_csv, index=False)
    print(f"\n[output] {out_csv}")
    print(head[have_cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
