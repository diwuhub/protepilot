"""
generative_engineer.py  ·  ProtePilot — Milestone 15
===========================================================
Generative Protein Engineering: In-Silico Mutagenesis + Pareto Filtering

Version   : 2.0 (Data-Driven Multi-Objective Optimization)
Author    : Di (ProtePilot)
Depends   : biopython (optional), numpy

Purpose
------------------------------------------------------------
Proactively suggests optimized sequences when the analytical
pipeline detects liabilities or purification bottlenecks.
Closes the DMTA (Design-Make-Test-Analyze) loop in-silico.

Key Capabilities
------------------------------------------------------------
  1. Liability-Targeted Mutagenesis
     - Oxidation risk (Met M → Leu L; Trp W → Phe F)
     - Deamidation hotspots (NG → NA/QG; NS → QS/NA)
     - Asp-Pro clipping (DP → EP or DA)
     - Isomerization (DG → EG; DS → ES)

  2. Charge Engineering for Bispecific Resolution
     - If Rs < 1.0 (homodimer co-elution), systematically mutate
       surface-exposed neutral/basic residues to acidic (D/E) on the
       offending chain to force a pI shift and widen resolution

  3. pI-Targeted Engineering
     - Shift pI toward the optimal 6.0-8.5 window for FcRn recycling
     - Acidify: K→Q, R→Q, H→N surface mutations
     - Basify: E→Q, D→N surface mutations

  4. Hydrophobicity Reduction
     - Replace exposed hydrophobic patches (V/I/L/F → T/S/N/Q)
     - Targets CDR surface residues preferentially

References
------------------------------------------------------------
  Estep et al. (2013) mAbs 5:270 — Sequence mutagenesis for developability
  Raybould et al. (2019) Bioinformatics 35:4983 — Therapeutic antibody profiling
  Shan et al. (2020) Nature Biomed Eng 4:1 — Computational antibody design
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("ProtePilot.GenerativeEngineer")


# ===========================================================================
# 1. Mutation Definitions & Conservation Rules
# ===========================================================================

# Conservative amino acid substitutions that preserve structure
# Format: original_aa -> [(replacement, penalty_score, rationale)]
# Penalty 0.0 = perfect conservation; 1.0 = disruptive
CONSERVATIVE_SUBS: Dict[str, List[Tuple[str, float, str]]] = {
    "M": [("L", 0.1, "Hydrophobic isostere; eliminates oxidation"), ("I", 0.15, "Hydrophobic; slightly bulkier")],
    "W": [("F", 0.2, "Aromatic; removes indole oxidation"), ("Y", 0.25, "Aromatic; adds hydroxyl")],
    "N": [("Q", 0.1, "Amide isostere; blocks NG/NS deamidation"), ("A", 0.3, "Small; may affect H-bond network")],
    "D": [("E", 0.1, "Acidic isostere; blocks DP clipping / DG isomerization"), ("N", 0.2, "Polar; removes charge")],
}

# Liability motif patterns with fix strategies
# Each strategy: (regex_pattern, fix_function_name, description)
LIABILITY_FIX_STRATEGIES = {
    "Oxidation (Met)": {
        "pattern": re.compile(r"M"),
        "fix": lambda seq, pos: seq[:pos] + "L" + seq[pos + 1:],
        "replacement": "L",
        "rationale": "Met→Leu: hydrophobic isostere, eliminates sulfoxide formation",
    },
    "Oxidation (Trp)": {
        "pattern": re.compile(r"W"),
        "fix": lambda seq, pos: seq[:pos] + "F" + seq[pos + 1:],
        "replacement": "F",
        "rationale": "Trp→Phe: aromatic ring preserved, removes indole oxidation",
    },
    "Deamidation (NG)": {
        "pattern": re.compile(r"NG"),
        "fix": lambda seq, pos: seq[:pos] + "QG" + seq[pos + 2:],
        "replacement": "QG",
        "rationale": "Asn→Gln at NG: amide isostere blocks succinimide intermediate",
    },
    "Deamidation (NS)": {
        "pattern": re.compile(r"NS"),
        "fix": lambda seq, pos: seq[:pos] + "QS" + seq[pos + 2:],
        "replacement": "QS",
        "rationale": "Asn→Gln at NS: amide isostere blocks deamidation",
    },
    "Deamidation (NT)": {
        "pattern": re.compile(r"NT"),
        "fix": lambda seq, pos: seq[:pos] + "QT" + seq[pos + 2:],
        "replacement": "QT",
        "rationale": "Asn→Gln at NT: amide isostere blocks deamidation",
    },
    "Asp-Pro Clip": {
        "pattern": re.compile(r"DP"),
        "fix": lambda seq, pos: seq[:pos] + "EP" + seq[pos + 2:],
        "replacement": "EP",
        "rationale": "Asp→Glu at DP: acidic isostere, blocks acid-labile clipping",
    },
    "Isomerization (DG)": {
        "pattern": re.compile(r"DG"),
        "fix": lambda seq, pos: seq[:pos] + "EG" + seq[pos + 2:],
        "replacement": "EG",
        "rationale": "Asp→Glu at DG: blocks Asp isomerization to isoAsp",
    },
    "Isomerization (DS)": {
        "pattern": re.compile(r"DS"),
        "fix": lambda seq, pos: seq[:pos] + "ES" + seq[pos + 2:],
        "replacement": "ES",
        "rationale": "Asp→Glu at DS: blocks Asp isomerization",
    },
}

# Charge engineering: residue substitutions to shift pI
# Acidify (lower pI): replace basic/neutral with acidic
ACIDIFY_SUBS = {
    "K": ("E", "Lys→Glu: +1 → -1 charge at pH 7, Delta charge = -2"),
    "R": ("E", "Arg→Glu: +1 → -1 charge, strongest pI reduction"),
    "Q": ("E", "Gln→Glu: neutral → -1 charge"),
    "N": ("D", "Asn→Asp: neutral → -1 charge"),
    "S": ("D", "Ser→Asp: neutral → -1 charge, minor structural impact"),
}

# Basify (raise pI): replace acidic/neutral with basic
BASIFY_SUBS = {
    "E": ("K", "Glu→Lys: -1 → +1 charge, Delta charge = +2"),
    "D": ("K", "Asp→Lys: -1 → +1 charge"),
    "Q": ("K", "Gln→Lys: neutral → +1 charge"),
    "N": ("K", "Asn→Lys: neutral → +1 charge"),
}

# Hydrophobicity reduction: replace exposed hydrophobics
HYDRO_REDUCTION_SUBS = {
    "V": ("T", "Val→Thr: reduces hydrophobicity, adds H-bond donor"),
    "I": ("T", "Ile→Thr: reduces hydrophobicity"),
    "L": ("S", "Leu→Ser: polar replacement"),
    "F": ("Y", "Phe→Tyr: adds hydroxyl, reduces hydrophobicity"),
}


# ===========================================================================
# 2. Liability-Targeted Mutagenesis Engine
# ===========================================================================

def identify_liabilities_for_mutagenesis(
    sequence: str,
    chain_name: str = "Chain",
    exclude_framework: bool = True,
) -> List[Dict[str, Any]]:
    """
    Scan a sequence for liability motifs eligible for mutagenesis.

    Parameters
    ----------
    sequence          : Amino acid sequence (single letter)
    chain_name        : Label for the chain (e.g., "HC", "LC")
    exclude_framework : If True, deprioritize mutations in framework
                        conserved regions (first 25 aa, last 10 aa of VR)

    Returns
    -------
    list of dicts: [{
        "position": int (0-indexed),
        "motif_type": str,
        "original_residues": str,
        "replacement": str,
        "rationale": str,
        "chain": str,
        "priority": "high" | "medium" | "low",
        "region": "CDR" | "Framework" | "Fc",
    }]
    """
    seq = sequence.upper()
    liabilities = []

    for motif_name, strategy in LIABILITY_FIX_STRATEGIES.items():
        pattern = strategy["pattern"]
        replacement = strategy["replacement"]
        rationale = strategy["rationale"]

        for match in pattern.finditer(seq):
            pos = match.start()
            original = match.group()

            # Determine region (heuristic)
            if pos < 120:
                # Variable region
                if 26 <= pos <= 38 or 50 <= pos <= 65 or 95 <= pos <= 115:
                    region = "CDR"
                    priority = "high"  # CDR liabilities are most critical
                else:
                    region = "Framework"
                    priority = "medium"
            elif pos < 230:
                region = "Hinge/CH1"
                priority = "medium"
            else:
                region = "Fc"
                priority = "low"  # Fc mutations are riskier (FcRn, effector function)

            liabilities.append({
                "position": pos,
                "motif_type": motif_name,
                "original_residues": original,
                "replacement": replacement,
                "rationale": rationale,
                "chain": chain_name,
                "priority": priority,
                "region": region,
            })

    # Sort by priority (high first, then by position)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    liabilities.sort(key=lambda x: (priority_order.get(x["priority"], 3), x["position"]))

    return liabilities


def apply_liability_mutations(
    sequence: str,
    liabilities: List[Dict[str, Any]],
    max_mutations: int = 8,
    skip_fc: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Apply liability-targeted mutations to a sequence.

    Applies mutations in reverse position order to avoid index shifting.
    Only targets high/medium priority liabilities.

    Parameters
    ----------
    sequence       : Original amino acid sequence
    liabilities    : Output from identify_liabilities_for_mutagenesis
    max_mutations  : Maximum number of mutations to apply
    skip_fc        : If True, skip Fc region mutations (preserve effector function)

    Returns
    -------
    tuple: (mutated_sequence, list_of_applied_mutations)
    """
    seq = sequence.upper()
    eligible = [
        l for l in liabilities
        if l["priority"] in ("high", "medium")
        and (not skip_fc or l["region"] != "Fc")
    ]

    # Limit to max_mutations
    to_apply = eligible[:max_mutations]

    # Apply in reverse position order to preserve indexing
    to_apply_sorted = sorted(to_apply, key=lambda x: x["position"], reverse=True)

    applied = []
    for mut in to_apply_sorted:
        pos = mut["position"]
        original = mut["original_residues"]
        replacement = mut["replacement"]

        # Verify the position still has the expected residues
        if seq[pos:pos + len(original)] == original:
            seq = seq[:pos] + replacement + seq[pos + len(original):]
            applied.append({
                "position": pos + 1,  # 1-indexed for display
                "chain": mut["chain"],
                "original": original,
                "replacement": replacement,
                "motif_type": mut["motif_type"],
                "rationale": mut["rationale"],
                "region": mut["region"],
                "notation": f"{original}{pos + 1}{replacement}",
            })

    # Reverse applied list so it's in position order
    applied.reverse()

    return seq, applied


# ===========================================================================
# 3. Charge Engineering for Bispecific Resolution
# ===========================================================================

def engineer_charge_shift(
    sequence: str,
    chain_name: str = "Chain_B",
    target_delta_pi: float = -0.5,
    max_mutations: int = 3,
    prefer_surface: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Systematically mutate residues to shift the chain pI.

    For bispecific homodimer resolution: if Rs < 1.0, shift Chain B pI
    to create wider separation from Chain A.

    Strategy:
    - Negative delta_pi (acidify): replace K/R/Q/N with E/D
    - Positive delta_pi (basify): replace E/D/Q/N with K
    - Prefer CDR-distal positions to preserve binding

    Parameters
    ----------
    sequence         : Chain amino acid sequence
    chain_name       : Chain label
    target_delta_pi  : Desired pI change (negative = more acidic)
    max_mutations    : Max number of mutations
    prefer_surface   : Prefer surface-exposed heuristic positions

    Returns
    -------
    tuple: (mutated_sequence, list_of_applied_mutations)
    """
    seq = sequence.upper()
    subs = ACIDIFY_SUBS if target_delta_pi < 0 else BASIFY_SUBS

    # Find candidate positions (prefer non-CDR, non-Fc positions)
    # Heuristic: surface-exposed = framework residues in loops (40-50, 65-80, 115-130 range)
    candidates = []
    surface_ranges = [(40, 50), (65, 80), (115, 130), (140, 170), (180, 220)]

    for pos, aa in enumerate(seq):
        if aa in subs:
            replacement, rationale = subs[aa]
            # Prioritize surface positions
            is_surface = any(start <= pos <= end for start, end in surface_ranges)
            # Avoid CDR core (26-38, 50-65, 95-115)
            is_cdr_core = 26 <= pos <= 38 or 50 <= pos <= 65 or 95 <= pos <= 115
            # Avoid Fc region critical residues (after pos 230 for full-length)
            is_fc = pos > 230 and len(seq) > 300

            if is_cdr_core:
                priority = 2  # Avoid CDR modifications
            elif is_fc:
                priority = 3  # Avoid Fc modifications
            elif is_surface and prefer_surface:
                priority = 0  # Best candidates
            else:
                priority = 1  # Acceptable

            candidates.append({
                "position": pos,
                "original": aa,
                "replacement": replacement,
                "rationale": rationale,
                "priority": priority,
                "chain": chain_name,
                "is_surface": is_surface,
            })

    # Sort by priority, then by position
    candidates.sort(key=lambda x: (x["priority"], x["position"]))

    # Apply top N mutations
    to_apply = candidates[:max_mutations]
    to_apply_sorted = sorted(to_apply, key=lambda x: x["position"], reverse=True)

    applied = []
    for mut in to_apply_sorted:
        pos = mut["position"]
        if seq[pos] == mut["original"]:
            seq = seq[:pos] + mut["replacement"] + seq[pos + 1:]
            applied.append({
                "position": pos + 1,  # 1-indexed
                "chain": mut["chain"],
                "original": mut["original"],
                "replacement": mut["replacement"],
                "rationale": mut["rationale"],
                "region": "Surface" if mut["is_surface"] else "Internal",
                "notation": f"{mut['original']}{pos + 1}{mut['replacement']}",
            })

    applied.reverse()  # Position order
    return seq, applied


# ===========================================================================
# 4. Hydrophobicity Reduction
# ===========================================================================

def reduce_hydrophobicity(
    sequence: str,
    chain_name: str = "Chain",
    max_mutations: int = 3,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Replace surface-exposed hydrophobic residues to reduce aggregation risk.

    Targets V/I/L/F in CDR-adjacent and loop regions.

    Parameters
    ----------
    sequence     : Amino acid sequence
    chain_name   : Chain label
    max_mutations: Maximum mutations

    Returns
    -------
    tuple: (mutated_sequence, applied_mutations)
    """
    seq = sequence.upper()
    candidates = []

    # Target hydrophobics in CDR-adjacent loops (heuristic)
    loop_ranges = [(26, 42), (50, 70), (93, 120)]

    for pos, aa in enumerate(seq):
        if aa in HYDRO_REDUCTION_SUBS:
            replacement, rationale = HYDRO_REDUCTION_SUBS[aa]
            in_loop = any(start <= pos <= end for start, end in loop_ranges)

            if in_loop:
                candidates.append({
                    "position": pos,
                    "original": aa,
                    "replacement": replacement,
                    "rationale": rationale,
                    "chain": chain_name,
                })

    # Limit to max mutations
    to_apply = candidates[:max_mutations]
    to_apply_sorted = sorted(to_apply, key=lambda x: x["position"], reverse=True)

    applied = []
    for mut in to_apply_sorted:
        pos = mut["position"]
        if seq[pos] == mut["original"]:
            seq = seq[:pos] + mut["replacement"] + seq[pos + 1:]
            applied.append({
                "position": pos + 1,
                "chain": mut["chain"],
                "original": mut["original"],
                "replacement": mut["replacement"],
                "rationale": mut["rationale"],
                "region": "CDR-adjacent",
                "notation": f"{mut['original']}{pos + 1}{mut['replacement']}",
            })

    applied.reverse()
    return seq, applied


# ===========================================================================
# 5. Variant Generator (Orchestrator)
# ===========================================================================

def generate_optimized_variants(
    chains: List[Dict[str, Any]],
    dev_score: float = 0.0,
    dev_grade: str = "Low",
    pk_half_life: float = 21.0,
    pk_risk: str = "Low",
    bispecific_rs: Optional[float] = None,
    bispecific_chain_b_idx: Optional[int] = None,
    n_variants: int = 3,
) -> Dict[str, Any]:
    """
    Generate N optimized sequence variants based on detected liabilities.

    Creates a spectrum of optimization strategies:
      - Variant 1 (Conservative): Fix only high-priority liabilities
      - Variant 2 (Moderate): Fix liabilities + charge engineering
      - Variant 3 (Aggressive): Fix liabilities + charge + hydrophobicity

    Parameters
    ----------
    chains               : List of chain dicts [{sequence, copy_number, name, chain_type}]
    dev_score            : Developability Score (0-1, higher = worse)
    dev_grade            : Developability grade ("Low", "Medium", "High")
    pk_half_life         : Predicted half-life (days)
    pk_risk              : PK risk assessment
    bispecific_rs        : Minimum resolution Rs (if bispecific)
    bispecific_chain_b_idx: Index of Chain B in chains list (for charge engineering)
    n_variants           : Number of variants to generate (default 3)

    Returns
    -------
    dict : {
        "status": "success" | "no_optimization_needed",
        "wild_type_chains": list,
        "variants": [{
            "name": str,
            "strategy": str,
            "chains": list (mutated),
            "mutations": list of all applied mutations,
            "mutation_count": int,
            "mutation_summary": str,
        }],
        "optimization_triggers": list of str,
        "summary": str,
    }
    """
    # Determine what needs optimization
    triggers = []
    needs_liability_fix = False
    needs_charge_eng = False
    needs_hydro_fix = False

    if dev_score > 0.35 or dev_grade in ("Medium", "High"):
        triggers.append(f"Developability Score {dev_score:.2f} ({dev_grade} risk)")
        needs_liability_fix = True
        if dev_score > 0.5:
            needs_hydro_fix = True

    if pk_half_life < 15.0 or pk_risk in ("High", "Very High"):
        triggers.append(f"PK half-life {pk_half_life:.1f} days ({pk_risk} risk)")
        needs_liability_fix = True

    if bispecific_rs is not None and bispecific_rs < 1.0:
        triggers.append(f"Bispecific resolution Rs={bispecific_rs:.3f} (<1.0)")
        needs_charge_eng = True

    # Scan all chains for liabilities
    all_chain_liabilities = {}
    for i, ch in enumerate(chains):
        seq = ch.get("sequence", "")
        name = ch.get("name", f"Chain{i + 1}")
        if len(seq) >= 10:
            liabs = identify_liabilities_for_mutagenesis(seq, chain_name=name)
            all_chain_liabilities[i] = liabs
            if len([l for l in liabs if l["priority"] in ("high", "medium")]) > 0:
                needs_liability_fix = True

    # If nothing needs optimization, check if there are any liabilities at all
    if not triggers:
        total_liabs = sum(len(l) for l in all_chain_liabilities.values())
        if total_liabs > 3:
            triggers.append(f"{total_liabs} liability motifs detected across chains")
            needs_liability_fix = True
        else:
            return {
                "status": "no_optimization_needed",
                "wild_type_chains": chains,
                "variants": [],
                "optimization_triggers": [],
                "summary": "No significant liabilities or bottlenecks detected. Wild-type sequence is acceptable.",
            }

    # Generate variants
    variants = []

    # -- Variant 1: Conservative (liability fixes only) -----------------------
    v1_chains = []
    v1_mutations = []
    for i, ch in enumerate(chains):
        seq = ch.get("sequence", "").upper()
        name = ch.get("name", f"Chain{i + 1}")
        liabs = all_chain_liabilities.get(i, [])

        if liabs and needs_liability_fix:
            mutated, applied = apply_liability_mutations(seq, liabs, max_mutations=4, skip_fc=True)
            v1_chains.append({**ch, "sequence": mutated})
            v1_mutations.extend(applied)
        else:
            v1_chains.append({**ch})

    variants.append({
        "name": "Variant 1 (Conservative)",
        "strategy": "Liability-targeted mutagenesis only — fix high-priority oxidation and deamidation hotspots",
        "chains": v1_chains,
        "mutations": v1_mutations,
        "mutation_count": len(v1_mutations),
        "mutation_summary": "; ".join(
            f"{m['chain']}: {m['notation']}" for m in v1_mutations
        ) if v1_mutations else "No mutations applied",
    })

    # -- Variant 2: Moderate (liabilities + charge engineering) ----------------
    v2_chains = []
    v2_mutations = []
    for i, ch in enumerate(chains):
        seq = ch.get("sequence", "").upper()
        name = ch.get("name", f"Chain{i + 1}")
        liabs = all_chain_liabilities.get(i, [])

        # First apply liability fixes
        if liabs and needs_liability_fix:
            seq, applied = apply_liability_mutations(seq, liabs, max_mutations=5, skip_fc=True)
            v2_mutations.extend(applied)

        # Then apply charge engineering if needed
        if needs_charge_eng and bispecific_chain_b_idx is not None and i == bispecific_chain_b_idx:
            seq, charge_applied = engineer_charge_shift(
                seq, chain_name=name, target_delta_pi=-0.5, max_mutations=2,
            )
            v2_mutations.extend(charge_applied)
        elif pk_half_life < 15.0 and ch.get("chain_type", "").upper().replace(" ", "") in ("HC", "HEAVY"):
            # Acidify HC slightly if pI is too high (reduce half-life penalty)
            seq, charge_applied = engineer_charge_shift(
                seq, chain_name=name, target_delta_pi=-0.3, max_mutations=1,
            )
            v2_mutations.extend(charge_applied)

        v2_chains.append({**ch, "sequence": seq})

    variants.append({
        "name": "Variant 2 (Moderate)",
        "strategy": "Liability fixes + charge engineering for pI optimization / resolution improvement",
        "chains": v2_chains,
        "mutations": v2_mutations,
        "mutation_count": len(v2_mutations),
        "mutation_summary": "; ".join(
            f"{m['chain']}: {m['notation']}" for m in v2_mutations
        ) if v2_mutations else "No mutations applied",
    })

    # -- Variant 3: Aggressive (all optimizations) ----------------------------
    v3_chains = []
    v3_mutations = []
    for i, ch in enumerate(chains):
        seq = ch.get("sequence", "").upper()
        name = ch.get("name", f"Chain{i + 1}")
        liabs = all_chain_liabilities.get(i, [])

        # Liability fixes (more aggressive)
        if liabs:
            seq, applied = apply_liability_mutations(seq, liabs, max_mutations=8, skip_fc=False)
            v3_mutations.extend(applied)

        # Charge engineering
        if needs_charge_eng and bispecific_chain_b_idx is not None and i == bispecific_chain_b_idx:
            seq, charge_applied = engineer_charge_shift(
                seq, chain_name=name, target_delta_pi=-0.8, max_mutations=3,
            )
            v3_mutations.extend(charge_applied)
        elif ch.get("chain_type", "").upper().replace(" ", "") in ("HC", "HEAVY"):
            seq, charge_applied = engineer_charge_shift(
                seq, chain_name=name, target_delta_pi=-0.3, max_mutations=2,
            )
            v3_mutations.extend(charge_applied)

        # Hydrophobicity reduction
        if needs_hydro_fix or dev_score > 0.4:
            seq, hydro_applied = reduce_hydrophobicity(seq, chain_name=name, max_mutations=3)
            v3_mutations.extend(hydro_applied)

        v3_chains.append({**ch, "sequence": seq})

    variants.append({
        "name": "Variant 3 (Aggressive)",
        "strategy": "Full optimization — liabilities + charge + hydrophobicity reduction",
        "chains": v3_chains,
        "mutations": v3_mutations,
        "mutation_count": len(v3_mutations),
        "mutation_summary": "; ".join(
            f"{m['chain']}: {m['notation']}" for m in v3_mutations
        ) if v3_mutations else "No mutations applied",
    })

    # Trim to requested count
    variants = variants[:n_variants]

    summary = (
        f"Generated {len(variants)} optimized variants from {len(triggers)} trigger(s): "
        + "; ".join(triggers)
    )

    return {
        "status": "success",
        "wild_type_chains": chains,
        "variants": variants,
        "optimization_triggers": triggers,
        "summary": summary,
    }


def variants_to_fasta(
    variants: List[Dict[str, Any]],
    wt_chains: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Convert variant chains to multi-FASTA format for download.

    Parameters
    ----------
    variants   : List of variant dicts from generate_optimized_variants
    wt_chains  : Optional wild-type chains to include first

    Returns
    -------
    str : Multi-FASTA formatted string
    """
    lines = []

    # Wild-type first
    if wt_chains:
        for ch in wt_chains:
            name = ch.get("name", "Chain")
            ctype = ch.get("chain_type", "")
            copies = ch.get("copy_number", 1)
            seq = ch.get("sequence", "")
            lines.append(f">WT_{name}|{ctype}|copies={copies}|len={len(seq)}")
            # Wrap sequence at 60 chars
            for i in range(0, len(seq), 60):
                lines.append(seq[i:i + 60])

    # Variants
    for var in variants:
        var_name = var.get("name", "Variant").replace(" ", "_")
        muts = var.get("mutation_summary", "")
        for ch in var.get("chains", []):
            name = ch.get("name", "Chain")
            ctype = ch.get("chain_type", "")
            copies = ch.get("copy_number", 1)
            seq = ch.get("sequence", "")
            lines.append(f">{var_name}_{name}|{ctype}|copies={copies}|len={len(seq)}|mutations={muts[:80]}")
            for i in range(0, len(seq), 60):
                lines.append(seq[i:i + 60])

    return "\n".join(lines)


# ===========================================================================
# 6. Multi-Objective Pareto-Optimal Filtering (M15)
# ===========================================================================

def _dominates(a: Dict[str, float], b: Dict[str, float], objectives: List[Dict[str, Any]]) -> bool:
    """
    Check if solution 'a' Pareto-dominates solution 'b'.

    A dominates B if A is at least as good on ALL objectives and
    strictly better on at least one.

    Parameters
    ----------
    a, b : dicts with objective values
    objectives : list of {"name": str, "minimize": bool}
    """
    at_least_as_good = True
    strictly_better = False

    for obj in objectives:
        name = obj["name"]
        minimize = obj.get("minimize", True)
        va = a.get(name, 0.0)
        vb = b.get(name, 0.0)

        if minimize:
            if va > vb:
                at_least_as_good = False
            if va < vb:
                strictly_better = True
        else:
            if va < vb:
                at_least_as_good = False
            if va > vb:
                strictly_better = True

    return at_least_as_good and strictly_better


def pareto_filter_variants(
    variants: List[Dict[str, Any]],
    objectives: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Filter variants to return only Pareto-optimal solutions.

    Multi-objective criteria:
      - MINIMIZE: predicted aggregation %, liability density, mutation count
      - MAXIMIZE: predicted Tm (°C), half-life (days)

    Non-dominated variants are kept; dominated ones are removed.

    Parameters
    ----------
    variants : list of variant dicts, each must have a "wetlab_predictions" or
               "evaluation" sub-dict with objective values
    objectives : list of {"name": str, "minimize": bool}. Defaults to the
                 standard set if not provided.

    Returns
    -------
    list of Pareto-optimal variants (subset of input)
    """
    if not variants:
        return []

    if objectives is None:
        objectives = [
            {"name": "pred_aggregation_pct", "minimize": True},
            {"name": "pred_tm", "minimize": False},
            {"name": "liability_density", "minimize": True},
            {"name": "pk_half_life", "minimize": False},
            {"name": "mutation_count", "minimize": True},
        ]

    # Extract objective values from variant evaluation data
    def _get_obj_values(var: Dict[str, Any]) -> Dict[str, float]:
        vals = {}
        # Sources of objective values: evaluation, wetlab_predictions, top-level
        eval_data = var.get("evaluation", {})
        wetlab = var.get("wetlab_predictions", {})

        vals["pred_aggregation_pct"] = wetlab.get(
            "Exp_Aggregation_Percent",
            eval_data.get("pred_aggregation_pct", 5.0)
        )
        vals["pred_tm"] = wetlab.get(
            "Exp_Tm_MeltingTemp",
            eval_data.get("pred_tm", 70.0)
        )
        vals["liability_density"] = eval_data.get(
            "liability_density",
            var.get("liability_density", 30.0)
        )
        vals["pk_half_life"] = eval_data.get(
            "pk_half_life",
            var.get("pk_half_life", 21.0)
        )
        vals["mutation_count"] = var.get("mutation_count", 0)

        return vals

    indexed_variants = [
        {"variant": v, "obj_vals": _get_obj_values(v)}
        for v in variants
    ]

    # Find non-dominated set
    pareto = []
    for i, vi in enumerate(indexed_variants):
        dominated = False
        for j, vj in enumerate(indexed_variants):
            if i != j and _dominates(vj["obj_vals"], vi["obj_vals"], objectives):
                dominated = True
                break
        if not dominated:
            var = vi["variant"].copy()
            var["pareto_optimal"] = True
            var["pareto_obj_values"] = vi["obj_vals"]
            pareto.append(var)

    log.info("Pareto filter: %d/%d variants are non-dominated",
             len(pareto), len(variants))

    return pareto


def evaluate_and_filter_variants(
    variants: List[Dict[str, Any]],
    wild_type_chains: List[Dict[str, Any]],
    glycoform_profile: str = "standard_cho",
) -> Dict[str, Any]:
    """
    Evaluate all variants through the wet-lab model and filter by Pareto optimality.

    For each variant:
      1. Build combined sequence from mutated chains
      2. Extract biophysical features
      3. Run through trained WetLabPredictor (if available)
      4. If Agg% spikes or Tm drops below threshold → mark as rejected
      5. Apply Pareto filtering to keep only non-dominated variants

    Parameters
    ----------
    variants : list of variant dicts from generate_optimized_variants
    wild_type_chains : original WT chains for comparison
    glycoform_profile : active glycoform profile

    Returns
    -------
    dict : {
        "status": "success",
        "pareto_variants": list (filtered),
        "rejected_variants": list (dominated or failed safety),
        "all_evaluated": list (all with predictions),
        "wetlab_model_available": bool,
        "wt_predictions": dict (WT wet-lab predictions),
    }
    """
    import re as _re

    # Try to load the wet-lab model
    try:
        from src.ml_predictor import get_wetlab_model, extract_features
        wetlab_model = get_wetlab_model()
    except ImportError:
        wetlab_model = None

    # Try to load PK predictor
    try:
        from src.preclinical_twin import predict_human_half_life, check_fcrn_binding_motif
        from src.analytical_twin import calculate_liability_density, build_super_sequence
        has_pk = True
    except ImportError:
        has_pk = False

    # Helper to extract features from chain list
    def _extract_from_chains(chain_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        seqs = [ch.get("sequence", "") for ch in chain_list
                for _ in range(max(1, ch.get("copy_number", 1)))]
        combined = "".join(seqs).upper()
        combined = _re.sub(r'[^A-Z]', '', combined)
        if len(combined) < 50:
            return None

        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            pa = ProteinAnalysis(combined)
            pI = pa.isoelectric_point()
            mw_kda = pa.molecular_weight() / 1000.0
            gravy = pa.gravy()
        except Exception:
            pI, mw_kda, gravy = 8.0, 150.0, -0.3

        hydrophobicity = max(0.0, min(1.0, (gravy + 2.0) / 4.0))
        deam = len(_re.findall(r"N[GS]", combined))
        ox = combined.count("M") + combined.count("W")
        acidic = combined.count("D") + combined.count("E")
        basic = combined.count("K") + combined.count("R") + combined.count("H")

        feat_vec = extract_features(pI, mw_kda, deam, ox, acidic, basic, hydrophobicity)

        # PK
        pk_hl = 21.0
        liab_dens = 30.0
        if has_pk:
            try:
                ld = calculate_liability_density(chain_list)
                liab_dens = ld.get("density_per_1000", 30.0)
                fcrn = check_fcrn_binding_motif(combined)
                pk = predict_human_half_life(
                    global_pi=pI, hydrophobicity=hydrophobicity,
                    liability_density=liab_dens,
                    fcrn_binding_motif_intact=fcrn.get("intact", True),
                    mw_kda=mw_kda, glycoform_profile=glycoform_profile,
                )
                pk_hl = pk.get("half_life_days", 21.0)
            except Exception:
                pass

        return {
            "pI": round(pI, 2),
            "mw_kda": round(mw_kda, 1),
            "hydrophobicity": round(hydrophobicity, 3),
            "feature_vector": feat_vec,
            "liability_density": round(liab_dens, 1),
            "pk_half_life": round(pk_hl, 1),
            "combined_sequence": combined,
        }

    # Evaluate WT
    wt_eval = _extract_from_chains(wild_type_chains)
    wt_wetlab = {}
    if wt_eval and wetlab_model and wetlab_model.trained:
        wt_pred = wetlab_model.predict_single(wt_eval["feature_vector"])
        wt_wetlab = wt_pred
        wt_eval["wetlab_predictions"] = wt_pred

    # Evaluate each variant
    all_evaluated = []
    rejected = []

    for var in variants:
        var_chains = var.get("chains", [])
        if not var_chains:
            continue

        var_eval = _extract_from_chains(var_chains)
        if var_eval is None:
            rejected.append({**var, "rejection_reason": "Could not extract features"})
            continue

        var_enriched = {**var}
        var_enriched["evaluation"] = var_eval

        # Run wet-lab model if available
        if wetlab_model and wetlab_model.trained:
            wetlab_pred = wetlab_model.predict_single(var_eval["feature_vector"])
            var_enriched["wetlab_predictions"] = wetlab_pred

            # Safety check: reject if aggregation spikes badly
            wt_agg = wt_wetlab.get("Exp_Aggregation_Percent", 5.0)
            var_agg = wetlab_pred.get("Exp_Aggregation_Percent", 5.0)
            wt_tm = wt_wetlab.get("Exp_Tm_MeltingTemp", 70.0)
            var_tm = wetlab_pred.get("Exp_Tm_MeltingTemp", 70.0)

            # Rejection criteria: Agg% increases by >50% relative OR Tm drops >5°C
            if var_agg > wt_agg * 1.5 and var_agg > 10.0:
                var_enriched["rejected"] = True
                var_enriched["rejection_reason"] = (
                    f"Predicted Agg% spike: {var_agg:.1f}% vs WT {wt_agg:.1f}%"
                )
                rejected.append(var_enriched)
                continue

            if var_tm < wt_tm - 5.0 and var_tm < 60.0:
                var_enriched["rejected"] = True
                var_enriched["rejection_reason"] = (
                    f"Predicted Tm drop: {var_tm:.1f}°C vs WT {wt_tm:.1f}°C"
                )
                rejected.append(var_enriched)
                continue

            # Enrich with deltas
            var_enriched["delta_agg_pct"] = round(var_agg - wt_agg, 2)
            var_enriched["delta_tm"] = round(var_tm - wt_tm, 2)

        var_enriched["liability_density"] = var_eval["liability_density"]
        var_enriched["pk_half_life"] = var_eval["pk_half_life"]

        all_evaluated.append(var_enriched)

    # Pareto filter
    if all_evaluated:
        pareto = pareto_filter_variants(all_evaluated)
    else:
        pareto = []

    return {
        "status": "success",
        "pareto_variants": pareto,
        "rejected_variants": rejected,
        "all_evaluated": all_evaluated,
        "wetlab_model_available": wetlab_model is not None and wetlab_model.trained,
        "wt_predictions": wt_wetlab,
        "wt_evaluation": wt_eval,
    }


# ===========================================================================
# __main__: Test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  Generative Engineer v1.0 — In-Silico Mutagenesis Test")
    print("=" * 60)

    # Test sequence with known liabilities
    test_hc = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTY"
        "YADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGYDYWGQGTLVTVSSAST"
        "KGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGL"
        "YSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGP"
        "SVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYN"
        "STYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDE"
        "LTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRW"
        "QQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
    )

    print("\n--- Test 1: Identify Liabilities ---")
    liabs = identify_liabilities_for_mutagenesis(test_hc, chain_name="HC")
    print(f"  Found {len(liabs)} liability motifs")
    for l in liabs[:5]:
        print(f"    [{l['priority']}] pos {l['position']+1}: {l['motif_type']} "
              f"({l['original_residues']}→{l['replacement']}) in {l['region']}")

    print("\n--- Test 2: Apply Liability Mutations ---")
    mutated, applied = apply_liability_mutations(test_hc, liabs, max_mutations=4)
    print(f"  Applied {len(applied)} mutations")
    for m in applied:
        print(f"    {m['chain']} {m['notation']}: {m['rationale'][:50]}...")
    assert mutated != test_hc, "Sequence should be different"
    assert len(mutated) == len(test_hc) or abs(len(mutated) - len(test_hc)) <= 4

    print("\n--- Test 3: Charge Engineering ---")
    eng_seq, eng_muts = engineer_charge_shift(test_hc, chain_name="HC", target_delta_pi=-0.5, max_mutations=3)
    print(f"  Applied {len(eng_muts)} charge mutations")
    for m in eng_muts:
        print(f"    {m['notation']}: {m['rationale'][:50]}...")

    print("\n--- Test 4: Hydrophobicity Reduction ---")
    hydro_seq, hydro_muts = reduce_hydrophobicity(test_hc, chain_name="HC", max_mutations=3)
    print(f"  Applied {len(hydro_muts)} hydrophobicity mutations")

    print("\n--- Test 5: Full Variant Generation ---")
    test_lc = (
        "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKLLIYAASSLQSGVP"
        "SRFSGSGSGTDFTLTISSLQPEDFATYYCLQHNSYPLTFGQGTRLEIKRTVAAPSVFIFP"
        "PSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTL"
        "TLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    )
    chains = [
        {"sequence": test_hc, "copy_number": 2, "name": "HC", "chain_type": "HC"},
        {"sequence": test_lc, "copy_number": 2, "name": "LC", "chain_type": "LC"},
    ]
    result = generate_optimized_variants(
        chains, dev_score=0.55, dev_grade="Medium",
        pk_half_life=14.0, pk_risk="Medium",
    )
    print(f"  Status: {result['status']}")
    print(f"  Triggers: {result['optimization_triggers']}")
    for var in result["variants"]:
        print(f"  {var['name']}: {var['mutation_count']} mutations")
        print(f"    Strategy: {var['strategy'][:60]}...")
        print(f"    Mutations: {var['mutation_summary'][:80]}...")

    print("\n--- Test 6: FASTA Export ---")
    fasta = variants_to_fasta(result["variants"], wt_chains=chains)
    n_headers = fasta.count(">")
    print(f"  FASTA: {n_headers} sequence headers, {len(fasta)} chars")

    print("\nGenerative Engineer v1.0 test complete")
