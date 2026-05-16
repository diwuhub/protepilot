"""Core structure analyzer.

Given a `StructureInput` (or a PDB path + chain IDs), compute a
`StructureMetrics` report with:
    - per-residue SASA (Shrake-Rupley via biopython)
    - paratope identification (CDR residues with exposed SASA)
    - VH-VL interface BSA (by SASA-subtraction)
    - aggregation-prone spatial patches
    - structure-based TAP metric upgrades

No MD, no ΔΔG. CDR boundaries come from the same regex-based heuristic
used by the sequence baseline.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

from src.structure.schema import (
    AggregationPatch,
    InterfaceAnalysis,
    ParatopeProfile,
    StructureInput,
    StructureMetrics,
    StructureSource,
)


# Three-letter → one-letter AA translation.
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Kyte-Doolittle hydrophobicity scale (used in ab_benchmark; duplicated
# here to keep structure module self-contained).
_KYTE_DOOLITTLE = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# SASA exposure cutoff (Å²) to call a residue "exposed".
# Chothia + Janin classical cutoffs put ~7 Å² for buried, ~20 Å² for exposed.
DEFAULT_EXPOSURE_CUTOFF = 7.5


def _ensure_ab_benchmark_on_path() -> None:
    """Lazy sys.path shim for ab_benchmark.seqprops (same pattern as mutation/)."""
    _AB = os.environ.get(
        "AB_BENCHMARK_PATH",
        str(Path(__file__).resolve().parents[3] / "ab-benchmark"),
    )
    if os.path.isdir(_AB) and _AB not in sys.path:
        sys.path.append(_AB)


# ---------------------------------------------------------------------------


def _parse_chain_sequence(structure, chain_id: str) -> tuple[list[str], list[int]]:
    """Extract one-letter sequence + residue numbers for a chain."""
    seq: list[str] = []
    resnums: list[int] = []
    for model in structure:
        if chain_id not in [c.id for c in model]:
            continue
        chain = model[chain_id]
        for residue in chain:
            # Skip non-standard residues (HETATM, water, etc.)
            hetflag, resseq, icode = residue.id
            if hetflag != " ":
                continue
            three = residue.get_resname()
            one = _THREE_TO_ONE.get(three)
            if one is None:
                continue
            seq.append(one)
            resnums.append(resseq)
        break  # first model only
    return seq, resnums


def _compute_sasa(structure, chain_ids: list[str]) -> dict[tuple[str, int], float]:
    """Return {(chain_id, resnum): sasa_A2} for the given chains."""
    from Bio.PDB.SASA import ShrakeRupley

    sr = ShrakeRupley(probe_radius=1.4, n_points=100)
    sr.compute(structure, level="R")

    out: dict[tuple[str, int], float] = {}
    for model in structure:
        for chain in model:
            if chain.id not in chain_ids:
                continue
            for residue in chain:
                hetflag, resseq, icode = residue.id
                if hetflag != " ":
                    continue
                sasa = getattr(residue, "sasa", None)
                if sasa is None:
                    continue
                out[(chain.id, resseq)] = float(sasa)
        break  # first model only
    return out


def _locate_cdrs(vh: str, vl: str = "") -> dict[str, tuple[int, int]]:
    """Return {cdr_label: (start, end)} — inclusive-exclusive chain-local indices.

    Uses ab_benchmark.seqprops.extract_cdrs when available; otherwise
    returns an empty dict (every position becomes framework).
    """
    _ensure_ab_benchmark_on_path()
    try:
        from ab_benchmark.seqprops import extract_cdrs  # type: ignore
    except ImportError:
        return {}
    cdrs = extract_cdrs(vh, vl)
    out: dict[str, tuple[int, int]] = {}
    for key, label in {"h1": "H:cdr_h1", "h2": "H:cdr_h2", "h3": "H:cdr_h3", "l3": "L:cdr_l3"}.items():
        if key not in cdrs:
            continue
        chain_seq = vh if key.startswith("h") else vl
        idx = chain_seq.find(cdrs[key])
        if idx < 0:
            continue
        out[label] = (idx, idx + len(cdrs[key]))
    return out


# ---------------------------------------------------------------------------


def _compute_paratope(
    vh_seq: list[str],
    vl_seq: list[str],
    vh_sasa: list[float],
    vl_sasa: list[float],
    cdr_ranges: dict[str, tuple[int, int]],
    exposure_cutoff: float,
) -> ParatopeProfile:
    cdr_positions: list[int] = []
    chains: list[str] = []

    # VH chain contributions.
    for label, (s, e) in cdr_ranges.items():
        chain = label.split(":")[0]
        chain_seq = vh_seq if chain == "H" else vl_seq
        chain_sasa = vh_sasa if chain == "H" else vl_sasa
        for i in range(s, min(e, len(chain_seq), len(chain_sasa))):
            if chain_sasa[i] >= exposure_cutoff:
                # Use joint index: VH positions are [0, len(vh_seq));
                # VL positions start at len(vh_seq).
                joint_idx = i if chain == "H" else len(vh_seq) + i
                cdr_positions.append(joint_idx)
                chains.append(chain)

    total_sasa = sum(
        (vh_sasa[i] if chain == "H" else vl_sasa[i - len(vh_seq)])
        for i, chain in zip(cdr_positions, chains)
    )

    # Hydrophobicity and charge fraction.
    pos_cnt = neg_cnt = neu_cnt = 0
    hydro_cnt = 0
    for i, chain in zip(cdr_positions, chains):
        seq = vh_seq if chain == "H" else vl_seq
        local_idx = i if chain == "H" else i - len(vh_seq)
        if local_idx >= len(seq):
            continue
        aa = seq[local_idx]
        if aa in {"K", "R"}:
            pos_cnt += 1
        elif aa in {"D", "E"}:
            neg_cnt += 1
        else:
            neu_cnt += 1
        if _KYTE_DOOLITTLE.get(aa, 0.0) > 0:
            hydro_cnt += 1

    n = len(cdr_positions)
    hydro_frac = hydro_cnt / n if n > 0 else 0.0
    # Flatness proxy: simple variance of SASA values — flatter paratope
    # has lower variance. Bounded to [0, 1] after transformation.
    if n > 1:
        sasa_vals = [
            (vh_sasa[i] if chain == "H" else vl_sasa[i - len(vh_seq)])
            for i, chain in zip(cdr_positions, chains)
        ]
        mean_sasa = sum(sasa_vals) / len(sasa_vals)
        variance = sum((s - mean_sasa) ** 2 for s in sasa_vals) / len(sasa_vals)
        flatness = 1.0 / (1.0 + variance / 100.0)
    else:
        flatness = 1.0

    return ParatopeProfile(
        cdr_residues=cdr_positions,
        chain_of_residue=chains,
        total_surface_area_a2=float(total_sasa),
        hydrophobic_fraction=float(hydro_frac),
        charge_breakdown={"positive": pos_cnt, "negative": neg_cnt, "neutral": neu_cnt},
        flatness_proxy=float(flatness),
    )


def _compute_interface(
    structure,
    vh_chain_id: str,
    vl_chain_id: str,
    complex_sasa: dict[tuple[str, int], float],
    exposure_cutoff: float,
) -> InterfaceAnalysis | None:
    """VH-VL interface BSA by chain-alone vs complex SASA subtraction."""
    from Bio.PDB import PDBParser, Select
    from Bio.PDB.SASA import ShrakeRupley

    # Method: compute SASA of each chain in isolation, subtract from complex.
    # Interface BSA = 0.5 × (vh_alone_sasa + vl_alone_sasa - vh_complex_sasa - vl_complex_sasa).

    def _chain_only_sasa(target_chain_id: str) -> dict[tuple[str, int], float]:
        # Create a detached copy with only the target chain.
        from copy import deepcopy
        s = deepcopy(structure)
        for model in s:
            for chain in list(model.get_chains()):
                if chain.id != target_chain_id:
                    model.detach_child(chain.id)
        sr = ShrakeRupley(probe_radius=1.4, n_points=100)
        sr.compute(s, level="R")
        out: dict[tuple[str, int], float] = {}
        for model in s:
            for chain in model:
                for res in chain:
                    if res.id[0] != " ":
                        continue
                    sa = getattr(res, "sasa", None)
                    if sa is None:
                        continue
                    out[(chain.id, res.id[1])] = float(sa)
            break
        return out

    vh_alone = _chain_only_sasa(vh_chain_id)
    vl_alone = _chain_only_sasa(vl_chain_id)

    if not vh_alone or not vl_alone:
        return None

    vh_complex_total = sum(v for (c, _), v in complex_sasa.items() if c == vh_chain_id)
    vl_complex_total = sum(v for (c, _), v in complex_sasa.items() if c == vl_chain_id)
    vh_alone_total = sum(vh_alone.values())
    vl_alone_total = sum(vl_alone.values())

    bsa = 0.5 * ((vh_alone_total - vh_complex_total) + (vl_alone_total - vl_complex_total))

    # Residues at interface: those whose SASA dropped by ≥ cutoff.
    vh_iface: list[int] = []
    for (chain_id, resnum), alone_sa in vh_alone.items():
        complex_sa = complex_sasa.get((chain_id, resnum), alone_sa)
        if alone_sa - complex_sa >= exposure_cutoff:
            vh_iface.append(resnum)
    vl_iface: list[int] = []
    for (chain_id, resnum), alone_sa in vl_alone.items():
        complex_sa = complex_sasa.get((chain_id, resnum), alone_sa)
        if alone_sa - complex_sa >= exposure_cutoff:
            vl_iface.append(resnum)

    # Packing density: heuristic from BSA and residue counts at interface.
    n_iface = len(vh_iface) + len(vl_iface)
    packing = min(1.0, bsa / (n_iface * 50.0)) if n_iface > 0 else 0.0

    # H-bond count left unfilled here; BioPython doesn't ship a trivial
    # H-bond detector. Tag as 0 with a note so readers know why.
    return InterfaceAnalysis(
        buried_surface_area_a2=float(bsa),
        packing_density=float(packing),
        n_hbonds_across_interface=0,
        vh_residues_at_interface=sorted(vh_iface),
        vl_residues_at_interface=sorted(vl_iface),
        notes="H-bonds: not computed (Phase 3 heuristic; upgrade with DSSP later)",
    )


def _detect_aggregation_patches(
    vh_seq: list[str],
    vl_seq: list[str],
    vh_sasa: list[float],
    vl_sasa: list[float],
    structure,
    vh_chain_id: str,
    vl_chain_id: str,
    exposure_cutoff: float = 20.0,
    spatial_cutoff_a: float = 8.0,
    min_patch_size: int = 3,
) -> list[AggregationPatch]:
    """Spatial clusters of exposed hydrophobic residues (SAP-style)."""
    # Collect candidate residues (exposed + hydrophobic).
    candidates: list[tuple[str, int, int, str, float]] = []
    # (chain_id, chain_resnum_in_pdb, joint_index, aa, sasa)

    for i, (aa, sasa) in enumerate(zip(vh_seq, vh_sasa)):
        if sasa >= exposure_cutoff and _KYTE_DOOLITTLE.get(aa, 0) > 0:
            candidates.append((vh_chain_id, i + 1, i, aa, sasa))  # PDB resnum ~1-indexed
    for i, (aa, sasa) in enumerate(zip(vl_seq, vl_sasa)):
        if sasa >= exposure_cutoff and _KYTE_DOOLITTLE.get(aa, 0) > 0:
            candidates.append((vl_chain_id, i + 1, len(vh_seq) + i, aa, sasa))

    if len(candidates) < min_patch_size:
        return []

    # Build Cα coordinates lookup per (chain, resnum).
    ca_coords: dict[tuple[str, int], tuple[float, float, float]] = {}
    for model in structure:
        for chain in model:
            if chain.id not in {vh_chain_id, vl_chain_id}:
                continue
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    x, y, z = residue["CA"].coord
                    ca_coords[(chain.id, residue.id[1])] = (float(x), float(y), float(z))
        break

    # Build adjacency by spatial distance.
    n = len(candidates)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        ci, ri, _, _, _ = candidates[i]
        if (ci, ri) not in ca_coords:
            continue
        xi, yi, zi = ca_coords[(ci, ri)]
        for j in range(i + 1, n):
            cj, rj, _, _, _ = candidates[j]
            if (cj, rj) not in ca_coords:
                continue
            xj, yj, zj = ca_coords[(cj, rj)]
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
            if d <= spatial_cutoff_a:
                union(i, j)

    # Group by cluster root.
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    patches: list[AggregationPatch] = []
    for root, members in clusters.items():
        if len(members) < min_patch_size:
            continue
        member_joint = [candidates[m][2] for m in members]
        member_chains = [candidates[m][0] for m in members]
        # Center = member with max SASA.
        center_idx = max(members, key=lambda m: candidates[m][4])
        total_sasa = sum(candidates[m][4] for m in members)
        # Patch score: bounded product of size, total SASA, mean hydrophobicity.
        hydro_mean = sum(_KYTE_DOOLITTLE.get(candidates[m][3], 0) for m in members) / len(members)
        raw = (len(members) / 10.0) * (total_sasa / 500.0) * (hydro_mean / 4.5)
        score = float(max(0.0, min(1.0, raw)))
        patches.append(
            AggregationPatch(
                center_residue=candidates[center_idx][2],
                member_residues=sorted(member_joint),
                total_hydrophobic_sasa_a2=float(total_sasa),
                patch_score=score,
                chain_of_residues=member_chains,
            )
        )
    return patches


# ---------------------------------------------------------------------------
# Structure-based TAP and DI
# ---------------------------------------------------------------------------


def _compute_structural_tap_di(
    vh_seq: list[str],
    vl_seq: list[str],
    vh_sasa: list[float],
    vl_sasa: list[float],
    patches: list[AggregationPatch],
    cdr_ranges: dict[str, tuple[int, int]],
    exposure_cutoff: float,
) -> dict[str, Any]:
    """Upgrade Phase 0 sequence-proxy TAP/DI to structure-based versions.

    Structural TAP guidelines (Raybould 2019):
      PSH — Patches of Surface Hydrophobicity (count of spatial hydrophobic patches)
      PPC — Patches of Positive Charge (exposed K+R clusters)
      PNC — Patches of Negative Charge (exposed D+E clusters)
      SFvCSP — net charge asymmetry between VH and VL (surface-weighted)
      CDR length — unchanged from sequence version

    Structural DI (Lauer 2012):
      DI = -SFvCSP + β·SAP
      SAP (Spatial Aggregation Propensity) ≈ total_hydrophobic_SASA of patches.
    """
    # --- PSH via counted patches ---
    psh = float(len(patches))

    # --- PPC / PNC via exposed charged residues ---
    def _count_exposed_charged(seq, sasa, charges):
        n = 0
        for aa, sa in zip(seq, sasa):
            if aa in charges and sa >= exposure_cutoff:
                n += 1
        return n

    vh_pos = _count_exposed_charged(vh_seq, vh_sasa, {"K", "R"})
    vl_pos = _count_exposed_charged(vl_seq, vl_sasa, {"K", "R"})
    vh_neg = _count_exposed_charged(vh_seq, vh_sasa, {"D", "E"})
    vl_neg = _count_exposed_charged(vl_seq, vl_sasa, {"D", "E"})
    ppc = float(vh_pos + vl_pos)
    pnc = float(vh_neg + vl_neg)

    # --- SFvCSP (surface-weighted net charges) ---
    def _sasa_weighted_charge(seq, sasa):
        q = 0.0
        for aa, sa in zip(seq, sasa):
            if aa in {"K", "R"}:
                q += sa
            elif aa in {"D", "E"}:
                q -= sa
        return q

    vh_q = _sasa_weighted_charge(vh_seq, vh_sasa)
    vl_q = _sasa_weighted_charge(vl_seq, vl_sasa)
    sfvcsp = (vh_q * vl_q) / 10000.0  # scale to sane range

    # --- CDR total length ---
    total_cdr = sum(max(0, e - s) for s, e in cdr_ranges.values())

    # --- Risk flags (Raybould 2019 amber bands, adapted to structure) ---
    flags = 0
    if total_cdr >= 65:          # very long cumulative CDR
        flags += 1
    if psh >= 3:                 # three+ hydrophobic patches
        flags += 1
    if ppc >= 12 or pnc >= 12:   # charge-patch overload
        flags += 1
    if abs(sfvcsp) >= 4:         # heavy asymmetry
        flags += 1

    # --- DI = -SFvCSP + β·SAP ---
    sap = sum(p.total_hydrophobic_sasa_a2 for p in patches) / 100.0
    beta = 0.815  # Lauer 2012
    di_full = -sfvcsp + beta * sap

    return {
        "tap_psh": psh,
        "tap_ppc": ppc,
        "tap_pnc": pnc,
        "tap_sfvcsp": float(sfvcsp),
        "tap_total_cdr_length": int(total_cdr),
        "tap_risk_flag_count": int(flags),
        "di_full": float(di_full),
        "di_sap": float(sap),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_structure(
    input: StructureInput,
    exposure_cutoff: float = DEFAULT_EXPOSURE_CUTOFF,
) -> StructureMetrics:
    """Produce a full StructureMetrics report from a PDB path."""
    if input.source is StructureSource.UNAVAILABLE or not input.pdb_path:
        return StructureMetrics(input=input, available=False, notes=input.notes or "no pdb")

    if not os.path.exists(input.pdb_path):
        return StructureMetrics(
            input=input, available=False, notes=f"pdb missing: {input.pdb_path}",
        )

    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("ab", input.pdb_path)
    except Exception as e:
        return StructureMetrics(input=input, available=False, notes=f"PDB parse failed: {e}")

    vh_seq, _ = _parse_chain_sequence(structure, input.vh_chain_id)
    vl_seq, _ = _parse_chain_sequence(structure, input.vl_chain_id)
    if not vh_seq:
        return StructureMetrics(
            input=input, available=False,
            notes=f"no VH chain {input.vh_chain_id} in PDB",
        )

    complex_sasa = _compute_sasa(structure, [input.vh_chain_id, input.vl_chain_id])
    vh_sasa = [complex_sasa.get((input.vh_chain_id, i + 1), 0.0) for i in range(len(vh_seq))]
    vl_sasa = (
        [complex_sasa.get((input.vl_chain_id, i + 1), 0.0) for i in range(len(vl_seq))]
        if vl_seq else []
    )

    cdr_ranges = _locate_cdrs("".join(vh_seq), "".join(vl_seq))
    paratope = _compute_paratope(vh_seq, vl_seq, vh_sasa, vl_sasa, cdr_ranges, exposure_cutoff)
    interface = (
        _compute_interface(structure, input.vh_chain_id, input.vl_chain_id, complex_sasa, exposure_cutoff)
        if vl_seq else None
    )
    patches = _detect_aggregation_patches(
        vh_seq, vl_seq, vh_sasa, vl_sasa, structure,
        input.vh_chain_id, input.vl_chain_id, exposure_cutoff=20.0,
    )
    tap_di = _compute_structural_tap_di(
        vh_seq, vl_seq, vh_sasa, vl_sasa, patches, cdr_ranges, exposure_cutoff,
    )

    # Composite structure_risk_score: bounded [0, 1].
    flag_comp = min(1.0, tap_di["tap_risk_flag_count"] / 4.0)
    patch_comp = min(1.0, sum(p.patch_score for p in patches) / 2.0)
    structure_risk_score = 0.6 * flag_comp + 0.4 * patch_comp

    return StructureMetrics(
        input=input,
        vh_sasa=vh_sasa,
        vl_sasa=vl_sasa,
        vh_resnames=vh_seq,
        vl_resnames=vl_seq,
        paratope=paratope,
        interface=interface,
        aggregation_patches=patches,
        structure_risk_score=float(structure_risk_score),
        available=True,
        **tap_di,
    )
