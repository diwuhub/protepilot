"""Enumerate single-residue mutation candidates for an antibody.

Per-chain strategy:
    - Every position in VH and VL is a candidate site unless filtered.
    - At each site, generate all 19 non-identity substitutions.
    - Label the region using regex-based CDR extraction from
      ab_benchmark.seqprops (same one used by the TAP proxy), so CDR vs
      framework labels are consistent across tools.
    - Positions in framework regions can be skipped entirely with
      `exclude_framework=True` (recommended when the caller is generating
      designs for affinity maturation — framework mutations usually
      regress humanness + manufacturability).

Performance: 137 VH (avg ~115 AA) × 19 AAs ≈ 2 200 candidates per VH.
Add VL (~108 AA) → another ~2 000. ~4.2k candidates per antibody is
tractable for downstream masked-marginal scoring (O(seq_length) forward
passes).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

from src.mutation.schema import (
    CDR_REGIONS,
    FRAMEWORK_REGIONS,
    STANDARD_AA,
    AntibodyChain,
    MutationCandidate,
)


def _region_map(seq: str, cdr_ranges: dict[str, tuple[int, int]]) -> list[str]:
    """Per-position region labels ('framework' or 'cdr_hX'/'cdr_lX') for a chain."""
    labels = ["framework"] * len(seq)
    for name, (start, end) in cdr_ranges.items():
        for i in range(max(0, start), min(len(seq), end)):
            labels[i] = name
    return labels


def _locate_cdrs(vh: str, vl: str) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """Find 0-indexed half-open [start, end) ranges for each CDR via regex.

    Uses ab_benchmark.seqprops.extract_cdrs (when available), then locates
    each extracted substring within the parent chain. If extraction fails,
    returns empty range maps — every position becomes 'framework'.
    """
    vh_ranges: dict[str, tuple[int, int]] = {}
    vl_ranges: dict[str, tuple[int, int]] = {}
    # Lazy sys.path shim for sibling ab-benchmark package. Done inside the
    # function (not at module import time) so it doesn't shadow ProtePilot's
    # own `tests` namespace package during pytest collection.
    _AB_BENCHMARK = os.environ.get(
        "AB_BENCHMARK_PATH",
        str(Path(__file__).resolve().parents[3] / "ab-benchmark"),
    )
    if os.path.isdir(_AB_BENCHMARK) and _AB_BENCHMARK not in sys.path:
        sys.path.append(_AB_BENCHMARK)
    try:
        from ab_benchmark.seqprops import extract_cdrs  # type: ignore
    except ImportError:
        cdrs = _extract_cdrs_regex(vh, vl)
    else:
        cdrs = extract_cdrs(vh, vl)

    region_name = {"h1": "cdr_h1", "h2": "cdr_h2", "h3": "cdr_h3", "l3": "cdr_l3"}
    for key, label in region_name.items():
        if key not in cdrs:
            continue
        sub = cdrs[key]
        chain_seq = vh if key.startswith("h") else vl
        idx = chain_seq.find(sub)
        if idx < 0:
            continue
        target = vh_ranges if key.startswith("h") else vl_ranges
        target[label] = (idx, idx + len(sub))
    return vh_ranges, vl_ranges


_HCDR3_RE = re.compile(r"C([A-Z]{3,30}?)W[GT][A-Z]G")
_LCDR3_RE = re.compile(r"C([A-Z]{3,30}?)[FW]G[A-Z]G")
_HCDR1_RE = re.compile(r"C[A-Z]{4}([A-Z]{5,10})W[IV]R")
_HCDR2_RE = re.compile(r"W[IV]R[A-Z]{14}([A-Z]{15,20})")


def _extract_cdrs_regex(vh: str, vl: str = "") -> dict[str, str]:
    """Local fallback matching ab-benchmark's heuristic CDR extraction."""
    out: dict[str, str] = {}

    h3_matches = list(_HCDR3_RE.finditer(vh))
    if h3_matches:
        out["h3"] = h3_matches[-1].group(1)

    m = _HCDR1_RE.search(vh)
    if m:
        out["h1"] = m.group(1)

    m = _HCDR2_RE.search(vh)
    if m:
        out["h2"] = m.group(1)

    if vl:
        l3_matches = list(_LCDR3_RE.finditer(vl))
        if l3_matches:
            out["l3"] = l3_matches[-1].group(1)

    return out


def enumerate_single_mutations(
    vh: str,
    vl: str = "",
    allowed_aas: str | Iterable[str] = STANDARD_AA,
    exclude_framework: bool = False,
    restrict_to_positions: dict[AntibodyChain, list[int]] | None = None,
) -> list[MutationCandidate]:
    """Yield every valid single-residue substitution across VH and VL.

    Parameters
    ----------
    vh, vl : parent sequences (AA strings; VL may be empty for heavy-only)
    allowed_aas : which mutant residues to consider (default: all 20 standard)
    exclude_framework : if True, skip positions labeled 'framework'
    restrict_to_positions : optional per-chain allow-list of position indices;
        when set for a chain, ONLY those positions produce candidates. VH
        positions are 0-indexed within VH (same for VL).
    """
    vh = (vh or "").upper()
    vl = (vl or "").upper()
    if not vh:
        raise ValueError("VH sequence is required")
    for s, name in ((vh, "VH"), (vl, "VL")):
        if s and any(c not in STANDARD_AA for c in s):
            raise ValueError(f"{name} contains non-standard residue(s)")

    allowed_set = set(allowed_aas)
    if not allowed_set.issubset(STANDARD_AA):
        raise ValueError(f"allowed_aas {allowed_set!r} must be subset of {STANDARD_AA!r}")

    vh_cdr_ranges, vl_cdr_ranges = _locate_cdrs(vh, vl)
    vh_regions = _region_map(vh, vh_cdr_ranges)
    vl_regions = _region_map(vl, vl_cdr_ranges) if vl else []

    out: list[MutationCandidate] = []

    def _emit(chain: AntibodyChain, seq: str, region_labels: list[str]) -> None:
        pos_filter = None
        if restrict_to_positions and chain in restrict_to_positions:
            pos_filter = set(restrict_to_positions[chain])

        for position, wt in enumerate(seq):
            if pos_filter is not None and position not in pos_filter:
                continue
            region = region_labels[position]
            if exclude_framework and region in FRAMEWORK_REGIONS:
                continue
            for mut in allowed_set:
                if mut == wt:
                    continue
                out.append(
                    MutationCandidate(
                        chain=chain,
                        position=position,
                        wildtype_aa=wt,
                        mutant_aa=mut,
                        region=region,
                    )
                )

    _emit(AntibodyChain.VH, vh, vh_regions)
    if vl:
        _emit(AntibodyChain.VL, vl, vl_regions)

    return out


def cdr_only(vh: str, vl: str = "") -> list[MutationCandidate]:
    """Convenience: all single mutations restricted to CDR regions."""
    return enumerate_single_mutations(vh, vl, exclude_framework=True)
