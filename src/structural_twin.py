"""
src/structural_twin.py — 3D Structural Twin & SASA-Aware Liability Filtering
=============================================================================
ProtePilot — Milestone 25 · Version 1.0

Integrates 3D protein structure prediction and solvent accessibility
analysis to transform 1D sequence-based liability scanning into
biologically meaningful, structure-aware risk assessment.

Pipeline
--------
  1. Sequence → ESMFold API → PDB coordinates
  2. PDB → Shrake-Rupley SASA → per-residue accessibility (A^2)
  3. 1D liabilities × SASA → filter buried (safe) vs exposed (risk)

Science Background
------------------
  - A deamidation motif (NG) buried in the hydrophobic core has near-zero
    risk because it is sterically shielded from water attack.
  - An oxidation-prone Met on the surface is genuinely at risk because
    solvent and reactive oxygen species can access it.
  - SASA threshold: residues with < 10 A^2 are considered buried/safe.
  - The Shrake-Rupley algorithm (1973) computes SASA by placing test
    points on a sphere around each atom and counting solvent-exposed ones.

Graceful Degradation
--------------------
  - If ESMFold API is unreachable → use mock SASA from empirical tables
  - If Bio.PDB not installed → fallback to empirical residue accessibility
  - All functions return structured results regardless of 3D availability
"""

from __future__ import annotations

import io
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ESMFold public API endpoint
ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

# SASA threshold (Angstrom^2) — below this, residue is considered buried
SASA_BURIED_THRESHOLD = 10.0

# Empirical average SASA per amino acid in Gly-X-Gly tripeptides (A^2)
# Used as fallback when 3D structure is unavailable
_EMPIRICAL_MAX_SASA = {
    'A': 113, 'R': 241, 'N': 158, 'D': 151, 'C': 140,
    'Q': 189, 'E': 183, 'G': 85, 'H': 194, 'I': 182,
    'L': 180, 'K': 211, 'M': 204, 'F': 218, 'P': 143,
    'S': 122, 'T': 146, 'W': 259, 'Y': 229, 'V': 160,
}


# ===========================================================================
# 1. Data Classes
# ===========================================================================

@dataclass
class ResidueSASA:
    """SASA data for a single residue."""
    position: int           # 0-indexed
    residue: str            # 1-letter amino acid
    sasa: float             # Solvent Accessible Surface Area (A^2)
    relative_sasa: float    # SASA / max_SASA for this residue type
    is_buried: bool         # SASA < threshold
    is_exposed: bool        # SASA >= threshold


@dataclass
class LiabilityAssessment3D:
    """A single liability with 3D structural context."""
    liability_type: str     # "Deamidation", "Oxidation_Met", "Oxidation_Trp", etc.
    motif: str              # e.g., "NG", "M", "DP"
    position: int           # 0-indexed in sequence
    residue_1letter: str    # single AA at this position
    sasa: float             # A^2
    relative_sasa: float    # fraction of max
    status: str             # "Exposed (High Risk)" or "Buried (Safe)"
    original_1d_flag: bool  # was this flagged by 1D scan?
    retained_3d: bool       # still flagged after 3D filtering?
    chain_name: str = ""    # HC, LC, etc.


@dataclass
class StructuralResult:
    """Full structural analysis result."""
    # Structure
    pdb_available: bool
    pdb_source: str             # "ESMFold API", "Mock/Empirical"
    pdb_data: Optional[str]     # raw PDB text (for 3D viewer)
    structure_confidence: float # mean pLDDT (0-1)

    # SASA
    sasa_per_residue: List[ResidueSASA]
    mean_sasa: float
    n_buried: int
    n_exposed: int

    # 3D-filtered liabilities
    liabilities_3d: List[LiabilityAssessment3D]
    n_total_1d: int             # original 1D count
    n_retained_3d: int          # after 3D filtering
    n_removed_buried: int       # removed because buried
    filtering_ratio: float      # fraction retained

    wall_time_s: float
    summary: str


# ===========================================================================
# 2. ESMFold API Integration
# ===========================================================================

def fetch_esmfold_pdb(sequence: str, timeout: int = 60) -> Optional[str]:
    """
    Send sequence to ESMFold API and retrieve PDB structure.

    Parameters
    ----------
    sequence : Amino acid sequence (max ~400 aa for public API)
    timeout  : Request timeout in seconds

    Returns
    -------
    str or None : PDB file text, or None if failed
    """
    import requests

    seq = sequence.upper().strip()
    # ESMFold public API has a length limit
    if len(seq) > 400:
        log.warning(f"Sequence ({len(seq)} aa) exceeds ESMFold public API limit (~400 aa). "
                    f"Truncating to first 400 residues.")
        seq = seq[:400]

    log.info(f"Requesting ESMFold structure for {len(seq)} aa...")
    try:
        response = requests.post(
            ESMFOLD_URL,
            data=seq,
            headers={"Content-Type": "text/plain"},
            timeout=timeout,
        )
        if response.status_code == 200:
            pdb_text = response.text
            if pdb_text and "ATOM" in pdb_text:
                log.info(f"ESMFold returned PDB: {len(pdb_text)} chars")
                return pdb_text
            else:
                log.warning("ESMFold returned response without ATOM records")
                return None
        else:
            log.warning(f"ESMFold API returned status {response.status_code}: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        log.warning(f"ESMFold API timed out after {timeout}s")
        return None
    except requests.exceptions.ConnectionError:
        log.warning("ESMFold API connection failed (network/DNS error)")
        return None
    except Exception as e:
        log.warning(f"ESMFold API request failed: {e}")
        return None


# ===========================================================================
# 3. SASA Calculation — Bio.PDB Shrake-Rupley
# ===========================================================================

def compute_sasa_from_pdb(pdb_text: str) -> List[ResidueSASA]:
    """
    Parse PDB and compute per-residue SASA using Shrake-Rupley algorithm.

    Parameters
    ----------
    pdb_text : PDB file content as string

    Returns
    -------
    List[ResidueSASA] for each residue in the structure
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", io.StringIO(pdb_text))
    model = structure[0]

    # Compute SASA
    sr = ShrakeRupley()
    sr.compute(model, level="R")  # Residue level

    results = []
    for chain in model:
        for residue in chain:
            if residue.get_id()[0] != " ":
                continue  # skip HETATM

            resname = residue.get_resname()
            # Convert 3-letter to 1-letter
            aa_1letter = _three_to_one(resname)
            if not aa_1letter:
                continue

            resnum = residue.get_id()[1]
            sasa_value = residue.sasa if hasattr(residue, 'sasa') else 0.0

            max_sasa = _EMPIRICAL_MAX_SASA.get(aa_1letter, 180.0)
            rel_sasa = sasa_value / max_sasa if max_sasa > 0 else 0.0

            results.append(ResidueSASA(
                position=resnum - 1,  # convert to 0-indexed
                residue=aa_1letter,
                sasa=round(sasa_value, 2),
                relative_sasa=round(rel_sasa, 4),
                is_buried=sasa_value < SASA_BURIED_THRESHOLD,
                is_exposed=sasa_value >= SASA_BURIED_THRESHOLD,
            ))

    return results


def _three_to_one(resname: str) -> Optional[str]:
    """Convert 3-letter amino acid code to 1-letter."""
    mapping = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }
    return mapping.get(resname.upper().strip())


# ===========================================================================
# 4. Empirical SASA Fallback (When No 3D Structure Available)
# ===========================================================================

def compute_sasa_empirical(sequence: str) -> List[ResidueSASA]:
    """
    Estimate per-residue SASA using empirical accessibility model.

    Uses a sliding-window hydrophobic burial heuristic:
      - Residues in hydrophobic-rich windows are predicted as buried
      - Terminal residues are more exposed
      - Pro, Gly at turns are exposed
      - Core hydrophobic residues (V, I, L, F, A) in long runs are buried

    This is a rough approximation for when ESMFold/Bio.PDB are unavailable.
    """
    seq = sequence.upper()
    n = len(seq)
    results = []

    # Hydrophobicity values (Kyte-Doolittle)
    kd = {
        'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
        'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
        'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
        'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
    }

    # Compute local hydrophobicity (window=7)
    window = 7
    half_w = window // 2
    local_hydro = np.zeros(n)
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        vals = [kd.get(seq[j], 0.0) for j in range(start, end)]
        local_hydro[i] = np.mean(vals)

    for i in range(n):
        aa = seq[i]
        max_sasa = _EMPIRICAL_MAX_SASA.get(aa, 180.0)

        # Base prediction: higher hydrophobicity → more buried
        # Map local_hydro from [-4.5, 4.5] to [0.1, 0.9] fraction exposed
        hydro_norm = (local_hydro[i] + 4.5) / 9.0  # 0 to 1
        fraction_buried = min(0.9, max(0.1, hydro_norm))

        # Terminal residues are always more exposed
        dist_from_end = min(i, n - 1 - i)
        if dist_from_end < 5:
            fraction_buried *= 0.3

        # Pro and Gly tend to be at turns (exposed)
        if aa in ('P', 'G'):
            fraction_buried *= 0.5

        # Charged residues tend to be surface-exposed
        if aa in ('D', 'E', 'K', 'R', 'H'):
            fraction_buried *= 0.4

        estimated_sasa = max_sasa * (1.0 - fraction_buried)
        rel_sasa = estimated_sasa / max_sasa if max_sasa > 0 else 0

        results.append(ResidueSASA(
            position=i,
            residue=aa,
            sasa=round(estimated_sasa, 2),
            relative_sasa=round(rel_sasa, 4),
            is_buried=estimated_sasa < SASA_BURIED_THRESHOLD,
            is_exposed=estimated_sasa >= SASA_BURIED_THRESHOLD,
        ))

    return results


# ===========================================================================
# 5. Extract pLDDT Confidence from PDB B-factors
# ===========================================================================

def _extract_mean_plddt(pdb_text: str) -> float:
    """Extract mean pLDDT from B-factor column of ESMFold PDB."""
    plddts = []
    for line in pdb_text.split("\n"):
        if line.startswith("ATOM") and len(line) >= 66:
            try:
                bfactor = float(line[60:66].strip())
                plddts.append(bfactor)
            except ValueError:
                pass
    if plddts:
        return round(np.mean(plddts) / 100.0, 3)  # normalize 0-100 to 0-1
    return 0.0


# ===========================================================================
# 6. 3D Liability Filtering — The Core Logic
# ===========================================================================

def _collect_1d_liabilities(
    sequence: str,
    liabilities_dict: Optional[Dict] = None,
    chain_name: str = "",
) -> List[Dict]:
    """
    Collect all 1D liability positions from a liability scan result.

    Returns list of {type, motif, position, residue} dicts.
    """
    seq = sequence.upper()
    items = []

    if liabilities_dict is None:
        # Do a quick inline scan
        for i, aa in enumerate(seq):
            if aa == 'M':
                items.append({"type": "Oxidation_Met", "motif": "M", "position": i, "residue": "M"})
            elif aa == 'W':
                items.append({"type": "Oxidation_Trp", "motif": "W", "position": i, "residue": "W"})

        for i in range(len(seq) - 1):
            if seq[i] == 'N' and seq[i+1] in ('G', 'S'):
                items.append({"type": "Deamidation", "motif": seq[i:i+2], "position": i, "residue": "N"})
            if seq[i] == 'D' and seq[i+1] == 'P':
                items.append({"type": "Asp-Pro_Clip", "motif": "DP", "position": i, "residue": "D"})
            if seq[i] == 'D' and seq[i+1] in ('G', 'S'):
                items.append({"type": "Isomerization", "motif": seq[i:i+2], "position": i, "residue": "D"})

        for i in range(len(seq) - 2):
            if seq[i] == 'N' and seq[i+1] != 'P' and seq[i+2] in ('S', 'T'):
                items.append({"type": "N-Glycosylation", "motif": seq[i:i+3], "position": i, "residue": "N"})
    else:
        # Use existing liability dict from scan_sequence_liabilities()
        for pos in liabilities_dict.get("met_positions", []):
            items.append({"type": "Oxidation_Met", "motif": "M", "position": pos, "residue": "M"})
        for pos in liabilities_dict.get("trp_positions", []):
            items.append({"type": "Oxidation_Trp", "motif": "W", "position": pos, "residue": "W"})
        for h in liabilities_dict.get("deamidation_hotspots", []):
            items.append({"type": "Deamidation", "motif": h.get("motif", "N?"),
                          "position": h["pos"], "residue": "N"})
        for pos in liabilities_dict.get("dp_positions", []):
            items.append({"type": "Asp-Pro_Clip", "motif": "DP", "position": pos, "residue": "D"})
        for h in liabilities_dict.get("isomerization_hotspots", []):
            items.append({"type": "Isomerization", "motif": h.get("motif", "D?"),
                          "position": h["pos"], "residue": "D"})
        for g in liabilities_dict.get("n_glyco_motifs", []):
            items.append({"type": "N-Glycosylation", "motif": g.get("motif", "NxT"),
                          "position": g["pos"], "residue": "N"})

    for item in items:
        item["chain_name"] = chain_name

    return items


def filter_liabilities_3d(
    liabilities_1d: List[Dict],
    sasa_data: List[ResidueSASA],
    threshold: float = SASA_BURIED_THRESHOLD,
) -> List[LiabilityAssessment3D]:
    """
    Cross-reference 1D liabilities with 3D SASA to filter buried motifs.

    The Rule: If a liability motif has SASA < threshold (buried), remove it.
    Only retain liabilities that are surface-exposed.
    """
    # Build position → SASA lookup
    sasa_map = {}
    for r in sasa_data:
        sasa_map[r.position] = r

    results = []
    for liab in liabilities_1d:
        pos = liab["position"]
        sasa_info = sasa_map.get(pos)

        if sasa_info:
            sasa_val = sasa_info.sasa
            rel_sasa = sasa_info.relative_sasa
            is_exposed = sasa_val >= threshold
        else:
            # Position not in SASA data (edge case) — assume exposed
            sasa_val = 999.0
            rel_sasa = 1.0
            is_exposed = True

        status = "Exposed (High Risk)" if is_exposed else "Buried (Safe)"

        results.append(LiabilityAssessment3D(
            liability_type=liab["type"],
            motif=liab["motif"],
            position=liab["position"],
            residue_1letter=liab["residue"],
            sasa=sasa_val,
            relative_sasa=rel_sasa,
            status=status,
            original_1d_flag=True,
            retained_3d=is_exposed,
            chain_name=liab.get("chain_name", ""),
        ))

    return results


# ===========================================================================
# 7. Main Structural Analysis Pipeline
# ===========================================================================

def run_structural_analysis(
    sequence: str,
    liabilities_dict: Optional[Dict] = None,
    chain_name: str = "",
    use_api: bool = True,
    sasa_threshold: float = SASA_BURIED_THRESHOLD,
) -> StructuralResult:
    """
    Run the full 3D structural analysis pipeline.

    Parameters
    ----------
    sequence         : Amino acid sequence
    liabilities_dict : Pre-computed liabilities from scan_sequence_liabilities()
    chain_name       : Chain identifier (HC, LC, etc.)
    use_api          : Whether to attempt ESMFold API call
    sasa_threshold   : SASA threshold in A^2 for buried/exposed classification

    Returns
    -------
    StructuralResult with PDB data, SASA, and filtered liabilities
    """
    t0 = time.time()
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    n = len(seq)
    log.info(f"Structural analysis for {chain_name}: {n} residues")

    pdb_data = None
    pdb_source = "Mock/Empirical"
    pdb_available = False
    confidence = 0.0

    # ---- Step 1: Try ESMFold API ----
    if use_api and n <= 400:
        try:
            pdb_data = fetch_esmfold_pdb(seq, timeout=60)
            if pdb_data:
                pdb_available = True
                pdb_source = "ESMFold API"
                confidence = _extract_mean_plddt(pdb_data)
                log.info(f"ESMFold success: pLDDT={confidence:.2f}")
        except Exception as e:
            log.warning(f"ESMFold failed: {e}")

    # ---- Step 2: Compute SASA ----
    sasa_data = []
    if pdb_available and pdb_data:
        try:
            sasa_data = compute_sasa_from_pdb(pdb_data)
            log.info(f"Bio.PDB SASA computed: {len(sasa_data)} residues")
        except ImportError:
            log.warning("Bio.PDB.SASA not available; falling back to empirical")
            sasa_data = compute_sasa_empirical(seq)
            pdb_source = "ESMFold + Empirical SASA"
        except Exception as e:
            log.warning(f"SASA computation failed: {e}; falling back to empirical")
            sasa_data = compute_sasa_empirical(seq)
            pdb_source = "ESMFold + Empirical SASA"
    else:
        sasa_data = compute_sasa_empirical(seq)

    # ---- Step 3: Collect 1D liabilities ----
    liabilities_1d = _collect_1d_liabilities(seq, liabilities_dict, chain_name)

    # ---- Step 4: Cross-reference with SASA → 3D filtering ----
    liabilities_3d = filter_liabilities_3d(liabilities_1d, sasa_data, sasa_threshold)

    # ---- Compute stats ----
    n_buried = sum(1 for r in sasa_data if r.is_buried)
    n_exposed = sum(1 for r in sasa_data if r.is_exposed)
    mean_sasa = float(np.mean([r.sasa for r in sasa_data])) if sasa_data else 0.0

    n_total_1d = len(liabilities_3d)
    n_retained = sum(1 for l in liabilities_3d if l.retained_3d)
    n_removed = n_total_1d - n_retained
    ratio = n_retained / n_total_1d if n_total_1d > 0 else 0.0

    wall_time = time.time() - t0

    # ---- Summary ----
    summary_lines = [
        f"3D Structural Analysis for {chain_name or 'sequence'} ({n} residues):",
        f"  Structure source: {pdb_source}",
        f"  {'pLDDT confidence: ' + f'{confidence:.2f}' if pdb_available else 'Using empirical SASA model'}",
        f"  Residues: {n_buried} buried, {n_exposed} exposed (mean SASA: {mean_sasa:.1f} A^2)",
        f"",
        f"  Liability filtering (SASA threshold: {sasa_threshold} A^2):",
        f"    1D liabilities identified: {n_total_1d}",
        f"    Retained after 3D filtering: {n_retained} ({ratio:.0%})",
        f"    Removed (buried/safe): {n_removed}",
    ]

    return StructuralResult(
        pdb_available=pdb_available,
        pdb_source=pdb_source,
        pdb_data=pdb_data,
        structure_confidence=confidence,
        sasa_per_residue=sasa_data,
        mean_sasa=round(mean_sasa, 2),
        n_buried=n_buried,
        n_exposed=n_exposed,
        liabilities_3d=liabilities_3d,
        n_total_1d=n_total_1d,
        n_retained_3d=n_retained,
        n_removed_buried=n_removed,
        filtering_ratio=round(ratio, 3),
        wall_time_s=round(wall_time, 3),
        summary="\n".join(summary_lines),
    )


# ===========================================================================
# 8. Multi-Chain Structural Analysis
# ===========================================================================

def run_structural_analysis_multi_chain(
    chain_analyses: List[Dict],
    use_api: bool = True,
    sasa_threshold: float = SASA_BURIED_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run structural analysis on multiple chains from chain_analyses list.

    Parameters
    ----------
    chain_analyses : List from intent["chain_analyses"]
    use_api        : Whether to attempt ESMFold API
    sasa_threshold : SASA buried/exposed threshold

    Returns
    -------
    Dict with per-chain results and aggregate statistics
    """
    results = {}
    total_1d = 0
    total_retained = 0
    total_removed = 0
    all_liabilities = []

    for chain in chain_analyses:
        name = chain.get("name", chain.get("chain_type", "Unknown"))
        seq = chain.get("sequence", "")
        liab_dict = chain.get("liabilities")

        if not seq:
            continue

        result = run_structural_analysis(
            sequence=seq,
            liabilities_dict=liab_dict,
            chain_name=name,
            use_api=use_api,
            sasa_threshold=sasa_threshold,
        )
        results[name] = result
        total_1d += result.n_total_1d
        total_retained += result.n_retained_3d
        total_removed += result.n_removed_buried
        all_liabilities.extend(result.liabilities_3d)

    return {
        "chain_results": results,
        "total_1d_liabilities": total_1d,
        "total_retained_3d": total_retained,
        "total_removed_buried": total_removed,
        "filtering_ratio": total_retained / total_1d if total_1d > 0 else 0.0,
        "all_liabilities": all_liabilities,
    }


# ===========================================================================
# 9. Utility — Generate PDB Viewer HTML with Highlighted Residues
# ===========================================================================

def generate_3d_viewer_html(
    pdb_data: str,
    exposed_residues: List[int],
    buried_residues: List[int],
    width: int = 700,
    height: int = 500,
) -> str:
    """
    Generate an HTML snippet for 3Dmol.js viewer with highlighted residues.

    Exposed liabilities → Red
    Buried (safe) liabilities → Green
    """
    # Escape PDB data for embedding in JavaScript
    pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    exposed_str = ",".join(str(r + 1) for r in exposed_residues)  # 1-indexed for PDB
    buried_str = ",".join(str(r + 1) for r in buried_residues)

    html = f"""
<div id="viewport3d" style="width:{width}px;height:{height}px;position:relative;"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
<script>
(function() {{
    var viewer = $3Dmol.createViewer("viewport3d", {{backgroundColor: "white"}});
    var pdb = `{pdb_escaped}`;
    viewer.addModel(pdb, "pdb");
    viewer.setStyle({{}}, {{cartoon: {{color: "spectrum", opacity: 0.85}}}});
"""
    if exposed_str:
        html += f"""
    // Exposed liabilities — RED spheres
    var exposed = [{exposed_str}];
    exposed.forEach(function(resi) {{
        viewer.addStyle({{resi: resi}}, {{stick: {{color: "red", radius: 0.3}}, cartoon: {{color: "red"}}}});
    }});
"""
    if buried_str:
        html += f"""
    // Buried (safe) — GREEN
    var buried = [{buried_str}];
    buried.forEach(function(resi) {{
        viewer.addStyle({{resi: resi}}, {{stick: {{color: "green", radius: 0.2}}, cartoon: {{color: "green"}}}});
    }});
"""
    html += """
    viewer.zoomTo();
    viewer.render();
})();
</script>
"""
    return html


# ===========================================================================
# 10. Self-Test
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("structural_twin.py — Self-Test")
    print("=" * 60)

    # Test sequence (VH domain)
    test_seq = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDSWIHWVRQAPGKGLEWVAWISPYGGSTYYADSVKG"
        "RFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGFDYWGQGTLVTVSS"
    )

    passed = 0
    total = 7

    # Test 1: Empirical SASA
    sasa = compute_sasa_empirical(test_seq)
    assert len(sasa) == len(test_seq)
    assert all(isinstance(r, ResidueSASA) for r in sasa)
    n_bur = sum(1 for r in sasa if r.is_buried)
    n_exp = sum(1 for r in sasa if r.is_exposed)
    print(f"  [1/7] Empirical SASA: {len(sasa)} residues, "
          f"{n_bur} buried, {n_exp} exposed ✅")
    passed += 1

    # Test 2: 1D liability collection
    liab_1d = _collect_1d_liabilities(test_seq, chain_name="VH")
    assert len(liab_1d) > 0
    types = set(l["type"] for l in liab_1d)
    print(f"  [2/7] 1D liabilities: {len(liab_1d)} items, "
          f"types: {types} ✅")
    passed += 1

    # Test 3: 3D filtering
    filtered = filter_liabilities_3d(liab_1d, sasa)
    n_retained = sum(1 for l in filtered if l.retained_3d)
    n_removed = sum(1 for l in filtered if not l.retained_3d)
    print(f"  [3/7] 3D filtering: {n_retained} retained, "
          f"{n_removed} removed (buried) ✅")
    passed += 1

    # Test 4: Full pipeline (no API)
    result = run_structural_analysis(
        test_seq, chain_name="VH", use_api=False
    )
    assert result.n_total_1d > 0
    assert result.n_retained_3d <= result.n_total_1d
    assert result.pdb_source == "Mock/Empirical"
    print(f"  [4/7] Full pipeline: {result.n_total_1d} 1D → "
          f"{result.n_retained_3d} retained ({result.filtering_ratio:.0%}) ✅")
    passed += 1

    # Test 5: Verify buried motifs are removed
    buried_liabs = [l for l in result.liabilities_3d if not l.retained_3d]
    for bl in buried_liabs:
        assert bl.sasa < SASA_BURIED_THRESHOLD
        assert bl.status == "Buried (Safe)"
    print(f"  [5/7] Buried verification: {len(buried_liabs)} correctly classified ✅")
    passed += 1

    # Test 6: Result data structure
    assert isinstance(result.summary, str)
    assert len(result.sasa_per_residue) == len(test_seq)
    print(f"  [6/7] Data structure: summary={len(result.summary)} chars ✅")
    passed += 1

    # Test 7: 3D viewer HTML (mock PDB)
    mock_pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C\n"
    html = generate_3d_viewer_html(mock_pdb, [0, 5], [3, 8])
    assert "3Dmol" in html and "red" in html and "green" in html
    print(f"  [7/7] 3D viewer HTML: {len(html)} chars ✅")
    passed += 1

    print(f"\nSummary:\n{result.summary}")
    print(f"\n{'=' * 60}")
    print(f"Self-test: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
