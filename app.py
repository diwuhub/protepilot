"""
ProtePilot — Copilot Chat Console
====================================
Version 32.1: System Simplification + Unified Acceptance Criteria + Foundation Fixes

Features:
  - [M27] Virtual Analytical QC Lab: cIEF charge variant, CE-SDS purity, Glycan Profile
  - [M27] Time-Series Stability Twin: Arrhenius kinetics, 5C/40C ICH projections
  - [M27] Multi-Objective Pareto Frontier: NSGA-II dominance, crowding distance
  - [M27] Excipient stabilization: sucrose, trehalose, PS80, arginine modifiers
  - [M27] Pareto scatter in Discovery & HT Screening page (frontier vs dominated)
  - [M21] COGS Digital Twin: commercial manufacturing cost ($/gram) calculator
  - [M21] LLM Co-Pilot: context-aware scientific chatbot (OpenAI or mock fallback)
  - [M21] Manufacturability & COGS page with Plotly pie chart cost breakdown
  - [M21] Global molecule context preserved across all workshop pages
  - [M20] Enterprise UI Overhaul: sidebar radio navigation with 7 Workshop pages
  - [M20] Upstream Bioreactor Digital Twin: CHO Fed-Batch ODE simulation
  - [M20] HT Data Viewer: st.dataframe() for CSV candidate inspection
  - [M20] Factory Reset: delete trained models and revert to baseline heuristics
  - [M20] DoE prominently displayed in dedicated Downstream Purification page
  - [M19] Model Persistence: joblib/torch.save auto-save/load for all ML models
  - [M19] Model Status Indicator: sidebar shows Trained vs Heuristic per predictor
  - [M19] GoSilico-Style In-Silico DoE: grid search over pH × gradient for purification
  - [M19] DoE Contour Plot: Plotly heatmap of Resolution across Design Space
  - [M19] Optimal Sweet Spot identification with Rs, Yield, and peak retention times
  - [M18] IND-Ready Executive Report generation (Word .docx)
  - [M18] Headless Plotly chart capture via kaleido (chromatogram, SHAP, Magic Quadrant)
  - [M18] Professional report sections: Molecular Profile, Developability, Formulation, Purification
  - [M18] Sidebar "Generate Executive Report" button with st.download_button
  - [M17] Formulation Digital Twin: Buffer pH, Buffer Type, Excipient simulation
  - [M17] Henderson-Hasselbalch net charge at buffer pH (physics feedback loop)
  - [M17] Real-time Developability Score adjustment from formulation conditions
  - [M17] Excipient catalog: Trehalose, Sucrose, PS80 with mechanistic models
  - [M17] Exhaustive help= tooltips on every interactive widget (UX Masterclass)
  - [M16] Early Discovery HT Screening tab with CSV bulk upload
  - [M16] High-Throughput Virtual Screening Engine (hundreds of candidates)
  - [M16] Potency/Affinity Predictor (ELISA/Kd → Predicted Potency Score)
  - [M16] Magic Quadrant interactive scatter plot (Developability vs Potency)
  - [M16] Auto-select Star quadrant candidates (top-right)
  - [M16] Downloadable CSV screening report with quadrant classification
  - [M16] Mock Discovery dataset generator (200 synthetic candidates)
  - [M15] Data Foundation & Model Training Dashboard
  - [M15] Jain-137 mAb dataset ingestion (HC+LC sequences + wet-lab assays)
  - [M15] Mock Jain-137 generator (50 synthetic clinical mAbs for testing)
  - [M15] XGBoost supervised training on real wet-lab CSV data (Agg%, Tm)
  - [M15] R² / RMSE model performance metrics display
  - [M15] Multi-objective Pareto-optimal variant filtering
  - [M15] Wet-lab safety checks: reject variants that spike Agg% or drop Tm
  - [M14] One-Click Auto-Optimize: generative AI sequence engineering
  - [M14] Liability-targeted mutagenesis (Met→Leu, NG→QG, DP→EP, etc.)
  - [M14] Charge engineering for bispecific resolution improvement
  - [M14] Hydrophobicity reduction for aggregation risk mitigation
  - [M14] Comparative table: WT vs 3 Optimized Variants (pI, PK, liabilities)
  - [M14] FASTA download for optimized sequences
  - [M13] Host Cell Glycoform Profile selector (Standard CHO, High-Mannose, Sialylated)
  - [M13] Sialylation pI shift → Physics feedback loop → earlier CADET elution
  - [M13] Preclinical PK / Half-Life predictor (empirical model, ~21 day baseline)
  - [M13] PK Risk gauge with penalty breakdown and engineering recommendations
  - [M13] Glycoform impact on intact mass, pI, and chromatography
  - [M12] Dynamic multi-chain stoichiometry UI (Add Chain + Copy Number)
  - [M12] Super-Sequence construction: global pI/MW from assembled tetramer
  - [M12] Stoichiometric Intact Mass: (Mass_HC * 2) + (Mass_LC * 2)
  - [M12] Source Chain column in peptide map (HC, LC, Fusion Arm)
  - [M12] Liability Density: motifs per 1000 assembled residues
  - [M11+] Deep-cache workspace history — click any past run to re-render instantly
  - [M11+] In-silico Mass Spectrometry panel (intact mass + tryptic digest)
  - [M11+] Glycoform profiling (G0F, G1F, G2F) for mAb intact mass
  - [M11+] Searchable peptide map with liability motif highlighting
  - [M11] Bispecific / Fusion protein mode with dual FASTA input
  - [M11] Homodimer AA / Heterodimer AB / Homodimer BB species simulation
  - [M11] 3-component chromatogram visualization with species overlay
  - [M11] Resolution (Rs) calculation and homodimer co-elution risk alerts
  - [M11] Charge asymmetry engineering recommendations
  - [M10] Enterprise workspace management (session isolation, run history, deletion)
  - [M10] FASTA file upload via st.file_uploader (large sequence support)
  - [M10] CSV data pipeline for historical candidate bulk import
  - [M10] Expert labeling UI (human-in-the-loop correction)
  - [M10] Continuous Learning dashboard (retraining, accuracy tracker, Plotly charts)
  - [M10] Report export (Markdown / CSV) via st.download_button
  - [M9] 3D protein visualization (py3Dmol/stmol) with spatial liability mapping
  - [M9] Production-grade ML validation (time-based split, batch-shift split, bootstrap CI)
  - [M8] pLM (ESM-2) protein language model embeddings
  - [M8] XGBoost developability risk prediction (aggregation, stability, viscosity)
  - [M8] Composite Developability Score with SHAP TreeExplainer
  - [M8] Actionable engineering advice from AI insights
  - [M8] Analytical validation plan generation (SEC, DSF, MAM, etc.)
  - ML-First pipeline: PyTorch MLP drives ka/nu prediction (static formulas = fallback)
  - RT-targeted learning: neural network trained to hit 15-20 min elution window
  - 7-feature input: pI, MW, deam, ox, acidic, basic, hydrophobicity (GRAVY)
  - SHAP interpretability (waterfall + summary plots)
  - FASTA sequence parsing via Biopython ProteinAnalysis
  - Multi-chain support (Heavy Chain / Light Chain detection)
  - Comprehensive liability scanning (oxidation, glycosylation, clipping, etc.)
  - CDR highlighting framework (heuristic-based)
  - PropertyMapper v7.0 with ml_override + bispecific support

Dependencies: streamlit, numpy, biopython, plotly, pandas, torch, shap, matplotlib,
              xgboost (optional), transformers (optional), py3Dmol (optional), stmol (optional)

Launch:
    cd ProtePilot
    streamlit run app.py
"""

import sys
import os
import re
import time
import uuid
import logging

log = logging.getLogger(__name__)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import numpy as np

# -- Project paths ----------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.agents import PharmaAgentManager, predict_physical_params, list_tools
from src.batch_processor import HighThroughputOrchestrator
from src.workspace_manager import WorkspaceStore, generate_report
from src.data_pipeline import parse_csv_upload, parse_fasta_file, ExpertLabelStore, validate_csv_data
from src.continuous_learning import (
    ContinuousLearningEngine,
    build_loss_over_epochs_chart,
    build_improvement_over_runs_chart,
)

# M30: FDA 21 CFR Part 11 Audit Trail
from src.audit_logger import (
    get_audit_logger,
    log_model_training as audit_log_model_training,
    log_batch_prediction as audit_log_batch_prediction,
    log_report_generation as audit_log_report_generation,
    log_plm_embedding as audit_log_plm_embedding,
    log_data_import as audit_log_data_import,
    log_factory_reset as audit_log_factory_reset,
    get_summary_stats as audit_get_summary_stats,
    get_all_records as audit_get_all_records,
    verify_integrity as audit_verify_integrity,
)


# ===========================================================================
#  0A. FASTA Sequence Utilities
# ===========================================================================

def is_fasta_input(text: str) -> bool:
    """Detect FASTA-format amino acid sequence input.

    Handles # comment lines (e.g. assembly headers) that may precede
    standard FASTA content.
    """
    text = text.strip()
    # Strip leading comment lines (# or ;) before checking
    lines = text.split("\n")
    _non_comment_lines = [l for l in lines if not l.strip().startswith("#") and not l.strip().startswith(";")]
    _stripped = "\n".join(_non_comment_lines).strip()
    if _stripped.startswith(">"):
        return True
    if text.startswith(">"):
        return True
    clean = re.sub(r'\s+', '', _stripped)
    if len(clean) > 20 and re.match(r'^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]+$', clean):
        return True
    return False


def parse_multi_chain_fasta(text: str) -> List[Dict[str, Any]]:
    """
    Parse multi-FASTA input into individual chains.
    Detects HC/LC from header keywords or sequence length heuristics.

    Handles # and ; comment lines (e.g. assembly headers) that may
    precede standard FASTA content.

    Returns list of {"name": str, "sequence": str, "chain_type": "HC"|"LC"|"unknown"}
    """
    text = text.strip()
    chains = []
    current_header = None
    current_lines = []

    # Strip comment lines (# or ;) before parsing — standard FASTA convention
    lines_raw = text.split("\n")
    lines = [l for l in lines_raw if not l.strip().startswith("#") and not l.strip().startswith(";")]
    text_clean = "\n".join(lines).strip()

    # Handle raw sequence (no header)
    if not text_clean.startswith(">"):
        seq = re.sub(r'[^A-Za-z]', '', text_clean).upper()
        if len(seq) >= 10:
            chain_type = "HC" if len(seq) > 300 else ("LC" if len(seq) > 100 else "unknown")
            return [{"name": f"Chain_{uuid.uuid4().hex[:6]}", "sequence": seq, "chain_type": chain_type}]
        return []

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            # Save previous chain
            if current_header is not None and current_lines:
                seq = re.sub(r'[^A-Za-z]', '', "".join(current_lines)).upper()
                if len(seq) >= 10:
                    chains.append({"name": current_header, "sequence": seq, "chain_type": "unknown"})
            # New header
            header = line[1:].strip()
            current_header = header.split()[0] if header else f"Chain_{uuid.uuid4().hex[:6]}"
            current_lines = []
        else:
            current_lines.append(line)

    # Save last chain
    if current_header is not None and current_lines:
        seq = re.sub(r'[^A-Za-z]', '', "".join(current_lines)).upper()
        if len(seq) >= 10:
            chains.append({"name": current_header, "sequence": seq, "chain_type": "unknown"})

    # Classify chains by header keywords or length
    # Long keywords use substring match; short ones (hc, lc, vh, vl) use word-boundary
    # to avoid false positives like "hc" matching inside "lightchain"
    hc_long = ["heavy", "gamma"]              # substring safe (unique enough)
    hc_short_pats = [r'\bhc\b', r'\bvh\b', r'\bigh\b']  # need word boundary
    lc_long = ["light", "kappa", "lambda"]
    lc_short_pats = [r'\blc\b', r'\bvl\b', r'\bigk\b', r'\bigl\b']

    for chain in chains:
        name_lower = chain["name"].lower()
        name_spaced = re.sub(r'[|_\-/]', ' ', name_lower)

        is_hc = any(kw in name_lower for kw in hc_long) or \
                any(re.search(p, name_spaced) for p in hc_short_pats)
        is_lc = any(kw in name_lower for kw in lc_long) or \
                any(re.search(p, name_spaced) for p in lc_short_pats)

        if is_hc and not is_lc:
            chain["chain_type"] = "HC"
        elif is_lc and not is_hc:
            chain["chain_type"] = "LC"
        elif is_hc and is_lc:
            # Both matched — use length heuristic as tiebreaker
            chain["chain_type"] = "HC" if len(chain["sequence"]) > 250 else "LC"
        elif chain["chain_type"] == "unknown":
            # Length heuristic: HC ~ 440-460 aa, LC ~ 210-230 aa
            if len(chain["sequence"]) > 300:
                chain["chain_type"] = "HC"
            elif len(chain["sequence"]) > 100:
                chain["chain_type"] = "LC"

    return chains


def parse_fasta_sequence(text: str) -> Optional[Dict[str, Any]]:
    """Parse FASTA and compute biophysical properties using Biopython."""
    _biopython_available = True
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
    except ImportError:
        log.warning("Biopython not installed — using fallback FASTA parser")
        _biopython_available = False

    chains = parse_multi_chain_fasta(text)
    if not chains:
        log.warning("parse_multi_chain_fasta returned no chains for input len=%d", len(text))
        return None

    # Combine all chains for overall properties
    combined_seq = "".join(c["sequence"] for c in chains)
    name = chains[0]["name"]

    if _biopython_available:
        try:
            analysis = ProteinAnalysis(combined_seq)
            pI = analysis.isoelectric_point()
            mw_da = analysis.molecular_weight()
            mw_kda = mw_da / 1000.0
            gravy = analysis.gravy()
            hydrophobicity = max(0.0, min(1.0, (gravy + 2.0) / 4.0))

            return {
                "name": name,
                "pI": round(pI, 2),
                "mw": round(mw_kda, 1),
                "sequence": combined_seq,
                "seq_length": len(combined_seq),
                "gravy": round(gravy, 3),
                "hydrophobicity": round(hydrophobicity, 3),
                "chains": chains,
                "num_chains": len(chains),
                "multi_chain_detected": len(chains) > 1,
                "n_chains_detected": len(chains),
            }
        except Exception as e:
            log.error("Biopython ProteinAnalysis failed: %s — using fallback", e)

    # ---------- Fallback: estimate biophysical properties without Biopython ------
    seq = combined_seq.upper()
    n = len(seq)

    # Amino acid pKa-based pI estimation (Henderson-Hasselbalch isoelectric)
    _aa_counts = {aa: seq.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    _pos_charge_aa = {"K": 10.5, "R": 12.4, "H": 6.0}   # pKa
    _neg_charge_aa = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
    _n_term_pka = 9.69
    _c_term_pka = 2.34

    def _net_charge(ph):
        charge = 0.0
        # N-terminus
        charge += 1.0 / (1.0 + 10.0 ** (ph - _n_term_pka))
        # C-terminus
        charge -= 1.0 / (1.0 + 10.0 ** (_c_term_pka - ph))
        for aa, pka in _pos_charge_aa.items():
            charge += _aa_counts.get(aa, 0) * (1.0 / (1.0 + 10.0 ** (ph - pka)))
        for aa, pka in _neg_charge_aa.items():
            charge -= _aa_counts.get(aa, 0) * (1.0 / (1.0 + 10.0 ** (pka - ph)))
        return charge

    # Bisection method for pI
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _net_charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    pI_est = round((lo + hi) / 2.0, 2)

    # MW estimation: average amino acid MW ~111.1 Da + 18 (water)
    _avg_mw = {"A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
               "E": 147.1, "Q": 146.2, "G": 75.0, "H": 155.2, "I": 131.2,
               "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
               "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1}
    mw_da_est = sum(_avg_mw.get(aa, 111.1) for aa in seq) - (n - 1) * 18.015
    mw_kda_est = round(mw_da_est / 1000.0, 1)

    # GRAVY estimation
    _kd_hydrophobicity = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
                          "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
                          "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
                          "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
    gravy_est = sum(_kd_hydrophobicity.get(aa, 0.0) for aa in seq) / max(n, 1)
    gravy_est = round(gravy_est, 3)
    hydro_est = round(max(0.0, min(1.0, (gravy_est + 2.0) / 4.0)), 3)

    log.info("Fallback FASTA analysis: pI=%.2f, MW=%.1f kDa, GRAVY=%.3f",
             pI_est, mw_kda_est, gravy_est)

    return {
        "name": name,
        "pI": pI_est,
        "mw": mw_kda_est,
        "sequence": combined_seq,
        "seq_length": n,
        "gravy": gravy_est,
        "hydrophobicity": hydro_est,
        "chains": chains,
        "num_chains": len(chains),
        "multi_chain_detected": len(chains) > 1,
        "n_chains_detected": len(chains),
    }


def _compute_stoichiometric_properties(
    chains: list,
    glycan_mass_per_site_da: float = 2400.0,
    n_glycan_sites_per_hc: int = 1,
) -> dict:
    """
    Compute stoichiometric MW, pI, GRAVY for assembled multi-chain molecule.

    For mAb (2×HC + 2×LC): MW = 2×MW_HC + 2×MW_LC + glycan_mass
    """
    if not chains:
        return {}

    total_mw_da = 0.0
    assembled_seq = ""
    n_hc = 0

    for ch in chains:
        seq = ch.get("sequence", "")
        copy = ch.get("copy_number", 1)
        chain_mw_da = len(seq) * 110.0  # avg AA mass ~110 Da
        total_mw_da += chain_mw_da * copy
        assembled_seq += seq * copy
        if ch.get("chain_type", "").upper() in ("HC", "HEAVY", "heavy"):
            n_hc += copy

    # Add glycan mass
    glycan_total = n_hc * n_glycan_sites_per_hc * glycan_mass_per_site_da
    total_mw_with_glycan = total_mw_da + glycan_total

    # Compute pI and GRAVY on assembled sequence
    result = {
        "mw_kda_assembled": round(total_mw_with_glycan / 1000.0, 2),
        "mw_kda_protein_only": round(total_mw_da / 1000.0, 2),
        "n_chains_total": sum(ch.get("copy_number", 1) for ch in chains),
    }

    if assembled_seq:
        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            analysis = ProteinAnalysis(assembled_seq)
            result["pI_assembled"] = round(analysis.isoelectric_point(), 2)
            result["gravy_assembled"] = round(analysis.gravy(), 4)
        except Exception:
            n = len(assembled_seq)
            basic = assembled_seq.upper().count("K") + assembled_seq.upper().count("R") + assembled_seq.upper().count("H")
            acidic = assembled_seq.upper().count("D") + assembled_seq.upper().count("E")
            result["pI_assembled"] = round(7.0 + 0.5 * (basic - acidic) / max(n, 1), 2)

    return result


# ===========================================================================
#  0B. Sequence Liability Scanner
# ===========================================================================

def scan_sequence_liabilities(sequence: str) -> Dict[str, Any]:
    """
    Comprehensive sequence liability analysis for antibody developability.

    Scans for:
      - Oxidation-prone residues: Met (M) and Trp (W) positions
      - N-Glycosylation motifs: N-X-S/T where X != P (Asn-linked glycosylation)
      - Disulfide potential: Cys (C) positions
      - Asp-Pro clipping: D-P motifs (acid-labile)
      - Deamidation hotspots: N-G, N-S motifs (Asn deamidation)
      - Isomerization risk: D-G, D-S motifs (Asp isomerization)
      - Acidic residues: D, E counts
      - Basic residues: K, R, H counts

    Returns dict with counts and position lists.
    """
    seq = sequence.upper()
    n = len(seq)

    liabilities = {
        "length": n,
        # Oxidation-prone residues
        "met_positions": [i for i, aa in enumerate(seq) if aa == "M"],
        "trp_positions": [i for i, aa in enumerate(seq) if aa == "W"],
        "met_count": seq.count("M"),
        "trp_count": seq.count("W"),
        # N-Glycosylation motifs: N-X-S/T where X != P
        "n_glyco_motifs": [],
        # Disulfide potential
        "cys_positions": [i for i, aa in enumerate(seq) if aa == "C"],
        "cys_count": seq.count("C"),
        # Asp-Pro clipping
        "dp_positions": [],
        # Deamidation hotspots: N-G, N-S
        "deamidation_hotspots": [],
        # Isomerization risk: D-G, D-S
        "isomerization_hotspots": [],
        # Charge composition
        "acidic_count": seq.count("D") + seq.count("E"),
        "basic_count": seq.count("K") + seq.count("R") + seq.count("H"),
        "asp_count": seq.count("D"),
        "glu_count": seq.count("E"),
        "lys_count": seq.count("K"),
        "arg_count": seq.count("R"),
        "his_count": seq.count("H"),
    }

    # Scan for motifs
    for i in range(n - 2):
        # N-X-S/T where X != P
        if seq[i] == "N" and seq[i+1] != "P" and seq[i+2] in ("S", "T"):
            liabilities["n_glyco_motifs"].append({"pos": i, "motif": seq[i:i+3]})

    for i in range(n - 1):
        # Asp-Pro clipping
        if seq[i] == "D" and seq[i+1] == "P":
            liabilities["dp_positions"].append(i)
        # Deamidation hotspots
        if seq[i] == "N" and seq[i+1] in ("G", "S"):
            liabilities["deamidation_hotspots"].append({"pos": i, "motif": seq[i:i+2]})
        # Isomerization
        if seq[i] == "D" and seq[i+1] in ("G", "S"):
            liabilities["isomerization_hotspots"].append({"pos": i, "motif": seq[i:i+2]})

    liabilities["n_glyco_count"] = len(liabilities["n_glyco_motifs"])
    liabilities["dp_count"] = len(liabilities["dp_positions"])
    liabilities["deamidation_count"] = len(liabilities["deamidation_hotspots"])
    liabilities["isomerization_count"] = len(liabilities["isomerization_hotspots"])

    # Risk summary
    risk_flags = []
    if liabilities["met_count"] > 5:
        risk_flags.append("High Met oxidation risk")
    if liabilities["trp_count"] > 4:
        risk_flags.append("High Trp oxidation risk")
    if liabilities["n_glyco_count"] > 2:
        risk_flags.append("Multiple N-glycosylation sites")
    if liabilities["dp_count"] > 0:
        risk_flags.append("Asp-Pro clipping risk")
    if liabilities["deamidation_count"] > 3:
        risk_flags.append("High deamidation risk")
    liabilities["risk_flags"] = risk_flags

    return liabilities


# ===========================================================================
#  0C. CDR Identification Framework (Heuristic-Based)
# ===========================================================================

def identify_cdrs_heuristic(sequence: str, chain_type: str = "HC") -> List[Dict[str, Any]]:
    """
    Heuristic CDR identification based on conserved residue patterns.

    For Heavy Chain (VH, ~120 aa variable region):
      - Find first Cys (typically pos ~22): marks start of CDR-H1 region
      - CDR-H1: Cys+4 to Cys+15 (approx. 10-12 residues after first Cys)
      - Find Trp after CDR-H1 (typically pos ~36): FR2 anchor
      - CDR-H2: Trp+15 to Trp+30 (approx. 15 residues)
      - Find second Cys (typically pos ~92-96): marks CDR-H3 region
      - CDR-H3: Cys2+3 to Cys2+15 (highly variable length)

    For Light Chain (VL, ~107 aa variable region):
      - CDR-L1: first Cys+1 to Cys+16 (approx. 11-16 residues)
      - CDR-L2: approx. positions 49-56 relative to first Cys
      - CDR-L3: second Cys+1 to Cys2+10

    Returns list of {"name": str, "start": int, "end": int, "sequence": str}
    Limited to first ~130 aa (variable region).
    """
    seq = sequence.upper()
    vr_len = min(len(seq), 130)  # Variable region is first ~120-130 aa
    vr = seq[:vr_len]

    cdrs = []

    # Find conserved Cys residues in variable region
    cys_positions = [i for i, aa in enumerate(vr) if aa == "C"]

    if len(cys_positions) < 2:
        return cdrs  # Cannot identify CDRs without conserved Cys

    cys1 = cys_positions[0]   # First conserved Cys (~position 22-23)
    cys2 = cys_positions[-1]  # Second conserved Cys (~position 88-96)

    # Find Trp after first Cys (FR2 anchor, typically ~14 residues after Cys1)
    trp_pos = None
    for i in range(cys1 + 10, min(cys1 + 25, vr_len)):
        if vr[i] == "W":
            trp_pos = i
            break

    if chain_type == "HC":
        # CDR-H1: from ~Cys1+4 to ~Cys1+15
        cdr1_start = cys1 + 4
        cdr1_end = min(cys1 + 16, vr_len)
        if cdr1_end > cdr1_start:
            cdrs.append({"name": "CDR-H1", "start": cdr1_start, "end": cdr1_end,
                         "sequence": vr[cdr1_start:cdr1_end]})

        # CDR-H2: from ~Trp+15 or Cys1+30 for ~16 residues
        if trp_pos is not None:
            cdr2_start = trp_pos + 15
        else:
            cdr2_start = cys1 + 30
        cdr2_end = min(cdr2_start + 17, vr_len)
        if cdr2_end > cdr2_start and cdr2_start < vr_len:
            cdrs.append({"name": "CDR-H2", "start": cdr2_start, "end": cdr2_end,
                         "sequence": vr[cdr2_start:cdr2_end]})

        # CDR-H3: from ~Cys2+3 for variable length
        cdr3_start = cys2 + 3
        cdr3_end = min(cdr3_start + 15, vr_len)
        if cdr3_end > cdr3_start and cdr3_start < vr_len:
            cdrs.append({"name": "CDR-H3", "start": cdr3_start, "end": cdr3_end,
                         "sequence": vr[cdr3_start:cdr3_end]})

    elif chain_type == "LC":
        # CDR-L1: from ~Cys1+1 for ~16 residues
        cdr1_start = cys1 + 1
        cdr1_end = min(cys1 + 17, vr_len)
        if cdr1_end > cdr1_start:
            cdrs.append({"name": "CDR-L1", "start": cdr1_start, "end": cdr1_end,
                         "sequence": vr[cdr1_start:cdr1_end]})

        # CDR-L2: from ~Cys1+28 for ~7 residues
        cdr2_start = cys1 + 28
        cdr2_end = min(cdr2_start + 7, vr_len)
        if cdr2_end > cdr2_start and cdr2_start < vr_len:
            cdrs.append({"name": "CDR-L2", "start": cdr2_start, "end": cdr2_end,
                         "sequence": vr[cdr2_start:cdr2_end]})

        # CDR-L3: from ~Cys2+1 for ~9 residues
        cdr3_start = cys2 + 1
        cdr3_end = min(cdr3_start + 10, vr_len)
        if cdr3_end > cdr3_start and cdr3_start < vr_len:
            cdrs.append({"name": "CDR-L3", "start": cdr3_start, "end": cdr3_end,
                         "sequence": vr[cdr3_start:cdr3_end]})

    return cdrs


def render_sequence_html(
    sequence: str,
    cdrs: List[Dict[str, Any]],
    liabilities: Dict[str, Any],
    line_width: int = 60,
) -> str:
    """
    Render an annotated HTML sequence with CDR and liability highlighting.

    Color scheme:
      - CDR regions: golden background (#FEF3C7)
      - Met (M) oxidation: red text on pink bg
      - Trp (W) oxidation: dark red text on light pink bg
      - N-glycosylation motif: blue underline
      - Asp-Pro (D-P): purple background
      - Cys (C): green text
      - Deamidation (N-G, N-S): orange underline
    """
    seq = sequence.upper()
    n = len(seq)

    # Build annotation map for each position
    # Priority: CDR bg < specific liability highlight
    annotations = [{"bg": None, "fg": None, "underline": False, "tooltip": ""} for _ in range(n)]

    # Mark CDR regions
    for cdr in cdrs:
        for i in range(cdr["start"], min(cdr["end"], n)):
            annotations[i]["bg"] = "#FEF3C7"
            annotations[i]["tooltip"] = cdr["name"]

    # Mark liabilities
    for pos in liabilities.get("met_positions", []):
        if pos < n:
            annotations[pos]["fg"] = "#EF4444"
            annotations[pos]["bg"] = "#FEE2E2"
            annotations[pos]["tooltip"] += " Met-Ox" if annotations[pos]["tooltip"] else "Met-Ox"

    for pos in liabilities.get("trp_positions", []):
        if pos < n:
            annotations[pos]["fg"] = "#991B1B"
            annotations[pos]["bg"] = "#FECACA"
            annotations[pos]["tooltip"] += " Trp-Ox" if annotations[pos]["tooltip"] else "Trp-Ox"

    for motif in liabilities.get("n_glyco_motifs", []):
        pos = motif["pos"]
        for i in range(pos, min(pos + 3, n)):
            annotations[i]["underline"] = True
            annotations[i]["fg"] = annotations[i]["fg"] or "#1D4ED8"
            annotations[i]["tooltip"] += " N-Glyco" if annotations[i]["tooltip"] else "N-Glyco"

    for pos in liabilities.get("dp_positions", []):
        for i in range(pos, min(pos + 2, n)):
            if i < n:
                annotations[i]["bg"] = "#E9D5FF"
                annotations[i]["tooltip"] += " DP-Clip" if annotations[i]["tooltip"] else "DP-Clip"

    for pos in liabilities.get("cys_positions", []):
        if pos < n:
            annotations[pos]["fg"] = annotations[pos]["fg"] or "#10B981"

    for hs in liabilities.get("deamidation_hotspots", []):
        pos = hs["pos"]
        for i in range(pos, min(pos + 2, n)):
            if i < n:
                annotations[i]["underline"] = True
                annotations[i]["fg"] = annotations[i]["fg"] or "#F59E0B"

    # Build HTML
    html_parts = [
        '<div style="font-family: \'Courier New\', monospace; font-size: 0.8rem; ',
        'line-height: 1.6; background: #FAFBFC; padding: 12px; border-radius: 8px; ',
        'border: 1px solid #E2E8F0; overflow-x: auto; white-space: pre-wrap;">\n',
    ]

    for line_start in range(0, n, line_width):
        line_end = min(line_start + line_width, n)
        # Position label
        html_parts.append(f'<span style="color:#94A3B8; font-size:0.7rem">{line_start+1:>5} </span>')

        for i in range(line_start, line_end):
            aa = seq[i]
            ann = annotations[i]
            styles = []
            if ann["bg"]:
                styles.append(f"background:{ann['bg']}")
            if ann["fg"]:
                styles.append(f"color:{ann['fg']}")
            if ann["underline"]:
                styles.append("text-decoration:underline")
            styles.append("padding:0 1px")

            style_str = ";".join(styles)
            tooltip = f' title="{ann["tooltip"]}"' if ann["tooltip"] else ""
            html_parts.append(f'<span style="{style_str}"{tooltip}>{aa}</span>')

            # Space every 10 residues
            if (i - line_start + 1) % 10 == 0 and i < line_end - 1:
                html_parts.append(" ")

        html_parts.append("\n")

    html_parts.append("</div>")
    return "".join(html_parts)


# ===========================================================================
#  0D. Chromatogram Plot Generation (Plotly)
# ===========================================================================

def plot_chromatogram_from_cqa(
    cqa: Dict[str, Any],
    gradient_slope: float = 15.0,
    c_salt_start: float = 50.0,
) -> "plotly.graph_objects.Figure":
    """
    Reconstruct a chromatogram visualization from CQA peak data.

    Uses Gaussian peak shapes centered at retention times with widths
    derived from FWHM. Overlays a linear salt gradient.

    Parameters
    ----------
    cqa           : CQA dict with 'peaks', 'area_pct', 'resolution'
    gradient_slope: Salt gradient slope (mM/min)
    c_salt_start  : Starting salt concentration (mM)

    Returns
    -------
    plotly Figure object
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    peaks = cqa.get("peaks", {})
    area_pct = cqa.get("area_pct", {})

    if not peaks:
        return None

    # Determine time range
    all_rts = [p["rt_min"] for p in peaks.values()]
    t_max = max(all_rts) * 1.4 + 2.0  # Extra space after last peak
    t = np.linspace(0, t_max, 2000)

    # Gaussian peak reconstruction
    # FWHM = 2*sqrt(2*ln2)*sigma => sigma = FWHM / 2.355
    colors = {
        "Acidic": "#EF4444",  # Red
        "Main":   "#3B82F6",  # Blue
        "Basic":  "#10B981",  # Green
    }
    labels = {
        "Acidic": "Acidic Variant",
        "Main":   "Main Peak",
        "Basic":  "Basic Variant",
    }

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=("UV Absorbance (280 nm)", "Salt Gradient"),
    )

    total_signal = np.zeros_like(t)

    for comp in ("Acidic", "Main", "Basic"):
        if comp not in peaks:
            continue
        pk = peaks[comp]
        rt = pk["rt_min"]
        fwhm = pk["fwhm_min"]
        sigma = fwhm / 2.355
        pct = area_pct.get(comp, 33.0)

        # Amplitude scaled by area percentage (arbitrary mAU units)
        amplitude = pct / 100.0 * (1.0 / (sigma * np.sqrt(2 * np.pi)))
        signal = amplitude * np.exp(-0.5 * ((t - rt) / sigma) ** 2)
        total_signal += signal

        fig.add_trace(
            go.Scatter(
                x=t, y=signal,
                mode="lines",
                name=f"{labels[comp]} (RT={rt:.1f} min)",
                line=dict(color=colors[comp], width=2),
                fill="tozeroy",
                opacity=0.6,
            ),
            row=1, col=1,
        )

    # Total signal (envelope)
    fig.add_trace(
        go.Scatter(
            x=t, y=total_signal,
            mode="lines",
            name="Total UV280",
            line=dict(color="#1F2937", width=2.5, dash="dot"),
        ),
        row=1, col=1,
    )

    # Salt gradient
    c_salt = c_salt_start + gradient_slope * t
    fig.add_trace(
        go.Scatter(
            x=t, y=c_salt,
            mode="lines",
            name="NaCl Gradient",
            line=dict(color="#F59E0B", width=2),
        ),
        row=2, col=1,
    )

    # Resolution annotations
    resolution = cqa.get("resolution", {})
    for label, rs in resolution.items():
        quality = "Baseline" if rs >= 1.5 else ("Partial" if rs >= 0.8 else "Overlap")
        pair = label.split("_vs_")
        if len(pair) == 2 and pair[0] in peaks and pair[1] in peaks:
            mid_rt = (peaks[pair[0]]["rt_min"] + peaks[pair[1]]["rt_min"]) / 2
            mid_y = max(total_signal) * 0.85
            fig.add_annotation(
                x=mid_rt, y=mid_y,
                text=f"Rs={rs:.2f}<br>({quality})",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor="#64748B",
                font=dict(size=10, color="#334155"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#E2E8F0",
                borderwidth=1,
                row=1, col=1,
            )

    fig.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig.update_yaxes(title_text="Absorbance (mAU)", row=1, col=1)
    fig.update_yaxes(title_text="NaCl (mM)", row=2, col=1)

    _apply_pharma_theme(fig,
        height=550,
        title="IEX Chromatogram — Competitive SMA Multi-Component",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plot_estimated_chromatogram(
    variants: Dict[str, Any],
    gradient_slope: float = 15.0,
    c_salt_start: float = 50.0,
    column_length: float = 0.25,
    velocity: float = 5.75e-4,
    porosity: float = 0.37,
) -> "plotly.graph_objects.Figure":
    """
    Generate an estimated chromatogram from SMA parameters (no CADET needed).

    Uses simplified analytical retention model:
        RT ≈ dead_time + (c_elution - c_start) / gradient_slope
    where c_elution ≈ Lambda * (Keq)^(1/nu) is the salt concentration
    at which the protein desorbs from the resin.

    Parameters
    ----------
    variants       : Three-variant parameter dict from Tool 1
    gradient_slope : Salt gradient slope (mM/min)
    c_salt_start   : Starting salt concentration (mM)
    column_length  : Column length (m)
    velocity       : Linear velocity (m/s)
    porosity       : Column porosity

    Returns
    -------
    plotly Figure object
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    lambda_ = variants.get("lambda_", 1200.0)
    kd = variants.get("kd", 1000.0)

    # Dead time (minutes)
    dead_time = (column_length / velocity) / 60.0  # ~7.25 min

    # Estimate elution salt concentration for each variant
    # Simplified: c_elution ≈ Lambda * (Keq * phase_ratio)^(-1/nu)
    # Phase ratio F = (1-eps)/eps
    F = (1 - porosity) / porosity
    # Convert lambda from mol/m3 to mM: lambda is already in mol/m3, 1 mol/m3 = 1 mM
    lambda_mM = lambda_

    estimated_peaks = {}
    for comp in ("acidic", "main", "basic"):
        if comp not in variants:
            continue
        v = variants[comp]
        nu = v["nu"]
        ka = v["ka"]
        Keq = ka / kd

        # Analytical estimate: at elution, K_eff ≈ 1/F
        # K_eff = Keq * (Lambda/c_salt)^nu => c_salt = Lambda * (Keq*F)^(1/nu)
        # v7.3: dynamic_factor=1.0 for standard-range ka (1-5).
        # With ka~3, Keq~0.003, equilibrium formula directly gives RT~17 min.
        dynamic_factor = 1.0
        if Keq > 0 and nu > 0:
            c_elution_eq = lambda_mM * (Keq * F) ** (1.0 / nu)
            c_elution = c_elution_eq * dynamic_factor
        else:
            c_elution = c_salt_start + 100
        c_elution = max(c_salt_start + 5.0, min(c_elution, c_salt_start + 600.0))

        rt = dead_time + (c_elution - c_salt_start) / gradient_slope
        # FWHM estimate: ~0.5-1.5 min for typical mAb IEX
        fwhm = 0.3 + 0.05 * nu

        estimated_peaks[comp] = {"rt_min": rt, "fwhm_min": fwhm, "c_elution": c_elution}

    if not estimated_peaks:
        return None

    # Build Plotly figure using same Gaussian reconstruction
    all_rts = [p["rt_min"] for p in estimated_peaks.values()]
    t_max = max(all_rts) * 1.4 + 2.0
    t = np.linspace(0, t_max, 2000)

    colors = {"acidic": "#EF4444", "main": "#3B82F6", "basic": "#10B981"}
    labels = {"acidic": "Acidic (est.)", "main": "Main (est.)", "basic": "Basic (est.)"}
    fractions = variants.get("c_fractions", [0.12, 0.80, 0.08])
    frac_map = {"acidic": fractions[0], "main": fractions[1], "basic": fractions[2]}

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=("Estimated UV Absorbance (280 nm)", "Salt Gradient"),
    )

    total_signal = np.zeros_like(t)

    for comp in ("acidic", "main", "basic"):
        if comp not in estimated_peaks:
            continue
        pk = estimated_peaks[comp]
        rt = pk["rt_min"]
        fwhm = pk["fwhm_min"]
        sigma = fwhm / 2.355
        pct = frac_map.get(comp, 0.33)

        amplitude = pct * (1.0 / (sigma * np.sqrt(2 * np.pi)))
        signal = amplitude * np.exp(-0.5 * ((t - rt) / sigma) ** 2)
        total_signal += signal

        fig.add_trace(
            go.Scatter(
                x=t, y=signal,
                mode="lines",
                name=f"{labels[comp]} (RT~{rt:.1f} min, {pk['c_elution']:.0f} mM)",
                line=dict(color=colors[comp], width=2),
                fill="tozeroy",
                opacity=0.6,
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=t, y=total_signal,
            mode="lines",
            name="Total (est.)",
            line=dict(color="#1F2937", width=2.5, dash="dot"),
        ),
        row=1, col=1,
    )

    # Salt gradient
    c_salt = c_salt_start + gradient_slope * t
    fig.add_trace(
        go.Scatter(
            x=t, y=c_salt,
            mode="lines",
            name="NaCl Gradient",
            line=dict(color="#F59E0B", width=2),
        ),
        row=2, col=1,
    )

    fig.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig.update_yaxes(title_text="Absorbance (mAU)", row=1, col=1)
    fig.update_yaxes(title_text="NaCl (mM)", row=2, col=1)

    _apply_pharma_theme(fig,
        height=550,
        title="IEX Chromatogram — Analytical Estimate (PropertyMapper v7.3)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Add "ESTIMATED" watermark
    fig.add_annotation(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text="ESTIMATED",
        showarrow=False,
        font=dict(size=40, color="rgba(200,200,200,0.3)"),
    )

    return fig


# ===========================================================================
#  0E. ML Predictor & SHAP Explainability Rendering
# ===========================================================================

def render_ml_shap_panel(intent: Dict[str, Any]) -> None:
    """
    Render the AI Predictor & SHAP Explainability panel in Streamlit.

    Runs the PyTorch MLP, displays predictions vs mechanistic model,
    shows training loss curves, and renders SHAP waterfall/summary plots.
    Silently skips if torch/shap are not installed.
    """
    # Early exit if PyTorch or SHAP not available — no error, no panel
    try:
        import torch as _tc
    except ImportError:
        return
    try:
        import shap as _shap_check
    except ImportError:
        return

    with st.expander("AI Predictor & SHAP Explainability", expanded=True):
        try:
            from src.ml_predictor import (
                predict_and_explain, plot_shap_waterfall, plot_shap_summary,
                FEATURE_NAMES,
            )
        except ImportError as e:
            st.info(
                f"ML/SHAP libraries not fully available ({e}). "
                "Install with: `pip install torch shap matplotlib`"
            )
            return

        with st.spinner("Training PyTorch MLP on biophysical features and computing SHAP explanations..."):
            try:
                ml_result = predict_and_explain(intent)
            except Exception as e:
                st.error(f"ML prediction failed: {e}")
                return

        pred = ml_result["prediction"]
        info = ml_result["model_info"]
        history = ml_result["training_history"]
        shap_result = ml_result["shap_result"]

        # -- Model info card ------------------------------------------------
        st.markdown("#### PyTorch MLP Model (v2.0 RT-Targeted)")
        mi1, mi2, mi3, mi4, mi5 = st.columns(5)
        with mi1:
            st.metric("Architecture", info["architecture"].split("(")[0])
        with mi2:
            st.metric("Training Samples", info["n_training_samples"])
        with mi3:
            st.metric("Train MSE", f"{info['final_train_mse']:.6f}")
        with mi4:
            st.metric("Val MSE", f"{info['final_val_mse']:.6f}")
        with mi5:
            st.metric("Target RT", info.get("target_rt_window", "15-20 min"))

        # -- Training loss curve --------------------------------------------
        st.markdown("#### Training Loss Curve")
        import plotly.graph_objects as go
        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]

        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            x=epochs, y=train_losses,
            mode="lines", name="Train MSE",
            line=dict(color=PLOTLY_COLORS[0], width=2),
        ))
        loss_fig.add_trace(go.Scatter(
            x=epochs, y=val_losses,
            mode="lines", name="Validation MSE",
            line=dict(color=PLOTLY_COLORS[3], width=2, dash="dash"),
        ))
        _apply_pharma_theme(loss_fig,
            title_text="",
            height=300,
            xaxis_title="Epoch",
            yaxis_title="MSE Loss",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        )
        st.plotly_chart(loss_fig, use_container_width=True)

        # -- Predictions comparison -----------------------------------------
        st.markdown("#### ML Prediction (Driving Parameters)")
        est_rt = pred.get('estimated_rt_min', 0.0)
        rt_color = PLOTLY_COLORS[1] if 15.0 <= est_rt <= 20.0 else PLOTLY_COLORS[3]
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown(f"""
            <div class="cqa-card">
                <div class="cqa-title">ML Predicted ka</div>
                <div class="cqa-value">{pred['ka']:.4f}</div>
                <div style="font-size:0.8rem; color:#64748B;">
                    PyTorch MLP v2.0
                </div>
            </div>
            """, unsafe_allow_html=True)
        with pc2:
            st.markdown(f"""
            <div class="cqa-card">
                <div class="cqa-title">ML Predicted nu</div>
                <div class="cqa-value">{pred['nu']:.3f}</div>
                <div style="font-size:0.8rem; color:#64748B;">
                    PyTorch MLP v2.0
                </div>
            </div>
            """, unsafe_allow_html=True)
        with pc3:
            st.markdown(f"""
            <div class="cqa-card">
                <div class="cqa-title">Estimated RT</div>
                <div class="cqa-value" style="color:{rt_color};">{est_rt:.1f} min</div>
                <div style="font-size:0.8rem; color:#64748B;">
                    Target: 15-20 min
                </div>
            </div>
            """, unsafe_allow_html=True)

        # -- Input features -------------------------------------------------
        st.markdown("#### Input Features")
        features = ml_result["features"]
        feat_cols = st.columns(len(features))
        for col, (fname, fval) in zip(feat_cols, zip(FEATURE_NAMES, features)):
            with col:
                st.metric(fname, f"{fval:.2f}")

        # -- SHAP Waterfall plots -------------------------------------------
        st.markdown("#### SHAP Feature Attribution")
        st.markdown(
            "SHAP values show how each input feature pushes the prediction "
            "above or below the model's baseline. Positive values (red) increase "
            "the output; negative values (blue) decrease it."
        )

        shap_c1, shap_c2 = st.columns(2)
        with shap_c1:
            try:
                fig_ka = plot_shap_waterfall(shap_result, output_idx=0, output_name="ka")
                st.pyplot(fig_ka, use_container_width=True)
            except Exception as e:
                st.caption(f"(ka waterfall unavailable: {e})")
        with shap_c2:
            try:
                fig_nu = plot_shap_waterfall(shap_result, output_idx=1, output_name="nu")
                st.pyplot(fig_nu, use_container_width=True)
            except Exception as e:
                st.caption(f"(nu waterfall unavailable: {e})")

        # -- SHAP Summary (feature importance) ------------------------------
        st.markdown("#### SHAP Feature Importance (Global)")
        sum_c1, sum_c2 = st.columns(2)
        with sum_c1:
            try:
                # Need multiple samples for summary plot
                from src.ml_predictor import get_trained_model, compute_shap_values as csv_fn
                model, X_train, _ = get_trained_model()
                # Explain a subset of training data
                X_subset = X_train[np.random.choice(len(X_train), min(30, len(X_train)), replace=False)]
                shap_multi = csv_fn(model, X_train, X_subset, max_background=50)
                fig_sum_ka = plot_shap_summary(shap_multi, output_idx=0, output_name="ka")
                st.pyplot(fig_sum_ka, use_container_width=True)
            except Exception as e:
                st.caption(f"(ka summary unavailable: {e})")
        with sum_c2:
            try:
                fig_sum_nu = plot_shap_summary(shap_multi, output_idx=1, output_name="nu")
                st.pyplot(fig_sum_nu, use_container_width=True)
            except Exception as e:
                st.caption(f"(nu summary unavailable: {e})")


# ===========================================================================
#  1. Page Configuration & Theme
# ===========================================================================

st.set_page_config(
    page_title="ProtePilot Platform",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Enterprise Color Palette (Monochrome) -----------------------------------
SLATE   = "#64748B"
DARK    = "#0F172A"
ACCENT  = "#334155"       # Dark slate (was blue)
SUCCESS = "#10B981"       # T.PASS  — unified with ui_colors.py
WARN    = "#F59E0B"       # T.CAUTION — unified with ui_colors.py
ERROR   = "#EF4444"       # T.FAIL  — unified with ui_colors.py
BG      = "#FFFFFF"
CARD_BG = "#FFFFFF"
SIDEBAR_BG = "#F8FAFC"   # Light gray (monochrome)
SIDEBAR_TEXT = "#334155"
BORDER  = "#E2E8F0"

# -- Unified Plotly color-blind-friendly palette (muted scientific) ----------
# Import chart palette from central color system; PLOTLY_COLORS kept as alias
from src.ui_colors import CHART_PALETTE as _CHART_PAL, CHART_COLORS as _CHART_CLR
PLOTLY_COLORS = [
    _CHART_CLR["primary"],    # Blue   #3B82F6
    _CHART_CLR["secondary"],  # Green  #10B981
    _CHART_CLR["accent"],     # Amber  #F59E0B
    _CHART_CLR["danger"],     # Red    #EF4444
    _CHART_CLR["purple"],     # Purple #8B5CF6
    _CHART_CLR["muted"],      # Gray   #64748B
    "#0891B2",                # Cyan   (extended)
    "#BE185D",                # Magenta (extended)
]

PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155"),
    title_font=dict(size=15, color=DARK, family="Inter, system-ui, sans-serif"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=48, r=24, t=56, b=48),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        showgrid=True, gridcolor="#F1F5F9", gridwidth=1,
        linecolor=BORDER, linewidth=1,
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#F1F5F9", gridwidth=1,
        linecolor=BORDER, linewidth=1,
        zeroline=False,
    ),
)


def _apply_pharma_theme(fig, **overrides):
    """Apply the ProtePilot enterprise Plotly theme to any figure."""
    layout = {**PLOTLY_LAYOUT_DEFAULTS, **overrides}
    # Dark mode: override font, grid, and legend colors for readability
    if st.session_state.get("dark_mode", False):
        _dk_font = dict(family="Inter, system-ui, sans-serif", size=12, color="#E2E8F0")
        _dk_title = dict(size=15, color="#F8FAFC", family="Inter, system-ui, sans-serif")
        _dk_axis = dict(
            showgrid=True, gridcolor="#334155", gridwidth=1,
            linecolor="#475569", linewidth=1, zeroline=False,
            tickfont=dict(color="#94A3B8"),
        )
        _dk_legend = dict(
            bgcolor="rgba(37,51,69,0.9)", bordercolor="#334155",
            borderwidth=1, font=dict(size=11, color="#E2E8F0"),
        )
        layout.update(font=_dk_font, title_font=_dk_title,
                      xaxis=_dk_axis, yaxis=_dk_axis, legend=_dk_legend)
        # Re-apply any user overrides on top of dark defaults
        layout.update(overrides)
    fig.update_layout(**layout)
    return fig


# -- Unified Design System CSS Theme (v2) ------------------------------------
st.markdown("""
<style>
  /* === Hide Streamlit chrome === */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header[data-testid="stHeader"] {background: rgba(255,255,255,0.95); backdrop-filter: blur(8px); border-bottom: 1px solid #E2E8F0;}

  /* === 1. Surfaces === */
  [data-testid="stSidebar"] {
    background-color: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
  }
  .stApp {
    background-color: #FFFFFF !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
  }

  /* === 2. Global Typography === */
  html, body, [class*="css"], .stMarkdown p, .stMarkdown, .stText,
  p, span, label, li, td, th {
    color: #0F172A !important;
  }
  .block-container {padding-top: 2.5rem; max-width: 1400px; color: #0F172A;}

  /* === Sidebar text === */
  section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stMarkdown {color: #334155 !important;}
  section[data-testid="stSidebar"] .stRadio label {
    color: #334155 !important; font-weight: 600; font-size: 0.85rem;
    padding: 4px 0;
  }
  section[data-testid="stSidebar"] .stRadio label:hover {color: #0F172A !important;}
  section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
  section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"] {
    color: #0F172A !important; font-weight: 700;
  }
  section[data-testid="stSidebar"] hr {border-color: #E2E8F0 !important; opacity: 0.7;}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stNumberInput label {color: #64748B !important; font-size: 0.8rem;}
  section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {color: #64748B !important;}
  section[data-testid="stSidebar"] .stButton > button {color: #334155 !important; border-color: #CBD5E1 !important; background: #FFFFFF !important;}
  section[data-testid="stSidebar"] .stButton > button:hover {background: #E2E8F0 !important; color: #0F172A !important; box-shadow: 0 4px 10px rgba(0,0,0,0.06) !important;}
  section[data-testid="stSidebar"] .stExpander summary {color: #334155 !important;}

  /* === 3. Inputs (faint gray bg, transparent border) === */
  .stTextInput input,
  .stNumberInput input,
  .stSelectbox div[data-baseweb="select"] > div,
  .stFileUploader,
  .stTextArea textarea {
    background-color: #F1F5F9 !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    color: #0F172A !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    transition: all 0.2s ease;
  }

  /* === 4. Input Hover & Focus === */
  .stTextInput input:hover,
  .stNumberInput input:hover,
  .stSelectbox div[data-baseweb="select"] > div:hover,
  .stFileUploader:hover,
  .stTextArea textarea:hover {
    border-color: #94A3B8 !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.04) !important;
  }
  .stTextInput input:focus,
  .stNumberInput input:focus,
  .stSelectbox div[data-baseweb="select"] > div:focus-within,
  .stTextArea textarea:focus {
    border-color: #0F172A !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.04) !important;
  }

  /* === 5. Dropdown & Label Colors === */
  div[data-baseweb="select"] span,
  div[data-baseweb="input"] input {
    color: #0F172A !important;
  }
  .stSlider label, .stNumberInput label, .stTextInput label,
  .stSelectbox label, .stFileUploader label, .stRadio label,
  .stCheckbox label, .stTextArea label {color: #334155 !important;}

  /* === Sidebar inputs === */
  section[data-testid="stSidebar"] .stTextInput input,
  section[data-testid="stSidebar"] div[data-baseweb="select"] span,
  section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    color: #0F172A !important;
    background-color: #F1F5F9 !important;
  }

  /* === 6. Standard Buttons (outline) === */
  .stButton > button {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    cursor: pointer !important;
    transition: all 0.2s ease;
    padding: 0.45rem 1.2rem !important;
    min-width: 120px !important;
    height: auto !important;
    line-height: 1.5 !important;
    white-space: nowrap !important;
  }
  .stButton > button:hover {
    border-color: #94A3B8 !important;
    color: #0F172A !important;
    background-color: #F8FAFC !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08) !important;
  }
  .stButton > button:active {
    background-color: #E2E8F0 !important;
    color: #0F172A !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.08) !important;
  }

  /* === Primary Buttons (filled gray, darker on hover) === */
  /* Cover ALL Streamlit primary button selector variants across versions */
  .stButton > button[kind="primary"],
  .stButton > button[data-testid="baseButton-primary"],
  .stButton > button[data-testid="stFormSubmitButton"],
  div[data-testid="stBaseButton-primary"] > button,
  div[data-testid="baseButton-primary"] > button,
  .stFormSubmitButton > button,
  button[kind="primaryFormSubmit"] {
    background-color: #CBD5E1 !important;
    border: 1px solid #CBD5E1 !important;
    color: #0F172A !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    padding: 0.45rem 1.2rem !important;
    min-width: 120px !important;
    height: auto !important;
    line-height: 1.5 !important;
    white-space: nowrap !important;
  }
  .stButton > button[kind="primary"]:hover,
  .stButton > button[data-testid="baseButton-primary"]:hover,
  .stButton > button[data-testid="stFormSubmitButton"]:hover,
  div[data-testid="stBaseButton-primary"] > button:hover,
  div[data-testid="baseButton-primary"] > button:hover,
  .stFormSubmitButton > button:hover,
  button[kind="primaryFormSubmit"]:hover {
    background-color: #94A3B8 !important;
    border-color: #94A3B8 !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
    transform: translateY(-1px);
  }
  .stButton > button[kind="primary"]:active,
  .stButton > button[data-testid="baseButton-primary"]:active,
  .stButton > button[data-testid="stFormSubmitButton"]:active,
  div[data-testid="stBaseButton-primary"] > button:active,
  div[data-testid="baseButton-primary"] > button:active,
  .stFormSubmitButton > button:active,
  button[kind="primaryFormSubmit"]:active {
    background-color: #64748B !important;
    color: #FFFFFF !important;
    transform: translateY(0);
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.12) !important;
  }

  /* === JS-tagged primary buttons (fallback for any Streamlit version) === */
  button.pp-primary-btn {
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    padding: 0.45rem 1.2rem !important;
    min-width: 120px !important;
    height: auto !important;
    line-height: 1.5 !important;
    white-space: nowrap !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
  }
  button.pp-primary-btn:hover {
    background-color: #94A3B8 !important;
    border-color: #94A3B8 !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
    transform: translateY(-1px) !important;
  }
  button.pp-primary-btn:active {
    background-color: #64748B !important;
    color: #FFFFFF !important;
    transform: translateY(0) !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.12) !important;
  }
  [data-theme="dark"] button.pp-primary-btn {
    background-color: #334155 !important;
    border: 1px solid #475569 !important;
    color: #F8FAFC !important;
  }
  [data-theme="dark"] button.pp-primary-btn:hover {
    background-color: #475569 !important;
    border-color: #64748B !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
  }

  /* === Danger Button (Reset AI — red, irreversible action warning) === */
  button.pp-danger-btn {
    background-color: #FEE2E2 !important;
    border: 1px solid #FECACA !important;
    color: #B91C1C !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 2px 8px rgba(185,28,28,0.10) !important;
    padding: 0.45rem 1.2rem !important;
    min-width: 120px !important;
    height: auto !important;
    line-height: 1.5 !important;
    white-space: nowrap !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
  }
  button.pp-danger-btn:hover {
    background-color: #FECACA !important;
    border-color: #F87171 !important;
    color: #991B1B !important;
    box-shadow: 0 4px 16px rgba(185,28,28,0.18) !important;
    transform: translateY(-1px) !important;
  }
  button.pp-danger-btn:active {
    background-color: #FCA5A5 !important;
    border-color: #EF4444 !important;
    color: #7F1D1D !important;
    transform: translateY(0) !important;
    box-shadow: inset 0 1px 3px rgba(185,28,28,0.15) !important;
  }
  [data-theme="dark"] button.pp-danger-btn {
    background-color: #451A1A !important;
    border: 1px solid #7F1D1D !important;
    color: #FCA5A5 !important;
    box-shadow: 0 2px 8px rgba(185,28,28,0.15) !important;
  }
  [data-theme="dark"] button.pp-danger-btn:hover {
    background-color: #5C1D1D !important;
    border-color: #B91C1C !important;
    color: #FEE2E2 !important;
    box-shadow: 0 4px 16px rgba(185,28,28,0.25) !important;
    transform: translateY(-1px) !important;
  }

  /* === Download Buttons (outline, darker fill on hover) === */
  .stDownloadButton > button {
    border: 1px solid #CBD5E1 !important;
    color: #334155 !important;
    background-color: transparent !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease;
  }
  .stDownloadButton > button:hover {
    background-color: #E2E8F0 !important;
    color: #0F172A !important;
    border-color: #94A3B8 !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08) !important;
  }
  .stDownloadButton > button:active {
    background-color: #CBD5E1 !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.08) !important;
  }

  /* === Disabled Buttons === */
  .stButton > button:disabled,
  .stDownloadButton > button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
  }

  /* === Metric Cards === */
  div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
    transition: box-shadow 0.15s ease;
  }
  div[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
  }
  div[data-testid="stMetric"] label {
    font-size: 0.75rem !important; font-weight: 600 !important;
    color: #64748B !important; text-transform: uppercase; letter-spacing: 0.04em;
  }
  div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.6rem !important; font-weight: 700 !important; color: #0F172A !important;
  }
  div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important; font-weight: 600 !important;
  }

  /* === Expanders === */
  details[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important; border-radius: 8px !important;
    background: #FFFFFF; margin-bottom: 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
  }
  details[data-testid="stExpander"] summary {
    font-weight: 600 !important; color: #334155 !important; font-size: 0.9rem;
  }

  /* === Preserve Streamlit alert/status colors === */
  .stAlert p, .stAlert span, .stAlert div {color: inherit !important;}
  .stSpinner > div {color: #64748B !important;}
  div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span {color: #334155 !important;}
  .stProgress > div {color: #64748B !important;}
  div[data-baseweb="notification"] {color: inherit !important;}

  /* === Dataframes === */
  .stDataFrame {border-radius: 8px; overflow: hidden; border: 1px solid #E2E8F0;}

  /* === Tabs === */
  .stTabs [data-baseweb="tab"] {font-weight: 600; font-size: 0.85rem; color: #64748B;}
  .stTabs [data-baseweb="tab"][aria-selected="true"] {color: #0F172A; border-bottom-color: #0F172A;}

  /* === Custom component classes === */
  .copilot-header {font-size: 1.6rem; font-weight: 700; color: #0F172A !important; margin-bottom: 2px; letter-spacing: -0.02em;}
  .copilot-sub {font-size: 0.85rem; color: #64748B !important; margin-bottom: 1.2rem; font-weight: 400;}
  .status-pill {display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;}
  .status-ok   {background: #E2E8F0; color: #0F172A;}
  .status-warn {background: #FFFBEB; color: #F59E0B;}
  .status-err  {background: #FEF2F2; color: #EF4444;}
  .cqa-card {background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04);}
  .cqa-title {font-size: 0.75rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;}
  .cqa-value {font-size: 1.6rem; font-weight: 700; color: #0F172A;}
  .cqa-unit {font-size: 0.8rem; color: #94A3B8;}
  .fasta-card {background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;}
  .fasta-title {font-size: 0.85rem; font-weight: 600; color: #334155; margin-bottom: 0.5rem;}
  .liability-card {background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.8rem; margin: 0.4rem 0;}
  .liability-title {font-size: 0.8rem; font-weight: 600; color: #64748B; margin-bottom: 0.3rem;}
  .liability-value {font-size: 1.0rem; font-weight: 700; color: #0F172A;}

  /* === Section dividers === */
  hr {border: none; border-top: 1px solid #E2E8F0; margin: 1.5rem 0;}

  /* === Page titles === */
  h2 {color: #0F172A !important; font-weight: 700 !important; letter-spacing: -0.02em; font-size: 1.6rem !important;}
  h3 {color: #0F172A !important; font-weight: 700 !important; font-size: 1.15rem !important;}

  /* === Dark/Light toggle (injected into toolbar via JS) === */
  .pp-theme-toggle {
    display: inline-flex; align-items: center; gap: 6px;
    background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 20px;
    padding: 3px 12px; font-size: 0.75rem; font-weight: 600; color: #334155;
    cursor: pointer; transition: all 0.25s ease; user-select: none;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-right: 8px;
    line-height: 1.4; white-space: nowrap; vertical-align: middle;
  }
  .pp-theme-toggle:hover {background: #E2E8F0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
  .pp-theme-toggle svg {width: 14px; height: 14px; flex-shrink: 0;}
  /* Dark mode overrides for the toggle itself */
  [data-theme="dark"] .pp-theme-toggle {
    background: #334155; border-color: #475569; color: #E2E8F0;
  }
  [data-theme="dark"] .pp-theme-toggle:hover {background: #475569;}
</style>
""", unsafe_allow_html=True)

# -- Dark / Light Mode Toggle (JS via components.html → parent DOM) -----------
# Uses localStorage for persistence; st.components.v1.html for JS execution.
# The toggle is injected into the Streamlit header (right side, next to Run/Deploy).
import streamlit.components.v1 as _components

# Dark mode CSS (in st.markdown — this works; only <script> gets stripped)
st.markdown("""
<style id="pp-dark-mode-css">
  /* === Dark Mode Overrides (activated by data-theme="dark" on .stApp) === */
  /* Palette: base=#1E293B  card=#253345  elevated=#334155  border=#334155 */
  /* NOTE: .stApp itself carries data-theme attr → compound selector, no space */
  .stApp[data-theme="dark"] {background-color: #1E293B !important;}
  [data-theme="dark"] [data-testid="stSidebar"] {background-color: #1E293B !important; border-right: 1px solid #334155 !important;}
  [data-theme="dark"] header[data-testid="stHeader"] {background: rgba(30,41,59,0.95) !important; border-bottom: 1px solid #334155 !important;}
  [data-theme="dark"] .block-container {color: #E2E8F0 !important;}

  [data-theme="dark"] html, [data-theme="dark"] body,
  [data-theme="dark"] [class*="css"], [data-theme="dark"] .stMarkdown p,
  [data-theme="dark"] .stMarkdown, [data-theme="dark"] .stText,
  [data-theme="dark"] p, [data-theme="dark"] span, [data-theme="dark"] label,
  [data-theme="dark"] li, [data-theme="dark"] td, [data-theme="dark"] th {color: #E2E8F0 !important;}

  [data-theme="dark"] section[data-testid="stSidebar"] p,
  [data-theme="dark"] section[data-testid="stSidebar"] span,
  [data-theme="dark"] section[data-testid="stSidebar"] label,
  [data-theme="dark"] section[data-testid="stSidebar"] .stMarkdown {color: #CBD5E1 !important;}
  [data-theme="dark"] section[data-testid="stSidebar"] .stRadio label {color: #CBD5E1 !important;}
  [data-theme="dark"] section[data-testid="stSidebar"] .stRadio label:hover {color: #F8FAFC !important;}
  [data-theme="dark"] section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
  [data-theme="dark"] section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"] {color: #F8FAFC !important; font-weight: 700;}
  [data-theme="dark"] section[data-testid="stSidebar"] hr {border-color: #334155 !important;}
  [data-theme="dark"] section[data-testid="stSidebar"] .stSelectbox label,
  [data-theme="dark"] section[data-testid="stSidebar"] .stNumberInput label {color: #94A3B8 !important;}
  [data-theme="dark"] section[data-testid="stSidebar"] .stCaption,
  [data-theme="dark"] section[data-testid="stSidebar"] small {color: #94A3B8 !important;}

  [data-theme="dark"] .stSlider label, [data-theme="dark"] .stNumberInput label,
  [data-theme="dark"] .stTextInput label, [data-theme="dark"] .stSelectbox label,
  [data-theme="dark"] .stFileUploader label, [data-theme="dark"] .stRadio label,
  [data-theme="dark"] .stCheckbox label, [data-theme="dark"] .stTextArea label {color: #CBD5E1 !important;}

  [data-theme="dark"] .stTextInput input, [data-theme="dark"] .stNumberInput input,
  [data-theme="dark"] .stSelectbox div[data-baseweb="select"] > div,
  [data-theme="dark"] .stFileUploader, [data-theme="dark"] .stTextArea textarea {
    background-color: #253345 !important; border: 1px solid #334155 !important; color: #E2E8F0 !important;
  }
  [data-theme="dark"] .stTextInput input:hover, [data-theme="dark"] .stNumberInput input:hover,
  [data-theme="dark"] .stSelectbox div[data-baseweb="select"] > div:hover,
  [data-theme="dark"] .stTextArea textarea:hover {
    border-color: #64748B !important; background-color: #253345 !important;
  }
  [data-theme="dark"] div[data-baseweb="select"] span,
  [data-theme="dark"] div[data-baseweb="input"] input {color: #E2E8F0 !important;}
  [data-theme="dark"] section[data-testid="stSidebar"] .stTextInput input,
  [data-theme="dark"] section[data-testid="stSidebar"] div[data-baseweb="select"] span,
  [data-theme="dark"] section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    color: #E2E8F0 !important; background-color: #253345 !important;
  }

  [data-theme="dark"] .stButton > button {
    background-color: #253345 !important; border: 1px solid #334155 !important; color: #E2E8F0 !important;
  }
  [data-theme="dark"] .stButton > button:hover {
    background-color: #334155 !important; border-color: #64748B !important;
    color: #F8FAFC !important; box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
  }
  [data-theme="dark"] .stButton > button:active {background-color: #475569 !important;}
  [data-theme="dark"] .stButton > button[kind="primary"],
  [data-theme="dark"] .stButton > button[data-testid="baseButton-primary"],
  [data-theme="dark"] .stButton > button[data-testid="stFormSubmitButton"],
  [data-theme="dark"] div[data-testid="stBaseButton-primary"] > button,
  [data-theme="dark"] div[data-testid="baseButton-primary"] > button,
  [data-theme="dark"] .stFormSubmitButton > button,
  [data-theme="dark"] button[kind="primaryFormSubmit"] {
    background-color: #334155 !important; border: 1px solid #475569 !important; color: #F8FAFC !important;
  }
  [data-theme="dark"] .stButton > button[kind="primary"]:hover,
  [data-theme="dark"] .stButton > button[data-testid="baseButton-primary"]:hover,
  [data-theme="dark"] .stButton > button[data-testid="stFormSubmitButton"]:hover,
  [data-theme="dark"] div[data-testid="stBaseButton-primary"] > button:hover,
  [data-theme="dark"] div[data-testid="baseButton-primary"] > button:hover,
  [data-theme="dark"] .stFormSubmitButton > button:hover,
  [data-theme="dark"] button[kind="primaryFormSubmit"]:hover {
    background-color: #475569 !important; border-color: #64748B !important;
    color: #FFFFFF !important; box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
  }
  [data-theme="dark"] .stDownloadButton > button {
    border: 1px solid #334155 !important; color: #E2E8F0 !important; background-color: transparent !important;
  }
  [data-theme="dark"] .stDownloadButton > button:hover {
    background-color: #334155 !important; border-color: #64748B !important; color: #F8FAFC !important;
  }
  [data-theme="dark"] section[data-testid="stSidebar"] .stButton > button {
    color: #E2E8F0 !important; border-color: #334155 !important; background: #253345 !important;
  }
  [data-theme="dark"] section[data-testid="stSidebar"] .stButton > button:hover {
    background: #334155 !important; color: #F8FAFC !important;
  }

  [data-theme="dark"] div[data-testid="stMetric"] {background: #253345 !important; border: 1px solid #334155 !important;}
  [data-theme="dark"] div[data-testid="stMetric"]:hover {box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;}
  [data-theme="dark"] div[data-testid="stMetric"] label {color: #94A3B8 !important;}
  [data-theme="dark"] div[data-testid="stMetric"] div[data-testid="stMetricValue"] {color: #F8FAFC !important;}
  [data-theme="dark"] details[data-testid="stExpander"] {border: 1px solid #334155 !important; background: #253345 !important;}
  [data-theme="dark"] details[data-testid="stExpander"] summary {color: #CBD5E1 !important;}
  [data-theme="dark"] .stDataFrame {border: 1px solid #334155 !important;}
  [data-theme="dark"] .stTabs [data-baseweb="tab"] {color: #94A3B8 !important;}
  [data-theme="dark"] .stTabs [data-baseweb="tab"][aria-selected="true"] {color: #F8FAFC !important; border-bottom-color: #F8FAFC !important;}
  [data-theme="dark"] div[data-testid="stChatMessage"] p,
  [data-theme="dark"] div[data-testid="stChatMessage"] span {color: #E2E8F0 !important;}

  [data-theme="dark"] .copilot-header {color: #F8FAFC !important;}
  [data-theme="dark"] .copilot-sub {color: #94A3B8 !important;}
  [data-theme="dark"] .status-ok {background: #334155 !important; color: #E2E8F0 !important;}
  [data-theme="dark"] .status-warn {background: rgba(245,158,11,0.15) !important; color: #FCD34D !important;}
  [data-theme="dark"] .status-err {background: rgba(239,68,68,0.15) !important; color: #FCA5A5 !important;}
  [data-theme="dark"] .cqa-card {background: #253345 !important; border: 1px solid #334155 !important;}
  [data-theme="dark"] .cqa-title {color: #94A3B8 !important;}
  [data-theme="dark"] .cqa-value {color: #F8FAFC !important;}
  [data-theme="dark"] .cqa-unit {color: #64748B !important;}
  [data-theme="dark"] .fasta-card {background: #253345 !important; border: 1px solid #334155 !important;}
  [data-theme="dark"] .fasta-title {color: #CBD5E1 !important;}
  [data-theme="dark"] .liability-card {background: #253345 !important; border: 1px solid #334155 !important;}
  [data-theme="dark"] .liability-title {color: #94A3B8 !important;}
  [data-theme="dark"] .liability-value {color: #F8FAFC !important;}
  [data-theme="dark"] hr {border-top: 1px solid #334155 !important;}
  [data-theme="dark"] h1 {color: #F8FAFC !important;}
  [data-theme="dark"] h2 {color: #F8FAFC !important;}
  [data-theme="dark"] h3 {color: #E2E8F0 !important;}
  [data-theme="dark"] h4, [data-theme="dark"] h5, [data-theme="dark"] h6 {color: #CBD5E1 !important;}
  [data-theme="dark"] section[data-testid="stSidebar"] .stExpander summary {color: #CBD5E1 !important;}

  /* ====================================================================
     Dark Mode: Inline HTML Background Overrides (attribute selectors)
     These override hardcoded light backgrounds in st.markdown() HTML
     ==================================================================== */

  /* --- Neutral light backgrounds → dark card surface #253345 --- */
  [data-theme="dark"] .stMarkdown div[style*="background:#F9FAFB"],
  [data-theme="dark"] .stMarkdown div[style*="background:#f9fafb"],
  [data-theme="dark"] .stMarkdown div[style*="background: #F9FAFB"],
  [data-theme="dark"] .stMarkdown div[style*="background:#f0f4f8"],
  [data-theme="dark"] .stMarkdown div[style*="background:#F8FAFC"],
  [data-theme="dark"] .stMarkdown div[style*="background:#f8fafc"],
  [data-theme="dark"] .stMarkdown div[style*="background:#f8f9fa"],
  [data-theme="dark"] .stMarkdown div[style*="background:#F1F5F9"],
  [data-theme="dark"] .stMarkdown div[style*="background:#f3f4f6"],
  [data-theme="dark"] .stMarkdown div[style*="background: #f3f4f6"],
  [data-theme="dark"] .stMarkdown div[style*="background:#FAFBFC"],
  [data-theme="dark"] .stMarkdown div[style*="background:#FFFFFF"],
  [data-theme="dark"] .stMarkdown div[style*="background: #FFFFFF"],
  [data-theme="dark"] .stMarkdown div[style*="background:white"],
  [data-theme="dark"] .stMarkdown div[style*="background: white"] {
    background: #253345 !important;
    color: #E2E8F0 !important;
    border-color: #334155 !important;
  }

  /* --- Blue-tinted light backgrounds → dark blue tint --- */
  [data-theme="dark"] .stMarkdown div[style*="background:#F0F4FF"],
  [data-theme="dark"] .stMarkdown div[style*="background:#f0f4ff"],
  [data-theme="dark"] .stMarkdown div[style*="background:#EFF6FF"],
  [data-theme="dark"] .stMarkdown div[style*="background:#eff6ff"] {
    background: rgba(59,130,246,0.12) !important;
    color: #E2E8F0 !important;
    border-color: #334155 !important;
  }

  /* --- Green-tinted light backgrounds → dark green tint --- */
  [data-theme="dark"] .stMarkdown div[style*="background:#F0FFF4"],
  [data-theme="dark"] .stMarkdown div[style*="background:#F0FDF4"],
  [data-theme="dark"] .stMarkdown div[style*="background:#f0fff4"],
  [data-theme="dark"] .stMarkdown div[style*="background:#f0fdf4"],
  [data-theme="dark"] .stMarkdown div[style*="background:#ECFDF5"],
  [data-theme="dark"] .stMarkdown div[style*="background:#ecfdf5"] {
    background: rgba(16,185,129,0.12) !important;
    color: #E2E8F0 !important;
    border-color: #334155 !important;
  }

  /* --- Amber/Yellow-tinted light backgrounds → dark amber tint --- */
  [data-theme="dark"] .stMarkdown div[style*="background:#FFFBEB"],
  [data-theme="dark"] .stMarkdown div[style*="background:#FEF3C7"],
  [data-theme="dark"] .stMarkdown div[style*="background:#fef3c7"] {
    background: rgba(245,158,11,0.12) !important;
    color: #E2E8F0 !important;
    border-color: #334155 !important;
  }

  /* --- Red-tinted light backgrounds → dark red tint --- */
  [data-theme="dark"] .stMarkdown div[style*="background:#FEF2F2"],
  [data-theme="dark"] .stMarkdown div[style*="background:#FEE2E2"],
  [data-theme="dark"] .stMarkdown div[style*="background:#fee2e2"],
  [data-theme="dark"] .stMarkdown div[style*="background:#FECACA"],
  [data-theme="dark"] .stMarkdown div[style*="background:#fecaca"] {
    background: rgba(239,68,68,0.12) !important;
    color: #E2E8F0 !important;
    border-color: #334155 !important;
  }

  /* --- Purple-tinted light backgrounds → dark purple tint --- */
  [data-theme="dark"] .stMarkdown div[style*="background:#F5F3FF"],
  [data-theme="dark"] .stMarkdown div[style*="background:#f5f3ff"],
  [data-theme="dark"] .stMarkdown div[style*="background:#EDE9FE"] {
    background: rgba(139,92,246,0.12) !important;
    color: #E2E8F0 !important;
    border-color: #334155 !important;
  }

  /* --- Orange-tinted light backgrounds → dark orange tint --- */
  [data-theme="dark"] .stMarkdown div[style*="background:#FFF7ED"],
  [data-theme="dark"] .stMarkdown div[style*="background:#fff7ed"] {
    background: rgba(249,115,22,0.12) !important;
    color: #E2E8F0 !important;
    border-color: #334155 !important;
  }

  /* --- Inline HTML: Dark text colors → light text --- */
  [data-theme="dark"] .stMarkdown div[style*="color:#0F172A"],
  [data-theme="dark"] .stMarkdown div[style*="color: #0F172A"],
  [data-theme="dark"] .stMarkdown div[style*="color:#334155"],
  [data-theme="dark"] .stMarkdown div[style*="color: #334155"],
  [data-theme="dark"] .stMarkdown div[style*="color:#1E293B"],
  [data-theme="dark"] .stMarkdown p[style*="color:#0F172A"],
  [data-theme="dark"] .stMarkdown p[style*="color:#334155"] {
    color: #E2E8F0 !important;
  }
  [data-theme="dark"] .stMarkdown span[style*="color:#0F172A"],
  [data-theme="dark"] .stMarkdown span[style*="color: #0F172A"],
  [data-theme="dark"] .stMarkdown span[style*="color:#334155"],
  [data-theme="dark"] .stMarkdown span[style*="color: #334155"],
  [data-theme="dark"] .stMarkdown span[style*="color:#1E293B"] {
    color: #E2E8F0 !important;
  }

  /* --- Inline HTML: Muted text → lighter muted --- */
  [data-theme="dark"] .stMarkdown div[style*="color:#64748B"],
  [data-theme="dark"] .stMarkdown div[style*="color: #64748B"],
  [data-theme="dark"] .stMarkdown span[style*="color:#64748B"],
  [data-theme="dark"] .stMarkdown span[style*="color: #64748B"] {
    color: #94A3B8 !important;
  }

  /* --- Inline HTML: Warning text colors → legible on dark --- */
  [data-theme="dark"] .stMarkdown div[style*="color:#92400E"],
  [data-theme="dark"] .stMarkdown span[style*="color:#92400E"] {
    color: #FCD34D !important;
  }

  /* --- Inline badge spans with light backgrounds --- */
  [data-theme="dark"] .stMarkdown span[style*="background:#fef3c7"],
  [data-theme="dark"] .stMarkdown span[style*="background:#FEF3C7"] {
    background: rgba(245,158,11,0.2) !important;
    color: #FCD34D !important;
  }
  [data-theme="dark"] .stMarkdown span[style*="background:#FEE2E2"],
  [data-theme="dark"] .stMarkdown span[style*="background:#fee2e2"] {
    background: rgba(239,68,68,0.2) !important;
    color: #FCA5A5 !important;
  }
  [data-theme="dark"] .stMarkdown span[style*="background:#FECACA"],
  [data-theme="dark"] .stMarkdown span[style*="background:#fecaca"] {
    background: rgba(239,68,68,0.25) !important;
    color: #FCA5A5 !important;
  }

  /* --- Inline border colors → dark borders --- */
  [data-theme="dark"] .stMarkdown div[style*="border:1px solid #E2E8F0"],
  [data-theme="dark"] .stMarkdown div[style*="border: 1px solid #E2E8F0"] {
    border-color: #334155 !important;
  }
  [data-theme="dark"] .stMarkdown div[style*="border:1px solid #FDE68A"] {
    border-color: rgba(245,158,11,0.4) !important;
  }
  [data-theme="dark"] .stMarkdown div[style*="border:1px solid #C4B5FD"] {
    border-color: rgba(139,92,246,0.4) !important;
  }

  /* --- Progress bar track --- */
  [data-theme="dark"] .stMarkdown div[style*="background: #f3f4f6"] {
    background: #334155 !important;
  }

  /* ====================================================================
     Dark Mode: Glide Data Grid (DataFrame) Theme Variables
     ==================================================================== */
  [data-theme="dark"] .stDataFrame {
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    overflow: hidden;
    --gdg-bg-cell: #253345;
    --gdg-bg-cell-medium: #2C3E52;
    --gdg-bg-header: #1A2535;
    --gdg-bg-header-has: #253345;
    --gdg-bg-header-hovered: #334155;
    --gdg-text-dark: #E2E8F0;
    --gdg-text-medium: #94A3B8;
    --gdg-text-light: #64748B;
    --gdg-text-header: #CBD5E1;
    --gdg-text-header-selected: #F8FAFC;
    --gdg-border-color: #334155;
    --gdg-horizontal-border-color: #334155;
    --gdg-accent-color: #3B82F6;
    --gdg-accent-light: rgba(59,130,246,0.15);
    --gdg-bg-bubble: #334155;
    --gdg-bg-bubble-selected: #475569;
    --gdg-link-color: #60A5FA;
    --gdg-cell-horizontal-padding: 8px;
    --gdg-cell-vertical-padding: 3px;
  }

  /* --- DataFrame header row dark styling (non-canvas fallback) --- */
  [data-theme="dark"] .stDataFrame [data-testid="glideDataEditor"] {
    background: #1A2535 !important;
  }

  /* ====================================================================
     Dark Mode: Alert / Info / Success / Warning / Error boxes
     ==================================================================== */
  [data-theme="dark"] .stAlert {
    background: #253345 !important;
    border-color: #334155 !important;
  }
  [data-theme="dark"] .stAlert p,
  [data-theme="dark"] .stAlert span,
  [data-theme="dark"] .stAlert div {
    color: #E2E8F0 !important;
  }

  /* ====================================================================
     Dark Mode: Additional catch-all for remaining light surfaces
     ==================================================================== */
  [data-theme="dark"] .stMarkdown table {
    background: #253345 !important;
    border-color: #334155 !important;
  }
  [data-theme="dark"] .stMarkdown table th {
    background: #1A2535 !important;
    color: #CBD5E1 !important;
    border-color: #334155 !important;
  }
  [data-theme="dark"] .stMarkdown table td {
    background: #253345 !important;
    color: #E2E8F0 !important;
    border-color: #334155 !important;
  }
  [data-theme="dark"] .stMarkdown code {
    background: #334155 !important;
    color: #E2E8F0 !important;
  }
  [data-theme="dark"] .stMarkdown pre {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
  }
</style>
""", unsafe_allow_html=True)

# Dark mode JS: apply data-theme attribute based on session_state.
# The toggle itself is a native Streamlit widget in the sidebar (see sidebar section).
# This JS only applies the CSS attribute on the parent document.
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
_dm_theme = "dark" if st.session_state.dark_mode else "light"
_components.html(f"""
<script>
(function() {{
  var doc = window.parent.document;
  var root = doc.querySelector('.stApp');
  if (root) root.setAttribute('data-theme', '{_dm_theme}');

  /* --- Force primary button style via JS (covers all Streamlit versions) --- */
  function stylePrimaryButtons() {{
    // Target every possible primary button wrapper/element
    var selectors = [
      'button[kind="primary"]',
      '[data-testid="baseButton-primary"] button',
      '[data-testid="stBaseButton-primary"] button',
      '[data-testid="baseButton-primary"]',
      '[data-testid="stBaseButton-primary"]',
      '.stFormSubmitButton button',
      '[data-testid="stBaseButton-primaryFormSubmit"] button',
      '[data-testid="stFormSubmitButton"] button',
      'button[kind="primaryFormSubmit"]'
    ];
    selectors.forEach(function(sel) {{
      var els = doc.querySelectorAll(sel);
      els.forEach(function(el) {{
        if (el.tagName === 'BUTTON' || el.querySelector('button')) {{
          var btn = el.tagName === 'BUTTON' ? el : el.querySelector('button');
          if (!btn) return;
          // Check if this is the danger "Reset" button
          var txt = (btn.textContent || '').trim();
          var isDark = root && root.getAttribute('data-theme') === 'dark';
          if (txt === 'Reset AI to Baseline') {{
            btn.classList.remove('pp-primary-btn');
            btn.classList.add('pp-danger-btn');
            btn.style.setProperty('background-color', isDark ? 'rgba(239,68,68,0.15)' : '#FEE2E2', 'important');
            btn.style.setProperty('border', isDark ? '1px solid #7F1D1D' : '1px solid #FECACA', 'important');
            btn.style.setProperty('color', isDark ? '#FCA5A5' : '#B91C1C', 'important');
          }} else {{
            btn.style.setProperty('background-color', isDark ? '#334155' : '#CBD5E1', 'important');
            btn.style.setProperty('border', isDark ? '1px solid #475569' : '1px solid #CBD5E1', 'important');
            btn.style.setProperty('color', isDark ? '#F8FAFC' : '#0F172A', 'important');
            btn.classList.add('pp-primary-btn');
          }}
        }}
      }});
    }});
  }}

  // Run immediately and on every DOM mutation
  stylePrimaryButtons();
  new MutationObserver(function() {{
    var r = doc.querySelector('.stApp');
    if (r && r.getAttribute('data-theme') !== '{_dm_theme}')
      r.setAttribute('data-theme', '{_dm_theme}');
    stylePrimaryButtons();
  }}).observe(doc.body, {{ childList: true, subtree: true }});
}})();
</script>
""", height=0)


# ===========================================================================
#  2. Session State (M10: Workspace-based isolation)
# ===========================================================================

if "manager" not in st.session_state:
    st.session_state.manager = PharmaAgentManager(workspace="data", engine_dir="engine")
if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "uploaded_fasta_text" not in st.session_state:
    st.session_state.uploaded_fasta_text = None

# M10: Workspace Store
ws_store = WorkspaceStore.from_session_state(st.session_state)

# M10: Expert Label Store
if "label_store" not in st.session_state:
    st.session_state.label_store = ExpertLabelStore()

# M10: Continuous Learning Engine
if "cl_engine" not in st.session_state:
    st.session_state.cl_engine = ContinuousLearningEngine()

# Backward compat: ensure messages list exists for legacy code
if "messages" not in st.session_state:
    st.session_state.messages = []

# M10: Active tab tracking
if "active_main_tab" not in st.session_state:
    st.session_state.active_main_tab = "Copilot Chat"

# M11: Bispecific mode pending input
if "pending_bispecific" not in st.session_state:
    st.session_state.pending_bispecific = None

# M12: Dynamic stoichiometry chains
if "stoich_chains" not in st.session_state:
    st.session_state.stoich_chains = []  # list of {"sequence", "copy_number", "name"}
if "pending_assembly" not in st.session_state:
    st.session_state.pending_assembly = None

# M13: Glycoform profile & PK
if "glycoform_profile" not in st.session_state:
    st.session_state.glycoform_profile = "standard_cho"

# M14: Generative optimization
if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None
if "pending_optimize" not in st.session_state:
    st.session_state.pending_optimize = None

# M15: Data Foundation & Wet-Lab Model Training
if "wetlab_model_trained" not in st.session_state:
    st.session_state.wetlab_model_trained = False
if "wetlab_training_metrics" not in st.session_state:
    st.session_state.wetlab_training_metrics = None
if "wetlab_dataset_info" not in st.session_state:
    st.session_state.wetlab_dataset_info = None
if "wetlab_csv_data" not in st.session_state:
    st.session_state.wetlab_csv_data = None

# M28: PLM Embedding Model
if "plm_model_trained" not in st.session_state:
    st.session_state.plm_model_trained = False
if "plm_training_metrics" not in st.session_state:
    st.session_state.plm_training_metrics = None
if "plm_csv_data" not in st.session_state:
    st.session_state.plm_csv_data = None

# M16: HT Screening & Potency
if "ht_screening_results" not in st.session_state:
    st.session_state.ht_screening_results = None
if "ht_screening_csv" not in st.session_state:
    st.session_state.ht_screening_csv = None
if "potency_model_trained" not in st.session_state:
    st.session_state.potency_model_trained = False
if "potency_training_metrics" not in st.session_state:
    st.session_state.potency_training_metrics = None

# M17: Formulation Digital Twin
if "formulation_buffer_ph" not in st.session_state:
    st.session_state.formulation_buffer_ph = 6.0
if "formulation_buffer_type" not in st.session_state:
    st.session_state.formulation_buffer_type = "histidine"
if "formulation_excipients" not in st.session_state:
    st.session_state.formulation_excipients = []

# M18: Executive Report cache
if "_exec_report_bytes" not in st.session_state:
    st.session_state["_exec_report_bytes"] = None
if "_exec_report_name" not in st.session_state:
    st.session_state["_exec_report_name"] = None

# M21: LLM Copilot chat history
if "copilot_messages" not in st.session_state:
    st.session_state.copilot_messages = []
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "copilot_show_details" not in st.session_state:
    st.session_state.copilot_show_details = False
if "copilot_execution_log" not in st.session_state:
    st.session_state.copilot_execution_log = []

# M21: Global molecule context (preserved across page navigation)
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None


# ===========================================================================
#  2B-pre. Molecule-Bound State Invalidation
# ===========================================================================
# Keys that are computed FOR a specific molecule and MUST be cleared when the
# molecule changes.  User assets (trained models, uploaded CSVs, glycoform
# preference) are deliberately NOT listed here.

_MOLECULE_BOUND_KEYS = [
    # ── Global molecule context ──
    "last_intent",
    # ── Process Development ──
    "upstream_result", "upstream_result_dict", "_up_logged_hash",
    "doe_result_ds", "doe_result_dict", "_doe_logged_hash",
    "scaleup_result",
    "cogs_result", "cogs_result_dict",
    # ── Clinical / Immunogenicity ──
    "ada_result", "ada_sequence_used",
    # ── Discovery / Advisory ──
    "vml_result", "advisory_panel_result",
    "_sasa_results", "_sasa_chain_names",
    "ht_parsed_candidates",
    # ── Executive Report artefacts ──
    "_exec_report_bytes", "_exec_report_name",
    "_json_report_bytes", "_json_report_name",
    # ── eCTD ──
    "ectd_markdown", "ectd_docx_bytes",
    # ── Analytical QC acceptance criteria ──
    "qc_sec_min", "qc_cesds_min", "qc_acidic_max", "qc_main_min",
    # ── Bispecific ──
    "pending_bispecific",
    # ── Optimization ──
    "optimization_result", "pending_optimize",
    # ── OOD / Training Center (molecule-specific baselines) ──
    "ood_computed_baseline",
    # ── Assembly / FASTA chain cache ──
    "uploaded_fasta_text", "fasta_assembly_chains", "fasta_assembly_confirmed",
    "_fasta_assembly_file",
]


def _invalidate_molecule_bound_state(ws_store=None, new_intent=None) -> None:
    """Clear ALL state tied to a specific molecule — both session_state AND workspace.

    Called when a new molecule is assembled or submitted, so that stale
    results from the previous molecule never bleed into the new analysis.

    User assets (trained models, uploaded data, glycoform preference) are
    intentionally preserved.

    Parameters
    ----------
    ws_store : WorkspaceStore, optional
        If provided, also clears workspace-level stale fields.
    new_intent : dict, optional
        If provided, sets last_intent to this value (the new molecule)
        instead of None, so downstream pages see the correct molecule.
    """
    for key in _MOLECULE_BOUND_KEYS:
        if key in st.session_state:
            st.session_state[key] = None

    # Reset assembly chain state to proper defaults (list/bool, not None)
    st.session_state.stoich_chains = []
    st.session_state.show_assembly_chains = False

    # Set last_intent to the new molecule (if provided) instead of None,
    # so that downstream pages immediately see the correct context.
    if new_intent is not None:
        st.session_state.last_intent = new_intent

    # Clear workspace-level stale fields
    if ws_store is not None:
        ws = ws_store.get_active()
        if ws:
            ws["bispecific_result"] = None
            ws["dev_result"] = None
            ws["ml_prediction"] = None
            ws["validation_report"] = None
            ws["export_ready"] = False
            # Clear stale analysis_cache fields that are molecule-specific
            cache = ws.get("analysis_cache")
            if cache and isinstance(cache, dict):
                for _ck in ("dev_result", "bispecific_result", "ms_characterization",
                            "pk_result", "optimization_result", "analytical_qc",
                            "cqa", "sim_summary", "sim_elapsed", "variants",
                            "glycoform_impact"):
                    cache[_ck] = None

    log.info("Molecule-bound state invalidated (session + workspace)")


# ===========================================================================
#  2B. Molecule Gate Helper (DRY — replaces 5 identical gate blocks)
# ===========================================================================

def enforce_molecule_state(ws_store) -> None:
    """Hard gate: call st.stop() if no molecule sequence is loaded.

    Used by Developability, Process, Analytical, Preclinical, and CMC pages
    to require a molecule before rendering the page content.
    """
    _gate_ws = ws_store.get_active() if ws_store else None
    _gate_intent = (_gate_ws.get("intent") or {}) if _gate_ws else {}
    if isinstance(_gate_intent, dict) and _gate_intent.get("sequence"):
        return
    _last_intent = st.session_state.get("last_intent") or {}
    if isinstance(_last_intent, dict) and _last_intent.get("sequence"):
        return
    st.warning(
        "**No molecule loaded.** Please upload a FASTA sequence in the "
        "**Molecule Setup** page to unlock this module."
    )
    st.stop()


# ===========================================================================
#  3. Mock LLM Intent Parser (Regex / Keyword + FASTA)
# ===========================================================================

def parse_intent(user_input: str) -> Optional[Dict[str, Any]]:
    """Extract protein chromatography intent from natural language or FASTA."""

    # Priority 1: FASTA sequence parsing
    _is_fasta = is_fasta_input(user_input)
    log.info("parse_intent: input len=%d, is_fasta=%s, starts_with='%s'",
             len(user_input), _is_fasta, user_input[:50].replace('\n', '\\n'))
    if _is_fasta:
        fasta_result = parse_fasta_sequence(user_input)
        log.info("parse_intent: parse_fasta_sequence returned %s",
                 "valid dict" if fasta_result else "None")
        if fasta_result is not None:
            header_text = user_input.split("\n")[0].lower() if user_input.startswith(">") else ""
            deam_match = re.search(r'(?:deam|deamidation)\s*[=:.]?\s*(\d+)', header_text)
            ox_match = re.search(r'(?:ox|oxidation)\s*[=:.]?\s*(\d+)', header_text)
            deam = int(deam_match.group(1)) if deam_match else 1
            ox = int(ox_match.group(1)) if ox_match else 1

            # Run liability analysis on all chains
            chain_analyses = []
            for chain in fasta_result.get("chains", []):
                liab = scan_sequence_liabilities(chain["sequence"])
                cdrs = identify_cdrs_heuristic(chain["sequence"], chain["chain_type"])
                chain_analyses.append({
                    "name": chain["name"],
                    "chain_type": chain["chain_type"],
                    "sequence": chain["sequence"],
                    "length": len(chain["sequence"]),
                    "liabilities": liab,
                    "cdrs": cdrs,
                })

            # M12: Build assembly_chains for MS characterization
            _asm_chains = [
                {"sequence": ch["sequence"], "copy_number": 1,
                 "name": ch["name"], "chain_type": ch["chain_type"]}
                for ch in fasta_result.get("chains", [])
            ]

            # Use user-confirmed assembly stoichiometry if available
            _user_asm = st.session_state.get("fasta_assembly_chains")
            if _user_asm and st.session_state.get("fasta_assembly_confirmed"):
                # Apply user-specified copy numbers
                for c in _asm_chains:
                    for uc in _user_asm:
                        if uc.get("name") == c.get("name"):
                            c["copy_number"] = uc.get("copy_number", 1)
                            break
            else:
                # Fallback: auto-detect HC/LC and default to 2×HC + 2×LC
                _n_hc = sum(1 for c in _asm_chains if c.get("chain_type", "").upper() in ("HC", "HEAVY"))
                _n_lc = sum(1 for c in _asm_chains if c.get("chain_type", "").upper() in ("LC", "LIGHT"))
                if _n_hc > 0 and _n_lc > 0:
                    for c in _asm_chains:
                        c["copy_number"] = 2

            # Compute stoichiometric properties
            _stoich = _compute_stoichiometric_properties(_asm_chains)

            # Use stoichiometric assembly MW for multi-chain molecules
            # (accounts for 2×HC + 2×LC + glycans) instead of single-chain MW
            _mw_display = fasta_result["mw"]
            if _stoich and _stoich.get("mw_kda_assembled"):
                _mw_display = _stoich["mw_kda_assembled"]

            # ── Phase 1: Molecule Classification & Feature Registry ──
            # Classify molecule type BEFORE any analysis begins.
            # This drives downstream feature selection, model routing,
            # risk weights, and validation recommendations.
            try:
                from src.molecule_classifier import classify_molecule as _classify_mol
                _classification = _classify_mol(
                    sequence=fasta_result["sequence"],
                    chains=fasta_result.get("chains", []),
                    assembly_chains=_asm_chains,
                    name=fasta_result["name"],
                )
                _mol_class = _classification.effective_class.value
                _mol_class_info = _classification.to_dict()
                log.info(
                    "MoleculeClassifier: %s → %s (confidence=%s, evidence=%s)",
                    fasta_result["name"], _mol_class,
                    _classification.confidence,
                    "; ".join(_classification.evidence[:2]),
                )
            except Exception as _cls_err:
                log.warning("MoleculeClassifier failed: %s — defaulting to unknown", _cls_err)
                _mol_class = "unknown"
                _mol_class_info = {"molecule_class": "unknown", "confidence": "Low",
                                   "evidence": [f"Classification error: {_cls_err}"]}

            # Compute ALL features through the centralized registry.
            # This is the single source of truth — no module should
            # compute features independently.
            try:
                from src.feature_registry import compute_features as _compute_feats
                _feature_set = _compute_feats(
                    sequence=fasta_result["sequence"],
                    molecule_class=_mol_class,
                    chains=_asm_chains,
                )
                _feature_dict = _feature_set.to_dict()
                _liability_summary = _feature_set.liability_summary()
                log.info(
                    "FeatureRegistry: %d features computed (ML vector dim=%d)",
                    len(_feature_set.features), len(_feature_set.ml_vector()),
                )
            except Exception as _feat_err:
                log.warning("FeatureRegistry failed: %s — continuing without", _feat_err)
                _feature_set = None
                _feature_dict = {}
                _liability_summary = {}

            # Use feature_registry values (assembly-aware, glycan-corrected)
            # when available; fall back to fasta_result for robustness.
            _fv = _feature_set.features if _feature_set else {}
            _fr_pI = _fv["pI"].value if "pI" in _fv else fasta_result["pI"]
            _fr_mw = _fv["mw_kda"].value if "mw_kda" in _fv else _mw_display
            _fr_hydro = _fv["hydrophobicity"].value if "hydrophobicity" in _fv else fasta_result["hydrophobicity"]
            _fr_gravy = _fv["gravy"].value if "gravy" in _fv else fasta_result["gravy"]
            _fr_seq_len = _fv["seq_length"].value if "seq_length" in _fv else fasta_result["seq_length"]
            _fr_deam = _fv["deam_sites"].value if "deam_sites" in _fv else deam
            _fr_ox = _fv["ox_sites"].value if "ox_sites" in _fv else ox
            _fr_cys = _fv["cysteine_count"].value if "cysteine_count" in _fv else fasta_result["sequence"].upper().count("C")
            _fr_acidic = _fv["acidic_residues"].value if "acidic_residues" in _fv else 0
            _fr_basic = _fv["basic_residues"].value if "basic_residues" in _fv else 0

            return {
                "name": fasta_result["name"],
                "pI": _fr_pI,
                "mw": _fr_mw,
                "mw_single_chain": fasta_result["mw"],  # preserve raw MW for reference
                "hydrophobicity": _fr_hydro,
                "pH_working": 7.0,
                "deam_sites": _fr_deam,
                "ox_sites": _fr_ox,
                "acidic_residues": _fr_acidic,
                "basic_residues": _fr_basic,
                "cysteine_count": _fr_cys,
                "gradient_slope": 15.0,
                "source": "fasta",
                "sequence": fasta_result["sequence"],
                "seq_length": _fr_seq_len,
                "gravy": _fr_gravy,
                "chains": fasta_result.get("chains", []),
                "chain_analyses": chain_analyses,
                "assembly_chains": _asm_chains,
                "stoichiometric_properties": _stoich,
                # Phase 1: New fields — molecule routing & unified features
                "molecule_class": _mol_class,
                "molecule_class_info": _mol_class_info,
                "feature_set": _feature_dict,
                "feature_set_obj": _feature_set,  # Live object for module consumption
                "liability_summary": _liability_summary,
            }

    # Priority 2: Regex-based natural language parsing
    text = user_input.lower().strip()

    pi_match = re.search(r'pi\s*[=:.]?\s*(\d+\.?\d*)', text)
    if not pi_match:
        pi_match = re.search(r'isoelectric\s*(?:point)?\s*[=:.]?\s*(\d+\.?\d*)', text)
    pI = float(pi_match.group(1)) if pi_match else None
    if pI is None:
        return None

    mw_match = re.search(r'(?:mw|molecular\s*weight)\s*[=:.]?\s*(\d+\.?\d*)', text)
    mw = float(mw_match.group(1)) if mw_match else 150.0

    h_match = re.search(r'(?:hydro(?:phobicity)?)\s*[=:.]?\s*(\d+\.?\d*)', text)
    hydro = float(h_match.group(1)) if h_match else 0.35

    ph_match = re.search(r'(?:^|[^a-z])ph\s*[=:.]?\s*(\d+\.?\d*)', text)
    pH = float(ph_match.group(1)) if ph_match else 7.0
    if ph_match and pi_match and ph_match.start() <= pi_match.start() + 1:
        pH = 7.0

    deam_match = re.search(r'(?:deamidation|deamidated|deam)\s*[=:.]?\s*(?:sites?\s*[=:.]?\s*)?(\d+)', text)
    if not deam_match:
        deam_match = re.search(r'(?<!\.)(\d+)\s*(?:sites?)?\s*(?:of\s+)?(?:deamidation|deamidated|deam)', text)
    deam = int(deam_match.group(1)) if deam_match else 1

    ox_match = re.search(r'(?:oxidation|oxidized|ox)\s*[=:.]?\s*(?:sites?\s*[=:.]?\s*)?(\d+)', text)
    if not ox_match:
        ox_match = re.search(r'(?<!\.)(\d+)\s*(?:sites?)?\s*(?:of\s+)?(?:oxidation|oxidized|ox)', text)
    ox = int(ox_match.group(1)) if ox_match else 1

    name_match = re.search(r'(?:name|molecule|protein)\s*[=:.]?\s*["\'\']?(\w+)', text)
    name = name_match.group(1) if name_match else f"mAb_Chat_{uuid.uuid4().hex[:6]}"

    grad_match = re.search(r'(?:gradient|slope)\s*[=:.]?\s*(\d+\.?\d*)', text)
    gradient = float(grad_match.group(1)) if grad_match else 15.0

    return {
        "name": name, "pI": pI, "mw": mw, "hydrophobicity": hydro,
        "gravy": None,  # No sequence available — upstream will use default
        "pH_working": pH, "deam_sites": deam, "ox_sites": ox,
        "gradient_slope": gradient, "source": "text",
    }


# ===========================================================================
#  3B. Render Characterization Panel
# ===========================================================================

def render_characterization_panel(intent: Dict[str, Any]) -> None:
    """
    Render a professional Antibody Characterization & Liabilities expander
    in Streamlit, showing per-chain stats, motif counts, CDR table,
    and annotated sequence visualization.
    """
    chain_analyses = intent.get("chain_analyses", [])
    if not chain_analyses:
        return

    with st.expander("Antibody Characterization & Liabilities", expanded=True):

        # -- Per-chain summary cards ----------------------------------------
        st.markdown("#### Chain Summary")
        cols = st.columns(min(len(chain_analyses), 4))
        for idx, ca in enumerate(chain_analyses):
            col = cols[idx % len(cols)]
            with col:
                chain_label = f"{ca['chain_type']} ({ca['name']})" if ca['chain_type'] != 'unknown' else ca['name']
                st.markdown(f"""
                <div class="liability-card">
                    <div class="liability-title">{chain_label}</div>
                    <div style="font-size:0.85rem; color:#334155;">
                        Length: <b>{ca['length']}</b> aa<br>
                        CDRs identified: <b>{len(ca['cdrs'])}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # -- Liability overview per chain -----------------------------------
        st.markdown("#### Developability Liabilities")
        for ca in chain_analyses:
            liab = ca["liabilities"]
            chain_label = f"{ca['chain_type']} — {ca['name']}" if ca['chain_type'] != 'unknown' else ca['name']
            st.markdown(f"**{chain_label}** ({ca['length']} aa)")

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Met (oxidation)", liab["met_count"])
            with mc2:
                st.metric("Trp (oxidation)", liab["trp_count"])
            with mc3:
                st.metric("N-Glyco motifs", liab["n_glyco_count"])
            with mc4:
                st.metric("Asp-Pro clips", liab["dp_count"])

            mc5, mc6, mc7, mc8 = st.columns(4)
            with mc5:
                st.metric("Deamidation hotspots", liab["deamidation_count"])
            with mc6:
                st.metric("Isomerization risk", liab["isomerization_count"])
            with mc7:
                st.metric("Cys (disulfide)", liab["cys_count"])
            with mc8:
                st.metric("Acidic (D+E) / Basic (K+R+H)",
                          f"{liab['acidic_count']} / {liab['basic_count']}")

            # Contextual interpretation of liabilities
            _interp_items = []
            _met = liab["met_count"]
            _trp = liab["trp_count"]
            _deam = liab["deamidation_count"]
            _nglyco = liab["n_glyco_count"]
            _dp = liab["dp_count"]
            _iso = liab["isomerization_count"]

            if _met + _trp > 4:
                _interp_items.append(
                    f"<b>Oxidation:</b> {_met} Met + {_trp} Trp sites — elevated oxidation risk. "
                    "Met oxidation can reduce binding affinity (especially in CDRs) and increase "
                    "aggregation. Consider Met→Leu mutation at non-critical positions."
                )
            elif _met + _trp > 0:
                _interp_items.append(
                    f"<b>Oxidation:</b> {_met} Met + {_trp} Trp sites — typical for IgG. "
                    "Monitor under accelerated stability; CDR-located sites are higher priority."
                )

            if _deam > 3:
                _interp_items.append(
                    f"<b>Deamidation:</b> {_deam} NG/NS hotspots — significant charge heterogeneity "
                    "expected. Deamidation introduces acidic variants that may affect potency over "
                    "shelf life. Screen formulation pH (target pH 5.5–6.0 to minimize)."
                )
            elif _deam > 0:
                _interp_items.append(
                    f"<b>Deamidation:</b> {_deam} NG/NS hotspot(s) — manageable with optimized "
                    "formulation. Track by icIEF or peptide mapping."
                )

            if _nglyco > 1:
                _interp_items.append(
                    f"<b>N-Glycosylation:</b> {_nglyco} NX[ST] motifs — multiple glycosylation sites "
                    "increase glycoform heterogeneity. Non-canonical sites (outside Fc Asn297) may "
                    "affect binding and require extended glycan profiling."
                )

            if _dp > 0:
                _interp_items.append(
                    f"<b>Asp-Pro clipping:</b> {_dp} DP motif(s) — acid-labile peptide bond prone to "
                    "backbone cleavage under low-pH conditions (viral inactivation, Protein A elution). "
                    "Monitor fragmentation by CE-SDS."
                )

            if _iso > 2:
                _interp_items.append(
                    f"<b>Isomerization:</b> {_iso} Asp isomerization sites — can create succinimide "
                    "intermediates and isoAsp, affecting charge profile and potentially potency."
                )

            if _interp_items:
                _interp_html = "".join(
                    f'<div style="margin:2px 0;">{item}</div>' for item in _interp_items
                )
                st.markdown(f"""
                <div style="padding:8px 12px; margin:6px 0; border-radius:6px;
                            background:#FFFBEB; border:1px solid #FDE68A;
                            font-size:0.8rem; color:#78350F;">
                    {_interp_html}
                </div>
                """, unsafe_allow_html=True)

            # Risk flags
            if liab["risk_flags"]:
                for flag in liab["risk_flags"]:
                    st.markdown(f'<span class="status-pill status-warn">{flag}</span>',
                                unsafe_allow_html=True)

            st.markdown("---")

        # -- CDR table ------------------------------------------------------
        st.markdown("#### CDR Regions (Heuristic)")
        cdr_rows = []
        for ca in chain_analyses:
            for cdr in ca["cdrs"]:
                cdr_rows.append({
                    "Chain": f"{ca['chain_type']} ({ca['name']})",
                    "CDR": cdr["name"],
                    "Start": cdr["start"] + 1,  # 1-indexed
                    "End": cdr["end"],
                    "Length": cdr["end"] - cdr["start"],
                    "Sequence": cdr["sequence"],
                })
        if cdr_rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(cdr_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No CDR regions identified (requires conserved Cys/Trp anchors).")

        # -- CDR-liability cross-reference ----------------------------------
        st.markdown("#### CDR Liability Cross-Reference")
        for ca in chain_analyses:
            cdrs = ca["cdrs"]
            liab = ca["liabilities"]
            if not cdrs:
                continue
            chain_label = f"{ca['chain_type']} ({ca['name']})"
            for cdr in cdrs:
                cdr_start, cdr_end = cdr["start"], cdr["end"]
                cdr_liabs = []
                for pos in liab.get("met_positions", []):
                    if cdr_start <= pos < cdr_end:
                        cdr_liabs.append(f"Met-Ox @{pos+1}")
                for pos in liab.get("trp_positions", []):
                    if cdr_start <= pos < cdr_end:
                        cdr_liabs.append(f"Trp-Ox @{pos+1}")
                for motif in liab.get("n_glyco_motifs", []):
                    if cdr_start <= motif["pos"] < cdr_end:
                        cdr_liabs.append(f"N-Glyco ({motif['motif']}) @{motif['pos']+1}")
                for pos in liab.get("dp_positions", []):
                    if cdr_start <= pos < cdr_end:
                        cdr_liabs.append(f"DP-Clip @{pos+1}")
                for hs in liab.get("deamidation_hotspots", []):
                    if cdr_start <= hs["pos"] < cdr_end:
                        cdr_liabs.append(f"Deamid ({hs['motif']}) @{hs['pos']+1}")

                if cdr_liabs:
                    st.markdown(
                        f'<span class="status-pill status-warn">{chain_label} {cdr["name"]}: '
                        f'{", ".join(cdr_liabs)}</span>',
                        unsafe_allow_html=True,
                    )

        # -- Annotated sequence visualization --------------------------------
        st.markdown("#### Annotated Sequence")
        for ca in chain_analyses:
            chain_label = f"{ca['chain_type']} ({ca['name']})"
            st.markdown(f"**{chain_label}**")
            html = render_sequence_html(ca["sequence"], ca["cdrs"], ca["liabilities"])
            st.markdown(html, unsafe_allow_html=True)

        # -- Color legend ---------------------------------------------------
        st.markdown("""
        <div style="font-size:0.75rem; color:#64748B; margin-top:8px; padding:8px; background:#F9FAFB; border-radius:6px;">
            <b>Legend:</b>
            <span style="background:#FEF3C7; padding:1px 4px; border-radius:3px;">CDR region</span>
            <span style="color:#EF4444; background:#FEE2E2; padding:1px 4px; border-radius:3px;">Met-Ox</span>
            <span style="color:#991B1B; background:#FECACA; padding:1px 4px; border-radius:3px;">Trp-Ox</span>
            <span style="color:#1D4ED8; text-decoration:underline;">N-Glyco</span>
            <span style="background:#E9D5FF; padding:1px 4px; border-radius:3px;">DP-Clip</span>
            <span style="color:#10B981;">Cys</span>
            <span style="color:#F59E0B; text-decoration:underline;">Deamid</span>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
#  3B1b. Lightweight MS Summary (for Discovery / Multi-Chain pages)
# ===========================================================================

def render_ms_summary_metrics(intent: Dict[str, Any]) -> None:
    """
    Show MW, pI, GRAVY as quick-reference metric cards.
    Used in Discovery & Multi-Chain Assembly pages to avoid
    duplicating the full peptide map / glycan table.
    """
    mw = intent.get("mw") or intent.get("molecular_weight")
    pi = intent.get("pI") or intent.get("isoelectric_point")
    gravy = intent.get("hydrophobicity") or intent.get("gravy")
    if mw is None and pi is None and gravy is None:
        return
    st.markdown("##### Biophysical Summary")
    _mc1, _mc2, _mc3 = st.columns(3)
    with _mc1:
        st.metric("Molecular Weight", f"{mw:,.1f} kDa" if mw else "—")
    with _mc2:
        st.metric("Isoelectric Point (pI)", f"{pi:.2f}" if pi else "—")
    with _mc3:
        st.metric("GRAVY Score", f"{gravy:.3f}" if gravy else "—")

    # Interpretive notes for biophysical values
    _notes = []
    if mw is not None:
        if mw > 200:
            _notes.append(f"MW {mw:.0f} kDa — large molecule; consider viscosity at high concentration")
        elif mw < 30:
            _notes.append(f"MW {mw:.0f} kDa — small molecule; may undergo rapid renal clearance (< 60 kDa threshold)")
        else:
            _notes.append(f"MW {mw:.0f} kDa — typical range for IgG-class biologics")
    if pi is not None:
        if pi > 9.0:
            _notes.append(f"pI {pi:.2f} — highly basic; may bind non-specifically to cell surfaces and increase clearance")
        elif pi < 6.5:
            _notes.append(f"pI {pi:.2f} — acidic; generally favorable for solubility but verify charge-mediated interactions")
        elif pi > 7.5:
            _notes.append(f"pI {pi:.2f} — moderately basic; typical for many approved mAbs (pI 7.5–9.0)")
        else:
            _notes.append(f"pI {pi:.2f} — near-neutral; good general solubility profile")
    if gravy is not None:
        if gravy > 0:
            _notes.append(f"GRAVY {gravy:.3f} — positive (hydrophobic); elevated aggregation and viscosity risk")
        elif gravy < -0.8:
            _notes.append(f"GRAVY {gravy:.3f} — very hydrophilic; generally favorable for solubility")
        else:
            _notes.append(f"GRAVY {gravy:.3f} — moderately hydrophilic; typical range")
    if _notes:
        st.markdown(
            '<div style="font-size:0.78rem; color:#334155; padding:4px 0;">'
            + " · ".join(_notes) + '</div>',
            unsafe_allow_html=True,
        )
    st.caption("Full mass spectrometry, peptide mapping & glycan profiling available in the **Analytical & Mass Spec** tab.")


# ===========================================================================
#  3B2. In-Silico Mass Spectrometry Panel (M11+)
# ===========================================================================

def render_ms_characterization_panel(
    intent: Dict[str, Any],
    ms_result: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Render the In-Silico Mass Spectrometry (Intact & Peptide Map) expander.

    If ms_result is None, runs the characterization on-the-fly.
    Returns the ms_result dict (for caching).
    """
    sequence = intent.get("sequence", "")
    if not sequence or len(sequence) < 20:
        return None

    try:
        from src.analytical_twin import run_ms_characterization, peptide_map_to_dataframe
    except ImportError:
        st.caption("(In-silico MS module not available)")
        return None

    if ms_result is None:
        try:
            # Derive molecule_class from intent (preferred) or crude heuristic
            _ms_mol_class = intent.get("molecule_class")
            is_mab = intent.get("source") == "fasta" and len(sequence) > 400

            # M12: Use stoichiometric chains if available
            asm_chains = intent.get("assembly_chains")
            if asm_chains and len(asm_chains) > 0:
                ms_result = run_ms_characterization(
                    protein_name=intent.get("name", "Protein"),
                    is_mab=is_mab,
                    missed_cleavages=1,
                    chains=asm_chains,
                    molecule_class=_ms_mol_class,
                )
            else:
                # Legacy: build chains from intent if multi-chain FASTA
                parsed_chains = intent.get("chains", [])
                if len(parsed_chains) > 1:
                    ms_chains = [
                        {"sequence": ch["sequence"], "copy_number": 1,
                         "name": ch.get("name", "Chain"), "chain_type": ch.get("chain_type", "unknown")}
                        for ch in parsed_chains
                    ]
                    ms_result = run_ms_characterization(
                        protein_name=intent.get("name", "Protein"),
                        is_mab=is_mab,
                        missed_cleavages=1,
                        chains=ms_chains,
                        molecule_class=_ms_mol_class,
                    )
                else:
                    ms_result = run_ms_characterization(
                        sequence=sequence,
                        protein_name=intent.get("name", "Protein"),
                        is_mab=is_mab,
                        missed_cleavages=1,
                        molecule_class=_ms_mol_class,
                    )
        except Exception as ms_err:
            st.caption(f"(MS characterization error: {ms_err})")
            return None

    if ms_result.get("status") != "success":
        return ms_result

    data = ms_result.get("data", ms_result)
    intact = data.get("intact_mass", {})
    peptides = data.get("peptide_map", [])
    summary = data.get("summary", {})

    with st.expander("In-Silico Mass Spectrometry (Intact & Peptide Map)", expanded=True):

        # -- Intact Mass Section -----------------------------------------------
        st.markdown("#### Theoretical Intact Mass")
        bare_mass = intact.get("bare_mass_da", 0)
        ds_mass = intact.get("disulfide_corrected_da", 0)
        n_ds = intact.get("n_disulfide_bonds", 0)

        im1, im2, im3 = st.columns(3)
        with im1:
            st.metric("Bare Mass (Da)", f"{bare_mass:,.1f}")
        with im2:
            st.metric("Disulfide-Corrected", f"{ds_mass:,.1f}")
        with im3:
            st.metric("Disulfide Bonds", str(n_ds))

        # M12: Per-chain mass breakdown (if stoichiometric)
        per_chain_masses = intact.get("per_chain_masses", [])
        stoich_summary = intact.get("stoichiometry_summary", "")
        if per_chain_masses:
            st.markdown(f"**Assembly:** {stoich_summary}")
            import pandas as pd
            pcm_rows = []
            for pc in per_chain_masses:
                pcm_rows.append({
                    "Chain": pc.get("name", ""),
                    "Type": pc.get("chain_type", ""),
                    "Length (aa)": pc.get("sequence_length", 0),
                    "Copies": pc.get("copy_number", 1),
                    "Chain Mass (Da)": f"{pc.get('chain_mass_da', 0):,.1f}",
                    "Total (Da)": f"{pc.get('total_mass_da', 0):,.1f}",
                })
            st.dataframe(pd.DataFrame(pcm_rows), use_container_width=True, hide_index=True)

        # M12: Liability density
        liab_density = ms_result.get("liability_density") or data.get("liability_density")
        if liab_density:
            ld_val = liab_density.get("density_per_1000", 0)
            ld_level = liab_density.get("risk_level", "Low")
            ld_pill = "ok" if ld_level == "Low" else ("warn" if ld_level == "Medium" else "err")
            st.markdown(f"""
            <div class="cqa-card" style="display:inline-block;">
                <div class="cqa-title">Liability Density</div>
                <div class="cqa-value">{ld_val:.1f} <span class="cqa-unit">motifs / 1000 aa</span></div>
                <span class="status-pill status-{ld_pill}">{ld_level}</span>
            </div>
            """, unsafe_allow_html=True)

        # Glycoform table — qualitative species list only
        # (Quantitative occupancies require experimental LC-MS or HILIC data;
        #  in-silico prediction of relative abundance is not reliable.)
        glycoforms = intact.get("glycoforms", [])
        if glycoforms:
            st.markdown("**Predicted Glycoform Species (N-linked)**")
            st.caption(
                "Qualitative: lists expected glycoform species and their theoretical "
                "intact masses. Relative abundances require experimental LC-MS/HILIC "
                "data and are not shown."
            )
            import pandas as pd
            gf_rows = []
            for gf in glycoforms:
                gf_rows.append({
                    "Glycoform": gf.get("name", ""),
                    "Intact Mass (Da)": f"{gf.get('intact_mass_da', 0):,.1f}",
                })
            st.dataframe(pd.DataFrame(gf_rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # -- Tryptic Digest / Peptide Map Section ------------------------------
        st.markdown("#### In-Silico Tryptic Digest (Peptide Map)")
        total_peptides = summary.get("total_peptides", len(peptides))
        liab_peptides = summary.get("liability_peptides", 0)

        tp1, tp2 = st.columns(2)
        with tp1:
            st.metric("Total Peptides", total_peptides)
        with tp2:
            st.metric("With Liabilities", liab_peptides)

        if peptides:
            import pandas as pd
            df = peptide_map_to_dataframe(peptides)
            if df is not None and not df.empty:
                # Searchable peptide map table
                search_query = st.text_input(
                    "Search peptides (by sequence, liability, or index):",
                    key="peptide_search",
                    placeholder="e.g. NG, M, CDR, T12...",
                    help=(
                        "Filter the tryptic peptide map. Enter a motif (NG for deamidation), "
                        "a residue letter (M for methionine/oxidation), 'CDR' to filter CDR peptides, "
                        "or a peptide index like 'T12'. Case-insensitive."
                    ),
                )
                display_df = df.copy()
                if search_query:
                    mask = display_df.apply(
                        lambda row: row.astype(str).str.contains(search_query, case=False).any(),
                        axis=1,
                    )
                    display_df = display_df[mask]

                # Highlight rows with PTM liabilities using Pandas Styler
                def _highlight_liabilities(row):
                    """Apply amber background to rows with PTM liabilities."""
                    if row.get("# Liab", 0) > 0:
                        return ["background-color: #FEF3C7; color: #92400E;"] * len(row)
                    return [""] * len(row)

                _styled = display_df.style.apply(_highlight_liabilities, axis=1)
                st.dataframe(
                    _styled,
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )
            else:
                st.info("Peptide map could not be converted to DataFrame.")
        else:
            st.info("No peptides generated (sequence may be too short).")

    return ms_result


# ===========================================================================
#  3B3. Cached Analysis Re-Renderer (Workspace History Fix)
# ===========================================================================

def render_cached_analysis(ws: Dict[str, Any], ws_store: "WorkspaceStore") -> None:
    """
    Re-render a previously cached analysis from workspace history.

    This function reads from ws["analysis_cache"] and renders all panels
    WITHOUT re-running ML inference, CADET simulation, or pLM embeddings.
    """
    cache = ws.get("analysis_cache")
    if not cache:
        return

    mode = cache.get("mode", "standard")
    intent = cache.get("intent", {})

    # Cross-check: use workspace-level intent as secondary guard.
    # If workspace intent says "canonical_mab" but cache says "bispecific",
    # the cache is stale — skip bispecific rendering.
    _ws_intent = ws.get("intent") or {}
    _ws_level_mol_class = (_ws_intent.get("molecule_class", "") or "").lower()

    with st.chat_message("assistant"):

        if mode == "bispecific":
            # -- Bispecific mode re-render -------------------------------------
            bispec_result = cache.get("bispecific_result")
            # Guard: only render if BOTH the cache intent AND workspace intent
            # agree this is a bispecific molecule. Prevents stale bispecific
            # chromatograms from contaminating canonical mAb views.
            _cache_mol_class = (intent.get("molecule_class", "") or "").lower()
            _is_bispec_cache = ("bispecific" in _cache_mol_class
                                or "fusion" in _cache_mol_class
                                or intent.get("is_bispecific", False))
            _is_bispec_ws = ("bispecific" in _ws_level_mol_class
                             or "fusion" in _ws_level_mol_class
                             or _ws_intent.get("is_bispecific", False))
            # Require at least one positive signal (cache OR workspace intent)
            # and NO contradicting signal from the other.
            _ws_is_canonical = (
                _ws_level_mol_class
                and "bispecific" not in _ws_level_mol_class
                and "fusion" not in _ws_level_mol_class
                and not _ws_intent.get("is_bispecific", False)
            )
            _should_render_bispec = (
                bispec_result
                and bispec_result.get("status") == "success"
                and (_is_bispec_cache or _is_bispec_ws)
                and not _ws_is_canonical  # workspace intent actively says non-bispecific = stale
            )
            if _should_render_bispec:
                st.markdown(f"""
                <div class="fasta-card" style="border-left: 4px solid #64748B;">
                    <div class="fasta-title">Bispecific / Fusion Mode — Cached</div>
                    <i>Re-rendered from workspace history</i>
                </div>
                """, unsafe_allow_html=True)

                render_bispecific_species_panel(bispec_result)
                render_bispecific_chromatogram(bispec_result)
                render_bispecific_risk_panel(bispec_result)
            return

        # -- Standard mAb mode re-render ---------------------------------------
        source = cache.get("source", "text")

        # FASTA card or text card
        if source == "fasta":
            st.markdown(f"""
            <div class="fasta-card">
                <div class="fasta-title">FASTA Sequence Analysis (Biopython) — Cached</div>
                <b>{intent.get('name', 'Protein')}</b><br>
                pI = {intent.get('pI', 'N/A')} | MW = {intent.get('mw', 'N/A')} kDa |
                GRAVY = {intent.get('gravy', 'N/A')} |
                Hydrophobicity = {intent.get('hydrophobicity', 'N/A')}<br>
                Chains: {len(intent.get('chains', []))} |
                Sequence length: {intent.get('seq_length', 'N/A')} aa
            </div>
            """, unsafe_allow_html=True)

            # ── Molecule Classification Badge (cached re-render) ──
            _cls_info_c = intent.get("molecule_class_info", {})
            if _cls_info_c:
                _cls_name_c = _cls_info_c.get("display_name", "Unclassified")
                _cls_conf_c = _cls_info_c.get("confidence", "Low")
                _cls_evidence_c = _cls_info_c.get("evidence", [])
                _cls_warnings_c = _cls_info_c.get("warnings", [])
                _conf_color_c = {"High": "#10B981", "Medium": "#F59E0B", "Low": "#EF4444"}.get(_cls_conf_c, "#94A3B8")
                _has_fc_c = _cls_info_c.get("has_fc_region", False)
                _expects_glyco_c = _cls_info_c.get("expects_glycosylation", False)

                _cls_ev_text_c = " &middot; ".join(
                    e.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    for e in _cls_evidence_c[:3]
                )
                _fc_badge_c = "<span style='background:#e0e7ff; padding:2px 6px; border-radius:8px; font-size:0.75em; margin-left:6px;'>Fc region</span>" if _has_fc_c else ""
                _glyco_badge_c = "<span style='background:#fef3c7; padding:2px 6px; border-radius:8px; font-size:0.75em; margin-left:6px;'>N-glycosylation expected</span>" if _expects_glyco_c else ""
                st.markdown(
                    f"<div style='background:#f0f4f8;border-left:4px solid {_conf_color_c};"
                    f"padding:10px 14px;border-radius:6px;margin:8px 0 12px 0;'>"
                    f"<span style='font-weight:600;font-size:0.95em;'>Molecule Type: {_cls_name_c}</span>"
                    f"<span style='background:{_conf_color_c};color:white;padding:2px 8px;"
                    f"border-radius:8px;font-size:0.78em;margin-left:8px;'>{_cls_conf_c} confidence</span>"
                    f"{_fc_badge_c}{_glyco_badge_c}"
                    f"<br/><span style='font-size:0.82em;color:#64748B;'>{_cls_ev_text_c}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if _cls_warnings_c:
                    for _w_c in _cls_warnings_c:
                        st.warning(_w_c)

            # Re-render characterization panel
            render_characterization_panel(intent)
        else:
            st.markdown(f"""
            <div class="fasta-card">
                <div class="fasta-title">Protein Parameters (Text Input) — Cached</div>
                <b>{intent.get('name', 'mAb')}</b><br>
                pI = {intent.get('pI', 'N/A')} | MW = {intent.get('mw', 150)} kDa |
                Hydrophobicity = {intent.get('hydrophobicity', 0.35)}
            </div>
            """, unsafe_allow_html=True)

        # ML override info
        ml_override = cache.get("ml_override")
        source_label = cache.get("source_label", "Unknown")
        if ml_override:
            st.success(
                f"ML Override (cached): ka={ml_override['ka']:.4f}, "
                f"nu={ml_override['nu']:.3f}"
            )

        # C11: SMA parameter cards removed from Characterization tab.
        # CADET parameters are internal and displayed in Downstream Purification tab.
        variants = cache.get("variants")

        # Developability panel
        dev_result = cache.get("dev_result")
        if dev_result and dev_result.get("status") == "success":
            try:
                render_developability_panel(intent, dev_result)
                render_actionable_insights_panel(dev_result)
                # Validation Plan moved to Virtual QC Lab tab (Tab 2)
            except Exception:
                pass

        # MS Characterization panel — NOT rendered here.
        # Peptide mapping / intact mass is shown exclusively in the Mass Spec tab
        # to avoid duplication when render_cached_analysis runs inside the
        # "Sequence & Liability Analysis" tab context.

        # M13: PK panel
        pk_result = cache.get("pk_result")
        if pk_result:
            try:
                glyco_impact = cache.get("glycoform_impact")
                render_pk_panel(pk_result, glycoform_impact=glyco_impact)
            except Exception:
                pass

        # M14: Optimization results
        opt_result = cache.get("optimization_result")
        if opt_result:
            try:
                render_optimization_panel(opt_result)
            except Exception:
                pass

        # Bispecific auto-detection (assembly path stores mode="standard"
        # but may include a bispecific_result from auto-detected distinct chains).
        # Guard: only render if the molecule is NOT a canonical mAb — canonical
        # mAbs have HC+LC which are naturally different and should never show
        # bispecific analysis.
        _std_bispec = cache.get("bispecific_result")
        _std_intent_class = (intent.get("molecule_class", "") or "").lower()
        _std_is_canonical = "canonical" in _std_intent_class
        if _std_bispec and not _std_is_canonical:
            _std_bispec_ok = (
                (isinstance(_std_bispec, dict) and _std_bispec.get("status") == "success")
                or (isinstance(_std_bispec, dict) and _std_bispec.get("chromatogram"))
                or (isinstance(_std_bispec, dict) and _std_bispec.get("data", {}).get("chromatogram"))
            )
            if _std_bispec_ok:
                try:
                    st.markdown("---")
                    # Check for multispecific pairwise results
                    _cached_pairwise = cache.get("bispecific_pairwise_results", [])
                    if _cached_pairwise and len(_cached_pairwise) > 1:
                        st.markdown(
                            f"#### Multispecific Separation Analysis -- Cached "
                            f"({len(_cached_pairwise)} pairwise analyses)"
                        )
                        for _pw_idx, _pw_result in enumerate(_cached_pairwise):
                            _pw_data = _pw_result.get("data", _pw_result)
                            st.markdown(f"**Pair {_pw_idx+1}/{len(_cached_pairwise)}**")
                            render_bispecific_species_panel(_pw_data)
                            render_bispecific_chromatogram(_pw_data)
                            render_bispecific_risk_panel(_pw_data)
                    else:
                        st.markdown("#### Bispecific Separation Analysis -- Cached")
                        render_bispecific_species_panel(_std_bispec)
                        render_bispecific_chromatogram(_std_bispec)
                        render_bispecific_risk_panel(_std_bispec)
                except Exception as _bispec_std_err:
                    st.caption(f"(Bispecific render error: {_bispec_std_err})")

        # CQA / Chromatogram
        cqa = cache.get("cqa")
        if cqa:
            st.markdown("#### Separation Quality (CQA)")
            c1, c2, c3 = st.columns(3)
            for col, comp in zip((c1, c2, c3), ("Acidic", "Main", "Basic")):
                pk = cqa.get("peaks", {}).get(comp)
                if pk:
                    with col:
                        st.markdown(f"""
                        <div class="cqa-card">
                            <div class="cqa-title">{comp} Variant</div>
                            <div class="cqa-value">{pk['rt_min']:.2f} <span class="cqa-unit">min</span></div>
                            <div style="font-size:0.8rem; color:#64748B;">
                                FWHM: {pk['fwhm_min']:.3f} min |
                                Area: {cqa.get('area_pct', {}).get(comp, 0):.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Resolution
            for label, rs in cqa.get("resolution", {}).items():
                quality = "Baseline" if rs >= 1.5 else ("Partial" if rs >= 0.8 else "Overlap")
                pill_status = "ok" if rs >= 1.5 else ("warn" if rs >= 0.8 else "err")
                st.markdown(f"""
                <div class="cqa-card" style="display:inline-block; margin-right:12px;">
                    <div class="cqa-title">{label} Resolution</div>
                    <div class="cqa-value">Rs = {rs:.3f}</div>
                    <span class="status-pill status-{pill_status}">{quality}</span>
                </div>
                """, unsafe_allow_html=True)

            # C11: IEX chromatogram removed from Characterization tab.
            # Chromatography analysis available in Downstream Purification tab.

            sim_summary = cache.get("sim_summary")
            if sim_summary:
                elapsed = cache.get("sim_elapsed", 0)
                st.markdown(f"Simulation completed in {elapsed:.1f}s.")

        st.caption("*Re-rendered from workspace history cache*")


# ===========================================================================
#  3C. Developability Risk Assessment Panel (M8)
# ===========================================================================

def render_developability_panel(intent: Dict[str, Any], dev_result: Dict[str, Any]) -> None:
    """
    Render the Developability Score dashboard with risk metrics.

    Displays:
      - Composite Developability Score with colored gauge
      - Individual risk cards (Aggregation, Stability, Viscosity)
      - Grade badge (Low/Medium/High Risk)
      - Embedding and prediction mode info
    """
    data = dev_result.get("data", dev_result)
    score_info = data.get("score", {})
    predictions = data.get("predictions", {})
    score_val = score_info.get("score", 0)
    grade = score_info.get("grade", "Unknown")
    color = score_info.get("color", "#64748B")
    embed_mode = data.get("embedding_mode", "mock")
    pred_mode = data.get("prediction_mode", "rule_based")

    with st.expander("Developability Risk Assessment (pLM + XGBoost)", expanded=True):
        # Mode indicators
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            embed_badge = "ESM-2" if embed_mode == "esm2" else "Mock (Composition)"
            st.markdown(f"**Embedding:** `{embed_badge}`")
        with mode_col2:
            pred_badge = "XGBoost" if pred_mode == "xgboost" else "Rule-Based Heuristic"
            st.markdown(f"**Predictor:** `{pred_badge}`")

        # Composite score with interpretation — score is RISK (0=best, 1=worst)
        if score_val <= 0.30:
            _score_interp = (
                "This molecule shows a favorable developability profile (low risk). "
                "Standard CMC development pathway is recommended with routine monitoring."
            )
        elif score_val <= 0.55:
            _score_interp = (
                "This molecule shows moderate developability risks. Targeted optimization "
                "of identified risk areas is recommended before advancing to late-stage development."
            )
        else:
            _score_interp = (
                "This molecule has significant developability concerns. Consider engineering "
                "modifications or variant screening to address the high-risk dimensions before proceeding."
            )

        st.markdown(f"""
        <div style="text-align:center; padding:15px; margin:10px 0;
                    background:linear-gradient(135deg, {color}15, {color}30);
                    border-radius:12px; border:2px solid {color};">
            <div style="font-size:0.85rem; color:#64748B; text-transform:uppercase;
                        letter-spacing:1px;">Developability Score</div>
            <div style="font-size:2.8rem; font-weight:700; color:{color};
                        margin:5px 0;">{score_val:.3f}</div>
            <div style="display:inline-block; padding:4px 16px; border-radius:20px;
                        background:{color}; color:white; font-weight:600;
                        font-size:0.9rem;">{grade}</div>
            <div style="font-size:0.8rem; color:#334155; margin-top:8px;
                        max-width:500px; display:inline-block; text-align:center;">
                {_score_interp}</div>
        </div>
        """, unsafe_allow_html=True)

        # Individual risk metrics
        rc1, rc2, rc3 = st.columns(3)

        agg = predictions.get("agg_risk", 0)
        stab = predictions.get("stability", 0)
        visc = predictions.get("viscosity_risk", 0)

        agg_color = "#EF4444" if agg > 0.5 else ("#F59E0B" if agg > 0.3 else "#10B981")
        stab_color = "#EF4444" if stab < 0.65 else ("#F59E0B" if stab < 0.8 else "#10B981")
        visc_color = "#EF4444" if visc > 0.4 else ("#F59E0B" if visc > 0.2 else "#10B981")

        # Interpretation text for each dimension
        _agg_interp = (
            "High aggregation propensity — consider hydrophobic patch engineering or formulation optimization"
            if agg > 0.5 else
            "Moderate aggregation risk — monitor with SEC during formulation development"
            if agg > 0.3 else
            "Low aggregation propensity — favorable for manufacturing and storage"
        )
        _stab_interp = (
            "Low thermal stability — high risk of degradation during processing and storage"
            if stab < 0.65 else
            "Moderate stability — may require optimized formulation (pH, excipients)"
            if stab < 0.8 else
            "Good thermal stability — expected to maintain integrity under standard conditions"
        )
        _visc_interp = (
            "High viscosity risk — may impede subcutaneous delivery at high concentration (>100 mg/mL)"
            if visc > 0.4 else
            "Moderate viscosity concern — measure experimentally at target concentration"
            if visc > 0.2 else
            "Low viscosity risk — suitable for high-concentration formulation"
        )

        for col, (label, val, clr, desc, interp) in zip(
            (rc1, rc2, rc3),
            [
                ("Aggregation Risk", agg, agg_color, "Lower is better", _agg_interp),
                ("Thermal Stability", stab, stab_color, "Higher is better", _stab_interp),
                ("Viscosity Risk", visc, visc_color, "Lower is better", _visc_interp),
            ],
        ):
            with col:
                st.markdown(f"""
                <div class="cqa-card">
                    <div class="cqa-title">{label}</div>
                    <div class="cqa-value" style="color:{clr};">{val:.3f}</div>
                    <div style="font-size:0.75rem; color:#94A3B8;">{desc}</div>
                    <div style="font-size:0.75rem; color:#334155; margin-top:4px;">
                        {interp}</div>
                </div>
                """, unsafe_allow_html=True)


def render_actionable_insights_panel(dev_result: Dict[str, Any]) -> None:
    """
    Render actionable engineering advice from SHAP analysis.

    Displays:
      - SHAP contribution breakdown (embedding vs biophysical)
      - Actionable advice with priority badges
      - Per-target SHAP biophysical feature values
    """
    data = dev_result.get("data", dev_result)
    advice_list = data.get("advice", [])
    shap_info = data.get("shap", {})

    with st.expander("Actionable AI Insights & SHAP Explanations", expanded=True):

        # Priority-sorted advice
        priority_icons = {"high": "[!]", "medium": "[~]", "low": "[ok]", "info": "[i]"}
        priority_order = {"high": 0, "medium": 1, "low": 2, "info": 3}

        sorted_advice = sorted(advice_list, key=lambda x: priority_order.get(x.get("priority", "info"), 99))

        if sorted_advice:
            st.markdown("#### Engineering Recommendations")
            for a in sorted_advice:
                icon = priority_icons.get(a.get("priority", "info"), "")
                cat = a.get("category", "")
                risk = a.get("risk_level", "")
                msg = a.get("message", "")
                st.markdown(f"""
                <div style="padding:10px 15px; margin:6px 0; border-radius:8px;
                            background:#F9FAFB; border-left:4px solid
                            {'#EF4444' if a.get('priority')=='high' else '#F59E0B' if a.get('priority')=='medium' else '#10B981'};">
                    <div style="font-weight:600; font-size:0.9rem;">
                        {icon} {cat} — <span style="color:#64748B;">{risk}</span>
                    </div>
                    <div style="font-size:0.85rem; color:#334155; margin-top:4px;">
                        {msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # SHAP breakdown (if available)
        if shap_info.get("available"):
            st.markdown("#### SHAP Feature Attribution")
            for target_name in ["agg_risk", "stability", "viscosity_risk"]:
                target_data = shap_info.get("targets", {}).get(target_name, {})
                if "error" in target_data:
                    continue

                embed_c = target_data.get("embed_contribution", 0)
                biophys_c = target_data.get("biophys_contribution", 0)
                total_c = embed_c + biophys_c

                if total_c > 0:
                    embed_pct = 100 * embed_c / total_c
                    biophys_pct = 100 * biophys_c / total_c
                else:
                    embed_pct = 50
                    biophys_pct = 50

                display_name = target_name.replace("_", " ").title()
                st.markdown(f"**{display_name}** — Embedding: {embed_pct:.0f}% | Biophysical: {biophys_pct:.0f}%")

                # Show biophysical SHAP values
                biophys_shap = target_data.get("biophys_shap", {})
                if biophys_shap:
                    import plotly.graph_objects as go
                    names = list(biophys_shap.keys())
                    values = [biophys_shap[n] for n in names]
                    colors = ["#EF4444" if v > 0 else "#3B82F6" for v in values]

                    fig = go.Figure(go.Bar(
                        x=values, y=names,
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:+.4f}" for v in values],
                        textposition="outside",
                    ))
                    _apply_pharma_theme(fig,
                        title=f"SHAP: {display_name} (Biophysical Features)",
                        xaxis_title="SHAP Value",
                        height=250,
                        margin=dict(l=120, r=50, t=40, b=30),
                    )
                    st.plotly_chart(fig, use_container_width=True)


def render_validation_plan_panel(validation_plan: Dict[str, Any]) -> None:
    """
    Render the analytical validation plan as a professional table.

    Displays:
      - Molecule format note (why this panel is tailored)
      - Required assays (always recommended)
      - Format-specific assays (molecule-class-driven)
      - Risk-triggered assays with threshold explanations
      - Excluded assays (not applicable for this format)
      - Summary and recommendations
    """
    with st.expander("Analytical Validation Plan", expanded=True):
        total = validation_plan.get("total_assays", 0)
        summary = validation_plan.get("risk_summary", "")
        mol_cls = validation_plan.get("molecule_class", "canonical_mab")

        st.markdown(f"**{total} assays recommended** — {summary}")

        # Format guidance note
        format_note = validation_plan.get("format_note", "")
        if format_note:
            cls_label = mol_cls.replace("_", " ").title()
            st.markdown(f"""
            <div style="padding:8px 12px; margin:6px 0; border-radius:6px;
                        background:#EFF6FF; border:1px solid #BFDBFE;">
                <span style="font-weight:600; color:#1E40AF;">
                    Format: {cls_label}</span>
                <div style="font-size:0.82rem; color:#1E3A5F; margin-top:3px;">
                    {format_note}</div>
            </div>
            """, unsafe_allow_html=True)

        # Required assays table
        required = validation_plan.get("required_assays", [])
        if required:
            st.markdown("##### Required Assays (ICH Q6B Standard Panel)")
            for a in required:
                st.markdown(f"""
                <div style="padding:8px 12px; margin:4px 0; border-radius:6px;
                            background:#ECFDF5; border:1px solid #A7F3D0;">
                    <span style="font-weight:600; color:#065F46;">\u2714 {a.get('name', '')}</span>
                    <span style="float:right; font-size:0.8rem; color:#64748B;">
                        {a.get('timeline', '')} | {a.get('priority', '').upper()}</span>
                    <div style="font-size:0.8rem; color:#334155; margin-top:2px;">
                        Measures: {a.get('measures', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Format-specific assays (NEW)
        format_specific = validation_plan.get("format_specific_assays", [])
        if format_specific:
            cls_label = mol_cls.replace("_", " ").title()
            st.markdown(f"##### Format-Specific Assays ({cls_label})")
            for a in format_specific:
                explanation = a.get("explanation", "")
                explanation_html = ""
                if explanation:
                    explanation_html = f"""
                    <div style="font-size:0.78rem; color:#5B21B6; margin-top:3px;
                                font-style:italic;">
                        Why: {explanation}</div>"""
                st.markdown(f"""
                <div style="padding:8px 12px; margin:4px 0; border-radius:6px;
                            background:#F5F3FF; border:1px solid #C4B5FD;">
                    <span style="font-weight:600; color:#5B21B6;">\u2605 {a.get('name', '')}</span>
                    <span style="float:right; font-size:0.8rem; color:#64748B;">
                        {a.get('timeline', '')} | {a.get('priority', '').upper()}</span>
                    <div style="font-size:0.8rem; color:#334155; margin-top:2px;">
                        Measures: {a.get('measures', '')}
                    </div>{explanation_html}
                </div>
                """, unsafe_allow_html=True)

        # Risk-triggered assays
        triggered = validation_plan.get("risk_triggered_assays", [])
        if triggered:
            st.markdown("##### Risk-Triggered Assays")
            for a in triggered:
                border_color = "#EF4444" if a.get("priority") == "high" else "#F59E0B"
                st.markdown(f"""
                <div style="padding:8px 12px; margin:4px 0; border-radius:6px;
                            background:#FFF7ED; border:1px solid {border_color}40;">
                    <span style="font-weight:600; color:#92400E;">\u26A0 {a.get('name', '')}</span>
                    <span style="float:right; font-size:0.8rem; color:#64748B;">
                        {a.get('timeline', '')} | {a.get('priority', '').upper()}</span>
                    <div style="font-size:0.8rem; color:#334155; margin-top:2px;">
                        Trigger: {a.get('trigger_reason', '')}
                    </div>
                    <div style="font-size:0.78rem; color:#64748B;">
                        Measures: {a.get('measures', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        elif not required and not format_specific:
            st.info("No risk-triggered assays. Standard panel is sufficient.")

        # Excluded assays (show what was removed and why)
        excluded = validation_plan.get("excluded_assays", [])
        if excluded:
            st.markdown("##### Excluded for This Format")
            for a in excluded:
                st.markdown(f"""
                <div style="padding:6px 12px; margin:3px 0; border-radius:6px;
                            background:#F9FAFB; border:1px solid #E2E8F0;">
                    <span style="color:#94A3B8; text-decoration:line-through;">
                        {a.get('name', '')}</span>
                    <span style="font-size:0.78rem; color:#64748B; margin-left:8px;">
                        {a.get('reason', '')}</span>
                </div>
                """, unsafe_allow_html=True)

        # Recommendations
        recommendations = validation_plan.get("recommendations", [])
        if recommendations:
            st.markdown("##### Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")


# ===========================================================================
#  3C1b. Formulation Digital Twin Panel (M17)
# ===========================================================================

def render_formulation_twin_panel(
    intent: Dict[str, Any],
    form_result: Dict[str, Any],
    base_predictions: Dict[str, float],
) -> None:
    """
    Render the Formulation Digital Twin panel showing how buffer pH,
    buffer type, and excipients affect the Developability Score.

    Displays:
      - Formulation summary bar
      - Net charge at buffer pH
      - Before vs After Developability Score comparison
      - Per-risk modifier breakdown (Agg, Stability, Viscosity)
      - Warnings and recommendations
    """
    adj_preds = form_result.get("adjusted_predictions", {})
    base_score = form_result.get("base_score", {})
    adj_score = form_result.get("adjusted_score", {})
    delta = form_result.get("score_delta", 0)
    net_charge = form_result.get("net_charge", 0)
    ph_pI_dist = form_result.get("ph_pI_distance", 0)
    modifiers = form_result.get("modifiers", {})

    with st.expander("Formulation Digital Twin (Buffer & Excipient Feedback)", expanded=True):
        st.caption(
            "Real-time simulation of how your formulation conditions affect developability. "
            "Adjust Buffer pH, Buffer Type, and Excipients in the sidebar to see changes."
        )

        # Formulation summary
        form_info = form_result.get("formulation", {})
        st.markdown(f"""
        <div style="padding:12px 16px; border-radius:8px; margin-bottom:12px;
                    background:#F8FAFC;
                    border:1px solid #E2E8F0;">
            <div style="font-size:0.85rem; color:#334155; font-weight:600;">
                Formulation: {form_info.get('buffer_full_name', '')} pH {form_info.get('buffer_ph', 6.0):.1f}
                | Excipients: {', '.join(form_info.get('excipients', [])) or 'None'}
            </div>
            <div style="font-size:0.8rem; color:#3B82F6; margin-top:4px;">
                Net Charge at pH {form_info.get('buffer_ph', 6.0):.1f}: <b>{net_charge:+.1f}</b>
                | Distance from pI: <b>{ph_pI_dist:.2f}</b> pH units
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Before vs After Score comparison
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            _base_c = base_score.get("color", "#64748B")
            st.markdown(f"""
            <div class="cqa-card">
                <div class="cqa-title">Base Score</div>
                <div class="cqa-value" style="color:{_base_c};">{base_score.get('score', 0):.3f}</div>
                <div style="font-size:0.75rem; color:#94A3B8;">{base_score.get('grade', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            _adj_c = adj_score.get("color", "#64748B")
            st.markdown(f"""
            <div class="cqa-card">
                <div class="cqa-title">Formulated Score</div>
                <div class="cqa-value" style="color:{_adj_c};">{adj_score.get('score', 0):.3f}</div>
                <div style="font-size:0.75rem; color:#94A3B8;">{adj_score.get('grade', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc3:
            _delta_color = "#10B981" if delta < 0 else ("#EF4444" if delta > 0 else "#64748B")
            _delta_arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "—")
            st.markdown(f"""
            <div class="cqa-card">
                <div class="cqa-title">Score Delta</div>
                <div class="cqa-value" style="color:{_delta_color};">{_delta_arrow} {abs(delta):.3f}</div>
                <div style="font-size:0.75rem; color:#94A3B8;">
                    {'Improved' if delta < 0 else ('Worsened' if delta > 0 else 'Unchanged')}</div>
            </div>
            """, unsafe_allow_html=True)

        # Per-risk modifier breakdown
        st.markdown("##### Risk Modifier Breakdown")
        rm1, rm2, rm3 = st.columns(3)
        _risk_items = [
            ("Agg Risk", modifiers.get("agg_risk", 0),
             base_predictions.get("agg_risk", 0), adj_preds.get("agg_risk", 0)),
            ("Stability", modifiers.get("stability", 0),
             base_predictions.get("stability", 0), adj_preds.get("stability", 0)),
            ("Viscosity", modifiers.get("viscosity_risk", 0),
             base_predictions.get("viscosity_risk", 0), adj_preds.get("viscosity_risk", 0)),
        ]
        for col, (label, mod, base_v, adj_v) in zip((rm1, rm2, rm3), _risk_items):
            with col:
                _mod_color = "#10B981" if mod < -0.01 else ("#EF4444" if mod > 0.01 else "#64748B")
                _mod_sign = "+" if mod > 0 else ""
                st.markdown(f"""
                <div style="padding:8px; border-radius:6px; text-align:center;
                            background:#F9FAFB; border:1px solid #E2E8F0;">
                    <div style="font-size:0.78rem; color:#64748B;">{label}</div>
                    <div style="font-size:0.85rem;">{base_v:.3f} → <b>{adj_v:.3f}</b></div>
                    <div style="font-size:0.9rem; font-weight:600; color:{_mod_color};">
                        {_mod_sign}{mod:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Warnings
        warnings = form_result.get("warnings", [])
        if warnings:
            for w in warnings:
                st.warning(w)

        # Recommendations
        recs = form_result.get("recommendations", [])
        if recs:
            with st.expander("Formulation Recommendations", expanded=True):
                for i, rec in enumerate(recs, 1):
                    st.markdown(f"**{i}.** {rec}")

        # Excipient detail cards
        exc_effects = form_result.get("excipient_effects", [])
        if exc_effects:
            with st.expander("Excipient Mechanism Details", expanded=True):
                for exc in exc_effects:
                    st.markdown(f"""
                    **{exc['name']}** ({exc['category']})
                    — Agg reduction: -{exc['agg_reduction']:.0%},
                    Stability boost: +{exc['stability_boost']:.0%},
                    Viscosity: {exc['viscosity_change']:+.0%}

                    _{exc['mechanism']}_

                    Typical concentration: {exc['concentration']}

                    ---
                    """)


# ===========================================================================
#  3C2. Preclinical PK & Half-Life Panel (M13)
# ===========================================================================

def render_pk_panel(
    pk_result: Dict[str, Any],
    glycoform_impact: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render the Preclinical PK & In-Vivo Efficacy panel.

    Displays:
      - Predicted half-life gauge/metric
      - PK risk assessment with color-coded badge
      - Penalty breakdown table
      - Glycoform impact on pI and chromatography
      - Engineering recommendations
    """
    with st.expander("Preclinical PK & In-Vivo Efficacy", expanded=True):
        t_half = pk_result.get("half_life_days", 0)
        baseline = pk_result.get("baseline_days", 21.0)
        risk = pk_result.get("risk_assessment", "Unknown")
        risk_color = pk_result.get("risk_color", "#64748B")
        effective_pi = pk_result.get("effective_pi", 0)
        cl = pk_result.get("clearance_ml_day_kg", 0)
        vd = pk_result.get("vd_ml_kg", 0)

        # -- Half-life gauge metric (large, prominent) -------------------------
        # Color-coded based on risk
        risk_pill = "ok" if risk == "Low" else ("warn" if risk == "Medium" else "err")
        st.markdown(f"""
        <div class="cqa-card" style="border-left: 5px solid {risk_color}; text-align:center;">
            <div class="cqa-title" style="font-size:1.1rem;">Predicted Human Half-Life</div>
            <div style="font-size:2.8rem; font-weight:700; color:{risk_color};">
                {t_half:.1f} <span style="font-size:1.2rem; font-weight:400;">days</span>
            </div>
            <div style="font-size:0.85rem; color:#64748B; margin:4px 0;">
                Baseline: {baseline:.0f} days | Effective pI: {effective_pi:.2f}
            </div>
            <span class="status-pill status-{risk_pill}">{risk} PK Risk</span>
        </div>
        """, unsafe_allow_html=True)

        # -- PK parameter cards ------------------------------------------------
        pk1, pk2, pk3 = st.columns(3)
        with pk1:
            st.metric("Clearance (CL)", f"{cl:.2f} mL/day/kg")
        with pk2:
            st.metric("Volume of Distribution", f"{vd:.1f} mL/kg")
        with pk3:
            total_mult = pk_result.get("total_multiplier", 1.0)
            st.metric("Total Penalty Factor", f"{total_mult:.3f}")

        # -- Penalty breakdown -------------------------------------------------
        penalties = pk_result.get("penalties", [])
        if penalties:
            st.markdown("**Half-Life Penalty Factors:**")
            for p in penalties:
                mult = p.get("multiplier", 1.0)
                color = "#10B981" if mult >= 0.90 else ("#F59E0B" if mult >= 0.70 else "#EF4444")
                st.markdown(f"""
                <div style="padding:6px 12px; margin:3px 0; border-radius:6px;
                            background:#F9FAFB; border-left:4px solid {color};">
                    <span style="font-weight:600;">{p['factor']}</span>
                    <span style="float:right; font-weight:600; color:{color};">
                        x {mult:.3f}
                    </span>
                    <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">
                        {p['reason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No PK penalties — standard IgG1 half-life expected.")

        # -- Glycoform impact section ------------------------------------------
        glyco_impact = pk_result.get("glycoform_impact", {})
        if glyco_impact:
            st.markdown("---")
            st.markdown("#### Glycoform Impact on PK & Chromatography")

            gi1, gi2 = st.columns(2)
            with gi1:
                profile_name = glyco_impact.get("name", "Standard CHO")
                hl_mult = glyco_impact.get("multiplier", 1.0)
                # Safeguard: glycoform half-life multiplier must be in [0.3, 1.2]
                # Standard G0F/G1F = always 1.0x. Clamp outliers.
                if not isinstance(hl_mult, (int, float)) or hl_mult > 1.2 or hl_mult < 0.1:
                    hl_mult = 1.0
                # If profile is standard_cho, force 1.0x regardless
                if glyco_impact.get("profile") in ("standard_cho", None):
                    hl_mult = 1.0
                hl_color = "#10B981" if hl_mult >= 0.90 else ("#F59E0B" if hl_mult >= 0.70 else "#EF4444")
                st.markdown(f"""
                <div class="cqa-card">
                    <div class="cqa-title">Glycoform Profile</div>
                    <div style="font-size:1.0rem; font-weight:600;">{profile_name}</div>
                    <div style="font-size:0.85rem; color:{hl_color};">
                        Half-life modifier: x{hl_mult:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with gi2:
                pi_shift = glyco_impact.get("pi_shift", 0.0)
                if abs(pi_shift) > 0.01:
                    shift_dir = "lower" if pi_shift < 0 else "higher"
                    st.markdown(f"""
                    <div class="cqa-card" style="border-left:4px solid #6366F1;">
                        <div class="cqa-title">pI Shift (Sialylation)</div>
                        <div style="font-size:1.0rem; font-weight:600; color:#6366F1;">
                            {pi_shift:+.1f} units ({shift_dir})
                        </div>
                        <div style="font-size:0.8rem; color:#64748B;">
                            Effective pI: {effective_pi:.2f} |
                            CEX elution shifts {'earlier' if pi_shift < 0 else 'later'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="cqa-card">
                        <div class="cqa-title">pI Shift</div>
                        <div style="font-size:1.0rem; color:#64748B;">None (neutral glycoform)</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Description
            desc = glyco_impact.get("description", "")
            if desc:
                st.caption(desc)

        # -- Glycoform chromatography feedback (if available) -------------------
        if glycoform_impact:
            chrom_text = glycoform_impact.get("chromatography_impact", "")
            if chrom_text:
                st.markdown("**Chromatography Impact:**")
                st.info(chrom_text)

        # -- Recommendations ---------------------------------------------------
        recommendations = pk_result.get("recommendations", [])
        if recommendations:
            st.markdown("**PK Engineering Recommendations:**")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")


# ===========================================================================
#  3C3. Generative Protein Engineering Panel (M14)
# ===========================================================================

def render_optimization_panel(
    opt_result: Dict[str, Any],
) -> None:
    """
    Render the Auto-Optimize comparative dashboard (M14).

    Displays:
      - Optimization triggers
      - Comparative table: WT vs Variant 1-3
      - Per-variant mutation details
      - Radar chart (WT vs best variant)
      - FASTA download button
    """
    if opt_result.get("status") == "no_optimization_needed":
        st.success(opt_result.get("summary", "No optimization needed."))
        return

    if opt_result.get("status") != "success":
        st.error(f"Optimization failed: {opt_result.get('message', 'Unknown error')}")
        return

    with st.expander("Generative Protein Engineering — Auto-Optimized Variants", expanded=True):
        # -- M15: Pareto / Wet-lab status banner --------------------------------
        if opt_result.get("pareto_applied"):
            n_pareto = sum(1 for v in opt_result.get("variants", []) if v.get("pareto_optimal"))
            n_rejected = len(opt_result.get("rejected_variants", []))
            st.markdown(f"""
            <div style="background:{ACCENT}10; border:1px solid {ACCENT}40;
                        border-radius:6px; padding:10px 16px; margin-bottom:12px;">
                <b style="color:{ACCENT};">Data-Driven Mode</b> —
                Wet-lab XGBoost model active.
                <b>{n_pareto}</b> Pareto-optimal variant(s),
                <b>{n_rejected}</b> rejected by safety checks.
            </div>
            """, unsafe_allow_html=True)

        # -- Triggers ----------------------------------------------------------
        triggers = opt_result.get("optimization_triggers", [])
        if triggers:
            st.markdown("**Optimization Triggers:**")
            for t in triggers:
                st.markdown(f"""
                <span class="status-pill status-warn" style="margin:2px;">
                    {t}
                </span>
                """, unsafe_allow_html=True)

        # -- Comparative Table -------------------------------------------------
        st.markdown("#### WT vs Optimized Variants")

        wt = opt_result.get("wild_type", {})
        variants = opt_result.get("variants", [])

        if wt and variants:
            import pandas as pd

            # Check if wet-lab predictions are available
            has_wetlab = any(v.get("wetlab_predictions") for v in variants)

            rows = []
            # Wild-type row
            wt_row = {
                "Variant": "Wild-Type (WT)",
                "Mutations": 0,
                "pI": wt.get("pI", 0),
                "MW (kDa)": wt.get("mw_kda", 0),
                "Hydrophobicity": wt.get("hydrophobicity", 0),
                "Liab. Density": wt.get("liability_density", 0),
                "Half-Life (days)": wt.get("pk_half_life", 0),
                "PK Risk": wt.get("pk_risk", "N/A"),
                "Dev Score": wt.get("dev_score", 0),
            }
            if has_wetlab:
                wt_wl = wt.get("wetlab_predictions", {})
                wt_row["Pred Agg%"] = round(wt_wl.get("Exp_Aggregation_Percent", 0), 1) if wt_wl else "—"
                wt_row["Pred Tm°C"] = round(wt_wl.get("Exp_Tm_MeltingTemp", 0), 1) if wt_wl else "—"
                wt_row["Pareto"] = "—"
            rows.append(wt_row)

            # Variant rows
            for var in variants:
                var_row = {
                    "Variant": var.get("name", "Variant"),
                    "Mutations": var.get("mutation_count", 0),
                    "pI": var.get("pI", 0),
                    "MW (kDa)": var.get("mw_kda", 0),
                    "Hydrophobicity": var.get("hydrophobicity", 0),
                    "Liab. Density": var.get("liability_density", 0),
                    "Half-Life (days)": var.get("pk_half_life", 0),
                    "PK Risk": var.get("pk_risk", "N/A"),
                    "Dev Score": "—",
                }
                if has_wetlab:
                    vwl = var.get("wetlab_predictions", {})
                    var_row["Pred Agg%"] = round(vwl.get("Exp_Aggregation_Percent", 0), 1) if vwl else "—"
                    var_row["Pred Tm°C"] = round(vwl.get("Exp_Tm_MeltingTemp", 0), 1) if vwl else "—"
                    if var.get("rejected"):
                        var_row["Pareto"] = "REJECTED"
                    elif var.get("pareto_optimal"):
                        var_row["Pareto"] = "OPTIMAL"
                    else:
                        var_row["Pareto"] = "dominated"
                rows.append(var_row)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Show rejected variant warnings
            rejected = opt_result.get("rejected_variants", [])
            for rej in rejected:
                reason = rej.get("rejection_reason", "Unknown")
                rname = rej.get("name", "Variant")
                st.warning(f"**{rname}** rejected: {reason}")

        # -- Per-variant mutation details --------------------------------------
        st.markdown("#### Mutation Details")
        for var in variants:
            mutations = var.get("mutations", [])
            if not mutations:
                continue

            var_name = var.get("name", "Variant")
            strategy = var.get("strategy", "")
            n_muts = var.get("mutation_count", 0)

            # Delta indicators
            delta_hl = var.get("delta_pk_half_life", 0)
            delta_ld = var.get("delta_liability_density", 0)
            hl_arrow = "+" if delta_hl > 0 else ""
            ld_arrow = "" if delta_ld >= 0 else ""

            hl_color = "#10B981" if delta_hl > 0 else ("#EF4444" if delta_hl < -1.0 else "#64748B")
            ld_color = "#10B981" if delta_ld < 0 else ("#EF4444" if delta_ld > 2.0 else "#64748B")

            st.markdown(f"""
            <div class="cqa-card" style="margin-bottom:12px;">
                <div class="cqa-title">{var_name} — {n_muts} mutations</div>
                <div style="font-size:0.85rem; color:#64748B; margin-bottom:6px;">{strategy}</div>
                <div style="display:flex; gap:16px; margin-bottom:6px;">
                    <span>pI: <b>{var.get('pI', 0):.2f}</b></span>
                    <span style="color:{hl_color};">Half-Life: <b>{var.get('pk_half_life', 0):.1f}d</b>
                        ({hl_arrow}{delta_hl:.1f}d)</span>
                    <span style="color:{ld_color};">Liab: <b>{var.get('liability_density', 0):.1f}/1k</b>
                        ({ld_arrow}{delta_ld:.1f})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Mutation list (compact)
            mut_lines = []
            for m in mutations[:12]:  # Cap display at 12
                chain = m.get("chain", "")
                notation = m.get("notation", "")
                rationale = m.get("rationale", "")[:60]
                region = m.get("region", "")
                mut_lines.append(
                    f"`{chain}: {notation}` — {rationale} _{region}_"
                )
            st.markdown("\n\n".join(mut_lines))
            if len(mutations) > 12:
                st.caption(f"... and {len(mutations) - 12} more mutations")

        # -- Radar chart: WT vs Best Variant -----------------------------------
        try:
            import plotly.graph_objects as go

            if wt and len(variants) >= 1:
                best_var = max(variants, key=lambda v: v.get("pk_half_life", 0))

                categories = ["pI (norm)", "Half-Life", "Low Liability", "Low Hydro", "FcRn Intact"]

                def _norm(val, lo, hi):
                    return max(0, min(1, (val - lo) / (hi - lo))) if hi > lo else 0.5

                wt_vals = [
                    _norm(wt.get("pI", 7), 5, 10),
                    _norm(wt.get("pk_half_life", 21), 0, 25),
                    1.0 - _norm(wt.get("liability_density", 50), 0, 100),
                    1.0 - wt.get("hydrophobicity", 0.35),
                    1.0 if wt.get("fcrn_intact", True) else 0.3,
                ]
                best_vals = [
                    _norm(best_var.get("pI", 7), 5, 10),
                    _norm(best_var.get("pk_half_life", 21), 0, 25),
                    1.0 - _norm(best_var.get("liability_density", 50), 0, 100),
                    1.0 - best_var.get("hydrophobicity", 0.35),
                    1.0 if best_var.get("fcrn_intact", True) else 0.3,
                ]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=wt_vals + [wt_vals[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name="Wild-Type",
                    line_color="#64748B",
                    opacity=0.6,
                ))
                fig.add_trace(go.Scatterpolar(
                    r=best_vals + [best_vals[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=best_var.get("name", "Best Variant"),
                    line_color="#10B981",
                    opacity=0.6,
                ))
                _apply_pharma_theme(fig,
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    height=350,
                    title="WT vs Best Variant (Radar)",
                )
                st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass  # Plotly not available

        # -- FASTA download ----------------------------------------------------
        fasta = opt_result.get("fasta_download", "")
        if fasta:
            st.download_button(
                label="Download Optimized FASTA Sequences",
                data=fasta,
                file_name="optimized_variants.fasta",
                mime="text/plain",
                type="primary",
            )

        # Summary
        st.caption(opt_result.get("summary", ""))


def render_optimize_button(
    intent: Dict[str, Any],
    dev_result: Optional[Dict[str, Any]] = None,
    pk_result: Optional[Dict[str, Any]] = None,
    bispec_result: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Render the One-Click Optimize button and trigger optimization.

    C9: Temporarily disabled. The backend function (src.agents.optimize_candidate)
    is functional, but the UI button is hidden pending full Agent pipeline validation.

    Only shows if there are detectable issues:
      - Developability Score > 0.35 (Medium or High risk)
      - PK half-life < 18 days
      - Bispecific Rs < 1.0
      - Liability density > 50/1k residues
    """
    # C9: Auto-Optimize button temporarily disabled
    return

    # Determine if optimization is warranted
    show_button = False
    reasons = []

    dev_score = 0.0
    dev_grade = "Low"
    if dev_result and dev_result.get("status") == "success":
        d = dev_result.get("data", dev_result)
        score_info = d.get("score", {})
        dev_score = score_info.get("score", 0)
        dev_grade = score_info.get("grade", "Low")
        if dev_score > 0.35:
            show_button = True
            reasons.append(f"Developability {dev_grade}")

    pk_hl = 21.0
    pk_risk_str = "Low"
    if pk_result:
        pk_hl = pk_result.get("half_life_days", 21.0)
        pk_risk_str = pk_result.get("risk_assessment", "Low")
        if pk_hl < 18.0:
            show_button = True
            reasons.append(f"PK {pk_risk_str}")

    bispec_rs = None
    bispec_chain_b_idx = None
    if bispec_result and bispec_result.get("status") == "success":
        res_data = bispec_result.get("data", {}).get("resolution", {})
        bispec_rs = res_data.get("min_rs", 99)
        if bispec_rs < 1.0:
            show_button = True
            reasons.append(f"Rs={bispec_rs:.2f}")
        bispec_chain_b_idx = 1  # Chain B is typically index 1

    # Check liability density from cache
    ms_char = (cache or {}).get("ms_characterization")
    if ms_char:
        ld = ms_char.get("liability_density") or ms_char.get("data", {}).get("liability_density")
        if ld and ld.get("density_per_1000", 0) > 50:
            show_button = True
            reasons.append(f"High liability density")

    # Also always show if user has chains available (allow proactive optimization)
    chains = intent.get("assembly_chains") or intent.get("chains", [])
    # Convert to proper chain format if needed
    proper_chains = []
    for ch in chains:
        seq = ch.get("sequence", "")
        if len(seq) >= 10:
            proper_chains.append({
                "sequence": seq,
                "copy_number": ch.get("copy_number", 1),
                "name": ch.get("name", "Chain"),
                "chain_type": ch.get("chain_type", "unknown"),
            })

    # If no proper chains, try to build from the full sequence
    if not proper_chains and intent.get("sequence"):
        seq = intent["sequence"]
        if len(seq) >= 20:
            proper_chains = [{"sequence": seq, "copy_number": 1, "name": "mAb", "chain_type": "HC"}]

    if not proper_chains:
        return  # Nothing to optimize

    # Show even without detected issues if chains exist (let users explore)
    if not show_button:
        # Still show, but less prominent
        if len(proper_chains) > 0:
            show_button = True
            reasons.append("Proactive exploration")

    if show_button:
        reason_text = " | ".join(reasons) if reasons else "Explore variants"
        btn_label = f"Auto-Optimize Sequence (Generative AI)"

        if st.button(btn_label, key="optimize_btn", type="primary"):
            with st.spinner("Engineering optimized variants via targeted mutagenesis (Met\u2192Leu, NG\u2192QG, charge balancing)..."):
                try:
                    from src.agents import optimize_candidate
                    opt_result = optimize_candidate(
                        chains=proper_chains,
                        dev_score=dev_score,
                        dev_grade=dev_grade,
                        pk_half_life=pk_hl,
                        pk_risk=pk_risk_str,
                        bispecific_rs=bispec_rs,
                        bispecific_chain_b_idx=bispec_chain_b_idx,
                        glycoform_profile=st.session_state.glycoform_profile,
                    )
                    st.session_state.optimization_result = opt_result

                    # Cache it
                    if cache is not None:
                        cache["optimization_result"] = opt_result

                except Exception as opt_err:
                    st.error(f"Optimization failed: {opt_err}")
                    return

    # Render results if available (outside show_button guard so results persist)
    opt_result = st.session_state.optimization_result
    if opt_result and opt_result.get("status") in ("success", "no_optimization_needed"):
        render_optimization_panel(opt_result)


# ===========================================================================
#  3C-HT. Early Discovery HT Screening Dashboard (M16)
# ===========================================================================

def render_ht_screening_tab() -> None:
    """
    Render the M16 Early Discovery HT Screening dashboard.

    Linear flow with three sections:
      1. Data Input: Upload Discovery CSV with real candidate sequences
      2. Screening Configuration & Run: Set thresholds, preview data, and run HT screening
      3. Results Dashboard: Interactive Magic Quadrant plot, Star candidates, CSV export
    """
    st.markdown("### Early Discovery — High-Throughput Virtual Screening")
    st.markdown(
        "Upload a CSV of candidate antibody sequences from Discovery. "
        "The HT engine screens each candidate for **Developability** and "
        "**Potency**, then classifies into the **Magic Quadrant**."
    )

    # ──────────────────────────────────────────────────────────────────
    # Section 1: Data Input (Upload Only — mock data removed for CMC integrity)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("### Data Input")

    st.markdown("**Upload Discovery CSV**")
    st.caption(
        "Expected columns: `Candidate_ID`, `Sequence_HC`, `Sequence_LC` (opt), "
        "`Exp_ELISA_OD` (opt), `Exp_Kd_nM` (opt). Column aliases accepted."
    )
    uploaded_ht = st.file_uploader(
        "Upload Discovery CSV",
        type=["csv"],
        key="ht_csv_uploader",
        help="Upload a CSV containing antibody discovery candidates.",
    )
    if uploaded_ht is not None:
        try:
            from src.ht_screening import parse_discovery_csv
            parsed = parse_discovery_csv(uploaded_ht.getvalue())
        except Exception as e:
            st.error(f"CSV parse error: {e}")
            parsed = None

        if parsed and parsed["status"] == "success":
            st.success(f"Parsed **{parsed['n_candidates']}** candidates")
            for w in parsed.get("warnings", []):
                st.warning(w)
            st.session_state.ht_parsed_candidates = parsed
        elif parsed:
            st.error(parsed.get("message", "Unknown error"))

    # ──────────────────────────────────────────────────────────────────
    # Section 2: Screening Configuration & Run
    # ──────────────────────────────────────────────────────────────────
    _has_candidates = (
        st.session_state.get("ht_parsed_candidates") is not None
        and isinstance(st.session_state.get("ht_parsed_candidates"), dict)
        and len(st.session_state["ht_parsed_candidates"].get("candidates", [])) > 0
    )
    if not _has_candidates:
        st.markdown("---")
        st.warning(
            "**No sequence data loaded.** Upload a Discovery CSV "
            "above before running the screening pipeline.",
            
        )
    if _has_candidates:
        st.markdown("---")
        st.markdown("### Screening Configuration & Run")
        _parsed_data = st.session_state.ht_parsed_candidates

        # Preview
        with st.expander(f"Preview ({_parsed_data['n_candidates']} candidates)", expanded=True):
            try:
                import pandas as _pd
                preview_rows = []
                for c in _parsed_data["candidates"][:10]:
                    preview_rows.append({
                        "ID": c["candidate_id"],
                        "HC Length": len(c["sequence_hc"]),
                        "LC Length": len(c.get("sequence_lc", "")),
                        "ELISA OD": c.get("exp_elisa_od", "—"),
                        "Kd (nM)": c.get("exp_kd_nm", "—"),
                    })
                st.dataframe(_pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
            except Exception:
                pass

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            dev_thresh = st.slider(
                "Developability Threshold", 0.1, 0.9, 0.5, 0.05,
                key="ht_dev_threshold",
                help="Minimum developability score to qualify as 'developable'.",
            )
        with col_t2:
            pot_thresh = st.slider(
                "Potency Threshold", 0.1, 0.9, 0.5, 0.05,
                key="ht_pot_threshold",
                help="Minimum potency score to qualify as 'potent'.",
            )

        if st.button("Run HT Screening", key="btn_run_ht", type="primary"):
            from src.ht_screening import HTScreeningEngine
            engine = HTScreeningEngine(dev_threshold=dev_thresh, potency_threshold=pot_thresh)
            n_cands = len(_parsed_data["candidates"])
            progress_bar = st.progress(0, text="Initializing screening...")
            status_text = st.empty()

            def _ht_progress(frac, msg):
                progress_bar.progress(min(frac, 1.0), text=msg)
                done = int(frac * n_cands)
                status_text.caption(f"Screening: {done}/{n_cands} completed...")

            _ht_t0 = time.time()
            result = engine.screen_candidates(_parsed_data["candidates"], progress_callback=_ht_progress)
            _ht_dur = time.time() - _ht_t0
            progress_bar.progress(1.0, text="Screening complete!")
            status_text.empty()

            audit_log_batch_prediction(n_candidates=n_cands, model_type="HT_Screening", duration_sec=_ht_dur)
            st.session_state.ht_screening_results = result
            st.session_state.ht_screening_csv = engine.get_results_as_csv()

            st.success(
                f"Screened **{result['n_candidates']}** candidates in "
                f"**{result['screening_time_sec']:.1f}s** — "
                f"**{result['summary']['quadrant_counts'].get('Star', 0)}** Stars found!"
            )

    # ──────────────────────────────────────────────────────────────────
    # Section 3: Results Dashboard (Magic Quadrant + Data Viewer)
    # ──────────────────────────────────────────────────────────────────
    _render_magic_quadrant()


def _render_magic_quadrant() -> None:
    """Render the Magic Quadrant interactive scatter plot and results table."""

    results = st.session_state.get("ht_screening_results")
    if results is None:
        st.info(
            "No screening results yet. Upload a Discovery CSV "
            "in the Data Input section above."
        )
        return

    data = results.get("results", [])
    summary = results.get("summary", {})

    if not data:
        st.warning("No candidate results available.")
        return

    st.markdown("### Magic Quadrant — Developability vs. Potency")

    # Summary metrics row
    qc = summary.get("quadrant_counts", {})
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    col_s1.metric("Total Screened", summary.get("n_total", 0))
    col_s2.metric("Stars", qc.get("Star", 0))
    col_s3.metric("Developable", qc.get("Developable", 0))
    col_s4.metric("Potent", qc.get("Potent", 0))
    col_s5.metric("Risky", qc.get("Risky", 0))

    # Models used
    models = results.get("models_used", {})
    model_tags = []
    if models.get("esm2_embedder"):
        model_tags.append("ESM-2")
    if models.get("xgb_developability"):
        model_tags.append("XGB-Dev")
    if models.get("xgb_potency"):
        model_tags.append("XGB-Potency")
    if not model_tags:
        model_tags.append("Heuristic-Only")
    st.caption(f"Models: {' + '.join(model_tags)} | Screening time: {results.get('screening_time_sec', 0):.1f}s")

    # -- Magic Quadrant Plotly Scatter --
    try:
        import plotly.graph_objects as go

        dev_scores = [r["developability_score"] for r in data]
        pot_scores = [r["potency_score"] for r in data]
        quadrants = [r["quadrant"] for r in data]
        ids = [r["candidate_id"] for r in data]

        # Color map
        color_map = {
            "Star": "#FFD700",        # Gold
            "Developable": "#4CAF50", # Green
            "Potent": "#2196F3",      # Blue
            "Risky": "#F44336",       # Red
        }
        colors = [color_map.get(q, "#94A3B8") for q in quadrants]

        fig = go.Figure()

        # Plot each quadrant separately for legend
        for quad_name, quad_color in color_map.items():
            idx = [i for i, q in enumerate(quadrants) if q == quad_name]
            if not idx:
                continue
            fig.add_trace(go.Scatter(
                x=[dev_scores[i] for i in idx],
                y=[pot_scores[i] for i in idx],
                mode="markers",
                name=f"{quad_name} ({len(idx)})",
                marker=dict(
                    size=8,
                    color=quad_color,
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                text=[f"{ids[i]}<br>Dev: {dev_scores[i]:.3f}<br>Pot: {pot_scores[i]:.3f}"
                      for i in idx],
                hovertemplate="%{text}<extra></extra>",
            ))

        # Add quadrant lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)

        # Quadrant labels
        fig.add_annotation(x=0.25, y=0.95, text="Developable", showarrow=False,
                           font=dict(size=12, color="#4CAF50"), opacity=0.6)
        fig.add_annotation(x=0.75, y=0.95, text="STAR", showarrow=False,
                           font=dict(size=14, color="#FFD700", weight="bold" if hasattr(dict, 'weight') else None), opacity=0.8)
        fig.add_annotation(x=0.25, y=0.05, text="Risky", showarrow=False,
                           font=dict(size=12, color="#F44336"), opacity=0.6)
        fig.add_annotation(x=0.75, y=0.05, text="Potent", showarrow=False,
                           font=dict(size=12, color="#2196F3"), opacity=0.6)

        _apply_pharma_theme(fig,
            title="Magic Quadrant — Candidate Classification",
            xaxis_title="Developability Score",
            yaxis_title="Potency Score",
            xaxis=dict(range=[0, 1], dtick=0.1, showgrid=True, gridcolor="#F1F5F9"),
            yaxis=dict(range=[0, 1], dtick=0.1, showgrid=True, gridcolor="#F1F5F9"),
            height=550,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Plotly not available — install plotly for interactive Magic Quadrant plot.")
        # Fallback: simple table
        st.markdown("**Quadrant Distribution:**")
        for q, count in qc.items():
            st.write(f"  - {q}: {count}")

    # -- Top Star Candidates Table --
    st.markdown("---")
    st.markdown("### Top Star Candidates")

    stars = [r for r in data if r["quadrant"] == "Star"]
    stars_sorted = sorted(stars,
                          key=lambda x: x["developability_score"] + x["potency_score"],
                          reverse=True)

    if stars_sorted:
        try:
            import pandas as _pd
            star_rows = []
            for s in stars_sorted[:20]:
                row = {
                    "Candidate": s["candidate_id"],
                    "Dev Score": round(s["developability_score"], 3),
                    "Potency Score": round(s["potency_score"], 3),
                    "Combined": round(s["developability_score"] + s["potency_score"], 3),
                    "pI": s["pI"],
                    "MW (kDa)": s["mw_kda"],
                    "Deam Sites": s["deam_sites"],
                    "Hydrophobicity": s["hydrophobicity"],
                }
                if "exp_elisa_od" in s:
                    row["ELISA OD"] = s["exp_elisa_od"]
                if "exp_kd_nm" in s:
                    row["Kd (nM)"] = s["exp_kd_nm"]
                star_rows.append(row)

            st.dataframe(
                _pd.DataFrame(star_rows),
                use_container_width=True,
                hide_index=True,
            )
        except ImportError:
            for s in stars_sorted[:10]:
                st.write(
                    f"**{s['candidate_id']}** — Dev: {s['developability_score']:.3f}, "
                    f"Potency: {s['potency_score']:.3f}"
                )
    else:
        st.warning(
            "No Star candidates found. Try adjusting the thresholds or training "
            "ML models on real data for more accurate scoring."
        )

    # -- Distribution Summary --
    with st.expander("Score Distribution Statistics", expanded=True):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.metric("Dev Score (mean)", f"{summary.get('dev_score_mean', 0):.3f}")
            st.metric("Dev Score (std)", f"{summary.get('dev_score_std', 0):.3f}")
        with col_d2:
            st.metric("Potency Score (mean)", f"{summary.get('potency_score_mean', 0):.3f}")
            st.metric("Potency Score (std)", f"{summary.get('potency_score_std', 0):.3f}")
        st.metric("Star Rate", f"{summary.get('star_rate_pct', 0):.1f}%")

    # -- CSV Download --
    st.markdown("---")
    csv_export = st.session_state.get("ht_screening_csv", "")
    if csv_export:
        st.download_button(
            "Download Full Screening Report (CSV)",
            data=csv_export,
            file_name="ht_screening_report.csv",
            mime="text/csv",
            key="dl_ht_report",
        )

        # Stars-only download
        star_csv_lines = [csv_export.split("\n")[0]]  # header
        for r in stars_sorted:
            for line in csv_export.split("\n")[1:]:
                if line.startswith(r["candidate_id"] + ","):
                    star_csv_lines.append(line)
                    break
        if len(star_csv_lines) > 1:
            st.download_button(
                f"Download Star Candidates Only ({len(stars_sorted)} candidates)",
                data="\n".join(star_csv_lines),
                file_name="star_candidates.csv",
                mime="text/csv",
                key="dl_stars_csv",
            )


# ===========================================================================
#  3C-bis. Data Foundation & Model Training Dashboard (M15)
# ===========================================================================

def render_data_foundation_panel() -> None:
    """
    Render the M15 Data Foundation & Model Training dashboard.

    Allows users to:
      1. Upload a Jain-137-format CSV with wet-lab data
      2. Trigger XGBoost training on the ingested data
      3. View R², RMSE, feature importance metrics
    """
    with st.expander("Data Foundation & Model Training", expanded=True):
        st.caption("Quick single-target XGBoost training for wet-lab data validation. For comprehensive 8-task deep learning, use AI Training Center.")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, {ACCENT}22, {SLATE}11);
                    border-left:4px solid {ACCENT}; border-radius:8px;
                    padding:16px; margin-bottom:16px;">
            <div style="font-size:1.1rem; font-weight:700; color:{DARK};">
                Data-Driven AI Platform
            </div>
            <div style="font-size:0.85rem; color:{SLATE};">
                Train supervised ML models on real wet-lab data (Jain-137 schema).
                Replace heuristics with empirical predictions for Aggregation% and Tm.
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_upload, tab_plm, tab_metrics = st.tabs([
            "Upload CSV", "PLM Embeddings", "Model Metrics",
        ])

        # -- Tab 1: Upload CSV --------------------------------------------------
        with tab_upload:
            st.markdown("**Upload Wet-Lab Dataset (CSV)**")
            st.caption(
                "Expected columns: `Name`, `Sequence_HC`, `Sequence_LC`, "
                "`Exp_Aggregation_Percent`, `Exp_Tm_MeltingTemp` (and optionally "
                "`Exp_SEC_RetentionTime`, `Exp_CIC_RetentionTime`, `Exp_Viscosity_cP`)"
            )

            # Check for data from AI Training Center
            _tc_data = st.session_state.get("wetlab_csv_data")
            if _tc_data:
                st.success(f"Training data available: **{len(_tc_data)}** samples")
                st.caption("Data was loaded in the AI Training Center → Data & Model Training tab.")
                try:
                    import pandas as _pd
                    preview_cols = [c for c in _tc_data[0].keys() if not c.startswith("Sequence")][:8]
                    preview = [{c: row.get(c) for c in preview_cols} for row in _tc_data[:8]]
                    st.dataframe(_pd.DataFrame(preview), use_container_width=True, hide_index=True)
                except Exception:
                    pass
                if st.button("Train XGBoost on This Dataset", key="train_uploaded_btn", type="primary"):
                    _train_wetlab_model(_tc_data)
            else:
                st.info(
                    "No training data loaded yet. Please upload a CSV in the "
                    "**AI Training Center** → **Data & Model Training** tab."
                )

        # -- Tab 2: PLM Embeddings -----------------------------------------------
        with tab_plm:
            st.markdown("**Train Foundation Model via ESM-2 Embeddings**")
            st.caption(
                "Upload a clinical benchmark CSV (Jain-137 or compatible) to extract "
                "320-dimensional Protein Language Model embeddings per sequence, then "
                "train an XGBoost regressor on high-dimensional representations."
            )

            # Check for data from AI Training Center or wet-lab upload
            _plm_data = st.session_state.get("wetlab_csv_data")
            if _plm_data is None:
                st.info(
                    "No training data loaded yet. Please upload a CSV in the "
                    "**AI Training Center** → **Data & Model Training** tab first."
                )
                plm_parsed = None
            else:
                st.success(f"Using training data: **{len(_plm_data)}** samples")
                plm_parsed = {
                    "status": "success",
                    "n_rows": len(_plm_data),
                    "schema": "jain137",
                    "data": _plm_data,
                    "columns": list(_plm_data[0].keys()) if _plm_data else [],
                }

                # Data preview
                try:
                    import pandas as _pd
                    preview_cols = [c for c in plm_parsed["columns"]
                                    if not c.startswith("Sequence")][:8]
                    if preview_cols:
                        preview_data = [
                            {c: row.get(c) for c in preview_cols}
                            for row in plm_parsed["data"][:8]
                        ]
                        st.dataframe(_pd.DataFrame(preview_data), use_container_width=True, hide_index=True)
                except Exception:
                    pass

                st.session_state.plm_csv_data = plm_parsed["data"]

                if st.button("Train Foundation Model via ESM-2 Embeddings",
                             key="train_plm_btn", type="primary"):
                    _train_plm_model(plm_parsed["data"])

            # Show PLM model metrics if available
            plm_metrics = st.session_state.get("plm_training_metrics")
            if plm_metrics and plm_metrics.get("status") == "success":
                st.markdown("---")
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, #7C3AED22, {SLATE}11);
                            border-left:4px solid #7C3AED; border-radius:8px;
                            padding:12px; margin-top:8px;">
                    <b>PLM Foundation Model</b> &mdash;
                    {plm_metrics.get('n_samples', 0)} samples &times;
                    {plm_metrics.get('n_features', 320)}-dim embeddings
                    ({plm_metrics.get('feature_mode', 'plm')})
                </div>
                """, unsafe_allow_html=True)

                for tname, tm in plm_metrics.get("per_target", {}).items():
                    r2 = tm.get("r2", 0)
                    rmse = tm.get("rmse", 0)
                    r2_color = SUCCESS if r2 > 0.7 else (WARN if r2 > 0.4 else ERROR)
                    st.markdown(f"""
                    <div style="display:inline-block; margin-right:20px; padding:8px 16px;
                                background:{r2_color}15; border-left:4px solid {r2_color};
                                border-radius:6px; margin-top:4px;">
                        <b>{tname}</b><br>
                        R² = <b style="color:{r2_color}">{r2:.3f}</b>
                        &nbsp;|&nbsp; RMSE = {rmse:.3f}
                    </div>
                    """, unsafe_allow_html=True)

        # -- Tab 4: Model Metrics -----------------------------------------------
        with tab_metrics:
            _render_wetlab_model_metrics()


def _train_wetlab_model(data: List[Dict[str, Any]]) -> None:
    """Train the wet-lab model and update session state."""
    with st.spinner("Encoding sequences via ESM-2 embeddings and training XGBoost ensemble on wet-lab assays..."):
        from src.data_pipeline import build_training_dataset
        from src.ml_predictor import train_wetlab_model

        # Auto-detect target columns from available data
        _preferred_targets = ["Exp_Aggregation_Percent", "Exp_Tm_MeltingTemp"]
        _alt_target_map = {
            "Exp_Aggregation_Percent": ["Aggregation", "Agg", "Agg%", "HMW", "aggregation_percent", "agg_pct"],
            "Exp_Tm_MeltingTemp": ["Tm", "MeltingTemp", "Tm1", "DSF_Tm", "melting_temp", "thermal_stability"],
        }
        _detected_targets = []
        if data:
            _sample_keys = set(data[0].keys())
            for _pref in _preferred_targets:
                if _pref in _sample_keys:
                    _detected_targets.append(_pref)
                else:
                    # Try alternative names
                    for _alt in _alt_target_map.get(_pref, []):
                        if _alt in _sample_keys:
                            _detected_targets.append(_alt)
                            break
        if not _detected_targets:
            # Show diagnostic
            _avail_cols = sorted(data[0].keys()) if data else []
            st.error(
                f"**No trainable target columns found.**\n\n"
                f"Expected: `Exp_Aggregation_Percent`, `Exp_Tm_MeltingTemp` (or similar).\n\n"
                f"Available columns: {', '.join(f'`{c}`' for c in _avail_cols)}\n\n"
                f"Please upload a CSV with experimental target data."
            )
            return

        train_ds = build_training_dataset(
            data,
            target_columns=_detected_targets,
        )

        if train_ds["status"] != "success":
            _avail_cols = sorted(data[0].keys()) if data else []
            _n_with_seq = sum(1 for r in data if (r.get("Sequence_HC") or r.get("Sequence", "")))
            st.error(
                f"**Failed to build training data:** {train_ds.get('message', 'Unknown error')}\n\n"
                f"Diagnostics: {len(data)} rows, {_n_with_seq} with sequences, "
                f"targets: {_detected_targets}\n\n"
                f"Columns: {', '.join(f'`{c}`' for c in _avail_cols)}"
            )
            return

        model, metrics = train_wetlab_model(
            X=train_ds["X"],
            y=train_ds["y"],
            target_names=train_ds["target_names"],
        )

        if metrics.get("status") == "success":
            st.session_state.wetlab_model_trained = True
            st.session_state.wetlab_training_metrics = metrics
            st.success(
                f"Model trained on **{train_ds['n_samples']}** samples "
                f"({train_ds['n_skipped']} skipped)"
            )

            # M30: Audit — model training
            audit_log_model_training(
                model_type="XGBoost_WetLab",
                n_samples=train_ds["n_samples"],
                metrics=metrics,
            )

            # Show quick metrics
            for tname, tm in metrics.get("per_target", {}).items():
                r2 = tm.get("r2", 0)
                rmse = tm.get("rmse", 0)
                r2_color = SUCCESS if r2 > 0.7 else (WARN if r2 > 0.4 else ERROR)
                st.markdown(f"""
                <div style="display:inline-block; margin-right:20px; padding:8px 16px;
                            background:{r2_color}15; border-left:4px solid {r2_color};
                            border-radius:6px;">
                    <b>{tname}</b><br>
                    R² = <b style="color:{r2_color}">{r2:.3f}</b>
                    &nbsp;|&nbsp; RMSE = {rmse:.3f}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Training failed")


def _train_plm_model(data: List[Dict[str, Any]]) -> None:
    """Train the PLM-embedding-based wet-lab model and update session state.

    M30: Uses chunked embedding extraction with real-time progress bar
    and FDA audit trail logging.
    """
    from src.data_pipeline import process_jain_dataset
    from src.ml_predictor import train_plm_wetlab_model

    n_total = len(data)

    # -- Phase 1: Chunked embedding extraction with real-time progress ---
    progress_bar = st.progress(0, text="Initializing ESM-2 embedder...")
    status_text = st.empty()

    def _progress_cb(current: int, total: int) -> None:
        pct = current / max(total, 1)
        progress_bar.progress(pct, text=f"Extracting ESM-2 Embeddings: {current}/{total} completed...")
        status_text.caption(
            f"Processing chunk {(current - 1) // 10 + 1} "
            f"(sequences {current}/{total})"
        )

    plm_ds = process_jain_dataset(
        data,
        target_columns=["Exp_Aggregation_Percent", "Exp_Tm_MeltingTemp"],
        progress_cb=_progress_cb,
    )
    progress_bar.progress(1.0, text="Embedding extraction complete!")
    status_text.empty()

    if plm_ds["status"] != "success":
        progress_bar.empty()
        st.error(f"Failed to build PLM dataset: {plm_ds.get('message', 'Unknown error')}")
        return

    st.info(
        f"Extracted **{plm_ds['n_samples']}** x {plm_ds['embed_dim']}-dim "
        f"embeddings ({plm_ds['mode']} mode, {plm_ds['n_skipped']} skipped)"
    )

    # M30: Audit — PLM embedding batch
    audit_log_plm_embedding(
        n_sequences=plm_ds["n_samples"],
        embed_dim=plm_ds["embed_dim"],
        mode=plm_ds["mode"],
    )

    # -- Phase 2: XGBoost training with spinner ----
    progress_bar.progress(0, text="Training XGBoost on PLM embeddings...")
    with st.spinner("Training XGBoost on high-dimensional PLM embeddings..."):
        model, metrics = train_plm_wetlab_model(
            X=plm_ds["X"],
            y=plm_ds["y"],
            target_names=plm_ds["target_names"],
            feature_names=plm_ds["feature_names"],
        )
    progress_bar.empty()

    if metrics.get("status") == "success":
        st.session_state.plm_model_trained = True
        st.session_state.plm_training_metrics = metrics
        st.success(
            f"PLM Foundation Model trained on **{plm_ds['n_samples']}** samples "
            f"({plm_ds['embed_dim']}-dim embeddings)"
        )

        # M30: Audit — model training
        audit_log_model_training(
            model_type="XGBoost_PLM",
            n_samples=plm_ds["n_samples"],
            metrics=metrics,
        )

        # Show quick metrics
        for tname, tm in metrics.get("per_target", {}).items():
            r2 = tm.get("r2", 0)
            rmse = tm.get("rmse", 0)
            r2_color = SUCCESS if r2 > 0.7 else (WARN if r2 > 0.4 else ERROR)
            st.markdown(f"""
            <div style="display:inline-block; margin-right:20px; padding:8px 16px;
                        background:{r2_color}15; border-left:4px solid {r2_color};
                        border-radius:6px;">
                <b>{tname}</b> (PLM)<br>
                R² = <b style="color:{r2_color}">{r2:.3f}</b>
                &nbsp;|&nbsp; RMSE = {rmse:.3f}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("PLM model training failed")


def _render_wetlab_model_metrics() -> None:
    """Render detailed model metrics in the Metrics tab."""
    metrics = st.session_state.wetlab_training_metrics
    if not metrics or not st.session_state.wetlab_model_trained:
        st.info("No wet-lab model trained yet. Upload a CSV in the Data & Model Training tab to begin.")
        return

    st.markdown(f"**Model Type:** {metrics.get('model_type', 'XGBoost')}")
    st.markdown(f"**Training Samples:** {metrics.get('n_samples', 0)}")

    ds_info = st.session_state.wetlab_dataset_info or {}
    if ds_info:
        st.caption(f"Dataset: {ds_info.get('filename', 'unknown')} | "
                   f"Schema: {ds_info.get('schema', 'unknown')} | "
                   f"Agg data: {ds_info.get('n_with_agg', 0)} | "
                   f"Tm data: {ds_info.get('n_with_tm', 0)}")

    # Per-target metrics
    st.markdown("---")
    st.markdown("#### Per-Target Performance")

    per_target = metrics.get("per_target", {})
    for tname, tm in per_target.items():
        r2 = tm.get("r2", 0)
        rmse = tm.get("rmse", 0)
        mae = tm.get("mae", 0)
        train_rmse = tm.get("train_rmse", 0)

        r2_color = SUCCESS if r2 > 0.7 else (WARN if r2 > 0.4 else ERROR)
        grade = "Excellent" if r2 > 0.8 else ("Good" if r2 > 0.6 else ("Fair" if r2 > 0.4 else "Poor"))

        st.markdown(f"""
        <div class="cqa-card" style="border-left: 4px solid {r2_color};">
            <div class="cqa-title">{tname}</div>
            <div style="display:flex; gap:24px; align-items:center;">
                <div>
                    <div style="font-size:2rem; font-weight:700; color:{r2_color};">
                        R² = {r2:.3f}
                    </div>
                    <span class="status-pill status-{'ok' if r2 > 0.6 else ('warn' if r2 > 0.3 else 'err')}">{grade}</span>
                </div>
                <div style="font-size:0.85rem; color:{SLATE};">
                    Val RMSE: <b>{rmse:.3f}</b><br>
                    Train RMSE: <b>{train_rmse:.3f}</b><br>
                    MAE: <b>{mae:.3f}</b><br>
                    N(train): {tm.get('n_train', 0)} | N(val): {tm.get('n_val', 0)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance
    importances = metrics.get("feature_importances", [])
    if importances:
        st.markdown("---")
        st.markdown("#### Feature Importance")

        try:
            import plotly.graph_objects as go

            for t_idx, fi_dict in enumerate(importances):
                if not fi_dict:
                    continue
                tname = metrics["target_names"][t_idx] if t_idx < len(metrics.get("target_names", [])) else f"Target {t_idx}"

                sorted_fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
                feat_names = [x[0] for x in sorted_fi]
                feat_vals = [x[1] for x in sorted_fi]

                fig = go.Figure(go.Bar(
                    x=feat_vals, y=feat_names,
                    orientation="h",
                    marker_color=ACCENT,
                ))
                _apply_pharma_theme(fig,
                    title=f"Feature Importance: {tname}",
                    xaxis_title="Importance",
                    yaxis=dict(autorange="reversed"),
                    height=280,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"fi_chart_{t_idx}")
        except ImportError:
            # Fallback: text display
            for t_idx, fi_dict in enumerate(importances):
                if fi_dict:
                    tname = metrics["target_names"][t_idx] if t_idx < len(metrics.get("target_names", [])) else f"Target {t_idx}"
                    st.markdown(f"**{tname}:**")
                    for fname, fval in sorted(fi_dict.items(), key=lambda x: -x[1]):
                        bar_len = int(fval * 200)
                        st.markdown(f"  `{fname:20s}` {'█' * bar_len} {fval:.3f}")

    # Wet-lab model status badge
    st.markdown("---")
    st.markdown(f"""
    <div style="background:{SUCCESS}15; border:1px solid {SUCCESS};
                border-radius:8px; padding:12px; text-align:center;">
        <span style="font-size:1.1rem; font-weight:700; color:{SUCCESS};">
            PASS: Wet-Lab Model Active
        </span>
        <br>
        <span style="font-size:0.8rem; color:{SLATE};">
            The Auto-Optimize pipeline now uses supervised predictions
            to validate variants before presenting them.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
#  3D. 3D Structural Twin & Spatial Liabilities (M9)
# ===========================================================================

def render_3d_structural_panel(intent: Dict[str, Any], dev_result: Optional[Dict[str, Any]] = None) -> None:
    """
    Render the 3D Structural Twin tab showing an interactive py3Dmol viewer
    with liability residues highlighted on the structure.
    """
    with st.expander("3D Structural Twin & Spatial Liabilities", expanded=True):
        st.markdown(
            '<div style="font-size:0.85rem; color:#64748B; margin-bottom:8px;">'
            'Interactive 3D viewer with liability mapping — '
            'Red: High-risk | Yellow: Moderate | Orange: SHAP hotspots | Cyan: CDRs'
            '</div>',
            unsafe_allow_html=True,
        )

        # Note about mAb Y-shape limitation
        st.caption(
            "**Structural model limitation:** The 3D viewer uses a single-chain backbone representation "
            "with liability mapping. Full mAb Y-shape (2×HC + 2×LC tetramer) requires experimental PDB "
            "coordinates (X-ray/cryo-EM) or AlphaFold-Multimer prediction, which is not yet integrated. "
            "The spatial liability assessment below remains valid for individual chain analysis."
        )

        try:
            from src.visualizer import (
                extract_liability_residues,
                map_shap_to_residues,
                build_3d_viewer,
                render_3d_in_streamlit,
                build_liability_legend,
                generate_liability_summary,
            )

            # Extract SHAP hotspots if available
            shap_hotspots = []
            if dev_result and dev_result.get("status") == "success":
                shap_data = dev_result.get("data", {}).get("shap", {})
                shap_hotspots = map_shap_to_residues(
                    shap_result=shap_data,
                    intent=intent,
                    top_n=5,
                )

            # Extract all liability residues
            liabilities = extract_liability_residues(intent, shap_hotspots=shap_hotspots)

            if not liabilities:
                st.info("No chain analyses available for 3D mapping. Provide a multi-chain FASTA for full structural visualization.")
                return

            # Build and render 3D viewer
            viewer = build_3d_viewer(
                pdb_data=None,  # Use default 1IGT mock
                liability_residues=liabilities,
                width=700,
                height=480,
            )
            render_3d_in_streamlit(viewer, height=480)

            # Legend
            legend_html = build_liability_legend()
            st.markdown(legend_html, unsafe_allow_html=True)

            # Liability summary cards
            summary = generate_liability_summary(liabilities)
            if summary["total"] > 0:
                st.markdown(f"**{summary['total']} liabilities mapped** — "
                            f"{len(summary['high_risk'])} high-risk, "
                            f"{len(summary['moderate_risk'])} moderate-risk")

                # Detail table
                cols = st.columns(2)
                with cols[0]:
                    if summary["high_risk"]:
                        st.markdown("**High Risk Residues**")
                        for r in summary["high_risk"][:8]:
                            st.markdown(
                                f'<span style="color:#EF4444; font-weight:600;">'
                                f'{r["chain"]}:{r["resnum"]} — {r["label"]}</span>',
                                unsafe_allow_html=True,
                            )
                with cols[1]:
                    if summary["moderate_risk"]:
                        st.markdown("**Moderate Risk Residues**")
                        for r in summary["moderate_risk"][:8]:
                            st.markdown(
                                f'<span style="color:#F59E0B; font-weight:600;">'
                                f'{r["chain"]}:{r["resnum"]} — {r["label"]}</span>',
                                unsafe_allow_html=True,
                            )
            else:
                st.success("No liabilities detected. Clean sequence profile.")

        except Exception as viz_err:
            st.caption(f"(3D visualization unavailable: {viz_err})")
            st.info(
                "Install py3Dmol and stmol for interactive 3D protein visualization:\n\n"
                "`pip install py3Dmol stmol`"
            )


# ===========================================================================
#  3D-b. 3D Structural Liability Assessment — SASA-Based (M25)
# ===========================================================================

def render_3d_sasa_liability_panel(intent: Dict[str, Any]) -> None:
    """
    Render the SASA-based 3D Structural Liability Assessment panel (M25).
    Uses ESMFold structure prediction + Shrake-Rupley SASA to reclassify
    1D sequence liabilities as Buried (Safe) or Exposed (High Risk).
    """
    with st.expander("3D Structural Liability Assessment (SASA)", expanded=True):
        st.markdown(
            '<div style="font-size:0.85rem; color:#64748B; margin-bottom:8px;">'
            'SASA-aware liability filtering — buried motifs (&lt;10 Å²) are safe; '
            'only solvent-exposed liabilities pose real developability risk'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            from src.structural_twin import (
                run_structural_analysis,
                run_structural_analysis_multi_chain,
                generate_3d_viewer_html,
                StructuralResult,
            )

            # --- Collect chain sequences and 1D liabilities ---
            chain_analyses = intent.get("chain_analyses", [])
            results: list = []

            if chain_analyses:
                # Multi-chain mode — iterate each chain
                for i, ca in enumerate(chain_analyses):
                    seq = ca.get("sequence", "")
                    chain_name = ca.get("chain_name", f"Chain {i+1}")
                    liab_dict = ca.get("liabilities", {})
                    if not seq:
                        continue
                    sr = run_structural_analysis(
                        sequence=seq,
                        liabilities_dict=liab_dict if liab_dict else None,
                        chain_name=chain_name,
                    )
                    results.append(sr)
            elif intent.get("sequence"):
                # Single sequence fallback
                seq = intent["sequence"]
                liab_dict = scan_sequence_liabilities(seq)
                sr = run_structural_analysis(
                    sequence=seq,
                    liabilities_dict=liab_dict,
                    chain_name="Query",
                )
                results.append(sr)

            if not results:
                st.info("No sequences available for 3D structural liability assessment.")
                return

            # Build (chain_name, StructuralResult) pairs for display
            chain_names: list = []
            if chain_analyses:
                for i, ca in enumerate(chain_analyses):
                    if ca.get("sequence"):
                        chain_names.append(ca.get("chain_name", f"Chain {i+1}"))
            else:
                chain_names = ["Query"]

            # --- Summary metrics ---
            total_1d = sum(r.n_total_1d for r in results)
            total_retained = sum(r.n_retained_3d for r in results)
            total_buried = sum(r.n_removed_buried for r in results)
            pct_removed = (total_buried / total_1d * 100) if total_1d > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("1D Liabilities", f"{total_1d}")
            m2.metric("Exposed (Risk)", f"{total_retained}", delta=None)
            m3.metric("Buried (Safe)", f"{total_buried}",
                       delta=f"-{pct_removed:.0f}% removed" if total_buried > 0 else None,
                       delta_color="normal")
            m4.metric("Net Reduction", f"{pct_removed:.0f}%")

            # --- Per-chain liability table ---
            for idx, sr in enumerate(results):
                cname = chain_names[idx] if idx < len(chain_names) else f"Chain {idx+1}"
                n_res = len(sr.sasa_per_residue)
                st.markdown(f"##### {cname} ({n_res} residues)")
                st.caption(f"Structure: {sr.pdb_source} | "
                           f"Mean SASA: {sr.mean_sasa:.1f} Å² | "
                           f"Buried: {sr.n_buried} | Exposed: {sr.n_exposed}")

                if sr.liabilities_3d:
                    rows = []
                    for la in sr.liabilities_3d:
                        is_safe = la.status == "Buried (Safe)"
                        status_emoji = "PASS" if is_safe else "FLAG"
                        rec = "No action needed (buried)" if is_safe else "Monitor — solvent-exposed liability"
                        rows.append({
                            "Status": status_emoji,
                            "Type": la.liability_type,
                            "Motif": la.motif,
                            "Position": la.position + 1,  # 1-indexed
                            "SASA (Å²)": f"{la.sasa:.1f}",
                            "Classification": la.status,
                            "Recommendation": rec,
                        })
                    import pandas as pd
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.success(f"No liabilities identified in {cname}.")

            # --- Cache results for 3D viewer persistence across reruns ---
            st.session_state["_sasa_results"] = results
            st.session_state["_sasa_chain_names"] = chain_names

            # --- 3D PDB viewer (always shown below the table) ---
            _sasa_has_pdb = any(sr.pdb_data for sr in results)
            if _sasa_has_pdb:
                st.markdown("##### 3D Structure Viewer")
                for idx, sr in enumerate(results):
                    cname = chain_names[idx] if idx < len(chain_names) else f"Chain {idx+1}"
                    if sr.pdb_data:
                        _exposed = [la.position for la in sr.liabilities_3d
                                    if la.status != "Buried (Safe)"]
                        _buried = [la.position for la in sr.liabilities_3d
                                   if la.status == "Buried (Safe)"]
                        viewer_html = generate_3d_viewer_html(
                            pdb_data=sr.pdb_data,
                            exposed_residues=_exposed,
                            buried_residues=_buried,
                            width=700,
                            height=450,
                        )
                        st.components.v1.html(viewer_html, height=470, scrolling=False)

        except Exception as sasa_err:
            st.caption(f"(3D SASA liability assessment unavailable: {sasa_err})")


# ===========================================================================
#  3E. Model Validation & Robustness Panel (M9)
# ===========================================================================

def render_validation_robustness_panel(intent: Dict[str, Any]) -> None:
    """
    Render the ML validation and robustness panel showing time-based
    and batch-shift validation results with confidence intervals.
    """
    with st.expander("Model Validation & Robustness", expanded=True):
        st.markdown(
            '<div style="font-size:0.85rem; color:#64748B; margin-bottom:8px;">'
            'Production-grade validation: temporal drift, batch-shift robustness, '
            'and bootstrap confidence intervals'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            from src.validation_strategy import (
                validate_chromatography_model,
                validate_developability_model,
            )

            # --- Chromatography MLP Validation ---
            st.markdown("##### Chromatography MLP (ka/nu Prediction)")

            # Try to use actual model if available
            chrom_report = None
            try:
                from src.ml_predictor import get_trained_model
                model, X_train, _ = get_trained_model()
                from src.validation_strategy import run_validation_report
                import torch

                def predict_fn(X):
                    with torch.no_grad():
                        X_t = torch.FloatTensor(X)
                        return model(X_t).numpy()

                y_train = predict_fn(X_train)
                chrom_report = run_validation_report(
                    predict_fn, X_train, y_train,
                    model_name="ChromatographyMLP (ka/nu)"
                )
            except Exception:
                chrom_report = validate_chromatography_model()

            if chrom_report and "error" not in chrom_report:
                _render_validation_report_card(chrom_report)
            else:
                st.info("Chromatography model validation unavailable.")

            st.markdown("---")

            # --- Developability XGBoost Validation ---
            st.markdown("##### Developability Predictor (XGBoost)")

            dev_report = None
            try:
                from src.developability_predictor import get_predictor
                predictor = get_predictor()
                dev_report = validate_developability_model(predictor=predictor)
            except Exception:
                dev_report = validate_developability_model()

            if dev_report and "error" not in dev_report:
                _render_validation_report_card(dev_report)
            else:
                st.info("Developability model validation unavailable.")

        except Exception as val_err:
            st.caption(f"(Model validation panel unavailable: {val_err})")


def _render_validation_report_card(report: Dict[str, Any]) -> None:
    """Render a single validation report as styled cards."""
    # Grade badge
    grade = report.get("grade", "Unknown")
    grade_color_map = {
        "Production-Ready": ("#065F46", "#ECFDF5", "#A7F3D0"),
        "Acceptable": ("#92400E", "#FFFBEB", "#FDE68A"),
        "Needs Improvement": ("#EF4444", "#FEF2F2", "#FECACA"),
    }
    fg, bg, border = grade_color_map.get(grade, ("#334155", "#F3F4F6", "#E2E8F0"))

    st.markdown(f"""
    <div style="display:inline-block; padding:4px 12px; border-radius:12px;
                background:{bg}; border:1px solid {border};
                font-weight:600; color:{fg}; font-size:0.9rem;">
        {grade}
    </div>
    """, unsafe_allow_html=True)

    # Overall metrics
    overall = report.get("overall_metrics", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R-squared", f"{overall.get('r2', 0):.4f}")
    m2.metric("RMSE", f"{overall.get('rmse', 0):.4f}")
    m3.metric("MAE", f"{overall.get('mae', 0):.4f}")
    m4.metric("MAPE", f"{overall.get('mape', 0):.2f}%")

    # Confidence intervals
    ci = report.get("confidence_intervals", {})
    ci_rmse = ci.get("rmse", {})
    if ci_rmse:
        st.markdown(
            f"**95% CI (RMSE):** [{ci_rmse.get('ci_lower', 0):.4f}, "
            f"{ci_rmse.get('ci_upper', 0):.4f}]"
        )

    # Time-based drift
    time_data = report.get("time_based", {})
    drift = time_data.get("drift_ratio", 1.0)
    drift_status = "ok" if drift < 1.5 else ("warn" if drift < 2.0 else "err")
    drift_label = "Stable" if drift < 1.5 else ("Moderate drift" if drift < 2.0 else "Significant drift")

    # Batch-shift robustness
    batch_data = report.get("batch_shift", {})
    shift = batch_data.get("shift_ratio", 1.0)
    shift_status = "ok" if shift < 1.5 else ("warn" if shift < 2.0 else "err")
    shift_label = "Robust" if shift < 1.5 else ("Moderate shift" if shift < 2.0 else "Sensitive to shifts")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="padding:8px 12px; border-radius:6px; background:#F9FAFB; border:1px solid #E2E8F0;">
            <div style="font-weight:600; font-size:0.85rem;">Temporal Drift</div>
            <div style="font-size:1.1rem; font-weight:700;">{drift:.2f}x
                <span class="status-pill status-{drift_status}" style="font-size:0.75rem;">
                    {drift_label}</span>
            </div>
            <div style="font-size:0.75rem; color:#64748B;">
                {time_data.get('n_splits', 0)} forward-chaining splits</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="padding:8px 12px; border-radius:6px; background:#F9FAFB; border:1px solid #E2E8F0;">
            <div style="font-weight:600; font-size:0.85rem;">Batch Robustness</div>
            <div style="font-size:1.1rem; font-weight:700;">{shift:.2f}x
                <span class="status-pill status-{shift_status}" style="font-size:0.75rem;">
                    {shift_label}</span>
            </div>
            <div style="font-size:0.75rem; color:#64748B;">
                {batch_data.get('n_batches', 0)} simulated batches</div>
        </div>
        """, unsafe_allow_html=True)

    # Per-split detail (collapsible)
    time_results = time_data.get("results", [])
    if time_results:
        with st.expander("Time-Split Details", expanded=True):
            for tr in time_results:
                if "metrics" in tr:
                    m = tr["metrics"]
                    st.markdown(
                        f"**{tr['description']}** — "
                        f"R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}"
                    )

    batch_results = batch_data.get("results", [])
    if batch_results:
        with st.expander("Batch-Shift Details", expanded=True):
            for br in batch_results:
                if "metrics" in br:
                    m = br["metrics"]
                    st.markdown(
                        f"**{br['description']}** — "
                        f"R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}"
                    )


# ===========================================================================
#  3F. Expert Labeling Panel (M10)
# ===========================================================================

def render_expert_labeling_panel(intent: Dict[str, Any], ml_prediction: Optional[Dict[str, Any]] = None) -> None:
    """Render inline expert labeling form for human-in-the-loop corrections."""
    with st.expander("Expert Label / Correct Prediction", expanded=True):
        st.markdown(
            '<div style="font-size:0.85rem; color:#64748B; margin-bottom:8px;">'
            'Review and correct the model prediction. Corrections are saved for '
            'Continuous Learning retraining.'
            '</div>',
            unsafe_allow_html=True,
        )

        pred_rt = "N/A"
        pred_ka = "N/A"
        pred_nu = "N/A"
        if ml_prediction:
            pred_rt = f"{ml_prediction.get('estimated_rt_min', 'N/A')}"
            pred_ka = f"{ml_prediction.get('ka', 'N/A')}"
            pred_nu = f"{ml_prediction.get('nu', 'N/A')}"

        st.markdown(f"**Model predicted:** RT={pred_rt} min, ka={pred_ka}, nu={pred_nu}")

        col1, col2 = st.columns(2)
        with col1:
            actual_rt = st.number_input(
                "Actual RT (min)",
                min_value=0.0, max_value=60.0,
                value=float(ml_prediction.get("estimated_rt_min", 15.0)) if ml_prediction else 15.0,
                step=0.1,
                key=f"label_rt_{id(intent)}",
                help=(
                    "Enter the experimentally observed retention time (minutes) from your IEX run. "
                    "This human-in-the-loop correction trains the ML model to better predict future RT values."
                ),
            )
        with col2:
            label_tag = st.text_input(
                "Tag / Note",
                value="",
                placeholder="e.g., Exp batch 2024-12",
                key=f"label_tag_{id(intent)}",
                help=(
                    "Optional tag for this correction. Use to track experiment batch, "
                    "column type, or run conditions (e.g., 'SP-HP 20240815')."
                ),
            )

        if st.button("Save Correction", key=f"save_label_{id(intent)}", type="primary"):
            features = [
                intent.get("pI", 8.0),
                intent.get("mw", 150.0),
                float(intent.get("deam_sites", 1)),
                float(intent.get("ox_sites", 1)),
                40.0, 50.0,
                intent.get("hydrophobicity", 0.35),
            ]
            st.session_state.label_store.add_label(
                feature_vector=features,
                predicted_value=pred_rt,
                actual_value=actual_rt,
                metric_type="RT",
                tag=label_tag or f"Correction for {intent.get('name', 'mAb')}",
                source="manual",
            )
            # Also store in workspace
            ws_store.add_labeled_data({
                "predicted_value": pred_rt,
                "actual_value": actual_rt,
                "tag": label_tag,
                "metric_type": "RT",
            })
            st.success(f"Saved correction: RT {pred_rt} -> {actual_rt} min")


# ===========================================================================
#  3G. Continuous Learning Dashboard Tab (M10)
# ===========================================================================

def render_continuous_learning_tab() -> None:
    """Render the Model Training & Accuracy Tracker dashboard."""
    st.markdown("### Model Training & Accuracy Tracker")
    st.markdown(
        "This dashboard tracks how your expert corrections improve prediction accuracy over time. "
        "Each time you correct a model prediction in the Copilot Chat (e.g., adjusting a predicted "
        "retention time or aggregation score), that correction is stored as an **expert label**. "
        "When you click 'Retrain Model Now', the system retrains the chromatography MLP using your "
        "corrections combined with synthetic calibration data."
    )

    cl_engine = st.session_state.cl_engine
    label_store = st.session_state.label_store

    # --- Stats row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Labels", label_store.count,
              help="Number of expert corrections stored. Each label = one prediction you manually corrected.")
    c2.metric("Training Runs", cl_engine.run_counter,
              help="How many times you have retrained the model. Each run incorporates all accumulated labels.")

    improvement = cl_engine.get_improvement_data()
    if improvement["n_runs"] > 0:
        c3.metric("Latest Val Loss", f"{improvement['val_loss'][-1]:.6f}",
                  help="Validation loss (MSE) of the most recent training run. Lower = better. "
                       "This is evaluated on a held-out portion of the training data.")
        _imp_pct = improvement['improvement_pct']
        c4.metric("Improvement", f"{_imp_pct:.1f}%",
                  help="Percentage reduction in validation loss compared to the first training run. "
                       "Positive = model is getting more accurate over time.")
    else:
        c3.metric("Latest Val Loss", "N/A",
                  help="No training runs yet — retrain to see validation loss.")
        c4.metric("Improvement", "N/A")

    # --- What happens when you reset / go back to baseline ---
    if cl_engine.run_counter > 0:
        with st.expander("What does each run change?", expanded=True):
            st.markdown(
                "**Each retrain run:**\n"
                "- Generates 500 synthetic training samples (physics-based, calibrated to RT 15-20 min window)\n"
                "- Merges your expert labels (if any) with the synthetic data\n"
                "- Retrains the Chromatography MLP neural network from scratch\n"
                "- Evaluates on a 20% held-out validation set\n"
                "- Reports validation loss (MSE) — lower means predictions match reality better\n\n"
                "**If you use 'Factory Reset' (in Data & Model Training tab):**\n"
                "- All trained models are deleted (WetLab XGBoost, Potency XGBoost, Chromatography MLP)\n"
                "- System reverts to **Baseline Heuristics** — simple rule-based formulas with no ML\n"
                "- Baseline heuristics use fixed weights: 30% pI + 30% hydrophobicity + 25% liability count + 15% MW\n"
                "- This is the lowest accuracy tier; recommended to at least run Jain-137 calibration after reset"
            )

    # --- Retrain button ---
    st.markdown("---")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        n_epochs = st.slider("Epochs for retraining", 20, 200, 80, step=10, key="cl_epochs",
                             help=(
                                 "Number of training epochs for retraining the MLP model with expert corrections. "
                                 "More epochs = better fit to your labels but risk overfitting. "
                                 "80 epochs is a good default; increase to 150+ if you have many corrections."
                             ))
    with col_b:
        st.markdown("")
        st.markdown("")
        retrain_clicked = st.button("Retrain Model Now", type="primary", key="retrain_btn")

    if retrain_clicked:
        with st.spinner("Retraining XGBoost model with expert-curated labels (continual learning cycle)..."):
            X_labeled, y_labeled = label_store.get_training_data("RT")
            if X_labeled is not None and len(X_labeled) > 0:
                # Convert RT targets to 2-col (ka, nu approx)
                y_2col = np.column_stack([
                    np.full(len(y_labeled), 1.5) + y_labeled * 0.05,
                    np.full(len(y_labeled), 3.5) + y_labeled * 0.02,
                ])
                result = cl_engine.retrain_model(
                    labeled_features=X_labeled,
                    labeled_targets=y_2col,
                    n_synthetic=500,
                    epochs=n_epochs,
                )
            else:
                result = cl_engine.retrain_model(
                    n_synthetic=500,
                    epochs=n_epochs,
                )

        if result.get("status") == "success":
            st.success(
                f"Retraining complete (Run #{result['run_id']}): "
                f"Val Loss = {result['final_val_loss']:.6f}, "
                f"{result['n_labeled']} expert labels used"
            )
        else:
            st.error(f"Retraining failed: {result.get('message', 'Unknown error')}")

    # --- Charts ---
    st.markdown("---")

    # Epoch-level loss chart (latest run)
    latest_history = cl_engine.get_latest_epoch_history()
    if latest_history:
        fig_epoch = build_loss_over_epochs_chart(
            latest_history,
            title=f"Training Loss — Run #{cl_engine.run_counter}",
        )
        if fig_epoch:
            st.plotly_chart(fig_epoch, use_container_width=True)
        else:
            st.info("Install plotly for interactive charts: `pip install plotly`")

    # Run-over-run improvement chart
    improvement = cl_engine.get_improvement_data()
    if improvement["n_runs"] >= 2:
        fig_improve = build_improvement_over_runs_chart(improvement)
        if fig_improve:
            st.plotly_chart(fig_improve, use_container_width=True)

    # --- Label Store Details ---
    if label_store.count > 0:
        with st.expander(f"Expert Label Store ({label_store.count} entries)", expanded=True):
            summary = label_store.get_summary()
            st.markdown(f"**By type:** {summary['by_type']}")
            st.markdown(f"**By source:** {summary['by_source']}")

            # Show recent labels
            for lb in label_store.labels[-10:]:
                st.markdown(
                    f"- `{lb['timestamp']}` | {lb['metric_type']} | "
                    f"Predicted: {lb['predicted_value']} -> Actual: {lb['actual_value']} "
                    f"| *{lb['tag']}*"
                )

    elif cl_engine.run_counter == 0:
        st.info(
            "No training runs yet. Add expert labels via the Copilot Chat tab "
            "or upload a CSV with historical data, then click 'Retrain Model Now'."
        )


# ===========================================================================
#  3C. Bispecific Antibody Rendering (M11)
# ===========================================================================

def render_bispecific_species_panel(bispec_result: Dict[str, Any]) -> None:
    """Render the 3-species comparison panel for bispecific analysis."""
    data = bispec_result.get("data", bispec_result)
    species = data.get("species", {})
    peaks = data.get("peaks", {})
    sma = data.get("sma_params", {})

    st.markdown("#### Assembly Species Comparison")
    cols = st.columns(3)
    for col, key in zip(cols, ("AA", "AB", "BB")):
        sp = species.get(key, {})
        pk = peaks.get(key, {})
        sm = sma.get(key, {})
        is_target = sp.get("is_target", False)
        border_color = "#10B981" if is_target else "#EF4444"
        badge = "TARGET" if is_target else "IMPURITY"

        with col:
            st.markdown(f"""
            <div class="cqa-card" style="border-left: 4px solid {border_color};">
                <div class="cqa-title">{sp.get('display_name', key)}
                    <span class="status-pill status-{'ok' if is_target else 'err'}">{badge}</span>
                </div>
                <div style="font-size:0.9rem; color:#334155;">
                    pI = <b>{sp.get('pI', 'N/A')}</b><br>
                    MW = <b>{sp.get('mw_kda', 'N/A')}</b> kDa<br>
                    h = <b>{sp.get('hydrophobicity', 'N/A')}</b><br>
                    Seq = {sp.get('seq_length', 'N/A')} aa
                </div>
                <hr style="margin:4px 0;">
                <div style="font-size:0.85rem; color:#64748B;">
                    RT = <b>{pk.get('rt_min', 'N/A')}</b> min | FWHM = {pk.get('fwhm_min', 'N/A')} min
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_bispecific_chromatogram(bispec_result: Dict[str, Any]) -> None:
    """Render the 3-component chromatogram for bispecific analysis."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Install plotly for bispecific chromatogram: `pip install plotly`")
        return

    data = bispec_result.get("data", bispec_result)
    chrom = data.get("chromatogram", {})
    species = data.get("species", {})
    peaks_data = data.get("peaks", {})

    t = chrom.get("time", [])
    signals = chrom.get("signals", {})
    gradient = chrom.get("gradient", [])
    total = chrom.get("total", [])

    if not t:
        st.info("No chromatogram data available.")
        return

    fig = go.Figure()

    # Individual species traces
    colors = {"AA": "#EF4444", "AB": "#10B981", "BB": "#3B82F6"}
    labels = {
        "AA": species.get("AA", {}).get("display_name", "Homodimer AA"),
        "AB": species.get("AB", {}).get("display_name", "Heterodimer AB"),
        "BB": species.get("BB", {}).get("display_name", "Homodimer BB"),
    }

    for key in ("AA", "AB", "BB"):
        sig = signals.get(key, [])
        if sig:
            dash = "solid" if key == "AB" else "dash"
            width = 3 if key == "AB" else 2
            fig.add_trace(go.Scatter(
                x=t, y=sig,
                name=labels[key],
                mode="lines",
                line=dict(color=colors[key], width=width, dash=dash),
                fill="tozeroy" if key == "AB" else None,
                opacity=0.8 if key == "AB" else 0.6,
            ))

    # Total signal
    if total:
        fig.add_trace(go.Scatter(
            x=t, y=total,
            name="Total Signal",
            mode="lines",
            line=dict(color="#1F2937", width=1.5, dash="dot"),
            opacity=0.5,
        ))

    # Salt gradient on secondary y-axis
    if gradient:
        fig.add_trace(go.Scatter(
            x=t, y=gradient,
            name="Salt Gradient (mM)",
            mode="lines",
            line=dict(color="#94A3B8", width=1, dash="dashdot"),
            yaxis="y2",
            opacity=0.4,
        ))

    # Add vertical lines for peak positions
    for key in ("AA", "AB", "BB"):
        pk = peaks_data.get(key, {})
        rt = pk.get("rt_min")
        if rt:
            fig.add_vline(
                x=rt,
                line_dash="dot",
                line_color=colors[key],
                opacity=0.5,
                annotation_text=f"{key}: {rt:.1f}",
                annotation_position="top",
            )

    _apply_pharma_theme(fig,
        title="Bispecific Separation Chromatogram (AA / AB / BB)",
        xaxis_title="Time (min)",
        yaxis_title="UV Signal (mAU, normalized)",
        yaxis2=dict(
            title="NaCl (mM)",
            overlaying="y",
            side="right",
            range=[0, 600],
            showgrid=False,
        ),
        height=450,
        legend=dict(orientation="h", y=-0.15),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_bispecific_risk_panel(bispec_result: Dict[str, Any]) -> None:
    """Render the homodimer co-elution risk assessment panel."""
    data = bispec_result.get("data", bispec_result)
    resolution = data.get("resolution", {})
    risk = data.get("risk", {})

    risk_level = risk.get("risk_level", "Unknown")
    risk_colors = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
    risk_color = risk_colors.get(risk_level, "#64748B")

    st.markdown("#### Downstream Manufacturability Risk")

    # Resolution metrics
    r1, r2, r3 = st.columns(3)
    with r1:
        rs_ab_aa = resolution.get("rs_AB_AA", 0)
        pill = "ok" if rs_ab_aa >= 1.5 else ("warn" if rs_ab_aa >= 0.8 else "err")
        st.markdown(f"""
        <div class="cqa-card">
            <div class="cqa-title">Rs (AB vs AA)</div>
            <div class="cqa-value">{rs_ab_aa:.3f}</div>
            <span class="status-pill status-{pill}">
                {'Baseline' if rs_ab_aa >= 1.5 else ('Partial' if rs_ab_aa >= 0.8 else 'Overlap')}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        rs_ab_bb = resolution.get("rs_AB_BB", 0)
        pill = "ok" if rs_ab_bb >= 1.5 else ("warn" if rs_ab_bb >= 0.8 else "err")
        st.markdown(f"""
        <div class="cqa-card">
            <div class="cqa-title">Rs (AB vs BB)</div>
            <div class="cqa-value">{rs_ab_bb:.3f}</div>
            <span class="status-pill status-{pill}">
                {'Baseline' if rs_ab_bb >= 1.5 else ('Partial' if rs_ab_bb >= 0.8 else 'Overlap')}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="cqa-card" style="border-left: 4px solid {risk_color};">
            <div class="cqa-title">Co-elution Risk</div>
            <div class="cqa-value" style="color: {risk_color};">{risk_level}</div>
            <span class="status-pill status-{'ok' if risk_level == 'Low' else ('warn' if risk_level == 'Medium' else 'err')}">
                Min Rs = {resolution.get('min_rs', 0):.3f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Risk details
    for detail in risk.get("risk_details", []):
        if risk_level == "High":
            st.error(detail)
        elif risk_level == "Medium":
            st.warning(detail)
        else:
            st.success(detail)

    # Recommendations
    recs = risk.get("recommendations", [])
    if recs:
        with st.expander("Engineering Recommendations", expanded=(risk_level == "High")):
            for i, rec in enumerate(recs, 1):
                st.markdown(f"**{i}.** {rec}")


# ===========================================================================
#  4. Sidebar (M10: Workspace Management + File Upload + Export)
# ===========================================================================

with st.sidebar:
    st.markdown('<div class="copilot-header">ProtePilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="copilot-sub">Protein Developability Pilot Platform &middot; v32.1</div>', unsafe_allow_html=True)

    # -- Workshop Navigation (grouped: Development → Discovery & AI) ---------
    st.markdown("---")
    st.markdown(
        '<span style="color:#64748B;font-size:0.75rem;font-weight:600;letter-spacing:0.05em;">'
        'DEVELOPMENT</span>',
        unsafe_allow_html=True,
    )
    # -- Callback helpers to clear the OTHER radio when one is clicked ----------
    def _on_dev_nav_change():
        """When user clicks a Development page, clear Discovery selection."""
        st.session_state["nav_data_page"] = None

    def _on_data_nav_change():
        """When user clicks a Discovery/AI page, deselect Development radio."""
        # Streamlit radios can't truly deselect; instead we use a sentinel.
        st.session_state["_data_page_active"] = True

    # Initialise sentinel if absent
    if "_data_page_active" not in st.session_state:
        st.session_state["_data_page_active"] = False

    active_page = st.radio(
        "Workshop",
        [
            "Molecule Setup",
            "Developability Dashboard",
            "Analytical & Characterization",
            "Process Development",
            "Preclinical & Clinical",
        ],
        key="nav_page",
        label_visibility="collapsed",
        on_change=_on_dev_nav_change,
    )
    # Advanced section (collapsed by default)
    st.markdown(
        '<span style="color:#64748B;font-size:0.75rem;font-weight:600;letter-spacing:0.05em;">'
        'ADVANCED</span>',
        unsafe_allow_html=True,
    )
    _data_page = st.radio(
        "Advanced",
        [
            "Model Management",
        ],
        key="nav_data_page",
        label_visibility="collapsed",
        index=None,
        on_change=_on_data_nav_change,
    )
    # Coordinate the two radios: if advanced page selected, override active_page
    if _data_page is not None:
        active_page = _data_page
        st.session_state["_data_page_active"] = True
    else:
        st.session_state["_data_page_active"] = False

    # -- Molecule Badge (read-only summary) ---------------------------------
    _badge_ws = ws_store.get_active() if ws_store else None
    _badge_intent = _badge_ws.get("intent") if _badge_ws else None
    if _badge_intent and isinstance(_badge_intent, dict):
        _badge_name = _badge_intent.get("name", "")
        _badge_class = _badge_intent.get("molecule_class", "unknown")
        _badge_chains = (
            _badge_intent.get("n_chains")
            or _badge_intent.get("n_chains_detected")
            or _badge_intent.get("n_chains_total")
            or _badge_intent.get("num_chains")
            or (len(_badge_intent["chains"]) if isinstance(_badge_intent.get("chains"), list) and _badge_intent.get("chains") else None)
            or "?"
        )
        _badge_mw = _badge_intent.get("total_mw")
        _badge_mw_str = f" | {_badge_mw/1000:.1f} kDa" if _badge_mw else ""
        st.markdown(
            f'<div style="background:#F0F4FF;border-left:4px solid #3B82F6;'
            f'padding:8px 10px;margin:8px 0;border-radius:6px;font-size:0.85rem;">'
            f'<b>{_badge_name or "Unnamed"}</b><br/>'
            f'<span style="color:#64748B;">{_badge_class} | {_badge_chains} chains{_badge_mw_str}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # -- Workspaces ---------------------------------------------------------
    st.markdown("---")
    st.markdown("**Workspaces**")

    ws_list = ws_store.list_workspaces()
    if ws_list:
        for ws_info in ws_list[:10]:  # Show last 10
            col_ws, col_del = st.columns([5, 1])
            with col_ws:
                btn_label = ws_info["display_name"]
                if ws_info["is_active"]:
                    btn_label = f"**{btn_label}**"
                if st.button(
                    btn_label,
                    key=f"ws_select_{ws_info['id']}",
                    use_container_width=True,
                ):
                    ws_store.set_active(ws_info["id"])
                    # Sync messages from workspace
                    active_ws = ws_store.get_active()
                    if active_ws:
                        st.session_state.messages = active_ws["messages"]
                        # M21: Sync global intent for cross-page context
                        if active_ws.get("intent"):
                            st.session_state.last_intent = active_ws["intent"]
                    # Invalidate all molecule-bound state on workspace switch
                    # to prevent stale results from leaking across molecules.
                    _ws_intent = active_ws.get("intent") if active_ws else None
                    _invalidate_molecule_bound_state(ws_store=ws_store, new_intent=_ws_intent)
                    st.rerun()
            with col_del:
                if st.button(
                    "X",
                    key=f"ws_del_{ws_info['id']}",
                    help="Delete this workspace",
                ):
                    ws_store.delete_workspace(ws_info["id"])
                    # Full state reset — clear all molecule-bound data
                    _invalidate_molecule_bound_state(ws_store=ws_store, new_intent=None)
                    st.session_state.messages = []
                    st.session_state.pending_input = None
                    st.session_state.pending_assembly = None
                    st.session_state.pending_bispecific = None
                    st.session_state.run_count = 0
                    st.rerun()
    else:
        st.caption("No workspaces yet. Start a chat to create one.")

    # -- Global Report Export --------------------------------------------------
    st.markdown("---")
    st.markdown("**Report Export**")

    # Resolve intent + analysis_cache from workspace store (primary)
    # then fall back to session_state
    _rpt_ws = ws_store.get_active() if ws_store else None
    _rpt_intent = (
        (_rpt_ws.get("intent") if _rpt_ws else None)
        or st.session_state.get("last_intent")
    )
    _rpt_cache = (
        (_rpt_ws.get("analysis_cache") if _rpt_ws else None)
        or {}
    )
    _has_analysis = bool(_rpt_intent and _rpt_cache)

    if not _has_analysis:
        st.caption("Run an analysis first to enable report export.")
    else:
        _export_col1, _export_col2 = st.columns(2)
        with _export_col1:
            _btn_docx = st.button(
                "Export DOCX",
                key="btn_export_docx",
                use_container_width=True,
                help="Generate a comprehensive Global Analysis Report as a Word document.",
            )
        with _export_col2:
            _btn_json = st.button(
                "Export JSON",
                key="btn_export_json",
                use_container_width=True,
                help="Export the structured report data as JSON for programmatic use.",
            )

        if _btn_docx or _btn_json:
            with st.spinner("Generating report..."):
                try:
                    from src.report_export import export_global_report
                    import tempfile, shutil, os
                    from datetime import datetime

                    _rpt_extras = {
                        "glycoform_profile": st.session_state.get("glycoform_profile", "standard_cho"),
                        "upstream_result_dict": st.session_state.get("upstream_result_dict", {}),
                        "ada_result": (
                            st.session_state.get("ada_result")
                            or (_rpt_ws.get("ada_result") if _rpt_ws else None)
                            or {}
                        ),
                    }

                    _rpt_out_dir = tempfile.mkdtemp(prefix="pharma_report_")
                    _rpt_docx_path, _rpt_json_path = export_global_report(
                        intent=_rpt_intent,
                        analysis_cache=_rpt_cache,
                        session_extras=_rpt_extras,
                        output_dir=_rpt_out_dir,
                    )

                    _mol_label = (_rpt_intent.get("name", "molecule") if isinstance(_rpt_intent, dict) else "molecule").replace(" ", "_")
                    _ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # ── Auto-save to Reports/ subfolder ──
                    _reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reports")
                    os.makedirs(_reports_dir, exist_ok=True)
                    _saved_files = []
                    try:
                        if _rpt_docx_path and os.path.exists(_rpt_docx_path):
                            _auto_docx = os.path.join(_reports_dir, f"ProtePilot_Report_{_mol_label}_{_ts_label}.docx")
                            shutil.copy2(_rpt_docx_path, _auto_docx)
                            _saved_files.append(_auto_docx)
                        if _rpt_json_path and os.path.exists(_rpt_json_path):
                            _auto_json = os.path.join(_reports_dir, f"ProtePilot_Report_{_mol_label}_{_ts_label}.json")
                            shutil.copy2(_rpt_json_path, _auto_json)
                            _saved_files.append(_auto_json)
                    except Exception as _save_err:
                        st.warning(f"Auto-save to Reports/ folder encountered an issue: {_save_err}")

                    # Store in session state for download
                    if _btn_docx:
                        with open(_rpt_docx_path, "rb") as _df:
                            _docx_bytes = _df.read()
                        st.session_state["_exec_report_bytes"] = _docx_bytes
                        st.session_state["_exec_report_name"] = f"ProtePilot_Report_{_mol_label}.docx"
                        st.success("DOCX report generated!")

                    if _btn_json:
                        with open(_rpt_json_path, "r") as _jf:
                            _json_bytes = _jf.read().encode("utf-8")
                        st.session_state["_json_report_bytes"] = _json_bytes
                        st.session_state["_json_report_name"] = f"ProtePilot_Report_{_mol_label}.json"
                        st.success("JSON report generated!")

                    if _saved_files:
                        st.caption(f"Auto-saved to `Reports/` folder: {', '.join(os.path.basename(f) for f in _saved_files)}")

                except Exception as _rpt_err:
                    st.error(f"Report generation failed: {_rpt_err}")
                    import traceback
                    st.code(traceback.format_exc(), language="text")

        # Download buttons (appear after generation)
        if st.session_state.get("_exec_report_bytes"):
            st.download_button(
                label=f"Download {st.session_state.get('_exec_report_name', 'report.docx')}",
                data=st.session_state["_exec_report_bytes"],
                file_name=st.session_state.get("_exec_report_name", "ProtePilot_Report.docx"),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_docx_report",
                use_container_width=True,
            )

        if st.session_state.get("_json_report_bytes"):
            st.download_button(
                label=f"Download {st.session_state.get('_json_report_name', 'report.json')}",
                data=st.session_state["_json_report_bytes"],
                file_name=st.session_state.get("_json_report_name", "ProtePilot_Report.json"),
                mime="application/json",
                key="dl_json_report",
                use_container_width=True,
            )

    st.markdown("---")
    # Dark mode toggle — sidebar bottom, collapses with sidebar
    _dm_col1, _dm_col2 = st.columns([3, 2])
    with _dm_col1:
        st.caption("ProtePilot v32.1")
    with _dm_col2:
        _dm_val = st.toggle("Dark", value=st.session_state.get("dark_mode", False), key="dark_mode_toggle", label_visibility="visible")
        if _dm_val != st.session_state.get("dark_mode", False):
            st.session_state.dark_mode = _dm_val
            st.rerun()


# ===========================================================================
#  5. Main Content Area (M20: Page-Based Workshop Navigation)
# ===========================================================================


# ===========================================================================
#  5-SETUP. Molecule Setup (Global Entry Point)
# ===========================================================================
if active_page == "Molecule Setup":
    st.markdown("## Molecule Setup")
    st.caption("Define your molecule: upload FASTA or manually specify chains and assembly.")

    # ── Current Molecule Context (read-only summary if already set) ──────────
    _setup_ws = ws_store.get_active() if ws_store else None
    _setup_intent = _setup_ws.get("intent") if _setup_ws else None
    if _setup_intent and isinstance(_setup_intent, dict) and _setup_intent.get("name"):
        st.markdown(
            f'<div style="background:#F0FFF4;border-left:4px solid #10B981;'
            f'padding:12px 16px;margin:8px 0 16px 0;border-radius:6px;">'
            f'<b style="font-size:1.1rem;">{_setup_intent.get("name", "Unnamed")}</b><br/>'
            f'<span style="color:#64748B;">Class: {_setup_intent.get("molecule_class", "unknown")} '
            f'| Chains: {_setup_intent.get("n_chains") or _setup_intent.get("n_chains_detected") or _setup_intent.get("n_chains_total") or _setup_intent.get("num_chains") or (len(_setup_intent["chains"]) if isinstance(_setup_intent.get("chains"), list) and _setup_intent.get("chains") else "?")} '
            f'| MW: {_setup_intent.get("total_mw", 0)/1000:.1f} kDa</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption("Molecule context is set. Navigate to Characterization or other modules to continue analysis.")
        st.markdown("---")

    # ── Protein Assembly (chain definition) ──────────────────────────────────
    st.markdown("### Define Assembly")
    st.caption(
        "Add chains with their sequences and copy numbers. "
        "Standard IgG: 2x HC + 2x LC. Bispecific: 1 per arm. "
        "Single-domain: 1 chain, 1 copy."
    )

    # Initialize visibility flag
    if "show_assembly_chains" not in st.session_state:
        st.session_state.show_assembly_chains = len(st.session_state.stoich_chains) > 0 and any(
            ch.get("sequence", "").strip() for ch in st.session_state.stoich_chains
        )

    if not st.session_state.show_assembly_chains:
        if st.button("Add Chain", key="initial_add_chain", type="primary"):
            st.session_state.stoich_chains = [
                {"name": "Chain1", "sequence": "", "copy_number": 1},
            ]
            st.session_state.show_assembly_chains = True
            st.rerun()
    else:
        _chains_to_delete = None
        for i in range(len(st.session_state.stoich_chains)):
            ch = st.session_state.stoich_chains[i]
            _hdr_left, _hdr_right = st.columns([4, 1])
            with _hdr_left:
                st.markdown(f"**Chain {i+1}**")
            with _hdr_right:
                if st.button("X", key=f"del_chain_{i}",
                             help=f"Remove {ch.get('name', f'Chain{i+1}')}"):
                    _chains_to_delete = i
            c_name, c_copy = st.columns([3, 1])
            with c_name:
                new_name = st.text_input(
                    "Name", value=ch.get("name", f"Chain{i+1}"),
                    key=f"stoich_name_{i}",
                    help="Label for this chain (e.g., HC, LC, ScFv, Fc).",
                )
            with c_copy:
                new_copy = st.number_input(
                    "Copies", min_value=1, max_value=10,
                    value=int(ch.get("copy_number", 1)),
                    key=f"stoich_copy_{i}",
                    help="Stoichiometric copy number.",
                )
            new_seq = st.text_area(
                f"Sequence for {new_name}",
                value=ch.get("sequence", ""),
                height=100,
                key=f"stoich_seq_{i}",
                placeholder=f">Chain{i+1}\nEVQLVESGGG...",
                help="Paste FASTA (with >header) or raw amino acid sequence.",
            )
            st.session_state.stoich_chains[i] = {
                "name": new_name,
                "sequence": new_seq.strip(),
                "copy_number": new_copy,
            }

        if _chains_to_delete is not None:
            st.session_state.stoich_chains.pop(_chains_to_delete)
            if len(st.session_state.stoich_chains) == 0:
                st.session_state.stoich_chains = []
                st.session_state.show_assembly_chains = False
            st.rerun()

        if st.button("Add Chain", key="add_chain_btn", type="primary"):
            idx = len(st.session_state.stoich_chains) + 1
            st.session_state.stoich_chains.append(
                {"name": f"Chain{idx}", "sequence": "", "copy_number": 1}
            )
            st.rerun()

        # Assembly summary
        total_seqs = sum(1 for ch in st.session_state.stoich_chains if ch["sequence"].strip())
        if total_seqs > 0:
            stoich_desc = " + ".join(
                f"{ch['name']}(x{ch['copy_number']})"
                for ch in st.session_state.stoich_chains if ch["sequence"].strip()
            )
            st.caption(f"Assembly: {stoich_desc}")

        if st.button("Analyze Assembly", key="run_assembly_btn", type="primary"):
            valid = [ch for ch in st.session_state.stoich_chains if ch["sequence"].strip()]
            if len(valid) >= 1:
                # Auto-assembly: 1 HC + 1 LC with copies=1 → infer standard IgG tetramer
                _auto_hc = [c for c in valid if c.get("chain_type", c.get("name", "")).upper() in ("HC", "HEAVY")]
                _auto_lc = [c for c in valid if c.get("chain_type", c.get("name", "")).upper() in ("LC", "LIGHT")]
                if (len(_auto_hc) == 1 and len(_auto_lc) == 1
                        and _auto_hc[0].get("copy_number", 1) == 1
                        and _auto_lc[0].get("copy_number", 1) == 1):
                    _auto_hc[0]["copy_number"] = 2
                    _auto_lc[0]["copy_number"] = 2
                    st.info(
                        "Auto-assembly: detected 1 HC + 1 LC. "
                        "Inferred standard IgG tetramer (2HC + 2LC)."
                    )
                st.session_state.pending_assembly = valid
                st.rerun()
            else:
                st.warning("Enter at least one chain sequence.")

    # ── FASTA File Upload ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Upload FASTA")
    uploaded_fasta = st.file_uploader(
        "Upload a FASTA file",
        type=["fasta", "fa", "faa", "txt"],
        key="fasta_uploader",
        help="Upload a .fasta file for automated sequence analysis.",
    )
    if uploaded_fasta is not None:
        fasta_result = parse_fasta_file(uploaded_fasta, uploaded_fasta.name)
        if fasta_result["status"] == "success" and fasta_result["n_sequences"] > 0:
            fasta_text = fasta_result["combined_text"]
            st.session_state.uploaded_fasta_text = fasta_text
            _fasta_chains = parse_fasta_sequence(fasta_text)
            _n_chains = _fasta_chains.get("num_chains", 1) if _fasta_chains else 1

            if _n_chains > 1:
                st.info(f"Detected **{_n_chains} chains** in the uploaded FASTA.")
                if "fasta_assembly_chains" not in st.session_state or st.session_state.get("_fasta_assembly_file") != uploaded_fasta.name:
                    _init_chains = []
                    for ch in _fasta_chains.get("chains", []):
                        _ct = ch.get("chain_type", "unknown").upper()
                        _default_copies = 2 if _ct in ("HC", "HEAVY", "LC", "LIGHT") else 1
                        _init_chains.append({
                            "name": ch.get("name", f"Chain_{len(_init_chains)+1}"),
                            "chain_type": _ct,
                            "sequence_length": len(ch.get("sequence", "")),
                            "copy_number": _default_copies,
                        })
                    st.session_state.fasta_assembly_chains = _init_chains
                    st.session_state._fasta_assembly_file = uploaded_fasta.name

                with st.form("fasta_assembly_form"):
                    st.markdown("**Define Assembly Stoichiometry**")
                    _updated_chains = []
                    for i, ch in enumerate(st.session_state.fasta_assembly_chains):
                        _cols = st.columns([3, 2, 1])
                        with _cols[0]:
                            st.text(f"{ch['name']} ({ch['chain_type']}, {ch['sequence_length']} aa)")
                        with _cols[1]:
                            st.text(f"Type: {ch['chain_type']}")
                        with _cols[2]:
                            _copies = st.number_input(
                                "Copies", min_value=1, max_value=10,
                                value=ch["copy_number"], key=f"fasta_asm_copies_{i}",
                            )
                        _updated_chains.append({**ch, "copy_number": _copies})
                    _asm_desc = " + ".join(f"{ch['name']}(x{ch['copy_number']})" for ch in _updated_chains)
                    st.caption(f"Assembly: {_asm_desc}")
                    _submit = st.form_submit_button("Confirm Assembly", type="primary")
                    if _submit:
                        st.session_state.fasta_assembly_chains = _updated_chains
                        st.session_state.fasta_assembly_confirmed = True
                        _stoich_update = []
                        for ch_info, ch_data in zip(_updated_chains, _fasta_chains.get("chains", [])):
                            _stoich_update.append({
                                "name": ch_info["name"],
                                "sequence": ch_data.get("sequence", ""),
                                "copy_number": ch_info["copy_number"],
                                "chain_type": ch_info["chain_type"],
                            })
                        st.session_state.stoich_chains = _stoich_update
                        st.session_state.show_assembly_chains = True
                        st.rerun()
            else:
                # Single-chain upload — show Confirm Assembly form (same as multi-chain)
                _single_chain = _fasta_chains.get("chains", [{}])[0] if _fasta_chains and _fasta_chains.get("chains") else {}
                _sc_name = _single_chain.get("name", "Chain_1")
                _sc_type = _single_chain.get("chain_type", "unknown").upper()
                _sc_len = len(_single_chain.get("sequence", ""))
                _sc_default_copies = 2 if _sc_type in ("HC", "HEAVY", "LC", "LIGHT") else 1

                st.success(f"Parsed 1 sequence from {uploaded_fasta.name}")

                # Initialize single-chain assembly if needed
                if "fasta_assembly_chains" not in st.session_state or st.session_state.get("_fasta_assembly_file") != uploaded_fasta.name:
                    st.session_state.fasta_assembly_chains = [{
                        "name": _sc_name,
                        "chain_type": _sc_type,
                        "sequence_length": _sc_len,
                        "copy_number": _sc_default_copies,
                    }]
                    st.session_state._fasta_assembly_file = uploaded_fasta.name

                with st.form("fasta_single_assembly_form"):
                    st.markdown("**Define Assembly Stoichiometry**")
                    _updated_sc = []
                    for i, ch in enumerate(st.session_state.fasta_assembly_chains):
                        _sc_cols = st.columns([3, 2, 1])
                        with _sc_cols[0]:
                            st.text(f"{ch['name']} ({ch['chain_type']}, {ch['sequence_length']} aa)")
                        with _sc_cols[1]:
                            st.text(f"Type: {ch['chain_type']}")
                        with _sc_cols[2]:
                            _sc_copies = st.number_input(
                                "Copies", min_value=1, max_value=10,
                                value=ch["copy_number"], key=f"fasta_sc_copies_{i}",
                            )
                        _updated_sc.append({**ch, "copy_number": _sc_copies})
                    _asm_desc = " + ".join(f"{c['name']}(x{c['copy_number']})" for c in _updated_sc)
                    st.caption(f"Assembly: {_asm_desc}")
                    _sc_submit = st.form_submit_button("Confirm Assembly", type="primary")
                    if _sc_submit:
                        st.session_state.fasta_assembly_chains = _updated_sc
                        st.session_state.fasta_assembly_confirmed = True
                        st.session_state.stoich_chains = [{
                            "name": _sc_name,
                            "sequence": _single_chain.get("sequence", ""),
                            "copy_number": _updated_sc[0]["copy_number"],
                            "chain_type": _sc_type,
                        }]
                        st.session_state.show_assembly_chains = True
                        st.rerun()
        elif fasta_result["status"] == "error":
            st.error("FASTA parsing failed: check file format.")
    else:
        st.session_state.uploaded_fasta_text = None
        if "fasta_assembly_chains" in st.session_state:
            del st.session_state["fasta_assembly_chains"]
        if "_fasta_assembly_file" in st.session_state:
            del st.session_state["_fasta_assembly_file"]

    # ── Bulk Developability Analysis ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Bulk Developability Analysis")
    st.caption(
        "Analyze multiple molecules in a single batch. "
        "All molecules in one CSV must be the **same type** (e.g., all standard IgG). "
        "For mixed types, run separate batches."
    )

    from src.bulk_schema import BATCH_TYPES, generate_csv_template, parse_bulk_csv
    from src.bulk_runner import run_bulk_analysis
    from src.bulk_summary import (
        export_summary_csv, export_summary_json,
        generate_display_stats, rank_candidates,
        generate_bulk_report_docx, save_bulk_reports,
    )
    import pandas as _pd_bulk

    # Step 1: Select molecule type
    _bulk_type_options = {v.display_name: k for k, v in BATCH_TYPES.items()}
    _bulk_type_label = st.selectbox(
        "Molecule Type",
        options=list(_bulk_type_options.keys()),
        key="bulk_mol_type",
        help="All molecules in the CSV must be this type. This determines column schema and assembly rule.",
    )
    _bulk_type_key = _bulk_type_options[_bulk_type_label]
    _bulk_spec = BATCH_TYPES[_bulk_type_key]

    # Show assembly info
    st.info(
        f"**Assembly:** {_bulk_spec.assembly_description}  \n"
        f"**Required columns:** name, {', '.join(_bulk_spec.required_columns)}"
        + (f"  \n**Optional:** {', '.join(_bulk_spec.optional_columns)}" if _bulk_spec.optional_columns else ""),
    )

    # Download template
    _tmpl_csv = generate_csv_template(_bulk_type_key)
    st.download_button(
        f"Download CSV Template ({_bulk_type_label})",
        data=_tmpl_csv,
        file_name=f"bulk_template_{_bulk_type_key}.csv",
        mime="text/csv",
    )

    # Step 2: Upload CSV
    _bulk_file = st.file_uploader(
        "Upload Batch CSV",
        type=["csv"],
        key="bulk_csv_upload",
        help=f"CSV with columns: name, {', '.join(_bulk_spec.required_columns)}",
    )

    if _bulk_file is not None:
        _raw_csv = _bulk_file.read().decode("utf-8")
        _parse = parse_bulk_csv(_raw_csv, _bulk_type_key)

        if _parse.errors:
            for _err in _parse.errors:
                st.error(_err)
        else:
            # Preview parsed data
            st.markdown(f"**Parsed:** {_parse.n_valid} valid / {_parse.n_total} total molecules")
            if _parse.n_errors > 0:
                st.warning(f"{_parse.n_errors} rows have validation errors and will be skipped.")
            for _w in _parse.warnings:
                st.warning(_w)

            # Preview table
            _preview_data = []
            for _r in _parse.rows[:20]:
                _row_d = {"name": _r.name, "status": "valid" if _r.error is None else "error"}
                for _col in _bulk_spec.required_columns:
                    _seq = _r.sequences.get(_col, "")
                    _row_d[_col] = f"{_seq[:20]}... ({len(_seq)} aa)" if len(_seq) > 20 else _seq
                if _r.error:
                    _row_d["error"] = _r.error
                _preview_data.append(_row_d)
            st.dataframe(_pd_bulk.DataFrame(_preview_data), use_container_width=True, hide_index=True)

            # Step 3: Run analysis
            if st.button("Run Bulk Analysis", type="primary", key="bulk_run_btn"):
                _bulk_progress = st.progress(0, text="Initializing...")

                def _bulk_cb(current, total, name):
                    pct = current / max(total, 1)
                    _bulk_progress.progress(pct, text=f"Analyzing {name} ({current}/{total})")

                with st.spinner(f"Analyzing {_parse.n_valid} molecules..."):
                    _batch_result = run_bulk_analysis(_parse, progress_callback=_bulk_cb)

                _bulk_progress.progress(1.0, text="Complete!")
                st.session_state["bulk_result"] = _batch_result

    # Step 4: Display results
    if "bulk_result" in st.session_state:
        _br = st.session_state["bulk_result"]
        _ds = generate_display_stats(_br)

        st.markdown("---")

        # Close / Clear button — clears bulk state so user can re-upload
        _bulk_hdr_l, _bulk_hdr_r = st.columns([6, 1])
        with _bulk_hdr_l:
            st.markdown("#### Bulk Results")
        with _bulk_hdr_r:
            if st.button("Clear Results", key="btn_bulk_clear", type="secondary"):
                for _bk in ("bulk_result", "bulk_report_paths", "bulk_csv_text",
                            "bulk_csv_parsed", "bulk_analysis_running"):
                    st.session_state.pop(_bk, None)
                st.rerun()

        # Cap per-molecule detail tables at top 5 to keep UI responsive.
        # Full data is always available via CSV/JSON export.
        _BULK_DISPLAY_LIMIT = 5
        _n_total = len(_br.results)
        _results_display = _br.results[:_BULK_DISPLAY_LIMIT]
        _truncated = _n_total > _BULK_DISPLAY_LIMIT
        if _truncated:
            st.caption(
                f"Showing top {_BULK_DISPLAY_LIMIT} of {_n_total} molecules. "
                "Download CSV/JSON for full results."
            )

        # Overview metrics
        _ov = _ds["overview"]
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("Total", _ov["n_total"])
        _m2.metric("Success", _ov["n_success"])
        _m3.metric("Success Rate", _ov["success_rate"])
        _m4.metric("Wall Time", _ov["wall_time"])

        # Score statistics
        if _ds["score_stats"]:
            _ss = _ds["score_stats"]
            _s1, _s2, _s3, _s4 = st.columns(4)
            _s1.metric("Mean Score", f'{_ss["mean"]:.3f}')
            _s2.metric("Median", f'{_ss["median"]:.3f}')
            _s3.metric("Best (Min)", f'{_ss["min"]:.3f}')
            _s4.metric("Worst (Max)", f'{_ss["max"]:.3f}')

        # Grade distribution
        _gd = _ds["grade_distribution"]
        _g1, _g2, _g3 = st.columns(3)
        _g1.metric("Low Risk", _gd["Low"])
        _g2.metric("Medium Risk", _gd["Medium"])
        _g3.metric("High Risk", _gd["High"])

        # Biophysical properties + PTM hotspot table
        with st.expander("Biophysical Properties & PTM Hotspots (per molecule)", expanded=True):
            _bio_data = []
            for _rr in _results_display:
                _bio_data.append({
                    "Name": _rr.name,
                    "Length (aa)": _rr.seq_length or "-",
                    "MW (kDa)": f"{_rr.mw_kda:.1f}" if _rr.mw_kda else "-",
                    "pI": f"{_rr.pI:.2f}" if _rr.pI else "-",
                    "GRAVY": f"{_rr.gravy:.4f}" if _rr.gravy is not None else "-",
                    "Deam Sites": _rr.deam_sites if _rr.deam_sites is not None else "-",
                    "Ox Sites": _rr.ox_sites if _rr.ox_sites is not None else "-",
                    "Cys": _rr.cysteine_count if _rr.cysteine_count is not None else "-",
                    "Acidic (D+E)": _rr.acidic_residues if _rr.acidic_residues is not None else "-",
                    "Basic (K+R+H)": _rr.basic_residues if _rr.basic_residues is not None else "-",
                })
            if _bio_data:
                st.dataframe(_pd_bulk.DataFrame(_bio_data), use_container_width=True, height=250, hide_index=True)

        # Ranking table
        st.markdown("#### Candidate Ranking")
        _sort_col = st.selectbox(
            "Sort by", ["dev_score", "agg_risk", "stability", "viscosity_risk"],
            key="bulk_sort",
        )
        _ranked = rank_candidates(_br, sort_by=_sort_col)
        if _ranked:
            st.dataframe(_pd_bulk.DataFrame(_ranked), use_container_width=True, height=400, hide_index=True)

        # Top candidates
        if _ds["top_candidates"]:
            st.markdown("#### Top 5 Candidates (Lowest Risk)")
            for _tc in _ds["top_candidates"]:
                st.markdown(f"- **{_tc['name']}** -- Score: {_tc['score']:.3f} ({_tc['grade']})")

        # ── Comprehensive Analysis Sections ──────────────────────────
        # Liability / MS Characterization
        _has_analytical = any(r.intact_mass_da is not None for r in _results_display)
        if _has_analytical:
            with st.expander("Liability / MS Characterization (per molecule)", expanded=True):
                _liab_data = []
                for _rr in _results_display:
                    if _rr.intact_mass_da is not None:
                        _liab_data.append({
                            "Name": _rr.name,
                            "Intact Mass (Da)": f"{_rr.intact_mass_da:,.0f}" if _rr.intact_mass_da else "-",
                            "Liability Density": f"{_rr.liability_density:.1f}" if _rr.liability_density is not None else "-",
                            "N Liabilities": _rr.n_liabilities if _rr.n_liabilities is not None else "-",
                            "Top Liabilities": _rr.liability_summary or "-",
                        })
                if _liab_data:
                    st.dataframe(_pd_bulk.DataFrame(_liab_data), use_container_width=True, hide_index=True)

        # Analytical QC (cIEF + CE-SDS)
        _has_qc = any(r.cief_main_pct is not None for r in _results_display)
        if _has_qc:
            with st.expander("Analytical QC: cIEF & CE-SDS (per molecule)", expanded=True):
                _qc_data = []
                for _rr in _results_display:
                    if _rr.cief_main_pct is not None or _rr.ce_sds_purity_pct is not None:
                        _qc_data.append({
                            "Name": _rr.name,
                            "cIEF Main (%)": f"{_rr.cief_main_pct:.1f}" if _rr.cief_main_pct is not None else "-",
                            "cIEF Acidic (%)": f"{_rr.cief_acidic_pct:.1f}" if _rr.cief_acidic_pct is not None else "-",
                            "cIEF Basic (%)": f"{_rr.cief_basic_pct:.1f}" if _rr.cief_basic_pct is not None else "-",
                            "CE-SDS Purity (%)": f"{_rr.ce_sds_purity_pct:.1f}" if _rr.ce_sds_purity_pct is not None else "-",
                            "CE-SDS HMW (%)": f"{_rr.ce_sds_hmw_pct:.1f}" if _rr.ce_sds_hmw_pct is not None else "-",
                            "CE-SDS LMW (%)": f"{_rr.ce_sds_lmw_pct:.1f}" if _rr.ce_sds_lmw_pct is not None else "-",
                        })
                if _qc_data:
                    st.dataframe(_pd_bulk.DataFrame(_qc_data), use_container_width=True, hide_index=True)

        # Preclinical PK + Titer
        _has_pk = any(r.half_life_days is not None for r in _results_display)
        if _has_pk:
            with st.expander("Preclinical PK & Upstream Titer (per molecule)", expanded=True):
                _pk_data = []
                for _rr in _results_display:
                    if _rr.half_life_days is not None:
                        _pk_data.append({
                            "Name": _rr.name,
                            "Half-Life (days)": f"{_rr.half_life_days:.1f}" if _rr.half_life_days is not None else "-",
                            "Clearance (mL/day/kg)": f"{_rr.clearance_ml_day_kg:.2f}" if _rr.clearance_ml_day_kg is not None else "-",
                            "Predicted Titer (g/L)": f"{_rr.predicted_titer_g_L:.2f}" if _rr.predicted_titer_g_L is not None else "-",
                        })
                if _pk_data:
                    st.dataframe(_pd_bulk.DataFrame(_pk_data), use_container_width=True, hide_index=True)

        # Immunogenicity
        _has_imm = any(r.ada_risk_category is not None for r in _results_display)
        if _has_imm:
            with st.expander("Immunogenicity / ADA Risk (per molecule)", expanded=True):
                _imm_data = []
                for _rr in _results_display:
                    if _rr.ada_risk_category is not None:
                        _imm_data.append({
                            "Name": _rr.name,
                            "ADA Risk": _rr.ada_risk_category or "-",
                            "ADA Score": f"{_rr.ada_risk_score:.3f}" if _rr.ada_risk_score is not None else "-",
                            "MHC-II Hotspots": _rr.n_mhcii_hotspots if _rr.n_mhcii_hotspots is not None else "-",
                        })
                if _imm_data:
                    st.dataframe(_pd_bulk.DataFrame(_imm_data), use_container_width=True, hide_index=True)

        # OOD flagged — "Out-of-Distribution" means the molecule's properties
        # fall outside the training data range, so predictions may be unreliable
        if _ds["flagged"]:
            with st.expander(
                f"Out-of-Distribution (OOD) Flagged Molecules ({len(_ds['flagged'])})",
                expanded=True,
            ):
                st.caption(
                    "OOD = Out-of-Distribution. These molecules have biophysical properties "
                    "that deviate significantly from the training dataset. Predictions for "
                    "flagged molecules may have higher uncertainty."
                )
                for _f in _ds["flagged"]:
                    st.markdown(f"- **{_f['name']}**: {_f['details']}")

        # Export section
        st.markdown("---")
        st.markdown("#### Export Reports")

        # Generate all reports button
        if st.button("Generate Reports (CSV + JSON + DOCX)", type="primary", key="bulk_gen_reports"):
            with st.spinner("Generating reports..."):
                _ws_dir = os.path.join("data", "reports")
                _gen_paths = save_bulk_reports(_br, _ws_dir)
                st.session_state["bulk_report_paths"] = _gen_paths
            if _gen_paths:
                st.success(f"Reports saved to data/reports/bulk_analysis/ ({len(_gen_paths)} files)")

        # Download buttons
        _ex1, _ex2, _ex3 = st.columns(3)
        with _ex1:
            _csv_out = export_summary_csv(_br)
            st.download_button(
                "Download CSV",
                data=_csv_out,
                file_name=f"bulk_results_{_br.batch_type}.csv",
                mime="text/csv",
            )
        with _ex2:
            _json_out = export_summary_json(_br)
            st.download_button(
                "Download JSON",
                data=_json_out,
                file_name=f"bulk_results_{_br.batch_type}.json",
                mime="application/json",
            )
        with _ex3:
            _docx_bytes = generate_bulk_report_docx(_br)
            if _docx_bytes:
                st.download_button(
                    "Download DOCX",
                    data=_docx_bytes,
                    file_name=f"bulk_report_{_br.batch_type}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            else:
                st.caption("DOCX generation unavailable")

    # ── Reset AI to Baseline ──────────────────────────────────────────────────
    st.markdown("---")
    if st.button("Reset AI to Baseline", key="btn_factory_reset_top", type="primary",
                  help="Delete all trained model files and reset to baseline heuristics."):
        try:
            from src.ml_predictor import factory_reset
            reset_result = factory_reset()
            for k in ["wetlab_model_trained", "wetlab_training_metrics",
                       "wetlab_dataset_info", "wetlab_csv_data",
                       "potency_model_trained", "potency_training_metrics",
                       "ht_screening_results", "ht_screening_csv",
                       "optimization_result", "doe_result",
                       "_exec_report_bytes", "_exec_report_name"]:
                if k in st.session_state:
                    st.session_state[k] = None
            st.session_state.wetlab_model_trained = False
            st.session_state.potency_model_trained = False
            audit_log_factory_reset()
            st.success(reset_result["message"])
            st.rerun()
        except Exception as rst_err:
            st.error(f"Reset failed: {rst_err}")
# ===========================================================================
#  5A. Analytical & Characterization (formerly Molecular Characterization)
# ===========================================================================
elif active_page == "Analytical & Characterization":
    st.markdown("## Analytical & Characterization")
    st.caption(
        "Sequence liability analysis, virtual analytical QC lab, "
        "mass spectrometry predictions, and charge variant characterization."
    )

    # -- Glycoform selector (visible at top level for easy access) --
    _glyco_opts_ac = {
        "standard_cho": "Standard CHO (G0F/G1F)",
        "high_mannose": "High-Mannose (Man5/Man8)",
        "afucosylated": "Afucosylated (G0, ADCC+)",
        "highly_sialylated": "Highly Sialylated (pI shift)",
        "none_aglycosylated": "None (Aglycosylated)",
    }
    _glyco_keys_ac = list(_glyco_opts_ac.keys())
    _glyco_cur_ac = st.session_state.get("glycoform_profile", "standard_cho")
    _glyco_idx_ac = _glyco_keys_ac.index(_glyco_cur_ac) if _glyco_cur_ac in _glyco_keys_ac else 0
    _glyco_sel_ac = st.selectbox(
        "Host Cell Glycoform Profile",
        options=_glyco_keys_ac,
        format_func=lambda k: _glyco_opts_ac[k],
        index=_glyco_idx_ac,
        key="glycoform_select_ac",
        help=(
            "Glycoform selection affects cIEF charge variants (sialylation → acidic shift), "
            "PK clearance (high-mannose → faster clearance), and overall developability risk."
        ),
    )
    st.session_state.glycoform_profile = _glyco_sel_ac

    # -- Analytical QC Quick-Access (tabs always visible) ----------------------
    _ac_ws = ws_store.get_active() if ws_store else None
    _ac_intent = _ac_ws.get("intent") if _ac_ws else None
    _ac_cache = (_ac_ws.get("analysis_cache") or {}) if _ac_ws else {}
    _ac_qc = _ac_cache.get("analytical_qc") if _ac_cache else None

    _ac_tab_char, _ac_tab_qc, _ac_tab_ms = st.tabs([
        "Sequence & Liability Analysis",
        "Virtual Analytical QC Lab",
        "Mass Spec & Biophysical",
    ])

    with _ac_tab_qc:
        if _ac_intent and isinstance(_ac_intent, dict):
            st.markdown("### Virtual Analytical QC Lab")
            st.caption(
                "Analytical validation plan, editable acceptance criteria, "
                "cIEF charge variants, CE-SDS purity, SEC monomer analysis, and glycan profiling."
            )

            # ── Validation Plan (moved from Tab 1) ────────────────────
            _vp_dev = _ac_cache.get("dev_result", {}) if _ac_cache else {}
            if isinstance(_vp_dev, dict) and _vp_dev.get("status") == "success":
                _vp_data = _vp_dev.get("data", _vp_dev)
                _vp_plan = _vp_data.get("validation_plan")
                if _vp_plan:
                    render_validation_plan_panel(_vp_plan)

            # ── Editable Acceptance Criteria ───────────────────────────
            from src.developability_core import QTPP_ACCEPTANCE_DEFAULTS as _QAD
            with st.expander("Acceptance Criteria (editable)", expanded=True):
                st.caption(
                    "Adjust per-molecule acceptance thresholds for QC assays. "
                    "Changes apply to both the QTPP table and Virtual QC pass/fail below."
                )
                _cr1, _cr2, _cr3, _cr4 = st.columns(4)
                with _cr1:
                    _crit_sec = st.number_input(
                        "SEC Monomer min %", min_value=80.0, max_value=99.9,
                        value=float(st.session_state.get("qc_sec_min") or _QAD["sec_monomer"]["accept_lower"]),
                        step=0.5, key="qc_crit_sec_tab2")
                with _cr2:
                    _crit_cesds = st.number_input(
                        "CE-SDS Intact min %", min_value=80.0, max_value=99.9,
                        value=float(st.session_state.get("qc_cesds_min") or _QAD["cesds_intact"]["accept_lower"]),
                        step=0.5, key="qc_crit_cesds_tab2")
                with _cr3:
                    _crit_acidic = st.number_input(
                        "cIEF Acidic max %", min_value=5.0, max_value=60.0,
                        value=float(st.session_state.get("qc_acidic_max") or _QAD["cief_acidic"]["accept_upper"]),
                        step=1.0, key="qc_crit_acidic_tab2")
                with _cr4:
                    _crit_main = st.number_input(
                        "cIEF Main min %", min_value=30.0, max_value=90.0,
                        value=float(st.session_state.get("qc_main_min") or _QAD["cief_main"]["accept_lower"]),
                        step=1.0, key="qc_crit_main_tab2")
                # Write to unified session_state keys
                st.session_state["qc_sec_min"] = _crit_sec
                st.session_state["qc_cesds_min"] = _crit_cesds
                st.session_state["qc_acidic_max"] = _crit_acidic
                st.session_state["qc_main_min"] = _crit_main

            # Read unified criteria for QC cards and banner
            _qc_sec_min = float(st.session_state.get("qc_sec_min") or _QAD["sec_monomer"]["accept_lower"])
            _qc_cesds_min = float(st.session_state.get("qc_cesds_min") or _QAD["cesds_intact"]["accept_lower"])
            _qc_acidic_max = float(st.session_state.get("qc_acidic_max") or _QAD["cief_acidic"]["accept_upper"])
            _qc_main_min = float(st.session_state.get("qc_main_min") or _QAD["cief_main"]["accept_lower"])

            # Run QC if not cached
            _ac_seq = _ac_intent.get("sequence", "")
            _ac_pI = _ac_intent.get("pI", 7.0)
            _ac_mol_class = _ac_intent.get("molecule_class", "unknown")
            _ac_glyco = st.session_state.get("glycoform_profile", "standard_cho")

            if _ac_qc is None and _ac_seq:
                try:
                    from src.analytical_qc_twin import run_analytical_qc as _run_aqc
                    from src.molecule_classifier import MoleculeClass as _AcMC
                    _ac_agg = 0.0
                    _dev_c = _ac_cache.get("dev_result", {})
                    if isinstance(_dev_c, dict):
                        _dev_d = _dev_c.get("data", _dev_c)
                        _ac_agg_risk = (_dev_d.get("predictions", {}).get("agg_risk", 0) or 0)
                    _ac_agg = _ac_agg_risk * _ac_agg_risk * 20.0  # quadratic — aligned with bulk_runner + Step 1.10b
                    _sial_map = {"standard_cho": 0.0, "highly_sialylated": 0.60,
                                 "afucosylated": 0.0, "high_mannose": 0.0, "none_aglycosylated": 0.0}
                    _ac_is_mab = False
                    try:
                        _ac_is_mab = _AcMC(_ac_mol_class).is_mab_like
                    except (ValueError, KeyError):
                        pass
                    _ac_agg_clamped = max(0.5, min(_ac_agg, 10.0))  # aligned with bulk_runner
                    _ac_qc = _run_aqc(
                        sequence=_ac_seq, pI=float(_ac_pI),
                        aggregation_pct=_ac_agg_clamped,
                        is_mab=_ac_is_mab,
                        sialylation_fraction=_sial_map.get(_ac_glyco, 0.0),
                        molecule_class=_ac_mol_class,
                    )
                    _ac_qc_dict = {
                        "cief": _ac_qc.cief,
                        "ce_sds": _ac_qc.ce_sds,
                        "sec": {"monomer_pct": round(100.0 - _ac_agg, 2),
                                "hmw_pct": round(_ac_agg, 2)},
                        "glycan": _ac_qc.glycan,
                        "overall_qc_pass": _ac_qc.overall_qc_pass,
                        "source": "simulated",
                    }
                    if _ac_ws:
                        if "analysis_cache" not in _ac_ws:
                            _ac_ws["analysis_cache"] = {}
                        _ac_ws["analysis_cache"]["analytical_qc"] = _ac_qc_dict
                    _ac_qc = _ac_qc_dict
                except Exception as _qc_err:
                    st.error(f"Analytical QC error: {_qc_err}")

            if _ac_qc and isinstance(_ac_qc, dict):
                _cief = _ac_qc.get("cief", {})
                _sec = _ac_qc.get("sec", {})
                _cesds = _ac_qc.get("ce_sds", {})
                _glycan = _ac_qc.get("glycan", {})

                # ── QC PASS/FAIL banner — uses USER criteria, not hardcoded spec_pass ──
                _get_val = lambda obj, k: getattr(obj, k, obj.get(k, None) if isinstance(obj, dict) else None)
                _qc_checks = []
                _v_mono_raw = _get_val(_sec, 'monomer_pct')
                _v_intact_raw = _get_val(_cesds, 'intact_pct')
                _v_acidic_raw = _get_val(_cief, 'acidic_pct')
                _v_main_raw = _get_val(_cief, 'main_pct')
                try:
                    if _v_mono_raw is not None:
                        _qc_checks.append(float(_v_mono_raw) >= _qc_sec_min)
                    if _v_intact_raw is not None:
                        _qc_checks.append(float(_v_intact_raw) >= _qc_cesds_min)
                    if _v_acidic_raw is not None:
                        _qc_checks.append(float(_v_acidic_raw) <= _qc_acidic_max)
                    if _v_main_raw is not None:
                        _qc_checks.append(float(_v_main_raw) >= _qc_main_min)
                except (ValueError, TypeError):
                    pass
                _qc_pass = all(_qc_checks) if _qc_checks else True
                if _qc_pass:
                    st.success("QC Status: PASS — All assays within acceptance criteria")
                else:
                    st.error("QC Status: FAIL — One or more assays outside acceptance criteria")

                # ── Color-coded QC metric cards ────────────────────────
                def _qc_card(label, value, pass_condition, col):
                    """Render a QC metric card with left-border color indicator."""
                    try:
                        v = float(value)
                        ok = pass_condition(v)
                    except (ValueError, TypeError):
                        v = None
                        ok = True
                    _border = "#10B981" if ok else "#EF4444"
                    with col:
                        st.markdown(
                            f"<div style='border:1px solid #E2E8F0;border-left:4px solid {_border};"
                            f"background:#F9FAFB;padding:8px 12px;margin:3px 0;border-radius:6px;'>"
                            f"<span style='font-size:0.78rem;font-weight:600;color:#64748B;"
                            f"text-transform:uppercase;letter-spacing:0.05em;'>{label}</span><br/>"
                            f"<span style='font-size:1.4rem;font-weight:700;color:#0F172A;'>{value}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # cIEF
                st.markdown("#### cIEF Charge Variant Analysis")
                _ci1, _ci2, _ci3 = st.columns(3)
                _v_acidic = _v_acidic_raw if _v_acidic_raw is not None else "N/A"
                _v_main = _v_main_raw if _v_main_raw is not None else "N/A"
                _v_basic = _get_val(_cief, 'basic_pct')
                if _v_basic is None:
                    _v_basic = "N/A"
                _qc_card("Acidic %", _v_acidic, lambda v: v <= _qc_acidic_max, _ci1)
                _qc_card("Main Peak %", _v_main, lambda v: v >= _qc_main_min, _ci2)
                _qc_card("Basic %", _v_basic, lambda v: True, _ci3)

                # CE-SDS
                st.markdown("#### CE-SDS Purity Profile")
                _sd1, _sd2, _sd3 = st.columns(3)
                _v_intact = _v_intact_raw if _v_intact_raw is not None else "N/A"
                _v_frag = _get_val(_cesds, 'lmw_pct')
                _v_hmw_cesds = _get_val(_cesds, 'hmw_pct')
                if _v_frag is None:
                    _v_frag = "N/A"
                if _v_hmw_cesds is None:
                    _v_hmw_cesds = "N/A"
                _qc_card("Intact %", _v_intact, lambda v: v >= _qc_cesds_min, _sd1)
                _qc_card("Fragment %", _v_frag, lambda v: True, _sd2)
                _qc_card("HMW %", _v_hmw_cesds, lambda v: True, _sd3)

                # SEC
                st.markdown("#### SEC Monomer Analysis")
                _se1, _se2 = st.columns(2)
                _v_mono = _v_mono_raw if _v_mono_raw is not None else "N/A"
                _v_hmw_sec = _get_val(_sec, 'hmw_pct')
                if _v_hmw_sec is None:
                    _v_hmw_sec = "N/A"
                _qc_card("Monomer %", _v_mono, lambda v: v >= _qc_sec_min, _se1)
                _qc_card("HMW %", _v_hmw_sec, lambda v: True, _se2)

                # Acceptance criteria summary
                st.caption(
                    f"Acceptance criteria — SEC Monomer ≥ {_qc_sec_min:.0f}% · "
                    f"CE-SDS Intact ≥ {_qc_cesds_min:.0f}% · "
                    f"cIEF Acidic ≤ {_qc_acidic_max:.0f}% · "
                    f"cIEF Main ≥ {_qc_main_min:.0f}%  "
                    f"*(Adjust thresholds in the Acceptance Criteria panel above)*"
                )

                # Glycan
                # [REMOVED] N-Linked Glycan Profile section disabled — shows blank
        else:
            st.info("Upload a molecule sequence in **Molecule Setup** to run analytical QC.")

    with _ac_tab_ms:
        if _ac_intent and isinstance(_ac_intent, dict):
            st.markdown("### Mass Spec & Biophysical")
            st.caption("Intact mass predictions, peptide mapping, and biophysical summary metrics.")
            _ms_result = _ac_cache.get("ms_characterization") if _ac_cache else None
            if _ms_result:
                # Render full MS characterization panel (intact mass, peptide map, liability density)
                try:
                    render_ms_characterization_panel(intent=_ac_intent, ms_result=_ms_result)
                except Exception as _ms_render_err:
                    st.caption(f"(MS panel render error: {_ms_render_err})")
                # Also render biophysical summary metrics
                try:
                    render_ms_summary_metrics(intent=_ac_intent)
                except Exception:
                    pass
            else:
                # No cached result — try to generate on the fly if sequence exists
                _ms_seq = _ac_intent.get("sequence", "") if _ac_intent else ""
                if _ms_seq and len(_ms_seq) >= 20:
                    try:
                        _ms_fresh = render_ms_characterization_panel(intent=_ac_intent)
                        if _ms_fresh:
                            # Store in cache for next reload
                            if _ac_ws and "analysis_cache" in _ac_ws:
                                _ac_ws["analysis_cache"]["ms_characterization"] = _ms_fresh
                            elif _ac_ws:
                                _ac_ws["analysis_cache"] = {"ms_characterization": _ms_fresh}
                        render_ms_summary_metrics(intent=_ac_intent)
                    except Exception as _ms_gen_err:
                        st.info(f"Could not generate MS predictions: {_ms_gen_err}")
                else:
                    st.info("Upload a molecule sequence to generate mass spec predictions.")
        else:
            st.info("Upload a molecule sequence in **Molecule Setup** to view mass spec predictions.")

    # -- Analysis Pipeline (always inside Tab 1 "Sequence & Liability Analysis") --
    # Use __enter__() to render all subsequent content inside Tab 1 without
    # re-indenting ~1600 lines.  _ac_tab_char is always created now.
    _ac_tab_char.__enter__()
    st.markdown("### Sequence & Liability Analysis Pipeline")
    st.caption("Paste a FASTA sequence or describe your protein to begin analysis.")

    # -- Display current molecule analysis (no history) -------------------------
    active_ws = ws_store.get_active()

    # -- Input from sidebar FASTA upload (no chat_input needed) ------------------
    # All molecule input is handled via the sidebar FASTA uploader.
    active_input = None
    if st.session_state.pending_input is not None:
        # Sidebar button was clicked — message already in history, process it
        active_input = st.session_state.pending_input

    # Clear pending flag (consumed)
    st.session_state.pending_input = None

    # -- M11+: Re-render cached analysis for historical workspace ---------------
    # If no new input to process and no pending analysis of any kind, check if
    # the active workspace has cached analysis artifacts and re-render them.
    _pending_bispec_check = st.session_state.pending_bispecific
    _pending_asm_check = st.session_state.pending_assembly
    if (active_input is None
        and _pending_bispec_check is None
        and _pending_asm_check is None
        and active_ws is not None
        and active_ws.get("analysis_cache") is not None):
        render_cached_analysis(active_ws, ws_store)

    # -- Process active input ---------------------------------------------------
    if active_input:
        # M10: Create or update workspace for this run
        ws = ws_store.get_active()
        if ws is None:
            name_preview = active_input[:40].replace("\n", " ").strip()
            ws = ws_store.create_new(display_name=name_preview)
            ws_store.save_to_session_state(st.session_state)
        ws_store.add_message_to_active("user", active_input)

        with st.chat_message("user"):
            st.markdown(active_input)

        # -- Parse intent -------------------------------------------------------
        intent = parse_intent(active_input)

        if intent is None:
            # Diagnostic: show why FASTA might have failed
            _diag_is_fasta = is_fasta_input(active_input)
            _diag_msg = (
                f"**Debug:** input length={len(active_input)}, "
                f"is_fasta={_diag_is_fasta}, "
                f"starts_with=`{active_input[:40].replace(chr(10), '↵')}`"
            )
            if _diag_is_fasta:
                _diag_chains = parse_multi_chain_fasta(active_input)
                _diag_msg += f", chains_parsed={len(_diag_chains)}"
                if _diag_chains:
                    try:
                        from Bio.SeqUtils.ProtParam import ProteinAnalysis as _DA
                        _dc = "".join(c["sequence"] for c in _diag_chains)
                        _da = _DA(_dc)
                        _diag_msg += f", biopython_pI={_da.isoelectric_point():.2f}"
                    except Exception as _de:
                        _diag_msg += f", biopython_error={_de}"
            st.caption(_diag_msg)

            response = (
                "I could not extract chromatography parameters from your input. "
                "Please provide at least an isoelectric point (pI), or paste a FASTA sequence.\n\n"
                "**Examples:**\n"
                "- `pI 8.5 mw 150 hydrophobicity 0.35`\n"
                "- Paste a FASTA sequence starting with `>`"
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            ws_store.add_message_to_active("assistant", response)
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            # M10: Store intent in workspace
            ws_store.update_active_field("intent", intent)

            # M21: Persist intent globally for cross-page context
            st.session_state.last_intent = intent

            # M11+: Initialize analysis cache for this run
            _analysis_cache = {
                "mode": "standard",
                "intent": intent,
                "source": intent.get("source", "text"),
                "ml_override": None,
                "source_label": "Unknown",
                "predictor_source": "rule_based",   # v32: tracks actual predictor used
                "predictor_detail": "",              # v32: human-readable model provenance
                "ood_bypass_reason": None,           # v32: if model bypassed, why
                "variants": None,
                "dev_result": None,
                "cqa": None,
                "sim_summary": None,
                "sim_elapsed": None,
                "bispecific_result": None,
                "ms_characterization": None,
                "pk_result": None,              # M13
                "glycoform_profile": st.session_state.glycoform_profile,  # M13
                "glycoform_impact": None,       # M13
                "optimization_result": None,    # M14
            }

            # CRITICAL: Invalidate ALL molecule-bound state (session + workspace)
            # when a new molecule analysis starts.  Prevents stale results from
            # upstream, DoE, ADA, bispecific, etc. from contaminating the new run.
            _invalidate_molecule_bound_state(ws_store=ws_store, new_intent=intent)
            ws_store.update_active_field("analysis_cache", _analysis_cache)

            with st.chat_message("assistant"):
                source = intent.get("source", "text")

            # -- FASTA analysis card ----------------------------------------
                if source == "fasta":
                    st.markdown(f"""
                    <div class="fasta-card">
                        <div class="fasta-title">FASTA Sequence Analysis (Biopython)</div>
                        <b>{intent['name']}</b><br>
                        pI = {intent['pI']} | MW = {intent['mw']} kDa |
                        GRAVY = {intent.get('gravy', 'N/A')} |
                        Hydrophobicity = {intent['hydrophobicity']}<br>
                        Chains: {len(intent.get('chains', []))} |
                        Sequence length: {intent.get('seq_length', 'N/A')} aa
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Molecule Classification Badge (Phase 1) ──
                    _cls_info = intent.get("molecule_class_info", {})
                    _cls_name = _cls_info.get("display_name", "Unclassified")
                    _cls_conf = _cls_info.get("confidence", "Low")
                    _cls_evidence = _cls_info.get("evidence", [])
                    _cls_warnings = _cls_info.get("warnings", [])
                    _conf_color = {"High": "#10B981", "Medium": "#F59E0B", "Low": "#EF4444"}.get(_cls_conf, "#94A3B8")
                    _has_fc = _cls_info.get("has_fc_region", False)
                    _expects_glyco = _cls_info.get("expects_glycosylation", False)

                    _cls_ev_text = " &middot; ".join(
                        e.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        for e in _cls_evidence[:3]
                    )
                    _fc_badge = "<span style='background:#e0e7ff; padding:2px 6px; border-radius:8px; font-size:0.75em; margin-left:6px;'>Fc region</span>" if _has_fc else ""
                    _glyco_badge = "<span style='background:#fef3c7; padding:2px 6px; border-radius:8px; font-size:0.75em; margin-left:6px;'>N-glycosylation expected</span>" if _expects_glyco else ""
                    st.markdown(
                        f"<div style='background:#f0f4f8;border-left:4px solid {_conf_color};"
                        f"padding:10px 14px;border-radius:6px;margin:8px 0 12px 0;'>"
                        f"<span style='font-weight:600;font-size:0.95em;'>Molecule Type: {_cls_name}</span>"
                        f"<span style='background:{_conf_color};color:white;padding:2px 8px;"
                        f"border-radius:8px;font-size:0.78em;margin-left:8px;'>{_cls_conf} confidence</span>"
                        f"{_fc_badge}{_glyco_badge}"
                        f"<br/><span style='font-size:0.82em;color:#64748B;'>{_cls_ev_text}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    if _cls_warnings:
                        for _w in _cls_warnings:
                            st.warning(_w)

                    # Render characterization panel
                    render_characterization_panel(intent)

                # -- Text input parameter summary card --------------------------
                else:
                    st.markdown(f"""
                    <div class="fasta-card">
                        <div class="fasta-title">Protein Parameters (Text Input)</div>
                        <b>{intent.get('name', 'mAb')}</b><br>
                        pI = {intent['pI']} | MW = {intent.get('mw', 150)} kDa |
                        Hydrophobicity = {intent.get('hydrophobicity', 0.35)}<br>
                        Deamidation sites: {intent.get('deam_sites', 1)} |
                        Oxidation sites: {intent.get('ox_sites', 1)} |
                        Gradient: {intent.get('gradient_slope', 15.0)} mM/min
                    </div>
                    """, unsafe_allow_html=True)

                # -- Step 0: ML Prediction (Neural Network Steering) --------
                ml_override = None
                _torch_available = False
                _predictor_source = "rule_based"     # v32: default
                _predictor_detail = "Rule-Based Heuristic (PropertyMapper v5.0)"
                _ood_bypass_reason = None
                try:
                    import torch as _torch_check
                    _torch_available = True
                except ImportError:
                    pass

                if _torch_available:
                    # v32: First, try the trained unified multitask model (.pt)
                    _unified_model_used = False
                    try:
                        import os as _os_step0
                        _um_path = _os_step0.path.join(str(ROOT), "models", "unified_multitask_best.pt")
                        if _os_step0.path.exists(_um_path):
                            # v32: OOD pre-check before invoking unified model
                            _seq = intent.get("sequence", "")
                            _ood_flags = {}
                            _ood_blocked = False
                            if _seq and len(_seq) > 10:
                                try:
                                    from src.developability_predictor import compute_ood_flags
                                    _ood_flags = compute_ood_flags(
                                        sequence=_seq,
                                        pI=intent.get("pI"),
                                        mw_kda=intent.get("mw"),
                                        molecule_class=intent.get("molecule_class"),
                                    )
                                    if _ood_flags.get("is_ood", False):
                                        _max_z = _ood_flags.get("max_z_score", 0)
                                        _warn = _ood_flags.get("warning_message", "")
                                        _flag_details = "; ".join(
                                            f"{f['metric']}={f['value']:.3f} (z={f['z_score']:.1f})"
                                            for f in _ood_flags.get("flags", [])
                                            if f.get("z_score", 0) > 2.0
                                        )
                                        _ood_bypass_reason = (
                                            f"OOD detected (max z={_max_z:.1f}): {_flag_details}"
                                        )
                                        st.warning(
                                            f"**OOD Warning:** This sequence is outside the "
                                            f"training distribution (max z-score: {_max_z:.1f}). "
                                            f"{_warn}  \n"
                                            f"Unified model predictions may be less reliable. "
                                            f"Flagged metrics: {_flag_details}"
                                        )
                                        # Note: we still attempt the model — OOD is a warning,
                                        # not a hard block. The user should be informed.
                                except Exception:
                                    pass  # OOD check failure doesn't block prediction

                            from src.agents import predict_unified_multitask
                            _um_result = predict_unified_multitask(
                                pI=intent["pI"],
                                mw=intent.get("mw", 150.0),
                                hydrophobicity=intent.get("hydrophobicity", 0.35),
                                deam_sites=intent.get("deam_sites", 1),
                                ox_sites=intent.get("ox_sites", 1),
                                sequence=intent.get("sequence", None),
                            )
                            if _um_result["status"] == "success":
                                _um_preds = _um_result["data"].get("predictions", {})
                                _um_ka = _um_preds.get("ka")
                                _um_nu = _um_preds.get("nu")
                                if _um_ka is not None and _um_nu is not None:
                                    ml_override = {"ka": float(_um_ka), "nu": float(_um_nu)}
                                    # v7.3.2: Propagate R² for PropertyMapper quality gate
                                    _um_ml_ov = _um_result["data"].get("ml_override", {})
                                    if "val_r2" in _um_ml_ov:
                                        ml_override["val_r2"] = _um_ml_ov["val_r2"]
                                    _unified_model_used = True
                                    # Get model metadata for provenance
                                    _um_stat = _os_step0.stat(_um_path)
                                    import datetime as _dt_step0
                                    _um_date = _dt_step0.datetime.fromtimestamp(
                                        _um_stat.st_mtime
                                    ).strftime("%Y-%m-%d %H:%M")
                                    _um_size_kb = _um_stat.st_size / 1024
                                    _predictor_source = "unified_multitask"
                                    _predictor_detail = (
                                        f"Unified PyTorch 8-Task Model "
                                        f"(trained {_um_date}, {_um_size_kb:.0f} KB)"
                                    )
                                    if _ood_bypass_reason:
                                        _predictor_detail += f" [OOD: {_ood_bypass_reason}]"
                                    st.success(
                                        f"Unified Model active: ka={ml_override['ka']:.4f}, "
                                        f"nu={ml_override['nu']:.3f} "
                                        f"(model: {_um_date}, {_um_size_kb:.0f} KB)"
                                    )
                            else:
                                _um_msg = _um_result.get("message", "Unknown error")
                                st.info(
                                    f"Unified model returned non-success: {_um_msg}. "
                                    f"Falling back to MLP/heuristic."
                                )
                    except Exception as _um_err:
                        st.caption(f"(Unified model check: {_um_err})")

                    # Fallback to MLP-based prediction if unified model not available
                    if not _unified_model_used:
                        st.markdown("**Step 0:** ML Prediction (PyTorch MLP v2.0)...")
                        try:
                            from src.agents import predict_ml_with_shap
                            ml_result = predict_ml_with_shap(
                                pI=intent["pI"],
                                mw=intent.get("mw", 150.0),
                                deam_sites=intent.get("deam_sites", 1),
                                ox_sites=intent.get("ox_sites", 1),
                                hydrophobicity=intent.get("hydrophobicity", 0.35),
                                sequence=intent.get("sequence", None),
                            )
                            if ml_result["status"] == "success":
                                ml_override = ml_result["data"]["ml_override"]
                                est_rt = ml_result["data"]["prediction"].get("estimated_rt_min", 0)
                                _predictor_source = "mlp_chromatography"
                                _predictor_detail = "PyTorch MLP v2.0 (ChromatographyMLP)"
                                st.success(
                                    f"ML Override ready: ka={ml_override['ka']:.4f}, "
                                    f"nu={ml_override['nu']:.3f}, est. RT={est_rt:.1f} min"
                                )
                            else:
                                st.info("ML prediction returned non-success; using static fallback.")
                        except Exception as ml_err:
                            st.info(f"ML prediction skipped: {ml_err}")
                    else:
                        st.markdown("**Step 0:** Unified 8-Task Model prediction active.")
                else:
                    st.info(
                        "PyTorch not installed — using static PropertyMapper v5.0 formulas. "
                        "To enable ML-First mode, run: `pip install torch shap matplotlib`"
                    )

                # v32: Persist predictor source into analysis cache
                _analysis_cache["predictor_source"] = _predictor_source
                _analysis_cache["predictor_detail"] = _predictor_detail
                _analysis_cache["ood_bypass_reason"] = _ood_bypass_reason

                # v32: Predictor transparency badge
                _pred_color = {
                    "unified_multitask": "#10B981",
                    "mlp_chromatography": "#3B82F6",
                    "rule_based": "#F59E0B",
                }.get(_predictor_source, "#64748B")
                _pred_icon = {
                    "unified_multitask": "Multitask",
                    "mlp_chromatography": "Chromatography",
                    "rule_based": "Rule-based",
                }.get(_predictor_source, "Unknown")
                _ood_html = ('  <span style="color:#F59E0B;font-weight:600;">OOD: </span>' + _ood_bypass_reason) if _ood_bypass_reason else ""
                st.markdown(
                    f'<div style="background:{_pred_color}15; border-left:4px solid {_pred_color}; '
                    f'padding:8px 12px; border-radius:6px; margin:8px 0; font-size:0.9em;">'
                    f'{_pred_icon} <b>Predictor:</b> {_predictor_detail}'
                    f'{_ood_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # -- M13: Glycoform pI Feedback Loop ---------------------------
                _glyco_profile = st.session_state.glycoform_profile
                _glyco_pi_shift = 0.0
                _glyco_impact = None
                try:
                    from src.preclinical_twin import (
                        get_glycoform_pi_shift as _pk_pi_shift,
                        assess_glycoform_pk_impact,
                    )
                    _glyco_pi_shift = _pk_pi_shift(_glyco_profile)
                    _glyco_impact = assess_glycoform_pk_impact(
                        _glyco_profile, base_pi=intent["pI"],
                    )
                    _analysis_cache["glycoform_impact"] = _glyco_impact
                except Exception:
                    pass

                # Apply pI shift from glycoform (sialylation lowers pI)
                _effective_pI = intent["pI"] + _glyco_pi_shift
                if abs(_glyco_pi_shift) > 0.01:
                    st.info(
                        f"Glycoform pI shift: {_glyco_pi_shift:+.1f} units "
                        f"(effective pI: {_effective_pI:.2f}). "
                        f"SMA parameters will use adjusted pI."
                    )

                # -- Step 1: Physical parameter prediction (ML-First Override) --
                source_label = "ML Override" if ml_override else "Static v5.0 Fallback"
                _analysis_cache["ml_override"] = ml_override
                _analysis_cache["source_label"] = "ML OVERRIDE" if ml_override else "STATIC FALLBACK"
                st.markdown(f"**Step 1/2:** Predicting SMA parameters ({source_label})...")

                try:
                    step1 = predict_physical_params(
                        name           = intent.get("name", "mAb"),
                        pI             = _effective_pI,
                        mw             = intent.get("mw", 150.0),
                        hydrophobicity = intent.get("hydrophobicity", 0.35),
                        pH_working     = intent.get("pH_working", 7.0),
                        deam_sites     = intent.get("deam_sites", 1),
                        ox_sites       = intent.get("ox_sites", 1),
                        sequence       = intent.get("sequence", None),
                        ml_override    = ml_override,
                    )
                except Exception as e:
                    step1 = {"status": "error", "message": str(e), "data": {}}

                # Initialize dev_result before conditional block so auto-triggers
                # always have a defined variable regardless of step1 outcome.
                dev_result = None

                if step1["status"] == "success":
                    v = step1["data"]["variants"]
                    param_source = step1["data"].get("source", "unknown")
                    source_badge = "ML OVERRIDE" if param_source == "ml_override" else "STATIC FALLBACK"
                    _analysis_cache["variants"] = v
                    _analysis_cache["source_label"] = source_badge

                    # SMA parameters computed — stored in cache for Downstream Purification tab
                    # (ka, nu, sigma, Keq NOT shown here per Separation of Concerns)

                    # -- ML Predictor & SHAP Explainability --------------------
                    try:
                        render_ml_shap_panel(intent)
                    except Exception as ml_err:
                        st.caption(f"(ML/SHAP panel unavailable: {ml_err})")
                    try:
                        from src.agents import predict_developability_risk

                        # Extract VH/VL sequences from chains
                        _vh_seq, _vl_seq = "", ""
                        for _chain in intent.get("chains", []):
                            _ct = _chain.get("chain_type", "").upper()
                            if _ct in ("HC", "HEAVY") and _chain.get("sequence"):
                                _vh_seq = _chain["sequence"]
                            elif _ct in ("LC", "LIGHT") and _chain.get("sequence"):
                                _vl_seq = _chain["sequence"]

                        st.markdown("**Developability Assessment:** Running pLM + XGBoost risk analysis...")
                        dev_result = predict_developability_risk(
                            pI=intent["pI"],
                            mw=intent.get("mw", 150.0),
                            hydrophobicity=intent.get("hydrophobicity", 0.35),
                            deam_sites=intent.get("deam_sites", 1),
                            ox_sites=intent.get("ox_sites", 1),
                            sequence=intent.get("sequence", None),
                            vh_sequence=_vh_seq,
                            vl_sequence=_vl_seq,
                            feature_set=intent.get("feature_set_obj"),
                            molecule_class=intent.get("molecule_class"),
                        )

                        if dev_result and dev_result.get("status") == "success":
                            # -- OOD Warning Banner (before developability panel) --
                            _ood = dev_result.get("data", {}).get("ood_info", {})
                            if _ood.get("is_ood"):
                                _ood_msg = _ood.get("warning_message", "Sequence may be out-of-distribution.")
                                _ood_conf = _ood.get("confidence", "Low")
                                _ood_color = "#F59E0B" if _ood_conf == "Medium" else "#EF4444"
                                st.markdown(
                                    f'<div style="background:{_ood_color}12;border-left:4px solid {_ood_color};'
                                    f'padding:12px;border-radius:8px;margin:8px 0;">'
                                    f'<span style="font-size:1.05rem;font-weight:700;color:{_ood_color};">'
                                    f'OOD Warning (Confidence: {_ood_conf})</span><br/>'
                                    f'<span style="color:#334155;font-size:0.92rem;">{_ood_msg}</span></div>',
                                    unsafe_allow_html=True,
                                )
                                # Show flagged metrics
                                _ood_flags = [f for f in _ood.get("flags", []) if f.get("z_score", 0) > 2.0]
                                if _ood_flags:
                                    _flag_items = [
                                        f"{f['metric']}: {f['value']:.2f} (z={f['z_score']:.1f}, ref: {f['reference_range']})"
                                        for f in _ood_flags
                                    ]
                                    st.caption("Outlier metrics: " + " | ".join(_flag_items))
                            elif _ood.get("confidence") == "Medium":
                                st.caption(
                                    "Model confidence: Medium -- some biophysical metrics deviate "
                                    "from typical IgG ranges. Predictions may have higher uncertainty."
                                )

                            # v32.1: Persist pipeline OOD into cache so report_assembler
                            # sees the same is_ood as bulk. Previously, only the pre-check
                            # (line ~7196) wrote ood_bypass_reason; if that was skipped or
                            # the pipeline detected OOD independently, the cache was empty.
                            if _ood.get("is_ood") and not _analysis_cache.get("ood_bypass_reason"):
                                _pipeline_max_z = _ood.get("max_z_score", 0)
                                _pipeline_flag_details = "; ".join(
                                    f"{f['metric']}={f['value']:.3f} (z={f['z_score']:.1f})"
                                    for f in _ood.get("flags", [])
                                    if f.get("z_score", 0) > 2.0
                                )
                                _analysis_cache["ood_bypass_reason"] = (
                                    f"OOD detected (max z={_pipeline_max_z:.1f}): {_pipeline_flag_details}"
                                )

                            render_developability_panel(intent, dev_result)
                            render_actionable_insights_panel(dev_result)
                            st.caption(
                                "Note: Predictions are based on XGBoost trained on ~370 public antibodies. "
                                "Scores reflect ranking correlation (Spearman rho), not absolute values. "
                                "stability_slope predictions are currently unreliable (rho=0.11). "
                                "Always validate with experimental data before making development decisions."
                            )
                            # Validation Plan moved to Virtual QC Lab tab (Tab 2)

                            # -- M17: Formulation Digital Twin Feedback Loop --------
                            try:
                                from src.formulation_twin import (
                                    run_formulation_assessment,
                                    BUFFER_CATALOG,
                                    EXCIPIENT_CATALOG,
                                )

                                _form_ph = st.session_state.formulation_buffer_ph
                                _form_buf = st.session_state.formulation_buffer_type
                                _form_exc = st.session_state.formulation_excipients

                                # Get base predictions from dev_result
                                _dev_data = dev_result.get("data", dev_result)
                                _base_preds = _dev_data.get("predictions", {})
                                _base_dict = {
                                    "agg_risk": _base_preds.get("agg_risk", 0.2),
                                    "stability": _base_preds.get("stability", 0.82),
                                    "viscosity_risk": _base_preds.get("viscosity_risk", 0.15),
                                }

                                _form_result = run_formulation_assessment(
                                    pI=intent["pI"],
                                    buffer_ph=_form_ph,
                                    buffer_type=_form_buf,
                                    excipients=_form_exc,
                                    sequence=intent.get("sequence"),
                                    hydrophobicity=intent.get("hydrophobicity", 0.35),
                                    base_predictions=_base_dict,
                                )

                                render_formulation_twin_panel(intent, _form_result, _base_dict)
                            except Exception as form_err:
                                st.caption(f"(Formulation Twin unavailable: {form_err})")

                        elif dev_result:
                            st.caption(f"(Developability assessment returned: {dev_result.get('message', 'N/A')})")
                    except Exception as dev_err:
                        st.caption(f"(Developability assessment unavailable: {dev_err})")

                    # -- Step 1.7: 3D Structural Twin & SASA Liability Assessment ---
                    # Single consolidated 3D viewer (SASA-based — no py3Dmol dependency)
                    try:
                        render_3d_sasa_liability_panel(intent)
                    except Exception as sasa_err:
                        st.caption(f"(3D structural liability panel unavailable: {sasa_err})")

                    # -- Step 1.8: Model Validation & Robustness (M9) -----------
                    try:
                        render_validation_robustness_panel(intent)
                    except Exception as val_err:
                        st.caption(f"(Model validation panel unavailable: {val_err})")

                    # -- M10: Store results in workspace & show labeling UI -----
                    if dev_result and dev_result.get("status") == "success":
                        ws_store.update_active_field("dev_result", dev_result)
                        _analysis_cache["dev_result"] = dev_result
                    # Retrieve ML prediction for labeling panel
                    _ml_pred = None
                    try:
                        from src.ml_predictor import predict_and_explain
                        _ml_result = predict_and_explain(intent)
                        _ml_pred = _ml_result.get("prediction")
                        ws_store.update_active_field("ml_prediction", _ml_pred)
                    except Exception:
                        pass
                    try:
                        render_expert_labeling_panel(intent, ml_prediction=_ml_pred)
                    except Exception as lab_err:
                        st.caption(f"(Expert labeling panel unavailable: {lab_err})")

                    # -- Step 1.9: MS Characterization + Biophysical Summary --
                    # Note: on first run this renders inline. On subsequent loads,
                    # the Mass Spec tab handles rendering (render_cached_analysis
                    # skips MS to avoid Sequence/Liability tab duplication).
                    try:
                        _ms_char_result = render_ms_characterization_panel(intent)
                        if _ms_char_result:
                            _analysis_cache["ms_characterization"] = _ms_char_result
                        render_ms_summary_metrics(intent)
                    except Exception as ms_err:
                        st.caption(f"(Biophysical summary unavailable: {ms_err})")

                    # -- Step 1.10: Preclinical PK Prediction (M13) ------------
                    try:
                        from src.preclinical_twin import predict_human_half_life, check_fcrn_binding_motif
                        from src.analytical_twin import calculate_liability_density as _calc_ld

                        # Compute liability density for PK input
                        _pk_liab_density = 30.0  # default
                        _pk_chains = intent.get("assembly_chains") or intent.get("chains", [])
                        if _pk_chains:
                            _pk_chain_data = [
                                {"sequence": ch.get("sequence", ""), "copy_number": ch.get("copy_number", 1)}
                                for ch in _pk_chains if ch.get("sequence")
                            ]
                            if _pk_chain_data:
                                _ld_result = _calc_ld(_pk_chain_data)
                                _pk_liab_density = _ld_result.get("density_per_1000", 30.0)
                        elif intent.get("sequence"):
                            _ld_result = _calc_ld([{"sequence": intent["sequence"], "copy_number": 1}])
                            _pk_liab_density = _ld_result.get("density_per_1000", 30.0)

                        # Check FcRn binding motif
                        _fcrn_intact = True
                        _seq_for_fcrn = intent.get("sequence", "")
                        if _seq_for_fcrn and len(_seq_for_fcrn) > 200:
                            _fcrn_check = check_fcrn_binding_motif(_seq_for_fcrn)
                            _fcrn_intact = _fcrn_check.get("intact", True)

                        _pk_result = predict_human_half_life(
                            global_pi=intent["pI"],
                            hydrophobicity=intent.get("hydrophobicity", 0.35),
                            liability_density=_pk_liab_density,
                            fcrn_binding_motif_intact=_fcrn_intact,
                            mw_kda=intent.get("mw", 150.0),
                            glycoform_profile=_glyco_profile,
                        )
                        _analysis_cache["pk_result"] = _pk_result
                        _analysis_cache["glycoform_profile"] = _glyco_profile

                        render_pk_panel(_pk_result, glycoform_impact=_glyco_impact)

                    except Exception as pk_err:
                        st.caption(f"(Preclinical PK prediction unavailable: {pk_err})")

                    # -- Step 1.10b: Auto-trigger Analytical QC Simulation ------
                    # Runs virtual SEC/CE-SDS/cIEF/glycan panel so Section 4 of the
                    # report is populated.  Marked as "simulated" — not experimental.
                    try:
                        from src.analytical_qc_twin import run_analytical_qc as _auto_qc
                        from src.molecule_classifier import MoleculeClass as _MC

                        _auto_seq = intent.get("sequence", "")
                        _auto_pI = intent.get("pI", 8.0)
                        _auto_mol_cls = intent.get("molecule_class", "unknown")
                        try:
                            _auto_is_mab = _MC(_auto_mol_cls).is_mab_like
                        except (ValueError, KeyError):
                            _auto_is_mab = False  # safe default: don't assume IgG

                        # Derive aggregation % from dev_result
                        _auto_agg_risk = 0.02
                        if 'dev_result' in dir() and dev_result and dev_result.get("status") == "success":
                            _auto_agg_risk = dev_result.get("data", {}).get("predictions", {}).get("agg_risk", 0.02)
                        _auto_agg_pct = _auto_agg_risk * _auto_agg_risk * 20.0

                        # Glycoform sialylation
                        _auto_sial_map = {
                            "standard_cho": 0.0, "high_mannose": 0.0,
                            "afucosylated": 0.0, "highly_sialylated": 0.60,
                            "none_aglycosylated": 0.0,
                        }
                        _auto_sial = _auto_sial_map.get(st.session_state.glycoform_profile, 0.0)

                        if _auto_seq and len(_auto_seq) > 30:
                            _auto_agg_pct_clamped = max(0.5, min(_auto_agg_pct, 10.0))  # aligned with bulk_runner
                            _auto_qc_result = _auto_qc(
                                sequence=_auto_seq,
                                pI=float(_auto_pI),
                                aggregation_pct=_auto_agg_pct_clamped,
                                is_mab=_auto_is_mab,
                                sialylation_fraction=_auto_sial,
                                molecule_class=_auto_mol_cls,
                            )
                            # Store as serializable dict in cache (mirrors Analytical tab structure)
                            _analysis_cache["analytical_qc"] = {
                                "cief": _auto_qc_result.cief,
                                "ce_sds": _auto_qc_result.ce_sds,
                                "sec": {"monomer_pct": round(100.0 - _auto_agg_pct, 2),
                                        "hmw_pct": round(_auto_agg_pct, 2)},
                                "glycan": _auto_qc_result.glycan,
                                "overall_qc_pass": _auto_qc_result.overall_qc_pass,
                                "source": "simulated",  # NOT experimental
                            }
                            log.info("Auto-QC: SEC monomer=%.1f%%, cIEF main=%.1f%%, CE-SDS intact=%.1f%%",
                                     100.0 - _auto_agg_pct,
                                     _auto_qc_result.cief.main_pct,
                                     _auto_qc_result.ce_sds.intact_pct)
                    except Exception as _auto_qc_err:
                        log.warning("Auto-QC simulation failed: %s", _auto_qc_err)

                    # -- Step 1.10c: Auto-trigger Upstream Simulation ----------
                    # Populates bioreactor fed-batch data so Section 5 of the
                    # report is available regardless of Process Dev page visit.
                    try:
                        from src.upstream_twin import (
                            run_upstream_simulation as _auto_upstream,
                            result_to_dict as _upstream_to_dict,
                        )

                        _up_dev_score = None
                        _up_agg_risk = None
                        if 'dev_result' in dir() and dev_result and dev_result.get("status") == "success":
                            _up_data = dev_result.get("data", {})
                            _up_preds = _up_data.get("predictions", {})
                            # Extract score: main path has data.score.score,
                            # fallback path has data.developability_score
                            _up_dev_score = (
                                _up_data.get("composite_score")                     # 5-dim (if set)
                                or _up_data.get("developability_score")             # fallback path
                                or (_up_data.get("score", {}) or {}).get("score")   # main path
                            )
                            _up_agg_risk = _up_preds.get("agg_risk") or _up_preds.get("aggregation_risk")

                        _up_result = _auto_upstream(
                            seed_density=0.5,
                            temp_shift_day=5.0,
                            dev_score=_up_dev_score,
                            agg_risk=_up_agg_risk,
                            culture_days=14.0,
                            hydrophobicity=intent.get("gravy"),
                            sequence=intent.get("sequence"),
                            molecule_class=intent.get("molecule_class"),
                        )
                        st.session_state["upstream_result"] = _up_result
                        _up_dict = _upstream_to_dict(_up_result)
                        st.session_state["upstream_result_dict"] = _up_dict
                        _analysis_cache["upstream_result"] = _up_dict
                        log.info("Auto-upstream: peak_vcd=%.1f, titer=%.1f mg/L",
                                 _up_result.peak_vcd, _up_result.final_titer)
                    except Exception as _auto_up_err:
                        log.warning("Auto-upstream simulation failed: %s", _auto_up_err)

                    # -- Step 1.10d: Auto-trigger DoE Purification Optimization -
                    # Populates DoE results so formulation_result is available
                    # in reports without visiting Process Dev page.
                    try:
                        from src.purification_optimizer import (
                            run_doe_optimization as _auto_doe,
                            doe_to_dict as _auto_doe_to_dict,
                        )

                        _doe_pI = float(intent.get("pI", 8.0))
                        _doe_mw = float(intent.get("mw", 150.0))
                        _doe_hydro = float(intent.get("gravy", 0.35) or 0.35)

                        _doe_result = _auto_doe(
                            pI=_doe_pI,
                            mw=_doe_mw,
                            hydrophobicity=_doe_hydro,
                        )
                        st.session_state["doe_result_ds"] = _doe_result
                        _doe_dict = _auto_doe_to_dict(_doe_result)
                        st.session_state["doe_result_dict"] = _doe_dict
                        _analysis_cache["formulation_result"] = _doe_dict
                        log.info("Auto-DoE: best_yield=%.1f%%, best_purity=%.1f%%",
                                 _doe_dict.get("best_yield", 0),
                                 _doe_dict.get("best_purity", 0))
                    except Exception as _auto_doe_err:
                        log.warning("Auto-DoE optimization failed: %s", _auto_doe_err)

                    # -- Step 1.10e: Auto-trigger Stability Projection ----------
                    # Runs ICH shelf-life simulation so stability data is available
                    # in report and 5-dim composite without visiting Preclinical page.
                    # Mirrors bulk_runner.py twin #6 parameters for alignment.
                    try:
                        from src.stability_twin import run_stability_study as _auto_stab

                        # Derive inputs from auto-QC results or defaults
                        _stab_hmw = 1.0  # default starting HMW%
                        _stab_acidic = 15.0  # default starting acidic%
                        if "_auto_qc_result" in dir() and _auto_qc_result:
                            _cief_obj = getattr(_auto_qc_result, "cief", None)
                            if _cief_obj:
                                _stab_acidic = getattr(_cief_obj, "acidic_pct", 15.0)
                            _cesds_obj = getattr(_auto_qc_result, "ce_sds", None)
                            if _cesds_obj:
                                _stab_hmw = getattr(_cesds_obj, "hmw_pct", 1.0)

                        _stab_deam = intent.get("deam_sites", 5)
                        _stab_pI = float(intent.get("pI", 8.0))
                        # stability_twin expects normalized hydrophobicity (0-1), not raw GRAVY
                        _stab_hydro = float(intent.get("hydrophobicity", 0.35))

                        _stab_result_auto = _auto_stab(
                            starting_hmw_pct=max(0.1, min(_stab_hmw, 10.0)),
                            starting_acidic_pct=max(5.0, min(_stab_acidic, 40.0)),
                            formulation_ph=6.0,
                            pI=_stab_pI,
                            deamidation_sites=_stab_deam,
                            dp_clip_sites=1,
                            hydrophobicity=_stab_hydro,
                        )
                        if _stab_result_auto:
                            _analysis_cache["stability_result"] = {
                                "shelf_life_months": getattr(
                                    _stab_result_auto, "predicted_shelf_life_months", None),
                                "stability_grade": getattr(
                                    _stab_result_auto, "overall_stability_grade", None),
                            }
                            log.info("Auto-stability: shelf_life=%.0f mo, grade=%s",
                                     _stab_result_auto.predicted_shelf_life_months,
                                     _stab_result_auto.overall_stability_grade)
                    except Exception as _auto_stab_err:
                        log.warning("Auto-stability projection failed: %s", _auto_stab_err)

                    # -- Step 1.10f: Auto-trigger Immunogenicity Assessment -----
                    # Runs ADA risk / MHC-II hotspot analysis so the 5-dim
                    # composite score includes immunogenicity evidence, matching
                    # the bulk_runner path (twin #4).
                    try:
                        from src.immunogenicity_twin import run_immunogenicity_assessment as _auto_imm

                        _imm_seq = intent.get("sequence", "")
                        _imm_agg_risk = 0.02
                        _imm_dev_score = None
                        if 'dev_result' in dir() and dev_result and dev_result.get("status") == "success":
                            _imm_data = dev_result.get("data", {})
                            _imm_preds = _imm_data.get("predictions", {})
                            _imm_agg_risk = _imm_preds.get("agg_risk", 0.02)
                            _imm_dev_score = (
                                _imm_data.get("composite_score")
                                or _imm_data.get("developability_score")
                                or (_imm_data.get("score", {}) or {}).get("score")
                            )

                        if _imm_seq and len(_imm_seq) > 30:
                            _imm_result = _auto_imm(
                                sequence=_imm_seq,
                                agg_risk=_imm_agg_risk,
                                dev_score=_imm_dev_score,
                                molecule_name=intent.get("name", "mAb"),
                                molecule_class=intent.get("molecule_class"),
                            )
                            if _imm_result:
                                # Store in session_state so Developability Dashboard picks it up
                                st.session_state["ada_result"] = _imm_result
                                # Also store in analysis_cache for report_assembler
                                _ada_level = getattr(_imm_result, "ada_risk_level", None)
                                _ada_score = getattr(_imm_result, "ada_risk_score", None)
                                _analysis_cache["ada_result"] = {
                                    "ada_risk_level": _ada_level,
                                    "ada_risk_score": _ada_score,
                                    "n_high_risk": getattr(_imm_result, "n_high_risk", 0),
                                    "n_medium_risk": getattr(_imm_result, "n_medium_risk", 0),
                                }
                                log.info("Auto-immunogenicity: ADA risk=%s, score=%.3f",
                                         _ada_level or "?", _ada_score or 0)
                    except Exception as _auto_imm_err:
                        log.warning("Auto-immunogenicity assessment failed: %s", _auto_imm_err)

                    # -- Step 1.10h: Auto-trigger COGS Calculation ---------------
                    # Calculates commercial manufacturing cost using titer from
                    # upstream auto-trigger and yield from DoE auto-trigger.
                    # Mirrors bulk_runner alignment — both paths produce COGS data.
                    try:
                        from src.cogs_twin import COGSInputs as _AutoCOGSInputs
                        from src.cogs_twin import calculate_cogs as _auto_cogs
                        from src.cogs_twin import cogs_to_dict as _auto_cogs_to_dict

                        _cogs_auto_titer = 3.50  # default fallback (typical mAb ~3-4 g/L)
                        _cogs_auto_yield = 0.70  # default
                        _up_d = st.session_state.get("upstream_result_dict") or {}
                        _doe_d = st.session_state.get("doe_result_dict") or {}
                        if _up_d.get("final_titer"):
                            _cogs_auto_titer = float(_up_d["final_titer"])
                        if _doe_d.get("optimal_yield"):
                            _cogs_auto_yield = float(_doe_d["optimal_yield"])
                        # Round to match manual tab widget precision so auto vs manual COGS are identical
                        _cogs_auto_titer = round(_cogs_auto_titer, 2)
                        _cogs_auto_yield = round(_cogs_auto_yield * 100, 1) / 100.0

                        _cogs_auto_inputs = _AutoCOGSInputs(
                            titer_g_per_L=_cogs_auto_titer,
                            downstream_yield=_cogs_auto_yield,
                        )
                        _cogs_auto_mol_cls = intent.get("molecule_class", "canonical_mab")
                        _cogs_auto_result = _auto_cogs(_cogs_auto_inputs, molecule_class=_cogs_auto_mol_cls)
                        st.session_state["cogs_result"] = _cogs_auto_result
                        _cogs_auto_dict = _auto_cogs_to_dict(_cogs_auto_result)
                        st.session_state["cogs_result_dict"] = _cogs_auto_dict
                        _analysis_cache["cogs_result"] = _cogs_auto_dict
                        log.info("Auto-COGS: $%.2f/g (%s), batch=%.0f g",
                                 _cogs_auto_result.cogs_per_gram,
                                 _cogs_auto_result.cost_rating,
                                 _cogs_auto_result.batch_output_g)
                    except Exception as _auto_cogs_err:
                        log.warning("Auto-COGS calculation failed: %s", _auto_cogs_err)

                    st.caption(
                        "Note: Digital twins use rule-based heuristics calibrated to public data. "
                        "They are educational tools for exploring developability dimensions, "
                        "not substitutes for experimental characterization."
                    )

                    # -- Step 1.11: Generative Optimization Button (M14) --------
                    try:
                        render_optimize_button(
                            intent=intent,
                            dev_result=dev_result if 'dev_result' in dir() else None,
                            pk_result=_pk_result if '_pk_result' in dir() else None,
                            bispec_result=None,
                            cache=_analysis_cache,
                        )
                    except Exception as opt_err:
                        st.caption(f"(Auto-optimize unavailable: {opt_err})")

                    # CADET chromatography simulation deferred to "Downstream Purification" tab
                    # (Separation of Concerns: Characterization = MW/pI/GRAVY only)
                    st.caption(
                        "SMA chromatography parameters computed. "
                        "View IEX simulation and chromatograms in the **Downstream Purification** tab."
                    )
                    char_msg = (
                        f"Characterization complete for **{intent.get('name', 'mAb')}** "
                        f"(pI={intent['pI']}, MW={intent.get('mw', 150)} kDa)."
                    )
                    st.session_state.messages.append({"role": "assistant", "content": char_msg})
                    st.session_state.run_count += 1

                    # M11+: Save analysis cache to workspace (always after step1 succeeds)
                    ws_store.update_active_field("analysis_cache", _analysis_cache)

                else:
                    # Step 1 failed entirely
                    error_msg = f"Parameter prediction failed: {step1.get('message', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # ===================================================================
    #  M12: Multi-Chain Stoichiometric Assembly Processing
    # ===================================================================
    pending_assembly = st.session_state.pending_assembly
    if pending_assembly is not None:
        st.session_state.pending_assembly = None  # consume

        # Parse each chain's FASTA header / raw sequence
        def _parse_chain_seq(raw_text: str) -> Tuple[str, str]:
            raw_text = raw_text.strip()
            if raw_text.startswith(">"):
                lines = raw_text.split("\n")
                name = lines[0][1:].strip().split()[0] if lines[0][1:].strip() else "Chain"
                seq = re.sub(r'[^A-Za-z]', '', "".join(lines[1:])).upper()
            else:
                name = "Chain"
                seq = re.sub(r'[^A-Za-z]', '', raw_text).upper()
            return name, seq

        assembly_chains = []
        for ch_input in pending_assembly:
            ch_name_parsed, ch_seq = _parse_chain_seq(ch_input.get("sequence", ""))
            if len(ch_seq) < 10:
                continue
            ch_name = ch_input.get("name", ch_name_parsed)
            ch_copies = max(1, int(ch_input.get("copy_number", 1)))
            # Detect chain type from name
            nm_lower = ch_name.lower()
            if any(kw in nm_lower for kw in ("hc", "heavy", "vh")):
                ch_type = "HC"
            elif any(kw in nm_lower for kw in ("lc", "light", "vl", "kappa", "lambda")):
                ch_type = "LC"
            else:
                ch_type = "HC" if len(ch_seq) > 300 else ("LC" if len(ch_seq) > 100 else "unknown")
            assembly_chains.append({
                "sequence": ch_seq, "copy_number": ch_copies,
                "name": ch_name, "chain_type": ch_type,
            })

        if not assembly_chains:
            st.error("No valid chain sequences provided (minimum 10 aa each).")
        else:
            # Build super-sequence for global properties
            from src.agents import assemble_super_sequence
            asm = assemble_super_sequence(assembly_chains)

            if "error" in asm:
                st.error(f"Assembly error: {asm['error']}")
            else:
                # Create workspace
                ws = ws_store.get_active()
                stoich_label = asm.get("stoichiometry", "Assembly")
                if ws is None:
                    ws = ws_store.create_new(display_name=f"Assembly: {stoich_label}")
                    ws_store.save_to_session_state(st.session_state)

                asm_msg = (
                    f"Multi-chain assembly: {stoich_label} | "
                    f"{asm['total_length']} aa | "
                    f"pI={asm['pI']} | MW={asm['mw_kda']} kDa"
                )
                st.session_state.messages.append({"role": "user", "content": asm_msg})
                ws_store.add_message_to_active("user", asm_msg)

                with st.chat_message("user"):
                    st.markdown(asm_msg)

                # Build intent from super-sequence
                super_seq = asm["super_sequence"]
                # Run liability analysis per unique chain
                chain_analyses = []
                for ch in assembly_chains:
                    liab = scan_sequence_liabilities(ch["sequence"])
                    cdrs = identify_cdrs_heuristic(ch["sequence"], ch["chain_type"])
                    chain_analyses.append({
                        "name": ch["name"], "chain_type": ch["chain_type"],
                        "sequence": ch["sequence"], "length": len(ch["sequence"]),
                        "copy_number": ch["copy_number"],
                        "liabilities": liab, "cdrs": cdrs,
                    })

                # ── Molecule Classification (mirrors FASTA pathway) ──
                try:
                    from src.molecule_classifier import classify_molecule as _asm_classify
                    _asm_classification = _asm_classify(
                        sequence=super_seq,
                        chains=[{"name": ch["name"], "sequence": ch["sequence"],
                                 "chain_type": ch["chain_type"]}
                                for ch in assembly_chains],
                        assembly_chains=assembly_chains,
                        name=stoich_label,
                    )
                    _asm_mol_class = _asm_classification.effective_class.value
                    _asm_mol_class_info = _asm_classification.to_dict()
                    log.info("Assembly classifier: %s → %s", stoich_label, _asm_mol_class)
                except Exception as _asm_cls_err:
                    log.warning("Assembly classification failed: %s", _asm_cls_err)
                    _asm_mol_class = "unknown"
                    _asm_mol_class_info = {"molecule_class": "unknown", "confidence": "Low",
                                           "evidence": [f"Classification error: {_asm_cls_err}"]}

                # ── Feature Registry (mirrors FASTA pathway) ──
                try:
                    from src.feature_registry import compute_features as _asm_compute_feats
                    _asm_feature_set = _asm_compute_feats(
                        sequence=super_seq,
                        molecule_class=_asm_mol_class,
                        chains=assembly_chains,
                    )
                    _asm_feature_dict = _asm_feature_set.to_dict()
                    _asm_liability_summary = _asm_feature_set.liability_summary()
                    log.info("Assembly FeatureRegistry: %d features", len(_asm_feature_set.features))
                except Exception as _asm_feat_err:
                    log.warning("Assembly FeatureRegistry failed: %s", _asm_feat_err)
                    _asm_feature_set = None
                    _asm_feature_dict = {}
                    _asm_liability_summary = {}

                # ── Compute real deam/ox/cys from chain analyses ──
                _asm_deam = sum(
                    ca.get("liabilities", {}).get("deamidation_sites", 0)
                    for ca in chain_analyses
                )
                _asm_ox = sum(
                    ca.get("liabilities", {}).get("oxidation_sites", 0)
                    for ca in chain_analyses
                )
                _asm_cys = sum(
                    ca.get("liabilities", {}).get("cysteine_count", 0)
                    for ca in chain_analyses
                )
                # Fallback: count from sequence directly if liabilities don't have cysteine
                if _asm_cys == 0:
                    _asm_cys = super_seq.upper().count("C")

                # Use feature_registry MW (includes glycan mass) when available;
                # fallback to agents.py MW (protein-only mass) otherwise.
                _asm_mw_final = asm["mw_kda"]
                if _asm_feature_set and hasattr(_asm_feature_set, "features"):
                    _asm_mw_feat = _asm_feature_set.features.get("mw_kda")
                    if _asm_mw_feat and getattr(_asm_mw_feat, "value", None):
                        _asm_mw_final = _asm_mw_feat.value

                # Use stoichiometric seq_length from feature_registry when available
                _asm_seq_len = len(super_seq)
                if _asm_feature_set and hasattr(_asm_feature_set, "features"):
                    _asm_sl_feat = _asm_feature_set.features.get("seq_length")
                    if _asm_sl_feat and getattr(_asm_sl_feat, "value", None):
                        _asm_seq_len = _asm_sl_feat.value

                intent = {
                    "name": stoich_label,
                    "pI": asm["pI"],
                    "mw": _asm_mw_final,
                    "hydrophobicity": asm["hydrophobicity"],
                    "pH_working": 7.0,
                    "deam_sites": _asm_deam if _asm_deam > 0 else 1,
                    "ox_sites": _asm_ox if _asm_ox > 0 else 1,
                    "cysteine_count": _asm_cys,
                    "gradient_slope": 15.0,
                    "source": "fasta",
                    "sequence": super_seq,
                    "seq_length": _asm_seq_len,
                    "gravy": asm["gravy"],
                    "chains": [
                        {"name": ch["name"], "sequence": ch["sequence"],
                         "chain_type": ch["chain_type"],
                         "copy_number": ch.get("copy_number", 1)}
                        for ch in assembly_chains
                    ],
                    "chain_analyses": chain_analyses,
                    "assembly_chains": assembly_chains,
                    "stoichiometry": stoich_label,
                    # Phase 1 fields — molecule routing & unified features
                    "molecule_class": _asm_mol_class,
                    "molecule_class_info": _asm_mol_class_info,
                    "feature_set": _asm_feature_dict,
                    "feature_set_obj": _asm_feature_set,
                    "liability_summary": _asm_liability_summary,
                }
                ws_store.update_active_field("intent", intent)

                # Persist intent globally for cross-page context (Fed-Batch, CMC Board, etc.)
                st.session_state.last_intent = intent

                # Initialize analysis cache
                _asm_cache = {
                    "mode": "standard",
                    "intent": intent,
                    "source": "fasta",
                    "ml_override": None,
                    "source_label": "Unknown",
                    "predictor_source": "rule_based",   # v32
                    "predictor_detail": "",              # v32
                    "ood_bypass_reason": None,           # v32
                    "variants": None,
                    "dev_result": None,
                    "cqa": None,
                    "sim_summary": None,
                    "sim_elapsed": None,
                    "bispecific_result": None,
                    "ms_characterization": None,
                    "pk_result": None,              # M13
                    "glycoform_profile": st.session_state.glycoform_profile,  # M13
                    "glycoform_impact": None,       # M13
                    "optimization_result": None,    # M14
                }

                # CRITICAL: Invalidate ALL molecule-bound state when new assembly starts.
                _invalidate_molecule_bound_state(ws_store=ws_store, new_intent=intent)
                ws_store.update_active_field("analysis_cache", _asm_cache)

                with st.chat_message("assistant"):
                    # Assembly info card
                    st.markdown(f"""
                    <div class="fasta-card" style="border-left: 4px solid #64748B;">
                        <div class="fasta-title">
                            Molecular Assembly — True Stoichiometry
                        </div>
                        <b>{stoich_label}</b><br>
                        pI = {asm['pI']} | MW = {asm['mw_kda']} kDa |
                        GRAVY = {asm['gravy']} |
                        Total residues: {asm['total_length']} aa
                    </div>
                    """, unsafe_allow_html=True)

                    # Per-chain table
                    st.markdown("#### Per-Chain Assembly")
                    for ch_info in asm.get("chains_summary", []):
                        st.markdown(
                            f"- **{ch_info['name']}** ({ch_info['chain_type']}): "
                            f"{ch_info['length']} aa x {ch_info['copy_number']} copies"
                        )

                    # Liability density (M12)
                    from src.analytical_twin import calculate_liability_density
                    liab_density = calculate_liability_density(assembly_chains)
                    ld_level = liab_density["risk_level"]
                    ld_color = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}.get(ld_level, "#64748B")
                    st.markdown(f"""
                    <div class="cqa-card" style="border-left: 4px solid {ld_color};">
                        <div class="cqa-title">Liability Density</div>
                        <div class="cqa-value">{liab_density['density_per_1000']:.1f}
                            <span class="cqa-unit">motifs / 1000 residues</span>
                        </div>
                        <span class="status-pill status-{'ok' if ld_level == 'Low' else ('warn' if ld_level == 'Medium' else 'err')}">{ld_level} Risk</span>
                        <div style="font-size:0.8rem; margin-top:4px; color:#64748B;">
                            Total motifs: {liab_density['total_motifs']} across {liab_density['total_residues']} assembled residues
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Characterization
                    render_characterization_panel(intent)

                    # -- Developability Assessment for Assembly --
                    _asm_dev_result = None
                    try:
                        from src.agents import predict_developability_risk
                        _vh_seq_asm, _vl_seq_asm = "", ""
                        for _ch in assembly_chains:
                            _ct = _ch.get("chain_type", "").upper()
                            if _ct in ("HC", "HEAVY") and _ch.get("sequence"):
                                _vh_seq_asm = _ch["sequence"]
                            elif _ct in ("LC", "LIGHT") and _ch.get("sequence"):
                                _vl_seq_asm = _ch["sequence"]

                        st.markdown("**Developability Assessment:** Running pLM + XGBoost risk analysis...")
                        _asm_dev_result = predict_developability_risk(
                            pI=intent["pI"],
                            mw=intent.get("mw", 150.0),
                            hydrophobicity=intent.get("hydrophobicity", 0.35),
                            deam_sites=intent.get("deam_sites", 1),
                            ox_sites=intent.get("ox_sites", 1),
                            sequence=super_seq,
                            vh_sequence=_vh_seq_asm,
                            vl_sequence=_vl_seq_asm,
                            feature_set=intent.get("feature_set_obj"),
                            molecule_class=intent.get("molecule_class"),
                        )

                        if _asm_dev_result and _asm_dev_result.get("status") == "success":
                            # OOD Warning
                            _ood_asm = _asm_dev_result.get("data", {}).get("ood_info", {})
                            if _ood_asm.get("is_ood"):
                                _ood_msg_asm = _ood_asm.get("warning_message", "OOD detected.")
                                _ood_conf_asm = _ood_asm.get("confidence", "Low")
                                _ood_c_asm = "#F59E0B" if _ood_conf_asm == "Medium" else "#EF4444"
                                st.markdown(
                                    f'<div style="background:{_ood_c_asm}12;border-left:4px solid {_ood_c_asm};'
                                    f'padding:12px;border-radius:8px;margin:8px 0;">'
                                    f'<span style="font-size:1.05rem;font-weight:700;color:{_ood_c_asm};">'
                                    f'OOD Warning (Confidence: {_ood_conf_asm})</span><br/>'
                                    f'<span style="color:#334155;font-size:0.92rem;">{_ood_msg_asm}</span></div>',
                                    unsafe_allow_html=True,
                                )

                            # v32.1: Persist pipeline OOD into cache (assembly path)
                            if _ood_asm.get("is_ood") and not _asm_cache.get("ood_bypass_reason"):
                                _asm_max_z = _ood_asm.get("max_z_score", 0)
                                _asm_flag_details = "; ".join(
                                    f"{f['metric']}={f['value']:.3f} (z={f['z_score']:.1f})"
                                    for f in _ood_asm.get("flags", [])
                                    if f.get("z_score", 0) > 2.0
                                )
                                _asm_cache["ood_bypass_reason"] = (
                                    f"OOD detected (max z={_asm_max_z:.1f}): {_asm_flag_details}"
                                )

                            render_developability_panel(intent, _asm_dev_result)
                            render_actionable_insights_panel(_asm_dev_result)

                            # Store dev_result in analysis cache AND workspace
                            _asm_cache["dev_result"] = _asm_dev_result
                            ws_store.update_active_field("dev_result", _asm_dev_result)
                    except Exception as asm_dev_err:
                        st.caption(f"(Developability assessment unavailable: {asm_dev_err})")

                    # Now run the standard pipeline with corrected global pI/MW
                    ml_override = None
                    _torch_available = False
                    try:
                        import torch as _t
                        _torch_available = True
                    except ImportError:
                        pass

                    if _torch_available:
                        st.markdown("**Step 0:** ML Prediction (PyTorch MLP)...")
                        try:
                            from src.agents import predict_ml_with_shap
                            ml_result = predict_ml_with_shap(
                                pI=intent["pI"], mw=intent["mw"],
                                hydrophobicity=intent["hydrophobicity"],
                                sequence=super_seq,
                            )
                            if ml_result["status"] == "success":
                                ml_override = ml_result["data"]["ml_override"]
                                st.success(
                                    f"ML Override: ka={ml_override['ka']:.4f}, "
                                    f"nu={ml_override['nu']:.3f}"
                                )
                        except Exception as ml_e:
                            st.info(f"ML prediction skipped: {ml_e}")

                    _asm_cache["ml_override"] = ml_override
                    source_label = "ML Override" if ml_override else "Static Fallback"
                    _asm_cache["source_label"] = "ML OVERRIDE" if ml_override else "STATIC FALLBACK"

                    # M13: Glycoform pI shift for assembly
                    _asm_glyco_shift = 0.0
                    try:
                        from src.preclinical_twin import get_glycoform_pi_shift as _asm_gpis
                        _asm_glyco_shift = _asm_gpis(st.session_state.glycoform_profile)
                    except Exception:
                        pass
                    _asm_effective_pI = intent["pI"] + _asm_glyco_shift
                    if abs(_asm_glyco_shift) > 0.01:
                        st.info(f"Glycoform pI shift: {_asm_glyco_shift:+.1f} (effective pI: {_asm_effective_pI:.2f})")

                    st.markdown(f"**Step 1/2:** SMA parameters ({source_label})...")

                    try:
                        step1 = predict_physical_params(
                            name=intent["name"], pI=_asm_effective_pI,
                            mw=intent["mw"], hydrophobicity=intent["hydrophobicity"],
                            sequence=super_seq, ml_override=ml_override,
                        )
                    except Exception as e:
                        step1 = {"status": "error", "message": str(e), "data": {}}

                    if step1["status"] == "success":
                        v = step1["data"]["variants"]
                        _asm_cache["variants"] = v
                        source_badge = step1["data"].get("source", "unknown").upper()
                        _asm_cache["source_label"] = source_badge

                        # SMA parameters stored in cache — displayed in Downstream Purification tab
                        # (ka, nu NOT shown here per Separation of Concerns)

                        # MS Characterization + Biophysical summary
                        try:
                            _asm_ms_result = render_ms_characterization_panel(intent)
                            if _asm_ms_result:
                                _asm_cache["ms_characterization"] = _asm_ms_result
                            render_ms_summary_metrics(intent)
                        except Exception:
                            pass

                        # M13: PK prediction for assembly (with assembly completeness penalty)
                        try:
                            from src.preclinical_twin import predict_human_half_life as _pk_predict_asm
                            from src.preclinical_twin import check_fcrn_binding_motif as _pk_fcrn_asm
                            from src.preclinical_twin import assess_assembly_completeness as _pk_asm_check
                            _fcrn_asm = True
                            if super_seq and len(super_seq) > 200:
                                _fcrn_asm = _pk_fcrn_asm(super_seq).get("intact", True)

                            # Assembly completeness check — penalize incomplete stoichiometry
                            _asm_completeness = _pk_asm_check(assembly_chains)
                            _asm_liab = liab_density.get("density_per_1000", 30.0)
                            _asm_hydro = asm["hydrophobicity"]

                            if not _asm_completeness["is_complete_igg"]:
                                # Feature-driven: adjust hydrophobicity and liability
                                # based on exposed interface area (not hardcoded penalties)
                                _hydro_delta = _asm_completeness.get("hydrophobicity_delta", 0.0)
                                _liab_delta = _asm_completeness.get("liability_density_delta", 0.0)
                                _asm_hydro = min(1.0, _asm_hydro + _hydro_delta)
                                _asm_liab = min(200.0, _asm_liab + _liab_delta)
                                # If Fc is missing entirely, flag FcRn as disrupted
                                if _asm_completeness.get("missing_fc", False):
                                    _fcrn_asm = False

                            _pk_asm = _pk_predict_asm(
                                global_pi=asm["pI"],
                                hydrophobicity=_asm_hydro,
                                liability_density=_asm_liab,
                                fcrn_binding_motif_intact=_fcrn_asm,
                                mw_kda=asm["mw_kda"],
                                glycoform_profile=st.session_state.glycoform_profile,
                                assembly_half_life_multiplier=_asm_completeness.get(
                                    "assembly_half_life_multiplier", 1.0),
                            )

                            # Display assembly risk info (scoring engine handles PK impact
                            # through the adjusted biophysical features above)
                            if not _asm_completeness["is_complete_igg"]:
                                if _asm_completeness["risk_level"] in ("Very High", "High"):
                                    _pk_asm["risk_assessment"] = _asm_completeness["risk_level"]
                                    _pk_asm["risk_color"] = "#EF4444"

                                # Show assembly warning card
                                st.markdown(
                                    f'<div style="background:#FEF2F215;border-left:5px solid #EF4444;'
                                    f'padding:12px;border-radius:8px;margin:8px 0;">'
                                    f'<span style="font-size:1.1rem;font-weight:700;color:#EF4444;">'
                                    f'Assembly Completeness: {_asm_completeness["risk_level"]} Risk</span>'
                                    f'<div style="font-size:0.9rem;color:#7F1D1D;margin-top:4px;">'
                                    f'{_asm_completeness["explanation"]}</div></div>',
                                    unsafe_allow_html=True,
                                )

                            _asm_cache["pk_result"] = _pk_asm
                            _asm_cache["glycoform_profile"] = st.session_state.glycoform_profile
                            render_pk_panel(_pk_asm)
                        except Exception:
                            pass

                        # -- Auto-trigger Analytical QC for Assembly pathway --
                        try:
                            from src.analytical_qc_twin import run_analytical_qc as _asm_auto_qc
                            from src.molecule_classifier import MoleculeClass as _AsmMC

                            _asm_auto_seq = intent.get("sequence", "")
                            _asm_auto_pI = intent.get("pI", 8.0)
                            _asm_auto_mol_cls = intent.get("molecule_class", "unknown")
                            try:
                                _asm_auto_is_mab = _AsmMC(_asm_auto_mol_cls).is_mab_like
                            except (ValueError, KeyError):
                                _asm_auto_is_mab = False  # safe default

                            _asm_auto_agg_risk = 0.02
                            if '_asm_dev_result' in dir() and _asm_dev_result and _asm_dev_result.get("status") == "success":
                                _asm_auto_agg_risk = _asm_dev_result.get("data", {}).get("predictions", {}).get("agg_risk", 0.02)
                            _asm_auto_agg_pct = _asm_auto_agg_risk * _asm_auto_agg_risk * 20.0

                            _asm_sial_map = {
                                "standard_cho": 0.0, "high_mannose": 0.0,
                                "afucosylated": 0.0, "highly_sialylated": 0.60,
                                "none_aglycosylated": 0.0,
                            }
                            _asm_sial = _asm_sial_map.get(st.session_state.glycoform_profile, 0.0)

                            if _asm_auto_seq and len(_asm_auto_seq) > 30:
                                _asm_auto_agg_pct_clamped = max(0.5, min(_asm_auto_agg_pct, 10.0))  # aligned with bulk_runner
                                _asm_qc_result = _asm_auto_qc(
                                    sequence=_asm_auto_seq,
                                    pI=float(_asm_auto_pI),
                                    aggregation_pct=_asm_auto_agg_pct_clamped,
                                    is_mab=_asm_auto_is_mab,
                                    sialylation_fraction=_asm_sial,
                                    molecule_class=_asm_auto_mol_cls,
                                )
                                _asm_cache["analytical_qc"] = {
                                    "cief": _asm_qc_result.cief,
                                    "ce_sds": _asm_qc_result.ce_sds,
                                    "sec": {"monomer_pct": round(100.0 - _asm_auto_agg_pct, 2),
                                            "hmw_pct": round(_asm_auto_agg_pct, 2)},
                                    "glycan": _asm_qc_result.glycan,
                                    "overall_qc_pass": _asm_qc_result.overall_qc_pass,
                                    "source": "simulated",
                                }
                                log.info("Assembly Auto-QC: SEC monomer=%.1f%%, cIEF main=%.1f%%",
                                         100.0 - _asm_auto_agg_pct,
                                         _asm_qc_result.cief.main_pct)
                        except Exception as _asm_qc_err:
                            log.warning("Assembly Auto-QC failed: %s", _asm_qc_err)

                        # -- Auto-trigger Upstream Simulation for Assembly pathway --
                        # Mirrors FASTA path Step 1.10c so fed-batch data is always
                        # available regardless of molecule type or input method.
                        try:
                            from src.upstream_twin import (
                                run_upstream_simulation as _asm_auto_upstream,
                                result_to_dict as _asm_upstream_to_dict,
                            )

                            _asm_up_dev_score = None
                            _asm_up_agg_risk = None
                            if '_asm_dev_result' in dir() and _asm_dev_result and _asm_dev_result.get("status") == "success":
                                _asm_up_data = _asm_dev_result.get("data", {})
                                _asm_up_preds = _asm_up_data.get("predictions", {})
                                # Extract score: main path has data.score.score,
                                # fallback path has data.developability_score
                                _asm_up_dev_score = (
                                    _asm_up_data.get("composite_score")
                                    or _asm_up_data.get("developability_score")
                                    or (_asm_up_data.get("score", {}) or {}).get("score")
                                )
                                _asm_up_agg_risk = _asm_up_preds.get("agg_risk") or _asm_up_preds.get("aggregation_risk")

                            _asm_up_result = _asm_auto_upstream(
                                seed_density=0.5,
                                temp_shift_day=5.0,
                                dev_score=_asm_up_dev_score,
                                agg_risk=_asm_up_agg_risk,
                                culture_days=14.0,
                                hydrophobicity=intent.get("gravy"),
                                sequence=intent.get("sequence"),
                                molecule_class=intent.get("molecule_class"),
                            )
                            st.session_state["upstream_result"] = _asm_up_result
                            _asm_up_dict = _asm_upstream_to_dict(_asm_up_result)
                            st.session_state["upstream_result_dict"] = _asm_up_dict
                            _asm_cache["upstream_result"] = _asm_up_dict
                            log.info("Assembly Auto-upstream: peak_vcd=%.1f, titer=%.1f mg/L",
                                     _asm_up_result.peak_vcd, _asm_up_result.final_titer)
                        except Exception as _asm_up_err:
                            log.warning("Assembly Auto-upstream simulation failed: %s", _asm_up_err)

                        # Auto-trigger Stability Projection (assembly path)
                        # Mirrors single-path Step 1.10e for bulk/single alignment.
                        try:
                            from src.stability_twin import run_stability_study as _asm_auto_stab

                            _asm_stab_hmw = 1.0
                            _asm_stab_acidic = 15.0
                            if "_asm_qc_result" in dir() and _asm_qc_result:
                                _c = getattr(_asm_qc_result, "cief", None)
                                if _c:
                                    _asm_stab_acidic = getattr(_c, "acidic_pct", 15.0)
                                _s = getattr(_asm_qc_result, "ce_sds", None)
                                if _s:
                                    _asm_stab_hmw = getattr(_s, "hmw_pct", 1.0)

                            _asm_stab_r = _asm_auto_stab(
                                starting_hmw_pct=max(0.1, min(_asm_stab_hmw, 10.0)),
                                starting_acidic_pct=max(5.0, min(_asm_stab_acidic, 40.0)),
                                formulation_ph=6.0,
                                pI=float(intent.get("pI", 8.0)),
                                deamidation_sites=intent.get("deam_sites", 5),
                                dp_clip_sites=1,
                                hydrophobicity=float(intent.get("hydrophobicity", 0.35)),
                            )
                            if _asm_stab_r:
                                _asm_cache["stability_result"] = {
                                    "shelf_life_months": getattr(
                                        _asm_stab_r, "predicted_shelf_life_months", None),
                                    "stability_grade": getattr(
                                        _asm_stab_r, "overall_stability_grade", None),
                                }
                                log.info("Assembly Auto-stability: shelf_life=%.0f mo, grade=%s",
                                         _asm_stab_r.predicted_shelf_life_months,
                                         _asm_stab_r.overall_stability_grade)
                        except Exception as _asm_stab_err:
                            log.warning("Assembly Auto-stability failed: %s", _asm_stab_err)

                        # Auto-trigger Immunogenicity Assessment (assembly path)
                        # Mirrors single-path Step 1.10f for bulk/single alignment.
                        try:
                            from src.immunogenicity_twin import run_immunogenicity_assessment as _asm_auto_imm

                            _asm_imm_seq = intent.get("sequence", "")
                            _asm_imm_agg_risk = 0.02
                            _asm_imm_dev_score = None
                            if '_asm_dev_result' in dir() and _asm_dev_result and _asm_dev_result.get("status") == "success":
                                _asm_imm_data = _asm_dev_result.get("data", {})
                                _asm_imm_preds = _asm_imm_data.get("predictions", {})
                                _asm_imm_agg_risk = _asm_imm_preds.get("agg_risk", 0.02)
                                _asm_imm_dev_score = (
                                    _asm_imm_data.get("composite_score")
                                    or _asm_imm_data.get("developability_score")
                                    or (_asm_imm_data.get("score", {}) or {}).get("score")
                                )

                            if _asm_imm_seq and len(_asm_imm_seq) > 30:
                                _asm_imm_result = _asm_auto_imm(
                                    sequence=_asm_imm_seq,
                                    agg_risk=_asm_imm_agg_risk,
                                    dev_score=_asm_imm_dev_score,
                                    molecule_name=intent.get("name", "mAb"),
                                    molecule_class=intent.get("molecule_class"),
                                )
                                if _asm_imm_result:
                                    st.session_state["ada_result"] = _asm_imm_result
                                    _asm_ada_level = getattr(_asm_imm_result, "ada_risk_level", None)
                                    _asm_ada_score = getattr(_asm_imm_result, "ada_risk_score", None)
                                    _asm_cache["ada_result"] = {
                                        "ada_risk_level": _asm_ada_level,
                                        "ada_risk_score": _asm_ada_score,
                                        "n_high_risk": getattr(_asm_imm_result, "n_high_risk", 0),
                                        "n_medium_risk": getattr(_asm_imm_result, "n_medium_risk", 0),
                                    }
                                    log.info("Assembly Auto-immunogenicity: ADA risk=%s, score=%.3f",
                                             _asm_ada_level or "?", _asm_ada_score or 0)
                        except Exception as _asm_imm_err:
                            log.warning("Assembly Auto-immunogenicity failed: %s", _asm_imm_err)

                        # Auto-trigger DoE Purification (assembly path)
                        # Mirrors single-path Step 1.10d — runs pI-adaptive grid search.
                        try:
                            from src.purification_optimizer import run_doe_optimization as _asm_auto_doe
                            from src.purification_optimizer import doe_to_dict as _asm_doe_to_dict
                            _asm_doe_pI = float(intent.get("pI", 8.0))
                            _asm_doe_mw = float(intent.get("mw", 150.0))
                            _asm_doe_hydro = float(intent.get("gravy", 0.35) or 0.35)
                            _asm_doe_result = _asm_auto_doe(
                                pI=_asm_doe_pI, mw=_asm_doe_mw,
                                hydrophobicity=_asm_doe_hydro,
                            )
                            if _asm_doe_result:
                                _asm_doe_dict = _asm_doe_to_dict(_asm_doe_result)
                                st.session_state["doe_result_ds"] = _asm_doe_result
                                st.session_state["doe_result_dict"] = _asm_doe_dict
                                _asm_cache["doe_result"] = _asm_doe_dict
                                _asm_cache["formulation_result"] = _asm_doe_dict
                                log.info("Assembly Auto-DoE: pH=%.2f yield=%.1f%%",
                                         _asm_doe_result.optimal_ph,
                                         _asm_doe_result.optimal.yield_main * 100
                                         if _asm_doe_result.optimal else 0)
                        except Exception as _asm_doe_err:
                            log.warning("Assembly Auto-DoE failed: %s", _asm_doe_err)

                        # Auto-trigger COGS Calculation (assembly path)
                        # Mirrors single-path Step 1.10h for bulk/single alignment.
                        try:
                            from src.cogs_twin import COGSInputs as _AsmCOGSInputs
                            from src.cogs_twin import calculate_cogs as _asm_auto_cogs
                            from src.cogs_twin import cogs_to_dict as _asm_cogs_to_dict

                            _asm_cogs_titer = 3.85  # default (matches upstream model)
                            _asm_cogs_yield = 0.70
                            _asm_up_d = st.session_state.get("upstream_result_dict") or {}
                            _asm_doe_d = st.session_state.get("doe_result_dict") or {}
                            if _asm_up_d.get("final_titer"):
                                _asm_cogs_titer = float(_asm_up_d["final_titer"])
                            if _asm_doe_d.get("optimal_yield"):
                                _asm_cogs_yield = float(_asm_doe_d["optimal_yield"])
                            # Round to match manual tab widget precision so auto vs manual COGS are identical
                            _asm_cogs_titer = round(_asm_cogs_titer, 2)
                            _asm_cogs_yield = round(_asm_cogs_yield * 100, 1) / 100.0

                            _asm_cogs_inputs = _AsmCOGSInputs(
                                titer_g_per_L=_asm_cogs_titer,
                                downstream_yield=_asm_cogs_yield,
                            )
                            _asm_cogs_mol_cls = intent.get("molecule_class", "canonical_mab")
                            _asm_cogs_r = _asm_auto_cogs(_asm_cogs_inputs, molecule_class=_asm_cogs_mol_cls)
                            st.session_state["cogs_result"] = _asm_cogs_r
                            _asm_cogs_dict = _asm_cogs_to_dict(_asm_cogs_r)
                            st.session_state["cogs_result_dict"] = _asm_cogs_dict
                            _asm_cache["cogs_result"] = _asm_cogs_dict
                            log.info("Assembly Auto-COGS: $%.2f/g (%s)",
                                     _asm_cogs_r.cogs_per_gram, _asm_cogs_r.cost_rating)
                        except Exception as _asm_cogs_err:
                            log.warning("Assembly Auto-COGS failed: %s", _asm_cogs_err)

                        # M14: Optimize button for assembly
                        try:
                            render_optimize_button(
                                intent=intent,
                                pk_result=_pk_asm if '_pk_asm' in dir() else None,
                                cache=_asm_cache,
                            )
                        except Exception:
                            pass

                        # Chromatogram deferred to Downstream Purification tab
                        st.caption(
                            "IEX chromatography simulation available in the **Downstream Purification** tab."
                        )

                        # ===================================================
                        # v7.3.2: Auto-detect bispecific (2 distinct chains)
                        # Only triggers when BOTH chains are the same type
                        # (e.g., two HCs in a bispecific) with <85% identity.
                        # Canonical mAbs (HC+LC) are EXCLUDED because HC/LC
                        # naturally have very low identity — that is NOT a
                        # bispecific signal.
                        # ===================================================
                        _bispec_auto_result = None
                        _should_auto_bispec = False
                        assembly_chains_for_bispec = []
                        _mol_class_lower = (intent.get("molecule_class", "") or "").lower()

                        # Gate 1: Molecule classifier already says bispecific/fusion.
                        # This overrides chain-count and chain-type restrictions
                        # because the classifier has already confirmed the molecule
                        # class.  We just need >=2 chains of sufficient length.
                        if ("bispecific" in _mol_class_lower or "fusion" in _mol_class_lower) and len(assembly_chains) >= 2:
                            # Collect ALL arm-like chains (HC or unknown, >=80 aa)
                            _hc_like = [ch for ch in assembly_chains
                                        if (ch.get("chain_type") or "").upper() in ("HC", "UNKNOWN")
                                        and len(ch.get("sequence", "")) >= 80]
                            if len(_hc_like) < 2:
                                # Fall back to all chains >=80 aa sorted by length
                                _hc_like = sorted(
                                    [ch for ch in assembly_chains if len(ch.get("sequence", "")) >= 80],
                                    key=lambda c: len(c.get("sequence", "")),
                                    reverse=True,
                                )
                            if len(_hc_like) >= 2:
                                _should_auto_bispec = True
                                # Keep ALL arm-like chains (not just 2) for pairwise analysis
                                assembly_chains_for_bispec = _hc_like

                        # Gates 2-4 only apply to exactly 2-chain assemblies
                        # where the classifier did NOT already confirm bispecific.
                        if not _should_auto_bispec and len(assembly_chains) == 2:
                            _ch_a = assembly_chains[0]
                            _ch_b = assembly_chains[1]
                            _seq_a = _ch_a.get("sequence", "")
                            _seq_b = _ch_b.get("sequence", "")
                            _type_a = (_ch_a.get("chain_type") or "unknown").upper()
                            _type_b = (_ch_b.get("chain_type") or "unknown").upper()

                            # Gate 2: Both chains are the SAME type (e.g., two HCs)
                            # and have low identity -> likely bispecific
                            if _type_a == _type_b and _type_a != "UNKNOWN":
                                if len(_seq_a) >= 80 and len(_seq_b) >= 80:
                                    from difflib import SequenceMatcher
                                    _seq_identity = SequenceMatcher(None, _seq_a, _seq_b).ratio()
                                    if _seq_identity < 0.85:
                                        _should_auto_bispec = True
                            # Gate 3: Both chains unknown type -- fall back to identity check
                            # but EXCLUDE canonical_mab classification
                            elif _type_a == "UNKNOWN" and _type_b == "UNKNOWN":
                                if "canonical" not in _mol_class_lower and "mab" not in _mol_class_lower:
                                    if len(_seq_a) >= 80 and len(_seq_b) >= 80:
                                        from difflib import SequenceMatcher
                                        _seq_identity = SequenceMatcher(None, _seq_a, _seq_b).ratio()
                                        if _seq_identity < 0.85:
                                            _should_auto_bispec = True
                            # Gate 4: Different chain types (HC+LC) -> canonical mAb, NOT bispecific
                            # This is the normal case -- do nothing.

                        # For non-Gate-1 paths, use assembly_chains directly
                        if _should_auto_bispec and not ("bispecific" in _mol_class_lower or "fusion" in _mol_class_lower):
                            assembly_chains_for_bispec = assembly_chains[:2]

                        # ── Pairwise bispecific separation analysis ──
                        # For 2 arm chains: single analysis (AA, AB, BB).
                        # For 3+ arm chains (trispecific): run ALL unique pairs
                        # so every arm-arm interaction is assessed.
                        if _should_auto_bispec:
                            from itertools import combinations
                            from difflib import SequenceMatcher
                            from src.agents import predict_bispecific_separation

                            _n_arms = len(assembly_chains_for_bispec)
                            _arm_pairs = list(combinations(range(_n_arms), 2))

                            st.markdown("---")
                            if _n_arms == 2:
                                _id_ab = SequenceMatcher(
                                    None,
                                    assembly_chains_for_bispec[0].get("sequence", ""),
                                    assembly_chains_for_bispec[1].get("sequence", ""),
                                ).ratio()
                                st.markdown(
                                    "**Bispecific Separation Analysis** -- 2 distinct chains detected "
                                    f"(sequence identity: {_id_ab:.0%}). "
                                    "Assessing homodimer/heterodimer co-elution risk..."
                                )
                            else:
                                st.markdown(
                                    f"**Multispecific Separation Analysis** -- {_n_arms} distinct arm chains detected. "
                                    f"Running {len(_arm_pairs)} pairwise homodimer/heterodimer analyses "
                                    "to assess all co-elution risks..."
                                )

                            _all_pairwise_results = []
                            _worst_risk = "Low"
                            _worst_rs = 999.0
                            _risk_order = {"Low": 0, "Medium": 1, "High": 2}

                            for _pair_idx, (_i, _j) in enumerate(_arm_pairs):
                                _ch_a = assembly_chains_for_bispec[_i]
                                _ch_b = assembly_chains_for_bispec[_j]
                                _seq_a = _ch_a.get("sequence", "")
                                _seq_b = _ch_b.get("sequence", "")
                                _name_a = _ch_a.get("name", f"Arm{_i+1}")
                                _name_b = _ch_b.get("name", f"Arm{_j+1}")

                                if _n_arms > 2:
                                    st.markdown(f"**Pair {_pair_idx+1}/{len(_arm_pairs)}: {_name_a} x {_name_b}**")

                                try:
                                    _bispec_pair_result = predict_bispecific_separation(
                                        chain_a_sequence=_seq_a,
                                        chain_b_sequence=_seq_b,
                                        chain_a_name=_name_a,
                                        chain_b_name=_name_b,
                                        gradient_slope=intent.get("gradient_slope", 15.0),
                                    )
                                    if _bispec_pair_result.get("status") == "success":
                                        _bispec_data = _bispec_pair_result.get("data", _bispec_pair_result)
                                        render_bispecific_species_panel(_bispec_data)
                                        render_bispecific_chromatogram(_bispec_data)
                                        render_bispecific_risk_panel(_bispec_data)
                                        _all_pairwise_results.append(_bispec_pair_result)

                                        # Track worst-case risk across all pairs
                                        _pair_risk = _bispec_data.get("risk", {}).get("risk_level", "Low")
                                        _pair_rs = _bispec_data.get("resolution", {}).get("min_rs", 999.0)
                                        if _risk_order.get(_pair_risk, 0) > _risk_order.get(_worst_risk, 0):
                                            _worst_risk = _pair_risk
                                        if _pair_rs < _worst_rs:
                                            _worst_rs = _pair_rs
                                    else:
                                        st.info(
                                            f"Bispecific analysis ({_name_a} x {_name_b}) returned: "
                                            f"{_bispec_pair_result.get('message', 'N/A')}"
                                        )
                                except Exception as _bispec_pair_err:
                                    st.caption(f"(Bispecific analysis {_name_a} x {_name_b} skipped: {_bispec_pair_err})")

                            # For 3+ arms, show overall risk summary
                            if _n_arms > 2 and _all_pairwise_results:
                                _risk_color = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}.get(_worst_risk, "#94A3B8")
                                st.markdown(
                                    f"**Overall Multispecific Separation Risk:** "
                                    f"<span style='color:{_risk_color};font-weight:700;'>{_worst_risk}</span> "
                                    f"(worst-case Rs = {_worst_rs:.3f} across {len(_all_pairwise_results)} pair analyses)",
                                    unsafe_allow_html=True,
                                )

                            # Store the primary result (first pair) for cache compatibility.
                            # For multispecific, also store all pairwise results.
                            if _all_pairwise_results:
                                _bispec_auto_result = _all_pairwise_results[0]
                                _asm_cache["bispecific_result"] = _bispec_auto_result
                                ws_store.update_active_field("bispecific_result", _bispec_auto_result)
                                if len(_all_pairwise_results) > 1:
                                    _asm_cache["bispecific_pairwise_results"] = _all_pairwise_results
                                    ws_store.update_active_field("bispecific_pairwise_results", _all_pairwise_results)

                        summary = (
                            f"Assembly analysis complete: **{stoich_label}**\n\n"
                            f"Global pI={asm['pI']}, MW={asm['mw_kda']} kDa, "
                            f"Liability density={liab_density['density_per_1000']:.1f}/1k residues"
                        )
                        st.markdown(summary)
                        st.session_state.messages.append({"role": "assistant", "content": summary})
                        ws_store.add_message_to_active("assistant", summary)

                        # ===================================================
                        # v7.3.1: Backend result logging + Sanity checks
                        # ===================================================
                        try:
                            from src.result_logger import ResultLogger
                            from src.sanity_check import run_sanity_checks

                            # Collect all available results for logging
                            _log_data = {
                                "molecule_name": stoich_label,
                                "assembly": {
                                    "pI": asm.get("pI"),
                                    "mw_kda": asm.get("mw_kda"),
                                    "hydrophobicity": asm.get("hydrophobicity"),
                                    "total_residues": asm.get("total_residues"),
                                    "chains": stoich_label,
                                },
                                "liability_density": liab_density.get("density_per_1000"),
                            }

                            # Add chromatography params if available
                            if "ml_override" in _asm_cache:
                                _ml = _asm_cache["ml_override"]
                                _log_data["chromatography"] = {
                                    "ka": _ml.get("ka"),
                                    "nu": _ml.get("nu"),
                                    "source": "ml_override",
                                }
                            elif "sma_params" in _asm_cache:
                                _sma = _asm_cache["sma_params"]
                                _log_data["chromatography"] = {
                                    "ka": _sma.get("ka"),
                                    "nu": _sma.get("nu"),
                                    "source": _sma.get("source", "static"),
                                }

                            # Add PK if available
                            if "_pk_asm" in dir() and _pk_asm:
                                _log_data["pk"] = {
                                    "half_life": _pk_asm.get("half_life_days"),
                                    "risk": _pk_asm.get("risk_assessment"),
                                    "total_multiplier": _pk_asm.get("total_multiplier"),
                                }

                            # Add developability if available
                            if "dev_scores" in _asm_cache:
                                _log_data["developability"] = _asm_cache["dev_scores"]

                            # Log the run
                            _logger = ResultLogger()
                            _run_id = _logger.log_run(_log_data)

                            # Run sanity checks
                            _sanity_input = {
                                "pi": asm.get("pI"),
                                "mw": asm.get("mw_kda"),
                            }
                            if "chromatography" in _log_data:
                                _sanity_input["ka"] = _log_data["chromatography"].get("ka")
                                _sanity_input["nu"] = _log_data["chromatography"].get("nu")
                            if "_pk_asm" in dir() and _pk_asm:
                                _sanity_input["half_life"] = _pk_asm.get("half_life_days")
                                _sanity_input["clearance"] = _pk_asm.get("clearance_ml_day_kg")

                            _warnings = run_sanity_checks(_sanity_input)
                            if _warnings:
                                _warn_html = (
                                    '<div style="background:#FEF3C715;border-left:5px solid #F59E0B;'
                                    'padding:12px;border-radius:8px;margin:8px 0;">'
                                    '<span style="font-size:1.05rem;font-weight:700;color:#92400E;">'
                                    f'Scientific Sanity Check: {len(_warnings)} warning(s)</span>'
                                    '<div style="font-size:0.85rem;color:#78350F;margin-top:6px;">'
                                )
                                from src.ui_colors import STATUS_DOT
                                for _w in _warnings:
                                    _sev_dot = STATUS_DOT["fail"] if _w.severity == "critical" else STATUS_DOT["caution"]
                                    _warn_html += f'{_sev_dot} {_w.message}<br/>'
                                _warn_html += '</div></div>'
                                st.markdown(_warn_html, unsafe_allow_html=True)

                            # Store run_id for reference
                            _asm_cache["last_run_id"] = _run_id

                        except Exception as _log_err:
                            import logging
                            logging.getLogger("ProtePilot").debug(
                                "Result logging/sanity check: %s", _log_err)

                        st.session_state.run_count += 1

                    else:
                        st.error(f"Parameter prediction failed: {step1.get('message', 'N/A')}")

                    ws_store.update_active_field("analysis_cache", _asm_cache)

    # ===================================================================
    #  M11: Bispecific Antibody Processing
    # ===================================================================
    pending_bispec = st.session_state.pending_bispecific
    if pending_bispec is not None:
        st.session_state.pending_bispecific = None  # consume

        chain_a_raw = pending_bispec["chain_a"]
        chain_b_raw = pending_bispec["chain_b"]
        bispec_grad = pending_bispec.get("gradient_slope", 15.0)

        # Parse chain sequences (strip FASTA headers if present)
        def _extract_seq(raw_text: str) -> Tuple[str, str]:
            """Extract name and sequence from FASTA or raw text."""
            raw_text = raw_text.strip()
            if raw_text.startswith(">"):
                lines = raw_text.split("\n")
                name = lines[0][1:].strip().split()[0] if lines[0][1:].strip() else "Chain"
                seq = re.sub(r'[^A-Za-z]', '', "".join(lines[1:])).upper()
            else:
                name = "Chain"
                seq = re.sub(r'[^A-Za-z]', '', raw_text).upper()
            return name, seq

        name_a, seq_a = _extract_seq(chain_a_raw)
        name_b, seq_b = _extract_seq(chain_b_raw)

        # Create workspace for bispecific run
        ws = ws_store.get_active()
        if ws is None:
            ws = ws_store.create_new(display_name=f"Bispecific: {name_a} x {name_b}")
            ws_store.save_to_session_state(st.session_state)

        bispec_input_msg = (
            f"Bispecific analysis: {name_a} ({len(seq_a)} aa) x {name_b} ({len(seq_b)} aa), "
            f"gradient {bispec_grad} mM/min"
        )
        st.session_state.messages.append({"role": "user", "content": bispec_input_msg})
        ws_store.add_message_to_active("user", bispec_input_msg)

        with st.chat_message("user"):
            st.markdown(bispec_input_msg)

        with st.chat_message("assistant"):
            st.markdown(f"""
            <div class="fasta-card" style="border-left: 4px solid #64748B;">
                <div class="fasta-title">Bispecific / Fusion Mode</div>
                <b>Chain A ({name_a}):</b> {len(seq_a)} aa<br>
                <b>Chain B ({name_b}):</b> {len(seq_b)} aa<br>
                Gradient: {bispec_grad} mM/min
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Running bispecific separation analysis...**")
            progress_bispec = st.progress(0, text="Building assembly species (AA, AB, BB)...")

            try:
                from src.agents import predict_bispecific_separation
                progress_bispec.progress(30, text="Mapping SMA parameters for 3 species...")

                bispec_result = predict_bispecific_separation(
                    chain_a_sequence=seq_a,
                    chain_b_sequence=seq_b,
                    chain_a_name=name_a,
                    chain_b_name=name_b,
                    gradient_slope=bispec_grad,
                )

                progress_bispec.progress(70, text="Calculating resolution and risk...")

                if bispec_result.get("status") == "success":
                    progress_bispec.progress(100, text="Analysis complete!")

                    # Store in workspace + cache
                    ws_store.update_active_field("bispecific_result", bispec_result)
                    _bispec_intent = {
                        "name": f"{name_a} x {name_b}",
                        "molecule_class": "bispecific",
                        "is_bispecific": True,
                    }
                    ws_store.update_active_field("intent", _bispec_intent)
                    _bispec_cache = {
                        "mode": "bispecific",
                        "intent": _bispec_intent,
                        "source": "bispecific",
                        "ml_override": None,
                        "source_label": "Bispecific",
                        "predictor_source": "rule_based",   # v32
                        "predictor_detail": "",              # v32
                        "ood_bypass_reason": None,           # v32
                        "variants": None,
                        "dev_result": None,
                        "cqa": None,
                        "sim_summary": None,
                        "sim_elapsed": None,
                        "bispecific_result": bispec_result,
                        "ms_characterization": None,
                    }
                    ws_store.update_active_field("analysis_cache", _bispec_cache)

                    # Render results
                    render_bispecific_species_panel(bispec_result)
                    render_bispecific_chromatogram(bispec_result)
                    render_bispecific_risk_panel(bispec_result)

                    # Summary message
                    bdata = bispec_result.get("data", bispec_result)
                    risk_info = bdata.get("risk", {})
                    res_info = bdata.get("resolution", {})
                    bispec_summary = (
                        f"Bispecific separation analysis complete for "
                        f"**{name_a} x {name_b}**.\n\n"
                        f"Risk level: **{risk_info.get('risk_level', 'N/A')}** | "
                        f"Min Rs = {res_info.get('min_rs', 0):.3f}\n\n"
                        f"Rs(AB-AA) = {res_info.get('rs_AB_AA', 0):.3f} | "
                        f"Rs(AB-BB) = {res_info.get('rs_AB_BB', 0):.3f}"
                    )
                    st.markdown(bispec_summary)
                    st.session_state.messages.append({"role": "assistant", "content": bispec_summary})
                    ws_store.add_message_to_active("assistant", bispec_summary)
                    st.session_state.run_count += 1

                else:
                    progress_bispec.progress(100, text="Analysis failed")
                    err_msg = bispec_result.get("message", "Unknown error")
                    st.error(f"Bispecific analysis failed: {err_msg}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Bispecific analysis error: {err_msg}"}
                    )

            except Exception as bispec_err:
                progress_bispec.progress(100, text="Error")
                st.error(f"Bispecific analysis error: {bispec_err}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Bispecific analysis error: {bispec_err}"}
                )

    # Close Tab 1 context (opened via _ac_tab_char.__enter__() above)
    try:
        _ac_tab_char.__exit__(None, None, None)
    except Exception:
        pass

    # If ANY analysis ran in this rerun (standard, assembly, or bispecific),
    # trigger a rerun so that the QC and Mass Spec tabs (rendered BEFORE the
    # pipeline) pick up the new cache data.  Use a one-shot flag to avoid
    # infinite rerun loops.
    _any_analysis_ran = (
        active_input is not None
        or pending_assembly is not None   # consumed at M12 block
        or pending_bispec is not None     # consumed at M11 bispecific block
    )
    if _any_analysis_ran and not st.session_state.get("_skip_post_analysis_rerun"):
        st.session_state["_skip_post_analysis_rerun"] = True
        st.rerun()
    else:
        st.session_state["_skip_post_analysis_rerun"] = False

# ===========================================================================
#  5B. Discovery & HT Screening Page (M16+M20)
# ===========================================================================
elif active_page == "Discovery & HT Screening":
    st.markdown("## Discovery & High-Throughput Screening")
    st.caption(
        "Upload a CSV of candidate sequences for bulk developability and potency scoring. "
        "Inspect data before analysis, view the Magic Quadrant, and export results."
    )

    render_ht_screening_tab()

    # -- M20: HT Data Viewer -----------------------------------------------
    # If screening results exist, show the raw dataframe for inspection
    _ht_res = st.session_state.get("ht_screening_results")
    if _ht_res and isinstance(_ht_res, dict):
        _ht_candidates = _ht_res.get("candidates")
        if _ht_candidates and isinstance(_ht_candidates, list) and len(_ht_candidates) > 0:
            st.markdown("---")
            st.markdown("### HT Data Viewer")
            st.caption("Visual inspection of all screened candidates with scores.")
            try:
                import pandas as _pd_ht
                _ht_df = _pd_ht.DataFrame(_ht_candidates)
                # Show key columns first if they exist
                _preferred_cols = ["name", "pI", "mw", "dev_score", "potency_score", "quadrant"]
                _available = [c for c in _preferred_cols if c in _ht_df.columns]
                _remaining = [c for c in _ht_df.columns if c not in _available]
                _ordered_cols = _available + _remaining
                st.dataframe(
                    _ht_df[_ordered_cols],
                    use_container_width=True,
                    height=400,
                    hide_index=True,
                )
                st.caption(f"Showing {len(_ht_df)} candidates.")
            except Exception as ht_view_err:
                st.caption(f"(Data viewer unavailable: {ht_view_err})")

    # -- Virtual Mutant Library (inference/generation only) --
    st.markdown("---")
    with st.expander("Virtual Mutant Library", expanded=True):
        st.caption(
            "Generate virtual mutants from a parent sequence, rank by expected improvement, "
            "and select candidates for experimental validation."
        )

        # Sequence input & configuration
        _vml_col1, _vml_col2 = st.columns([3, 1])
        with _vml_col1:
            _vml_seq = st.text_area(
                "Parent Sequence (amino acids)",
                value=st.session_state.get("vml_sequence", ""),
                height=100,
                key="vml_parent_seq",
                help="Enter the wild-type antibody sequence. Virtual mutants will be generated from this parent.",
            )
        with _vml_col2:
            _vml_n_mutants = st.number_input(
                "Virtual library size", min_value=100, max_value=10000,
                value=1000, step=100, key="vml_n_mutants",
                help="Number of in-silico point mutants to generate and evaluate.",
            )
            _vml_n_select = st.number_input(
                "Experiments to select", min_value=1, max_value=48,
                value=10, step=1, key="vml_n_select",
                help="Top-N candidates ranked by Expected Improvement.",
            )
            _vml_explore = st.slider(
                "Exploration weight", 0.0, 1.0, 0.5, 0.1,
                key="vml_explore_weight",
                help="0 = pure exploitation (high predicted performance), "
                     "1 = pure exploration (high uncertainty). "
                     "0.5 balances both.",
            )

        # Run Virtual Mutant Library
        if st.button("Generate & Rank Virtual Mutants", type="primary",
                      key="btn_run_vml"):
            if len(_vml_seq.strip()) < 50:
                st.error("Parent sequence must be at least 50 amino acids.")
            else:
                with st.spinner(f"Generating {_vml_n_mutants} virtual mutants and scoring..."):
                    try:
                        from src.uncertainty_engine import run_active_learning_cycle

                        # Try to load trained models for real uncertainty
                        _vml_mlp = None
                        _vml_xgb = None
                        try:
                            from src.ml_predictor import get_trained_model
                            _vml_mlp, _, _ = get_trained_model()
                        except Exception:
                            pass
                        try:
                            from src.ml_predictor import WetLabPredictor
                            _vml_xgb = st.session_state.get("wetlab_model")
                        except Exception:
                            pass

                        _vml_result = run_active_learning_cycle(
                            parent_sequence=_vml_seq.strip(),
                            n_virtual_mutants=_vml_n_mutants,
                            n_select=_vml_n_select,
                            exploration_weight=_vml_explore,
                            mlp_model=_vml_mlp,
                            xgb_model=_vml_xgb,
                        )
                        st.session_state["vml_result"] = _vml_result
                    except Exception as vml_err:
                        st.error(f"Virtual mutant generation failed: {vml_err}")
                        import traceback
                        st.caption(traceback.format_exc())

        # Display results
        _vml_result = st.session_state.get("vml_result")
        if _vml_result and _vml_result.get("status") == "success":
            al = _vml_result["al_result"]
            st.markdown("---")

            # Summary banner
            _hvml = sum(1 for s in al.selected if s.is_high_value_candidate)
            st.markdown(
                f'<div style="background:#3B82F615;border-left:4px solid #3B82F6;'
                f'padding:16px;border-radius:8px;margin:8px 0;">'
                f'<span style="font-size:1.1rem;font-weight:700;color:#3B82F6;">'
                f'Scanned {al.candidate_pool_size:,} virtual mutants '
                f'&rarr; {al.n_selected} recommended experiments</span>'
                f'<span style="margin-left:24px;font-size:0.9rem;color:#64748B;">'
                f'{_hvml} high-value candidates | '
                f'Best EI: {al.best_ei:.4f} | Time: {al.wall_time_s:.2f}s</span></div>',
                unsafe_allow_html=True,
            )

            # Metrics
            _vm1, _vm2, _vm3, _vm4 = st.columns(4)
            _vm1.metric("Library Size", f"{al.candidate_pool_size:,}")
            _vm2.metric("Selected", f"{al.n_selected}")
            _vm3.metric("High-Value", f"{_hvml}")
            _vm4.metric("Best EI", f"{al.best_ei:.4f}")

            # Recommended experiments table
            st.markdown("#### Recommended Experiments")
            _vml_rows = []
            for rank, s in enumerate(al.selected, 1):
                _vml_rows.append({
                    "Rank": rank,
                    "Name": s.name[:40],
                    "EI Score": f"{s.expected_improvement:.4f}",
                    "Performance": f"{s.predicted_performance:.3f}",
                    "Uncertainty": f"{s.combined_uncertainty:.3f}",
                    "Pred Agg%": f"{s.agg_mean:.1f} +/- {s.agg_std:.1f}",
                    "Pred Tm": f"{s.tm_mean:.1f} +/- {s.tm_std:.1f}",
                    "High-Value": "Yes" if s.is_high_value_candidate else "No",
                })
            try:
                import pandas as pd
                _vml_df = pd.DataFrame(_vml_rows)
                st.dataframe(_vml_df, use_container_width=True, hide_index=True)
            except ImportError:
                for row in _vml_rows:
                    st.text(f"  #{row['Rank']} {row['Name']} — EI={row['EI Score']}")

            st.info("Proceed to **AI Training Center** → **Active Learning** tab to generate plate layouts and upload results.")

    # -- M27: Pareto Frontier Analysis -----------------------------------------
    if _ht_res and isinstance(_ht_res, dict):
        _pareto_cands = _ht_res.get("candidates")
        if _pareto_cands and isinstance(_pareto_cands, list) and len(_pareto_cands) >= 3:
            st.markdown("---")
            st.markdown("### Multi-Objective Pareto Frontier")
            st.caption(
                "Identifies the Pareto-optimal set: candidates where no other is simultaneously "
                "better in **all** objectives. These represent the best realistic compromise "
                "between Efficacy and Manufacturability."
            )
            try:
                from src.pareto_optimizer import run_pareto_analysis, ParetoResult
                import plotly.graph_objects as _pgo_pareto

                # Run Pareto analysis on HT screening candidates
                _pareto_result: ParetoResult = run_pareto_analysis(_pareto_cands)

                # --- Summary Metrics ---
                _pm1, _pm2, _pm3, _pm4 = st.columns(4)
                with _pm1:
                    st.metric("Total Candidates", _pareto_result.n_total)
                with _pm2:
                    st.metric("Pareto-Optimal", _pareto_result.n_pareto)
                with _pm3:
                    st.metric("Frontier %", f"{_pareto_result.frontier_fraction:.1%}")
                with _pm4:
                    _best = _pareto_result.candidates[0] if _pareto_result.candidates else None
                    st.metric("Top Candidate", _best.name if _best else "—")

                # --- Pareto Frontier Scatter (Efficacy vs Developability) ---
                if len(_pareto_result.objective_names) >= 2:
                    _obj_x = _pareto_result.objective_names[0]  # Efficacy
                    _obj_y = _pareto_result.objective_names[1]  # Developability

                    _frontier_x = [c.objectives[_obj_x] for c in _pareto_result.candidates if c.is_pareto_optimal]
                    _frontier_y = [c.objectives[_obj_y] for c in _pareto_result.candidates if c.is_pareto_optimal]
                    _frontier_names = [c.name for c in _pareto_result.candidates if c.is_pareto_optimal]

                    _dom_x = [c.objectives[_obj_x] for c in _pareto_result.candidates if not c.is_pareto_optimal]
                    _dom_y = [c.objectives[_obj_y] for c in _pareto_result.candidates if not c.is_pareto_optimal]
                    _dom_names = [c.name for c in _pareto_result.candidates if not c.is_pareto_optimal]

                    _pfig = _pgo_pareto.Figure()

                    # Dominated points (grey)
                    _pfig.add_trace(_pgo_pareto.Scatter(
                        x=_dom_x, y=_dom_y,
                        mode="markers",
                        marker=dict(size=8, color=PLOTLY_COLORS[5], line=dict(width=1, color="#94A3B8")),
                        text=_dom_names,
                        name="Dominated",
                        hovertemplate="%{text}<br>" + _obj_x + ": %{x:.2f}<br>" + _obj_y + ": %{y:.2f}<extra></extra>",
                    ))

                    # Pareto frontier points (gold)
                    _pfig.add_trace(_pgo_pareto.Scatter(
                        x=_frontier_x, y=_frontier_y,
                        mode="markers+text",
                        marker=dict(size=14, color=PLOTLY_COLORS[2], symbol="star", line=dict(width=1.5, color="#F59E0B")),
                        text=_frontier_names,
                        textposition="top center",
                        name="Pareto Frontier",
                        hovertemplate="%{text}<br>" + _obj_x + ": %{x:.2f}<br>" + _obj_y + ": %{y:.2f}<extra></extra>",
                    ))

                    # Connect frontier with dashed line (sorted by x)
                    _sorted_f = sorted(zip(_frontier_x, _frontier_y), key=lambda p: p[0], reverse=True)
                    if len(_sorted_f) > 1:
                        _pfig.add_trace(_pgo_pareto.Scatter(
                            x=[p[0] for p in _sorted_f], y=[p[1] for p in _sorted_f],
                            mode="lines",
                            line=dict(dash="dash", color="#F59E0B", width=2),
                            showlegend=False,
                            hoverinfo="skip",
                        ))

                    _apply_pharma_theme(_pfig,
                        title="Pareto Frontier: Efficacy vs Developability",
                        xaxis_title=_obj_x,
                        yaxis_title=_obj_y,
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(_pfig, use_container_width=True)

                # --- Pareto Candidates Table ---
                with st.expander("Pareto-Optimal Candidates Detail", expanded=True):
                    import pandas as _pd_pareto
                    _pareto_rows = []
                    for _pc in _pareto_result.candidates:
                        _row = {"Candidate": _pc.name, "Rank": _pc.pareto_rank}
                        for _on, _ov in _pc.objectives.items():
                            _row[_on] = round(_ov, 3)
                        _row["Weighted Score"] = round(_pc.weighted_score, 3)
                        _row["Pareto Optimal"] = "Yes" if _pc.is_pareto_optimal else "No"
                        _pareto_rows.append(_row)
                    _pareto_df = _pd_pareto.DataFrame(_pareto_rows)
                    st.dataframe(_pareto_df, use_container_width=True, height=300, hide_index=True)

                st.success(f"Pareto analysis complete — {_pareto_result.n_pareto} frontier candidates identified.")

            except Exception as pareto_err:
                st.caption(f"(Pareto analysis unavailable: {pareto_err})")

    # -- Training modules are in AI Training Center ---
    st.markdown("---")
    st.info(
        "Model training, Mock Jain-137 dataset generation, and Continuous Learning "
        "are consolidated in the **AI Training Center** tab. Navigate there to "
        "train or retrain models before screening.",
    )


# ===========================================================================
#  5B-DEV. Developability Dashboard (Phase 2 — Integration Decision Layer)
# ===========================================================================
elif active_page == "Developability Dashboard":
    st.markdown("## Developability Dashboard")
    st.caption(
        "Unified risk assessment integrating evidence from all analytical modules. "
        "Molecule-class-aware scoring with ICH Q8 QTPP alignment."
    )

    # -- Hard gate: require molecule --
    enforce_molecule_state(ws_store)

    _active_ws = ws_store.get_active() if ws_store else None
    _dev_intent = _active_ws.get("intent") if _active_ws else st.session_state.get("last_intent")

    if _dev_intent:
        try:
            from src.developability_core import assess_developability, DevelopabilityAssessment

            # Collect evidence from all available modules
            _dev_cache = (_active_ws.get("analysis_cache") or {}) if _active_ws else {}
            _dev_result_raw = _dev_cache.get("dev_result", {})
            _dev_data = _dev_result_raw.get("data", _dev_result_raw) if isinstance(_dev_result_raw, dict) else {}
            _dev_preds = _dev_data.get("predictions", {})

            # Feature values from FeatureRegistry (or fallback from intent)
            _fs_obj = _dev_intent.get("feature_set_obj")
            if _fs_obj and hasattr(_fs_obj, "value"):
                _feat_vals = {
                    name: _fs_obj.value(name)
                    for name in ["pI", "mw_kda", "hydrophobicity", "deam_sites", "ox_sites",
                                 "acidic_residues", "basic_residues", "beta_sheet_propensity",
                                 "cdr_hydrophobicity", "asp_isomerization_sites",
                                 "pyroglutamate_risk", "n_glycosylation_sites",
                                 "seq_length", "cysteine_count"]
                    if _fs_obj.value(name) is not None
                }
            else:
                _feat_vals = {
                    "pI": _dev_intent.get("pI"),
                    "mw_kda": _dev_intent.get("mw"),
                    "hydrophobicity": _dev_intent.get("hydrophobicity"),
                    "deam_sites": _dev_intent.get("deam_sites", 1),
                    "ox_sites": _dev_intent.get("ox_sites", 1),
                    "seq_length": _dev_intent.get("seq_length", 0),
                }

            # Analytical QC results
            _anal_results = {}
            _qc = _dev_cache.get("analytical_qc")
            if _qc and isinstance(_qc, dict):
                _cief = _qc.get("cief", {})
                _sec = _qc.get("sec", {})
                _cesds = _qc.get("ce_sds", {})
                if hasattr(_cief, "main_pct"):
                    _anal_results["cief_main_pct"] = _cief.main_pct
                    _anal_results["cief_acidic_pct"] = getattr(_cief, "acidic_pct", None)
                elif isinstance(_cief, dict):
                    _anal_results["cief_main_pct"] = _cief.get("main_pct")
                    _anal_results["cief_acidic_pct"] = _cief.get("acidic_pct")
                if hasattr(_sec, "monomer_pct"):
                    _anal_results["sec_monomer_pct"] = _sec.monomer_pct
                    _anal_results["sec_hmw_pct"] = _sec.hmw_pct
                elif isinstance(_sec, dict):
                    _anal_results["sec_monomer_pct"] = _sec.get("monomer_pct")
                    _anal_results["sec_hmw_pct"] = _sec.get("hmw_pct")
                if hasattr(_cesds, "intact_pct"):
                    _anal_results["cesds_intact_pct"] = _cesds.intact_pct
                elif isinstance(_cesds, dict):
                    _anal_results["cesds_intact_pct"] = _cesds.get("intact_pct")

            # Stability results
            _stab_results = {}
            _stab_raw = _dev_cache.get("stability_result")
            if isinstance(_stab_raw, dict):
                _stab_results["shelf_life_months"] = _stab_raw.get("shelf_life_months")

            # PK results
            _pk_results = {}
            _pk_raw = _dev_cache.get("pk_result")
            if isinstance(_pk_raw, dict):
                _pk_data = _pk_raw.get("data", _pk_raw)
                _pk_results["half_life_days"] = _pk_data.get("half_life_days")

            # ADA results — extract from dataclass or dict (session_state or cache)
            _ada_results = {}
            _ada_raw = st.session_state.get("ada_result") or _dev_cache.get("ada_result")
            if _ada_raw is not None:
                if isinstance(_ada_raw, dict):
                    _ada_results["ada_risk_level"] = _ada_raw.get("ada_risk_level", "")
                    _ada_results["ada_risk_score"] = _ada_raw.get("ada_risk_score")
                else:
                    _ada_results["ada_risk_level"] = getattr(_ada_raw, "ada_risk_level", "")
                    _ada_results["ada_risk_score"] = getattr(_ada_raw, "ada_risk_score", None)

            # Upstream results
            _ups_results = {}
            _ups_raw = st.session_state.get("upstream_result_dict")
            if isinstance(_ups_raw, dict):
                _ups_results["final_titer"] = _ups_raw.get("final_titer")

            # ── Acceptance Criteria Editor (renders first for immediate reactivity) ──
            from src.developability_core import QTPP_ACCEPTANCE_DEFAULTS as _QTPP_AD
            with st.expander("Acceptance Criteria (editable)", expanded=True):
                st.caption(
                    "Adjust per-molecule acceptance thresholds. "
                    "Changes update the QTPP status and Virtual QC pass/fail in real time."
                )
                _qc1, _qc2, _qc3, _qc4 = st.columns(4)
                with _qc1:
                    _uc_sec = st.number_input(
                        "SEC Monomer min %", min_value=80.0, max_value=99.9,
                        value=float(st.session_state.get("qc_sec_min") or _QTPP_AD["sec_monomer"]["accept_lower"]),
                        step=0.5, key="qc_crit_sec_dev")
                with _qc2:
                    _uc_cesds = st.number_input(
                        "CE-SDS Intact min %", min_value=80.0, max_value=99.9,
                        value=float(st.session_state.get("qc_cesds_min") or _QTPP_AD["cesds_intact"]["accept_lower"]),
                        step=0.5, key="qc_crit_cesds_dev")
                with _qc3:
                    _uc_acidic = st.number_input(
                        "cIEF Acidic max %", min_value=5.0, max_value=60.0,
                        value=float(st.session_state.get("qc_acidic_max") or _QTPP_AD["cief_acidic"]["accept_upper"]),
                        step=1.0, key="qc_crit_acidic_dev")
                with _qc4:
                    _uc_main = st.number_input(
                        "cIEF Main min %", min_value=30.0, max_value=90.0,
                        value=float(st.session_state.get("qc_main_min") or _QTPP_AD["cief_main"]["accept_lower"]),
                        step=1.0, key="qc_crit_main_dev")
                # Sync to unified session_state keys (used by Analytical page too)
                st.session_state["qc_sec_min"] = _uc_sec
                st.session_state["qc_cesds_min"] = _uc_cesds
                st.session_state["qc_acidic_max"] = _uc_acidic
                st.session_state["qc_main_min"] = _uc_main

            # Build user_criteria from the LIVE widget values (not stale session_state)
            _uc = {
                "sec_monomer":  {"accept_lower": float(_uc_sec)},
                "cesds_intact": {"accept_lower": float(_uc_cesds)},
                "cief_acidic":  {"accept_upper": float(_uc_acidic)},
                "cief_main":    {"accept_lower": float(_uc_main)},
            }

            # Run assessment
            _mol_class = _dev_intent.get("molecule_class", "unknown")

            _assessment = assess_developability(
                molecule_name=_dev_intent.get("name", "Unknown"),
                molecule_class=_mol_class,
                feature_values=_feat_vals,
                dev_predictions=_dev_preds,
                analytical_results=_anal_results,
                stability_results=_stab_results,
                pk_results=_pk_results,
                ada_results=_ada_results,
                upstream_results=_ups_results,
                user_criteria=_uc,
            )

            # ── Header: Molecule + Class + Composite Score ────────────
            from src.ui_colors import COLORS as _UI_C
            _rec_colors = {
                "Proceed": _UI_C["pass"]["primary"],
                "Proceed with caution": _UI_C["caution"]["primary"],
                "Optimize before proceeding": _UI_C["fail"]["primary"],
            }
            _rec_color = _rec_colors.get(_assessment.recommendation, _UI_C["neutral"]["primary"])

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f0f4f8, #e8f0fe);
                        padding: 16px 20px; border-radius: 8px; margin-bottom: 16px;
                        border-left: 5px solid {_assessment.composite_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.3em; font-weight: 700;">{_assessment.molecule_name}</span>
                        <span style="background: #e0e7ff; padding: 3px 10px; border-radius: 12px;
                                     font-size: 0.82em; margin-left: 10px;">
                            {_assessment.molecule_class_display}
                        </span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 2.0em; font-weight: 700; color: {_assessment.composite_color};">
                            {_assessment.composite_score:.2f}
                        </span>
                        <br>
                        <span style="background: {_assessment.composite_color}; color: white;
                                     padding: 3px 12px; border-radius: 12px; font-size: 0.85em;">
                            {_assessment.composite_grade}
                        </span>
                    </div>
                </div>
                <div style="margin-top: 10px; padding: 8px 12px; background: white;
                            border-radius: 8px; border-left: 4px solid {_rec_color};">
                    <span style="font-weight: 600; color: {_rec_color};">
                        Recommendation: {_assessment.recommendation}
                    </span>
                    <br><span style="font-size: 0.88em; color: #64748B;">
                        {_assessment.recommendation_detail}
                    </span>
                </div>
                <div style="margin-top: 6px; font-size: 0.78em; color: #94A3B8;">
                    Confidence: {_assessment.confidence} · Sources: {", ".join(_assessment.model_sources[:3])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Radar Chart (Spider Plot) ─────────────────────────────
            st.markdown("### Risk Profile Radar")
            st.caption(
                "Spider chart showing each risk dimension (0 = best, 1 = worst). "
                "The shaded area and weighted overlay reveal where a molecule's key shortcomings lie."
            )

            _radar = _assessment.radar_data()
            if _radar["labels"]:
                import plotly.graph_objects as _go_radar

                _r_labels = _radar["labels"] + [_radar["labels"][0]]  # close the polygon
                _r_scores = _radar["scores"] + [_radar["scores"][0]]
                _r_weights = _radar["weights"] + [_radar["weights"][0]]
                # Weighted scores = score * weight (normalized to show relative contribution)
                _r_weighted = [s * w for s, w in zip(_radar["scores"], _radar["weights"])]
                _r_weighted_closed = _r_weighted + [_r_weighted[0]]

                _radar_fig = _go_radar.Figure()

                # Layer 1: Raw risk scores (outer boundary)
                _radar_fig.add_trace(_go_radar.Scatterpolar(
                    r=_r_scores,
                    theta=_r_labels,
                    fill="toself",
                    fillcolor="rgba(99, 102, 241, 0.12)",
                    line=dict(color="#6366f1", width=2),
                    name="Risk Score",
                    hovertemplate="%{theta}<br>Score: %{r:.2f}<extra></extra>",
                ))

                # Layer 2: Weighted contribution (inner shading)
                _radar_fig.add_trace(_go_radar.Scatterpolar(
                    r=_r_weighted_closed,
                    theta=_r_labels,
                    fill="toself",
                    fillcolor="rgba(239, 68, 68, 0.15)",
                    line=dict(color="#ef4444", width=1.5, dash="dot"),
                    name="Weighted Contribution",
                    hovertemplate="%{theta}<br>Weighted: %{r:.3f}<extra></extra>",
                ))

                # Reference ring at 0.5 (moderate risk threshold)
                _ref_r = [0.5] * (len(_radar["labels"]) + 1)
                _radar_fig.add_trace(_go_radar.Scatterpolar(
                    r=_ref_r,
                    theta=_r_labels,
                    line=dict(color="#d1d5db", width=1, dash="dash"),
                    name="Moderate Threshold (0.5)",
                    showlegend=True,
                    hoverinfo="skip",
                ))

                # Dimension markers with color coding
                for _i, (_lbl, _sc, _clr) in enumerate(
                    zip(_radar["labels"], _radar["scores"], _radar["colors"])
                ):
                    _radar_fig.add_trace(_go_radar.Scatterpolar(
                        r=[_sc],
                        theta=[_lbl],
                        mode="markers+text",
                        marker=dict(size=10, color=_clr, line=dict(width=1, color="white")),
                        text=[f"{_sc:.2f}"],
                        textposition="top center",
                        textfont=dict(size=10, color=_clr),
                        showlegend=False,
                        hovertemplate=f"{_lbl}<br>Score: {_sc:.2f}<br>Weight: {_radar['weights'][_i]:.0%}<extra></extra>",
                    ))

                _radar_is_dark = st.session_state.get("dark_mode", False)
                _radar_grid = "#334155" if _radar_is_dark else "#e5e7eb"
                _radar_line = "#475569" if _radar_is_dark else "#d1d5db"
                _radar_bg = "#1E293B" if _radar_is_dark else "white"
                _radar_txt = "#E2E8F0" if _radar_is_dark else None
                _radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True, range=[0, 1],
                            tickvals=[0.2, 0.4, 0.6, 0.8],
                            ticktext=["0.2", "0.4", "0.6", "0.8"],
                            gridcolor=_radar_grid,
                            linecolor=_radar_line,
                            tickfont=dict(color=_radar_txt) if _radar_txt else {},
                        ),
                        angularaxis=dict(
                            gridcolor=_radar_grid,
                            linecolor=_radar_line,
                            tickfont=dict(color=_radar_txt) if _radar_txt else {},
                        ),
                        bgcolor=_radar_bg,
                    ),
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=-0.15,
                        xanchor="center", x=0.5,
                        font=dict(size=11, color=_radar_txt or "#334155"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=60, r=60, t=30, b=60),
                    height=420,
                )

                st.plotly_chart(_radar_fig, use_container_width=True)

                # Weight breakdown bar (compact horizontal summary)
                _w_parts = []
                for _lbl, _wt, _clr in zip(_radar["labels"], _radar["weights"], _radar["colors"]):
                    _w_parts.append(
                        f'<span style="display:inline-block; background:{_clr}22; border:1px solid {_clr};'
                        f' border-radius:6px; padding:2px 8px; margin:2px; font-size:0.78em;">'
                        f'{_lbl}: <b>{_wt:.0%}</b></span>'
                    )
                st.markdown(
                    '<div style="text-align:center; margin-bottom:12px;">'
                    '<span style="font-size:0.8em; color:#94A3B8;">Weights: </span>'
                    + "".join(_w_parts) + '</div>',
                    unsafe_allow_html=True,
                )

            # ── Risk Dimension Cards ──────────────────────────────────
            st.markdown("### Risk Dimensions")
            st.caption("Each dimension is scored 0 (best) to 1 (worst) with molecule-class-specific weights.")

            for _dim in _assessment.dimensions:
                _bar_width = max(3, min(100, int(_dim.score * 100)))
                st.markdown(f"""
                <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px;
                            padding: 12px 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600;">{_dim.display_name}</span>
                        <span style="font-size: 1.1em; font-weight: 700; color: {_dim.color};">
                            {_dim.score:.2f}
                            <span style="font-size: 0.75em; font-weight: 400; color: #94A3B8;">
                                (weight: {_dim.weight:.0%})
                            </span>
                        </span>
                    </div>
                    <div style="background: #f3f4f6; border-radius: 6px; height: 8px; margin: 6px 0;">
                        <div style="background: {_dim.color}; height: 8px; border-radius: 6px;
                                    width: {_bar_width}%;"></div>
                    </div>
                    <div style="font-size: 0.85em; color: #64748B; margin-top: 4px;">
                        {_dim.explanation}
                    </div>
                """, unsafe_allow_html=True)

                # Evidence bullets
                if _dim.evidence:
                    _ev_html = "".join(
                        f'<div style="font-size:0.82em; color:#64748B; padding-left:12px;">• {e}</div>'
                        for e in _dim.evidence
                    )
                    st.markdown(_ev_html, unsafe_allow_html=True)

                st.markdown(f"""
                    <div style="font-size: 0.75em; color: #94A3B8; margin-top: 4px;">
                        Source: {_dim.source} · Confidence: {_dim.confidence}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── QTPP Table (ICH Q8) ───────────────────────────────────
            st.markdown("### Quality Target Product Profile (QTPP)")
            st.caption(
                "ICH Q8(R2) framework — Critical Quality Attributes with target values, "
                "acceptable ranges, and current platform predictions."
            )

            _qtpp_rows = _assessment.qtpp
            if _qtpp_rows:
                # Use pandas DataFrame for reliable rendering
                import pandas as _pd_qtpp

                _qtpp_data = []
                for _r in _qtpp_rows:
                    _flag = ""  # Risk flag handled via Status column color
                    _qtpp_data.append({
                        "CQA": _r.attribute,
                        "Target": _r.target,
                        "Acceptable Range": _r.acceptable_range,
                        "Current Prediction": f"{_r.current_prediction}{_flag}",
                        "Status": _r.status,
                        "Model / Source": getattr(_r, "model_source", "") or "",
                    })

                _qtpp_df = _pd_qtpp.DataFrame(_qtpp_data)

                # Color-coded status using unified palette
                from src.ui_colors import COLORS

                _status_color_map = {
                    "Within Target": {"bg": COLORS["pass"]["bg"],    "fg": COLORS["pass"]["primary"]},
                    "Within Range":  {"bg": COLORS["caution"]["bg"], "fg": COLORS["caution"]["primary"]},
                    "Out of Range":  {"bg": COLORS["fail"]["bg"],    "fg": COLORS["fail"]["primary"]},
                    "Not Assessed":  {"bg": COLORS["neutral"]["bg"], "fg": COLORS["neutral"]["primary"]},
                }

                def _style_qtpp_status(val):
                    c = _status_color_map.get(val, {})
                    if c:
                        return f"background-color: {c['bg']}; color: {c['fg']}; font-weight: 600"
                    return ""

                _qtpp_styled = _qtpp_df.style.applymap(
                    _style_qtpp_status, subset=["Status"]
                )

                st.dataframe(
                    _qtpp_styled,
                    use_container_width=True,
                    hide_index=True,
                    height=min(600, 40 + 35 * len(_qtpp_data)),
                    column_config={
                        "CQA": st.column_config.TextColumn("CQA", width="medium"),
                        "Target": st.column_config.TextColumn("Target", width="small"),
                        "Acceptable Range": st.column_config.TextColumn("Acceptable Range", width="small"),
                        "Current Prediction": st.column_config.TextColumn("Current Prediction", width="small"),
                        "Status": st.column_config.TextColumn("Status", width="small"),
                    },
                )

                # Summary stats with color-coded labels
                _n_target = sum(1 for r in _qtpp_rows if r.status == "Within Target")
                _n_range = sum(1 for r in _qtpp_rows if r.status == "Within Range")
                _n_oor = sum(1 for r in _qtpp_rows if r.status == "Out of Range")
                _n_na = sum(1 for r in _qtpp_rows if r.status == "Not Assessed")

                _sum_c1, _sum_c2, _sum_c3, _sum_c4 = st.columns(4)
                with _sum_c1:
                    st.markdown(
                        f'<div style="text-align:center;"><span style="color:{COLORS["pass"]["primary"]};'
                        f'font-size:1.8rem;font-weight:700;">{_n_target}</span><br/>'
                        f'<span style="color:{COLORS["pass"]["primary"]};font-size:0.8rem;">Within Target</span></div>',
                        unsafe_allow_html=True)
                with _sum_c2:
                    st.markdown(
                        f'<div style="text-align:center;"><span style="color:{COLORS["caution"]["primary"]};'
                        f'font-size:1.8rem;font-weight:700;">{_n_range}</span><br/>'
                        f'<span style="color:{COLORS["caution"]["primary"]};font-size:0.8rem;">Within Range</span></div>',
                        unsafe_allow_html=True)
                with _sum_c3:
                    st.markdown(
                        f'<div style="text-align:center;"><span style="color:{COLORS["fail"]["primary"]};'
                        f'font-size:1.8rem;font-weight:700;">{_n_oor}</span><br/>'
                        f'<span style="color:{COLORS["fail"]["primary"]};font-size:0.8rem;">Out of Range</span></div>',
                        unsafe_allow_html=True)
                with _sum_c4:
                    st.markdown(
                        f'<div style="text-align:center;"><span style="color:{COLORS["neutral"]["primary"]};'
                        f'font-size:1.8rem;font-weight:700;">{_n_na}</span><br/>'
                        f'<span style="color:{COLORS["neutral"]["primary"]};font-size:0.8rem;">Not Assessed</span></div>',
                        unsafe_allow_html=True)

                st.caption(
                    "QTPP based on ICH Q8(R2), Q6B, and Q5C guidelines. "
                    "Targets reflect industry standard acceptance criteria for clinical-stage biologics."
                )

            # ── Advisory Panel (3 Role Summaries) ───────────────────────
            st.markdown("---")
            st.markdown("### Advisory Panel")
            st.caption(
                "Three domain-expert perspectives on your molecule's development feasibility."
            )

            try:
                from src.advisory_panel import run_advisory_panel
                _panel_state = {k: v for k, v in st.session_state.items()
                                if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                _panel_state["_workspace_store"] = ws_store
                _panel_result = run_advisory_panel(_panel_state)
                st.session_state["advisory_panel_result"] = _panel_result

                # Risk banner
                _panel_risk_colors = {"Low": COLORS["pass"]["primary"], "Medium": COLORS["caution"]["primary"],
                                      "High": COLORS["fail"]["primary"], "Critical": "#EF4444"}
                _panel_risk_color = _panel_risk_colors.get(_panel_result.overall_risk, "#64748B")
                st.markdown(
                    f'<div style="background:{_panel_risk_color}15;border-left:4px solid {_panel_risk_color};'
                    f'padding:12px 16px;border-radius:8px;margin:8px 0;">'
                    f'<span style="font-size:1.1rem;font-weight:700;color:{_panel_risk_color};">'
                    f'Overall Risk Assessment: {_panel_result.overall_risk}</span><br/>'
                    f'<span style="font-size:0.9rem;color:#334155;">'
                    f'{_panel_result.overall_recommendation}</span></div>',
                    unsafe_allow_html=True,
                )

                if _panel_result.all_risk_flags:
                    with st.expander(f"Risk Flags ({len(_panel_result.all_risk_flags)})", expanded=True):
                        for _rf in _panel_result.all_risk_flags:
                            st.caption(f"WARNING: {_rf}")

                # Three role cards in columns
                _adv_c1, _adv_c2, _adv_c3 = st.columns(3)
                for _col, _role in zip(
                    [_adv_c1, _adv_c2, _adv_c3],
                    [_panel_result.upstream, _panel_result.downstream, _panel_result.quality],
                ):
                    _r_color = _panel_risk_colors.get(_role.risk_level, "#64748B")
                    with _col:
                        st.markdown(
                            f'<div style="background:#f8f9fa;border-radius:8px;padding:12px;'
                            f'border-top:3px solid {_r_color};min-height:180px;">'
                            f'<span style="font-weight:700;font-size:0.9rem;">{_role.role_name}</span><br/>'
                            f'<span style="font-size:0.75rem;color:#64748B;">{_role.role_title}</span><br/>'
                            f'<span style="display:inline-block;margin-top:6px;padding:2px 8px;'
                            f'border-radius:8px;font-size:0.75rem;font-weight:600;'
                            f'background:{_r_color}15;color:{_r_color};">{_role.risk_level}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        for _kf in _role.key_findings:
                            st.caption(_kf)
                        if _role.recommendation:
                            st.caption(f"→ {_role.recommendation}")
            except Exception as _panel_err:
                st.caption(f"Advisory panel: {_panel_err}")

        except Exception as _dev_dash_err:
            st.error(f"Developability Dashboard error: {_dev_dash_err}")
            log.error("Developability Dashboard error: %s", _dev_dash_err, exc_info=True)
    else:
        st.info("Load a molecule in the Molecule Setup page to see the developability assessment.")


# ===========================================================================
#  5C. Upstream Cell Culture Page (M20 — NEW)
# ===========================================================================
elif active_page == "Process Development":
    st.markdown("## Process Development")
    st.caption(
        "Upstream cell culture simulation and downstream purification optimization "
        "in a unified workspace."
    )

    # -- Hard gate: require molecule --
    enforce_molecule_state(ws_store)

    _proc_tab_up, _proc_tab_down, _proc_tab_cogs = st.tabs(["Upstream Cell Culture", "Downstream Purification", "Manufacturability & COGS"])

    with _proc_tab_up:
        st.markdown("### Upstream Bioreactor Digital Twin")
        st.caption(
            "Simulate a 14-day CHO Fed-Batch culture using ODE kinetics. "
            "Viable Cell Density, Titer, Glucose, and Lactate dynamics."
        )

        # -- Inputs ---------------------------------------------------------------
        _up_c1, _up_c2, _up_c3 = st.columns(3)
        with _up_c1:
            _up_seed = st.number_input(
                "Seed Density (10^6 cells/mL)",
                min_value=0.1, max_value=5.0, value=0.5, step=0.1,
                key="upstream_seed_density",
                help="Initial viable cell density at inoculation.",
            )
        with _up_c2:
            _up_temp_day = st.number_input(
                "Temperature Shift Day",
                min_value=0.0, max_value=14.0, value=5.0, step=1.0,
                key="upstream_temp_shift_day",
                help="Day of hypothermic shift (37 to 33 C). Set 0 for no shift.",
            )
        with _up_c3:
            _up_duration = st.number_input(
                "Culture Duration (days)",
                min_value=7.0, max_value=21.0, value=14.0, step=1.0,
                key="upstream_duration",
                help="Total culture duration before harvest.",
            )

        # -- Developability coupling ---
        _up_dev_score = None
        _up_agg_risk = None
        _last_intent = st.session_state.get("last_intent")
        _active_ws = ws_store.get_active()
        # Fallback: if last_intent is truly missing (None), use workspace intent
        if _last_intent is None and _active_ws and _active_ws.get("intent"):
            _last_intent = _active_ws["intent"]
            st.session_state.last_intent = _last_intent
        if not isinstance(_last_intent, dict):
            _last_intent = {}
        if _active_ws and _active_ws.get("analysis_cache"):
            _dev_data = (_active_ws["analysis_cache"].get("dev_result") or {}).get("data", {})
            if _dev_data:
                # Extract score: main path has data.score.score,
                # fallback path has data.developability_score
                _up_dev_score = (
                    _dev_data.get("composite_score")                     # 5-dim (if set)
                    or _dev_data.get("developability_score")             # fallback path
                    or (_dev_data.get("score", {}) or {}).get("score")   # main path
                )
                _up_agg_risk = (_dev_data.get("predictions") or {}).get("agg_risk")

        with st.expander("Advanced Engineering Parameters", expanded=True):
            _dev_col1, _dev_col2 = st.columns(2)
            with _dev_col1:
                if _up_dev_score is not None:
                    st.info(f"Developability Score coupled: **{_up_dev_score:.2f}** (from active workspace)")
                else:
                    st.caption("Note: No developability score available yet. Run the main analysis first (Step 0) to couple developability into the upstream model. Default: full productivity assumed.")
            with _dev_col2:
                if _up_agg_risk is not None:
                    st.info(f"Aggregation Risk coupled: **{_up_agg_risk:.2f}**")

        # -- Run simulation -------------------------------------------------------
        if st.button("Run Fed-Batch Simulation", type="primary", key="btn_upstream_run"):
            with st.spinner("Simulating CHO Fed-Batch culture: ODE kinetics for VCD, titer, glucose, and lactate dynamics..."):
                try:
                    from src.upstream_twin import run_upstream_simulation, result_to_dict

                    _up_result = run_upstream_simulation(
                        seed_density=_up_seed,
                        temp_shift_day=_up_temp_day,
                        dev_score=_up_dev_score,
                        agg_risk=_up_agg_risk,
                        culture_days=_up_duration,
                        hydrophobicity=_last_intent.get("gravy") if isinstance(_last_intent, dict) else None,
                        sequence=_last_intent.get("sequence") if isinstance(_last_intent, dict) else None,
                        molecule_class=_last_intent.get("molecule_class") if isinstance(_last_intent, dict) else None,
                    )
                    st.session_state["upstream_result"] = _up_result
                    st.session_state["upstream_result_dict"] = result_to_dict(_up_result)
                except Exception as up_err:
                    st.error(f"Simulation failed: {up_err}")
                    import traceback
                    st.caption(traceback.format_exc())

        # -- Display results -------------------------------------------------------
        _up_result = st.session_state.get("upstream_result")
        if _up_result is not None:
            st.markdown("---")
            st.markdown("### Culture Performance")

            _m1, _m2, _m3, _m4 = st.columns(4)
            with _m1:
                st.metric("Peak VCD", f"{_up_result.peak_vcd:.1f} x10^6/mL", f"Day {_up_result.peak_vcd_day:.0f}")
            with _m2:
                st.metric("Harvest Titer", f"{_up_result.final_titer:.2f} g/L")
            with _m3:
                st.metric("Viability", f"{_up_result.viability_at_harvest:.0f}%")
            with _m4:
                st.metric("IVCC", f"{_up_result.integral_vcc:.0f} x10^6 d/mL")

            if _up_result.dev_penalty_applied < 1.0:
                st.warning(
                    f"Developability penalty applied: q_p reduced to "
                    f"{_up_result.dev_penalty_applied:.0%} of maximum. "
                    f"High aggregation risk causes ER stress and lower titer."
                )

            # -- Dual-axis Plotly chart: VCD + Titer ---
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                _fig_up = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=("Viable Cell Density & Titer", "Glucose & Lactate"),
                )

                # VCD (left axis)
                _fig_up.add_trace(go.Scatter(
                    x=_up_result.time_days, y=_up_result.vcd,
                    name="VCD (10^6/mL)", line=dict(color="#3B82F6", width=2.5),
                ), row=1, col=1)

                # Titer on secondary y-axis via second trace
                _fig_up.add_trace(go.Scatter(
                    x=_up_result.time_days, y=_up_result.titer,
                    name="Titer (g/L)", line=dict(color="#EF4444", width=2.5, dash="dash"),
                    yaxis="y2",
                ), row=1, col=1)

                # Glucose
                _fig_up.add_trace(go.Scatter(
                    x=_up_result.time_days, y=_up_result.glucose,
                    name="Glucose (g/L)", line=dict(color="#10B981", width=2),
                ), row=2, col=1)

                # Lactate
                _fig_up.add_trace(go.Scatter(
                    x=_up_result.time_days, y=_up_result.lactate,
                    name="Lactate (g/L)", line=dict(color="#F59E0B", width=2),
                ), row=2, col=1)

                # Temperature shift line (>= 0 means shift is active)
                if _up_result.params.temp_shift_day >= 0:
                    for r in [1, 2]:
                        _fig_up.add_vline(
                            x=_up_result.params.temp_shift_day, row=r, col=1,
                            line_dash="dot", line_color="gray",
                            annotation_text="Temp Shift" if r == 1 else None,
                        )

                _apply_pharma_theme(_fig_up,
                    title_text="",
                    height=650,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    yaxis=dict(title="VCD (10^6 cells/mL)"),
                    yaxis2=dict(
                        title="Titer (g/L)", overlaying="y", side="right",
                        showgrid=False,
                    ),
                    xaxis2=dict(title="Culture Day"),
                    yaxis3=dict(title="Concentration (g/L)"),
                )
                st.plotly_chart(_fig_up, use_container_width=True)

            except Exception as plot_err:
                st.caption(f"(Chart unavailable: {plot_err})")

            st.success(
                f"Simulation complete: {_up_result.params.t_end_days:.0f}-day culture, "
                f"titer = {_up_result.final_titer:.2f} g/L"
            )

            # -- v7.3.2: Internal State Variables (μ, qP, qG, qL) ----------------
            if hasattr(_up_result, "mu") and _up_result.mu is not None and len(_up_result.mu) > 0:
                with st.expander("Internal State Variables (μ, q_P, q_G, q_L)", expanded=True):
                    import numpy as _np_sv
                    # Summary metrics
                    _sv_c1, _sv_c2, _sv_c3, _sv_c4, _sv_c5 = st.columns(5)
                    with _sv_c1:
                        _mu_max_obs = float(_np_sv.max(_up_result.mu))
                        st.metric("μ_max (1/h)", f"{_mu_max_obs:.4f}",
                                  help="Peak specific growth rate observed during culture")
                    with _sv_c2:
                        _qp_max_obs = float(_np_sv.max(_up_result.q_p))
                        st.metric("q_P max (pg/cell/d)", f"{_qp_max_obs:.1f}",
                                  help="Peak specific productivity")
                    with _sv_c3:
                        _qg_max_obs = float(_np_sv.max(_up_result.q_g))
                        st.metric("q_G max", f"{_qg_max_obs:.3f}",
                                  help="Peak specific glucose uptake rate")
                    with _sv_c4:
                        _lact_peak = float(_np_sv.max(_up_result.lactate))
                        st.metric("Lactate peak (g/L)", f"{_lact_peak:.2f}")
                    with _sv_c5:
                        _mud_harv = float(_up_result.mu_d[-1])
                        st.metric("μ_d harvest (1/h)", f"{_mud_harv:.4f}",
                                  help="Death rate at harvest")

                    # Dual-panel plots: Growth Kinetics + Metabolic Rates
                    try:
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots

                        _fig_sv = make_subplots(
                            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                            subplot_titles=(
                                "Growth & Death Kinetics",
                                "Specific Productivity & Substrate Uptake",
                            ),
                        )

                        # Row 1: μ and μ_d
                        _fig_sv.add_trace(go.Scatter(
                            x=_up_result.time_days, y=_up_result.mu * 24.0,
                            name="μ (1/day)", line=dict(color="#3B82F6", width=2),
                        ), row=1, col=1)
                        _fig_sv.add_trace(go.Scatter(
                            x=_up_result.time_days, y=_up_result.mu_d * 24.0,
                            name="μ_d (1/day)", line=dict(color="#EF4444", width=2, dash="dash"),
                        ), row=1, col=1)

                        # Row 2: q_P and q_G
                        _fig_sv.add_trace(go.Scatter(
                            x=_up_result.time_days, y=_up_result.q_p,
                            name="q_P (pg/cell/day)", line=dict(color="#8B5CF6", width=2),
                        ), row=2, col=1)
                        _fig_sv.add_trace(go.Scatter(
                            x=_up_result.time_days, y=_up_result.q_L,
                            name="q_L (specific lactate)", line=dict(color="#F59E0B", width=2, dash="dot"),
                            yaxis="y4",
                        ), row=2, col=1)

                        # Temp shift line (>= 0 means shift is active)
                        if _up_result.params.temp_shift_day >= 0:
                            for r in [1, 2]:
                                _fig_sv.add_vline(
                                    x=_up_result.params.temp_shift_day, row=r, col=1,
                                    line_dash="dot", line_color="gray",
                                    annotation_text="Temp Shift" if r == 1 else None,
                                )

                        _apply_pharma_theme(_fig_sv,
                            title_text="",
                            height=550,
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            yaxis=dict(title="Rate (1/day)"),
                            xaxis2=dict(title="Culture Day"),
                            yaxis3=dict(title="q_P (pg/cell/day)"),
                        )
                        st.plotly_chart(_fig_sv, use_container_width=True)
                    except Exception:
                        pass

            # -- v7.3.2: Upstream Sanity Check + Result Logging --
            # Dedup guard: only log once per simulation run
            _up_log_hash = f"{_up_result.final_titer:.6f}_{_up_result.peak_vcd:.4f}_{_up_result.viability_at_harvest:.2f}"
            _up_already_logged = st.session_state.get("_up_logged_hash")
            if _up_already_logged != _up_log_hash:
                st.session_state["_up_logged_hash"] = _up_log_hash
                try:
                    from src.result_logger import ResultLogger
                    from src.upstream_twin import result_to_dict as _up_to_dict
                    _up_mol_name = "unknown"
                    _up_log_intent = st.session_state.get("last_intent")
                    if _up_log_intent and isinstance(_up_log_intent, dict):
                        _up_mol_name = _up_log_intent.get("name",
                                        _up_log_intent.get("stoichiometry", "unknown"))
                    _up_log_data = _up_to_dict(_up_result)
                    _up_log_data["molecule_name"] = _up_mol_name
                    _up_log_data["module"] = "upstream_simulation"
                    ResultLogger().log_run(_up_log_data)
                except Exception:
                    pass

            # Sanity check on upstream outputs (always displayed)
            try:
                from src.sanity_check import check_upstream
                _up_sanity_w = check_upstream(
                    titer=_up_result.final_titer,
                    peak_vcd=_up_result.peak_vcd,
                    viability=_up_result.viability_at_harvest,
                )
                if _up_sanity_w:
                    st.markdown("##### Scientific Sanity Check")
                    for _w in _up_sanity_w:
                        if _w.severity == "critical":
                            st.error(_w.message)
                        else:
                            st.warning(_w.message)
            except Exception:
                pass

            # -- Upstream Process DoE: Parameter Sensitivity Sweep -------------------
            with st.expander("Upstream Process DoE — Parameter Sensitivity", expanded=True):
                st.caption(
                    "One-at-a-time parameter sweeps around the current operating point. "
                    "Shows how seed density and temperature shift day affect titer and peak VCD."
                )
                try:
                    import numpy as _np_doe_up
                    import plotly.graph_objects as _go_doe_up
                    from plotly.subplots import make_subplots as _ms_doe_up
                    from src.upstream_twin import run_upstream_simulation as _doe_up_sim

                    _base_seed = _up_result.params.X0
                    _base_tshift = _up_result.params.temp_shift_day
                    _base_dur = _up_result.params.t_end_days

                    # Sweep 1: Seed density (0.2 – 1.2) — industry range
                    # Standard CHO fed-batch: 0.3-0.8 x10^6; high-seed up to ~1.0
                    _seed_vals = _np_doe_up.linspace(0.2, 1.2, 11)
                    _seed_titers = []
                    _seed_vcds = []
                    for _sv in _seed_vals:
                        _sr = _doe_up_sim(
                            seed_density=float(_sv), temp_shift_day=_base_tshift,
                            dev_score=_up_dev_score, agg_risk=_up_agg_risk,
                            culture_days=_base_dur,
                            hydrophobicity=_last_intent.get("gravy") if isinstance(_last_intent, dict) else None,
                            sequence=_last_intent.get("sequence") if isinstance(_last_intent, dict) else None,
                            molecule_class=_last_intent.get("molecule_class") if isinstance(_last_intent, dict) else None,
                        )
                        _seed_titers.append(_sr.final_titer)
                        _seed_vcds.append(_sr.peak_vcd)

                    # Sweep 2: Temp shift day (2 – 8, plus "No Shift")
                    # Industry standard: day 3-5 for mAb production.
                    # Includes "No Shift" (temp_shift_day=-1) as reference.
                    _tshift_day_vals = [2, 3, 4, 5, 6, 7, 8]
                    _tshift_labels = [str(d) for d in _tshift_day_vals] + ["No Shift"]
                    _tshift_titers = []
                    _tshift_vcds = []
                    for _tv in _tshift_day_vals:
                        _tr = _doe_up_sim(
                            seed_density=_base_seed, temp_shift_day=float(_tv),
                            dev_score=_up_dev_score, agg_risk=_up_agg_risk,
                            culture_days=_base_dur,
                            hydrophobicity=_last_intent.get("gravy") if isinstance(_last_intent, dict) else None,
                            sequence=_last_intent.get("sequence") if isinstance(_last_intent, dict) else None,
                            molecule_class=_last_intent.get("molecule_class") if isinstance(_last_intent, dict) else None,
                        )
                        _tshift_titers.append(_tr.final_titer)
                        _tshift_vcds.append(_tr.peak_vcd)
                    # Add "No Shift" reference (temp_shift_day=-1)
                    _tr_noshift = _doe_up_sim(
                        seed_density=_base_seed, temp_shift_day=-1.0,
                        dev_score=_up_dev_score, agg_risk=_up_agg_risk,
                        culture_days=_base_dur,
                        hydrophobicity=_last_intent.get("gravy") if isinstance(_last_intent, dict) else None,
                        sequence=_last_intent.get("sequence") if isinstance(_last_intent, dict) else None,
                        molecule_class=_last_intent.get("molecule_class") if isinstance(_last_intent, dict) else None,
                    )
                    _tshift_titers.append(_tr_noshift.final_titer)
                    _tshift_vcds.append(_tr_noshift.peak_vcd)
                    _tshift_x_vals = list(range(len(_tshift_labels)))

                    # 2-panel chart
                    _fig_doe_up = _ms_doe_up(
                        rows=1, cols=2,
                        subplot_titles=(
                            "Seed Density Sensitivity",
                            "Temperature Shift Day Sensitivity",
                        ),
                    )
                    # Panel 1: seed density
                    _fig_doe_up.add_trace(_go_doe_up.Scatter(
                        x=_seed_vals, y=_seed_titers,
                        name="Titer (g/L)", line=dict(color="#EF4444", width=2.5),
                    ), row=1, col=1)
                    _fig_doe_up.add_trace(_go_doe_up.Scatter(
                        x=_seed_vals, y=_seed_vcds,
                        name="Peak VCD", line=dict(color="#3B82F6", width=2.5, dash="dash"),
                        yaxis="y2",
                    ), row=1, col=1)
                    # Mark current operating point
                    _fig_doe_up.add_vline(
                        x=_base_seed, row=1, col=1,
                        line_dash="dot", line_color="#10B981",
                        annotation_text="Current",
                    )

                    # Panel 2: temp shift day (categorical x-axis)
                    _fig_doe_up.add_trace(_go_doe_up.Scatter(
                        x=_tshift_labels, y=_tshift_titers,
                        name="Titer (g/L)", line=dict(color="#EF4444", width=2.5),
                        showlegend=False,
                    ), row=1, col=2)
                    _fig_doe_up.add_trace(_go_doe_up.Scatter(
                        x=_tshift_labels, y=_tshift_vcds,
                        name="Peak VCD", line=dict(color="#3B82F6", width=2.5, dash="dash"),
                        showlegend=False,
                        yaxis="y4",
                    ), row=1, col=2)
                    # Mark current operating point on the categorical axis.
                    # NOTE: Plotly's add_vline(annotation_text=...) crashes on
                    # category axes (tries to sum strings).  Use add_shape +
                    # add_annotation instead.
                    if _base_tshift >= 0:
                        _current_label = str(int(_base_tshift)) if _base_tshift == int(_base_tshift) else f"{_base_tshift:.1f}"
                        if _current_label in _tshift_labels:
                            _fig_doe_up.add_shape(
                                type="line",
                                x0=_current_label, x1=_current_label,
                                y0=0, y1=1, yref="y3 domain",
                                xref="x2",
                                line=dict(dash="dot", color="#10B981", width=1.5),
                            )
                            _fig_doe_up.add_annotation(
                                x=_current_label, y=1.02, yref="y3 domain",
                                xref="x2",
                                text="Current", showarrow=False,
                                font=dict(size=11, color="#10B981"),
                            )

                    _apply_pharma_theme(_fig_doe_up,
                        title_text="",
                        height=400,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.08),
                        xaxis=dict(title="Seed Density (10^6 cells/mL)"),
                        yaxis=dict(title="Titer (g/L)"),
                        yaxis2=dict(title="Peak VCD", overlaying="y", side="right", showgrid=False),
                        xaxis2=dict(title="Temp Shift Day", type="category"),
                        yaxis3=dict(title="Titer (g/L)"),
                        yaxis4=dict(title="Peak VCD", overlaying="y3", side="right", showgrid=False),
                    )
                    st.plotly_chart(_fig_doe_up, use_container_width=True)

                    # Summary table — find optimal within industry-standard ranges
                    _opt_seed_idx = int(_np_doe_up.argmax(_seed_titers))
                    # For temp shift, only consider day-based options (exclude "No Shift")
                    _opt_tshift_idx = int(_np_doe_up.argmax(_tshift_titers[:len(_tshift_day_vals)]))
                    _opt_tshift_label = _tshift_labels[_opt_tshift_idx]
                    st.markdown(
                        f"**Optimal seed density**: {_seed_vals[_opt_seed_idx]:.2f} x 10^6 cells/mL "
                        f"-> titer = {_seed_titers[_opt_seed_idx]:.2f} g/L  \n"
                        f"**Optimal temp shift day**: Day {_opt_tshift_label} "
                        f"-> titer = {_tshift_titers[_opt_tshift_idx]:.2f} g/L  \n"
                        f"**No shift reference**: titer = {_tshift_titers[-1]:.2f} g/L"
                    )
                except Exception as _doe_up_err:
                    st.caption(f"(Parameter sensitivity sweep unavailable: {_doe_up_err})")

            # -- Glycation & SVA Risk Assessment (C8: Qualitative diagnostics) ------
            _up_seq = _last_intent.get("sequence") if isinstance(_last_intent, dict) else None
            if _up_seq and len(_up_seq) > 50:
                st.markdown("---")
                st.markdown("### Post-Translational Modification Risk")
                try:
                    from src.upstream_twin import estimate_glycation_risk, estimate_sva_frequency
                    import numpy as _np_up

                    _avg_glucose = float(_np_up.mean(_up_result.glucose))
                    _glyc = estimate_glycation_risk(
                        _up_seq,
                        glucose_conc_gL=_avg_glucose,
                        culture_days=_up_result.params.t_end_days,
                    )
                    _sva = estimate_sva_frequency(
                        _up_seq,
                        peak_vcd=_up_result.peak_vcd,
                        culture_days=_up_result.params.t_end_days,
                        glucose_conc_gL=_avg_glucose,
                    )

                    # Qualitative text-based warnings (no quantitative metrics)
                    _ptm1, _ptm2 = st.columns(2)
                    with _ptm1:
                        _glyc_color = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}.get(
                            _glyc["risk_level"], "#64748B")
                        st.markdown(
                            f'<div style="background:{_glyc_color}15;border-left:4px solid {_glyc_color};'
                            f'padding:12px;border-radius:6px;">'
                            f'<b>Glycation Risk: {_glyc["risk_level"]}</b><br>'
                            f'<span style="font-size:0.9rem;">{_glyc["recommendation"]}</span></div>',
                            unsafe_allow_html=True,
                        )

                    with _ptm2:
                        _sva_color = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}.get(
                            _sva["risk_level"], "#64748B")
                        st.markdown(
                            f'<div style="background:{_sva_color}15;border-left:4px solid {_sva_color};'
                            f'padding:12px;border-radius:6px;">'
                            f'<b>SVA Risk: {_sva["risk_level"]}</b><br>'
                            f'<span style="font-size:0.9rem;">{_sva["recommendation"]}</span></div>',
                            unsafe_allow_html=True,
                        )

                except Exception as _ptm_err:
                    st.caption(f"PTM risk estimation unavailable: {_ptm_err}")

            # -- Tech Transfer / Scale-Up Section (M23 — NEW) ----------------------
            st.markdown("---")
            st.markdown("### Tech Transfer: 2L → Manufacturing Scale")
            st.caption(
                "Simulate scale-up from bench (2L) to manufacturing (2000L) bioreactor. "
                "Compare constant P/V vs. constant kLa strategies and assess CHO cell shear risk."
            )

            _su_c1, _su_c2, _su_c3 = st.columns(3)
            with _su_c1:
                _su_large_vol = st.number_input(
                    "Manufacturing Scale (L)",
                    min_value=50.0, max_value=20000.0, value=2000.0, step=500.0,
                    key="scaleup_large_vol",
                    help="Target manufacturing bioreactor working volume.",
                )
            with _su_c2:
                _su_bench_rpm = st.number_input(
                    "Bench RPM",
                    min_value=50.0, max_value=500.0, value=200.0, step=10.0,
                    key="scaleup_bench_rpm",
                    help="Agitation speed at bench scale (RPM).",
                )
            with _su_c3:
                _su_vvm = st.number_input(
                    "VVM (gas flow rate)",
                    min_value=0.005, max_value=0.10, value=0.02, step=0.005,
                    key="scaleup_vvm", format="%.3f",
                    help="Volume of gas per Volume of liquid per Minute.",
                )

            if st.button("Run Scale-Up Simulation", key="btn_scaleup_run", type="primary"):
                with st.spinner("Computing scale-up parameters: volumetric mass transfer, impeller tip speed, mixing time..."):
                    try:
                        from src.scaleup_twin import run_scaleup_simulation

                        _su_result = run_scaleup_simulation(
                            small_volume_L=2.0,
                            large_volume_L=_su_large_vol,
                            small_rpm=_su_bench_rpm,
                            bench_titer=_up_result.final_titer,
                            vvm=_su_vvm,
                        )
                        st.session_state["scaleup_result"] = _su_result
                    except Exception as su_err:
                        st.error(f"Scale-up simulation failed: {su_err}")
                        import traceback
                        st.caption(traceback.format_exc())

            _su_result = st.session_state.get("scaleup_result")
            if _su_result is not None:
                # Warnings first
                for _sw in _su_result.warnings:
                    if "CRITICAL" in _sw:
                        st.error(_sw)
                    else:
                        st.warning(_sw)

                # Comparison table
                st.markdown("#### Scale-Up Strategy Comparison")

                _suc1, _suc2, _suc3 = st.columns(3)
                with _suc1:
                    st.markdown("**Bench (2L)**")
                    st.metric("RPM", f"{_su_result.small_rpm:.0f}")
                    st.metric("Tip Speed", f"{_su_result.small_tip_speed:.3f} m/s")
                    st.metric("P/V", f"{_su_result.small_pv:.1f} W/m³")
                    st.metric("kLa", f"{_su_result.small_kla:.1f} 1/h")
                    st.metric("Mixing Time", f"{_su_result.mixing_time_small:.0f}s")
                with _suc2:
                    st.markdown(f"**Constant P/V ({_su_large_vol:.0f}L)**")
                    st.metric("RPM", f"{_su_result.pv_rpm:.1f}")
                    _pv_color = {"Safe": "normal", "Warning": "off", "Danger": "inverse"}.get(
                        _su_result.pv_shear_status, "normal")
                    st.metric("Tip Speed", f"{_su_result.pv_tip_speed:.3f} m/s",
                              delta=_su_result.pv_shear_status, delta_color=_pv_color)
                    st.metric("P/V", f"{_su_result.small_pv:.1f} W/m³ (matched)")
                    st.metric("kLa", f"{_su_result.pv_kla:.1f} 1/h")
                    st.metric("Mixing Time", f"{_su_result.mixing_time_large:.0f}s")
                with _suc3:
                    st.markdown(f"**Constant kLa ({_su_large_vol:.0f}L)**")
                    st.metric("RPM", f"{_su_result.kla_rpm:.1f}")
                    _kla_color = {"Safe": "normal", "Warning": "off", "Danger": "inverse"}.get(
                        _su_result.kla_shear_status, "normal")
                    st.metric("Tip Speed", f"{_su_result.kla_tip_speed:.3f} m/s",
                              delta=_su_result.kla_shear_status, delta_color=_kla_color)
                    st.metric("P/V", f"{_su_result.kla_pv:.1f} W/m³")
                    st.metric("kLa", f"{_su_result.small_kla:.1f} 1/h (matched)")
                    st.metric("Mixing Time", f"{_su_result.mixing_time_large:.0f}s")

                # Recommendation banner
                st.markdown(
                    f'<div style="background:#3B82F615;border-left:4px solid #3B82F6;'
                    f'padding:16px;border-radius:8px;margin:12px 0;">'
                    f'<span style="font-size:1.1rem;font-weight:700;color:#3B82F6;">'
                    f'Recommended: {_su_result.recommended_strategy}</span><br/>'
                    f'<span style="font-size:0.95rem;">'
                    f'Predicted titer at {_su_large_vol:.0f}L: '
                    f'<strong>{_su_result.predicted_titer_large:.2f} g/L</strong> '
                    f'({_su_result.titer_scaling_factor:.0%} of bench {_su_result.small_titer:.2f} g/L)'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

                # Geometry info
                with st.expander("Bioreactor Geometry Details", expanded=True):
                    st.markdown(
                        f"**Bench ({_su_result.small_scale.volume_L:.0f}L):** "
                        f"Tank Ø {_su_result.small_scale.tank_diameter_m*100:.1f} cm, "
                        f"Impeller Ø {_su_result.small_scale.impeller_diameter_m*100:.1f} cm, "
                        f"Height {_su_result.small_scale.liquid_height_m*100:.1f} cm, "
                        f"{_su_result.small_scale.n_impellers} impeller(s)\n\n"
                        f"**Manufacturing ({_su_result.large_scale.volume_L:.0f}L):** "
                        f"Tank Ø {_su_result.large_scale.tank_diameter_m*100:.1f} cm, "
                        f"Impeller Ø {_su_result.large_scale.impeller_diameter_m*100:.1f} cm, "
                        f"Height {_su_result.large_scale.liquid_height_m*100:.1f} cm, "
                        f"{_su_result.large_scale.n_impellers} impeller(s)"
                    )


    # ===========================================================================
    #  Downstream Purification Tab (inside Process Development)
    # ===========================================================================
    with _proc_tab_down:
        st.markdown("### Downstream Purification")
        st.caption(
            "In-Silico DoE optimization for Design Space exploration. "
            "Charge variant profiling is available in the Analytical & Mass Spec tab."
        )

        _active_ws = ws_store.get_active()

        # -- Upstream context: read titer from Upstream results (data flow connection) --
        _upstream_dict = st.session_state.get("upstream_result_dict") or {}
        _upstream_titer = _upstream_dict.get("final_titer")
        if _upstream_titer:
            st.markdown(
                f'<div style="background:#FFFBEB;border-left:4px solid #F59E0B;'
                f'padding:8px 12px;margin:0 0 12px 0;border-radius:6px;font-size:0.85rem;">'
                f'Upstream titer: <b>{_upstream_titer:.2f} g/L</b> '
                f'(from Cell Culture simulation)</div>',
                unsafe_allow_html=True,
            )

        # -- M19: GoSilico DoE Optimization (PROMINENTLY DISPLAYED) ----
        st.markdown("### In-Silico DoE Purification Optimization")
        st.caption(
            "GoSilico-style grid search over elution pH and gradient steepness. "
            "Identifies the optimal Design Space sweet spot for maximum Resolution and Yield."
        )

        _doe_intent = st.session_state.get("last_intent")
        _doe_pI = None
        if _doe_intent and isinstance(_doe_intent, dict):
            _doe_pI = _doe_intent.get("pI")

        if _doe_pI is None and _active_ws and _active_ws.get("intent"):
            _doe_pI = _active_ws["intent"].get("pI")

        if _doe_pI is None:
            _doe_pI = st.number_input(
                "Protein pI (required for DoE)", min_value=4.0, max_value=12.0, value=8.5, step=0.1,
                key="doe_manual_pI",
                help="If no molecule is loaded, enter pI manually to run DoE.",
            )

        # Compute pI-adaptive pH range so defaults are molecule-specific
        from src.purification_optimizer import compute_adaptive_ph_range as _comp_adaptive_ph
        _adaptive_ph = _comp_adaptive_ph(_doe_pI, mode="cex") if _doe_pI else (5.5, 6.8)

        _doe_c1, _doe_c2 = st.columns(2)
        with _doe_c1:
            _doe_ph_min = st.number_input(
                "pH range — min", value=float(_adaptive_ph[0]), min_value=3.5, max_value=9.0, step=0.1,
                key="doe_ph_min_ds",
                help=f"Lower bound of elution buffer pH. pI-adaptive default: {_adaptive_ph[0]:.1f}.",
            )
            _doe_ph_max = st.number_input(
                "pH range — max", value=float(_adaptive_ph[1]), min_value=3.5, max_value=9.0, step=0.1,
                key="doe_ph_max_ds",
                help=f"Upper bound of elution buffer pH. pI-adaptive default: {_adaptive_ph[1]:.1f}. "
                     f"CEX requires pH < pI ({_doe_pI:.1f}).",
            )
        with _doe_c2:
            _doe_grad_min = st.number_input(
                "Gradient — min (mM/min)", value=5.0, min_value=1.0, max_value=50.0, step=1.0,
                key="doe_grad_min_ds",
                help="Lower bound of salt gradient steepness. IgG1 optimal: 10-20 mM/min.",
            )
            _doe_grad_max = st.number_input(
                "Gradient — max (mM/min)", value=25.0, min_value=1.0, max_value=50.0, step=1.0,
                key="doe_grad_max_ds",
                help="Upper bound of salt gradient steepness. IgG1 optimal: 10-20 mM/min.",
            )

        # ── v2.0 Advanced DoE Settings ────────────────────────────────
        with st.expander("Advanced DoE v2.0 Settings", expanded=True):
            st.caption(
                "Extended 4D Design Space: pH × Gradient × Salt Elute × Column Loading. "
                "Includes HCP clearance model, gradient shape selection, and hard purity/resolution constraints."
            )
            _adv_c1, _adv_c2 = st.columns(2)
            with _adv_c1:
                _doe_salt_min = st.number_input(
                    "Salt elute — min (mM)", value=300.0, min_value=100.0, max_value=800.0, step=25.0,
                    key="doe_salt_min_ds",
                    help="Lower bound of NaCl elution concentration for pool collection.",
                )
                _doe_salt_max = st.number_input(
                    "Salt elute — max (mM)", value=600.0, min_value=100.0, max_value=800.0, step=25.0,
                    key="doe_salt_max_ds",
                )
                _doe_load_min = st.number_input(
                    "Column load — min (mg/mL)", value=20.0, min_value=5.0, max_value=80.0, step=5.0,
                    key="doe_load_min_ds",
                    help="Lower bound of column loading capacity for DoE.",
                )
                _doe_load_max = st.number_input(
                    "Column load — max (mg/mL)", value=50.0, min_value=5.0, max_value=80.0, step=5.0,
                    key="doe_load_max_ds",
                )
            with _adv_c2:
                _doe_gradient_shape = st.selectbox(
                    "Gradient shape",
                    options=["linear", "step", "concave", "convex"],
                    index=0, key="doe_grad_shape_ds",
                    help="Salt gradient profile. Linear is standard; concave provides slower initial rise for better early separation.",
                )
                _doe_hcp_feed = st.number_input(
                    "HCP feed (ppm)", value=500.0, min_value=50.0, max_value=5000.0, step=50.0,
                    key="doe_hcp_feed_ds",
                    help="Host Cell Protein level in feed material (post-Protein A). Typical: 300-1000 ppm.",
                )
                _doe_purity_floor = st.number_input(
                    "Purity floor (%)", value=95.0, min_value=80.0, max_value=99.9, step=0.5,
                    key="doe_purity_floor_ds",
                    help="Hard constraint: conditions below this purity are marked infeasible.",
                )
                _doe_rs_floor = st.number_input(
                    "Rs floor", value=1.5, min_value=0.5, max_value=3.0, step=0.1,
                    key="doe_rs_floor_ds",
                    help="Hard constraint: conditions below this resolution are marked infeasible.",
                )
            _doe_use_pi_adaptive = st.checkbox(
                "Override: use pI-adaptive pH range", value=False, key="doe_pi_adaptive_ds",
                help="When checked, ignores the manual pH min/max above and auto-computes "
                     f"the range from pI ({_doe_pI:.2f}). The manual inputs already default "
                     "to the pI-adaptive range, so only check this if you changed the "
                     "inputs and want to reset.",
            )
            if _doe_use_pi_adaptive and _doe_pI is not None:
                st.caption(
                    f"Override active: using pI-adaptive range "
                    f"[{_adaptive_ph[0]:.1f}, {_adaptive_ph[1]:.1f}] "
                    f"(computed from pI={_doe_pI:.2f}). Manual pH inputs above are ignored."
                )

        # v7.3.4: pI-aware pH soft constraint for CEX binding.
        # Use pI - 0.5 ceiling (consistent with compute_adaptive_ph_range).
        # Show warning but allow the user's choice when within safe range.
        _doe_ph_max_effective = _doe_ph_max
        if _doe_pI is not None and _doe_pI > 0 and not _doe_use_pi_adaptive:
            _pI_ceiling = round(_doe_pI - 0.5, 1)
            if _doe_ph_max > _pI_ceiling:
                _doe_ph_max_effective = max(_doe_ph_min + 0.1, _pI_ceiling)
                st.warning(
                    f"**pH constraint applied:** pH max clamped from {_doe_ph_max:.1f} "
                    f"to **{_doe_ph_max_effective:.1f}** (pI={_doe_pI:.2f} - 0.5 = {_pI_ceiling:.1f}). "
                    f"CEX requires pH < pI for cation binding."
                )

        if st.button(
            "Run In-Silico DoE Optimization",
            type="primary", key="btn_doe_run_ds",
            help="Execute grid search across pH x Gradient conditions using Yamamoto SMA theory",
        ):
            with st.spinner("Running GoSilico-style Design of Experiments: pH \u00d7 gradient grid search for optimal resolution..."):
                try:
                    from src.purification_optimizer import (
                        run_doe_optimization, doe_summary, doe_to_dict,
                    )
                    import plotly.graph_objects as go

                    _doe_mw = 150.0
                    _doe_hydro = 0.35
                    if _doe_intent and isinstance(_doe_intent, dict):
                        _doe_mw = _doe_intent.get("mw", 150.0)
                        _doe_hydro = _doe_intent.get("hydrophobicity", 0.35)

                    # When pI-adaptive override is ON, pass ph_range=None
                    # to let the optimizer auto-compute from pI.
                    # When OFF (default), use the user's manual pH inputs.
                    _doe_ph_range_arg = None if _doe_use_pi_adaptive else (_doe_ph_min, _doe_ph_max_effective)
                    _doe_kwargs = dict(
                        pI=_doe_pI,
                        mw=_doe_mw,
                        hydrophobicity=_doe_hydro,
                        ph_range=_doe_ph_range_arg,
                        gradient_range=(_doe_grad_min, _doe_grad_max),
                        salt_elute_range=(_doe_salt_min, _doe_salt_max),
                        load_range=(_doe_load_min, _doe_load_max),
                        ph_steps=16,
                        gradient_steps=12,
                        salt_steps=4,
                        load_steps=4,
                        purity_floor=_doe_purity_floor,
                        rs_floor=_doe_rs_floor,
                        gradient_shape=_doe_gradient_shape,
                        hcp_feed_ppm=_doe_hcp_feed,
                    )
                    _doe_result = run_doe_optimization(**_doe_kwargs)
                    st.session_state["doe_result_ds"] = _doe_result
                    _doe_dict = doe_to_dict(_doe_result)
                    st.session_state["doe_result_dict"] = _doe_dict
                    # Propagate to workspace analysis_cache so report can pick it up
                    _doe_ws = ws_store.get_active() if ws_store else None
                    if _doe_ws and isinstance(_doe_ws.get("analysis_cache"), dict):
                        _doe_ws["analysis_cache"]["formulation_result"] = _doe_dict
                    elif _doe_ws:
                        ws_store.update_active_field("formulation_result", _doe_dict)
                except Exception as doe_err:
                    st.error(f"DoE optimization failed: {doe_err}")
                    import traceback
                    st.caption(traceback.format_exc())

        # -- Display DoE results ------------------------------------------------
        _doe_result = st.session_state.get("doe_result_ds")
        if _doe_result is not None:
            st.markdown("#### Optimal Sweet Spot")
            _ds1, _ds2, _ds3, _ds4 = st.columns(4)
            with _ds1:
                st.metric("Elution pH", f"{_doe_result.optimal_ph:.2f}")
            with _ds2:
                st.metric("Gradient", f"{_doe_result.optimal_gradient:.1f} mM/min")
            with _ds3:
                _rs_label = "Baseline" if _doe_result.optimal.resolution_min >= 1.5 else (
                    "Partial" if _doe_result.optimal.resolution_min >= 0.8 else "Overlap"
                )
                st.metric("Resolution (Rs)", f"{_doe_result.optimal.resolution_min:.3f}", _rs_label)
            with _ds4:
                st.metric("Main-Peak Yield", f"{_doe_result.optimal.yield_main * 100:.1f}%")

            # Show combined process yield if upstream titer available
            if _upstream_titer:
                _ds_yield = _doe_result.optimal.yield_main
                _est_purified = _upstream_titer * _ds_yield
                st.markdown(
                    f'<div style="background:#F0FFF4;border-left:4px solid #10B981;'
                    f'padding:8px 12px;margin:8px 0;border-radius:6px;font-size:0.85rem;">'
                    f'Estimated purified product: <b>{_est_purified:.2f} g/L</b> '
                    f'(titer {_upstream_titer:.2f} x yield {_ds_yield*100:.1f}%)</div>',
                    unsafe_allow_html=True,
                )

            # v2.0 extended metrics row
            _dv1, _dv2, _dv3, _dv4 = st.columns(4)
            with _dv1:
                _purity_val = getattr(_doe_result.optimal, "pool_purity_pct", None)
                st.metric("Pool Purity", f"{_purity_val:.1f}%" if _purity_val else "N/A")
            with _dv2:
                _hcp_val = getattr(_doe_result.optimal, "hcp_ppm", None)
                st.metric("HCP", f"{_hcp_val:.0f} ppm" if _hcp_val else "N/A")
            with _dv3:
                _salt_val = getattr(_doe_result.optimal, "salt_elute", None)
                st.metric("Salt Elute", f"{_salt_val:.0f} mM" if _salt_val else "N/A")
            with _dv4:
                _load_val = getattr(_doe_result.optimal, "c_load_mg_ml", None)
                st.metric("Column Load", f"{_load_val:.0f} mg/mL" if _load_val else "N/A")

            _dp1, _dp2, _dp3, _dp4 = st.columns(4)
            with _dp1:
                st.metric("RT Acidic", f"{_doe_result.optimal.rt_acidic:.2f} min")
            with _dp2:
                st.metric("RT Main", f"{_doe_result.optimal.rt_main:.2f} min")
            with _dp3:
                st.metric("RT Basic", f"{_doe_result.optimal.rt_basic:.2f} min")
            with _dp4:
                _shape_val = getattr(_doe_result.optimal, "gradient_shape", "linear")
                _feasible = getattr(_doe_result.optimal, "feasible", True)
                st.metric("Gradient Shape", _shape_val.capitalize(),
                          delta="Feasible" if _feasible else "Infeasible",
                          delta_color="normal" if _feasible else "inverse")

            # -- Contour Plots ---
            import plotly.graph_objects as go
            import numpy as _np
            _ph_arr = _np.array(_doe_result.ph_values)
            _grad_arr = _np.array(_doe_result.gradient_values)

            _doe_col1, _doe_col2 = st.columns(2)
            with _doe_col1:
                _fig_contour = go.Figure()
                _fig_contour.add_trace(go.Contour(
                    z=_doe_result.resolution_matrix,
                    x=_grad_arr, y=_ph_arr,
                    colorscale="Viridis",
                    colorbar=dict(title="Rs (min)"),
                    contours=dict(showlabels=True, labelfont=dict(size=10)),
                    name="Resolution",
                ))
                _fig_contour.add_trace(go.Scatter(
                    x=[_doe_result.optimal_gradient], y=[_doe_result.optimal_ph],
                    mode="markers+text",
                    marker=dict(size=14, color=PLOTLY_COLORS[2], symbol="star"),
                    text=["OPTIMAL"], textposition="top center",
                    textfont=dict(color=PLOTLY_COLORS[2], size=12),
                    name="Sweet Spot",
                ))
                _apply_pharma_theme(_fig_contour,
                    title="Resolution vs Elution Conditions",
                    xaxis_title="Gradient (mM/min)",
                    yaxis_title="pH",
                    height=450,
                )
                st.plotly_chart(_fig_contour, use_container_width=True)

            with _doe_col2:
                _fig_yield = go.Figure()
                _fig_yield.add_trace(go.Contour(
                    z=_doe_result.yield_matrix,
                    x=_grad_arr, y=_ph_arr,
                    colorscale="YlOrRd",
                    colorbar=dict(title="Yield"),
                    contours=dict(showlabels=True, labelfont=dict(size=10)),
                    name="Yield",
                ))
                _fig_yield.add_trace(go.Scatter(
                    x=[_doe_result.optimal_gradient], y=[_doe_result.optimal_ph],
                    mode="markers+text",
                    marker=dict(size=14, color=PLOTLY_COLORS[0], symbol="star"),
                    text=["OPTIMAL"], textposition="top center",
                    textfont=dict(color=PLOTLY_COLORS[0], size=12),
                    name="Sweet Spot",
                ))
                _apply_pharma_theme(_fig_yield,
                    title="Yield vs Elution Conditions",
                    xaxis_title="Gradient (mM/min)",
                    yaxis_title="pH",
                    height=450,
                )
                st.plotly_chart(_fig_yield, use_container_width=True)

            # ── v2.0 Purity & HCP Contour Plots ──
            _has_purity = _doe_result.purity_matrix is not None
            _has_hcp = _doe_result.hcp_matrix is not None
            if _has_purity or _has_hcp:
                _doe_col3, _doe_col4 = st.columns(2)
                if _has_purity:
                    with _doe_col3:
                        _fig_purity = go.Figure()
                        _fig_purity.add_trace(go.Contour(
                            z=_doe_result.purity_matrix,
                            x=_grad_arr, y=_ph_arr,
                            colorscale="Greens",
                            colorbar=dict(title="Purity %"),
                            contours=dict(showlabels=True, labelfont=dict(size=10)),
                            name="Pool Purity",
                        ))
                        _fig_purity.add_trace(go.Scatter(
                            x=[_doe_result.optimal_gradient], y=[_doe_result.optimal_ph],
                            mode="markers", marker=dict(size=14, color=PLOTLY_COLORS[2], symbol="star"),
                            name="Sweet Spot",
                        ))
                        _apply_pharma_theme(_fig_purity,
                            title="Pool Purity vs Elution Conditions",
                            xaxis_title="Gradient (mM/min)", yaxis_title="pH", height=400,
                        )
                        st.plotly_chart(_fig_purity, use_container_width=True)
                if _has_hcp:
                    with _doe_col4:
                        _fig_hcp = go.Figure()
                        _fig_hcp.add_trace(go.Contour(
                            z=_doe_result.hcp_matrix,
                            x=_grad_arr, y=_ph_arr,
                            colorscale="Reds_r",
                            colorbar=dict(title="HCP (ppm)"),
                            contours=dict(showlabels=True, labelfont=dict(size=10)),
                            name="HCP Clearance",
                        ))
                        _fig_hcp.add_trace(go.Scatter(
                            x=[_doe_result.optimal_gradient], y=[_doe_result.optimal_ph],
                            mode="markers", marker=dict(size=14, color=PLOTLY_COLORS[0], symbol="star"),
                            name="Sweet Spot",
                        ))
                        _apply_pharma_theme(_fig_hcp,
                            title="HCP Clearance vs Elution Conditions",
                            xaxis_title="Gradient (mM/min)", yaxis_title="pH", height=400,
                        )
                        st.plotly_chart(_fig_hcp, use_container_width=True)

            # ── pI-adaptive pH source indicator ──
            _ph_src = getattr(_doe_result, "ph_range_source", "user")
            if _ph_src == "pI_adaptive":
                st.info(f"pH range auto-set from pI = {_doe_pI:.2f} (pI-adaptive mode)")

            from src.purification_optimizer import doe_summary
            st.code(doe_summary(_doe_result), language="text")

            st.success(
                f"DoE complete: {_doe_result.n_conditions} conditions evaluated "
                f"in {_doe_result.wall_time_s:.2f}s"
            )

            # v7.3.2: Log DoE results + mass balance + sanity check
            # Dedup guard: include molecule name + optimal values + wall_time
            # to prevent re-logging on Streamlit rerenders AND across molecules.
            _doe_dedup_mol = "unknown"
            _doe_dedup_intent = st.session_state.get("last_intent")
            if _doe_dedup_intent and isinstance(_doe_dedup_intent, dict):
                _doe_dedup_mol = _doe_dedup_intent.get("name",
                                _doe_dedup_intent.get("stoichiometry", "unknown"))
            _doe_run_hash = (
                f"{_doe_dedup_mol}_{_doe_result.optimal_ph:.4f}_"
                f"{_doe_result.optimal_gradient:.4f}_{_doe_result.wall_time_s:.4f}"
            )
            _already_logged = st.session_state.get("_doe_logged_hash")
            if _already_logged != _doe_run_hash:
                st.session_state["_doe_logged_hash"] = _doe_run_hash
                try:
                    from src.result_logger import ResultLogger
                    from src.purification_optimizer import compute_mass_balance

                    _opt = _doe_result.optimal
                    _mb = compute_mass_balance(_opt.resolution_acidic_main, _opt.resolution_main_basic)

                    # Derive molecule name from intent
                    _doe_mol_name = "unknown"
                    _doe_log_intent = st.session_state.get("last_intent")
                    if _doe_log_intent and isinstance(_doe_log_intent, dict):
                        _doe_mol_name = _doe_log_intent.get("name",
                                        _doe_log_intent.get("stoichiometry", "unknown"))

                    _logger = ResultLogger()
                    _logger.log_run({
                        "molecule_name": _doe_mol_name,
                        "module": "purification_optimizer",
                        "doe_optimal": {
                            "ph": _doe_result.optimal_ph,
                            "gradient": _doe_result.optimal_gradient,
                            "rs_min": _opt.resolution_min,
                            "yield": round(_opt.yield_main * 100.0, 1),
                            "rt_main": _opt.rt_main,
                        },
                        "mass_balance": _mb,
                    })
                except Exception as _doe_log_err:
                    log.warning("DoE result logging failed: %s", _doe_log_err)

            # Always display mass balance + sanity (even on re-render)
            try:
                from src.sanity_check import check_chromatography, check_purification
                from src.purification_optimizer import compute_mass_balance

                _opt = _doe_result.optimal
                _mb = compute_mass_balance(_opt.resolution_acidic_main, _opt.resolution_main_basic)

                st.markdown("##### Mass Balance Summary")
                _mb_cols = st.columns(4)
                _mb_cols[0].metric("Pool (Main)", f"{_mb['pool_main_pct']:.1f}%")
                _mb_cols[1].metric("Pool Purity", f"{_mb['pool_purity_pct']:.1f}%")
                _mb_cols[2].metric("Waste", f"{_mb['waste_total_pct']:.1f}%")
                _mb_cols[3].metric("Balance OK" if _mb['mass_balanced'] else "Balance FAIL",
                                  f"Δ={_mb['mass_balance_error']:.3f}%")

                # Sanity check
                _sc_w = check_chromatography(rs=_opt.resolution_min, rt_main=_opt.rt_main)
                _sc_w.extend(check_purification(yield_pct=_opt.yield_main * 100))
                if _sc_w:
                    for _w in _sc_w:
                        st.warning(f"Sanity: {_w.message}")
            except Exception:
                pass

        # ---- Multi-Step Platform Purification Process ----
        # C7: Removed — static plot with no dynamic data input. Purification
        # analysis is served by the DoE optimizer above.
        # (Code preserved in git history for future re-enablement.)

    # -- COGS Tab (moved from CMC Board page) --
    with _proc_tab_cogs:
        st.markdown("### Manufacturability & COGS")
        st.caption(
            "Calculate commercial manufacturing Cost of Goods Sold ($/gram purified API). "
            "Connects upstream titer and downstream yield with manufacturing cost inputs."
        )

        # Resolve molecule_class for COGS defaults
        _cogs_ws = ws_store.get_active() if ('ws_store' in dir() and ws_store) else None
        _cogs_mol_cls = (_cogs_ws.get("intent") or {}).get("molecule_class", "canonical_mab") if _cogs_ws else "canonical_mab"

        _up_dict_proc = st.session_state.get("upstream_result_dict") or {}
        _doe_dict_proc = st.session_state.get("doe_result_dict") or {}
        _auto_titer_proc = _up_dict_proc.get("final_titer", 5.0)
        _auto_yield_proc = _doe_dict_proc.get("optimal_yield", 0.70)

        # Molecule-class-aware defaults (aligned with cogs_twin.py calculate_cogs)
        _cogs_default_vol = 2000.0
        _cogs_default_resin = 50000.0
        _cogs_resin_label = "ProA Resin ($/batch)"
        if _cogs_mol_cls == "peptide":
            _cogs_default_vol = 200.0
            _cogs_default_resin = 15000.0
            _cogs_resin_label = "RP-HPLC Resin ($/batch)"
        elif _cogs_mol_cls == "single_domain":
            _cogs_default_vol = 500.0
            _cogs_default_resin = 20000.0
            _cogs_resin_label = "IEX Resin ($/batch)"

        _cogs_p1, _cogs_p2, _cogs_p3 = st.columns(3)
        with _cogs_p1:
            _cogs_titer_proc = st.number_input(
                "Upstream Titer (g/L)", min_value=0.1, max_value=30.0,
                value=float(round(_auto_titer_proc, 2)), step=0.5, key="cogs_titer_proc",
            )
        with _cogs_p2:
            _cogs_yield_proc = st.number_input(
                "DS Yield (%)", min_value=5.0, max_value=99.0,
                value=float(round(_auto_yield_proc * 100, 1)), step=5.0, key="cogs_yield_proc",
            )
        with _cogs_p3:
            _cogs_volume_proc = st.number_input(
                "Bioreactor (L)", min_value=50.0, max_value=20000.0,
                value=_cogs_default_vol, step=500.0, key="cogs_volume_proc",
            )
        _cp1, _cp2, _cp3 = st.columns(3)
        with _cp1:
            _cogs_media_proc = st.number_input("Media ($/L)", min_value=5.0, max_value=200.0, value=45.0, step=5.0, key="cogs_media_proc")
        with _cp2:
            _cogs_resin_proc = st.number_input(_cogs_resin_label, min_value=5000.0, max_value=200000.0, value=_cogs_default_resin, step=5000.0, key="cogs_resin_proc")
        with _cp3:
            _cogs_oh_proc = st.slider("Overhead Mult", min_value=0.5, max_value=3.0, value=1.0, step=0.1, key="cogs_oh_proc")

        if st.button("Calculate COGS", type="primary", key="btn_cogs_proc"):
            try:
                from src.cogs_twin import COGSInputs, calculate_cogs, cogs_to_dict
                # Use same overhead defaults as COGSInputs dataclass (aligned with auto-trigger):
                # qc=$65K, facility=$95K, labor=$55K, utilities=$8K (BioPhorum 2023)
                _inputs_proc = COGSInputs(
                    titer_g_per_L=_cogs_titer_proc,
                    downstream_yield=_cogs_yield_proc / 100.0,
                    bioreactor_volume_L=_cogs_volume_proc,
                    media_cost_per_L=_cogs_media_proc,
                    protein_a_resin_cost=_cogs_resin_proc,
                    qc_testing=65000.0 * _cogs_oh_proc,
                    facility_cost=95000.0 * _cogs_oh_proc,
                    labor_cost=55000.0 * _cogs_oh_proc,
                    utilities_cost=8000.0 * _cogs_oh_proc,
                )
                _cogs_r_proc = calculate_cogs(_inputs_proc, molecule_class=_cogs_mol_cls)
                st.session_state["cogs_result"] = _cogs_r_proc
                st.session_state["cogs_result_dict"] = cogs_to_dict(_cogs_r_proc)
            except Exception as cogs_err:
                st.error(f"COGS calculation failed: {cogs_err}")

        _cogs_r_proc = st.session_state.get("cogs_result")
        if _cogs_r_proc is not None:
            _ch1, _ch2, _ch3, _ch4 = st.columns(4)
            with _ch1:
                st.metric("COGS/g", f"${_cogs_r_proc.cogs_per_gram:.2f}", _cogs_r_proc.cost_rating)
            with _ch2:
                st.metric("Batch Cost", f"${_cogs_r_proc.total_batch_cost:,.0f}")
            with _ch3:
                st.metric("Batch Output", f"{_cogs_r_proc.batch_output_g:,.0f} g")
            with _ch4:
                st.metric("Output (kg)", f"{_cogs_r_proc.batch_output_kg:.1f} kg")
            if not _cogs_r_proc.is_commercial:
                st.error(f"COGS = ${_cogs_r_proc.cogs_per_gram:.2f}/g exceeds $150/g commercial viability threshold.")

            st.markdown("**COGS Formula:**")
            st.latex(r"\text{COGS}_{/g} = \frac{C_{\text{media}} \cdot V + C_{\text{resin}} + C_{\text{QC}} + C_{\text{facility}} + C_{\text{labor}} + C_{\text{utilities}}}{\text{Titer} \times V \times Y_{\text{DS}}}")
            st.caption("Where V = bioreactor volume (L), Y_DS = downstream yield")


# [REMOVED v32.1] Dead page code removed. See git history.


# ===========================================================================
#  5F. Preclinical & Formulation Page
# ===========================================================================
elif active_page == "Preclinical & Clinical":
    st.markdown("## Preclinical & Clinical")
    st.caption(
        "Formulation optimization, PK prediction, stability projection, "
        "and immunogenicity / ADA risk assessment."
    )

    # -- Hard gate: require molecule --
    enforce_molecule_state(ws_store)

    # -- Formulation & Buffer Conditions (top of page for real-time feedback) --
    st.markdown("### Formulation & Buffer Conditions")
    st.caption(
        "Adjust buffer pH, buffer type, and excipients to see real-time impact "
        "on aggregation risk, thermal stability, and viscosity below."
    )
    _form_c1, _form_c2 = st.columns(2)
    with _form_c1:
        formulation_ph = st.slider(
            "Buffer pH",
            min_value=4.5, max_value=8.0, value=st.session_state.formulation_buffer_ph,
            step=0.1,
            key="formulation_ph_slider",
            help=(
                "Target buffer pH. When pH approaches pI, aggregation risk spikes. "
                "For most IgG mAbs (pI 7.5-9.0), pH 5.5-6.5 is ideal."
            ),
        )
        st.session_state.formulation_buffer_ph = formulation_ph

        _buffer_options = {
            "histidine": "Histidine (pH 5.5-6.5, low viscosity)",
            "citrate": "Citrate (pH 3.0-6.5, strong capacity)",
            "phosphate": "Phosphate (pH 6.0-8.0, wide range)",
        }
        formulation_buffer = st.selectbox(
            "Buffer Type",
            options=list(_buffer_options.keys()),
            format_func=lambda k: _buffer_options[k],
            index=list(_buffer_options.keys()).index(st.session_state.formulation_buffer_type),
            key="formulation_buffer_select",
            help="Histidine for SC, Citrate for low-pH, Phosphate for IV.",
        )
        st.session_state.formulation_buffer_type = formulation_buffer

    with _form_c2:
        st.markdown("**Excipients**")
        _exc_trehalose = st.checkbox(
            "Trehalose (5-10% w/v)",
            value="trehalose" in st.session_state.formulation_excipients,
            key="exc_trehalose",
            help="Preferential exclusion stabilizer. Reduces Agg Risk by ~25%.",
        )
        _exc_sucrose = st.checkbox(
            "Sucrose (5-10% w/v)",
            value="sucrose" in st.session_state.formulation_excipients,
            key="exc_sucrose",
            help="Lyoprotectant and stabilizer. Reduces Agg Risk by ~22%.",
        )
        _exc_ps80 = st.checkbox(
            "Polysorbate 80 / PS80 (0.01-0.05%)",
            value="ps80" in st.session_state.formulation_excipients,
            key="exc_ps80",
            help="Surfactant for agitation stress protection. Reduces Agg Risk by ~15%.",
        )
        _selected_excipients = []
        if _exc_trehalose:
            _selected_excipients.append("trehalose")
        if _exc_sucrose:
            _selected_excipients.append("sucrose")
        if _exc_ps80:
            _selected_excipients.append("ps80")
        st.session_state.formulation_excipients = _selected_excipients
        if _selected_excipients:
            _exc_display = ", ".join(
                {"trehalose": "Trehalose", "sucrose": "Sucrose", "ps80": "PS80"}[e]
                for e in _selected_excipients
            )
            st.caption(f"Active excipients: {_exc_display}")

    st.markdown("---")

    _active_ws = ws_store.get_active()
    _ws_intent = _active_ws.get("intent") if _active_ws else None
    _ws_cache = _active_ws.get("analysis_cache") if _active_ws else None

    # Fallback to last_intent if workspace has no intent yet
    if not _ws_intent and st.session_state.get("last_intent"):
        _ws_intent = st.session_state.last_intent

    if _ws_intent:
        # -- Formulation Twin Panel --
        _dev_data = (_ws_cache.get("dev_result") or {}).get("data", {}) if _ws_cache else {}
        _base_preds = _dev_data.get("predictions", {})
        _base_dict = {
            "agg_risk": _base_preds.get("agg_risk", 0.2),
            "stability": _base_preds.get("stability", 0.82),
            "viscosity_risk": _base_preds.get("viscosity_risk", 0.15),
        }

        try:
            from src.formulation_twin import run_formulation_assessment
            _form_result = run_formulation_assessment(
                pI=_ws_intent.get("pI", 8.0),
                buffer_ph=st.session_state.formulation_buffer_ph,
                buffer_type=st.session_state.formulation_buffer_type,
                excipients=st.session_state.formulation_excipients,
                sequence=_ws_intent.get("sequence"),
                hydrophobicity=_ws_intent.get("hydrophobicity", 0.35),
                base_predictions=_base_dict,
            )
            render_formulation_twin_panel(_ws_intent, _form_result, _base_dict)
        except Exception as form_err:
            st.warning(f"Formulation twin unavailable: {form_err}")

        # -- M27: Time-Series Stability Projection --
        st.markdown("---")
        st.markdown("### Stability Projection")
        st.caption(
            "Arrhenius-based kinetic degradation simulation: project SEC HMW%, "
            "sub-visible particles, and potency over 24 months (5°C) and "
            "3 months accelerated (40°C)."
        )
        try:
            from src.stability_twin import run_stability_study
            import re as _re_stab

            _stab_seq = _ws_intent.get("sequence", "")
            _stab_pI = float(_ws_intent.get("pI", 8.5))
            _stab_hydro = float(_ws_intent.get("hydrophobicity", 0.35))
            _stab_deam = len(_re_stab.findall(r"N[GS]", _stab_seq.upper())) if _stab_seq else 5
            _stab_dp = len(_re_stab.findall(r"DP", _stab_seq.upper())) if _stab_seq else 1
            # Map ML risk score (0-1) to realistic starting HMW% for stability study
            # Low risk (0-0.2) → <1% HMW; Medium (0.2-0.5) → 1-3%; High (>0.5) → 3-8%
            _agg_risk_raw = _base_dict.get("agg_risk", 0.2)
            _stab_agg0 = _agg_risk_raw * _agg_risk_raw * 20.0  # Quadratic: 0.22→0.97%, 0.5→5%, 0.8→12.8%

            # Get formulation parameters from sidebar
            _stab_ph = st.session_state.get("formulation_buffer_ph", 6.0)
            _stab_exc = st.session_state.get("formulation_excipients", [])

            _stab_result = run_stability_study(
                starting_hmw_pct=max(_stab_agg0, 0.5),
                starting_acidic_pct=12.0,
                formulation_ph=_stab_ph,
                pI=_stab_pI,
                excipients=_stab_exc,
                deamidation_sites=_stab_deam,
                dp_clip_sites=_stab_dp,
                hydrophobicity=_stab_hydro,
            )

            # Grade banner
            _sg = _stab_result.overall_stability_grade
            from src.ui_colors import COLORS as _SG_C
            _sg_colors = {"Excellent": _SG_C["pass"]["primary"], "Good": _SG_C["info"]["primary"],
                          "At Risk": _SG_C["caution"]["primary"], "Poor": _SG_C["fail"]["primary"]}
            _sgc = _sg_colors.get(_sg, "#64748B")
            st.markdown(
                f'<div style="background:{_sgc}15;border-left:4px solid {_sgc};'
                f'padding:12px;border-radius:8px;margin:8px 0;">'
                f'<span style="font-size:1.1rem;font-weight:700;color:{_sgc};">'
                f'Stability Grade: {_sg}</span>'
                f'<span style="margin-left:16px;font-size:0.9rem;">'
                f'Predicted shelf life: ~{_stab_result.predicted_shelf_life_months:.0f} months</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            _stm1, _stm2, _stm3, _stm4 = st.columns(4)
            _lt = _stab_result.long_term
            _ac = _stab_result.accelerated
            _stm1.metric("HMW at 24mo (5°C)", f"{_lt.final_hmw_pct:.2f}%")
            _stm2.metric("HMW at 3mo (40°C)", f"{_ac.final_hmw_pct:.1f}%")
            _stm3.metric("Potency at 24mo", f"{_lt.final_potency_pct:.1f}%")
            _stm4.metric("Shelf Life", f"~{_stab_result.predicted_shelf_life_months:.0f} mo")

            # Time-series charts
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                _fig_stab = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "SEC HMW% — 5°C (24 months)", "SEC HMW% — 40°C (3 months)",
                        "Potency Decay — 5°C", "SVP (>=10um/mL) — 5°C",
                    ),
                    vertical_spacing=0.12,
                )

                # 5°C HMW
                _lt_months = [tp.month for tp in _lt.timepoints]
                _lt_hmw = [tp.sec_hmw_pct for tp in _lt.timepoints]
                _fig_stab.add_trace(go.Scatter(x=_lt_months, y=_lt_hmw,
                    name="5\u00b0C HMW%", line=dict(color=PLOTLY_COLORS[0], width=2.5)), row=1, col=1)
                _fig_stab.add_hline(y=5.0, line_dash="dash", line_color=PLOTLY_COLORS[2],
                    annotation_text="Spec Limit (5%)", row=1, col=1)

                # 40°C HMW
                _ac_months = [tp.month for tp in _ac.timepoints]
                _ac_hmw = [tp.sec_hmw_pct for tp in _ac.timepoints]
                _fig_stab.add_trace(go.Scatter(x=_ac_months, y=_ac_hmw,
                    name="40\u00b0C HMW%", line=dict(color=PLOTLY_COLORS[2], width=2.5)), row=1, col=2)

                # Potency
                _lt_pot = [tp.potency_pct for tp in _lt.timepoints]
                _fig_stab.add_trace(go.Scatter(x=_lt_months, y=_lt_pot,
                    name="Potency%", line=dict(color=PLOTLY_COLORS[1], width=2.5)), row=2, col=1)
                _fig_stab.add_hline(y=80.0, line_dash="dash", line_color=PLOTLY_COLORS[4],
                    annotation_text="Min Spec (80%)", row=2, col=1)

                # SVP
                _lt_svp = [tp.svp_per_ml for tp in _lt.timepoints]
                _fig_stab.add_trace(go.Scatter(x=_lt_months, y=_lt_svp,
                    name="SVP/mL", line=dict(color=PLOTLY_COLORS[3], width=2.5)), row=2, col=2)

                _apply_pharma_theme(_fig_stab,
                    height=550, showlegend=False,
                    title="Stability Projection: Kinetic Degradation Model",
                )
                _fig_stab.update_xaxes(title_text="Months", row=2, col=1)
                _fig_stab.update_xaxes(title_text="Months", row=2, col=2)

                st.plotly_chart(_fig_stab, use_container_width=True)
            except ImportError:
                st.caption("(Install plotly for stability charts)")

        except Exception as stab_err:
            st.caption(f"(Stability projection unavailable: {stab_err})")

    else:
        st.info(
            "No molecule loaded. Go to **Molecule Setup** and load a sequence first, "
            "then return here to view formulation, preclinical, and clinical results."
        )

    # -- Immunogenicity / ADA Risk Tab (merged from Clinical & Immunogenicity page) --
    st.markdown("---")
    st.markdown("### Immunogenicity & ADA Risk")
    st.caption(
        "MHC-II binding prediction, humanization scoring, and anti-drug antibody (ADA) risk classification."
    )

    _ada_ws = ws_store.get_active() if ws_store else None
    _ada_intent = _ada_ws.get("intent") if _ada_ws else None
    _ada_sequence = _ada_intent.get("sequence", "") if _ada_intent else ""
    _ada_mol_name = _ada_intent.get("name", "Unknown") if _ada_intent else "Unknown"

    if _ada_sequence:
        _ada_clean = "".join(c for c in _ada_sequence.upper() if c.isalpha())

        if st.button("Run ADA Assessment", type="primary",
                      key="btn_ada_run_preclin"):
            try:
                from src.immunogenicity_twin import run_immunogenicity_assessment
                _ada_dev_cache = (_ada_ws.get("analysis_cache") or {}) if _ada_ws else {}
                _ada_dev = (_ada_dev_cache.get("dev_result") or {})
                _ada_dev_data = _ada_dev.get("data", _ada_dev) if isinstance(_ada_dev, dict) else {}
                _ada_agg_risk = (_ada_dev_data.get("predictions", {}).get("agg_risk") or 0.0)
                _ada_dev_score = _ada_dev_data.get("composite_score", 0.0)

                _ada_mol_cls = (_ada_ws.get("intent") or {}).get("molecule_class", "unknown") if _ada_ws else "unknown"
                _ada_result = run_immunogenicity_assessment(
                    sequence=_ada_clean,
                    agg_risk=_ada_agg_risk,
                    dev_score=_ada_dev_score,
                    molecule_name=_ada_mol_name,
                    molecule_class=_ada_mol_cls,
                )
                st.session_state["ada_result"] = _ada_result
            except Exception as _ada_err:
                st.error(f"ADA assessment error: {_ada_err}")

        _ada_result = st.session_state.get("ada_result")
        if _ada_result is not None:
            from src.ui_colors import COLORS as _ADA_C
            _ada_risk_colors = {"Low": _ADA_C["pass"]["primary"], "Medium": _ADA_C["caution"]["primary"],
                                "High": _ADA_C["fail"]["primary"]}
            _ada_rl = getattr(_ada_result, "ada_risk_level", "Unknown")
            _ada_rs = getattr(_ada_result, "ada_risk_score", 0.0)
            _ada_color = _ada_risk_colors.get(_ada_rl, "#64748B")

            st.markdown(
                f'<div style="background:{_ada_color}15;border-left:4px solid {_ada_color};'
                f'padding:12px 16px;border-radius:8px;margin:8px 0;">'
                f'<span style="font-size:1.1rem;font-weight:700;color:{_ada_color};">'
                f'ADA Risk: {_ada_rl}</span>'
                f'<span style="margin-left:16px;color:#64748B;">Score: {_ada_rs:.3f}</span></div>',
                unsafe_allow_html=True,
            )

            # Key metrics
            _am1, _am2, _am3, _am4 = st.columns(4)
            with _am1:
                st.metric("Peptides Scanned", f"{getattr(_ada_result, 'total_peptides_scanned', getattr(_ada_result, 'n_peptides', 0)):,}")
            with _am2:
                st.metric("High-Risk Hotspots", str(getattr(_ada_result, 'n_high_risk', getattr(_ada_result, 'n_high', 0))))
            with _am3:
                st.metric("Humanization", f"{getattr(_ada_result, 'humanization_score', 0):.2f}")
            with _am4:
                st.metric("FR Identity", f"{getattr(_ada_result, 'framework_identity_pct', getattr(_ada_result, 'framework_identity', 0)):.0f}%")

            # Hotspot table
            _hotspots = getattr(_ada_result, "hotspots", [])
            if _hotspots:
                with st.expander(f"Top Immunogenic Hotspots ({min(15, len(_hotspots))})", expanded=True):
                    import pandas as _pd_ada
                    _hs_data = []
                    for _hs in _hotspots[:15]:
                        if isinstance(_hs, dict):
                            _hs_dict = _hs
                        else:
                            _hs_dict = {
                                "Position": getattr(_hs, "position", 0),
                                "Peptide": getattr(_hs, "peptide", ""),
                                "Score": round(getattr(_hs, "score", 0), 3),
                                "Risk": getattr(_hs, "risk_label", getattr(_hs, "risk", "")),
                                "In CDR": "Yes" if getattr(_hs, "in_cdr", False) else "No",
                                "Anchors": getattr(_hs, "anchor_residues", ""),
                            }
                        _hs_data.append(_hs_dict)
                    if _hs_data:
                        st.dataframe(_pd_ada.DataFrame(_hs_data), use_container_width=True, hide_index=True)

            # Clinical recommendations
            with st.expander("Clinical Recommendations", expanded=True):
                if _ada_rl == "High":
                    st.warning("Consider deimmunization engineering, in-vitro PBMC T-cell assay, and Fc-silent or PEGylation strategies.")
                elif _ada_rl == "Medium":
                    st.info("Include immunogenicity monitoring in Phase I protocol. In-vitro T-cell assay (ELISpot/PBMC) recommended.")
                else:
                    st.success("Standard immunogenicity monitoring sufficient. No deimmunization engineering required.")
    else:
        st.info("Load a molecule to run immunogenicity assessment.")


# [REMOVED v32.1] Dead page code removed. See git history.

# [REMOVED v32.1] Dead page code removed. See git history.

# ===========================================================================
#  AI Training Center (Consolidated: Training + Active Learning + Benchmark)
# ===========================================================================
elif active_page in ("AI Training Center", "Model Management"):
    st.markdown("## Model Management")
    st.caption(
        "Unified hub for model training, active learning experiment design, "
        "and benchmark validation. Upload data, train models, design experiments, "
        "and validate predictions against the NISTmAb gold standard."
    )

    # -- Model Training Status Dashboard (always visible) --
    try:
        from src.ml_predictor import get_model_status as _get_ms
        _mstatus = _get_ms()
    except Exception:
        _mstatus = {}
    _any_trained = any(
        v.get("loaded") or v.get("persisted") for v in _mstatus.values()
    ) if _mstatus else False

    with st.container(border=True):
        from src.ui_colors import COLORS, STATUS_DOT
        _cal_info = _mstatus.get("calibration", {})
        _is_calibrated = _cal_info.get("calibrated", False)
        if _any_trained and not _is_calibrated:
            _mode_dot = STATUS_DOT["pass"]
            _mode_text = "Custom Trained Model"
        elif _is_calibrated:
            _mode_dot = STATUS_DOT["pass"]
            _mode_text = "Literature-Calibrated (Jain-137)"
        else:
            _mode_dot = STATUS_DOT["caution"]
            _mode_text = "Baseline Heuristics"
        st.markdown(f"{_mode_dot} **Current Mode: {_mode_text}**", unsafe_allow_html=True)

        _wl_metrics = st.session_state.get("wetlab_training_metrics", {})
        _wl_info = st.session_state.get("wetlab_dataset_info", {})
        if _any_trained and (_wl_metrics or _wl_info):
            _n_data = _wl_metrics.get("n_samples") or _wl_info.get("n_sequences", "?")
            _ts = _wl_metrics.get("timestamp", "unknown")
            st.caption(f"Trained on {_n_data} sequences | Last updated: {_ts}")
        elif not _any_trained:
            st.caption("Upload training data and run model training to upgrade from baseline heuristics.")

        _SOURCE_LABELS = {
            "custom_trained": ("Custom Trained", "Trained on your uploaded data"),
            "literature_calibrated": ("Lit-Calibrated", "Calibrated from Jain-137 literature panel"),
            "persisted_on_disk": ("Persisted", "Previously trained model loaded from disk"),
            "baseline_heuristic": ("Baseline", "Rule-based heuristics (no ML); lowest accuracy"),
        }
        _MODEL_DESCRIPTIONS = {
            "wetlab": "Predicts aggregation % (SEC HMW) and thermal stability (Tm) from sequence features",
            "potency": "Predicts binding potency / functional activity score from biophysical properties",
            "chromatography_mlp": "Predicts ion-exchange binding parameters (ka, nu) for chromatographic separation",
        }
        _ms1, _ms2, _ms3 = st.columns(3)
        for _ms_col, (_ms_key, _ms_label) in zip(
            [_ms1, _ms2, _ms3],
            [("wetlab", "WetLab XGBoost"), ("potency", "Potency XGBoost"), ("chromatography_mlp", "Chromatography MLP")],
        ):
            _ms_info = _mstatus.get(_ms_key, {})
            _ms_src = _ms_info.get("source", "baseline_heuristic")
            _ms_display, _ms_hint = _SOURCE_LABELS.get(_ms_src, ("Unknown", ""))
            with _ms_col:
                st.metric(
                    _ms_label, _ms_display,
                    help=f"{_ms_hint}. {_MODEL_DESCRIPTIONS.get(_ms_key, '')}",
                )

    # -- v2.0: Quick Calibration from Jain-137 Literature Data --
    _cal_status = _mstatus.get("calibration", {})
    with st.expander("Quick Calibrate from Jain-137 Literature", expanded=not _any_trained):
        st.markdown(
            "**What is this?** This calibrates the prediction engine using a synthetic panel modeled after "
            "Jain et al. (PNAS 2017) — 137 clinical-stage mAbs with measured biophysical properties. "
            "The platform generates realistic antibody sequences with correlated assay outcomes "
            "(aggregation %, melting temperature Tm, viscosity, etc.) and trains an XGBoost model on them."
        )
        st.markdown(
            "**Why use it?** The uncalibrated baseline uses simple rule-of-thumb heuristics "
            "(e.g., hydrophobicity > 0.5 = high aggregation risk). After Jain-137 calibration, "
            "the model learns nonlinear feature interactions from literature-representative data, "
            "giving more accurate risk predictions without requiring your own wet-lab data."
        )
        st.caption(
            "After calibration, you can further improve accuracy by uploading your own experimental CSV "
            "in the 'Data & Model Training' tab — this produces a Custom Trained model that supersedes "
            "the literature calibration."
        )

        _cal_c1, _cal_c2 = st.columns([2, 1])
        with _cal_c1:
            _cal_n = st.slider(
                "Sample size (mAb panel)", 20, 200, 50, step=10, key="jain_cal_n",
                help="Number of synthetic mAbs to generate. 50 is sufficient for stable calibration; "
                     "increase to 100-200 for tighter confidence intervals.",
            )
        with _cal_c2:
            _cal_is_done = _cal_status.get("calibrated", False)
            if _cal_is_done:
                st.success("Literature-Calibrated")
                _cal_m = _cal_status.get("metrics") or {}
                if _cal_m:
                    # Show key metrics inline
                    for _tname, _tmetrics in _cal_m.items():
                        if isinstance(_tmetrics, dict) and "r2" in _tmetrics:
                            st.caption(f"{_tname}: R²={_tmetrics['r2']:.3f}, MAE={_tmetrics.get('mae', 0):.3f}")
            else:
                st.info("Not calibrated")

        if st.button("Calibrate Baseline from Jain-137", type="primary", key="btn_jain_cal"):
            with st.spinner("Generating Jain-137 panel and training foundation model..."):
                try:
                    from src.ml_predictor import calibrate_baseline_from_jain137
                    _cal_result = calibrate_baseline_from_jain137(n_samples=_cal_n)
                    if _cal_result.get("status") == "success":
                        st.success(
                            f"Calibration complete on {_cal_result.get('n_samples', '?')} mAbs. "
                            f"Predictions now use **Literature-Calibrated (Jain-137)** model."
                        )
                        _cal_metrics = _cal_result.get("metrics", {})
                        if _cal_metrics:
                            for _t, _m in _cal_metrics.items():
                                if isinstance(_m, dict) and "r2" in _m:
                                    st.markdown(
                                        f"- **{_t}**: R² = {_m['r2']:.3f}, "
                                        f"MAE = {_m.get('mae', 0):.3f}, "
                                        f"RMSE = {_m.get('rmse', 0):.3f}"
                                    )
                        st.rerun()
                    else:
                        st.error(f"Calibration failed: {_cal_result.get('message', 'unknown error')}")
                except Exception as _cal_err:
                    st.error(f"Calibration error: {_cal_err}")

    _aitc_tab1, _aitc_tab2 = st.tabs([
        "Data & Model Training",
        "Continuous Learning",
    ])

    # ------------------------------------------------------------------
    #  Tab 1: Data & Model Training
    # ------------------------------------------------------------------
    with _aitc_tab1:
        st.caption(
            "Upload CSV datasets to retrain models. Auto-detects VH/VL columns and prepares features for training."
        )

        # ---- Section 1: Upload & Harmonize ----
        st.markdown("### 1. Data Upload and Harmonization")

        _tc_csv = st.file_uploader(
            "Upload Training CSV",
            type=["csv", "tsv"],
            key="tc_csv_uploader",
            help=(
                "Upload a CSV with VH and/or VL sequence columns plus assay targets. "
                "Supported column names: VH, Heavy Chain, VL, Light Chain, Sequence, "
                "AC-SINS, BVP, Aggregation, Titer, Stability, etc."
            ),
        )

        if _tc_csv is not None:
            try:
                import pandas as pd

                _sep = "\t" if _tc_csv.name.lower().endswith(".tsv") else ","
                _tc_df_raw = pd.read_csv(_tc_csv, sep=_sep)
                st.caption(
                    f"Loaded file: {_tc_csv.name}  --  "
                    f"{len(_tc_df_raw)} rows x {len(_tc_df_raw.columns)} columns"
                )

                # Show raw column names for transparency
                with st.expander("Raw Columns Detected", expanded=True):
                    st.text(", ".join(_tc_df_raw.columns.tolist()))

                # Harmonize
                from src.data_harmonizer import DataHarmonizer
                _tc_harmonizer = DataHarmonizer(min_sequence_length=20)
                _tc_result = _tc_harmonizer.harmonize(_tc_df_raw)

                if _tc_result["status"] == "success":
                    _tc_data = _tc_result["data"]
                    _tc_targets = _tc_result["target_columns"]
                    _tc_nv = _tc_result["n_valid"]
                    _tc_nd = _tc_result["n_dropped"]
                    _det = _tc_result.get("detected_columns", {})

                    # Metrics banner
                    _b1, _b2, _b3 = st.columns(3)
                    _b1.metric("Sequences Harmonized", f"{_tc_nv}")
                    _b2.metric("Rows Dropped", f"{_tc_nd}")
                    _b3.metric("Target Assays", f"{len(_tc_targets)}")

                    # Show average sequence length from harmonized data
                    _seq_stats = _tc_result.get("sequence_stats", {})
                    if _seq_stats:
                        _stats_cols = st.columns(3)
                        with _stats_cols[0]:
                            st.metric("Avg Sequence Length", f"{_seq_stats.get('mean_length', 0):.0f} aa")
                        with _stats_cols[1]:
                            st.metric("Std Dev", f"± {_seq_stats.get('std_length', 0):.0f} aa")
                        with _stats_cols[2]:
                            st.metric("Assembly Mode", _seq_stats.get("assembly_mode", "single").replace("_", " ").title())

                        # Compute and store dynamic OOD baseline
                        try:
                            from src.ood_baseline import OODBaselineCalculator
                            _ood_calc = OODBaselineCalculator()
                            _ood_baseline = _ood_calc.compute_from_training_data(_tc_data)
                            st.session_state["ood_computed_baseline"] = _ood_calc.to_dict()
                            st.success(f"OOD baseline updated from {_seq_stats['n_sequences']} training sequences.")
                        except Exception as _ood_err:
                            st.caption(f"OOD baseline: using defaults ({_ood_err})")

                    # Column detection summary
                    _det_lines = []
                    if _det.get("vh"):
                        _det_lines.append(f"VH: {_det['vh']}")
                    if _det.get("vl"):
                        _det_lines.append(f"VL: {_det['vl']}")
                    if _det.get("single_seq"):
                        _det_lines.append(f"Sequence: {_det['single_seq']}")
                    if _det.get("id"):
                        _det_lines.append(f"ID: {_det['id']}")
                    if _det_lines:
                        st.caption("Column mapping: " + " | ".join(_det_lines))

                    for _w in _tc_result.get("warnings", []):
                        st.caption(_w)

                    # Preview
                    st.markdown("#### Harmonized Data Preview")
                    _preview_cols = ["Molecule_ID", "Combined_Sequence"] + _tc_targets
                    _preview_cols = [c for c in _preview_cols if c in _tc_data.columns]
                    _preview = _tc_data[_preview_cols].head(15).copy()
                    _preview["Combined_Sequence"] = _preview["Combined_Sequence"].str[:60] + "..."
                    st.dataframe(_preview, use_container_width=True, hide_index=True)

                    # Store for training
                    st.session_state["tc_harmonized_data"] = _tc_data
                    st.session_state["tc_target_columns"] = _tc_targets

                    # ---- Section 2: Target Selection ----
                    st.markdown("---")
                    st.markdown("### 2. Select Target Assay")

                    if _tc_targets:
                        _tc_selected_target = st.selectbox(
                            "Target for model training",
                            options=_tc_targets,
                            key="tc_target_select",
                            help="Select which assay column to use as the regression target for XGBoost.",
                        )

                        # Show target distribution
                        _valid_targets = _tc_data[_tc_selected_target].dropna()
                        if len(_valid_targets) > 0:
                            _t1, _t2, _t3, _t4 = st.columns(4)
                            _t1.metric("Valid Samples", f"{len(_valid_targets)}")
                            _t2.metric("Mean", f"{_valid_targets.mean():.4f}")
                            _t3.metric("Std Dev", f"{_valid_targets.std():.4f}")
                            _t4.metric("Range", f"{_valid_targets.min():.3f} - {_valid_targets.max():.3f}")
                    else:
                        st.warning(
                            "No target assay columns detected. Ensure your CSV contains "
                            "columns with names like AC-SINS, BVP, Aggregation, Stability, etc."
                        )
                        _tc_selected_target = None

                    # ---- Section 3: Feature Extraction & Training ----
                    if _tc_selected_target:
                        st.markdown("---")
                        st.markdown("### 3. Extract Features and Train Foundation Model")
                        st.caption("Train a single-target XGBoost model with biophysical features. Model is saved to disk for persistence across sessions.")

                        if st.button(
                            "Train Foundation Model",
                            key="btn_tc_train",
                            type="primary",
                        ):
                            with st.spinner("Extracting biophysical features and training XGBoost..."):
                                try:
                                    from src.data_harmonizer import (
                                        prepare_training_matrix,
                                        BIOPHYS_FEATURE_NAMES,
                                    )

                                    X_train, y_train, mol_ids = prepare_training_matrix(
                                        _tc_data, _tc_selected_target,
                                    )
                                    st.caption(
                                        f"Feature matrix: {X_train.shape[0]} samples x "
                                        f"{X_train.shape[1]} features ({', '.join(BIOPHYS_FEATURE_NAMES)})"
                                    )

                                    # Train XGBoost (or scikit-learn fallback)
                                    _xgb_available = False
                                    try:
                                        import xgboost as xgb
                                        _xgb_available = True
                                    except ImportError:
                                        pass

                                    from sklearn.model_selection import cross_val_score
                                    from sklearn.metrics import r2_score, mean_absolute_error
                                    import numpy as np

                                    if _xgb_available:
                                        model = xgb.XGBRegressor(
                                            n_estimators=100,
                                            max_depth=4,
                                            learning_rate=0.1,
                                            subsample=0.8,
                                            colsample_bytree=0.8,
                                            random_state=42,
                                            verbosity=0,
                                        )
                                        _model_label = "XGBoost"
                                    else:
                                        from sklearn.ensemble import GradientBoostingRegressor
                                        model = GradientBoostingRegressor(
                                            n_estimators=100,
                                            max_depth=4,
                                            learning_rate=0.1,
                                            subsample=0.8,
                                            random_state=42,
                                        )
                                        _model_label = "GradientBoosting (sklearn)"

                                    # Cross-validation
                                    _n_folds = min(5, max(2, len(X_train) // 5))
                                    cv_scores = cross_val_score(
                                        model, X_train, y_train,
                                        cv=_n_folds, scoring="r2",
                                    )

                                    # Fit on full dataset
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_train)
                                    r2_full = r2_score(y_train, y_pred)
                                    mae_full = mean_absolute_error(y_train, y_pred)

                                    # Feature importance
                                    if _xgb_available:
                                        importances = model.feature_importances_
                                    else:
                                        importances = model.feature_importances_

                                    # Display results
                                    st.success(
                                        f"Training complete ({_model_label}). "
                                        f"{_n_folds}-fold CV R2: {cv_scores.mean():.4f} "
                                        f"(+/- {cv_scores.std():.4f})  |  "
                                        f"Full-data R2: {r2_full:.4f}  |  MAE: {mae_full:.4f}"
                                    )

                                    # Metrics cards
                                    _m1, _m2, _m3, _m4 = st.columns(4)
                                    _m1.metric("CV R2 (mean)", f"{cv_scores.mean():.4f}")
                                    _m2.metric("CV R2 (std)", f"{cv_scores.std():.4f}")
                                    _m3.metric("Full R2", f"{r2_full:.4f}")
                                    _m4.metric("MAE", f"{mae_full:.4f}")

                                    # Feature importance table
                                    st.markdown("#### Feature Importance")
                                    _imp_df = pd.DataFrame({
                                        "Feature": BIOPHYS_FEATURE_NAMES,
                                        "Importance": [round(float(v), 4) for v in importances],
                                    }).sort_values("Importance", ascending=False)
                                    st.dataframe(_imp_df, use_container_width=True, hide_index=True)

                                    # Predicted vs Actual scatter
                                    try:
                                        import plotly.graph_objects as go
                                        _fig_scatter = go.Figure()
                                        _fig_scatter.add_trace(go.Scatter(
                                            x=y_train, y=y_pred,
                                            mode="markers",
                                            marker=dict(size=6, color="#3B82F6", opacity=0.7),
                                            text=mol_ids,
                                            name="Predictions",
                                        ))
                                        _min_val = min(float(y_train.min()), float(y_pred.min()))
                                        _max_val = max(float(y_train.max()), float(y_pred.max()))
                                        _fig_scatter.add_trace(go.Scatter(
                                            x=[_min_val, _max_val],
                                            y=[_min_val, _max_val],
                                            mode="lines",
                                            line=dict(color="#64748B", dash="dash", width=1),
                                            name="Ideal",
                                        ))
                                        _apply_pharma_theme(_fig_scatter,
                                            title=f"Predicted vs Actual: {_tc_selected_target}",
                                            xaxis_title="Actual",
                                            yaxis_title="Predicted",
                                            height=420,
                                            showlegend=False,
                                        )
                                        st.plotly_chart(_fig_scatter, use_container_width=True)
                                    except ImportError:
                                        st.caption("(Install plotly for scatter visualization)")

                                    # Store model in session state
                                    st.session_state["tc_trained_model"] = model
                                    st.session_state["tc_trained_target"] = _tc_selected_target
                                    st.session_state["tc_train_r2"] = r2_full
                                    st.session_state["tc_train_cv_r2"] = cv_scores.mean()

                                    # Persist model to disk
                                    import joblib
                                    _tc_model_path = os.path.join(str(ROOT), "models", f"xgboost_tc_{_tc_selected_target.lower().replace(' ', '_')}.pkl")
                                    os.makedirs(os.path.join(str(ROOT), "models"), exist_ok=True)
                                    joblib.dump({
                                        "model": model,
                                        "target": _tc_selected_target,
                                        "r2": r2_full,
                                        "cv_r2": float(cv_scores.mean()),
                                        "n_samples": len(X_train),
                                        "features": BIOPHYS_FEATURE_NAMES,
                                    }, _tc_model_path)
                                    st.info(f"Model saved to `models/{os.path.basename(_tc_model_path)}`")

                                except ImportError as imp_err:
                                    st.error(
                                        f"Missing dependency: {imp_err}. "
                                        "Install with: pip install scikit-learn xgboost"
                                    )
                                except Exception as train_err:
                                    st.error(f"Training failed: {train_err}")
                                    import traceback
                                    st.caption(traceback.format_exc())

                        # Show cached training results
                        if st.session_state.get("tc_trained_model") is not None:
                            _cached_target = st.session_state.get("tc_trained_target", "")
                            _cached_r2 = st.session_state.get("tc_train_r2", 0.0)
                            _cached_cv = st.session_state.get("tc_train_cv_r2", 0.0)
                            st.markdown(
                                f'<div style="background:#F0FDF4;border-left:4px solid #10B981;'
                                f'padding:12px;border-radius:8px;margin:12px 0;">'
                                f'<span style="font-weight:700;color:#10B981;">Model Active</span>'
                                f'<span style="margin-left:16px;color:#334155;">'
                                f'Target: {_cached_target}  |  '
                                f'R2: {_cached_r2:.4f}  |  CV R2: {_cached_cv:.4f}'
                                f'</span></div>',
                                unsafe_allow_html=True,
                            )

                    # ---- Section 4: Unified Multi-Task Model Training ----
                    st.markdown("---")
                    st.markdown("### 4. Unified Multi-Task Model (8-Task)")
                    st.caption("Train neural network to predict 8 properties with shared knowledge.")

                    # Detect unified targets from CSV columns
                    from src.data_harmonizer import detect_unified_targets
                    _unified_detected = detect_unified_targets(_tc_data.columns.tolist())

                    if _unified_detected:
                        st.success(f"Detected {len(_unified_detected)} unified targets in your CSV")
                        _det_table = [
                            {"Task": task, "CSV Column": col}
                            for task, col in sorted(_unified_detected.items())
                        ]
                        st.dataframe(
                            pd.DataFrame(_det_table),
                            use_container_width=True, hide_index=True,
                        )

                        # Target selection
                        _avail_tasks = list(_unified_detected.keys())
                        _selected_tasks = st.multiselect(
                            "Select tasks to include in unified training",
                            options=_avail_tasks,
                            default=_avail_tasks,
                            key="unified_task_select",
                            help="At least 2 tasks required for multi-task learning benefit.",
                        )

                        if len(_selected_tasks) >= 2:
                            # ---- Section 5: ESM-2 Precomputation ----
                            st.markdown("---")
                            st.markdown("### 5. Precompute ESM-2 Embeddings")
                            st.caption("Cache 480-dim embeddings for faster training.")

                            _esm2_cached = st.session_state.get("unified_esm2_cache") is not None
                            if _esm2_cached:
                                _cache = st.session_state["unified_esm2_cache"]
                                st.info(f"ESM-2 cache ready: {_cache.shape[0]} embeddings ({_cache.shape[1]}-dim)")
                            else:
                                if st.button("Precompute ESM-2 Cache", key="btn_esm2_cache", type="primary"):
                                    with st.spinner("Computing ESM-2 embeddings (this may take 1-5 minutes)..."):
                                        try:
                                            import sys as _sys
                                            _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
                                            from esm2_hybrid_encoder import ESM2HybridEncoder
                                            import torch

                                            _encoder = ESM2HybridEncoder()

                                            # Get sequences
                                            _det_cols = st.session_state.get("tc_harmonized_data", _tc_data)
                                            if "hc_sequence" in _det_cols.columns:
                                                _hc_list = _det_cols["hc_sequence"].astype(str).tolist()
                                                _lc_list = _det_cols["lc_sequence"].astype(str).tolist()
                                            elif "Combined_Sequence" in _det_cols.columns:
                                                _hc_list = _det_cols["Combined_Sequence"].astype(str).tolist()
                                                _lc_list = _hc_list  # fallback: same seq for HC/LC
                                            else:
                                                raise ValueError("No sequence column found")

                                            _embs = []
                                            _prog = st.progress(0)
                                            for _ei in range(len(_hc_list)):
                                                _emb = _encoder.encode_sequences([_hc_list[_ei]], [_lc_list[_ei]])
                                                _embs.append(_emb.squeeze(0))
                                                _prog.progress((_ei + 1) / len(_hc_list))

                                            st.session_state["unified_esm2_cache"] = torch.stack(_embs)
                                            st.success(f"Cached {len(_embs)} ESM-2 embeddings")
                                            st.rerun()
                                        except Exception as _esm_err:
                                            st.error(f"ESM-2 precomputation failed: {_esm_err}")

                            # ---- Section 6: Train Unified Model ----
                            st.markdown("---")
                            st.markdown("### 6. Train Unified Multi-Task Model")

                            _uf_epochs = st.slider("Epochs", 10, 200, 50, key="uf_epochs")
                            _uf_lr = st.select_slider(
                                "Learning Rate",
                                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                                value=0.001,
                                key="uf_lr",
                            )

                            if st.button(
                                "Train Unified Model",
                                key="btn_train_unified",
                                type="primary",
                            ):
                                with st.spinner("Training unified 8-task model..."):
                                    try:
                                        import sys as _sys
                                        import os as _os
                                        _project_root = _os.path.dirname(_os.path.abspath(__file__))
                                        _sys.path.insert(0, _os.path.join(_project_root, "src"))

                                        import torch
                                        from torch.utils.data import DataLoader, random_split
                                        from esm2_hybrid_encoder import ESM2HybridEncoder
                                        from unified_multitask_model import UnifiedMultiTaskModel, UNIFIED_TASKS
                                        from unified_dataset import UnifiedAntibodyDataset, unified_collate_fn
                                        from unified_trainer import UnifiedTrainer

                                        # Prepare CSV with unified column names
                                        _uf_df = _tc_data.copy()

                                        # Map detected CSV columns to unified task names
                                        for _task, _csv_col in _unified_detected.items():
                                            if _task in _selected_tasks and _csv_col in _uf_df.columns:
                                                if _task != _csv_col:
                                                    _uf_df[_task] = _uf_df[_csv_col]

                                        # Ensure sequence columns
                                        if "hc_sequence" not in _uf_df.columns:
                                            if "Combined_Sequence" in _uf_df.columns:
                                                _uf_df["hc_sequence"] = _uf_df["Combined_Sequence"]
                                                _uf_df["lc_sequence"] = _uf_df["Combined_Sequence"]

                                        # Save temp CSV for dataset loader
                                        import tempfile
                                        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as _tmp:
                                            _uf_df.to_csv(_tmp.name, index=False)
                                            _tmp_path = _tmp.name

                                        # Load dataset
                                        _uf_ds = UnifiedAntibodyDataset(
                                            _tmp_path,
                                            tasks=_selected_tasks,
                                            compute_biophys=True,
                                        )

                                        # Split
                                        _tr_size = int(0.8 * len(_uf_ds))
                                        _va_size = len(_uf_ds) - _tr_size
                                        _tr_ds, _va_ds = random_split(_uf_ds, [_tr_size, _va_size])

                                        _tr_loader = DataLoader(_tr_ds, batch_size=16, shuffle=True, collate_fn=unified_collate_fn)
                                        _va_loader = DataLoader(_va_ds, batch_size=16, shuffle=False, collate_fn=unified_collate_fn)

                                        # Model
                                        _uf_enc = ESM2HybridEncoder(hidden_dim=256)
                                        _uf_model = UnifiedMultiTaskModel(
                                            encoder=_uf_enc, encoder_dim=256, tasks=_selected_tasks,
                                        )

                                        _uf_optim = torch.optim.AdamW(_uf_model.parameters(), lr=_uf_lr, weight_decay=1e-4)
                                        _uf_save_dir = _os.path.join(_project_root, "models")
                                        _os.makedirs(_uf_save_dir, exist_ok=True)

                                        _uf_trainer = UnifiedTrainer(
                                            model=_uf_model,
                                            train_loader=_tr_loader,
                                            val_loader=_va_loader,
                                            optimizer=_uf_optim,
                                            patience=15,
                                            save_dir=_uf_save_dir,
                                        )

                                        _uf_summary = _uf_trainer.train(epochs=_uf_epochs, verbose=False)

                                        # Display results
                                        st.success(
                                            f"Training complete! "
                                            f"{_uf_summary['epochs_trained']} epochs, "
                                            f"best val loss: {_uf_summary['best_val_loss']:.4f}, "
                                            f"time: {_uf_summary['elapsed_seconds']:.1f}s"
                                        )

                                        _um1, _um2, _um3 = st.columns(3)
                                        _um1.metric("Val Loss", f"{_uf_summary['best_val_loss']:.4f}")
                                        _um2.metric("Epochs", f"{_uf_summary['epochs_trained']}")
                                        _um3.metric("Time", f"{_uf_summary['elapsed_seconds']:.1f}s")

                                        st.session_state["unified_model_trained"] = True
                                        st.session_state["unified_model_summary"] = _uf_summary

                                        st.info(
                                            f"Model saved to: models/unified_multitask_best.pt  \n"
                                            f"All analysis pages will now use the trained unified predictions."
                                        )

                                        # Cleanup temp file
                                        _os.unlink(_tmp_path)

                                    except Exception as _uf_err:
                                        st.error(f"Unified training failed: {_uf_err}")
                                        import traceback
                                        st.caption(traceback.format_exc())

                            # Show cached training status
                            if st.session_state.get("unified_model_trained"):
                                _uf_s = st.session_state.get("unified_model_summary", {})
                                st.markdown(
                                    f'<div style="background:#F0FDF4;border-left:4px solid #10B981;'
                                    f'padding:12px;border-radius:8px;margin:12px 0;">'
                                    f'<span style="font-weight:700;color:#10B981;">Unified Model Active</span>'
                                    f'<span style="margin-left:16px;color:#334155;">'
                                    f'Tasks: {len(_selected_tasks)}  |  '
                                    f'Val Loss: {_uf_s.get("best_val_loss", 0):.4f}  |  '
                                    f'Epochs: {_uf_s.get("epochs_trained", 0)}'
                                    f'</span></div>',
                                    unsafe_allow_html=True,
                                )

                        elif len(_selected_tasks) < 2 and _selected_tasks:
                            st.warning("Select at least 2 tasks for multi-task learning.")

                    else:
                        st.caption(
                            "No unified targets detected in your CSV. "
                            "Ensure columns contain names like: Tm, Aggregation, Stability, "
                            "Viscosity, Potency, Hydrophobicity, ka, nu."
                        )

                else:
                    st.error(f"Harmonization failed: {_tc_result.get('message', 'Unknown error')}")
                    for _w in _tc_result.get("warnings", []):
                        st.caption(_w)

            except Exception as tc_err:
                st.error(f"CSV processing failed: {tc_err}")
                import traceback
                st.caption(traceback.format_exc())

        # ---- Before/After Training Comparison ----
        st.markdown("---")
        st.markdown("### Before / After Training Comparison")
        st.caption(
            "Compare heuristic (baseline) predictions vs. trained model predictions "
            "for the currently loaded molecule. Run this after training to see improvement."
        )

        _cmp_intent = st.session_state.get("last_intent") or {}
        _cmp_seq = _cmp_intent.get("sequence") if isinstance(_cmp_intent, dict) else None
        _cmp_name = _cmp_intent.get("name", "Current molecule") if isinstance(_cmp_intent, dict) else "No molecule"

        if _cmp_seq and len(_cmp_seq) > 50:
            if st.button(f"Compare Models for: {_cmp_name}", key="btn_compare_models",
                          type="primary"):
                with st.spinner("Running heuristic and trained-model predictions side by side..."):
                    _cmp_results = {}
                    try:
                        # 1. Heuristic baseline (always available)
                        from src.data_pipeline import extract_features_from_jain_row
                        _heur_row = {"Sequence": _cmp_seq, "Name": _cmp_name}
                        _bpf = extract_features_from_jain_row(_heur_row)
                        if _bpf:
                            # Use stoichiometric assembly MW when available
                            # (accounts for 2×HC + 2×LC + glycans)
                            _heur_mw = _bpf.get("mw_kda")
                            _stoich_props = _cmp_intent.get("stoichiometric_properties")
                            if _stoich_props and _stoich_props.get("mw_kda_assembled"):
                                _heur_mw = _stoich_props["mw_kda_assembled"]
                            _cmp_results["Heuristic"] = {
                                "pI": _bpf.get("pI"),
                                "MW (kDa)": _heur_mw,
                                "Hydrophobicity": round(_bpf.get("hydrophobicity", 0.35), 4),
                                "Deamidation Sites": _bpf.get("deam_sites"),
                                "Oxidation Sites": _bpf.get("ox_sites"),
                                "Source": "Biopython + sequence analysis",
                            }
                        else:
                            _cmp_results["Heuristic"] = {"Error": "Could not extract features from sequence"}
                    except Exception as _he:
                        _cmp_results["Heuristic"] = {"Error": str(_he)}

                    try:
                        # 2. Trained unified model (if available)
                        import os as _os_cmp
                        _um_path = _os_cmp.path.join(str(ROOT), "models", "unified_multitask_best.pt")
                        if _os_cmp.path.exists(_um_path):
                            import torch
                            from src.esm2_hybrid_encoder import ESM2HybridEncoder
                            from src.unified_multitask_model import UnifiedMultiTaskModel, UNIFIED_TASKS

                            # Load checkpoint and detect which tasks were trained
                            _ckpt = torch.load(_um_path, map_location="cpu", weights_only=True)
                            _saved_tasks = sorted(set(
                                k.split(".")[1] for k in _ckpt.keys()
                                if k.startswith("heads.")
                            ))
                            _model_tasks = _saved_tasks if _saved_tasks else UNIFIED_TASKS

                            # Detect hidden_dim from checkpoint to avoid size mismatch
                            _hidden_dim = 256  # default fallback
                            if "encoder.esm2_proj.0.weight" in _ckpt:
                                _hidden_dim = _ckpt["encoder.esm2_proj.0.weight"].shape[0]
                            _enc = ESM2HybridEncoder(hidden_dim=_hidden_dim)
                            _mdl = UnifiedMultiTaskModel(
                                encoder=_enc, encoder_dim=_hidden_dim, tasks=_model_tasks,
                            )
                            _load_result = _mdl.load_state_dict(_ckpt, strict=False)
                            _mdl.eval()

                            hc = _cmp_seq[:451] if len(_cmp_seq) > 451 else _cmp_seq
                            lc = _cmp_seq[451:] if len(_cmp_seq) > 451 else _cmp_seq[:100]
                            _preds = _mdl.predict_numpy([hc], [lc])
                            import datetime as _dt_cmp
                            _mtime = _os_cmp.path.getmtime(_um_path)
                            _date = _dt_cmp.datetime.fromtimestamp(_mtime).strftime("%Y-%m-%d %H:%M")
                            _cmp_results["Unified Model"] = {
                                k: round(v, 4) if isinstance(v, float) else v
                                for k, v in _preds.items()
                            }
                            _cmp_results["Unified Model"]["Source"] = f"Trained {_date} ({len(_model_tasks)} tasks)"
                            if _load_result.missing_keys:
                                _cmp_results["Unified Model"]["Note"] = f"{len(_load_result.missing_keys)} weights initialized from scratch"
                        else:
                            _cmp_results["Unified Model"] = {"Status": "Not trained yet — train in Section 6 above"}
                    except Exception as _ume:
                        _cmp_results["Unified Model"] = {"Error": str(_ume)}

                    try:
                        # 3. XGBoost wetlab model (if available)
                        from src.ml_predictor import load_persisted_models as _load_ml, get_wetlab_model
                        _load_ml()  # ensure persisted models are loaded
                        _xgb = get_wetlab_model()
                        if _xgb is not None and _xgb.trained:
                            # Build 7-dim feature vector from heuristic features
                            # predict_single() expects np.ndarray, NOT a string
                            import numpy as _np_xgb
                            if _bpf:
                                _xgb_feats = _np_xgb.array([[
                                    float(_bpf.get("pI", 7.0)),
                                    float(_bpf.get("mw_kda", 50.0)),
                                    float(_bpf.get("deam_sites", 0)),
                                    float(_bpf.get("ox_sites", 0)),
                                    float(_bpf.get("acidic_residues", 0)),
                                    float(_bpf.get("basic_residues", 0)),
                                    float(_bpf.get("hydrophobicity", 0.35)),
                                ]])
                                _xgb_pred = _xgb.predict_single(_xgb_feats)
                                _cmp_results["XGBoost"] = {
                                    k: round(v, 4) if isinstance(v, (int, float)) else v
                                    for k, v in _xgb_pred.items()
                                }
                                _cmp_results["XGBoost"]["Source"] = "XGBoost (wetlab)"
                            else:
                                _cmp_results["XGBoost"] = {"Status": "Features unavailable — load molecule first"}
                        else:
                            _cmp_results["XGBoost"] = {"Status": "Not trained — upload wet-lab CSV data and train in Data & Model Training tab"}
                    except Exception as _xgb_err:
                        _cmp_results["XGBoost"] = {"Status": f"Not available ({type(_xgb_err).__name__}: {_xgb_err})"}

                    st.session_state["model_comparison"] = _cmp_results

            _cmp = st.session_state.get("model_comparison")
            if _cmp:
                import pandas as pd
                # Build comparison table
                _all_keys = set()
                for _mv in _cmp.values():
                    if isinstance(_mv, dict):
                        _all_keys.update(_mv.keys())
                _all_keys = sorted(_all_keys - {"Source", "Status", "Error"})
                _cmp_rows = []
                for _metric in _all_keys:
                    _row = {"Metric": _metric}
                    for _model_name, _mv in _cmp.items():
                        if isinstance(_mv, dict):
                            val = _mv.get(_metric, "—")
                            if isinstance(val, float):
                                val = f"{val:.4f}"
                            _row[_model_name] = str(val) if val is not None else "—"
                    _cmp_rows.append(_row)
                if _cmp_rows:
                    st.dataframe(pd.DataFrame(_cmp_rows), use_container_width=True, hide_index=True)

                # Show model sources
                for _mn, _mv in _cmp.items():
                    if isinstance(_mv, dict):
                        _src = _mv.get("Source") or _mv.get("Status") or _mv.get("Error", "")
                        st.caption(f"**{_mn}**: {_src}")
                st.caption(
                    "Note: The unified multi-task model (ESM-2 t12 + XGBoost) is research-grade. "
                    "Cross-validation shows moderate correlation for most targets. "
                    "See the Benchmark section for detailed performance metrics."
                )
        else:
            st.caption("Load a molecule first (via Discovery page) to compare predictions.")

        if not st.session_state.get("tc_uploaded_csv"):
            # No file uploaded yet -- show instructions
            st.info(
                "Upload a CSV file with antibody sequence data to get started. "
                "The harmonizer supports dual-chain (VH + VL) and single-sequence formats."
            )
            st.markdown("#### Supported Column Names")
            _help_data = {
                "Category": ["Molecule ID", "Heavy Chain (VH)", "Light Chain (VL)",
                             "Single Sequence", "Target Assay"],
                "Accepted Names": [
                    "name, antibody, id, clone",
                    "vh, heavy chain, heavy_chain, hc",
                    "vl, light chain, light_chain, lc",
                    "sequence, seq, amino_acid",
                    "ac-sins, bvp, titer, aggregation, stability, viscosity, tm, hmw",
                ],
            }
            import pandas as pd
            st.dataframe(pd.DataFrame(_help_data), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    #  NISTmAb Benchmark (collapsed expander at bottom of Tab 1)
    # ------------------------------------------------------------------
        st.markdown("---")
        with st.expander("NISTmAb Benchmark (RM 8671)", expanded=True):
            st.caption(
                "Validate the ProtePilot pipeline against the industry gold standard: "
                "NIST Reference Material 8671, an extensively characterized humanized IgG1k."
            )

            # Reference info
            st.markdown(
                "**NISTmAb** is a humanized IgG1k produced in NS0 cells (NIST RM 8671). "
                "Published values: pI ~9.15, intact mass ~148.038 kDa, aggregation <1%, Tm ~71C."
            )

            if st.button("Run NISTmAb Benchmark", type="primary",
                          key="btn_nistmab_run"):
                with st.spinner("Validating full pipeline against NISTmAb RM 8671..."):
                    try:
                        from src.nistmab_benchmark import run_nistmab_validation
                        _nist_result = run_nistmab_validation()
                        st.session_state["nistmab_result"] = _nist_result
                    except Exception as nist_err:
                        st.error(f"Benchmark failed: {nist_err}")
                        import traceback
                        st.caption(traceback.format_exc())

            _nist_result = st.session_state.get("nistmab_result")
            if _nist_result is not None:
                # Grade banner
                from src.ui_colors import COLORS as _GC_C
                _grade_colors = {"Excellent": _GC_C["pass"]["primary"], "Good": _GC_C["info"]["primary"], "Needs Improvement": _GC_C["caution"]["primary"]}
                _gc = _grade_colors.get(_nist_result.overall_grade, _GC_C["neutral"]["primary"])
                st.markdown(
                    f'<div style="background:{_gc}15;border-left:4px solid {_gc};'
                    f'padding:12px;border-radius:8px;margin:8px 0;">'
                    f'<span style="font-size:1.2rem;font-weight:700;color:{_gc};">'
                    f'Grade: {_nist_result.overall_grade}</span>'
                    f'<span style="margin-left:16px;font-size:0.9rem;">'
                    f'{_nist_result.n_passed}/{_nist_result.n_total} passed '
                    f'({_nist_result.pass_rate:.0%})</span></div>',
                    unsafe_allow_html=True,
                )

                # Comparison table
                _nist_rows = []
                for m in _nist_result.metrics:
                    _nist_rows.append({
                        "Status": "PASS" if m.within_range else "FAIL",
                        "Metric": m.metric_name,
                        "Predicted": str(m.predicted_value),
                        "Literature": str(m.literature_value),
                        "Unit": m.unit,
                        "Error %": f"{m.error_pct:.1f}%" if m.error_pct is not None else "\u2014",
                        "Method": getattr(m, "model_source", "heuristic"),
                        "Notes": m.notes[:80] + "..." if len(m.notes) > 80 else m.notes,
                    })
                try:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(_nist_rows), use_container_width=True, hide_index=True)
                except ImportError:
                    for row in _nist_rows:
                        st.text(f"  [{row['Status']}] {row['Metric']}: {row['Predicted']} vs {row['Literature']}")

                # Detailed notes + diagnostics per metric (expandable)
                with st.expander("Detailed Notes & Diagnostics", expanded=True):
                    for m in _nist_result.metrics:
                        _status_label = "[PASS]" if m.within_range else "[FAIL]"
                        _status_color = "#10B981" if m.within_range else "#EF4444"
                        st.markdown(
                            f'<span style="color:{_status_color};font-weight:700;">{_status_label}</span> '
                            f'**{m.metric_name}** ({getattr(m, "model_source", "heuristic")})',
                            unsafe_allow_html=True,
                        )
                        if m.notes:
                            st.caption(m.notes)
                        _diag = getattr(m, "diagnostic", "")
                        if _diag:
                            st.markdown(
                                f'<div style="background:#FFF7ED;border-left:4px solid #F59E0B;'
                                f'padding:8px 12px;margin:4px 0 12px 0;border-radius:6px;'
                                f'font-size:0.85em;color:#92400E;">'
                                f'{_diag}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown("---")

    # ------------------------------------------------------------------
    #  Tab 2: Continuous Learning & Accuracy Tracker
    # ------------------------------------------------------------------
    with _aitc_tab2:
        st.caption(
            "Track how expert labels improve model accuracy over retraining cycles. "
            "Add labels via the Copilot Chat or upload historical data, then retrain."
        )
        render_continuous_learning_tab()

    # ------------------------------------------------------------------
    #  Tab 3: Molecule Classifier & OOD Detector Training
    # ------------------------------------------------------------------
    with st.expander("\u2699\ufe0f Advanced: Molecule Classifier & OOD Detection", expanded=False):
        st.caption(
            "Train the molecule classifier and OOD detector from public datasets. "
            "Trained models automatically integrate into the classification pipeline as a "
            "'second opinion' alongside the rule-based classifier."
        )

        # ── Current model status ──
        _clf_meta_path = os.path.join("models", "classifier", "classifier_metadata.json")
        _ood_meta_path = os.path.join("models", "ood_detector", "ood_metadata.json")

        _mc1, _mc2 = st.columns(2)
        with _mc1:
            if os.path.exists(_clf_meta_path):
                import json as _json_tc
                _clf_meta = _json_tc.load(open(_clf_meta_path))
                st.metric("Classifier Accuracy", f"{_clf_meta.get('test_accuracy', 0):.1%}")
                st.caption(
                    f"Model: {_clf_meta.get('model_type', '?')} | "
                    f"Classes: {len(_clf_meta.get('classes', []))} | "
                    f"Trained: {_clf_meta.get('timestamp', '?')}"
                )
            else:
                st.metric("Classifier", "Not Trained")
                st.caption("No molecule classifier model found.")

        with _mc2:
            if os.path.exists(_ood_meta_path):
                import json as _json_tc
                _ood_meta = _json_tc.load(open(_ood_meta_path))
                st.metric("OOD Detector F1", f"{_ood_meta.get('test_f1', 0):.3f}")
                st.caption(
                    f"Threshold: {_ood_meta.get('threshold', 0):.2f} | "
                    f"Trained: {_ood_meta.get('timestamp', '?')}"
                )
            else:
                st.metric("OOD Detector", "Not Trained")
                st.caption("No OOD detector model found.")

        st.markdown("---")

        # ── Train Classifier ──
        st.markdown("### Train Molecule Classifier")
        st.markdown(
            "Trains a logistic regression classifier on all available data sources "
            "(Jain-137, TheraSAbDab, CoV-AbDab, DRAMP, ThPD, and synthetic entries). "
            "The trained model provides a probability-based 'second opinion' alongside the "
            "rule-based classifier. **It never silently overrides** the rule-based classification."
        )

        # ── Upload additional training data ──
        _tc3_upload = st.file_uploader(
            "Upload additional training data (optional)",
            type=["csv", "xlsx", "tsv"],
            key="tc3_data_uploader",
            help=(
                "Upload CSV/XLSX files with antibody or peptide sequences to expand the "
                "training dataset. Place them in the data/ folder or upload here. "
                "The harmonizer auto-detects column formats from Jain-137, TheraSAbDab, "
                "CoV-AbDab, DRAMP, ThPD, SAbDab, and IMGT/mAb-DB."
            ),
        )
        if _tc3_upload is not None:
            import shutil
            _upload_dest = os.path.join("data", _tc3_upload.name)
            with open(_upload_dest, "wb") as _uf:
                _uf.write(_tc3_upload.getbuffer())
            st.success(f"Saved `{_tc3_upload.name}` to data/ folder. Click **Train Molecule Classifier** to include it.")

        if st.button("Train Molecule Classifier", type="primary", key="btn_train_classifier"):
            with st.spinner("Harmonizing data and training classifier..."):
                try:
                    from src.training.data_harmonizer import harmonize as _harmonize_fn
                    from src.training.classifier_trainer import train_classifier as _train_fn

                    _harm_df = _harmonize_fn(data_dir="data", output_path="data/training/classifier_data.csv")

                    # Show data source summary
                    if "source" in _harm_df.columns:
                        _src_counts = _harm_df["source"].value_counts()
                        _src_text = " | ".join(f"{src}: {cnt:,}" for src, cnt in _src_counts.items())
                        st.info(f"Harmonized **{len(_harm_df):,}** rows from: {_src_text}")

                    _tc_result = _train_fn(
                        data_path="data/training/classifier_data.csv",
                        artifact_dir="models/classifier",
                        seed=42,
                    )
                    st.success(
                        f"Classifier trained: **{_tc_result.test_accuracy:.1%}** accuracy "
                        f"({_tc_result.n_classes} classes, {_tc_result.n_train} training samples). "
                        f"Improvement over rule-based: **{_tc_result.improvement_vs_rule:+.1%}**"
                    )
                    # Show per-class F1
                    if _tc_result.per_class_f1:
                        _f1_text = " | ".join(
                            f"{c}: {f1:.2f}" for c, f1 in sorted(_tc_result.per_class_f1.items())
                        )
                        st.caption(f"Per-class F1: {_f1_text}")
                    st.caption("Tip: Retrain the OOD Detector below to update the in-distribution baseline.")
                    st.rerun()
                except Exception as _tc_err:
                    st.error(f"Training failed: {_tc_err}")

        # ── Train OOD Detector ──
        st.markdown("### Train OOD Detector")
        st.markdown(
            "Learns the biophysical feature distribution of known biologics. "
            "Sequences outside this distribution are flagged as OOD, "
            "capping classification confidence to prevent overconfident predictions. "
            "**Retrain after updating the classifier** to keep the OOD baseline current."
        )

        if st.button("Train OOD Detector", type="primary", key="btn_train_ood"):
            with st.spinner("Generating OOD sequences and training detector..."):
                try:
                    from src.training.ood_trainer import train_ood_detector as _ood_fn
                    _ood_result = _ood_fn(
                        data_path="data/training/classifier_data.csv",
                        artifact_dir="models/ood_detector",
                    )
                    st.success(
                        f"OOD detector trained: F1 = **{_ood_result.test_f1:.3f}**, "
                        f"threshold = {_ood_result.threshold:.2f} "
                        f"({_ood_result.n_in_distribution} in-distribution, "
                        f"{_ood_result.n_ood} OOD samples)"
                    )
                    st.rerun()
                except Exception as _ood_err:
                    st.error(f"OOD training failed: {_ood_err}")

        # ── Benchmark Evaluator ──
        st.markdown("### Benchmark Evaluation")
        st.markdown(
            "Compare the trained classifier vs rule-based baseline on a fixed panel of "
            "7 reference molecules (NISTmAb, bispecific, Fc-fusion, nanobody, peptide, 2 OOD)."
        )
        if st.button("Run Benchmark Evaluation", key="btn_run_benchmark"):
            with st.spinner("Running benchmark comparison..."):
                try:
                    from src.training.benchmark_evaluator import run_benchmark, post_training_selftest
                    _bm = run_benchmark(artifact_dir="models/classifier")
                    _st_checks = post_training_selftest(artifact_dir="models/classifier")

                    # Show per-molecule results
                    import pandas as pd
                    _bm_rows = []
                    for _mol in _bm.results:
                        _bm_rows.append({
                            "Molecule": _mol.get("name", "?"),
                            "Expected": _mol.get("expected", "?"),
                            "Rule-Based": _mol.get("rule_class", "?"),
                            "Trained": _mol.get("trained_class", "?"),
                            "Confidence": _mol.get("trained_confidence", "?"),
                            "Changed": "Yes" if _mol.get("rule_class") != _mol.get("trained_class") else "",
                        })
                    if _bm_rows:
                        st.dataframe(pd.DataFrame(_bm_rows), use_container_width=True, hide_index=True)

                    # Selftest checks
                    _n_pass = sum(1 for v in _st_checks.values() if v is True)
                    _n_total = len(_st_checks)
                    if _n_pass == _n_total:
                        st.success(f"All {_n_total} selftest checks passed.")
                    else:
                        st.warning(f"{_n_pass}/{_n_total} selftest checks passed.")
                        for _ck, _cv in _st_checks.items():
                            if not _cv:
                                st.caption(f"  [FAIL] {_ck}")
                except Exception as _bm_err:
                    st.error(f"Benchmark failed: {_bm_err}")

        # ── Upload Pre-Trained Model Artifacts ──
        st.markdown("---")
        st.markdown("### Upload Pre-Trained Model Artifacts")
        st.caption(
            "Upload a pre-trained classifier or OOD detector to bypass training. "
            "Expects paired files: .npz (weights) + .json (metadata)."
        )
        _art_c1, _art_c2 = st.columns(2)
        with _art_c1:
            st.markdown("**Classifier**")
            _clf_npz = st.file_uploader("Classifier .npz", type=["npz"], key="upload_clf_npz")
            _clf_json = st.file_uploader("Classifier .json", type=["json"], key="upload_clf_json")
            if _clf_npz is not None and _clf_json is not None:
                if st.button("Install Classifier", key="btn_install_clf"):
                    _clf_dir = os.path.join("models", "classifier")
                    os.makedirs(_clf_dir, exist_ok=True)
                    with open(os.path.join(_clf_dir, "classifier_model.npz"), "wb") as _f:
                        _f.write(_clf_npz.getbuffer())
                    with open(os.path.join(_clf_dir, "classifier_metadata.json"), "wb") as _f:
                        _f.write(_clf_json.getbuffer())
                    st.success("Classifier artifacts installed. Refresh to see updated metrics.")
                    st.rerun()
        with _art_c2:
            st.markdown("**OOD Detector**")
            _ood_npz = st.file_uploader("OOD .npz", type=["npz"], key="upload_ood_npz")
            _ood_json = st.file_uploader("OOD .json", type=["json"], key="upload_ood_json")
            if _ood_npz is not None and _ood_json is not None:
                if st.button("Install OOD Detector", key="btn_install_ood"):
                    _ood_dir = os.path.join("models", "ood_detector")
                    os.makedirs(_ood_dir, exist_ok=True)
                    with open(os.path.join(_ood_dir, "ood_detector.npz"), "wb") as _f:
                        _f.write(_ood_npz.getbuffer())
                    with open(os.path.join(_ood_dir, "ood_metadata.json"), "wb") as _f:
                        _f.write(_ood_json.getbuffer())
                    st.success("OOD detector artifacts installed. Refresh to see updated metrics.")
                    st.rerun()


