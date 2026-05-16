"""
agents.py  ·  ProtePilot — Milestone 27
===========================================================
Multi-Agent Architecture: LLM-Callable Tool Layer + Deterministic Pipeline Dispatcher

Version   : 23.0 (Analytical QC + Stability + Pareto)
Author    : Di (ProtePilot)
Depends   : PropertyMapper (M2/v7.0), cadet_engine (M1), biopython, torch, shap,
             xgboost (optional), transformers (optional), bispecific_engine (M11),
             ht_screening (M16), formulation_twin (M17), report_generator (M18),
             purification_optimizer (M19), upstream_twin (M20), cogs_twin (M21),
             llm_copilot (M21), multi_agent_board (M22), regulatory_filer (M22/M24),
             immunogenicity_twin (M23), scaleup_twin (M23),
             nistmab_benchmark (M24), data_pipeline (M15/M24),
             structural_twin (M25),
             uncertainty_engine (M26), lab_automation (M26),
             analytical_qc_twin (M27), stability_twin (M27), pareto_optimizer (M27),
             joblib (M19), openai (optional, M21-M24), PyPDF2 (optional, M24)

Architecture
------------------------------------------------------------
Three-layer design:

    +---------------------------------------------+
    |  Layer 3 -- PharmaAgentManager               |
    |  (Workflow Engine / LLM Orchestrator)        |
    |  - run_ml_first_pipeline()                   |
    |  - run_developability_pipeline()   [M8]      |
    |  - run_bispecific_pipeline()       [M11]     |
    |  - run_ht_screening_pipeline()     [M16]     |
    |  - run_cmc_board_meeting()        [M22]     |
    |  - run_immunogenicity()          [M23]     |
    |  - run_scaleup_simulation()      [M23]     |
    |  - run_nistmab_validation()     [NEW M24] |
    |  - run_deterministic_pipeline()              |
    |  - [future] run_llm_pipeline() via LangChain |
    +---------------------------------------------+
    |  Layer 2 -- Standardized Tools               |
    |  - predict_physical_params()   [Tool 1]     |
    |  - run_chromatography_sim()    [Tool 2]     |
    |  - predict_ml_with_shap()      [Tool 3]     |
    |  - predict_developability()    [Tool 4]     |
    |  - predict_bispecific_sep()    [Tool 5]     |
    |  - run_ht_screen()             [Tool 6]     |
    |  - run_ada_assessment()        [Tool 7]     |
    |  - run_nistmab_benchmark()     [Tool 8] NEW |
    +---------------------------------------------+
    |  Layer 1 -- Core Engines                     |
    |  - PropertyMapper (M2/v7.0 ML-First+Bispec) |
    |  - CadetEngine    (M1)                      |
    |  - ChromatographyMLP (M7/v2.0)              |
    |  - ESM2Embedder   (M8)                      |
    |  - DevelopabilityPredictor (M8)              |
    |  - ValidationPlanner (M8)                    |
    |  - BispecificEngine  (M11)                   |
    |  - HTScreeningEngine (M16)                    |
    |  - PotencyPredictor  (M16)                    |
    |  - FormulationTwin   (M17)                     |
    |  - ReportGenerator   (M18)                     |
    |  - PurificationOptimizer (M19)                  |
    |  - ModelPersistence  (M19)                     |
    |  - UpstreamTwin      (M20)                     |
    |  - COGSTwin          (M21)                      |
    |  - LLMCoPilot        (M21)                      |
    |  - MultiAgentBoard   (M22)                       |
    |  - RegulatoryFiler   (M22)                       |
    |  - ImmunogenicityTwin (M23)                       |
    |  - ScaleUpTwin        (M23)                       |
    |  - NISTmAbBenchmark   (M24)                       |
    |  - LiteratureRAG      (M24)                       |
    |  - StructuralTwin     (M25)                       |
    |  - UncertaintyEngine  (M26)                       |
    |  - LabAutomation      (M26)                       |
    |  - AnalyticalQCTwin   (M27) NEW                  |
    |  - StabilityTwin      (M27) NEW                  |
    |  - ParetoOptimizer    (M27) NEW                  |
    +---------------------------------------------+

New in v23.0 (Analytical QC + Stability + Pareto — M27)
------------------------------------------------------------
  - analytical_qc_twin.py: cIEF charge variant, CE-SDS purity, Glycan Profile simulators
  - stability_twin.py: Arrhenius kinetic degradation, 5°C/40°C ICH stability projections
  - pareto_optimizer.py: NSGA-II Pareto dominance, crowding distance, multi-objective frontier
  - Excipient stabilization modeling (sucrose, trehalose, PS80, arginine)
  - Pareto frontier scatter in Discovery & HT Screening page
  - Analytical QC section in Analytical & Mass Spec page (cIEF electropherogram, glycan pie)
  - Stability projection in Preclinical & Formulation page (2x2 time-series subplot)

New in v22.0 (Active Learning & DoE Automation — M26)
------------------------------------------------------------
  - uncertainty_engine.py: MC Dropout + Ensemble Variance uncertainty quantification
  - Active Learning: Expected Improvement acquisition function for experiment selection
  - Virtual mutant library generation (in-silico point mutagenesis)
  - lab_automation.py: 96-well plate layout, Hamilton/Tecan robot worklist CSV
  - NISTmAb positive controls, buffer blanks, standard curve wells
  - Active Learning Center page: scan mutants, select experiments, export robot files
  - Upload Results & Retrain: parse wet-lab CSV and trigger continual learning

New in v21.0 (3D Structural Twin & SASA Liability Assessment — M25)
------------------------------------------------------------
  - structural_twin.py: ESMFold API structure prediction + Shrake-Rupley SASA
  - 3D liability filtering: buried motifs (<10 Å² SASA) reclassified as safe
  - Empirical SASA fallback when Bio.PDB / ESMFold unavailable
  - SASA-based liability assessment panel in Molecular Characterization tab
  - Interactive 3D PDB viewer with exposed liabilities highlighted in red

New in v20.0 (Deep eCTD + NISTmAb Benchmark + Literature RAG — M24)
------------------------------------------------------------
  - regulatory_filer.py v2.0: deep narrative eCTD with paragraph-level scientific prose
  - nistmab_benchmark.py: NISTmAb RM 8671 gold standard pipeline validation
  - data_pipeline.py v3.0: literature PDF extraction + biophysical metric parsing
  - NISTmAb Benchmark Validation page with side-by-side comparison table
  - Literature Data Extraction (RAG foundation) with PDF upload and text parsing

New in v19.0 (Immunogenicity/ADA + Tech Transfer + Docker — M23)
------------------------------------------------------------
  - immunogenicity_twin.py: MHC-II binding scan, humanization score, ADA risk
  - scaleup_twin.py: tech transfer 2L→2000L (constant P/V, constant kLa)
  - Clinical & Immunogenicity (ADA) page with hotspot table and annotated sequence
  - Tech Transfer section in Upstream page with shear risk assessment
  - Dockerfile, docker-compose.yml, requirements.txt for enterprise deployment

New in v18.0 (Multi-Agent CMC Board + eCTD Filer — M22)
------------------------------------------------------------
  - multi_agent_board.py: 3-agent CMC board (Upstream, Downstream, Regulatory/QA)
  - regulatory_filer.py: Manufacturability & Risk Assessment auto-generator (Markdown + Word)
  - CMC Board & Regulatory page with agent dialogue and risk banners
  - Download risk assessment drafts as .md or .docx directly from the app

New in v17.0 (COGS Digital Twin + LLM Co-Pilot — M21)
------------------------------------------------------------
  - cogs_twin.py: commercial COGS calculator ($/gram purified API)
  - llm_copilot.py: context-aware scientific chat (OpenAI or mock)
  - Manufacturability & COGS page with pie chart cost breakdown
  - Global molecule context preserved across workshop pages

New in v16.0 (Enterprise Modular UI + Upstream Twin — M20)
------------------------------------------------------------
  - app.py v23.0: sidebar radio navigation with 6 Workshop pages
  - upstream_twin.py: CHO Fed-Batch ODE bioreactor simulation
  - HT Data Viewer: st.dataframe() for candidate CSV inspection
  - Factory Reset: delete trained models and revert to baseline
  - DoE moved to dedicated Downstream Purification page

New in v15.0 (Model Persistence + GoSilico DoE — M19)
------------------------------------------------------------
  - ml_predictor.py v6.0: joblib/torch.save persistence for all ML models
  - purification_optimizer.py: GoSilico-style In-Silico DoE grid search
  - Model status API: get_model_status(), load_persisted_models()
  - Auto-save on training, auto-load on startup

New in v14.0 (IND-Ready Report Engine — M18)
------------------------------------------------------------
  - report_generator.py: Word .docx executive report generation
  - Headless Plotly chart capture via kaleido (PNG embedding)
  - Professional IND-ready report sections with chart images
  - Sidebar Generate Executive Report button + download

New in v13.0 (Formulation Digital Twin — M17)
------------------------------------------------------------
  - Formulation Digital Twin integration (formulation_twin.py)
  - Buffer pH, Buffer Type, Excipient simulation
  - Henderson-Hasselbalch net charge feedback loop
  - Real-time Developability Score adjustment from formulation

New in v12.0 (Early Discovery HT Screening — M16)
------------------------------------------------------------
  - Tool 6: run_ht_screen() — High-Throughput Virtual Screening
  - Bulk processing of candidate CSVs (100s-1000s of sequences)
  - ESM-2 embeddings + Biopython features per candidate
  - XGBoost Developability & Potency scoring
  - Magic Quadrant classification (Star/Developable/Potent/Risky)
  - run_ht_screening_pipeline() in PharmaAgentManager
  - PotencyPredictor integration from ml_predictor v5.0

New in v8.0 (Bispecific Antibody / Complex Modality Support)
------------------------------------------------------------
  - Tool 5: predict_bispecific_separation() — 3-species homodimer analysis
  - Builds AA (Homodimer A), AB (Heterodimer Target), BB (Homodimer B) species
  - Computes pI/MW/hydrophobicity per species via Biopython
  - Maps each species independently to SMA parameters
  - Estimates retention times and peak widths (Yamamoto theory)
  - Calculates Resolution (Rs) between AB-AA and AB-BB
  - Risk assessment: High/Medium/Low homodimer co-elution risk
  - Actionable recommendations for charge asymmetry engineering
  - Chromatogram data generation for 3-component visualization
  - run_bispecific_pipeline() in PharmaAgentManager

New in v7.0 (End-to-End Developability Digital Twin)
------------------------------------------------------------
  - Tool 4: predict_developability_risk() — pLM (ESM-2) embedding + XGBoost
  - Predicts aggregation risk, thermal stability, viscosity risk
  - Composite Developability Score with grade (Low/Medium/High)
  - SHAP TreeExplainer for actionable insights
  - Analytical validation plan generation (SEC, DSF, MAM, etc.)
  - Graceful fallback: rule-based heuristics if xgboost/transformers unavailable

New in v6.0 (ML-First Override)
------------------------------------------------------------
  - Tool 1 now accepts ml_override parameter: {"ka": float, "nu": float}
  - When ML predictions are available, they OVERRIDE static PropertyMapper formulas
  - Static v5.0 formulas are retained as FALLBACK only
  - Pipeline flow: Tool 3 (ML predict) -> Tool 1 (with override) -> Tool 2 (simulate)
  - Feature expansion: hydrophobicity (7th input via Biopython GRAVY)
  - RT-targeted training: synthetic data calibrated for 15-20 min elution window

New in v5.0
------------------------------------------------------------
  - Tool 3: predict_ml_with_shap() -- PyTorch MLP inference + SHAP explainability
  - ChromatographyMLP trained on synthetic data from PropertyMapper v5.0 physics
  - SHAP KernelExplainer provides feature attribution for ka/nu predictions

Tool Protocol
------------------------------------------------------------
Each Tool function follows these design specifications:
  1. Pure Python function; parameters and return values are JSON-serializable primitives.
  2. Contains a complete docstring (for LLM function-calling schema auto-extraction).
  3. Returns standardized dict: {"status": "success"|"error", "data": {...}, "message": str}
  4. Exceptions are caught and serialized as error responses; never leak to callers.
  5. Decorator placeholder: can seamlessly add @tool (LangChain) or @function (OpenAI).

References
------------------------------------------------------------
  LangChain Tool Protocol: https://python.langchain.com/docs/modules/tools/
  OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
"""

from __future__ import annotations

import logging
import time as _time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("ProtePilot.Agents")


# ===========================================================================
# Tool Registry -- Extensible Tool Metadata Framework
# ===========================================================================

@dataclass
class ToolSpec:
    """
    Standardized tool description metadata.

    Aligned with LangChain @tool / OpenAI function-calling schema;
    can be directly exported as JSON Schema for LLM consumption.
    """
    name:        str
    description: str
    func:        Callable
    parameters:  Dict[str, Any] = field(default_factory=dict)
    category:    str = "general"


# Global Tool Registry
_TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    category: str = "general",
):
    """Tool registration decorator."""
    def decorator(func: Callable) -> Callable:
        spec = ToolSpec(
            name=name,
            description=description,
            func=func,
            parameters=parameters or {},
            category=category,
        )
        _TOOL_REGISTRY[name] = spec
        log.debug("Tool registered: %s [%s]", name, category)
        return func
    return decorator


def get_tool(name: str) -> Optional[ToolSpec]:
    """Retrieve a registered Tool by name."""
    return _TOOL_REGISTRY.get(name)


def list_tools() -> List[ToolSpec]:
    """List all registered Tools."""
    return list(_TOOL_REGISTRY.values())


def export_tool_schemas() -> List[Dict[str, Any]]:
    """Export all Tool JSON Schemas in OpenAI function-calling format."""
    schemas = []
    for spec in _TOOL_REGISTRY.values():
        schemas.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        })
    return schemas


# ===========================================================================
# Standard Response Wrappers
# ===========================================================================

def _success(data: Dict[str, Any], message: str = "OK") -> Dict[str, Any]:
    """Construct a standard success response."""
    return {"status": "success", "data": data, "message": message}


def _error(message: str, details: str = "") -> Dict[str, Any]:
    """Construct a standard error response."""
    return {"status": "error", "data": {}, "message": message, "details": details}


# ===========================================================================
# M12: Super-Sequence Assembly Utility
# ===========================================================================

def assemble_super_sequence(
    chains: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a stoichiometry-aware super-sequence from multi-chain input.

    Takes a list of chain dicts (each with sequence, copy_number, name,
    chain_type) and constructs a single concatenated super-sequence that
    represents the TRUE molecular assembly. The global pI, MW, GRAVY, and
    hydrophobicity are computed from this super-sequence.

    Parameters
    ----------
    chains : List of chain dicts, each with:
        - "sequence": str (amino acid)
        - "copy_number": int (stoichiometric multiplier, default 1)
        - "name": str (chain label)
        - "chain_type": str ("HC", "LC", etc.)

    Returns
    -------
    dict : {
        "super_sequence": str,
        "total_length": int,
        "pI": float,
        "mw_kda": float,
        "gravy": float,
        "hydrophobicity": float (0-1 scale),
        "chains_summary": [{"name", "chain_type", "copy_number", "length"}],
        "stoichiometry": str (e.g., "HC(x2) + LC(x2)"),
    }
    """
    from src.analytical_twin import build_super_sequence

    super_seq = build_super_sequence(chains)
    if len(super_seq) < 10:
        return {"error": "Super-sequence too short"}

    # Compute biophysical properties (Biopython preferred, pure-Python fallback)
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        analysis = ProteinAnalysis(super_seq)
        pI = analysis.isoelectric_point()
        mw_da = analysis.molecular_weight()
        mw_kda = mw_da / 1000.0
        gravy = analysis.gravy()
    except ImportError:
        log.info("Biopython unavailable — using fallback biophysical calculations")
        from src.analytical_twin import calculate_sequence_mass
        mw_da = calculate_sequence_mass(super_seq, mass_type="average")
        mw_kda = mw_da / 1000.0
        # Fallback pI estimate (average of charged residue pKas)
        n_asp = super_seq.count("D") + super_seq.count("E")
        n_bas = super_seq.count("K") + super_seq.count("R") + super_seq.count("H")
        pI = 6.0 + (n_bas - n_asp) / max(len(super_seq), 1) * 8.0
        pI = max(4.0, min(12.0, pI))
        # Fallback GRAVY (Kyte-Doolittle)
        _kd = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
               "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
               "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
        gravy = sum(_kd.get(aa, 0.0) for aa in super_seq) / max(len(super_seq), 1)
    except Exception as e:
        log.error("ProteinAnalysis failed: %s", e)
        return {"error": str(e)}

    hydrophobicity = max(0.0, min(1.0, (gravy + 2.0) / 4.0))

    stoich_parts = []
    chain_summary = []
    for ch in chains:
        copies = max(1, int(ch.get("copy_number", 1)))
        name = ch.get("name", "Chain")
        ctype = ch.get("chain_type", "unknown")
        stoich_parts.append(f"{name}(x{copies})")
        chain_summary.append({
            "name": name, "chain_type": ctype,
            "copy_number": copies, "length": len(ch.get("sequence", "")),
        })

    return {
        "super_sequence": super_seq,
        "total_length": len(super_seq),
        "pI": round(pI, 2),
        "mw_kda": round(mw_kda, 1),
        "gravy": round(gravy, 3),
        "hydrophobicity": round(hydrophobicity, 3),
        "chains_summary": chain_summary,
        "stoichiometry": " + ".join(stoich_parts),
    }


# ===========================================================================
# Tool 1: Bio-Informatics Tool -- Protein Physical Parameter Prediction
#          v6.0: Accepts ml_override for ML-First dynamic steering
# ===========================================================================

@register_tool(
    name="predict_physical_params",
    description=(
        "Predict three-variant (acidic/main/basic) SMA model physical parameters "
        "for IEX chromatography from protein properties (pI, MW, hydrophobicity) "
        "and PTM modification site information. v6.0: Accepts ml_override dict "
        "with ML-predicted ka/nu to override static formulas. Static v5.0 formulas "
        "serve as fallback when ml_override is not provided."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name":        {"type": "string",  "description": "Protein name"},
            "pI":          {"type": "number",  "description": "Isoelectric point (pH units)"},
            "mw":          {"type": "number",  "description": "Molecular weight (kDa)"},
            "hydrophobicity": {"type": "number", "description": "Hydrophobicity [0,1]", "default": 0.35},
            "pH_working":  {"type": "number",  "description": "Working pH", "default": 7.0},
            "deam_sites":  {"type": "integer", "description": "Deamidation sites", "default": 1},
            "ox_sites":    {"type": "integer", "description": "Oxidation sites", "default": 1},
            "sequence":    {"type": "string",  "description": "Amino acid sequence (optional)", "default": ""},
            "ml_override": {"type": "object",  "description": "ML-predicted override {ka, nu} (optional)"},
        },
        "required": ["name", "pI", "mw"],
    },
    category="bioinformatics",
)
def predict_physical_params(
    name:            str,
    pI:              float,
    mw:              float,
    hydrophobicity:  float = 0.35,
    pH_working:      float = 7.0,
    deam_sites:      int   = 1,
    ox_sites:        int   = 1,
    sequence:        Optional[str] = None,
    ml_override:     Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Predict SMA three-variant physical parameters from protein properties and PTM profile.

    v6.0: When ml_override is provided ({"ka": float, "nu": float}), the ML-predicted
    values OVERRIDE the static v5.0 formulas. PropertyMapper applies only structural
    corrections (sigma, lambda, variant offsets) on top of the ML predictions.

    Parameters
    ----------
    name            : Protein name
    pI              : Isoelectric point (pH units), mAb typical range 7-9
    mw              : Molecular weight (kDa), mAb typical ~150
    hydrophobicity  : Hydrophobicity in [0, 1], default 0.35
    pH_working      : Working pH, default 7.0
    deam_sites      : Deamidation modification sites, default 1
    ox_sites        : Oxidation modification sites, default 1
    sequence        : Amino acid sequence string (optional)
    ml_override     : ML-predicted ka/nu override (optional, PRIMARY when provided)

    Returns
    -------
    dict : Standard response; data contains:
        - variants: {acidic: {ka, nu}, main: {ka, nu}, basic: {ka, nu}}
        - protein_input: Input parameter echo
        - source: "ml_override" or "static_v5"
    """
    try:
        from src.PropertyMapper import ProteinProperties, PropertyMapper

        # -- Build protein properties object --------------------------------
        ptm_profile = {
            "deamidation_sites": deam_sites,
            "oxidation_sites":   ox_sites,
        }

        protein = ProteinProperties(
            name           = name,
            pI             = pI,
            MW_kDa         = mw,
            hydrophobicity = hydrophobicity,
            pH_working     = pH_working,
            sequence       = sequence if sequence else None,
            ptm_profile    = ptm_profile,
        )

        # -- Run mapping (v6.0: ML-first override) -------------------------
        mapper = PropertyMapper()
        vp = mapper.map_variants(protein, ml_override=ml_override)

        # -- Serialize output (dict-based access) ---------------------------
        source = vp.get("source", "static_v5")
        variants_serialized = {
            "acidic": {"ka": vp["acidic"]["ka"], "nu": vp["acidic"]["nu"], "sigma": vp["acidic"]["sigma"]},
            "main":   {"ka": vp["main"]["ka"],   "nu": vp["main"]["nu"],   "sigma": vp["main"]["sigma"]},
            "basic":  {"ka": vp["basic"]["ka"],   "nu": vp["basic"]["nu"],  "sigma": vp["basic"]["sigma"]},
            "kd":     vp["kd"],
            "lambda_": vp["lambda_"],
            "c_fractions": list(vp["c_fractions"]),
        }

        # Build protein input echo
        protein_input = {
            "name": name, "pI": pI, "mw": mw,
            "hydrophobicity": hydrophobicity,
            "pH_working": pH_working,
            "ptm_profile": ptm_profile,
        }
        if sequence:
            protein_input["sequence_length"] = len(sequence)
            protein_input["has_sequence"] = True
        if ml_override:
            protein_input["ml_override"] = ml_override

        source_label = "ML OVERRIDE" if source == "ml_override" else "STATIC FALLBACK"
        log.info("Tool1 [predict_physical_params] -> %s: Three-variant parameters (%s)%s",
                 name, source_label,
                 f" [seq: {len(sequence)} aa]" if sequence else "")

        return _success(
            data={
                "variants": variants_serialized,
                "protein_input": protein_input,
                "source": source,
            },
            message=f"Three-variant physical parameter prediction complete: {name} ({source_label})",
        )

    except Exception as e:
        log.error("Tool1 failed: %s", e)
        return _error(
            message=f"predict_physical_params failed: {e}",
            details=traceback.format_exc(),
        )


# ===========================================================================
# Tool 2: Simulation & CQA Tool -- Chromatography Simulation & Quality Analysis
# ===========================================================================

@register_tool(
    name="run_chromatography_simulation",
    description=(
        "Run IEX chromatography simulation using CADET-Core SMA multi-component model. "
        "Accepts three-variant physical parameter dictionary, executes competitive adsorption "
        "simulation, and automatically computes CQA separation metrics (retention time, "
        "resolution Rs, area percentage)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "variants_dict": {
                "type": "object",
                "description": (
                    "Three-variant parameter dictionary, format: "
                    '{"acidic": {"ka":..., "nu":..., "sigma":...}, '
                    '"main": {...}, "basic": {...}, '
                    '"kd": 1000.0, "lambda_": 1200.0, '
                    '"c_fractions": [0.12, 0.80, 0.08]}'
                ),
            },
            "gradient_slope": {
                "type": "number",
                "description": "Salt gradient slope (mM/min), default 15.0",
                "default": 15.0,
            },
            "run_name": {
                "type": "string",
                "description": "Simulation filename prefix, auto-generated if empty",
                "default": "",
            },
        },
        "required": ["variants_dict"],
    },
    category="simulation",
)
def run_chromatography_simulation(
    variants_dict:  Dict[str, Any],
    gradient_slope: float = 15.0,
    run_name:       str   = "",
) -> Dict[str, Any]:
    """
    Run CADET multi-component competitive adsorption simulation and compute CQA.

    Parameters
    ----------
    variants_dict   : Three-variant parameter dictionary
    gradient_slope  : Salt gradient slope (mM/min), default 15.0
    run_name        : Simulation filename prefix

    Returns
    -------
    dict : Standard response; data contains:
        - cqa: {peaks, resolution, area_pct}
        - h5_path: Generated HDF5 file path
        - wall_time: Simulation wall time (s)
        - summary: Human-readable summary
    """
    try:
        from src.cadet_engine import CadetEngine, VariantParams, ProcessParams

        vparams = VariantParams.from_dict(variants_dict)
        process = ProcessParams(gradient_slope=gradient_slope)

        if not run_name:
            timestamp = _time.strftime("%Y%m%d_%H%M%S")
            run_name = f"agent_sim_{timestamp}"
        h5_filename = f"{run_name}.h5"

        engine = CadetEngine(workspace="data", engine_dir="engine")
        h5_path = engine.build_h5(h5_filename, vparams, process)
        result = engine.run_simulation(h5_path)

        cqa = result.compute_cqa()

        cqa_serialized = {
            "peaks": {},
            "resolution": {},
            "area_pct": {},
        }

        for comp_name, peak_info in cqa["peaks"].items():
            cqa_serialized["peaks"][comp_name] = {
                k: float(v) for k, v in peak_info.items()
            }

        for label, rs in cqa["resolution"].items():
            cqa_serialized["resolution"][label] = round(float(rs), 4)

        for comp_name, pct in cqa["area_pct"].items():
            cqa_serialized["area_pct"][comp_name] = round(float(pct), 2)

        summary_lines = [
            f"Simulation complete ({result.wall_time:.2f}s)",
            f"Retention times:",
        ]
        for comp in ("Acidic", "Main", "Basic"):
            p = cqa_serialized["peaks"][comp]
            summary_lines.append(
                f"  {comp:7s}: RT={p['rt_min']:.2f} min, "
                f"FWHM={p['fwhm_min']:.3f} min"
            )
        summary_lines.append("Resolution:")
        for label, rs in cqa_serialized["resolution"].items():
            quality = "Baseline resolved" if rs >= 1.5 else ("Partial overlap" if rs >= 0.8 else "Not resolved")
            summary_lines.append(f"  {label}: Rs={rs:.3f} ({quality})")

        log.info("Tool2 [run_chromatography_simulation] -> %s: CQA analysis complete", run_name)

        return _success(
            data={
                "cqa":       cqa_serialized,
                "h5_path":   str(h5_path),
                "wall_time": round(result.wall_time, 3),
                "summary":   "\n".join(summary_lines),
            },
            message=f"Simulation and CQA analysis complete: {run_name}",
        )

    except Exception as e:
        log.error("Tool2 failed: %s", e)
        return _error(
            message=f"run_chromatography_simulation failed: {e}",
            details=traceback.format_exc(),
        )


# ===========================================================================
# Tool 3: ML Predictor -- PyTorch Inference + SHAP Explainability
#          v2.0: 7 features (incl. hydrophobicity), RT-targeted
# ===========================================================================

@register_tool(
    name="predict_ml_with_shap",
    description=(
        "Run PyTorch MLP inference to predict IEX chromatographic parameters (ka, nu) "
        "from sequence-derived biophysical features (including hydrophobicity), then "
        "compute SHAP values for explainability. v2.0: RT-targeted training ensures "
        "predictions yield 15-20 min elution. Returns ML predictions suitable for "
        "overriding PropertyMapper static formulas."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pI":          {"type": "number",  "description": "Isoelectric point (pH units)"},
            "mw":          {"type": "number",  "description": "Molecular weight (kDa)"},
            "deam_sites":  {"type": "integer", "description": "Deamidation site count", "default": 1},
            "ox_sites":    {"type": "integer", "description": "Oxidation site count", "default": 1},
            "acidic_residues": {"type": "integer", "description": "Acidic residue count (D+E)", "default": 40},
            "basic_residues":  {"type": "integer", "description": "Basic residue count (K+R+H)", "default": 50},
            "hydrophobicity":  {"type": "number",  "description": "Hydrophobicity [0,1]", "default": 0.35},
            "sequence":        {"type": "string",  "description": "Amino acid sequence (optional, for GRAVY)", "default": ""},
        },
        "required": ["pI", "mw"],
    },
    category="ml_prediction",
)
def predict_ml_with_shap(
    pI:              float,
    mw:              float,
    deam_sites:      int   = 1,
    ox_sites:        int   = 1,
    acidic_residues: int   = 40,
    basic_residues:  int   = 50,
    hydrophobicity:  float = 0.35,
    sequence:        Optional[str] = None,
) -> Dict[str, Any]:
    """
    PyTorch MLP prediction with SHAP interpretability (v2.0).

    Trains (or uses cached) ChromatographyMLP on RT-targeted synthetic data,
    predicts ka/nu from 7 biophysical features (including hydrophobicity),
    and computes SHAP values for explainable AI.

    The returned prediction dict can be directly used as ml_override
    for predict_physical_params() to enable the ML-First pipeline.

    Parameters
    ----------
    pI              : Isoelectric point
    mw              : Molecular weight (kDa)
    deam_sites      : Deamidation site count
    ox_sites        : Oxidation site count
    acidic_residues : Acidic residue count (D+E)
    basic_residues  : Basic residue count (K+R+H)
    hydrophobicity  : Hydrophobicity [0,1], default 0.35
    sequence        : Amino acid sequence (optional, for GRAVY computation)

    Returns
    -------
    dict : Standard response; data contains:
        - prediction: {ka, nu, estimated_rt_min}
        - ml_override: {ka, nu} (ready for PropertyMapper override)
        - features: input feature values
        - feature_names: feature labels
        - shap: per-output SHAP attributions
        - model_info: architecture, training metrics
        - training_history: epoch-by-epoch loss
    """
    try:
        from src.ml_predictor import (
            extract_features, get_trained_model,
            compute_shap_values, compute_hydrophobicity_from_sequence,
            FEATURE_NAMES, estimate_rt_from_sma,
        )

        # Compute hydrophobicity from sequence if available
        if sequence and len(sequence) > 20:
            hydro = compute_hydrophobicity_from_sequence(sequence)
        else:
            hydro = hydrophobicity

        # Get or train model
        model, X_train, history = get_trained_model()

        # Build features (7 features now)
        features = extract_features(
            pI=pI, mw=mw,
            deam_sites=deam_sites, ox_sites=ox_sites,
            acidic_residues=acidic_residues,
            basic_residues=basic_residues,
            hydrophobicity=hydro,
        )

        # Predict
        prediction = model.predict_single(features)

        # SHAP
        shap_result = compute_shap_values(
            model=model,
            X_background=X_train,
            X_explain=features,
            max_background=80,
        )

        # Serialize SHAP values (convert numpy to lists for JSON)
        shap_serialized = {}
        for i, target_name in enumerate(["ka", "nu"]):
            sv = shap_result["shap_values"][i][0]  # first sample
            shap_serialized[target_name] = {
                "values": {fn: round(float(sv[j]), 6) for j, fn in enumerate(FEATURE_NAMES)},
                "base_value": float(
                    shap_result["base_values"][i]
                    if hasattr(shap_result["base_values"], "__len__")
                    else shap_result["base_values"]
                ),
            }

        # Build the ml_override dict ready for PropertyMapper
        ml_override_dict = {
            "ka": prediction["ka"],
            "nu": prediction["nu"],
        }
        # v7.3.2: Propagate validation R² for quality gating
        if "val_r2" in prediction:
            ml_override_dict["val_r2"] = prediction["val_r2"]

        log.info("Tool3 [predict_ml_with_shap] -> ka=%.4f, nu=%.3f, est_RT=%.1f min "
                 "(SHAP computed, ML override ready)",
                 prediction["ka"], prediction["nu"],
                 prediction.get("estimated_rt_min", 0.0))

        return _success(
            data={
                "prediction": prediction,
                "ml_override": ml_override_dict,
                "features": {fn: round(float(features[j]), 4)
                             for j, fn in enumerate(FEATURE_NAMES)},
                "shap": shap_serialized,
                "model_info": {
                    "architecture": "MLP (7->64->32->16->2)",
                    "n_training_samples": len(X_train),
                    "n_epochs": len(history),
                    "final_train_mse": history[-1]["train_loss"],
                    "final_val_mse": history[-1]["val_loss"],
                    "target_rt_window": "15-20 min",
                },
                "training_history": history,
            },
            message="ML prediction with SHAP explainability complete (v2.0 RT-targeted)",
        )

    except Exception as e:
        log.error("Tool3 failed: %s", e)
        return _error(
            message=f"predict_ml_with_shap failed: {e}",
            details=traceback.format_exc(),
        )


# ===========================================================================
# Tool 4: Developability Risk Assessment (M8)
#          pLM (ESM-2) Embedding + XGBoost + SHAP + Validation Planning
# ===========================================================================

@register_tool(
    name="predict_developability_risk",
    description=(
        "Predict end-to-end developability risk for a therapeutic antibody using "
        "Protein Language Model (ESM-2) embeddings + XGBoost multi-output regression. "
        "Returns aggregation risk, thermal stability, viscosity risk, composite "
        "developability score, actionable SHAP-based advice, and an analytical "
        "validation plan with recommended assays."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pI":          {"type": "number",  "description": "Isoelectric point (pH units)"},
            "mw":          {"type": "number",  "description": "Molecular weight (kDa)"},
            "hydrophobicity": {"type": "number", "description": "Hydrophobicity [0,1]", "default": 0.35},
            "deam_sites":  {"type": "integer", "description": "Deamidation site count", "default": 1},
            "ox_sites":    {"type": "integer", "description": "Oxidation site count", "default": 1},
            "acidic_residues": {"type": "integer", "description": "Acidic residue count (D+E)", "default": 40},
            "basic_residues":  {"type": "integer", "description": "Basic residue count (K+R+H)", "default": 50},
            "sequence":    {"type": "string",  "description": "Full amino acid sequence (optional)", "default": ""},
            "vh_sequence": {"type": "string",  "description": "Heavy chain variable region (optional)", "default": ""},
            "vl_sequence": {"type": "string",  "description": "Light chain variable region (optional)", "default": ""},
        },
        "required": ["pI", "mw"],
    },
    category="developability",
)
def predict_developability_risk(
    pI:              float,
    mw:              float,
    hydrophobicity:  float = 0.35,
    deam_sites:      int   = 1,
    ox_sites:        int   = 1,
    acidic_residues: int   = 40,
    basic_residues:  int   = 50,
    sequence:        Optional[str] = None,
    vh_sequence:     Optional[str] = None,
    vl_sequence:     Optional[str] = None,
    feature_set: Any = None,
    molecule_class:  Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end developability risk assessment (Milestone 8).

    Pipeline:
      1. Generate pLM embedding (ESM-2 or mock fallback)
      2. Build biophysical feature vector (7-dim)
      3. Predict risks with XGBoost (or rule-based fallback)
      4. Compute composite Developability Score
      5. Generate SHAP explanations
      6. Generate actionable engineering advice
      7. Generate analytical validation plan

    Parameters
    ----------
    pI, mw, hydrophobicity : Core biophysical properties
    deam_sites, ox_sites   : PTM site counts
    acidic_residues, basic_residues : Charged residue counts
    sequence               : Full amino acid sequence (for embedding)
    vh_sequence, vl_sequence : Separated VH/VL chains (preferred)

    Returns
    -------
    dict : Standard response; data contains:
        - embedding_mode: "esm2" or "mock"
        - predictions: {agg_risk, stability, viscosity_risk}
        - score: {score, grade, color}
        - shap: SHAP explanation dict
        - advice: list of actionable recommendations
        - validation_plan: recommended analytical testing plan
        - prediction_mode: "xgboost" or "rule_based"
    """
    try:
        from src.pLM_embedder import ESM2Embedder, get_embedder
        from src.developability_predictor import (
            predict_developability, get_predictor, BIOPHYS_NAMES,
        )
        from src.validation_planner import generate_validation_plan

        # -- Step 1: pLM Embedding -----------------------------------------
        embedder = get_embedder()
        embedding_mode = "mock" if embedder.is_mock else "esm2"

        if vh_sequence and vl_sequence:
            embedding = embedder.embed_antibody(vh_sequence, vl_sequence)
        elif sequence and len(sequence) > 20:
            single = embedder.embed_sequence(sequence)
            import numpy as np
            embedding = np.concatenate([single, single])  # pad to ANTIBODY_EMBED_DIM
        else:
            from src.pLM_embedder import mock_embedding, ANTIBODY_EMBED_DIM
            import numpy as np
            embedding = np.zeros(ANTIBODY_EMBED_DIM, dtype=np.float32)

        # -- Step 2: Biophysical features ----------------------------------
        # Prefer Feature Registry if available (Phase 5c: single source of truth)
        import numpy as np
        if feature_set is not None and hasattr(feature_set, "biophysical_7dim"):
            biophysical = np.array(feature_set.biophysical_7dim(), dtype=np.float32)
            log.info("Using Feature Registry biophysical_7dim for developability prediction")
        else:
            biophysical = np.array([
                pI, mw,
                float(deam_sites), float(ox_sites),
                float(acidic_residues), float(basic_residues),
                hydrophobicity,
            ], dtype=np.float32)

        # -- Step 3: Predict risks -----------------------------------------
        intent_for_context = {
            "pI": pI, "mw": mw, "hydrophobicity": hydrophobicity,
            "deam_sites": deam_sites, "ox_sites": ox_sites,
        }
        if molecule_class:
            intent_for_context["molecule_class"] = molecule_class
        dev_result = predict_developability(embedding, biophysical, intent_for_context,
                                            sequence=sequence)

        # -- Step 4: Validation plan (molecule-class-aware) -----------------
        _mol_cls = intent_for_context.get("molecule_class", "canonical_mab")
        if hasattr(_mol_cls, "value"):   # MoleculeClass enum → str
            _mol_cls = _mol_cls.value
        validation_plan = generate_validation_plan(
            risk_scores=dev_result["predictions"],
            intent=intent_for_context,
            molecule_class=_mol_cls,
        )

        # -- Serialize for JSON response -----------------------------------
        # Convert SHAP data to JSON-safe format
        shap_serialized = {"available": dev_result["shap"].get("available", False)}
        if shap_serialized["available"]:
            shap_serialized["targets"] = {}
            for target_name, target_data in dev_result["shap"].get("targets", {}).items():
                if "error" in target_data:
                    shap_serialized["targets"][target_name] = {"error": target_data["error"]}
                else:
                    shap_serialized["targets"][target_name] = {
                        "base_value": target_data.get("base_value", 0),
                        "embed_contribution": target_data.get("embed_contribution", 0),
                        "biophys_contribution": target_data.get("biophys_contribution", 0),
                        "biophys_shap": target_data.get("biophys_shap", {}),
                        "top_features": target_data.get("top_features", [])[:10],
                    }

        # Serialize advice
        advice_serialized = []
        for a in dev_result.get("advice", []):
            advice_serialized.append({
                "category": a["category"],
                "risk_level": a["risk_level"],
                "message": a["message"],
                "priority": a["priority"],
            })

        # Serialize validation plan (already JSON-safe)
        plan_serialized = {
            "total_assays": validation_plan["total_assays"],
            "risk_summary": validation_plan["risk_summary"],
            "recommendations": validation_plan["recommendations"],
            "required_assays": [
                {"id": a["id"], "name": a["name"], "priority": a["priority"],
                 "measures": a["measures"], "timeline": a["timeline"],
                 "trigger_reason": a.get("trigger_reason", "")}
                for a in validation_plan["required_assays"]
            ],
            "risk_triggered_assays": [
                {"id": a["id"], "name": a["name"], "priority": a["priority"],
                 "measures": a["measures"], "timeline": a["timeline"],
                 "trigger_reason": a.get("trigger_reason", "")}
                for a in validation_plan["risk_triggered_assays"]
            ],
        }
        if "timeline" in validation_plan:
            plan_serialized["timeline"] = validation_plan["timeline"]

        log.info("Tool4 [predict_developability_risk] -> score=%.4f (%s), "
                 "mode=%s, embedding=%s, assays=%d",
                 dev_result["score"]["score"],
                 dev_result["score"]["grade"],
                 dev_result["mode"],
                 embedding_mode,
                 validation_plan["total_assays"])

        # -- OOD Detection --------------------------------------------------
        ood_info = {}
        if sequence and len(sequence) > 20:
            try:
                from src.developability_predictor import compute_ood_flags
                ood_info = compute_ood_flags(
                    sequence=sequence,
                    pI=pI,
                    mw_kda=mw,
                    molecule_class=molecule_class,
                )
            except Exception as ood_err:
                log.warning("OOD detection failed: %s", ood_err)

        return _success(
            data={
                "embedding_mode": embedding_mode,
                "prediction_mode": dev_result["mode"],
                "predictions": dev_result["predictions"],
                "score": dev_result["score"],
                "shap": shap_serialized,
                "advice": advice_serialized,
                "validation_plan": plan_serialized,
                "ood_info": ood_info,
                "biophysical_input": {
                    name: round(float(biophysical[i]), 4)
                    for i, name in enumerate(BIOPHYS_NAMES)
                },
            },
            message=(
                f"Developability assessment complete: "
                f"Score={dev_result['score']['score']:.3f} ({dev_result['score']['grade']}), "
                f"Mode={dev_result['mode']}, Embedding={embedding_mode}"
            ),
        )

    except Exception as e:
        log.error("Tool4 failed: %s", e)
        return _error(
            message=f"predict_developability_risk failed: {e}",
            details=traceback.format_exc(),
        )


# ===========================================================================
# Tool 28: Unified Multi-Task Prediction (Biologics AI Integration)
#           8-task simultaneous prediction with CADET-ready output
# ===========================================================================

@register_tool(
    name="predict_unified_multitask",
    description=(
        "Unified 8-task simultaneous prediction using shared-backbone MultiTask model. "
        "Predicts ka, nu (SMA parameters for CADET), Tm, aggregation_risk, stability, "
        "viscosity_risk, hydrophobicity, and potency from HC/LC sequences and "
        "biophysical features. Returns CADET-ready ml_override and developability score."
    ),
    parameters={
        "pI": {"type": "number", "description": "Isoelectric point"},
        "mw": {"type": "number", "description": "Molecular weight (kDa)"},
        "hydrophobicity": {"type": "number", "description": "GRAVY hydrophobicity [0,1]"},
        "deam_sites": {"type": "integer", "description": "Deamidation site count"},
        "ox_sites": {"type": "integer", "description": "Oxidation site count"},
        "acidic_residues": {"type": "integer", "description": "D + E count"},
        "basic_residues": {"type": "integer", "description": "K + R + H count"},
        "hc_sequence": {"type": "string", "description": "Heavy chain sequence"},
        "lc_sequence": {"type": "string", "description": "Light chain sequence"},
    },
    category="ml_prediction",
)
def predict_unified_multitask(
    pI: float = 8.5,
    mw: float = 148.0,
    hydrophobicity: float = 0.35,
    deam_sites: int = 1,
    ox_sites: int = 1,
    acidic_residues: int = 40,
    basic_residues: int = 50,
    hc_sequence: Optional[str] = None,
    lc_sequence: Optional[str] = None,
    sequence: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool 28: Unified 8-task prediction via shared-backbone MultiTaskModel.

    Returns predictions in a format compatible with both existing tools
    and the new CADET closed-loop pipeline.
    """
    try:
        # Try loading the unified model via adapter
        from multitask_adapter import get_adapter
        adapter = get_adapter()

        if adapter.is_available:
            hc = hc_sequence or sequence or "EVQLVESGGGLVQPGGSLRLSCAAS"
            lc = lc_sequence or sequence or "DIQMTQSPSSLSASVGDRVTITC"
            biophys = {
                "pI": pI, "MW_kDa": mw,
                "deam_sites": float(deam_sites), "ox_sites": float(ox_sites),
                "acidic_residues": float(acidic_residues),
                "basic_residues": float(basic_residues),
                "hydrophobicity_gravy": hydrophobicity,
            }
            result = adapter.predict_all(hc, lc, biophys)
            return _success(
                data=result,
                message=(
                    f"Unified 8-task prediction complete: "
                    f"ka={result['predictions'].get('ka', 0):.4f}, "
                    f"nu={result['predictions'].get('nu', 0):.3f}, "
                    f"dev_score={result['developability_score']:.3f} "
                    f"({result['developability_grade']})"
                ),
            )
        else:
            # Fallback: aggregate results from existing independent predictors
            log.info("Unified model not available; assembling from independent predictors")
            return _predict_unified_fallback(
                pI=pI, mw=mw, hydrophobicity=hydrophobicity,
                deam_sites=deam_sites, ox_sites=ox_sites,
                acidic_residues=acidic_residues, basic_residues=basic_residues,
                hc_sequence=hc_sequence, lc_sequence=lc_sequence,
                sequence=sequence,
            )
    except Exception as e:
        log.warning("Unified prediction failed, using fallback: %s", e)
        return _predict_unified_fallback(
            pI=pI, mw=mw, hydrophobicity=hydrophobicity,
            deam_sites=deam_sites, ox_sites=ox_sites,
            acidic_residues=acidic_residues, basic_residues=basic_residues,
            hc_sequence=hc_sequence, lc_sequence=lc_sequence,
            sequence=sequence,
        )


def _predict_unified_fallback(
    pI: float = 8.5, mw: float = 148.0, hydrophobicity: float = 0.35,
    deam_sites: int = 1, ox_sites: int = 1,
    acidic_residues: int = 40, basic_residues: int = 50,
    hc_sequence: Optional[str] = None, lc_sequence: Optional[str] = None,
    sequence: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fallback: assemble unified predictions from existing independent predictors.
    Ensures backward compatibility when the unified model is not yet trained.
    """
    predictions = {}

    # ChromatographyMLP → ka, nu
    try:
        ml_result = predict_ml_with_shap(
            pI=pI, mw=mw, hydrophobicity=hydrophobicity,
            deam_sites=deam_sites, ox_sites=ox_sites,
            acidic_residues=acidic_residues, basic_residues=basic_residues,
            sequence=sequence,
        )
        if ml_result["status"] == "success":
            predictions["ka"] = ml_result["data"]["prediction"]["ka"]
            predictions["nu"] = ml_result["data"]["prediction"]["nu"]
            # v7.3.2: Propagate val_r2 for quality gating
            _ml_ov = ml_result["data"].get("ml_override", {})
            if "val_r2" in _ml_ov:
                predictions["val_r2"] = _ml_ov["val_r2"]
    except Exception:
        predictions["ka"] = 1.0
        predictions["nu"] = 3.0

    # DevelopabilityPredictor → agg_risk, stability, viscosity_risk
    try:
        dev_result = predict_developability_risk(
            pI=pI, mw=mw, hydrophobicity=hydrophobicity,
            deam_sites=deam_sites, ox_sites=ox_sites,
            acidic_residues=acidic_residues, basic_residues=basic_residues,
            vh_sequence=hc_sequence, vl_sequence=lc_sequence,
            sequence=sequence,
        )
        if dev_result["status"] == "success":
            dp = dev_result["data"]["predictions"]
            predictions["aggregation_risk"] = dp.get("agg_risk", 0.5)
            predictions["stability"] = dp.get("stability", 0.5)
            predictions["viscosity_risk"] = dp.get("viscosity_risk", 0.5)
    except Exception:
        predictions.setdefault("aggregation_risk", 0.5)
        predictions.setdefault("stability", 0.5)
        predictions.setdefault("viscosity_risk", 0.5)

    # Defaults for remaining tasks
    predictions.setdefault("tm", 65.0 + (pI - 7.0) * 2.0)
    predictions.setdefault("hydrophobicity", hydrophobicity)
    predictions.setdefault("potency", 0.5)

    # Compute developability score
    agg = predictions.get("aggregation_risk", 0.5)
    stab = predictions.get("stability", 0.5)
    visc = predictions.get("viscosity_risk", 0.5)
    dev_score = 0.40 * agg + 0.30 * (1.0 - stab) + 0.30 * visc
    # Use canonical grade boundaries from report_schema (SINGLE SOURCE OF TRUTH)
    from src.report_schema import GRADE_LOW_UPPER, GRADE_MEDIUM_UPPER
    grade = "Low Risk" if dev_score < GRADE_LOW_UPPER else ("Medium Risk" if dev_score < GRADE_MEDIUM_UPPER else "High Risk")

    return _success(
        data={
            "predictions": predictions,
            "developability_score": dev_score,
            "developability_grade": grade,
            "ml_override": {
                "ka": predictions.get("ka", 1.0),
                "nu": predictions.get("nu", 3.0),
                **({"val_r2": predictions["val_r2"]} if "val_r2" in predictions else {}),
            },
            "model": "fallback_ensemble",
        },
        message=(
            f"Unified prediction (fallback mode): "
            f"ka={predictions.get('ka', 0):.4f}, "
            f"nu={predictions.get('nu', 0):.3f}, "
            f"dev_score={dev_score:.3f} ({grade})"
        ),
    )


# ===========================================================================
# Tool 5: Bispecific Antibody Separation Analysis (M11)
#          3-Species Homodimer/Heterodimer Assessment
# ===========================================================================

@register_tool(
    name="predict_bispecific_separation",
    description=(
        "Analyze bispecific antibody purification feasibility by simulating "
        "the three assembly species (Homodimer AA, Heterodimer AB target, "
        "Homodimer BB). Computes per-species pI, MW, SMA parameters, "
        "estimated retention times, resolution (Rs), and homodimer "
        "co-elution risk with actionable recommendations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "chain_a_sequence": {
                "type": "string",
                "description": "Amino acid sequence for Chain A / Arm 1",
            },
            "chain_b_sequence": {
                "type": "string",
                "description": "Amino acid sequence for Chain B / Arm 2",
            },
            "chain_a_name": {
                "type": "string",
                "description": "Display name for Chain A",
                "default": "ArmA",
            },
            "chain_b_name": {
                "type": "string",
                "description": "Display name for Chain B",
                "default": "ArmB",
            },
            "gradient_slope": {
                "type": "number",
                "description": "Salt gradient slope (mM/min)",
                "default": 15.0,
            },
        },
        "required": ["chain_a_sequence", "chain_b_sequence"],
    },
    category="bispecific",
)
def predict_bispecific_separation(
    chain_a_sequence: str,
    chain_b_sequence: str,
    chain_a_name: str = "ArmA",
    chain_b_name: str = "ArmB",
    gradient_slope: float = 15.0,
) -> Dict[str, Any]:
    """
    Bispecific antibody 3-species separation analysis (Milestone 11).

    Pipeline:
      1. Build AntibodyChain objects with Biopython biophysical properties
      2. Assemble Homodimer AA, Heterodimer AB, Homodimer BB species
      3. Map each species to CADET SMA parameters via PropertyMapper
      4. Estimate retention times (Yamamoto gradient elution theory)
      5. Calculate Resolution (Rs) between AB-AA and AB-BB
      6. Assess homodimer co-elution risk (High/Medium/Low)
      7. Generate chromatogram visualization data

    Parameters
    ----------
    chain_a_sequence : Amino acid sequence for Chain A / Arm 1
    chain_b_sequence : Amino acid sequence for Chain B / Arm 2
    chain_a_name     : Display name for Chain A
    chain_b_name     : Display name for Chain B
    gradient_slope   : Salt gradient slope (mM/min), default 15.0

    Returns
    -------
    dict : Standard response; data contains:
        - chain_a, chain_b: Chain biophysical properties
        - species: {AA, AB, BB} with pI, MW, hydrophobicity
        - sma_params: Per-species SMA parameters
        - peaks: Per-species RT and FWHM
        - resolution: Rs_AB_AA, Rs_AB_BB, min_rs
        - risk: risk_level, risk_details, recommendations
        - chromatogram: time/signal arrays for visualization
    """
    try:
        from src.bispecific_engine import run_bispecific_analysis

        result = run_bispecific_analysis(
            chain_a_seq=chain_a_sequence,
            chain_b_seq=chain_b_sequence,
            chain_a_name=chain_a_name,
            chain_b_name=chain_b_name,
            gradient_slope=gradient_slope,
        )

        if result.get("status") != "success":
            return _error(
                message="Bispecific analysis did not complete successfully",
                details=str(result),
            )

        risk_level = result["risk"]["risk_level"]
        min_rs = result["resolution"]["min_rs"]

        log.info(
            "Tool5 [predict_bispecific_separation] -> %s x %s: "
            "risk=%s, Rs_min=%.3f",
            chain_a_name, chain_b_name, risk_level, min_rs,
        )

        return _success(
            data=result,
            message=(
                f"Bispecific separation analysis complete: "
                f"{chain_a_name} x {chain_b_name}, "
                f"Risk={risk_level}, Rs_min={min_rs:.3f}"
            ),
        )

    except Exception as e:
        log.error("Tool5 failed: %s", e)
        return _error(
            message=f"predict_bispecific_separation failed: {e}",
            details=traceback.format_exc(),
        )


# ===========================================================================
# Tool 6: High-Throughput Virtual Screening (Milestone 16)
# ===========================================================================

@register_tool(
    name="run_ht_screen",
    description=(
        "Run High-Throughput Virtual Screening on a batch of antibody candidate "
        "sequences. Scores each candidate for Developability (aggregation risk, "
        "thermal stability) and Potency (binding affinity prediction), then "
        "classifies into Magic Quadrant: Star (high dev + high potency), "
        "Developable, Potent, or Risky."
    ),
    parameters={
        "type": "object",
        "properties": {
            "csv_content": {
                "type": "string",
                "description": "Raw CSV content with Candidate_ID, Sequence_HC, Sequence_LC columns",
            },
            "dev_threshold": {
                "type": "number",
                "description": "Developability score threshold for quadrant classification (0-1)",
                "default": 0.5,
            },
            "potency_threshold": {
                "type": "number",
                "description": "Potency score threshold for quadrant classification (0-1)",
                "default": 0.5,
            },
        },
        "required": [],
    },
    category="discovery",
)
def run_ht_screen(
    candidates: Optional[List[Dict[str, Any]]] = None,
    csv_content: Optional[str] = None,
    dev_threshold: float = 0.5,
    potency_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Tool 6: Run High-Throughput Virtual Screening on candidate sequences.

    Accepts either a list of candidate dicts or raw CSV content.
    Each candidate is scored for Developability and Potency, then
    classified into Magic Quadrant (Star/Developable/Potent/Risky).

    Parameters
    ----------
    candidates : List of dicts with candidate_id, sequence_hc, sequence_lc
    csv_content : Raw CSV string (alternative to candidates list)
    dev_threshold : Score threshold for "high developability" (default 0.5)
    potency_threshold : Score threshold for "high potency" (default 0.5)

    Returns
    -------
    Standard response dict with screening results and quadrant classification.
    """
    try:
        from src.ht_screening import HTScreeningEngine, parse_discovery_csv

        # Parse CSV if provided instead of candidates list
        if candidates is None and csv_content is not None:
            parsed = parse_discovery_csv(csv_content)
            if parsed["status"] != "success":
                return _error(message=parsed.get("message", "CSV parse failed"))
            candidates = parsed["candidates"]

        if not candidates:
            return _error(message="No candidates provided. Supply candidates list or csv_content.")

        engine = HTScreeningEngine(
            dev_threshold=dev_threshold,
            potency_threshold=potency_threshold,
        )

        result = engine.screen_candidates(candidates)

        if result["status"] != "success":
            return _error(message="Screening failed")

        summary = result.get("summary", {})
        qc = summary.get("quadrant_counts", {})

        return _success(
            data={
                "results": result["results"],
                "summary": summary,
                "star_candidates": engine.get_star_candidates(),
                "csv_export": engine.get_results_as_csv(),
                "models_used": result.get("models_used", {}),
                "screening_time_sec": result.get("screening_time_sec", 0),
            },
            message=(
                f"HT Screening complete: {result['n_candidates']} candidates processed. "
                f"Stars={qc.get('Star', 0)}, Developable={qc.get('Developable', 0)}, "
                f"Potent={qc.get('Potent', 0)}, Risky={qc.get('Risky', 0)}"
            ),
        )

    except Exception as e:
        log.error("Tool6 failed: %s", e)
        return _error(
            message=f"run_ht_screen failed: {e}",
            details=traceback.format_exc(),
        )


# ===========================================================================
# Tool 9: Molecular Physics -- BioPython Protein Analysis
#          v1.0 (STEP 3): Sequence-only biophysical characterization
# ===========================================================================

@register_tool(
    name="analyze_molecular_physics",
    description=(
        "Compute comprehensive biophysical properties from an amino acid sequence "
        "using BioPython ProteinAnalysis. Returns isoelectric point (pI), molecular "
        "weight, GRAVY hydrophobicity, extinction coefficients, net charge at a "
        "given pH, deamidation/oxidation liability counts, and amino acid composition. "
        "Use this tool when the user provides a raw sequence and wants to understand "
        "its fundamental physical and chemical characteristics."
    ),
    parameters={
        "type": "object",
        "properties": {
            "sequence": {
                "type": "string",
                "description": "Amino acid sequence (single-letter code, e.g. 'EVQLVESGG...')",
            },
            "ph": {
                "type": "number",
                "description": "pH for net charge calculation (default 7.0)",
                "default": 7.0,
            },
        },
        "required": ["sequence"],
    },
    category="bioinformatics",
)
def analyze_molecular_physics(
    sequence: str,
    ph:       float = 7.0,
) -> Dict[str, Any]:
    """
    Tool 9: Compute biophysical properties from an amino acid sequence.

    Uses BioPython ProteinAnalysis to calculate pI, MW, GRAVY, extinction
    coefficients, net charge, and sequence liability motifs (NG/NS deamidation,
    Met oxidation).

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter code)
    ph : float
        pH for net charge calculation (default 7.0)

    Returns
    -------
    dict : Standard response; data contains pI, mw_kda, gravy, charge_at_ph,
           extinction_coeff_reduced, extinction_coeff_cystines, sequence_length,
           deamidation_sites, oxidation_sites, hydrophobicity_normalized
    """
    try:
        import re
        seq_clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", sequence.upper())
        if len(seq_clean) < 10:
            return _error(message=f"Sequence too short ({len(seq_clean)} aa). Minimum 10 residues.")

        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            pa = ProteinAnalysis(seq_clean)
            pI = round(pa.isoelectric_point(), 3)
            mw = round(pa.molecular_weight() / 1000.0, 3)  # kDa
            gravy = round(pa.gravy(), 4)
            charge = round(pa.charge_at_pH(ph), 3)
            ext = pa.molar_extinction_coefficient()  # (reduced, cystines)
            ext_reduced = ext[0]
            ext_cystines = ext[1]
            aa_pct = pa.get_amino_acids_percent()
        except ImportError:
            # Heuristic fallback
            aa_comp = {aa: seq_clean.count(aa) / len(seq_clean) for aa in "ACDEFGHIKLMNPQRSTVWY"}
            acidic = (seq_clean.count("D") + seq_clean.count("E")) / len(seq_clean)
            basic = (seq_clean.count("K") + seq_clean.count("R") + seq_clean.count("H")) / len(seq_clean)
            pI = round(7.0 + (basic - acidic) * 8.0, 3)
            mw = round(len(seq_clean) * 0.11, 3)
            gravy_table = {
                "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
                "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
                "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
                "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
            }
            gravy = round(sum(gravy_table.get(aa, 0) for aa in seq_clean) / len(seq_clean), 4)
            charge = round((basic - acidic) * len(seq_clean), 3)
            ext_reduced = 0
            ext_cystines = 0
            aa_pct = aa_comp

        # Liability counts
        deam_sites = len(re.findall(r"N[GS]", seq_clean))
        ox_sites = seq_clean.count("M")
        hydro_norm = round(max(0.0, min(1.0, (gravy + 2.0) / 4.0)), 4)

        data = {
            "sequence_length":           len(seq_clean),
            "pI":                        pI,
            "mw_kda":                    mw,
            "gravy":                     gravy,
            "charge_at_ph":              charge,
            "ph_used":                   ph,
            "extinction_coeff_reduced":  ext_reduced,
            "extinction_coeff_cystines": ext_cystines,
            "deamidation_sites":         deam_sites,
            "oxidation_sites":           ox_sites,
            "hydrophobicity_normalized": hydro_norm,
            "amino_acid_composition":    {k: round(v, 4) for k, v in aa_pct.items()},
        }

        log.info("Tool9 [analyze_molecular_physics] -> %d aa, pI=%.2f, MW=%.1f kDa", len(seq_clean), pI, mw)

        return _success(
            data=data,
            message=(
                f"Molecular physics: {len(seq_clean)} aa, pI={pI}, MW={mw} kDa, "
                f"GRAVY={gravy}, charge@pH{ph}={charge}, "
                f"{deam_sites} deamidation + {ox_sites} oxidation sites"
            ),
        )

    except Exception as e:
        log.error("Tool9 failed: %s", e)
        return _error(
            message=f"analyze_molecular_physics failed: {e}",
            details=traceback.format_exc(),
        )


# ===========================================================================
# Tool 10: Downstream Process Simulation -- PropertyMapper + CADET
#           v1.0 (STEP 3): Sequence → SMA params → chromatography simulation
# ===========================================================================

@register_tool(
    name="simulate_downstream_process",
    description=(
        "Simulate ion-exchange chromatography downstream purification for a protein. "
        "Takes protein properties (name, pI, MW, sequence) and process conditions "
        "(gradient slope, pH) and runs the PropertyMapper → CADET SMA pipeline. "
        "Returns three-variant SMA parameters, predicted retention times, resolution, "
        "and purity metrics. Use this tool when the user wants to explore purification "
        "conditions or compare gradient strategies for a molecule."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Protein name (e.g. 'adalimumab', 'trastuzumab')",
            },
            "pI": {
                "type": "number",
                "description": "Isoelectric point (pH units, mAb range 7-9)",
            },
            "mw": {
                "type": "number",
                "description": "Molecular weight in kDa (mAb ~150)",
            },
            "sequence": {
                "type": "string",
                "description": "Amino acid sequence (optional, used for GRAVY-based ka)",
                "default": "",
            },
            "gradient_slope": {
                "type": "number",
                "description": "Salt gradient slope in mM/min (typical 10-30, default 15)",
                "default": 15.0,
            },
            "ph": {
                "type": "number",
                "description": "Working buffer pH (typical 6.0-8.5, default 7.0)",
                "default": 7.0,
            },
        },
        "required": ["name", "pI", "mw"],
    },
    category="simulation",
)
def simulate_downstream_process(
    name:           str,
    pI:             float,
    mw:             float,
    sequence:       str   = "",
    gradient_slope: float = 15.0,
    ph:             float = 7.0,
) -> Dict[str, Any]:
    """
    Tool 10: End-to-end downstream process simulation.

    Chains PropertyMapper (protein → SMA params) with CADET simulation
    (SMA params → chromatogram) to predict retention times, resolution,
    and purity for a given protein and gradient condition.

    Parameters
    ----------
    name           : Protein name
    pI             : Isoelectric point
    mw             : Molecular weight (kDa)
    sequence       : Amino acid sequence (optional, enables GRAVY-based ka)
    gradient_slope : Salt gradient slope (mM/min)
    ph             : Working pH

    Returns
    -------
    dict : Standard response; data contains sma_params, simulation_result,
           retention_times, resolution, purity, summary
    """
    try:
        import re
        from src.PropertyMapper import ProteinProperties, PropertyMapper

        # -- Build protein properties --
        seq_clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", sequence.upper()) if sequence else None
        gravy_score = None
        hydro = 0.35
        deam = 1
        ox_sites = 1

        if seq_clean and len(seq_clean) >= 10:
            deam = len(re.findall(r"N[GS]", seq_clean))
            ox_sites = seq_clean.count("M")
            try:
                from Bio.SeqUtils.ProtParam import ProteinAnalysis
                pa = ProteinAnalysis(seq_clean)
                gravy_score = pa.gravy()
                hydro = max(0.0, min(1.0, (gravy_score + 2.0) / 4.0))
            except ImportError:
                gravy_table = {
                    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
                    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
                    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
                    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
                }
                gravy_score = sum(gravy_table.get(aa, 0) for aa in seq_clean) / len(seq_clean)
                hydro = max(0.0, min(1.0, (gravy_score + 2.0) / 4.0))

        protein = ProteinProperties(
            name=name, pI=pI, MW_kDa=mw,
            hydrophobicity=hydro, pH_working=ph,
            sequence=seq_clean,
            ptm_profile={"deamidation_sites": deam, "oxidation_sites": ox_sites},
            gravy_score=gravy_score,
        )

        mapper = PropertyMapper()
        variants = mapper.map_variants(protein)
        explanation = mapper.explain(protein)

        # -- Attempt CADET simulation --
        sim_result = None
        try:
            from src.cadet_engine import CadetEngine, VariantParams, ProcessParams
            vparams = VariantParams.from_dict(variants)
            process = ProcessParams(gradient_slope=gradient_slope)
            engine = CadetEngine(workspace="data", engine_dir="engine")
            timestamp = _time.strftime("%Y%m%d_%H%M%S")
            h5_path = engine.build_h5(f"agent_ds_{timestamp}.h5", vparams, process)
            sim_result = engine.run_simulation(h5_path)
        except Exception as sim_err:
            log.warning("CADET simulation unavailable: %s — returning SMA params only", sim_err)

        # -- Build response --
        data = {
            "sma_params": {
                "acidic": {k: round(float(v), 4) for k, v in variants.get("acidic", {}).items()},
                "main":   {k: round(float(v), 4) for k, v in variants.get("main", {}).items()},
                "basic":  {k: round(float(v), 4) for k, v in variants.get("basic", {}).items()},
                "kd":     round(float(variants.get("kd", 0)), 2),
                "lambda": round(float(variants.get("lambda_", 0)), 2),
            },
            "gradient_slope_mM_min": gradient_slope,
            "ph_working":            ph,
            "protein_input":         {"name": name, "pI": pI, "mw": mw, "has_sequence": bool(seq_clean)},
            "explanation":           explanation,
        }

        if sim_result:
            cqa = sim_result.compute_cqa()
            peaks = {}
            for comp_name, peak_info in cqa["peaks"].items():
                peaks[comp_name] = {k: round(float(v), 4) for k, v in peak_info.items()}
            resolution = {lbl: round(float(rs), 4) for lbl, rs in cqa["resolution"].items()}
            area_pct = {comp: round(float(pct), 2) for comp, pct in cqa["area_pct"].items()}

            data["simulation"] = {
                "peaks":      peaks,
                "resolution": resolution,
                "area_pct":   area_pct,
                "wall_time":  round(sim_result.wall_time, 3),
            }
            msg = (
                f"Downstream simulation for {name}: gradient={gradient_slope} mM/min, pH={ph}. "
                f"Main peak RT={peaks.get('Main', {}).get('rt_min', 'N/A')} min. "
                f"Resolution: {resolution}"
            )
        else:
            data["simulation"] = None
            msg = (
                f"PropertyMapper params computed for {name} (pI={pI}, MW={mw} kDa). "
                f"CADET engine not available; SMA parameters returned for manual simulation."
            )

        log.info("Tool10 [simulate_downstream_process] -> %s", name)
        return _success(data=data, message=msg)

    except Exception as e:
        log.error("Tool10 failed: %s", e)
        return _error(
            message=f"simulate_downstream_process failed: {e}",
            details=traceback.format_exc(),
        )


# ===========================================================================
# Tool 11: Retrain Predictive Models -- STEP 2 MLOps Pipeline
#           v1.0 (STEP 3): Deterministic XGBoost retraining
# ===========================================================================

@register_tool(
    name="retrain_predictive_models",
    description=(
        "Retrain XGBoost predictive models on a clinical dataset CSV (e.g. Jain-137). "
        "Executes the full STEP 2 MLOps pipeline: data curation (dual-chain fusion + "
        "ESM-2 embedding + biophysical features) → XGBoost training with train/val split "
        "→ model serialization to models/baseline_model.pkl. Returns per-target validation "
        "metrics (R², RMSE, MAE). Use this tool when the user provides new training data "
        "or asks to retrain/update the ML models."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dataset_path": {
                "type": "string",
                "description": "Path to training CSV (e.g. 'Jain137_Cleaned_Training_Data.csv')",
            },
            "seed": {
                "type": "integer",
                "description": "Random seed for reproducibility (default 42)",
                "default": 42,
            },
            "val_fraction": {
                "type": "number",
                "description": "Validation split fraction (default 0.2)",
                "default": 0.2,
            },
        },
        "required": ["dataset_path"],
    },
    category="ml_training",
)
def retrain_predictive_models(
    dataset_path: str,
    seed:         int   = 42,
    val_fraction: float = 0.2,
) -> Dict[str, Any]:
    """
    Tool 11: Retrain predictive models using the STEP 2 deterministic MLOps pipeline.

    Calls trigger_model_training() which orchestrates:
      1. DataCurator: CSV → (G4S)3 chain fusion → ESM-2 embed → feature matrix (N, 327)
      2. ModelTrainer: XGBoost training with train/val split → per-target metrics
      3. Model serialization to models/baseline_model.pkl

    Parameters
    ----------
    dataset_path : str
        Path to training CSV (must contain VH, VL columns)
    seed : int
        Random seed for deterministic training (default 42)
    val_fraction : float
        Fraction of data held out for validation (default 0.2)

    Returns
    -------
    dict : Standard response; data contains metrics, model_path, n_samples,
           target_names, timing
    """
    try:
        from src.model_trainer import trigger_model_training

        result = trigger_model_training(
            dataset_path=dataset_path,
            model_type="xgboost",
            seed=seed,
            val_fraction=val_fraction,
        )

        if result.get("status") != "success":
            return _error(message=result.get("message", "Training failed"))

        data = {
            "model_path":     result["model_path"],
            "n_samples":      result["n_samples"],
            "n_features":     result["n_features"],
            "n_targets":      result["n_targets"],
            "target_names":   result["target_names"],
            "metrics":        result["metrics"],
            "curator_time_s": round(result.get("curator_time_s", 0), 2),
            "training_time_s": round(result.get("training_time_s", 0), 2),
            "total_time_s":   round(result.get("total_time_s", 0), 2),
            "seed":           seed,
            "val_fraction":   val_fraction,
        }

        # Build human-readable summary
        lines = [f"Model retrained: {result['n_samples']} samples, {result['n_targets']} targets"]
        for tgt in result["target_names"]:
            m = result["metrics"].get(tgt, {})
            lines.append(f"  {tgt}: R²={m.get('R2', 'N/A'):.3f}, RMSE={m.get('RMSE', 'N/A'):.3f}")
        lines.append(f"Saved: {result['model_path']} ({result.get('total_time_s', 0):.1f}s)")

        log.info("Tool11 [retrain_predictive_models] -> %s targets, %.1fs",
                 result["n_targets"], result.get("total_time_s", 0))

        return _success(data=data, message="\n".join(lines))

    except Exception as e:
        log.error("Tool11 failed: %s", e)
        return _error(
            message=f"retrain_predictive_models failed: {e}",
            details=traceback.format_exc(),
        )


# ===========================================================================
# Layer 3: PharmaAgentManager -- Workflow Dispatcher
#           v12.0: HT Screening + Bispecific + Developability Digital Twin
# ===========================================================================

# ===========================================================================
# M14: Generative Protein Engineering — optimize_candidate
# ===========================================================================

def optimize_candidate(
    chains: List[Dict[str, Any]],
    dev_score: float = 0.0,
    dev_grade: str = "Low",
    pk_half_life: float = 21.0,
    pk_risk: str = "Low",
    bispecific_rs: Optional[float] = None,
    bispecific_chain_b_idx: Optional[int] = None,
    glycoform_profile: str = "standard_cho",
    n_variants: int = 3,
) -> Dict[str, Any]:
    """
    M14: Generate optimized sequence variants and silently evaluate each
    through the full analytical pipeline (pI/MW, developability, PK).

    This function:
      1. Calls the generative_engineer to produce N variant chain sets
      2. For each variant, computes the super-sequence properties (pI, MW, GRAVY)
      3. Runs liability density analysis
      4. Runs the PK half-life predictor
      5. Returns a comparative table: WT vs Variant 1 vs ... vs Variant N

    Parameters
    ----------
    chains               : Wild-type chain list [{sequence, copy_number, name, chain_type}]
    dev_score            : WT developability score (0-1)
    dev_grade            : WT developability grade
    pk_half_life         : WT predicted half-life
    pk_risk              : WT PK risk level
    bispecific_rs        : Min resolution (if bispecific)
    bispecific_chain_b_idx: Chain B index (for charge engineering)
    glycoform_profile    : Active glycoform profile key
    n_variants           : Number of variants to generate

    Returns
    -------
    dict : {
        "status": "success" | "no_optimization_needed" | "error",
        "wild_type": {pI, mw_kda, gravy, hydrophobicity, liability_density, pk_half_life, dev_score, ...},
        "variants": [{name, strategy, mutations, mutation_count, pI, mw_kda, pk_half_life, ...}],
        "optimization_triggers": list,
        "fasta_download": str (multi-FASTA),
        "summary": str,
    }
    """
    try:
        from src.generative_engineer import (
            generate_optimized_variants,
            variants_to_fasta,
        )
        from src.analytical_twin import (
            calculate_liability_density,
            build_super_sequence,
        )
        from src.preclinical_twin import (
            predict_human_half_life,
            check_fcrn_binding_motif,
        )
    except ImportError as ie:
        return _error(f"Required module not available: {ie}")

    # -- Step 1: Generate variants -------------------------------------------
    gen_result = generate_optimized_variants(
        chains=chains,
        dev_score=dev_score,
        dev_grade=dev_grade,
        pk_half_life=pk_half_life,
        pk_risk=pk_risk,
        bispecific_rs=bispecific_rs,
        bispecific_chain_b_idx=bispecific_chain_b_idx,
        n_variants=n_variants,
    )

    if gen_result["status"] == "no_optimization_needed":
        return {
            "status": "no_optimization_needed",
            "wild_type": {},
            "variants": [],
            "optimization_triggers": [],
            "fasta_download": "",
            "summary": gen_result["summary"],
        }

    # -- Step 2: Evaluate wild-type properties --------------------------------
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
    except ImportError:
        return _error("Biopython required for optimization evaluation")

    def _evaluate_chains(chain_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute biophysical + PK properties for a set of chains."""
        super_seq = build_super_sequence(chain_list)
        if len(super_seq) < 10:
            return {"error": "Sequence too short"}

        try:
            analysis = ProteinAnalysis(super_seq)
            pI = round(analysis.isoelectric_point(), 2)
            mw_da = analysis.molecular_weight()
            mw_kda = round(mw_da / 1000.0, 1)
            gravy = round(analysis.gravy(), 3)
            hydrophobicity = round(max(0.0, min(1.0, (gravy + 2.0) / 4.0)), 3)
        except Exception:
            return {"error": "ProteinAnalysis failed"}

        # Liability density
        ld = calculate_liability_density(chain_list)
        liab_density = ld.get("density_per_1000", 0.0)
        liab_risk = ld.get("risk_level", "Low")

        # FcRn check
        fcrn_intact = True
        if len(super_seq) > 200:
            fcrn_check = check_fcrn_binding_motif(super_seq)
            fcrn_intact = fcrn_check.get("intact", True)

        # PK prediction
        pk = predict_human_half_life(
            global_pi=pI,
            hydrophobicity=hydrophobicity,
            liability_density=liab_density,
            fcrn_binding_motif_intact=fcrn_intact,
            mw_kda=mw_kda,
            glycoform_profile=glycoform_profile,
        )

        return {
            "pI": pI,
            "mw_kda": mw_kda,
            "gravy": gravy,
            "hydrophobicity": hydrophobicity,
            "liability_density": round(liab_density, 1),
            "liability_risk": liab_risk,
            "total_motifs": ld.get("total_motifs", 0),
            "total_residues": ld.get("total_residues", 0),
            "pk_half_life": pk.get("half_life_days", 0),
            "pk_risk": pk.get("risk_assessment", "Unknown"),
            "pk_clearance": pk.get("clearance_ml_day_kg", 0),
            "fcrn_intact": fcrn_intact,
        }

    # Evaluate WT
    wt_props = _evaluate_chains(chains)
    wt_props["dev_score"] = dev_score
    wt_props["dev_grade"] = dev_grade

    # -- Step 3: Evaluate each variant (biophysics + PK) ----------------------
    evaluated_variants = []
    for var in gen_result["variants"]:
        var_chains = var.get("chains", [])
        var_props = _evaluate_chains(var_chains)

        # Compute deltas
        deltas = {}
        if "error" not in var_props and "error" not in wt_props:
            for key in ("pI", "mw_kda", "hydrophobicity", "liability_density", "pk_half_life"):
                wt_val = wt_props.get(key, 0)
                var_val = var_props.get(key, 0)
                if isinstance(wt_val, (int, float)) and isinstance(var_val, (int, float)):
                    deltas[f"delta_{key}"] = round(var_val - wt_val, 3)

        evaluated_variants.append({
            "name": var["name"],
            "strategy": var["strategy"],
            "mutations": var["mutations"],
            "mutation_count": var["mutation_count"],
            "mutation_summary": var["mutation_summary"],
            "chains": var_chains,
            **var_props,
            **deltas,
        })

    # -- Step 3b: M15 Wet-Lab Model Evaluation & Pareto Filtering ----------
    wetlab_available = False
    pareto_applied = False
    rejected_variants = []
    wt_wetlab_pred = {}

    try:
        from src.ml_predictor import get_wetlab_model, extract_features as _extract_feat
        from src.generative_engineer import evaluate_and_filter_variants

        wetlab_mdl = get_wetlab_model()
        if wetlab_mdl and wetlab_mdl.trained:
            wetlab_available = True

            # Run multi-objective evaluation and Pareto filter
            pareto_result = evaluate_and_filter_variants(
                variants=gen_result["variants"],
                wild_type_chains=chains,
                glycoform_profile=glycoform_profile,
            )

            if pareto_result["status"] == "success":
                pareto_applied = True
                rejected_variants = pareto_result.get("rejected_variants", [])
                wt_wetlab_pred = pareto_result.get("wt_predictions", {})

                # Enrich evaluated_variants with wet-lab predictions
                pareto_names = {
                    v.get("name"): v
                    for v in pareto_result.get("pareto_variants", [])
                }
                all_eval_map = {
                    v.get("name"): v
                    for v in pareto_result.get("all_evaluated", [])
                }

                for ev in evaluated_variants:
                    vname = ev["name"]
                    enriched = all_eval_map.get(vname) or pareto_names.get(vname)
                    if enriched:
                        ev["wetlab_predictions"] = enriched.get("wetlab_predictions", {})
                        ev["pareto_optimal"] = enriched.get("pareto_optimal", False)
                        ev["delta_agg_pct"] = enriched.get("delta_agg_pct", 0)
                        ev["delta_tm"] = enriched.get("delta_tm", 0)
                    else:
                        ev["wetlab_predictions"] = {}
                        ev["pareto_optimal"] = False

                    # Check if this variant was rejected
                    for rej in rejected_variants:
                        if rej.get("name") == vname:
                            ev["rejected"] = True
                            ev["rejection_reason"] = rej.get("rejection_reason", "")
                            ev["pareto_optimal"] = False
                            break

    except ImportError:
        log.debug("M15 wet-lab or Pareto modules not available — skipping")
    except Exception as e:
        log.warning("M15 Pareto evaluation failed: %s", e)

    # Enrich WT with wet-lab predictions if available
    if wt_wetlab_pred:
        wt_props["wetlab_predictions"] = wt_wetlab_pred

    # -- Step 4: Generate FASTA download --------------------------------------
    fasta = variants_to_fasta(gen_result["variants"], wt_chains=chains)

    # -- Summary ---------------------------------------------------------------
    best_var = None
    best_hl = pk_half_life
    for ev in evaluated_variants:
        hl = ev.get("pk_half_life", 0)
        if hl > best_hl and not ev.get("rejected", False):
            best_hl = hl
            best_var = ev["name"]

    summary_parts = [gen_result["summary"]]
    if best_var:
        summary_parts.append(
            f"Best improvement: {best_var} (half-life {best_hl:.1f} days, "
            f"up from {pk_half_life:.1f} days)"
        )
    if pareto_applied:
        n_pareto = sum(1 for ev in evaluated_variants if ev.get("pareto_optimal"))
        n_rejected = len(rejected_variants)
        summary_parts.append(
            f"Pareto filter: {n_pareto} optimal, {n_rejected} rejected by wet-lab model"
        )

    return {
        "status": "success",
        "wild_type": wt_props,
        "variants": evaluated_variants,
        "optimization_triggers": gen_result["optimization_triggers"],
        "fasta_download": fasta,
        "summary": ". ".join(summary_parts),
        "wetlab_available": wetlab_available,
        "pareto_applied": pareto_applied,
        "rejected_variants": rejected_variants,
    }


class PharmaAgentManager:
    """
    ProtePilot Multi-Agent Manager.

    v12.0 provides six execution modes:
      1. **HT Screening Pipeline** (run_ht_screening_pipeline) [NEW M16]:
         Tool 6 (HT screening) -> bulk candidate processing -> Magic Quadrant.
         Early Discovery integration with Developability + Potency scoring.

      2. **Bispecific Pipeline** (run_bispecific_pipeline) [M11]:
         Tool 5 (bispecific analysis) -> 3-species separation assessment.
         Homodimer AA / Heterodimer AB / Homodimer BB risk analysis.

      3. **Developability Pipeline** (run_developability_pipeline) [M8]:
         Tool 3 (ML) -> Tool 1 (params) -> Tool 4 (developability) -> Tool 2 (simulate).
         Full end-to-end digital twin with risk assessment.

      4. **ML-First Pipeline** (run_ml_first_pipeline):
         Tool 3 (ML predict) -> Tool 1 (with ml_override) -> Tool 2 (simulate).
         The neural network drives the parameter selection.

      5. **Deterministic Pipeline** (run_deterministic_pipeline):
         Tool 1 (static formulas) -> Tool 2 (simulate).
         No ML; pure mechanistic model.

      6. **LLM-Driven Pipeline** (run_llm_pipeline) [placeholder]:
         Connects to LangChain / Claude API for dynamic tool invocation.
    """

    def __init__(
        self,
        workspace:  str = "data",
        engine_dir: str = "engine",
    ):
        self.workspace  = workspace
        self.engine_dir = engine_dir
        self._execution_log: List[Dict[str, Any]] = []
        log.info("PharmaAgentManager initialized (workspace=%s)", workspace)

    # -- ML-First Pipeline (v6.0) -------------------------------------------

    def run_ml_first_pipeline(
        self,
        prompt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        ML-First end-to-end pipeline:
            Step 0: Tool 3 (ML predict) -> get ml_override {ka, nu}
            Step 1: Tool 1 (with ml_override) -> three-variant parameters
            Step 2: Tool 2 (simulate) -> CQA

        The neural network takes the steering wheel; static formulas
        are only used if ML prediction fails.

        Parameters
        ----------
        prompt_data : Input parameter dictionary (same as run_deterministic_pipeline)

        Returns
        -------
        dict with pipeline results including source (ml_override or static_v5)
        """
        t_start = _time.time()
        pipeline_results: List[Dict[str, Any]] = []

        source = prompt_data.get("source", "text")
        log.info("=" * 60)
        log.info("ML-First Pipeline: %s (source: %s)", prompt_data.get("name", "unknown"), source)
        log.info("=" * 60)

        # -- Step 0: ML Prediction (Tool 3) ---------------------------------
        log.info("[Step 0/2] Running ML prediction (PyTorch MLP)...")
        ml_override = None
        try:
            ml_result = predict_ml_with_shap(
                pI=prompt_data["pI"],
                mw=prompt_data["mw"],
                deam_sites=prompt_data.get("deam_sites", 1),
                ox_sites=prompt_data.get("ox_sites", 1),
                acidic_residues=prompt_data.get("acidic_residues", 40),
                basic_residues=prompt_data.get("basic_residues", 50),
                hydrophobicity=prompt_data.get("hydrophobicity", 0.35),
                sequence=prompt_data.get("sequence", None),
            )
            pipeline_results.append({"tool": "predict_ml_with_shap", "result": ml_result})

            if ml_result["status"] == "success":
                ml_override = ml_result["data"]["ml_override"]
                log.info("[Step 0/2] ML override ready: ka=%.4f, nu=%.3f, est_RT=%.1f min",
                         ml_override["ka"], ml_override["nu"],
                         ml_result["data"]["prediction"].get("estimated_rt_min", 0))
            else:
                log.warning("[Step 0/2] ML prediction failed, falling back to static formulas: %s",
                            ml_result.get("message", ""))
        except Exception as e:
            log.warning("[Step 0/2] ML prediction exception, using static fallback: %s", e)

        # -- Step 1: Physical Parameters (Tool 1, with ml_override) ---------
        log.info("[Step 1/2] Calling predict_physical_params (ml_override=%s)...",
                 "YES" if ml_override else "NO (static fallback)")

        step1 = predict_physical_params(
            name=prompt_data["name"],
            pI=prompt_data["pI"],
            mw=prompt_data["mw"],
            hydrophobicity=prompt_data.get("hydrophobicity", 0.35),
            pH_working=prompt_data.get("pH_working", 7.0),
            deam_sites=prompt_data.get("deam_sites", 1),
            ox_sites=prompt_data.get("ox_sites", 1),
            sequence=prompt_data.get("sequence", None),
            ml_override=ml_override,
        )

        pipeline_results.append({"tool": "predict_physical_params", "result": step1})

        if step1["status"] != "success":
            return {
                "status":   "error",
                "pipeline": pipeline_results,
                "final_cqa": None,
                "summary":  f"Pipeline failed at Step 1: {step1['message']}",
                "wall_time_total": _time.time() - t_start,
            }

        log.info("[Step 1/2] Complete (source: %s)", step1["data"].get("source", "unknown"))

        # -- Step 2: Simulation & CQA (Tool 2) ------------------------------
        log.info("[Step 2/2] Calling run_chromatography_simulation...")

        variants_dict = step1["data"]["variants"]
        gradient_slope = prompt_data.get("gradient_slope", 15.0)

        step2 = run_chromatography_simulation(
            variants_dict=variants_dict,
            gradient_slope=gradient_slope,
            run_name=f"ml_pipeline_{prompt_data['name']}",
        )

        pipeline_results.append({"tool": "run_chromatography_simulation", "result": step2})

        if step2["status"] != "success":
            return {
                "status":   "error",
                "pipeline": pipeline_results,
                "final_cqa": None,
                "summary":  f"Pipeline failed at Step 2: {step2['message']}",
                "wall_time_total": _time.time() - t_start,
            }

        wall_total = _time.time() - t_start
        param_source = step1["data"].get("source", "unknown")

        summary_lines = [
            "=" * 50,
            f"  ProtePilot -- ML-First Pipeline Report",
            "=" * 50,
            f"  Protein: {prompt_data['name']}",
            f"  Parameter Source: {param_source.upper()}",
            f"  Total wall time: {wall_total:.2f}s",
        ]
        if ml_override:
            summary_lines += [
                f"",
                f"  [Step 0] ML Prediction (PyTorch MLP v2.0):",
                f"    ML ka={ml_override['ka']:.4f}, ML nu={ml_override['nu']:.3f}",
            ]
        summary_lines += [
            f"",
            f"  [Step 1] Physical Parameters ({param_source}):",
            f"    Acidic: nu={step1['data']['variants']['acidic']['nu']:.3f}  "
            f"ka={step1['data']['variants']['acidic']['ka']:.6f}",
            f"    Main:   nu={step1['data']['variants']['main']['nu']:.3f}  "
            f"ka={step1['data']['variants']['main']['ka']:.6f}",
            f"    Basic:  nu={step1['data']['variants']['basic']['nu']:.3f}  "
            f"ka={step1['data']['variants']['basic']['ka']:.6f}",
            f"",
            f"  [Step 2] Simulation + CQA:",
            f"    {step2['data']['summary']}",
            "=" * 50,
        ]

        final_result = {
            "status":          "success",
            "pipeline":        pipeline_results,
            "final_cqa":       step2["data"]["cqa"],
            "summary":         "\n".join(summary_lines),
            "wall_time_total": round(wall_total, 3),
            "param_source":    param_source,
        }

        self._execution_log.append({
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "input":     prompt_data,
            "output":    final_result,
            "mode":      "ml_first",
        })

        log.info("ML-First Pipeline complete (%.2fs, source=%s)", wall_total, param_source)
        return final_result

    # -- Unified Pipeline (Biologics AI Integration) -------------------------

    def run_unified_pipeline(
        self,
        prompt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Unified end-to-end pipeline:
            Step 0: Tool 28 (unified multitask) → 8 predictions + ml_override
            Step 1: Tool 1 (physical params with ml_override) → SMA parameters
            Step 2: Tool 2 (CADET simulation) → chromatogram + retention time
            Step 3: Developability assessment from unified predictions

        This is the CADET closed-loop: Sequence → Prediction → Simulation.
        """
        t_start = _time.time()
        pipeline_results = []

        log.info("=" * 60)
        log.info("UNIFIED Pipeline: %s", prompt_data.get("name", "unknown"))
        log.info("=" * 60)

        # -- Step 0: Unified 8-task Prediction ---------------------------------
        log.info("[Step 0/3] Running unified multitask prediction...")
        ml_override = None
        unified_preds = {}

        try:
            t28 = predict_unified_multitask(
                pI=prompt_data["pI"],
                mw=prompt_data["mw"],
                hydrophobicity=prompt_data.get("hydrophobicity", 0.35),
                deam_sites=prompt_data.get("deam_sites", 1),
                ox_sites=prompt_data.get("ox_sites", 1),
                acidic_residues=prompt_data.get("acidic_residues", 40),
                basic_residues=prompt_data.get("basic_residues", 50),
                hc_sequence=prompt_data.get("hc_sequence"),
                lc_sequence=prompt_data.get("lc_sequence"),
                sequence=prompt_data.get("sequence"),
            )
            pipeline_results.append({"tool": "predict_unified_multitask", "result": t28})

            if t28["status"] == "success":
                ml_override = t28["data"]["ml_override"]
                unified_preds = t28["data"]["predictions"]
                log.info(
                    "[Step 0/3] Unified: ka=%.4f, nu=%.3f, tm=%.1f, dev=%s",
                    ml_override.get("ka", 0), ml_override.get("nu", 0),
                    unified_preds.get("tm", 0),
                    t28["data"].get("developability_grade", "?"),
                )
        except Exception as e:
            log.warning("[Step 0/3] Unified prediction failed: %s", e)

        # -- Step 1: Physical Parameters (with ml_override) --------------------
        log.info("[Step 1/3] Computing physical parameters...")
        step1 = predict_physical_params(
            name=prompt_data["name"],
            pI=prompt_data["pI"],
            mw=prompt_data["mw"],
            hydrophobicity=prompt_data.get("hydrophobicity", 0.35),
            pH_working=prompt_data.get("pH_working", 7.0),
            deam_sites=prompt_data.get("deam_sites", 1),
            ox_sites=prompt_data.get("ox_sites", 1),
            sequence=prompt_data.get("sequence"),
            ml_override=ml_override,
        )
        pipeline_results.append({"tool": "predict_physical_params", "result": step1})

        if step1["status"] != "success":
            return {
                "status": "error",
                "message": f"Physical params failed: {step1.get('message', '')}",
                "pipeline_results": pipeline_results,
            }

        # -- Step 2: CADET Chromatography Simulation ---------------------------
        log.info("[Step 2/3] Running CADET simulation...")
        step2 = run_chromatography_sim(
            name=prompt_data["name"],
            pI=prompt_data["pI"],
            mw=prompt_data["mw"],
            hydrophobicity=prompt_data.get("hydrophobicity", 0.35),
            pH_working=prompt_data.get("pH_working", 7.0),
            exp_rt_min=prompt_data.get("exp_rt_min"),
            override_params=step1["data"]["variants"]["main"],
        )
        pipeline_results.append({"tool": "run_chromatography_sim", "result": step2})

        # -- Step 3: Assemble comprehensive report ----------------------------
        elapsed = _time.time() - t_start

        result = {
            "status": "success",
            "pipeline": "unified",
            "elapsed_seconds": round(elapsed, 2),
            "unified_predictions": unified_preds,
            "ml_override": ml_override,
            "physical_params": step1["data"] if step1["status"] == "success" else None,
            "chromatography": step2["data"] if step2["status"] == "success" else None,
            "developability": {
                "score": t28["data"].get("developability_score") if t28 and t28.get("status") == "success" else None,
                "grade": t28["data"].get("developability_grade") if t28 and t28.get("status") == "success" else None,
            },
            "pipeline_results": pipeline_results,
            "message": (
                f"Unified pipeline complete in {elapsed:.1f}s: "
                f"8 predictions → SMA params → CADET simulation"
            ),
        }

        log.info("Unified pipeline completed in %.1fs", elapsed)
        return result

    # -- Developability Pipeline (v7.0 End-to-End Digital Twin) ---------------

    def run_developability_pipeline(
        self,
        prompt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        End-to-end developability digital twin pipeline (M8):
            Step 0: Tool 3 (ML predict) -> ml_override
            Step 1: Tool 1 (physical params with ml_override)
            Step 2: Tool 4 (developability risk assessment)
            Step 3: Tool 2 (CADET simulation)

        Combines ML-First chromatographic prediction with developability
        risk scoring, SHAP explainability, and validation planning.

        Parameters
        ----------
        prompt_data : Input parameter dictionary

        Returns
        -------
        dict with full pipeline results including developability assessment
        """
        t_start = _time.time()
        pipeline_results: List[Dict[str, Any]] = []

        source = prompt_data.get("source", "text")
        log.info("=" * 60)
        log.info("Developability Pipeline: %s (source: %s)",
                 prompt_data.get("name", "unknown"), source)
        log.info("=" * 60)

        # -- Step 0: ML Prediction (Tool 3) ---------------------------------
        log.info("[Step 0/3] Running ML prediction (PyTorch MLP)...")
        ml_override = None
        try:
            ml_result = predict_ml_with_shap(
                pI=prompt_data["pI"],
                mw=prompt_data["mw"],
                deam_sites=prompt_data.get("deam_sites", 1),
                ox_sites=prompt_data.get("ox_sites", 1),
                acidic_residues=prompt_data.get("acidic_residues", 40),
                basic_residues=prompt_data.get("basic_residues", 50),
                hydrophobicity=prompt_data.get("hydrophobicity", 0.35),
                sequence=prompt_data.get("sequence", None),
            )
            pipeline_results.append({"tool": "predict_ml_with_shap", "result": ml_result})
            if ml_result["status"] == "success":
                ml_override = ml_result["data"]["ml_override"]
                log.info("[Step 0/3] ML override ready: ka=%.4f, nu=%.3f",
                         ml_override["ka"], ml_override["nu"])
            else:
                log.warning("[Step 0/3] ML prediction failed, using static fallback")
        except Exception as e:
            log.warning("[Step 0/3] ML prediction exception: %s", e)

        # -- Step 1: Physical Parameters (Tool 1) ---------------------------
        log.info("[Step 1/3] Predicting physical parameters...")
        step1 = predict_physical_params(
            name=prompt_data["name"],
            pI=prompt_data["pI"],
            mw=prompt_data["mw"],
            hydrophobicity=prompt_data.get("hydrophobicity", 0.35),
            pH_working=prompt_data.get("pH_working", 7.0),
            deam_sites=prompt_data.get("deam_sites", 1),
            ox_sites=prompt_data.get("ox_sites", 1),
            sequence=prompt_data.get("sequence", None),
            ml_override=ml_override,
        )
        pipeline_results.append({"tool": "predict_physical_params", "result": step1})

        if step1["status"] != "success":
            return {
                "status": "error", "pipeline": pipeline_results,
                "summary": f"Pipeline failed at Step 1: {step1['message']}",
                "wall_time_total": _time.time() - t_start,
            }

        # -- Step 2: Developability Risk (Tool 4) ---------------------------
        log.info("[Step 2/3] Running developability risk assessment...")

        # Extract VH/VL sequences from chains if available.
        # For multi-chain molecules (bispecific, etc.), pick the longest
        # HC-like and LC-like chains so the embedding covers the primary
        # variable domains. scFv arms, HC2/LC2 are included via the
        # combined "sequence" field passed separately.
        vh_seq, vl_seq = "", ""
        _hc_types = {"HC", "HEAVY", "HC1", "HEAVY1"}
        _lc_types = {"LC", "LIGHT", "LC1", "LIGHT1"}
        _vh_candidates, _vl_candidates = [], []
        for chain in prompt_data.get("chains", []):
            ct = chain.get("chain_type", "").upper().replace(" ", "")
            ch_seq = chain.get("sequence", "")
            if not ch_seq:
                continue
            if ct in _hc_types:
                _vh_candidates.append(ch_seq)
            elif ct in _lc_types:
                _vl_candidates.append(ch_seq)
            elif "SCFV" in ct or "VHH" in ct:
                # scFv arms and VHH contribute to VH for embedding
                _vh_candidates.append(ch_seq)
            elif ct in {"HC2", "HEAVY2"}:
                _vh_candidates.append(ch_seq)
            elif ct in {"LC2", "LIGHT2"}:
                _vl_candidates.append(ch_seq)
        # Pick the longest candidate for VH and VL
        if _vh_candidates:
            vh_seq = max(_vh_candidates, key=len)
        if _vl_candidates:
            vl_seq = max(_vl_candidates, key=len)

        step2_dev = predict_developability_risk(
            pI=prompt_data["pI"],
            mw=prompt_data["mw"],
            hydrophobicity=prompt_data.get("hydrophobicity", 0.35),
            deam_sites=prompt_data.get("deam_sites", 1),
            ox_sites=prompt_data.get("ox_sites", 1),
            acidic_residues=prompt_data.get("acidic_residues", 40),
            basic_residues=prompt_data.get("basic_residues", 50),
            sequence=prompt_data.get("sequence", None),
            vh_sequence=vh_seq,
            vl_sequence=vl_seq,
            molecule_class=prompt_data.get("molecule_class"),
        )
        pipeline_results.append({"tool": "predict_developability_risk", "result": step2_dev})

        if step2_dev["status"] != "success":
            log.warning("[Step 2/3] Developability assessment failed: %s",
                       step2_dev.get("message", ""))

        # -- Step 3: CADET Simulation (Tool 2) ------------------------------
        log.info("[Step 3/3] Running CADET simulation...")
        variants_dict = step1["data"]["variants"]
        step3 = run_chromatography_simulation(
            variants_dict=variants_dict,
            gradient_slope=prompt_data.get("gradient_slope", 15.0),
            run_name=f"dev_pipeline_{prompt_data['name']}",
        )
        pipeline_results.append({"tool": "run_chromatography_simulation", "result": step3})

        wall_total = _time.time() - t_start

        summary_lines = [
            "=" * 50,
            f"  ProtePilot — Developability Pipeline Report",
            "=" * 50,
            f"  Protein: {prompt_data['name']}",
            f"  Total wall time: {wall_total:.2f}s",
        ]

        if step2_dev["status"] == "success":
            score = step2_dev["data"]["score"]
            summary_lines += [
                f"",
                f"  Developability Score: {score['score']:.3f} ({score['grade']})",
                f"  Assays recommended: {step2_dev['data']['validation_plan']['total_assays']}",
            ]

        summary_lines.append("=" * 50)

        final_result = {
            "status": "success",
            "pipeline": pipeline_results,
            "final_cqa": step3["data"]["cqa"] if step3["status"] == "success" else None,
            "developability": step2_dev["data"] if step2_dev["status"] == "success" else None,
            "summary": "\n".join(summary_lines),
            "wall_time_total": round(wall_total, 3),
            "param_source": step1["data"].get("source", "unknown"),
        }

        self._execution_log.append({
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "input": prompt_data,
            "output": final_result,
            "mode": "developability",
        })

        log.info("Developability Pipeline complete (%.2fs)", wall_total)
        return final_result

    # -- Bispecific Pipeline (v8.0 M11) --------------------------------------

    def run_bispecific_pipeline(
        self,
        chain_a_seq: str,
        chain_b_seq: str,
        chain_a_name: str = "ArmA",
        chain_b_name: str = "ArmB",
        gradient_slope: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Bispecific antibody separation analysis pipeline (M11):
            Tool 5 (bispecific analysis) -> 3-species assessment

        Analyzes Homodimer AA, Heterodimer AB, Homodimer BB species,
        maps to SMA parameters, estimates retention times, and assesses
        homodimer co-elution risk with actionable recommendations.

        Parameters
        ----------
        chain_a_seq    : Amino acid sequence for Chain A / Arm 1
        chain_b_seq    : Amino acid sequence for Chain B / Arm 2
        chain_a_name   : Display name for Chain A
        chain_b_name   : Display name for Chain B
        gradient_slope : Salt gradient slope (mM/min)

        Returns
        -------
        dict with bispecific analysis results including species,
        SMA params, peaks, resolution, risk, and chromatogram data
        """
        t_start = _time.time()

        log.info("=" * 60)
        log.info("Bispecific Pipeline: %s x %s", chain_a_name, chain_b_name)
        log.info("=" * 60)

        result = predict_bispecific_separation(
            chain_a_sequence=chain_a_seq,
            chain_b_sequence=chain_b_seq,
            chain_a_name=chain_a_name,
            chain_b_name=chain_b_name,
            gradient_slope=gradient_slope,
        )

        wall_total = _time.time() - t_start

        self._execution_log.append({
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "input": {
                "chain_a_name": chain_a_name,
                "chain_b_name": chain_b_name,
                "gradient_slope": gradient_slope,
            },
            "output": result,
            "mode": "bispecific",
        })

        log.info("Bispecific Pipeline complete (%.2fs)", wall_total)
        return result

    # -- HT Screening Pipeline (v12.0 M16) -----------------------------------

    def run_ht_screening_pipeline(
        self,
        candidates: Optional[List[Dict[str, Any]]] = None,
        csv_content: Optional[str] = None,
        dev_threshold: float = 0.5,
        potency_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        HT Screening pipeline: Tool 6 (bulk screening) -> Magic Quadrant.

        Processes a batch of candidate antibody sequences through
        developability and potency models, classifies into quadrants,
        and returns ranked results.

        Parameters
        ----------
        candidates : List of candidate dicts (alternative to csv_content)
        csv_content : Raw CSV string with Discovery format
        dev_threshold : Developability score threshold
        potency_threshold : Potency score threshold

        Returns
        -------
        dict with screening results, quadrant classification, and CSV export
        """
        log.info("=== HT Screening Pipeline (M16 v12.0) ===")
        t0 = _time.time()

        result = run_ht_screen(
            candidates=candidates,
            csv_content=csv_content,
            dev_threshold=dev_threshold,
            potency_threshold=potency_threshold,
        )

        wall_total = _time.time() - t0
        log.info("HT Screening Pipeline complete (%.2fs)", wall_total)

        return result

    # -- Deterministic Pipeline (static fallback) ---------------------------

    def run_deterministic_pipeline(
        self,
        prompt_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Deterministic pipeline: Tool 1 (static v5.0) -> Tool 2 (simulation).

        No ML override; pure mechanistic model. Preserved for backward
        compatibility and as a comparison baseline.
        """
        t_start = _time.time()
        pipeline_results: List[Dict[str, Any]] = []

        source = prompt_data.get("source", "text")
        log.info("=" * 60)
        log.info("Deterministic Pipeline: %s (source: %s)", prompt_data.get("name", "unknown"), source)
        log.info("=" * 60)

        # -- Step 1: Bio-Informatics Tool (no ML override) ------------------
        step1 = predict_physical_params(
            name=prompt_data["name"],
            pI=prompt_data["pI"],
            mw=prompt_data["mw"],
            hydrophobicity=prompt_data.get("hydrophobicity", 0.35),
            pH_working=prompt_data.get("pH_working", 7.0),
            deam_sites=prompt_data.get("deam_sites", 1),
            ox_sites=prompt_data.get("ox_sites", 1),
            sequence=prompt_data.get("sequence", None),
            ml_override=None,  # explicit: no ML override
        )

        pipeline_results.append({"tool": "predict_physical_params", "result": step1})

        if step1["status"] != "success":
            return {
                "status":   "error",
                "pipeline": pipeline_results,
                "final_cqa": None,
                "summary":  f"Pipeline failed at Step 1: {step1['message']}",
                "wall_time_total": _time.time() - t_start,
            }

        # -- Step 2: Simulation & CQA Tool -----------------------------------
        variants_dict = step1["data"]["variants"]
        gradient_slope = prompt_data.get("gradient_slope", 15.0)

        step2 = run_chromatography_simulation(
            variants_dict=variants_dict,
            gradient_slope=gradient_slope,
            run_name=f"pipeline_{prompt_data['name']}",
        )

        pipeline_results.append({"tool": "run_chromatography_simulation", "result": step2})

        if step2["status"] != "success":
            return {
                "status":   "error",
                "pipeline": pipeline_results,
                "final_cqa": None,
                "summary":  f"Pipeline failed at Step 2: {step2['message']}",
                "wall_time_total": _time.time() - t_start,
            }

        wall_total = _time.time() - t_start

        summary_lines = [
            "=" * 50,
            f"  ProtePilot -- Deterministic Pipeline Report",
            "=" * 50,
            f"  Protein: {prompt_data['name']}",
            f"  Input source: {source}",
            f"  Total wall time: {wall_total:.2f}s",
            "",
            "  [Step 1] Bio-Informatics -> Physical Parameters (v5.0 Static Fallback):",
            f"    Acidic: nu={step1['data']['variants']['acidic']['nu']:.3f}  "
            f"ka={step1['data']['variants']['acidic']['ka']:.6f}",
            f"    Main:   nu={step1['data']['variants']['main']['nu']:.3f}  "
            f"ka={step1['data']['variants']['main']['ka']:.6f}",
            f"    Basic:  nu={step1['data']['variants']['basic']['nu']:.3f}  "
            f"ka={step1['data']['variants']['basic']['ka']:.6f}",
            "",
            f"  [Step 2] Simulation + CQA:",
            f"    {step2['data']['summary']}",
            "=" * 50,
        ]

        final_result = {
            "status":          "success",
            "pipeline":        pipeline_results,
            "final_cqa":       step2["data"]["cqa"],
            "summary":         "\n".join(summary_lines),
            "wall_time_total": round(wall_total, 3),
        }

        self._execution_log.append({
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "input":     prompt_data,
            "output":    final_result,
            "mode":      "deterministic",
        })

        log.info("Deterministic Pipeline complete (%.2fs)", wall_total)
        return final_result

    # -- LLM-Driven Pipeline (placeholder) -----------------------------------

    def run_llm_pipeline(
        self,
        user_prompt: str,
        llm_client:  Any = None,
    ) -> Dict[str, Any]:
        """LLM-driven dynamic pipeline -- placeholder interface."""
        if llm_client is None:
            return _error(
                message="LLM pipeline requires an llm_client parameter. "
                        "Please configure an API Key, or use run_ml_first_pipeline().",
            )
        return _error(message="LLM pipeline not yet implemented. Use run_ml_first_pipeline().")

    # -- Tool management -----------------------------------------------------

    @staticmethod
    def list_available_tools() -> List[Dict[str, str]]:
        """List all registered Tools with descriptions."""
        return [
            {
                "name":        spec.name,
                "description": spec.description,
                "category":    spec.category,
            }
            for spec in list_tools()
        ]

    @staticmethod
    def get_tool_schemas() -> List[Dict[str, Any]]:
        """Export OpenAI function-calling compatible Tool schemas."""
        return export_tool_schemas()

    @property
    def execution_log(self) -> List[Dict[str, Any]]:
        """Return historical execution log."""
        return self._execution_log


# ===========================================================================
# __main__: Local Testing
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  ProtePilot -- Agent Architecture v7.0 (Developability Digital Twin)")
    print("=" * 60)

    manager = PharmaAgentManager()
    print("\nRegistered Tools:")
    for t in manager.list_available_tools():
        print(f"  [{t['category']}] {t['name']}: {t['description'][:60]}...")

    # -- Test Tool 1 with ML override ----------------------------------------
    print("\n--- Tool 1: predict_physical_params (ML override) ---")
    t1_ml = predict_physical_params(
        name="mAb_ML_Test",
        pI=8.45,
        mw=148.0,
        hydrophobicity=0.35,
        ml_override={"ka": 1.42, "nu": 2.58},
    )
    if t1_ml["status"] == "success":
        v = t1_ml["data"]["variants"]
        src = t1_ml["data"]["source"]
        print(f"  Source: {src}")
        print(f"  Main:   nu={v['main']['nu']:.3f}  ka={v['main']['ka']:.6f}")

    # -- Test Tool 1 without ML override (static fallback) -------------------
    print("\n--- Tool 1: predict_physical_params (static fallback) ---")
    t1_fb = predict_physical_params(
        name="mAb_Fallback",
        pI=8.45,
        mw=148.0,
        hydrophobicity=0.35,
    )
    if t1_fb["status"] == "success":
        v = t1_fb["data"]["variants"]
        src = t1_fb["data"]["source"]
        print(f"  Source: {src}")
        print(f"  Main:   nu={v['main']['nu']:.3f}  ka={v['main']['ka']:.6f}")

    # -- Test Tool 4 (developability risk) -----------------------------------
    print("\n--- Tool 4: predict_developability_risk ---")
    t4 = predict_developability_risk(
        pI=8.45, mw=148.0, hydrophobicity=0.35,
        deam_sites=1, ox_sites=1,
    )
    if t4["status"] == "success":
        d = t4["data"]
        print(f"  Embedding: {d['embedding_mode']}")
        print(f"  Mode: {d['prediction_mode']}")
        print(f"  Score: {d['score']['score']:.3f} ({d['score']['grade']})")
        print(f"  Assays: {d['validation_plan']['total_assays']}")
        print(f"  Advice items: {len(d['advice'])}")
    else:
        print(f"  Error: {t4['message']}")

    # -- Test Tool 28: Unified MultiTask Prediction --------------------------
    print("\n--- Tool 28: predict_unified_multitask ---")
    t28 = predict_unified_multitask(
        pI=8.45, mw=148.0, hydrophobicity=0.35,
        deam_sites=1, ox_sites=1,
        hc_sequence="EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH",
        lc_sequence="DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVA",
    )
    if t28["status"] == "success":
        preds = t28["data"]["predictions"]
        print(f"  Model: {t28['data']['model']}")
        for task, val in preds.items():
            print(f"  {task}: {val:.4f}")
        print(f"  Dev Score: {t28['data']['developability_score']:.3f} ({t28['data']['developability_grade']})")
        print(f"  ML Override: ka={t28['data']['ml_override']['ka']:.4f}, nu={t28['data']['ml_override']['nu']:.4f}")
    else:
        print(f"  Fallback: {t28['message']}")

    print(f"\nExported {len(manager.get_tool_schemas())} Tool schemas")
    print("\nAgent architecture v7.0 test complete")
