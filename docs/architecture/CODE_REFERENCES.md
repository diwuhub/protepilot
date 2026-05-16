# ProtePilot Deep Analysis — Code References & Evidence

## 1. EXPORT DATA QUALITY

### Validation Results
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/SelfTest/validation_results_v3.json`
**Lines:** 1–1976 (65 KB)

Key test passes:
```json
{
  "bispecific_format": {
    "status": "PASS",
    "checks": {
      "has_AA_species": true,
      "has_AB_species": true,
      "has_BB_species": true,
      "peaks_have_rt": true,
      "rs_realistic": true,
      "qtpp_has_rows": true,
      "recommendation_mentions_bispecific": true
    },
    "data": {
      "peaks": {
        "AA": {"rt_min": 13.629, "fwhm_min": 0.831},
        "AB": {"rt_min": 12.44, "fwhm_min": 0.772},
        "BB": {"rt_min": 10.537, "fwhm_min": 0.677}
      },
      "resolution": {
        "rs_AB_AA": 0.6299,
        "rs_AB_BB": 1.1153,
        "min_rs": 0.6299
      }
    }
  },
  "noncanonical_formats": {
    "status": "PASS",
    "checks": {
      "fc_fusion_no_crash": true,
      "fusion_no_crash": true,
      "sdab_no_crash": true,
      "fc_fusion_format_aware_rec": true,
      "fusion_format_aware_rec": true
    }
  }
}
```

### Report Assembly (Single Source of Truth)
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/report_assembler.py`

**Key Function:** `assemble_report()` (Lines 50–88)
```python
def assemble_report(
    intent: Dict[str, Any],
    analysis_cache: Optional[Dict[str, Any]] = None,
    session_extras: Optional[Dict[str, Any]] = None,
) -> ReportObject:
    """
    Assemble a complete ReportObject from current analysis state.
    v2.0: All data flows through a single ReportContext object.
    """
    cache = analysis_cache or {}
    extras = session_extras or {}

    # ── Step 0: Build unified context ──────────────────────────────
    ctx = _build_report_context(intent, cache, extras)

    report = ReportObject(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        context=ctx,
    )

    # ── Section builders — all read from ctx ──────────────────────
    report.executive_summary = _build_executive_summary(ctx, cache)
    report.molecule_overview = _build_molecule_overview(ctx)
    report.developability = _build_developability(ctx, cache, intent)
    report.analytical = _build_analytical(ctx, cache)
    report.process_pk = _build_process_pk(ctx, cache, extras)
    report.validation_plan = _build_validation_plan(ctx, cache, intent)
    report.model_metadata = _build_model_metadata(ctx, cache, intent)
    report.appendix = _build_appendix(ctx, intent)

    # ── Final: Cross-section consistency pass ─────────────────────
    _validate_cross_section_consistency(report)

    log.info("Report v2.0 assembled for '%s' (%s) — grade=%s, score=%.2f",
             ctx.molecule_name, ctx.molecule_class,
             ctx.overall_grade, ctx.overall_score or 0)

    return report
```

**ReportContext Data Priority:** (Lines 95–200)
```python
def _build_report_context(
    intent: Dict, cache: Dict, extras: Dict,
) -> ReportContext:
    """
    Extract ALL cross-section fields from the latest workspace context.

    Priority order for each field:
      1. feature_set_obj (FeatureRegistry computed values)
      2. intent dict (from parse_intent / molecule setup)
      3. None (NOT a hardcoded default)

    If a value is not available, it stays None.
    """
    ctx = ReportContext()
    ctx.context_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ctx.source = intent.get("source", "text")

    # ── Biophysical — priority: feature_set_obj > intent > None ──
    fs_obj = intent.get("feature_set_obj")
    if fs_obj and hasattr(fs_obj, "value"):
        ctx.molecular_weight_kda = fs_obj.value("mw_kda")
        ctx.isoelectric_point = fs_obj.value("pI")
        ctx.hydrophobicity = fs_obj.value("hydrophobicity")
        # ...
    else:
        ctx.molecular_weight_kda = intent.get("mw")  # may be None
        ctx.isoelectric_point = intent.get("pI")
        ctx.hydrophobicity = intent.get("hydrophobicity")
        # ...
```

---

## 2. DEPENDENCY & IMPORT ANALYSIS

### Torch Dependencies (6 modules)
**Files:**
- `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/esm2_hybrid_encoder.py` (Line 1: import torch)
- `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/multitask_adapter.py` (import torch)
- `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/unified_dataset.py` (import torch)
- `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/unified_multitask_model.py` (import torch)
- `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/unified_trainer.py` (import torch)
- `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/uncertainty_engine.py` (import torch)

### H5PY Dependencies (MISSING from requirements.txt)
**Files:**
- `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/cadet_engine.py` (Line 20+: import h5py)
- `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/CharacterizationAgent.py` (import h5py)

**Critical Finding:** requirements.txt (Line 1–42) does NOT list h5py
```
# -- Machine Learning --------
torch>=2.0.0
xgboost>=1.7.0
shap>=0.42.0
scikit-learn>=1.3.0

# NOTE: h5py is MISSING but required by cadet_engine.py!
```

**Fix:**
```bash
echo "h5py>=3.0.0" >> /sessions/optimistic-modest-pascal/mnt/ProtePilot/requirements.txt
```

### No Circular Imports
**Verification:** Imports traced for 5 key modules
```
bispecific_engine.py imports: logging, math, numpy, dataclasses, typing, Bio.SeqUtils.ProtParam
formulation_twin.py imports: logging, math, dataclasses, typing, numpy
preclinical_twin.py imports: logging, math, typing
immunogenicity_twin.py imports: logging, math, numpy, dataclasses, typing
generative_engineer.py imports: logging, re, typing
```
**Result:** No mutual/circular dependencies detected ✅

---

## 3. TEST INFRASTRUCTURE

### Validation Test Harness
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/SelfTest/run_validation.py`
**Lines:** 1–1200+ (comprehensive test runner)

**Entry Point:**
```python
def test_property_mapper() -> dict:
    log.info("=== 1. PropertyMapper ===")
    from src.PropertyMapper import ProteinProperties, PropertyMapper
    mapper = PropertyMapper()
    data = {}
    checks = {}
    for name, ref in REFERENCE_MABS.items():
        # ... test logic
```

### Missing pytest Configuration
**File:** NOT FOUND
- `/sessions/optimistic-modest-pascal/mnt/ProtePilot/pytest.ini` ❌
- `/sessions/optimistic-modest-pascal/mnt/ProtePilot/conftest.py` ❌

**Result:** Tests run via custom harness, not pytest

### Unit Test File
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/tests/test_unified_integration.py`
**Size:** 11 KB (single integration test)
**Coverage:** Limited to unified model integration

---

## 4. CONTEXT & STATE MANAGEMENT

### Workspace Manager
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/workspace_manager.py`
**Lines:** 1–444

**WorkspaceStore Class** (Lines 92–215):
```python
class WorkspaceStore:
    """
    Manages multiple workspaces in st.session_state.
    """

    def __init__(self):
        self.workspaces: Dict[str, Dict[str, Any]] = {}
        self.active_id: Optional[str] = None
        self.history_order: List[str] = []  # Most recent first

    @classmethod
    def from_session_state(cls, ss: Any) -> "WorkspaceStore":
        """Load or initialize from Streamlit session_state."""
        if "workspace_store" in ss:
            return ss["workspace_store"]
        store = cls()
        ss["workspace_store"] = store
        return store

    def create_new(
        self,
        display_name: Optional[str] = None,
        intent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new workspace and make it active."""
        ws = create_workspace(display_name=display_name, intent=intent)
        self.workspaces[ws["id"]] = ws
        self.history_order.insert(0, ws["id"])
        self.active_id = ws["id"]
        log.info("Created workspace %s: %s", ws["id"], ws["display_name"])
        return ws

    def get_active(self) -> Optional[Dict[str, Any]]:
        """Get the currently active workspace."""
        if self.active_id and self.active_id in self.workspaces:
            return self.workspaces[self.active_id]
        return None
```

**Workspace Data Model** (Lines 44–86):
```python
def create_workspace(
    display_name: Optional[str] = None,
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a new isolated workspace.
    Returns a workspace dict with a unique ID and empty state.
    """
    ws_id = f"ws_{uuid.uuid4().hex[:8]}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "id": ws_id,
        "display_name": display_name or f"Run {now}",
        "created_at": now,
        "intent": intent,
        "messages": [],
        "results": {},
        "labeled_data": [],
        "dev_result": None,
        "validation_report": None,
        "ml_prediction": None,
        "export_ready": False,
        "analysis_cache": None,
    }
```

### App.py State Management
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/app.py`
**Lines:** 10971 (total)

**Workspace Store Initialization** (Lines 1690–1691):
```python
# M10: Workspace Store
ws_store = WorkspaceStore.from_session_state(st.session_state)
```

**Molecule-Bound State Invalidation** (Lines 1822–1866):
```python
def _invalidate_molecule_bound_state(ws_store=None, new_intent=None) -> None:
    """Clear ALL state tied to a specific molecule — both session_state AND workspace.
    
    Called whenever:
      - User switches to a new molecule (new name / sequence)
      - User changes intent (charge, pI, binding target)
      - User clears analysis results
    
    Clears:
      - Session state keys: glycoform_profile, formulation_*, last_intent
      - Workspace fields: dev_result, validation_report, ml_prediction
    
    Preserves:
      - Workspace history (other molecules' runs remain)
      - UI state (dark_mode, sidebar state)
    
    ws_store : WorkspaceStore, optional
        If provided, also clears workspace-level stale fields.
    """
    # Clear workspace-level stale fields
    if ws_store is not None:
        ws = ws_store.get_active()
        if ws:
            ws["dev_result"] = None
            ws["validation_report"] = None
            ws["ml_prediction"] = None
            # ... other fields

    log.info("Molecule-bound state invalidated (session + workspace)")
```

---

## 5. SINGLE SOURCE OF TRUTH AUDIT

### Report Schema
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/report_schema.py`
**Lines:** 1–350+

**Evidence Tier Constants** (Lines 32–40):
```python
EVIDENCE_TIER_1 = "primary"     # Sequence-derived, high confidence
EVIDENCE_TIER_2 = "supporting"  # Predicted, moderate confidence
EVIDENCE_TIER_3 = "simulated"   # Virtual QC, low discriminatory

GRADE_LOW_UPPER = 0.25          # score < 0.25  → "Low Risk"
GRADE_MEDIUM_UPPER = 0.55       # 0.25 ≤ score < 0.55 → "Medium Risk"
#                               # score ≥ 0.55  → "High Risk"

NARRATIVE_MAP = {
    "Low": {
        "tone": ["favorable", "manageable", "low concern"],
        "banned": ["elevated", "significant", "action required"],
        "recommendation_base": "proceed",
    },
    "Medium": { ... },
    "High": { ... },
}
```

---

## 6. ALGORITHM GAPS

### Bispecific Engine
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/bispecific_engine.py`
**Lines:** 1–600+

**Chain Biophysical Computation** (Lines 70–150):
```python
def _compute_biophysical(self):
    """Compute biophysical properties from sequence using Biopython."""
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        seq_clean = self.sequence.upper()
        analysis = ProteinAnalysis(seq_clean)
        self.pI = round(analysis.isoelectric_point(), 2)
        self.mw_kda = round(analysis.molecular_weight() / 1000.0, 1)
        self.gravy = round(analysis.gravy(), 3)
        self.hydrophobicity = round(
            max(0.0, min(1.0, (self.gravy + 2.0) / 4.0)), 3
        )
        # Deamidation hotspots: N-G, N-S
        deam = 0
        for i in range(len(seq_clean) - 1):
            if seq_clean[i] == "N" and seq_clean[i + 1] in ("G", "S"):
                deam += 1
        self.deam_sites = max(1, deam)
        # Oxidation: Met count
        self.ox_sites = max(1, seq_clean.count("M"))
    except ImportError:
        log.warning("Biopython not available; using fallback estimation")
        self._estimate_biophysical_from_composition()
```

**Status:** ✅ COMPLETE (no stubs)

### Formulation Twin
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/formulation_twin.py`
**Lines:** 1–600+

**Buffer Catalog** (Lines 97–150):
```python
BUFFER_CATALOG: Dict[str, BufferSystem] = {
    "histidine": BufferSystem(
        name="histidine",
        full_name="L-Histidine / Histidine-HCl",
        optimal_ph_low=5.5,
        optimal_ph_high=6.5,
        pka_values=[6.0],
        ionic_strength_factor=0.8,
        viscosity_modifier=0.05,
        stabilization_bonus=0.08,
    ),
    "citrate": BufferSystem(...),
    # ... 4 more buffers
}
```

**Status:** ✅ COMPLETE (no stubs)

### Preclinical Twin
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/preclinical_twin.py`
**Lines:** 1–500+

**PK Prediction Function** (Lines 145–200):
```python
def predict_human_half_life(
    global_pi: float,
    hydrophobicity: float = 0.35,
    liability_density: float = 30.0,
    fcrn_binding_motif_intact: bool = True,
    mw_kda: float = 150.0,
    # ... more parameters
) -> Dict[str, Any]:
    """
    Empirical model: baseline ~21 days for standard IgG1
    - Penalizes extreme pI (>9.0 or <5.5)
    - Penalizes high hydrophobicity
    - Penalizes high liability density (aggregation proxy)
    - Accounts for FcRn binding motif integrity
    """
    # Full implementation present
```

**Status:** ✅ COMPLETE (no stubs)

### Immunogenicity Twin
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/immunogenicity_twin.py`
**Lines:** 1–500+

**MHC-II Scanning** (Lines 78–150):
```python
# MHC-II anchor position weights
_ANCHOR_WEIGHTS = {0: 2.5, 3: 2.0, 5: 1.5, 8: 2.0}  # 0-indexed

# Human germline frameworks
_HUMAN_VH_FR1 = "EVQLVESGGGLVQPGGSLRLSCAAS"
_HUMAN_VH_FR2 = "WVRQAPGKGLEWVS"
_HUMAN_VH_FR3 = "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR"
_HUMAN_VK_FR1 = "DIQMTQSPSSLSASVGDRVTITC"
_HUMAN_VK_FR2 = "WYQQKPGKAPKLLIY"
_HUMAN_VK_FR3 = "GVPSRFSGSGSGTDFTLTISSLQPEDFATYFC"
```

**Status:** ✅ COMPLETE (no stubs)

### Generative Engineer
**File:** `/sessions/optimistic-modest-pascal/mnt/ProtePilot/src/generative_engineer.py`
**Lines:** 1–600+

**Liability Fix Strategies** (Lines 70–120):
```python
LIABILITY_FIX_STRATEGIES = {
    "Oxidation (Met)": {
        "pattern": re.compile(r"M"),
        "fix": lambda seq, pos: seq[:pos] + "L" + seq[pos + 1:],
        "replacement": "L",
        "rationale": "Met→Leu: hydrophobic isostere, eliminates sulfoxide formation",
    },
    "Deamidation (NG)": {
        "pattern": re.compile(r"NG"),
        "fix": lambda seq, pos: seq[:pos] + "QG" + seq[pos + 2:],
        "replacement": "QG",
        "rationale": "Asn→Gln at NG: amide isostere blocks succinimide intermediate",
    },
    # ... more strategies
}
```

**Status:** ✅ COMPLETE (no stubs)

---

## Summary of Evidence

All findings in DEEP_ANALYSIS_REPORT.md and FINDINGS_SUMMARY.txt are directly sourced from:

1. **Validation Output:** validation_results_v3.json (16 tests, all PASS)
2. **Code Review:** 64 Python modules in src/; trace of imports, data flow
3. **State Management:** workspace_manager.py + app.py session_state usage
4. **Report Architecture:** report_assembler.py + report_schema.py unified design
5. **Algorithm Completeness:** 5 engine modules inspected for stubs/TODOs

**Last Updated:** 2026-03-14
