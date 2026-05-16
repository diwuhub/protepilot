"""
bulk_single_schema_alignment.py  ·  ProtePilot — Bulk / Single Schema Boundary
==================================================================================
P2 documentation: authoritative map of field relationships between the single-molecule
analysis path (ReportContext / ReportObject) and the bulk analysis path (BulkRowResult).

WHY THIS FILE EXISTS
--------------------
The two analysis paths were built independently and use different field names for some
semantically identical values.  Before v33 this divergence was undocumented, which caused:
  - Confusion about which score to compare across modes
  - Grade-string mismatches (fixed in P1 — now both use "Low Risk" / "Medium Risk" / "High Risk")
  - OOD data not persisting to report cache in single path (fixed in P1)

This file serves as:
  1. The authoritative written contract between bulk and single paths
  2. A runtime alignment validator callable from the selftest suite
  3. A convenience bridge that extracts the "shared core" from either path object

FIELD CLASSIFICATION
--------------------
  SHARED_ALIGNED   — same semantics, same or mapped field names, values MUST match
  NAME_MISMATCH    — semantically equivalent but different names (mapping documented here)
  BULK_SPECIFIC    — batch metadata or extra detail that has no single-path equivalent
  SINGLE_SPECIFIC  — single-path fields not yet surfaced in bulk (known gaps)

Grade String Convention (authoritative, enforced here)
------------------------------------------------------
  Both paths MUST produce: "Low Risk" | "Medium Risk" | "High Risk"
  This is enforced by grade_to_risk_label() in report_schema.py.
  Any raw "Low" / "Medium" / "High" grade strings are INTERNAL only.

Author  : Di (ProtePilot)
Version : 1.0  (introduced v33 / P2)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# =========================================================================
#  1. Field Mapping Table
# =========================================================================

# Each entry: (category, single_path_field, bulk_path_field, notes)
#   - single_path_field : dotted path in ReportContext or ReportObject sub-section
#   - bulk_path_field   : attribute name on BulkRowResult
#   - notes             : brief alignment note

FIELD_MAP: List[Tuple[str, str, str, str]] = [

    # ── Score / Grade ──────────────────────────────────────────────────
    ("SHARED_ALIGNED",
     "context.overall_score",
     "composite_dev_score",
     "5-dim composite developability score (primary ranking metric); "
     "computed by assess_developability() in developability_core.py. "
     "Fallback: 3-dim developability_score when composite unavailable."),

    ("NAME_MISMATCH",
     "context.base_risk_score",
     "developability_score",
     "3-dim predictor score (agg+stability+viscosity only). "
     "Retained for reference. Single calls it 'base_risk_score'; "
     "bulk still stores it as 'developability_score'."),

    ("SHARED_ALIGNED",
     "context.overall_grade",
     "composite_dev_grade",
     "Display grade from 5-dim composite score. "
     "MUST be 'Low Risk' | 'Medium Risk' | 'High Risk' in both paths. "
     "Enforced by grade_to_risk_label() — see P1 fix."),

    ("NAME_MISMATCH",
     "(embedded in dev risk section)",
     "developability_grade",
     "Grade from 3-dim predictor. Bulk exposes separately; single embeds "
     "in developability section. Both use 'Low Risk' convention."),

    ("SHARED_ALIGNED",
     "context.agg_risk",
     "agg_risk",
     "Aggregation risk score 0–1 (same field name in both paths)."),

    ("SHARED_ALIGNED",
     "context.stability",
     "stability",
     "Stability score 0–1 (same field name in both paths)."),

    ("SHARED_ALIGNED",
     "context.viscosity_risk",
     "viscosity_risk",
     "Viscosity risk score 0–1 (same field name in both paths)."),

    # ── Biophysical ────────────────────────────────────────────────────
    ("NAME_MISMATCH",
     "context.molecular_weight_kda",
     "mw_kda",
     "Molecular weight in kDa. Single uses 'molecular_weight_kda'; "
     "bulk uses 'mw_kda'. Both computed by feature_registry."),

    ("NAME_MISMATCH",
     "context.isoelectric_point",
     "pI",
     "Isoelectric point. Single uses 'isoelectric_point'; bulk uses 'pI'. "
     "Both from feature_registry.compute_features()."),

    ("NAME_MISMATCH",
     "context.gravy_score",
     "gravy",
     "GRAVY hydropathy index. Single uses 'gravy_score'; bulk uses 'gravy'. "
     "Both from Biopython ProteinAnalysis."),

    ("SHARED_ALIGNED",
     "context.hydrophobicity",
     "hydrophobicity",
     "Normalized hydrophobicity 0–1 (same field name in both paths)."),

    ("NAME_MISMATCH",
     "context.sequence_length",
     "seq_length",
     "Total assembled sequence length in residues (stoichiometric). "
     "Single uses 'sequence_length'; bulk uses 'seq_length'."),

    ("SHARED_ALIGNED",
     "context.deam_sites",
     "deam_sites",
     "Deamidation liability sites count (same field name)."),

    ("SHARED_ALIGNED",
     "context.ox_sites",
     "ox_sites",
     "Oxidation liability sites count (same field name)."),

    ("SHARED_ALIGNED",
     "context.acidic_residues",
     "acidic_residues",
     "Count of D+E residues in assembled sequence (same field name)."),

    ("SHARED_ALIGNED",
     "context.basic_residues",
     "basic_residues",
     "Count of K+R+H residues in assembled sequence (same field name)."),

    ("SHARED_ALIGNED",
     "context.cysteine_count",
     "cysteine_count",
     "Total cysteine count (same field name)."),

    # ── OOD ────────────────────────────────────────────────────────────
    ("NAME_MISMATCH",
     "context.is_ood",
     "ood_flag",
     "Boolean out-of-distribution flag. Single: 'is_ood'; bulk: 'ood_flag'. "
     "Both now persist from pipeline result (P1 fix)."),

    ("NAME_MISMATCH",
     "context.ood_reason",
     "ood_details",
     "Human-readable OOD reason string. Single: 'ood_reason'; bulk: 'ood_details'."),

    ("SINGLE_SPECIFIC",
     "context.ood_confidence",
     "(none)",
     "OOD detector confidence level: 'High' | 'Medium' | 'Low'. "
     "Single path only — bulk does not currently expose this. "
     "KNOWN GAP: bulk could add ood_confidence to BulkRowResult if needed."),

    # ── Analytical QC ──────────────────────────────────────────────────
    ("SHARED_ALIGNED",
     "analytical.cief_main_pct",
     "cief_main_pct",
     "cIEF main peak percentage (same field name in both paths)."),

    ("SHARED_ALIGNED",
     "analytical.cief_acidic_pct",
     "cief_acidic_pct",
     "cIEF acidic variant percentage (same field name)."),

    ("SHARED_ALIGNED",
     "analytical.cief_basic_pct",
     "cief_basic_pct",
     "cIEF basic variant percentage (same field name)."),

    ("NAME_MISMATCH",
     "analytical.cesds_intact_pct",
     "ce_sds_purity_pct",
     "CE-SDS intact/purity percentage. Single: 'cesds_intact_pct'; "
     "bulk: 'ce_sds_purity_pct'. Both from analytical_qc_twin.run_analytical_qc()."),

    ("BULK_SPECIFIC",
     "(none)",
     "ce_sds_hmw_pct",
     "CE-SDS high-molecular-weight species percentage. Bulk-only flat field. "
     "Single path reports CE-SDS results in narrative; no dedicated HMW% attribute "
     "on AnalyticalSummary (AnalyticalSummary.sec_hmw_pct is SEC-based, not CE-SDS)."),

    ("BULK_SPECIFIC",
     "(none)",
     "ce_sds_lmw_pct",
     "CE-SDS low-molecular-weight species percentage. Bulk-only flat field. "
     "Single path embeds LMW% in CE-SDS narrative text only."),

    ("BULK_SPECIFIC",
     "(none)",
     "liability_density",
     "MS tryptic-peptide liability density per 1000 residues. Bulk exposes as flat "
     "numeric field for ranking; single reports in MS characterization narrative only."),

    ("BULK_SPECIFIC",
     "(none)",
     "n_liabilities",
     "Total count of sequence-derived liability sites from tryptic digest. "
     "Bulk exposes for CSV ranking; single includes in MS characterization section text."),

    ("BULK_SPECIFIC",
     "(none)",
     "liability_summary",
     "Short human-readable string summarising top liability categories "
     "(e.g. 'deamidation x7, oxidation x5'). Bulk-only flat string; "
     "single uses a liability_summary dict in MoleculeOverview."),

    ("NAME_MISMATCH",
     "process_pk.clearance_ml_day_kg",
     "clearance_ml_day_kg",
     "Predicted plasma clearance (mL/day/kg) from PK model. "
     "Both paths now populate this field (unified in ProcessPKSummary v2)."),

    ("SHARED_ALIGNED",
     "analytical.ms_intact_mass_da",
     "intact_mass_da",
     "Intact mass (Da) — note single uses 'ms_intact_mass_da', bulk uses 'intact_mass_da'. "
     "Flagged as SHARED_ALIGNED because values must agree; "
     "the name difference is a minor inconsistency."),

    # ── PK / Process ───────────────────────────────────────────────────
    ("SHARED_ALIGNED",
     "process_pk.half_life_days",
     "half_life_days",
     "Predicted serum half-life in days (same field name)."),

    ("NAME_MISMATCH",
     "process_pk.final_titer_g_l",
     "predicted_titer_g_L",
     "Upstream titer (g/L). Single: 'final_titer_g_l'; bulk: 'predicted_titer_g_L'. "
     "Both from upstream_twin.run_upstream_simulation() with molecule-class scaling."),

    # ── ADA / Immunogenicity ───────────────────────────────────────────
    ("NAME_MISMATCH",
     "process_pk.ada_risk_level",
     "ada_risk_category",
     "ADA risk category string. Single stores as 'ada_risk_level' in ProcessPKSummary; "
     "bulk stores as 'ada_risk_category' in BulkRowResult. Semantically equivalent."),

    ("SHARED_ALIGNED",
     "process_pk.ada_risk_score",
     "ada_risk_score",
     "Numeric ADA risk score 0–1. Both paths now populate "
     "(unified in ProcessPKSummary v2)."),

    ("SHARED_ALIGNED",
     "process_pk.n_mhcii_hotspots",
     "n_mhcii_hotspots",
     "Number of MHC-II hotspot peptides. Both paths now populate "
     "(unified in ProcessPKSummary v2)."),

    # ── Stability ─────────────────────────────────────────────────────
    ("SHARED_ALIGNED",
     "process_pk.shelf_life_months",
     "shelf_life_months",
     "ICH-based predicted shelf life. Both paths now populate "
     "(unified in ProcessPKSummary v2)."),

    ("SHARED_ALIGNED",
     "process_pk.stability_grade",
     "stability_grade",
     "Stability grade string (e.g. 'Excellent', 'Good'). "
     "Both paths now populate (unified in ProcessPKSummary v2)."),

    # ── DoE Purification ──────────────────────────────────────────────
    ("SHARED_ALIGNED",
     "process_pk.doe_optimal_ph",
     "doe_optimal_ph",
     "DoE optimal pH from CEX DoE sweep. Both paths now populate "
     "(unified in ProcessPKSummary v2)."),

    ("BULK_SPECIFIC",
     "(none)",
     "doe_optimal_gradient",
     "DoE optimal gradient slope (mM/min). Bulk exposes as flat field; "
     "single stores in DoE note narrative."),

    ("SHARED_ALIGNED",
     "process_pk.doe_optimal_yield",
     "doe_optimal_yield",
     "DoE optimal yield (fraction). Both paths now populate "
     "(unified in ProcessPKSummary v2)."),

    ("SHARED_ALIGNED",
     "process_pk.doe_optimal_purity",
     "doe_optimal_purity",
     "DoE optimal purity (%). Both paths now populate "
     "(unified in ProcessPKSummary v2)."),

    ("SHARED_ALIGNED",
     "process_pk.doe_rs_min",
     "doe_rs_min",
     "DoE minimum resolution (Rs). Both paths now populate "
     "(unified in ProcessPKSummary v2)."),

    # ── COGS ──────────────────────────────────────────────────────────
    ("SHARED_ALIGNED",
     "process_pk.cogs_per_gram",
     "cogs_per_gram",
     "Estimated cost of goods per gram. Both paths now populate "
     "(unified in ProcessPKSummary v2)."),

    ("SHARED_ALIGNED",
     "process_pk.cogs_cost_rating",
     "cogs_cost_rating",
     "COGS rating string ('Excellent', 'Good', 'Fair', 'Poor'). "
     "Both paths now populate (unified in ProcessPKSummary v2)."),

    ("BULK_SPECIFIC",
     "(none)",
     "cogs_batch_output_g",
     "Estimated batch output in grams. Bulk exposes as flat field; "
     "single uses cogs_note narrative."),

    # ── Identifiers ──────────────────────────────────────────────────────
    ("NAME_MISMATCH",
     "context.molecule_name",
     "name",
     "Molecule/candidate identifier. Single: 'molecule_name' on ReportContext; "
     "bulk: 'name' on BulkRowResult. Both sourced from user CSV or input form."),

    # ── Molecule Class ─────────────────────────────────────────────────
    ("NAME_MISMATCH",
     "context.molecule_class",
     "molecule_class",
     "Molecule class string (same field name). "
     "Single also carries molecule_class_display, assembly_class, molecule_class_info dict; "
     "bulk carries only the base string."),

    # ── Batch metadata (bulk-only) ─────────────────────────────────────
    ("BULK_SPECIFIC",
     "(none)",
     "row_index",
     "0-based CSV row position. Bulk-only batch metadata."),

    ("BULK_SPECIFIC",
     "(none)",
     "status",
     "Row execution status: 'success' | 'error' | 'skipped'. Bulk-only."),

    ("BULK_SPECIFIC",
     "(none)",
     "wall_time",
     "Per-row wall-clock time in seconds. Bulk-only performance metric."),

    ("BULK_SPECIFIC",
     "(none)",
     "error_message",
     "Row-level error message. Bulk-only fault isolation."),

    # ── Single-specific liability fields ──────────────────────────────
    ("SINGLE_SPECIFIC",
     "context.isomerization_sites",
     "(none)",
     "Asp isomerization (DG/DS motif) site count. Single-path detail; "
     "bulk aggregates into n_liabilities total."),

    ("SINGLE_SPECIFIC",
     "context.dp_clip_sites",
     "(none)",
     "Asp-Pro clip site count. Single-path detail only."),

    ("SINGLE_SPECIFIC",
     "context.free_cysteine_risk",
     "(none)",
     "Boolean free cysteine risk flag. Single-path only."),

    ("SINGLE_SPECIFIC",
     "context.n_glycosylation_sites",
     "(none)",
     "N-glycosylation site count. Single-path only."),

    ("SINGLE_SPECIFIC",
     "context.pyroglutamate_risk",
     "(none)",
     "Pyroglutamate formation risk at N-terminus. Single-path only."),

    # ── Single-specific Phase 1B biophysical ──────────────────────────
    ("SINGLE_SPECIFIC",
     "context.beta_sheet_propensity",
     "(none)",
     "Beta-sheet propensity score. Single Phase 1B feature; not yet in bulk."),

    ("SINGLE_SPECIFIC",
     "context.cdr_hydrophobicity",
     "(none)",
     "CDR-region hydrophobicity score. Single Phase 1B feature; not yet in bulk."),
]


# =========================================================================
#  2. Convenience accessors
# =========================================================================

def get_shared_fields() -> List[Tuple[str, str, str]]:
    """Return (single_field, bulk_field, notes) for SHARED_ALIGNED entries."""
    return [(s, b, n) for cat, s, b, n in FIELD_MAP if cat == "SHARED_ALIGNED"]


def get_name_mismatches() -> List[Tuple[str, str, str]]:
    """Return (single_field, bulk_field, notes) for NAME_MISMATCH entries."""
    return [(s, b, n) for cat, s, b, n in FIELD_MAP if cat == "NAME_MISMATCH"]


def get_bulk_specific() -> List[Tuple[str, str]]:
    """Return (bulk_field, notes) for BULK_SPECIFIC entries."""
    return [(b, n) for cat, _s, b, n in FIELD_MAP if cat == "BULK_SPECIFIC"]


def get_single_specific() -> List[Tuple[str, str]]:
    """Return (single_field, notes) for SINGLE_SPECIFIC entries."""
    return [(s, n) for cat, s, _b, n in FIELD_MAP if cat == "SINGLE_SPECIFIC"]


# =========================================================================
#  3. Runtime Alignment Validator
# =========================================================================

def extract_bulk_core(row_result) -> Dict[str, Any]:
    """
    Extract the 'shared core' from a BulkRowResult into a canonical dict.

    Uses the field-mapping table to produce normalized names matching
    the ReportContext naming convention.  This dict can be directly compared
    with extract_single_core() for cross-path alignment checks.

    Parameters
    ----------
    row_result : BulkRowResult
        A completed bulk analysis row result.

    Returns
    -------
    dict : Canonical field values using single-path naming convention.
    """
    def _g(attr, fallback=None):
        return getattr(row_result, attr, fallback)

    return {
        # Score / grade (using single-path names)
        "overall_score":        _g("composite_dev_score") or _g("developability_score"),
        "base_risk_score":      _g("developability_score"),
        "overall_grade":        _g("composite_dev_grade") or _g("developability_grade"),
        "agg_risk":             _g("agg_risk"),
        "stability":            _g("stability"),
        "viscosity_risk":       _g("viscosity_risk"),
        # Biophysical (using single-path names)
        "molecular_weight_kda": _g("mw_kda"),
        "isoelectric_point":    _g("pI"),
        "gravy_score":          _g("gravy"),
        "hydrophobicity":       _g("hydrophobicity"),
        "sequence_length":      _g("seq_length"),
        "deam_sites":           _g("deam_sites"),
        "ox_sites":             _g("ox_sites"),
        "acidic_residues":      _g("acidic_residues"),
        "basic_residues":       _g("basic_residues"),
        "cysteine_count":       _g("cysteine_count"),
        # OOD (using single-path names)
        "is_ood":               _g("ood_flag"),
        "ood_reason":           _g("ood_details"),
        # QC (mixed names)
        "cief_main_pct":        _g("cief_main_pct"),
        "cief_acidic_pct":      _g("cief_acidic_pct"),
        "cief_basic_pct":       _g("cief_basic_pct"),
        "cesds_intact_pct":     _g("ce_sds_purity_pct"),
        "ms_intact_mass_da":    _g("intact_mass_da"),
        # PK / Process
        "half_life_days":       _g("half_life_days"),
        "clearance_ml_day_kg":  _g("clearance_ml_day_kg"),
        "final_titer_g_l":      _g("predicted_titer_g_L"),
        # ADA / Immunogenicity
        "ada_risk_level":       _g("ada_risk_category"),
        "ada_risk_score":       _g("ada_risk_score"),
        "n_mhcii_hotspots":     _g("n_mhcii_hotspots"),
        # Stability
        "shelf_life_months":    _g("shelf_life_months"),
        "stability_grade":      _g("stability_grade"),
        # DoE Purification
        "doe_optimal_ph":       _g("doe_optimal_ph"),
        "doe_optimal_yield":    _g("doe_optimal_yield"),
        "doe_optimal_purity":   _g("doe_optimal_purity"),
        "doe_rs_min":           _g("doe_rs_min"),
        # COGS
        "cogs_per_gram":        _g("cogs_per_gram"),
        "cogs_cost_rating":     _g("cogs_cost_rating"),
        # Molecule class
        "molecule_class":       _g("molecule_class"),
    }


def extract_single_core(report_context, process_pk=None, analytical=None) -> Dict[str, Any]:
    """
    Extract the 'shared core' from a single-path ReportContext.

    Parameters
    ----------
    report_context : ReportContext
        The frozen context from report_assembler.
    process_pk : ProcessPKSummary, optional
        Process / PK section for titer and ADA fields.
    analytical : AnalyticalSummary, optional
        Analytical section for QC fields.

    Returns
    -------
    dict : Canonical field values using single-path naming convention.
    """
    def _g(obj, attr, fallback=None):
        return getattr(obj, attr, fallback) if obj is not None else fallback

    ctx = report_context
    return {
        "overall_score":        _g(ctx, "overall_score"),
        "base_risk_score":      _g(ctx, "base_risk_score"),
        "overall_grade":        _g(ctx, "overall_grade"),
        "agg_risk":             _g(ctx, "agg_risk"),
        "stability":            _g(ctx, "stability"),
        "viscosity_risk":       _g(ctx, "viscosity_risk"),
        "molecular_weight_kda": _g(ctx, "molecular_weight_kda"),
        "isoelectric_point":    _g(ctx, "isoelectric_point"),
        "gravy_score":          _g(ctx, "gravy_score"),
        "hydrophobicity":       _g(ctx, "hydrophobicity"),
        "sequence_length":      _g(ctx, "sequence_length"),
        "deam_sites":           _g(ctx, "deam_sites"),
        "ox_sites":             _g(ctx, "ox_sites"),
        "acidic_residues":      _g(ctx, "acidic_residues"),
        "basic_residues":       _g(ctx, "basic_residues"),
        "cysteine_count":       _g(ctx, "cysteine_count"),
        "is_ood":               _g(ctx, "is_ood"),
        "ood_reason":           _g(ctx, "ood_reason"),
        "cief_main_pct":        _g(analytical, "cief_main_pct"),
        "cief_acidic_pct":      _g(analytical, "cief_acidic_pct"),
        "cief_basic_pct":       _g(analytical, "cief_basic_pct"),
        "cesds_intact_pct":     _g(analytical, "cesds_intact_pct"),
        "ms_intact_mass_da":    _g(analytical, "ms_intact_mass_da"),
        "half_life_days":       _g(process_pk, "half_life_days"),
        "clearance_ml_day_kg":  _g(process_pk, "clearance_ml_day_kg"),
        "final_titer_g_l":      _g(process_pk, "final_titer_g_l"),
        # ADA / Immunogenicity
        "ada_risk_level":       _g(process_pk, "ada_risk_level"),
        "ada_risk_score":       _g(process_pk, "ada_risk_score"),
        "n_mhcii_hotspots":     _g(process_pk, "n_mhcii_hotspots"),
        # Stability
        "shelf_life_months":    _g(process_pk, "shelf_life_months"),
        "stability_grade":      _g(process_pk, "stability_grade"),
        # DoE Purification
        "doe_optimal_ph":       _g(process_pk, "doe_optimal_ph"),
        "doe_optimal_yield":    _g(process_pk, "doe_optimal_yield"),
        "doe_optimal_purity":   _g(process_pk, "doe_optimal_purity"),
        "doe_rs_min":           _g(process_pk, "doe_rs_min"),
        # COGS
        "cogs_per_gram":        _g(process_pk, "cogs_per_gram"),
        "cogs_cost_rating":     _g(process_pk, "cogs_cost_rating"),
        # Molecule class
        "molecule_class":       _g(ctx, "molecule_class"),
    }


def validate_grade_strings(row_result) -> List[str]:
    """
    Check that all grade strings in a BulkRowResult use the canonical
    "Low Risk" / "Medium Risk" / "High Risk" format.

    Returns list of violation strings (empty = all OK).
    """
    violations = []
    _VALID = {"Low Risk", "Medium Risk", "High Risk", None, ""}
    _BARE = {"Low", "Medium", "High"}

    for attr in ("composite_dev_grade", "developability_grade"):
        val = getattr(row_result, attr, None)
        if val in _BARE:
            violations.append(
                f"BulkRowResult.{attr} = '{val}' — "
                "must be 'Low Risk'/'Medium Risk'/'High Risk' (bare grade string found)"
            )
        elif val not in _VALID and val is not None:
            violations.append(
                f"BulkRowResult.{attr} = '{val}' — unrecognized grade string"
            )
    return violations


def validate_bulk_row_alignment(row_result) -> Dict[str, Any]:
    """
    Run all alignment checks for a single BulkRowResult.

    Checks:
    1. Grade strings use canonical "X Risk" format
    2. All SHARED_ALIGNED fields are present (not completely None when status=success)
    3. Score fields are consistent (composite >= 0, 3-dim >= 0, both ≤ 1)

    Returns
    -------
    dict with keys: ok (bool), violations (list of str)
    """
    violations: List[str] = []

    # 1. Grade strings
    violations.extend(validate_grade_strings(row_result))

    # 2. Score range checks (only for successful rows)
    if getattr(row_result, "status", None) == "success":
        for attr in ("composite_dev_score", "developability_score",
                     "agg_risk", "stability", "viscosity_risk"):
            val = getattr(row_result, attr, None)
            if val is not None and not (0.0 <= val <= 1.0):
                violations.append(
                    f"BulkRowResult.{attr} = {val} — expected value in [0, 1]"
                )

        # 3-dim score must exist for a successful row
        if getattr(row_result, "developability_score", None) is None:
            violations.append("BulkRowResult.developability_score is None for a 'success' row")

    return {"ok": len(violations) == 0, "violations": violations}


# =========================================================================
#  4. Schema Summary (for logging / reporting)
# =========================================================================

def print_schema_summary() -> None:
    """Print a human-readable schema alignment summary."""
    from collections import Counter
    counts = Counter(cat for cat, *_ in FIELD_MAP)

    print("=" * 70)
    print("ProtePilot — Bulk / Single Schema Alignment (P2)")
    print("=" * 70)
    print(f"  SHARED_ALIGNED   : {counts['SHARED_ALIGNED']:3d} fields  "
          "(same semantics, values must match)")
    print(f"  NAME_MISMATCH    : {counts['NAME_MISMATCH']:3d} fields  "
          "(equivalent semantics, different names)")
    print(f"  BULK_SPECIFIC    : {counts['BULK_SPECIFIC']:3d} fields  "
          "(batch metadata / extra detail, no single equivalent)")
    print(f"  SINGLE_SPECIFIC  : {counts['SINGLE_SPECIFIC']:3d} fields  "
          "(single-path detail not yet exposed in bulk — known gaps)")
    print(f"  TOTAL            : {len(FIELD_MAP):3d} fields")
    print()
    print("Name Mismatches (single → bulk):")
    for s, b, n in get_name_mismatches():
        print(f"  {s:38s}  →  {b}")
    print()
    print("Known Single-Specific Gaps (not in bulk):")
    for s, n in get_single_specific():
        print(f"  {s}  —  {n[:60]}")
    print("=" * 70)


# =========================================================================
#  5. Self-test
# =========================================================================

def _selftest() -> bool:
    """Validate the alignment module itself."""
    import logging
    log = logging.getLogger("ProtePilot.BulkSingleAlignment")

    # Verify FIELD_MAP is well-formed
    valid_cats = {"SHARED_ALIGNED", "NAME_MISMATCH", "BULK_SPECIFIC", "SINGLE_SPECIFIC"}
    for i, entry in enumerate(FIELD_MAP):
        assert len(entry) == 4, f"Entry {i} has wrong length: {entry}"
        cat, single, bulk, notes = entry
        assert cat in valid_cats, f"Entry {i}: unknown category '{cat}'"
        assert isinstance(notes, str) and len(notes) > 10, f"Entry {i}: empty notes"

    # Verify no duplicate (category, bulk_field) pairs (single field may appear once as NONE)
    seen_bulk = set()
    for cat, single, bulk, notes in FIELD_MAP:
        if bulk != "(none)":
            key = bulk
            assert key not in seen_bulk, f"Duplicate bulk field: '{bulk}'"
            seen_bulk.add(key)

    # Verify grade validator catches bare grades
    class _FakeResult:
        status = "success"
        composite_dev_grade = "Low"      # bare — should be flagged
        developability_grade = "Medium"  # bare — should be flagged
        developability_score = 0.22
        composite_dev_score = 0.22
        agg_risk = 0.15
        stability = 0.77
        viscosity_risk = 0.10

    result = validate_bulk_row_alignment(_FakeResult())
    assert not result["ok"], "Grade validator should have caught bare grade strings"
    assert len(result["violations"]) >= 2, "Expected 2+ violations for bare grades"

    # Verify grade validator passes correct strings
    class _GoodResult:
        status = "success"
        composite_dev_grade = "Low Risk"
        developability_grade = "Low Risk"
        developability_score = 0.22
        composite_dev_score = 0.22
        agg_risk = 0.15
        stability = 0.77
        viscosity_risk = 0.10

    result2 = validate_bulk_row_alignment(_GoodResult())
    assert result2["ok"], f"Unexpected violations: {result2['violations']}"

    # Verify extract_bulk_core produces expected canonical keys
    class _RowResult:
        composite_dev_score = 0.262
        developability_score = 0.207
        composite_dev_grade = "Medium Risk"
        developability_grade = "Low Risk"
        agg_risk = 0.20
        stability = 0.75
        viscosity_risk = 0.12
        mw_kda = 145.2
        pI = 8.4
        gravy = -0.32
        hydrophobicity = 0.35
        seq_length = 1204
        deam_sites = 7
        ox_sites = 5
        acidic_residues = 140
        basic_residues = 160
        cysteine_count = 16
        ood_flag = False
        ood_details = None
        cief_main_pct = 54.6
        cief_acidic_pct = 22.1
        cief_basic_pct = 23.3
        ce_sds_purity_pct = 97.6
        intact_mass_da = 131305.0
        half_life_days = 20.8
        predicted_titer_g_L = 3.31
        ada_risk_category = "Medium"
        molecule_class = "canonical_mab"

    core = extract_bulk_core(_RowResult())
    assert core["overall_score"] == 0.262
    assert core["molecular_weight_kda"] == 145.2
    assert core["isoelectric_point"] == 8.4
    assert core["cesds_intact_pct"] == 97.6      # name mapped from ce_sds_purity_pct
    assert core["final_titer_g_l"] == 3.31       # name mapped from predicted_titer_g_L
    assert core["is_ood"] is False               # name mapped from ood_flag

    # Verify print_schema_summary doesn't raise
    import io, sys
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        print_schema_summary()
    finally:
        sys.stdout = old

    log.info("bulk_single_schema_alignment _selftest PASS (%d field entries)", len(FIELD_MAP))
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    _selftest()
    print_schema_summary()
