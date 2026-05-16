"""
bulk_schema.py  ·  ProtePilot — Bulk Analysis Input Schema & Validation
=========================================================================
B7: Type-homogeneous batch design — one CSV = one molecule type.

Users select a molecule type before uploading. The CSV column schema
adapts to that type, and every row shares the same assembly rule.
This avoids the complexity of per-row assembly specification while
keeping the bulk pipeline simple and robust.

Supported batch types
---------------------
- canonical_mab   : name, HC, LC              → 2×HC + 2×LC
- bispecific      : name, HC1, LC1, HC2, LC2  → HC1·LC1 / HC2·LC2
- scfv            : name, scFv                → single chain
- nanobody        : name, VHH                 → single chain
- fc_fusion       : name, Fc, partner         → 2×(Fc+partner)
- peptide         : name, peptide             → single chain
- adc             : name, HC, LC, DAR         → IgG + DAR metadata
- fusion_protein  : name, chain1, chain2      → chain1 + chain2

Design decision (B1): homogeneous batches. Mixed-type CSVs are rejected.
Users group molecules by type and upload separate batches.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("ProtePilot.BulkSchema")


# ═══════════════════════════════════════════════════════════════════════
#  Batch Type Definitions
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BatchTypeSpec:
    """Schema specification for a single batch type."""
    key: str                          # Internal identifier
    display_name: str                 # UI label
    molecule_class: str               # Maps to MoleculeClass enum value
    required_columns: Tuple[str, ...] # CSV columns (lowercase, order matters)
    optional_columns: Tuple[str, ...] = ()
    chain_count: int = 1              # How many distinct chain sequences per row
    assembly_description: str = ""    # Human-readable assembly rule

    @property
    def all_columns(self) -> Tuple[str, ...]:
        return ("name",) + self.required_columns + self.optional_columns


BATCH_TYPES: Dict[str, BatchTypeSpec] = {}

def _register(*specs: BatchTypeSpec) -> None:
    for s in specs:
        BATCH_TYPES[s.key] = s

_register(
    BatchTypeSpec(
        key="canonical_mab",
        display_name="Standard IgG (mAb)",
        molecule_class="canonical_mab",
        required_columns=("hc", "lc"),
        chain_count=2,
        assembly_description="2×HC + 2×LC  (standard IgG tetramer)",
    ),
    BatchTypeSpec(
        key="bispecific_4chain",
        display_name="Bispecific — 4-Chain (HC1·LC1 + HC2·LC2)",
        molecule_class="bispecific",
        required_columns=("hc1", "lc1", "hc2", "lc2"),
        chain_count=4,
        assembly_description="HC1·LC1 arm + HC2·LC2 arm  (standard knob-in-hole)",
    ),
    BatchTypeSpec(
        key="bispecific_3chain",
        display_name="Bispecific — 3-Chain (HC·LC + scFv arm)",
        molecule_class="bispecific",
        required_columns=("hc", "lc", "scfv_arm"),
        chain_count=3,
        assembly_description="HC·LC Fab arm + scFv arm  (e.g., BiTE-like with Fc)",
    ),
    BatchTypeSpec(
        key="bispecific_2chain",
        display_name="Bispecific — 2-Chain (KiH Fc heterodimer)",
        molecule_class="bispecific",
        required_columns=("chain_a", "chain_b"),
        chain_count=2,
        assembly_description="Chain A + Chain B heterodimer  (e.g., KiH Fc-fusion, DVD-Ig)",
    ),
    BatchTypeSpec(
        key="scfv",
        display_name="scFv (Single-Chain Variable Fragment)",
        molecule_class="single_domain",
        required_columns=("scfv",),
        chain_count=1,
        assembly_description="Single chain  (VH-linker-VL or VL-linker-VH)",
    ),
    BatchTypeSpec(
        key="nanobody",
        display_name="Nanobody / VHH",
        molecule_class="single_domain",
        required_columns=("vhh",),
        chain_count=1,
        assembly_description="Single-domain antibody (~12-15 kDa)",
    ),
    BatchTypeSpec(
        key="fc_fusion",
        display_name="Fc-Fusion Protein",
        molecule_class="fc_fusion",
        required_columns=("fc", "partner"),
        chain_count=2,
        assembly_description="2×(Fc + fusion partner)  (e.g., etanercept)",
    ),
    BatchTypeSpec(
        key="peptide",
        display_name="Therapeutic Peptide",
        molecule_class="peptide",
        required_columns=("peptide",),
        chain_count=1,
        assembly_description="Single peptide chain (<80 aa)",
    ),
    BatchTypeSpec(
        key="adc",
        display_name="Antibody-Drug Conjugate (ADC)",
        molecule_class="adc",
        required_columns=("hc", "lc"),
        optional_columns=("dar",),
        chain_count=2,
        assembly_description="IgG (2×HC + 2×LC) + drug-to-antibody ratio",
    ),
    BatchTypeSpec(
        key="fusion_protein",
        display_name="Fusion Protein (non-Fc)",
        molecule_class="fusion_protein",
        required_columns=("chain1", "chain2"),
        optional_columns=("chain3",),
        chain_count=2,
        assembly_description="Multi-domain fusion  (chain1 + chain2 [+ chain3])",
    ),
)


# ═══════════════════════════════════════════════════════════════════════
#  Parsed Row — one molecule candidate
# ═══════════════════════════════════════════════════════════════════════

_AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", re.IGNORECASE)

@dataclass
class BulkRow:
    """Validated row from a bulk CSV — ready for the pipeline."""
    row_index: int                    # 0-based row number in CSV
    name: str                         # Molecule name / identifier
    sequences: Dict[str, str]         # column_name → cleaned sequence
    metadata: Dict[str, Any] = field(default_factory=dict)  # DAR, notes, etc.
    combined_sequence: str = ""       # All chains concatenated (for feature calc)
    chains: List[Dict[str, Any]] = field(default_factory=list)
    assembly_chains: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None       # Validation error (None = valid)


# ═══════════════════════════════════════════════════════════════════════
#  CSV Template Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_csv_template(batch_type_key: str) -> str:
    """Generate a downloadable CSV template string for a given batch type."""
    spec = BATCH_TYPES.get(batch_type_key)
    if not spec:
        raise ValueError(f"Unknown batch type: {batch_type_key}")

    header = list(spec.all_columns)
    rows = [header]

    # Add one example row
    example = {"name": "Example_Molecule_1"}
    for col in spec.required_columns:
        if col in ("hc", "hc1", "hc2"):
            example[col] = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPT..."
        elif col in ("lc", "lc1", "lc2"):
            example[col] = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASF..."
        elif col in ("scfv", "vhh", "scfv_arm"):
            example[col] = "QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREF..."
        elif col in ("chain_a", "chain_b"):
            example[col] = "DKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVS..."
        elif col == "fc":
            example[col] = "DKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDP..."
        elif col == "partner":
            example[col] = "LPAQVAFTPYAPEPGSTCRLREYYDQTAQMCCSKCSPGQHAKVFCTKT..."
        elif col == "peptide":
            example[col] = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"
        elif col in ("chain1", "chain2", "chain3"):
            example[col] = "MKVLWAALLVTFLAGCQAKVEQAVE..."
        else:
            example[col] = "SEQUENCE"
    for col in spec.optional_columns:
        if col == "dar":
            example[col] = "4.0"
        else:
            example[col] = ""
    rows.append([example.get(c, "") for c in header])

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════
#  CSV Parsing & Validation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BulkParseResult:
    """Result of parsing a bulk CSV file."""
    batch_type: BatchTypeSpec
    rows: List[BulkRow]             # All rows (including invalid ones)
    valid_rows: List[BulkRow] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)  # Global errors
    warnings: List[str] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.rows)

    @property
    def n_valid(self) -> int:
        return len(self.valid_rows)

    @property
    def n_errors(self) -> int:
        return self.n_total - self.n_valid

    @property
    def is_ok(self) -> bool:
        return self.n_valid > 0 and len(self.errors) == 0


def _clean_sequence(raw: str) -> str:
    """Strip whitespace, numbers, special chars from a sequence string."""
    return re.sub(r"[^A-Za-z]", "", raw.strip())


def _validate_sequence(seq: str, col_name: str, row_idx: int,
                       min_len: int = 5, max_len: int = 3000) -> Optional[str]:
    """Validate a single amino acid sequence. Returns error string or None."""
    if not seq:
        return f"Row {row_idx + 1}: empty sequence in column '{col_name}'"
    if len(seq) < min_len:
        return f"Row {row_idx + 1}: sequence in '{col_name}' too short ({len(seq)} aa, min {min_len})"
    if len(seq) > max_len:
        return f"Row {row_idx + 1}: sequence in '{col_name}' too long ({len(seq)} aa, max {max_len})"
    if not _AA_PATTERN.match(seq):
        bad = set(seq.upper()) - set("ACDEFGHIKLMNPQRSTVWY")
        return f"Row {row_idx + 1}: invalid amino acid(s) {bad} in '{col_name}'"
    return None


def _build_chains_and_assembly(
    sequences: Dict[str, str],
    spec: BatchTypeSpec,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build chains and assembly_chains lists matching the single-molecule format."""
    chains: List[Dict[str, Any]] = []
    assembly: List[Dict[str, Any]] = []

    # Map column names to chain type labels.
    # IMPORTANT: Use "HC" / "LC" (not "Heavy" / "Light") to match the
    # system-wide convention expected by molecule_classifier._infer_chain_type,
    # analytical_twin, and the single-molecule path.  Numbered variants
    # (HC_1, HC_2, LC_1, LC_2) are also recognised by the classifier's
    # startswith/in checks after the fix below.
    _type_map = {
        "hc": "HC", "hc1": "HC_1", "hc2": "HC_2",
        "lc": "LC", "lc1": "LC_1", "lc2": "LC_2",
        "scfv": "scFv", "vhh": "VHH", "scfv_arm": "scFv_Arm",
        "chain_a": "Chain_A", "chain_b": "Chain_B",
        "fc": "Fc", "partner": "Fusion_Partner",
        "peptide": "Peptide",
        "chain1": "Chain_1", "chain2": "Chain_2", "chain3": "Chain_3",
    }

    # Assembly stoichiometry per batch type
    _stoich_map = {
        "canonical_mab": {"hc": 2, "lc": 2},
        "bispecific_4chain": {"hc1": 1, "lc1": 1, "hc2": 1, "lc2": 1},
        "bispecific_3chain": {"hc": 1, "lc": 2, "scfv_arm": 1},
        "bispecific_2chain": {"chain_a": 1, "chain_b": 1},
        "scfv": {"scfv": 1},
        "nanobody": {"vhh": 1},
        "fc_fusion": {"fc": 2, "partner": 2},
        "peptide": {"peptide": 1},
        "adc": {"hc": 2, "lc": 2},
        "fusion_protein": {"chain1": 1, "chain2": 1},
    }

    stoich = _stoich_map.get(spec.key, {})

    for col, seq in sequences.items():
        chain_type = _type_map.get(col, col.title())
        chain_entry = {
            "sequence": seq,
            "type": chain_type,
            "chain_type": chain_type,
            "length": len(seq),
        }
        chains.append(chain_entry)

        count = stoich.get(col, 1)
        asm_entry = {
            "sequence": seq,
            "type": chain_type,
            "chain_type": chain_type,
            "copy_number": count,
            "length": len(seq),
        }
        assembly.append(asm_entry)

    # ── Chain-length advisory: warn if a chain is unusually short ──
    # These are WARNINGS only — analysis always proceeds.  Short chains
    # may indicate truncated input, VH-only fragments, or copy-paste
    # errors.  The advisory helps users catch data-entry mistakes.
    _chain_expected_min = {
        "HC": 350, "HC_1": 350, "HC_2": 350,
        "LC": 150, "LC_1": 150, "LC_2": 150,
        "scFv": 200, "scFv_Arm": 200, "VHH": 80,
        "Fc": 200, "Fusion_Partner": 50, "Peptide": 5,
    }
    for asm_ch in assembly:
        ct = asm_ch.get("chain_type", "")
        expected = _chain_expected_min.get(ct, 0)
        if expected > 0 and asm_ch["length"] < expected:
            log.warning(
                "Chain %s is %d aa (expected >= %d for %s) — "
                "sequence may be truncated or incomplete",
                ct, asm_ch["length"], expected, ct,
            )

    return chains, assembly


def parse_bulk_csv(
    csv_content: str,
    batch_type_key: str,
    max_rows: int = 500,
) -> BulkParseResult:
    """
    Parse and validate a CSV for bulk analysis.

    Parameters
    ----------
    csv_content : str
        Raw CSV text content.
    batch_type_key : str
        Key into BATCH_TYPES (e.g., "canonical_mab").
    max_rows : int
        Safety limit on number of molecules.

    Returns
    -------
    BulkParseResult with valid_rows ready for pipeline processing.
    """
    spec = BATCH_TYPES.get(batch_type_key)
    if not spec:
        return BulkParseResult(
            batch_type=BatchTypeSpec(key="unknown", display_name="Unknown",
                                     molecule_class="unknown",
                                     required_columns=()),
            rows=[],
            errors=[f"Unknown batch type: {batch_type_key}"],
        )

    result = BulkParseResult(batch_type=spec, rows=[])

    # Parse CSV
    try:
        reader = csv.DictReader(io.StringIO(csv_content))
        if reader.fieldnames is None:
            result.errors.append("CSV file is empty or has no header row")
            return result
    except Exception as e:
        result.errors.append(f"Failed to parse CSV: {e}")
        return result

    # Normalize header columns
    raw_fields = [f.strip().lower() for f in reader.fieldnames]

    # Check required columns present
    missing = []
    for col in ("name",) + spec.required_columns:
        if col not in raw_fields:
            missing.append(col)
    if missing:
        result.errors.append(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected: name, {', '.join(spec.required_columns)}"
        )
        return result

    # Parse rows
    for i, raw_row in enumerate(reader):
        if i >= max_rows:
            result.warnings.append(
                f"CSV truncated at {max_rows} rows (safety limit)."
            )
            break

        # Normalize keys
        row = {k.strip().lower(): (v or "").strip() for k, v in raw_row.items()}

        name = row.get("name", "").strip()
        if not name:
            name = f"Molecule_{i + 1}"

        # Extract and validate sequences
        sequences: Dict[str, str] = {}
        row_error = None

        for col in spec.required_columns:
            raw_seq = row.get(col, "")
            cleaned = _clean_sequence(raw_seq)

            # Set min_len based on type
            min_len = 5 if spec.molecule_class != "peptide" else 3
            err = _validate_sequence(cleaned, col, i, min_len=min_len)
            if err:
                row_error = err
                break
            sequences[col] = cleaned.upper()

        # Optional columns
        metadata: Dict[str, Any] = {}
        for col in spec.optional_columns:
            val = row.get(col, "").strip()
            if val:
                if col == "dar":
                    try:
                        metadata["dar"] = float(val)
                    except ValueError:
                        result.warnings.append(
                            f"Row {i + 1}: invalid DAR value '{val}', ignored"
                        )
                else:
                    metadata[col] = val

        # Build chains and assembly
        chains, assembly = _build_chains_and_assembly(sequences, spec)

        # ── Chain-length advisory (warning only — never blocks analysis) ──
        # Checks for unusually short chains that may indicate a truncated or
        # fragment-only input.  These are WARNINGS, not errors: the row still
        # runs so users can analyse VH-only, Fab-only, or engineered short
        # constructs without being blocked.  Only sequences shorter than a
        # very conservative floor (50 aa for long-chain types, 10 aa for
        # everything else) generate an advisory.
        if row_error is None and spec.chain_count >= 2:
            _ADVISORY_FLOOR = {
                "hc": 50, "hc1": 50, "hc2": 50,
                "lc": 50, "lc1": 50, "lc2": 50,
                "scfv_arm": 50,
                "chain_a": 20, "chain_b": 20,
                "fc": 50, "partner": 10,
                "chain1": 10, "chain2": 10, "chain3": 10,
            }
            for col, seq_val in sequences.items():
                _floor = _ADVISORY_FLOOR.get(col, 0)
                if _floor > 0 and len(seq_val) < _floor:
                    result.warnings.append(
                        f"Row {i + 1} ({name}): chain '{col}' is only {len(seq_val)} residues — "
                        f"this looks like a fragment. "
                        f"Full-length sequences are recommended for accurate MW, pI, and liability analysis."
                    )

        # Compute combined sequence using stoichiometric copy numbers
        # so that seq_length, deam_sites, ox_sites, etc. reflect the
        # full assembled molecule (e.g., mAb = 2×HC + 2×LC, not HC + LC).
        combined_parts = []
        for asm_ch in assembly:
            ch_seq = asm_ch.get("sequence", "")
            ch_copy = asm_ch.get("copy_number", 1)
            combined_parts.append(ch_seq * ch_copy)
        combined = "".join(combined_parts)

        bulk_row = BulkRow(
            row_index=i,
            name=name,
            sequences=sequences,
            metadata=metadata,
            combined_sequence=combined,
            chains=chains,
            assembly_chains=assembly,
            error=row_error,
        )

        result.rows.append(bulk_row)
        if row_error is None:
            result.valid_rows.append(bulk_row)

    if not result.rows:
        result.errors.append("CSV contains no data rows")

    log.info(
        "Bulk CSV parsed: %d total, %d valid, %d errors, type=%s",
        result.n_total, result.n_valid, result.n_errors, spec.key,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Intent Builder — converts BulkRow to pipeline-compatible dict
# ═══════════════════════════════════════════════════════════════════════

def row_to_intent(
    row: BulkRow,
    spec: BatchTypeSpec,
) -> Dict[str, Any]:
    """
    Convert a validated BulkRow into the prompt_data / intent dict
    expected by the single-molecule pipeline.

    Uses feature_registry.compute_features() — the same authoritative path
    as single-molecule analysis — so every molecule gets real, distinct
    pI, MW, GRAVY, hydrophobicity, acidic/basic residue counts, etc.
    """
    seq = row.combined_sequence

    # ── Use the authoritative feature_registry for real computation ──
    try:
        from src.feature_registry import compute_features
        fs = compute_features(
            sequence=seq,
            molecule_class=spec.molecule_class,
            chains=row.assembly_chains,
        )
        # Extract values from FeatureSet
        _fv = fs.features
        pI = _fv["pI"].value if "pI" in _fv else 7.0
        mw_kda = _fv["mw_kda"].value if "mw_kda" in _fv else len(seq) * 0.110
        gravy = _fv["gravy"].value if "gravy" in _fv else 0.0
        hydrophobicity = _fv["hydrophobicity"].value if "hydrophobicity" in _fv else 0.35
        deam_sites = _fv["deam_sites"].value if "deam_sites" in _fv else 0
        ox_sites = _fv["ox_sites"].value if "ox_sites" in _fv else 0
        acidic = _fv["acidic_residues"].value if "acidic_residues" in _fv else 0
        basic = _fv["basic_residues"].value if "basic_residues" in _fv else 0
        cysteine = _fv["cysteine_count"].value if "cysteine_count" in _fv else seq.upper().count("C")
        log.info(
            "Row %d (%s): feature_registry -> pI=%.2f, MW=%.1f kDa, GRAVY=%.4f, "
            "acidic=%d, basic=%d",
            row.row_index + 1, row.name, pI, mw_kda, gravy, acidic, basic,
        )
    except Exception as e:
        log.warning(
            "Row %d (%s): feature_registry unavailable (%s), using fallback",
            row.row_index + 1, row.name, e,
        )
        fb = _fallback_features(seq)
        pI = fb["pI"]
        mw_kda = fb["MW_kDa"]
        gravy = fb["gravy"]
        hydrophobicity = fb["hydrophobicity"]
        deam_sites = fb["deam_sites"]
        ox_sites = fb["ox_sites"]
        acidic = seq.upper().count("D") + seq.upper().count("E")
        basic = seq.upper().count("K") + seq.upper().count("R") + seq.upper().count("H")
        cysteine = seq.upper().count("C")

    # ── Derive molecule-class properties (deferred until after auto-detection) ──
    # Placeholder; actual derivation after Issue 3 auto-detect block below.

    # ── Issue 3+4: Auto-detect molecule class for scfv template ──
    # scfv template defaults to single_domain, but fc_fusion molecules
    # (e.g., Etanercept, aflibercept) need correct classification and stoichiometry.
    _effective_mol_class = spec.molecule_class
    if spec.key == "scfv" and seq:
        try:
            from src.molecule_classifier import classify_molecule
            _clf_result = classify_molecule(seq)
            # classify_molecule returns a ClassificationResult dataclass; extract string value
            detected_class = (
                _clf_result.molecule_class.value
                if _clf_result and hasattr(_clf_result, "molecule_class")
                else str(_clf_result) if _clf_result else None
            )
            if detected_class and detected_class != "single_domain":
                log.info(
                    "Row %d (%s): auto-detected molecule_class=%s (template default=%s)",
                    row.row_index + 1, row.name, detected_class, spec.molecule_class,
                )
                _effective_mol_class = detected_class
                # adjust stoichiometry for fc_fusion homodimers
                if detected_class == "fc_fusion" and row.chains:
                    for ch in row.chains:
                        ch["copy_number"] = 2
                    for ch in row.assembly_chains:
                        ch["copy_number"] = 2
        except Exception as e:
            log.debug("Auto-classification unavailable: %s", e)

    # ── Derive molecule-class properties from effective (possibly auto-detected) class ──
    try:
        from src.molecule_classifier import MoleculeClass as _MC
        _mc = _MC(_effective_mol_class)
    except (ValueError, KeyError, ImportError):
        _mc = None
    _has_fc = _mc.has_fc_region if _mc else ("mab" in _effective_mol_class or "fc" in _effective_mol_class)
    _expects_glyco = _mc.expects_glycosylation if _mc else _has_fc
    _is_mab_like = _mc.is_mab_like if _mc else ("mab" in _effective_mol_class)

    # ── Issue 5: Assembly length validation ──
    _total_assembly_len = sum(
        len(ch.get("sequence", "")) * ch.get("copy_number", 1)
        for ch in row.assembly_chains
    )
    _expected_min = {
        "canonical_mab": 800, "bispecific_4chain": 800,
        "bispecific_3chain": 600, "bispecific": 600,
        "scfv": 100, "fc_fusion": 400, "adc": 800,
    }.get(_effective_mol_class, 100)
    _assembly_warnings = []
    if _total_assembly_len < _expected_min:
        _assembly_warnings.append(
            f"Assembly length {_total_assembly_len} aa is below expected minimum "
            f"{_expected_min} aa for {_effective_mol_class}. Check input sequences."
        )

    intent = {
        "name": row.name,
        "pI": pI,
        "mw": mw_kda,
        "hydrophobicity": hydrophobicity,
        "pH_working": 7.0,
        "deam_sites": deam_sites,
        "ox_sites": ox_sites,
        "acidic_residues": acidic,
        "basic_residues": basic,
        "cysteine_count": cysteine,
        "gradient_slope": 15.0,
        "source": "bulk_csv",
        "sequence": seq,
        "seq_length": _total_assembly_len if _total_assembly_len > len(seq) else len(seq),
        "gravy": gravy,
        "chains": row.chains,
        "assembly_chains": row.assembly_chains,
        "molecule_class": _effective_mol_class,
        "bulk_row_index": row.row_index,
        "bulk_metadata": row.metadata,
        # ── Molecule-class-aware fields (align with single-upload intent) ──
        "has_fc": _has_fc,
        "has_fc_region": _has_fc,
        "expects_glycosylation": _expects_glyco,
        "glycoform_profile": "standard_cho" if _expects_glyco else "none_aglycosylated",
        "is_mab_like": _is_mab_like,
        "format_display": spec.display_name,
        "molecule_class_info": {
            "class": _effective_mol_class,
            "has_fc_region": _has_fc,
            "expects_glycosylation": _expects_glyco,
            "is_mab_like": _is_mab_like,
        },
    }
    if _assembly_warnings:
        intent["warnings"] = _assembly_warnings
    return intent


def _fallback_features(seq: str) -> Dict[str, Any]:
    """Minimal feature calculation when biophysical_features is unavailable."""
    s = seq.upper()
    length = len(s)
    d_count = s.count("D") + s.count("E")
    k_count = s.count("K") + s.count("R") + s.count("H")
    n_count = s.count("N")
    q_count = s.count("Q")
    m_count = s.count("M")
    w_count = s.count("W")

    # Rough pI estimate
    if k_count > d_count:
        pi_est = 8.0 + 0.5 * (k_count - d_count) / max(length, 1) * 100
    else:
        pi_est = 6.0 - 0.5 * (d_count - k_count) / max(length, 1) * 100
    pi_est = max(3.0, min(12.0, pi_est))

    return {
        "pI": round(pi_est, 2),
        "MW_kDa": round(length * 0.110, 2),
        "hydrophobicity": 0.35,
        "deam_sites": n_count + q_count,
        "ox_sites": m_count + w_count,
        "gravy": 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════

def _selftest() -> None:
    """Quick self-test for bulk schema module."""
    # Test template generation
    for key in BATCH_TYPES:
        tmpl = generate_csv_template(key)
        assert "name" in tmpl, f"Template for {key} missing 'name' column"

    # Test CSV parsing — canonical_mab
    csv_text = (
        "name,HC,LC\n"
        "Test_mAb_1,EVQLVESGGGLVQPGGSLRLSCAAS,DIQMTQSPSSLSASVGDRVTITC\n"
        "Test_mAb_2,QVQLVQSGAEVKKPGASVKVSCKASGYTFT,EIVLTQSPATLSLSPGERATLSCRASQS\n"
    )
    result = parse_bulk_csv(csv_text, "canonical_mab")
    assert result.is_ok, f"Parse failed: {result.errors}"
    assert result.n_valid == 2
    assert result.rows[0].assembly_chains[0]["copy_number"] == 2  # HC ×2

    # Test CSV parsing — peptide
    csv_text2 = "name,peptide\nSema,HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG\n"
    result2 = parse_bulk_csv(csv_text2, "peptide")
    assert result2.is_ok
    assert result2.rows[0].combined_sequence == "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"

    # Test CSV parsing — nanobody / VHH
    csv_vhh = "name,vhh\nCapla,QVQLVESGGGLVQAGGSLRLSCAASGSIFSINAMGWYRQAPGKQRELVA\n"
    result_vhh = parse_bulk_csv(csv_vhh, "nanobody")
    assert result_vhh.is_ok, f"Nanobody parse failed: {result_vhh.errors}"
    assert result_vhh.n_valid == 1
    assert len(result_vhh.rows[0].combined_sequence) > 0

    # Test CSV parsing — bispecific 4-chain
    csv_bspc = (
        "name,hc1,lc1,hc2,lc2\n"
        "BiTE_test,EVQLVESGGGLVQ,DIQMTQSPSSL,QVQLVQSGAEVK,EIVLTQSPATL\n"
    )
    result_bspc = parse_bulk_csv(csv_bspc, "bispecific_4chain")
    assert result_bspc.is_ok, f"Bispecific 4-chain parse failed: {result_bspc.errors}"
    assert result_bspc.n_valid == 1

    # Test CSV parsing — ADC (hc + lc, optional dar)
    csv_adc = "name,hc,lc,dar\nTDM1,EVQLVESGGGLVQ,DIQMTQSPSSL,3.5\n"
    result_adc = parse_bulk_csv(csv_adc, "adc")
    assert result_adc.is_ok, f"ADC parse failed: {result_adc.errors}"
    assert result_adc.n_valid == 1

    # Test CSV parsing — fc_fusion (fc + partner)
    csv_fc = "name,fc,partner\nEtanercept,APELLGGPSVFLFPPKPKDTL,LPAQVAFTPYAPEPGSTCRLREYYDQTAQMC\n"
    result_fc = parse_bulk_csv(csv_fc, "fc_fusion")
    assert result_fc.is_ok, f"Fc-fusion parse failed: {result_fc.errors}"
    assert result_fc.n_valid == 1

    # Test missing column error
    csv_text3 = "name,sequence\nBad,ACDEF\n"
    result3 = parse_bulk_csv(csv_text3, "canonical_mab")
    assert not result3.is_ok
    assert "Missing required" in result3.errors[0]

    log.info("bulk_schema self-test: PASS (7 format checks)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
    print("bulk_schema: all self-tests passed")
