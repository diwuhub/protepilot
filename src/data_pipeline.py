"""
data_pipeline.py  ·  ProtePilot — Milestone 15
===========================================================
Experimental Data Ingestion Engine — Jain-137 mAb Dataset Support

Version   : 2.0 (Data-Driven AI Platform)
Author    : Di (ProtePilot)
Depends   : pandas, numpy, biopython (optional)

Purpose
------------------------------------------------------------
Robust data ingestion pipeline that parses standardized CSV datasets
containing wet-lab biophysical assay results. The canonical schema
follows Jain et al. (2017) — 137 clinical-stage mAbs with empirical
SEC, CIC (cross-interaction chromatography), and DSF data.

v2.0 Changes
------------------------------------------------------------
  - Jain-137 schema support: Sequence_HC, Sequence_LC, Exp_SEC_RetentionTime,
    Exp_Aggregation_Percent, Exp_Tm_MeltingTemp
  - generate_mock_jain137(): Synthetic 50-row mAb dataset with realistic
    biophysical distributions for immediate training/testing
  - build_training_dataset(): Convert parsed CSV → (X, y) for supervised ML
  - Feature extraction from raw sequences (pI, MW, GRAVY, liability counts)
  - Backward-compatible: all v1.0 APIs preserved

CSV Schema (Jain-137 Compatible):
  - Sequence_HC        : Heavy chain amino acid sequence
  - Sequence_LC        : Light chain amino acid sequence
  - Name               : Candidate name / ID
  - Exp_SEC_RetentionTime : SEC retention time (min)
  - Exp_Aggregation_Percent : %HMW by SEC (aggregation)
  - Exp_Tm_MeltingTemp : Differential Scanning Fluorimetry Tm (°C)
  - Exp_CIC_RetentionTime : Cross-interaction chromatography RT (optional)
  - Exp_Viscosity_cP   : Viscosity at 150 mg/mL (optional)

References
------------------------------------------------------------
  Jain et al. (2017) PNAS 114(5):944-949 — Biophysical properties of
  the clinical-stage antibody landscape
"""

from __future__ import annotations

import io
import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("ProtePilot.DataPipeline")

_HAS_PANDAS = False
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    log.info("pandas not installed — CSV pipeline will use basic parsing")


# ===========================================================================
# 1. CSV Parsing & Validation (v1.0 backward-compatible + Jain schema)
# ===========================================================================

# Expected columns (normalized to lowercase for matching)
EXPECTED_COLUMNS = {
    "sequence", "name", "pi", "mw_kda", "mw", "hydrophobicity",
    "rt_experimental", "rt_exp", "rt", "agg_score", "aggregation",
    "stability", "viscosity", "label_tag", "tag", "notes",
    # Jain-137 schema columns (M15)
    "sequence_hc", "sequence_lc",
    "exp_sec_retentiontime", "exp_aggregation_percent",
    "exp_tm_meltingtemp", "exp_cic_retentiontime",
    "exp_viscosity_cp",
}

# Column aliases for normalization
COLUMN_ALIASES = {
    # v1.0 aliases
    "pi": "pI",
    "mw_kda": "MW_kDa",
    "mw": "MW_kDa",
    "rt_experimental": "RT_experimental",
    "rt_exp": "RT_experimental",
    "rt": "RT_experimental",
    "agg_score": "Agg_Score",
    "aggregation": "Agg_Score",
    "hydrophobicity": "Hydrophobicity",
    "stability": "Stability",
    "viscosity": "Viscosity",
    "label_tag": "Label_Tag",
    "tag": "Label_Tag",
    "notes": "Notes",
    "sequence": "Sequence",
    "name": "Name",
    # Jain-137 aliases (M15)
    "sequence_hc": "Sequence_HC",
    "sequence_lc": "Sequence_LC",
    "exp_sec_retentiontime": "Exp_SEC_RetentionTime",
    "exp_aggregation_percent": "Exp_Aggregation_Percent",
    "exp_tm_meltingtemp": "Exp_Tm_MeltingTemp",
    "exp_cic_retentiontime": "Exp_CIC_RetentionTime",
    "exp_viscosity_cp": "Exp_Viscosity_cP",
    # Additional aliases for flexible header naming
    "heavy_chain": "Sequence_HC",
    "light_chain": "Sequence_LC",
    "hc": "Sequence_HC",
    "lc": "Sequence_LC",
    "sec_rt": "Exp_SEC_RetentionTime",
    "sec_retention": "Exp_SEC_RetentionTime",
    "aggregation_percent": "Exp_Aggregation_Percent",
    "agg_percent": "Exp_Aggregation_Percent",
    "hmw_percent": "Exp_Aggregation_Percent",
    "%hmw": "Exp_Aggregation_Percent",
    "tm": "Exp_Tm_MeltingTemp",
    "melting_temp": "Exp_Tm_MeltingTemp",
    "dsf_tm": "Exp_Tm_MeltingTemp",
    "tm1": "Exp_Tm_MeltingTemp",
    "cic_rt": "Exp_CIC_RetentionTime",
    "viscosity_cp": "Exp_Viscosity_cP",
}


def parse_csv_upload(
    file_content: Any,
    filename: str = "upload.csv",
) -> Dict[str, Any]:
    """
    Parse an uploaded CSV file into a structured dataset.

    Supports both v1.0 schema (Sequence, RT, Agg_Score) and
    v2.0 Jain-137 schema (Sequence_HC, Sequence_LC, Exp_* fields).

    Parameters
    ----------
    file_content : File-like object (from st.file_uploader) or string
    filename : Original filename

    Returns
    -------
    Dict with:
      - status: "success" or "error"
      - message: Description
      - data: List[Dict] of parsed rows
      - columns: List of detected column names
      - n_rows: Number of data rows
      - warnings: List of parsing warnings
      - schema: "jain137" or "legacy" (v2.0)
    """
    warnings_list = []

    try:
        if _HAS_PANDAS:
            if isinstance(file_content, str):
                df = pd.read_csv(io.StringIO(file_content))
            else:
                df = pd.read_csv(file_content)

            # Normalize column names
            rename_map = {}
            for col in df.columns:
                col_lower = col.strip().lower().replace(" ", "_")
                if col_lower in COLUMN_ALIASES:
                    rename_map[col] = COLUMN_ALIASES[col_lower]
                else:
                    rename_map[col] = col.strip()
            df = df.rename(columns=rename_map)

            # Convert to list of dicts
            data = df.to_dict(orient="records")

            # Clean NaN values
            for row in data:
                for k, v in row.items():
                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                        row[k] = None

            columns = list(df.columns)
        else:
            # Basic CSV parsing without pandas
            if isinstance(file_content, str):
                lines = file_content.strip().split("\n")
            else:
                content = file_content.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                lines = content.strip().split("\n")

            if not lines:
                return {"status": "error", "message": "Empty CSV file",
                        "data": [], "columns": [], "n_rows": 0,
                        "warnings": [], "schema": "unknown"}

            headers = [h.strip() for h in lines[0].split(",")]
            columns = []
            for h in headers:
                h_lower = h.lower().replace(" ", "_")
                columns.append(COLUMN_ALIASES.get(h_lower, h))

            data = []
            for line in lines[1:]:
                if not line.strip():
                    continue
                vals = [v.strip() for v in line.split(",")]
                row = {}
                for i, col in enumerate(columns):
                    if i < len(vals):
                        val = vals[i]
                        try:
                            val = float(val)
                            if val == int(val) and "." not in vals[i]:
                                val = int(val)
                        except (ValueError, TypeError):
                            if val.lower() in ("", "na", "nan", "none", "null"):
                                val = None
                        row[col] = val
                    else:
                        row[col] = None
                data.append(row)

        # Detect schema type
        jain_cols = {"Sequence_HC", "Sequence_LC", "Exp_Aggregation_Percent", "Exp_Tm_MeltingTemp"}
        has_jain = len(jain_cols.intersection(set(columns))) >= 2
        schema = "jain137" if has_jain else "legacy"

        # Validation warnings
        if schema == "jain137":
            if "Sequence_HC" not in columns:
                warnings_list.append("Missing 'Sequence_HC' column — limited feature extraction")
            if "Exp_Aggregation_Percent" not in columns and "Exp_Tm_MeltingTemp" not in columns:
                warnings_list.append("No wet-lab target columns (Exp_Aggregation_Percent, Exp_Tm_MeltingTemp)")
        else:
            if "Sequence" not in columns and "Name" not in columns:
                warnings_list.append("No 'Sequence' or 'Name' column found.")
            if "RT_experimental" not in columns and "Agg_Score" not in columns:
                warnings_list.append("No experimental data columns found. Limited use for retraining.")

        n_rows = len(data)
        log.info("Parsed CSV '%s': %d rows, %d columns, schema=%s",
                 filename, n_rows, len(columns), schema)

        return {
            "status": "success",
            "message": f"Parsed {n_rows} rows from {filename} ({schema} schema)",
            "data": data,
            "columns": columns,
            "n_rows": n_rows,
            "warnings": warnings_list,
            "schema": schema,
        }

    except Exception as e:
        log.error("CSV parsing failed: %s", e)
        return {
            "status": "error",
            "message": f"Failed to parse CSV: {e}",
            "data": [],
            "columns": [],
            "n_rows": 0,
            "warnings": [str(e)],
            "schema": "unknown",
        }


# ===========================================================================
# 2. FASTA File Parser (for st.file_uploader)
# ===========================================================================

def parse_fasta_file(
    file_content: Any,
    filename: str = "upload.fasta",
) -> Dict[str, Any]:
    """
    Parse an uploaded .fasta file into sequences.

    Returns
    -------
    Dict with:
      - status: "success" or "error"
      - sequences: List of {"header": str, "sequence": str}
      - n_sequences: int
      - combined_text: Full FASTA text (for passing to existing parser)
    """
    try:
        if isinstance(file_content, str):
            content = file_content
        else:
            # Reset seek position — Streamlit UploadedFile may retain position
            # from a previous read() in the same or prior render cycle.
            if hasattr(file_content, "seek"):
                file_content.seek(0)
            content = file_content.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")

        sequences = []
        current_header = None
        current_seq_lines = []

        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None and current_seq_lines:
                    seq = "".join(current_seq_lines).upper()
                    seq = "".join(c for c in seq if c.isalpha())
                    if len(seq) >= 10:
                        sequences.append({"header": current_header, "sequence": seq})
                current_header = line[1:].strip()
                current_seq_lines = []
            elif line:
                current_seq_lines.append(line)

        # Last sequence
        if current_header is not None and current_seq_lines:
            seq = "".join(current_seq_lines).upper()
            seq = "".join(c for c in seq if c.isalpha())
            if len(seq) >= 10:
                sequences.append({"header": current_header, "sequence": seq})

        # Handle raw sequence (no header)
        if not sequences and not content.startswith(">"):
            seq = "".join(c for c in content.upper() if c.isalpha())
            if len(seq) >= 10:
                sequences.append({"header": "Unnamed_Sequence", "sequence": seq})

        log.info("Parsed FASTA '%s': %d sequences", filename, len(sequences))

        return {
            "status": "success",
            "sequences": sequences,
            "n_sequences": len(sequences),
            "combined_text": content,
        }

    except Exception as e:
        log.error("FASTA parsing failed: %s", e)
        return {
            "status": "error",
            "sequences": [],
            "n_sequences": 0,
            "combined_text": "",
        }


# ===========================================================================
# 3. Expert Labeling Manager
# ===========================================================================

class ExpertLabelStore:
    """
    Manages expert-labeled corrections for the Continuous Learning pipeline.

    Each label contains:
      - feature_vector : Original model input features (7-dim)
      - predicted_value : Model's prediction (dict or float)
      - actual_value   : Scientist's corrected value
      - metric_type    : What was corrected ("RT", "ka", "nu", "agg_score", etc.)
      - tag            : Free-text label
      - timestamp      : When the correction was made
      - source         : "manual" or "csv_import"
    """

    def __init__(self):
        self.labels: List[Dict[str, Any]] = []

    def add_label(
        self,
        feature_vector: Optional[List[float]] = None,
        predicted_value: Any = None,
        actual_value: Any = None,
        metric_type: str = "RT",
        tag: str = "",
        source: str = "manual",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a new expert label."""
        label = {
            "id": f"label_{len(self.labels):04d}",
            "feature_vector": feature_vector,
            "predicted_value": predicted_value,
            "actual_value": actual_value,
            "metric_type": metric_type,
            "tag": tag,
            "source": source,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "extra": extra or {},
        }
        self.labels.append(label)
        log.info("Added label %s: %s predicted=%s actual=%s",
                 label["id"], metric_type, predicted_value, actual_value)
        return label

    def add_labels_from_csv(self, csv_data: List[Dict[str, Any]]) -> int:
        """
        Import expert labels from parsed CSV data.

        Looks for rows with both predicted and actual values.
        Returns number of labels imported.
        """
        count = 0
        for row in csv_data:
            rt_exp = row.get("RT_experimental")
            agg = row.get("Agg_Score")
            stab = row.get("Stability")

            # Build feature vector from available columns
            features = None
            pi_val = row.get("pI")
            mw_val = row.get("MW_kDa")
            if pi_val is not None and mw_val is not None:
                features = [
                    float(pi_val or 8.0),
                    float(mw_val or 150.0),
                    float(row.get("deam_sites", 1)),
                    float(row.get("ox_sites", 1)),
                    float(row.get("acidic_residues", 40)),
                    float(row.get("basic_residues", 50)),
                    float(row.get("Hydrophobicity", 0.35)),
                ]

            name = row.get("Name", "Unknown")

            if rt_exp is not None:
                self.add_label(
                    feature_vector=features,
                    predicted_value=None,
                    actual_value=float(rt_exp),
                    metric_type="RT",
                    tag=f"CSV import: {name}",
                    source="csv_import",
                    extra=row,
                )
                count += 1

            if agg is not None:
                self.add_label(
                    feature_vector=features,
                    predicted_value=None,
                    actual_value=float(agg),
                    metric_type="agg_score",
                    tag=f"CSV import: {name}",
                    source="csv_import",
                    extra=row,
                )
                count += 1

            if stab is not None:
                self.add_label(
                    feature_vector=features,
                    predicted_value=None,
                    actual_value=float(stab),
                    metric_type="stability",
                    tag=f"CSV import: {name}",
                    source="csv_import",
                    extra=row,
                )
                count += 1

        log.info("Imported %d labels from CSV (%d rows)", count, len(csv_data))
        return count

    def get_training_data(
        self,
        metric_type: str = "RT",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract feature/target arrays for retraining from labeled data.

        Returns (X, y) where X is (n, 7) and y is (n,), or (None, None).
        """
        X_list = []
        y_list = []

        for label in self.labels:
            if label["metric_type"] != metric_type:
                continue
            if label["feature_vector"] is None or label["actual_value"] is None:
                continue
            X_list.append(label["feature_vector"])
            y_list.append(float(label["actual_value"]))

        if not X_list:
            return None, None

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the label store."""
        by_type = {}
        by_source = {}
        for label in self.labels:
            mt = label["metric_type"]
            src = label["source"]
            by_type[mt] = by_type.get(mt, 0) + 1
            by_source[src] = by_source.get(src, 0) + 1

        return {
            "total": len(self.labels),
            "by_type": by_type,
            "by_source": by_source,
        }

    @property
    def count(self) -> int:
        return len(self.labels)


# ===========================================================================
# 4. Data Quality Checks
# ===========================================================================

def validate_csv_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run quality checks on parsed CSV data.

    Returns dict with quality metrics and flagged rows.
    """
    n = len(data)
    if n == 0:
        return {"status": "empty", "n_rows": 0, "issues": ["No data rows"]}

    issues = []
    n_missing_seq = sum(1 for r in data if not r.get("Sequence") and not r.get("Sequence_HC"))
    n_missing_pi = sum(1 for r in data if r.get("pI") is None)
    n_outlier_pi = sum(1 for r in data
                       if r.get("pI") is not None
                       and (float(r["pI"]) < 3.0 or float(r["pI"]) > 12.0))

    # Jain-137 specific quality checks
    n_with_agg = sum(1 for r in data if r.get("Exp_Aggregation_Percent") is not None)
    n_with_tm = sum(1 for r in data if r.get("Exp_Tm_MeltingTemp") is not None)
    n_with_hc = sum(1 for r in data if r.get("Sequence_HC") and len(str(r["Sequence_HC"])) > 50)
    n_with_lc = sum(1 for r in data if r.get("Sequence_LC") and len(str(r["Sequence_LC"])) > 50)

    if n_missing_seq > n * 0.5:
        issues.append(f"{n_missing_seq}/{n} rows missing sequence data")
    if n_outlier_pi > 0:
        issues.append(f"{n_outlier_pi} rows with pI outside [3.0, 12.0]")

    # Check for Jain aggregation percent outliers
    for r in data:
        agg = r.get("Exp_Aggregation_Percent")
        if agg is not None:
            try:
                if float(agg) < 0 or float(agg) > 100:
                    issues.append(f"Aggregation% out of range [0, 100]: {agg}")
                    break
            except (ValueError, TypeError):
                pass

    return {
        "status": "ok" if not issues else "warnings",
        "n_rows": n,
        "n_missing_sequence": n_missing_seq,
        "n_missing_pi": n_missing_pi,
        "n_outlier_pi": n_outlier_pi,
        "n_with_aggregation": n_with_agg,
        "n_with_tm": n_with_tm,
        "n_with_hc_sequence": n_with_hc,
        "n_with_lc_sequence": n_with_lc,
        "issues": issues,
    }


# ===========================================================================
# 5. Mock Jain-137 Synthetic Dataset Generator (M15)
# ===========================================================================

# IgG1 framework fragments used for realistic sequence generation
_HC_FRAMEWORK = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFS"  # FR1
    "SYAMSWVRQAPGKGLEWVSAIS"           # CDR1 + FR2 start
)
_HC_FC = (
    "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTV"
    "PSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEV"
    "TCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPA"
    "PIEKTISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDG"
    "SFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
)
_LC_FRAMEWORK = (
    "DIQMTQSPSSLSASVGDRVTITC"  # FR1
    "RASQGIRNDLGWYQQKPGKAPKLLIY"  # CDR1 + FR2
)
_LC_CONSTANT = (
    "RTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTL"
    "TLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
)

# CDR amino acid alphabet (biased toward antibody CDR residues)
_CDR_AAS = "AGSTYDFNQHWRLIVPEKM"


def _random_cdr(rng: np.random.RandomState, min_len: int = 5, max_len: int = 18) -> str:
    """Generate a random CDR-like peptide."""
    length = rng.randint(min_len, max_len + 1)
    return "".join(rng.choice(list(_CDR_AAS)) for _ in range(length))


def _random_framework_filler(rng: np.random.RandomState, length: int = 15) -> str:
    """Generate framework-like filler residues."""
    fw_aas = "AGSTYQNDEKVLIP"
    return "".join(rng.choice(list(fw_aas)) for _ in range(length))


def generate_mock_jain137(
    n_samples: int = 50,
    seed: int = 137,
) -> Dict[str, Any]:
    """
    Generate a synthetic 'Mock Jain-137' dataset that mimics real clinical
    mAb sequences with realistic biophysical assay scores.

    Each row contains:
      - Name: mAb identifier (mAb_001 ... mAb_050)
      - Sequence_HC: Full-length heavy chain (~440-460 aa)
      - Sequence_LC: Full-length light chain (~210-230 aa)
      - Exp_SEC_RetentionTime: SEC retention time (9-14 min typical)
      - Exp_Aggregation_Percent: %HMW by SEC (0.5-30%)
      - Exp_Tm_MeltingTemp: DSF first Tm (55-80 °C)
      - Exp_CIC_RetentionTime: CIC retention time (10-25 min)
      - Exp_Viscosity_cP: Viscosity at 150 mg/mL (2-50 cP)

    Biophysical correlations enforced:
      - High pI → lower aggregation (typical for IgG1)
      - High hydrophobicity → higher aggregation, lower Tm
      - More liabilities → lower Tm, higher %HMW
      - Viscosity correlates with pI deviation from neutrality

    Parameters
    ----------
    n_samples : Number of synthetic mAbs to generate (default 50)
    seed : Random seed for reproducibility

    Returns
    -------
    dict : {
        "status": "success",
        "data": List[Dict] — parsed rows (same format as parse_csv_upload),
        "columns": List[str],
        "n_rows": int,
        "csv_string": str — CSV text for download/display,
        "schema": "jain137",
    }
    """
    rng = np.random.RandomState(seed)
    rows = []

    # Try to use Biopython for accurate pI
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        has_bio = True
    except ImportError:
        has_bio = False

    for i in range(n_samples):
        # -- Generate realistic antibody sequences --------------------------
        # Heavy chain: FR1 + CDR1 + FR2 + CDR2 + FR3 + CDR3 + Fc
        cdr_h1 = _random_cdr(rng, 5, 7)
        cdr_h2 = _random_cdr(rng, 10, 18)
        cdr_h3 = _random_cdr(rng, 4, 25)
        fr2 = _random_framework_filler(rng, 14)
        fr3 = _random_framework_filler(rng, 32)

        hc_seq = _HC_FRAMEWORK + cdr_h1 + fr2 + cdr_h2 + fr3 + cdr_h3 + _HC_FC
        # Clean
        hc_seq = re.sub(r'[^A-Z]', '', hc_seq.upper())

        # Light chain: FR1 + CDR_L1 + FR2 + CDR_L2 + FR3 + CDR_L3 + CL
        cdr_l1 = _random_cdr(rng, 6, 12)
        cdr_l2 = _random_cdr(rng, 7, 7)
        cdr_l3 = _random_cdr(rng, 7, 10)
        fr_l2 = _random_framework_filler(rng, 15)
        fr_l3 = _random_framework_filler(rng, 30)

        lc_seq = _LC_FRAMEWORK + cdr_l1 + fr_l2 + cdr_l2 + fr_l3 + cdr_l3 + _LC_CONSTANT
        lc_seq = re.sub(r'[^A-Z]', '', lc_seq.upper())

        name = f"mAb_{i + 1:03d}"

        # -- Compute biophysical features -----------------------------------
        combined = hc_seq + lc_seq
        if has_bio:
            try:
                pa = ProteinAnalysis(combined)
                pI = pa.isoelectric_point()
                mw_kda = pa.molecular_weight() / 1000.0
                gravy = pa.gravy()
            except Exception:
                pI = rng.uniform(6.5, 9.5)
                mw_kda = rng.uniform(140.0, 155.0)
                gravy = rng.uniform(-0.6, 0.0)
        else:
            pI = rng.uniform(6.5, 9.5)
            mw_kda = rng.uniform(140.0, 155.0)
            gravy = rng.uniform(-0.6, 0.0)

        hydrophobicity = max(0.0, min(1.0, (gravy + 2.0) / 4.0))

        # Count liabilities
        n_met = combined.count("M")
        n_deam = len(re.findall(r"N[GS]", combined))
        n_dp = combined.count("DP")

        # -- Generate correlated wet-lab data --------------------------------
        # Aggregation: baseline 2-5%, increases with hydrophobicity and liabilities
        base_agg = rng.uniform(0.5, 5.0)
        hydro_penalty = max(0, (hydrophobicity - 0.35)) * 30.0
        liab_penalty = (n_met * 0.5 + n_deam * 0.8 + n_dp * 1.5)
        # High pI mAbs (>8.0) tend to have lower aggregation for IgG1
        pi_benefit = max(0, (pI - 7.5)) * (-2.0)
        agg_pct = base_agg + hydro_penalty + liab_penalty + pi_benefit + rng.normal(0, 1.5)
        agg_pct = round(max(0.2, min(35.0, agg_pct)), 1)

        # Tm: baseline 68-75 °C, decreases with liabilities and hydrophobicity
        base_tm = rng.uniform(68.0, 75.0)
        tm_liab_penalty = (n_met * 0.3 + n_deam * 0.4) * -1
        tm_hydro_penalty = max(0, (hydrophobicity - 0.35)) * -15.0
        tm = base_tm + tm_liab_penalty + tm_hydro_penalty + rng.normal(0, 1.5)
        tm = round(max(52.0, min(82.0, tm)), 1)

        # SEC RT: mAb monomer typically 10-13 min
        sec_rt = round(rng.uniform(9.5, 13.5) + (mw_kda - 148.0) * 0.01, 2)

        # CIC RT: measures non-specific binding; higher = stickier
        cic_base = 12.0 + hydrophobicity * 15.0
        cic_rt = round(cic_base + rng.normal(0, 1.0), 2)
        cic_rt = max(8.0, min(28.0, cic_rt))

        # Viscosity: correlates with charge asymmetry and concentration
        visc_base = 5.0 + abs(pI - 7.0) * 3.0 + hydrophobicity * 20.0
        viscosity = round(visc_base + rng.normal(0, 2.0), 1)
        viscosity = max(2.0, min(55.0, viscosity))

        rows.append({
            "Name": name,
            "Sequence_HC": hc_seq,
            "Sequence_LC": lc_seq,
            "Exp_SEC_RetentionTime": sec_rt,
            "Exp_Aggregation_Percent": agg_pct,
            "Exp_Tm_MeltingTemp": tm,
            "Exp_CIC_RetentionTime": cic_rt,
            "Exp_Viscosity_cP": viscosity,
        })

    # Build CSV string
    columns = [
        "Name", "Sequence_HC", "Sequence_LC",
        "Exp_SEC_RetentionTime", "Exp_Aggregation_Percent",
        "Exp_Tm_MeltingTemp", "Exp_CIC_RetentionTime",
        "Exp_Viscosity_cP",
    ]
    csv_lines = [",".join(columns)]
    for row in rows:
        vals = [str(row.get(c, "")) for c in columns]
        csv_lines.append(",".join(vals))
    csv_string = "\n".join(csv_lines)

    log.info("Generated Mock Jain-137 dataset: %d mAbs", n_samples)

    return {
        "status": "success",
        "data": rows,
        "columns": columns,
        "n_rows": n_samples,
        "csv_string": csv_string,
        "schema": "jain137",
    }


# ===========================================================================
# 6. Feature Extraction from Jain-137 Rows (M15)
# ===========================================================================

def extract_features_from_jain_row(
    row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extract biophysical features from a Jain-137-format row.

    Computes pI, MW, GRAVY, liability counts from HC+LC sequences.

    Returns
    -------
    dict with: pI, mw_kda, hydrophobicity, deam_sites, ox_sites,
               acidic_residues, basic_residues, feature_vector (7-dim list),
               combined_sequence
    or None if insufficient data.
    """
    hc = str(row.get("Sequence_HC", "") or "").upper()
    lc = str(row.get("Sequence_LC", "") or "").upper()
    single = str(row.get("Sequence", "") or "").upper()

    # Use HC+LC if available, else fall back to single Sequence column
    if len(hc) > 50 and len(lc) > 50:
        combined = hc + lc
    elif len(single) > 50:
        combined = single
    else:
        return None

    # Clean
    combined = re.sub(r'[^A-Z]', '', combined)
    if len(combined) < 50:
        return None

    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        pa = ProteinAnalysis(combined)
        pI = pa.isoelectric_point()
        mw_kda = pa.molecular_weight() / 1000.0
        gravy = pa.gravy()
    except Exception:
        # Fallback: use row-level pI/MW if available
        pI = float(row.get("pI", 8.0) or 8.0)
        mw_kda = float(row.get("MW_kDa", 150.0) or 150.0)
        gravy = -0.3

    hydrophobicity = max(0.0, min(1.0, (gravy + 2.0) / 4.0))

    # Liability counts
    deam_sites = len(re.findall(r"N[GS]", combined))
    ox_sites = combined.count("M") + combined.count("W")
    acidic = combined.count("D") + combined.count("E")
    basic = combined.count("K") + combined.count("R") + combined.count("H")

    feature_vector = [
        round(pI, 4),
        round(mw_kda, 4),
        float(deam_sites),
        float(ox_sites),
        float(acidic),
        float(basic),
        round(hydrophobicity, 4),
    ]

    return {
        "pI": round(pI, 2),
        "mw_kda": round(mw_kda, 1),
        "hydrophobicity": round(hydrophobicity, 3),
        "gravy": round(gravy, 3),
        "deam_sites": deam_sites,
        "ox_sites": ox_sites,
        "acidic_residues": acidic,
        "basic_residues": basic,
        "feature_vector": feature_vector,
        "combined_sequence": combined,
    }


def build_training_dataset(
    data: List[Dict[str, Any]],
    target_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convert parsed CSV data into ML-ready (X, y) arrays.

    Extracts biophysical features from sequences and pairs them
    with wet-lab target variables.

    Parameters
    ----------
    data : List of row dicts (from parse_csv_upload or generate_mock_jain137)
    target_columns : List of target column names. Default:
                     ["Exp_Aggregation_Percent", "Exp_Tm_MeltingTemp"]

    Returns
    -------
    dict : {
        "status": "success" | "error",
        "X": np.ndarray (n, 7) — biophysical features,
        "y": np.ndarray (n, n_targets) — wet-lab targets,
        "feature_names": list of str,
        "target_names": list of str,
        "n_samples": int,
        "n_skipped": int,
        "sample_names": list of str,
    }
    """
    if target_columns is None:
        target_columns = ["Exp_Aggregation_Percent", "Exp_Tm_MeltingTemp"]

    feature_names = [
        "pI", "MW_kDa", "deam_sites", "ox_sites",
        "acidic_residues", "basic_residues", "hydrophobicity",
    ]

    X_list = []
    y_list = []
    names = []
    n_skipped = 0

    for row in data:
        # Extract features from sequence
        feat = extract_features_from_jain_row(row)
        if feat is None:
            n_skipped += 1
            continue

        # Extract target values
        targets = []
        valid = True
        for tc in target_columns:
            val = row.get(tc)
            if val is None:
                valid = False
                break
            try:
                targets.append(float(val))
            except (ValueError, TypeError):
                valid = False
                break

        if not valid:
            n_skipped += 1
            continue

        X_list.append(feat["feature_vector"])
        y_list.append(targets)
        names.append(str(row.get("Name", f"sample_{len(names)}")))

    if not X_list:
        return {
            "status": "error",
            "message": "No valid samples with both features and targets",
            "X": None, "y": None,
            "feature_names": feature_names,
            "target_names": target_columns,
            "n_samples": 0,
            "n_skipped": n_skipped,
            "sample_names": [],
        }

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    log.info("Built training dataset: %d samples, %d features, %d targets (skipped %d)",
             len(X), X.shape[1], y.shape[1], n_skipped)

    return {
        "status": "success",
        "message": f"Built {len(X)} samples x {len(target_columns)} targets",
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "target_names": target_columns,
        "n_samples": len(X),
        "n_skipped": n_skipped,
        "sample_names": names,
    }


# ===========================================================================
# 7b. PLM Embedding Dataset Builder (M28 — ESM-2 Upgrade)
# ===========================================================================

def process_jain_dataset(
    data: List[Dict[str, Any]],
    target_columns: Optional[List[str]] = None,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Build a PLM-embedding-based training dataset from Jain-137-format data.

    For each sample, extracts a 480-dimensional ESM-2 embedding (or mock
    fallback) from the heavy-chain sequence, and pairs it with wet-lab
    target values.  This replaces the 7-dim biophysical feature vectors
    used by ``build_training_dataset()`` with high-dimensional protein
    language model representations.

    Parameters
    ----------
    data : List of row dicts (from parse_csv_upload or generate_mock_jain137).
           Must contain at least one of:
           - ``Sequence_HC`` / ``Sequence_LC`` (preferred)
           - ``Sequence`` (single-chain fallback)
    target_columns : Target column names.  Default:
                     ["Exp_Aggregation_Percent", "Exp_Tm_MeltingTemp"]
    progress_cb : callable(current, total) — optional progress callback.

    Returns
    -------
    dict : {
        "status": "success" | "error",
        "X": np.ndarray (n, embed_dim) — PLM embeddings,
        "y": np.ndarray (n, n_targets) — wet-lab targets,
        "feature_names": list of str (["plm_0", "plm_1", ...]),
        "target_names": list of str,
        "n_samples": int,
        "n_skipped": int,
        "sample_names": list of str,
        "embed_dim": int,
        "mode": "esm2" | "mock",
    }
    """
    if target_columns is None:
        target_columns = ["Exp_Aggregation_Percent", "Exp_Tm_MeltingTemp"]

    # Lazy-import the PLM embedder
    try:
        from src.pLM_embedder import get_embedder, get_embedding_dim
    except ImportError:
        try:
            from pLM_embedder import get_embedder, get_embedding_dim
        except ImportError:
            return {
                "status": "error",
                "message": "PLM embedder module not available (src.plm_embedder)",
                "X": None, "y": None,
                "feature_names": [],
                "target_names": target_columns,
                "n_samples": 0, "n_skipped": 0,
                "sample_names": [], "embed_dim": 0, "mode": "unavailable",
            }

    embedder = get_embedder()
    embed_dim = get_embedding_dim()
    mode = "mock" if embedder.is_mock else "esm2"

    X_list: List[np.ndarray] = []
    y_list: List[List[float]] = []
    names: List[str] = []
    n_skipped = 0
    total = len(data)

    for idx, row in enumerate(data):
        # --- Resolve sequence ---
        hc = str(row.get("Sequence_HC", "") or "").strip()
        lc = str(row.get("Sequence_LC", "") or "").strip()
        single = str(row.get("Sequence", "") or "").strip()

        sequence = hc or single
        if not sequence or len(re.sub(r"[^A-Z]", "", sequence.upper())) < 10:
            n_skipped += 1
            if progress_cb:
                progress_cb(idx + 1, total)
            continue

        # --- Extract target values ---
        targets: List[float] = []
        valid = True
        for tc in target_columns:
            val = row.get(tc)
            if val is None:
                valid = False
                break
            try:
                targets.append(float(val))
            except (ValueError, TypeError):
                valid = False
                break

        if not valid:
            n_skipped += 1
            if progress_cb:
                progress_cb(idx + 1, total)
            continue

        # --- Generate PLM embedding ---
        try:
            embedding = embedder.embed_sequence(sequence)
        except Exception as e:
            log.warning("Embedding failed for sample %d: %s", idx, e)
            n_skipped += 1
            if progress_cb:
                progress_cb(idx + 1, total)
            continue

        X_list.append(embedding)
        y_list.append(targets)
        names.append(str(row.get("Name", f"sample_{len(names)}")))

        if progress_cb:
            progress_cb(idx + 1, total)

    if not X_list:
        feature_names = [f"plm_{i}" for i in range(embed_dim)]
        return {
            "status": "error",
            "message": "No valid samples with both sequence and targets",
            "X": None, "y": None,
            "feature_names": feature_names,
            "target_names": target_columns,
            "n_samples": 0, "n_skipped": n_skipped,
            "sample_names": [], "embed_dim": embed_dim, "mode": mode,
        }

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    feature_names = [f"plm_{i}" for i in range(X.shape[1])]

    log.info(
        "PLM dataset built: %d samples, %d-dim embeddings (%s mode), "
        "%d targets (skipped %d)",
        len(X), X.shape[1], mode, y.shape[1], n_skipped,
    )

    return {
        "status": "success",
        "message": f"Built {len(X)} samples × {X.shape[1]}-dim PLM embeddings ({mode})",
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "target_names": target_columns,
        "n_samples": len(X),
        "n_skipped": n_skipped,
        "sample_names": names,
        "embed_dim": X.shape[1],
        "mode": mode,
    }


# ===========================================================================
# 8. Literature PDF / Text Data Extraction (M24 — RAG Foundation)
# ===========================================================================

def extract_text_from_pdf(file_content: Any) -> str:
    """
    Extract text from a PDF file.

    Tries PyPDF2 first, falls back to pdfplumber, then basic byte decoding.

    Parameters
    ----------
    file_content : bytes or file-like object

    Returns
    -------
    str : Extracted text from all pages
    """
    text = ""

    # Ensure we have bytes
    if hasattr(file_content, "read"):
        raw_bytes = file_content.read()
        if hasattr(file_content, "seek"):
            file_content.seek(0)
    else:
        raw_bytes = file_content

    # Try PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        text = "\n\n".join(pages)
        if text.strip():
            log.info(f"Extracted {len(text)} chars from PDF via PyPDF2 ({len(reader.pages)} pages)")
            return text
    except ImportError:
        log.info("PyPDF2 not installed, trying alternatives")
    except Exception as e:
        log.warning(f"PyPDF2 extraction failed: {e}")

    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            text = "\n\n".join(pages)
            if text.strip():
                log.info(f"Extracted {len(text)} chars from PDF via pdfplumber")
                return text
    except ImportError:
        log.info("pdfplumber not installed, trying basic extraction")
    except Exception as e:
        log.warning(f"pdfplumber extraction failed: {e}")

    # Basic fallback: try to decode raw bytes
    try:
        decoded = raw_bytes.decode("utf-8", errors="ignore")
        # Extract printable text segments
        import re as _re
        segments = _re.findall(r'[\x20-\x7E]{10,}', decoded)
        text = " ".join(segments)
        log.info(f"Extracted {len(text)} chars via basic byte decoding")
    except Exception:
        text = ""

    return text


def extract_data_from_literature_text(text: str) -> Dict[str, Any]:
    """
    Parse literature text to extract protein sequences and biophysical metrics.

    Looks for:
      - FASTA-format sequences or long amino acid strings
      - Biophysical metrics: pI, Kd, Tm, aggregation%, titer, viscosity

    Parameters
    ----------
    text : Raw text from a PDF or pasted content

    Returns
    -------
    Dict with:
      - sequences: List of extracted sequences
      - metrics: List of {metric, value, unit, context} dicts
      - n_sequences: count
      - n_metrics: count
      - warnings: list of parsing notes
    """
    sequences = []
    metrics = []
    warnings = []

    # ---- Extract sequences ----
    # FASTA format: >header followed by amino acid lines
    fasta_pattern = re.compile(
        r'>([^\n]+)\n([A-Za-z\s\n]+?)(?=\n>|\n\n|\Z)',
        re.MULTILINE
    )
    for match in fasta_pattern.finditer(text):
        header = match.group(1).strip()
        seq = re.sub(r'[^A-Za-z]', '', match.group(2)).upper()
        if len(seq) >= 20 and re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', seq):
            sequences.append({"header": header, "sequence": seq, "length": len(seq)})

    # Raw amino acid strings (at least 30 consecutive valid AAs)
    raw_aa_pattern = re.compile(r'\b([ACDEFGHIKLMNPQRSTVWY]{30,})\b')
    for match in raw_aa_pattern.finditer(text.upper()):
        seq = match.group(1)
        # Avoid duplicates
        if not any(s["sequence"] == seq for s in sequences):
            sequences.append({"header": f"Extracted_seq_{len(sequences)+1}", "sequence": seq, "length": len(seq)})

    # ---- Extract biophysical metrics ----
    metric_patterns = [
        # pI
        (r'pI\s*[=:≈~]\s*(\d+\.?\d*)', "pI", "pH units"),
        (r'isoelectric\s+point\s*[=:≈~of]*\s*(\d+\.?\d*)', "pI", "pH units"),
        # Kd (affinity)
        (r'[Kk][Dd]\s*[=:≈~]\s*(\d+\.?\d*)\s*(nM|uM|pM|μM)', "Kd", None),
        # Aggregation
        (r'aggregation\s*[=:≈~of]*\s*(\d+\.?\d*)\s*%', "Aggregation", "%"),
        (r'(%?\s*HMW|%?\s*aggregate)\s*[=:≈~of]*\s*(\d+\.?\d*)\s*%?', "Aggregation", "%"),
        # Titer
        (r'titer\s*[=:≈~of]*\s*(\d+\.?\d*)\s*(g/L|mg/L)', "Titer", None),
        # Tm (melting temperature)
        (r'[Tt]m\s*[=:≈~]\s*(\d+\.?\d*)\s*[°]?[Cc]?', "Tm", "°C"),
        (r'melting\s*(?:temperature|temp)\s*[=:≈~of]*\s*(\d+\.?\d*)', "Tm", "°C"),
        # Viscosity
        (r'viscosity\s*[=:≈~of]*\s*(\d+\.?\d*)\s*(cP|mPa)', "Viscosity", None),
        # SEC retention time
        (r'SEC\s*(?:retention\s*time|RT)\s*[=:≈~of]*\s*(\d+\.?\d*)\s*(min)?', "SEC_RT", "min"),
        # Molecular weight
        (r'(?:molecular\s*weight|MW)\s*[=:≈~of]*\s*(\d+\.?\d*)\s*(kDa|Da)', "MW", None),
    ]

    for pattern, metric_name, default_unit in metric_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            try:
                value = float(groups[0])
                unit = groups[1] if len(groups) > 1 and groups[1] else default_unit or ""
                # Get surrounding context (30 chars before and after)
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].replace("\n", " ").strip()

                # Avoid duplicate metric entries
                is_dup = any(
                    m["metric"] == metric_name and abs(m["value"] - value) < 0.01
                    for m in metrics
                )
                if not is_dup:
                    metrics.append({
                        "metric": metric_name,
                        "value": value,
                        "unit": unit,
                        "context": context,
                    })
            except (ValueError, IndexError):
                pass

    if not sequences and not metrics:
        warnings.append("No protein sequences or biophysical metrics detected in the text.")

    return {
        "sequences": sequences,
        "metrics": metrics,
        "n_sequences": len(sequences),
        "n_metrics": len(metrics),
        "warnings": warnings,
    }


def literature_to_training_rows(
    extracted: Dict[str, Any],
    source_name: str = "Literature",
) -> List[Dict[str, Any]]:
    """
    Convert extracted literature data into training-compatible rows
    for the Continuous Learning CSV database.

    Parameters
    ----------
    extracted : Output from extract_data_from_literature_text()
    source_name : Name/citation of the source paper

    Returns
    -------
    List of dicts compatible with the Jain-137 training schema
    """
    rows = []
    sequences = extracted.get("sequences", [])
    metrics = extracted.get("metrics", [])

    # Build a metrics lookup
    metric_map = {}
    for m in metrics:
        key = m["metric"]
        if key not in metric_map:
            metric_map[key] = m["value"]

    for i, seq_entry in enumerate(sequences):
        row = {
            "Name": f"{source_name}_seq{i+1}",
            "Sequence_HC": seq_entry["sequence"] if seq_entry["length"] > 200 else "",
            "Sequence_LC": seq_entry["sequence"] if seq_entry["length"] <= 200 else "",
        }
        # Attach any extracted metrics
        if "Aggregation" in metric_map:
            row["Exp_Aggregation_Percent"] = metric_map["Aggregation"]
        if "Tm" in metric_map:
            row["Exp_Tm_MeltingTemp"] = metric_map["Tm"]
        if "SEC_RT" in metric_map:
            row["Exp_SEC_RetentionTime"] = metric_map["SEC_RT"]
        if "Viscosity" in metric_map:
            row["Exp_Viscosity_cP"] = metric_map["Viscosity"]

        rows.append(row)

    # If we have metrics but no sequences, create a partial row
    if not sequences and metrics:
        row = {"Name": f"{source_name}_metrics"}
        if "Aggregation" in metric_map:
            row["Exp_Aggregation_Percent"] = metric_map["Aggregation"]
        if "Tm" in metric_map:
            row["Exp_Tm_MeltingTemp"] = metric_map["Tm"]
        if "pI" in metric_map:
            row["pI"] = metric_map["pI"]
        if "Titer" in metric_map:
            row["Titer_g_L"] = metric_map["Titer"]
        rows.append(row)

    return rows


# ===========================================================================
# __main__: Test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Data Pipeline v2.0 Test (Jain-137 Support)")
    print("=" * 60)

    # Test 1: Generate Mock Jain-137
    print("\n--- Test 1: Generate Mock Jain-137 Dataset ---")
    mock = generate_mock_jain137(n_samples=50, seed=137)
    print(f"  Status: {mock['status']}")
    print(f"  Rows: {mock['n_rows']}, Schema: {mock['schema']}")
    print(f"  Columns: {mock['columns']}")
    print(f"  CSV length: {len(mock['csv_string'])} chars")
    row0 = mock["data"][0]
    print(f"  Sample row: Name={row0['Name']}, HC_len={len(row0['Sequence_HC'])}, "
          f"LC_len={len(row0['Sequence_LC'])}")
    print(f"    Agg%={row0['Exp_Aggregation_Percent']}, "
          f"Tm={row0['Exp_Tm_MeltingTemp']}°C, "
          f"SEC_RT={row0['Exp_SEC_RetentionTime']}min")

    # Test 2: Parse the CSV back
    print("\n--- Test 2: Parse Mock CSV ---")
    parsed = parse_csv_upload(mock["csv_string"], "mock_jain137.csv")
    print(f"  Status: {parsed['status']}, Schema: {parsed['schema']}")
    print(f"  Rows: {parsed['n_rows']}, Columns: {len(parsed['columns'])}")
    print(f"  Warnings: {parsed['warnings']}")

    # Test 3: Validate
    print("\n--- Test 3: Validate Data ---")
    quality = validate_csv_data(parsed["data"])
    print(f"  Status: {quality['status']}")
    print(f"  With Agg: {quality['n_with_aggregation']}, With Tm: {quality['n_with_tm']}")
    print(f"  With HC seq: {quality['n_with_hc_sequence']}, "
          f"With LC seq: {quality['n_with_lc_sequence']}")

    # Test 4: Build Training Dataset
    print("\n--- Test 4: Build Training Dataset ---")
    train = build_training_dataset(
        parsed["data"],
        target_columns=["Exp_Aggregation_Percent", "Exp_Tm_MeltingTemp"],
    )
    print(f"  Status: {train['status']}")
    print(f"  Samples: {train['n_samples']}, Skipped: {train['n_skipped']}")
    if train["X"] is not None:
        print(f"  X shape: {train['X'].shape}")
        print(f"  y shape: {train['y'].shape}")
        print(f"  Feature names: {train['feature_names']}")
        print(f"  Target names: {train['target_names']}")
        print(f"  pI range: [{train['X'][:, 0].min():.2f}, {train['X'][:, 0].max():.2f}]")
        print(f"  Agg% range: [{train['y'][:, 0].min():.1f}, {train['y'][:, 0].max():.1f}]")
        print(f"  Tm range: [{train['y'][:, 1].min():.1f}, {train['y'][:, 1].max():.1f}]")

    # Test 5: Feature extraction from single row
    print("\n--- Test 5: Feature Extraction ---")
    feat = extract_features_from_jain_row(parsed["data"][0])
    if feat:
        print(f"  pI={feat['pI']}, MW={feat['mw_kda']}kDa, "
              f"Hydro={feat['hydrophobicity']}")
        print(f"  Deam={feat['deam_sites']}, Ox={feat['ox_sites']}, "
              f"Acidic={feat['acidic_residues']}, Basic={feat['basic_residues']}")

    # Test 6: Literature PDF extraction
    print("\n--- Test 6: Literature PDF Text Extraction ---")
    test_text = (
        "Abstract: We report the characterization of mAb-X (IgG1). "
        "The sequence is: EVQLVESGGGLVQPGGSLRLSCAAS. "
        "Biophysical measurements: pI = 8.7, Kd = 1.2 nM, "
        "aggregation 3.5%, titer 5.2 g/L, Tm = 72.5 C."
    )
    extracted = extract_data_from_literature_text(test_text)
    print(f"  Sequences found: {len(extracted['sequences'])}")
    print(f"  Metrics found: {len(extracted['metrics'])}")
    for m in extracted['metrics']:
        print(f"    {m['metric']}: {m['value']} {m['unit']}")

    print("\nData Pipeline v3.0 test complete")
