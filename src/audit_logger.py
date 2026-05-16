"""
audit_logger.py  ·  ProtePilot — Milestone 30
===========================================================
FDA 21 CFR Part 11 Compliant Audit Trail Logger

Version   : 1.0
Author    : Di (ProtePilot)
Depends   : csv, hashlib, datetime, os, pathlib

Compliance Context
------------------------------------------------------------
21 CFR Part 11 mandates that electronic records used in
FDA-regulated processes maintain:

  1. Audit trails recording who did what and when
  2. Immutable records (append-only log design)
  3. Checksums / digital signatures for integrity
  4. Computer-generated, time-stamped entries

This module implements a secure, append-only CSV-based audit
trail suitable for GxP-compliant biopharma environments.

Logged Events
------------------------------------------------------------
  - MODEL_TRAIN       : Any ML model training event
  - BATCH_PREDICT     : Batch prediction / HT screening run
  - ECTD_REPORT       : eCTD / Executive report generation
  - PLM_EMBED         : PLM embedding extraction batch
  - FACTORY_RESET     : Model reset / deletion event
  - DATA_IMPORT       : Dataset upload / ingestion
  - SCREENING_RUN     : HT Screening pipeline execution

Audit Record Fields
------------------------------------------------------------
  Timestamp, User_ID, Action, Detail, Model_Checksum,
  Model_Version, Record_Hash

The Record_Hash is a SHA-256 hash of the entire record row
(excluding the hash itself), providing tamper-detection.

References
------------------------------------------------------------
  FDA 21 CFR Part 11: https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("ProtePilot.AuditTrail")

# ===========================================================================
#  Constants
# ===========================================================================

# Default audit trail location (relative to project root)
_DEFAULT_LOG_DIR = "logs"
_DEFAULT_LOG_FILE = "audit_trail.csv"

# CSV column headers (21 CFR Part 11 compliant fields)
AUDIT_COLUMNS = [
    "Timestamp",
    "User_ID",
    "Action",
    "Detail",
    "Model_Checksum",
    "Model_Version",
    "Record_Hash",
]

# Action type constants
ACTION_MODEL_TRAIN = "MODEL_TRAIN"
ACTION_BATCH_PREDICT = "BATCH_PREDICT"
ACTION_ECTD_REPORT = "ECTD_REPORT"
ACTION_PLM_EMBED = "PLM_EMBED"
ACTION_FACTORY_RESET = "FACTORY_RESET"
ACTION_DATA_IMPORT = "DATA_IMPORT"
ACTION_SCREENING_RUN = "SCREENING_RUN"

# Thread lock for concurrent-safe writes
_WRITE_LOCK = threading.Lock()


# ===========================================================================
#  Utility: Compute checksums and record hashes
# ===========================================================================

def compute_model_checksum(model_path: str) -> str:
    """
    Compute SHA-256 checksum of a model file for integrity verification.

    Parameters
    ----------
    model_path : Absolute or relative path to the model file.

    Returns
    -------
    str : First 16 hex characters of the SHA-256 digest, or 'N/A' if
          the file does not exist or cannot be read.
    """
    try:
        if not os.path.exists(model_path):
            return "N/A"
        sha = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()[:16]
    except Exception as e:
        log.warning("Checksum computation failed for %s: %s", model_path, e)
        return "ERROR"


def _compute_record_hash(row_values: List[str]) -> str:
    """
    Compute SHA-256 hash of all record fields for tamper detection.

    This hash covers every field in the audit row except the hash itself,
    providing a simple integrity check per 21 CFR Part 11.

    Parameters
    ----------
    row_values : List of string field values (Timestamp through Model_Version).

    Returns
    -------
    str : First 16 hex characters of the SHA-256 digest.
    """
    payload = "|".join(str(v) for v in row_values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ===========================================================================
#  Core: AuditLogger class
# ===========================================================================

class AuditLogger:
    """
    FDA 21 CFR Part 11 compliant audit trail logger.

    Writes append-only CSV records for every regulated action
    (model training, batch predictions, report generation, etc.).

    Usage
    -----
    >>> logger = AuditLogger()
    >>> logger.log_event(
    ...     action="MODEL_TRAIN",
    ...     detail="XGBoost wetlab model trained on 137 samples",
    ...     model_checksum="a1b2c3d4e5f6",
    ...     model_version="xgboost_wetlab_v1",
    ... )

    Thread Safety
    -------------
    All write operations are guarded by a threading lock to ensure
    safe concurrent access from Streamlit's rerun model.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_file: str = _DEFAULT_LOG_FILE,
        default_user: str = "System_Admin",
    ):
        """
        Initialize the audit logger.

        Parameters
        ----------
        log_dir : Directory for audit logs. Defaults to <project_root>/logs.
        log_file : Name of the CSV audit trail file.
        default_user : Default User_ID when none is provided.
        """
        if log_dir is None:
            root = Path(__file__).resolve().parent.parent
            log_dir = str(root / _DEFAULT_LOG_DIR)

        self._log_dir = log_dir
        self._log_path = os.path.join(log_dir, log_file)
        self._default_user = default_user

        # Ensure directory exists
        os.makedirs(self._log_dir, exist_ok=True)

        # Initialize CSV with headers if file does not exist
        if not os.path.exists(self._log_path):
            self._write_header()

        log.info("AuditLogger initialized: %s", self._log_path)

    def _write_header(self) -> None:
        """Write CSV header row (only on first creation)."""
        with _WRITE_LOCK:
            with open(self._log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(AUDIT_COLUMNS)

    def log_event(
        self,
        action: str,
        detail: str = "",
        model_checksum: str = "N/A",
        model_version: str = "N/A",
        user_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Append a single audit record to the trail.

        Parameters
        ----------
        action : Action type constant (e.g., ACTION_MODEL_TRAIN).
        detail : Human-readable description of the event.
        model_checksum : SHA-256 checksum of the model artifact.
        model_version : Model version identifier / filename.
        user_id : Operator ID. Defaults to self._default_user.

        Returns
        -------
        dict : The complete audit record as a dictionary.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        uid = user_id or self._default_user

        # Build row values (everything except record hash)
        row_values = [timestamp, uid, action, detail, model_checksum, model_version]
        record_hash = _compute_record_hash(row_values)
        full_row = row_values + [record_hash]

        # Append to CSV (thread-safe, append-only)
        with _WRITE_LOCK:
            with open(self._log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(full_row)

        record = dict(zip(AUDIT_COLUMNS, full_row))
        log.info("AUDIT | %s | %s | %s", uid, action, detail[:80])
        return record

    def log_model_training(
        self,
        model_type: str,
        n_samples: int,
        metrics: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Log a model training event with automatic checksum computation.

        Parameters
        ----------
        model_type : e.g., 'XGBoost_WetLab', 'XGBoost_PLM', 'XGBoost_Potency'
        n_samples : Number of training samples used.
        metrics : Optional training metrics dict (R2, RMSE, etc.).
        model_path : Path to the saved model file (for checksum).
        user_id : Operator ID.

        Returns
        -------
        dict : The audit record.
        """
        checksum = compute_model_checksum(model_path) if model_path else "N/A"
        detail = f"{model_type} trained on {n_samples} samples"
        if metrics:
            r2_vals = []
            for tname, tm in metrics.get("per_target", {}).items():
                r2_vals.append(f"{tname}:R2={tm.get('r2', 0):.3f}")
            if r2_vals:
                detail += f" [{', '.join(r2_vals)}]"

        return self.log_event(
            action=ACTION_MODEL_TRAIN,
            detail=detail,
            model_checksum=checksum,
            model_version=model_type,
            user_id=user_id,
        )

    def log_batch_prediction(
        self,
        n_candidates: int,
        model_type: str = "HT_Screening",
        duration_sec: float = 0.0,
        user_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Log a batch prediction / HT screening event.

        Parameters
        ----------
        n_candidates : Number of candidates screened.
        model_type : Model or pipeline identifier.
        duration_sec : Wall-clock time for the batch.
        user_id : Operator ID.
        """
        detail = (
            f"Batch prediction: {n_candidates} candidates via {model_type} "
            f"({duration_sec:.1f}s)"
        )
        return self.log_event(
            action=ACTION_BATCH_PREDICT,
            detail=detail,
            model_version=model_type,
            user_id=user_id,
        )

    def log_report_generation(
        self,
        report_type: str = "Executive_eCTD",
        molecule_name: str = "Unknown",
        user_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Log an eCTD or Executive Report generation event.

        Parameters
        ----------
        report_type : Type of report generated.
        molecule_name : Name of the molecule in the report.
        user_id : Operator ID.
        """
        detail = f"Report generated: {report_type} for {molecule_name}"
        return self.log_event(
            action=ACTION_ECTD_REPORT,
            detail=detail,
            model_version=report_type,
            user_id=user_id,
        )

    def log_plm_embedding(
        self,
        n_sequences: int,
        embed_dim: int = 480,
        mode: str = "esm2",
        user_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Log a PLM embedding extraction batch event.

        Parameters
        ----------
        n_sequences : Number of sequences embedded.
        embed_dim : Embedding dimensionality.
        mode : 'esm2' or 'mock'.
        user_id : Operator ID.
        """
        detail = (
            f"PLM embedding extraction: {n_sequences} sequences x {embed_dim}-dim "
            f"({mode} mode)"
        )
        return self.log_event(
            action=ACTION_PLM_EMBED,
            detail=detail,
            model_version=f"ESM2_{mode}",
            user_id=user_id,
        )

    def log_data_import(
        self,
        filename: str,
        n_rows: int,
        schema: str = "unknown",
        user_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Log a dataset upload / import event.

        Parameters
        ----------
        filename : Name of the uploaded file.
        n_rows : Number of rows imported.
        schema : Detected schema type.
        user_id : Operator ID.
        """
        detail = f"Data import: {filename} ({n_rows} rows, {schema} schema)"
        return self.log_event(
            action=ACTION_DATA_IMPORT,
            detail=detail,
            user_id=user_id,
        )

    def log_factory_reset(self, user_id: Optional[str] = None) -> Dict[str, str]:
        """Log a factory reset (model deletion) event."""
        return self.log_event(
            action=ACTION_FACTORY_RESET,
            detail="All trained models deleted and reset to baseline heuristics",
            user_id=user_id,
        )

    # -----------------------------------------------------------------------
    #  Read / query the audit trail
    # -----------------------------------------------------------------------

    def get_all_records(self) -> List[Dict[str, str]]:
        """
        Read all audit records from the CSV trail.

        Returns
        -------
        list[dict] : Each record as a dictionary keyed by AUDIT_COLUMNS.
        """
        records = []
        try:
            with open(self._log_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))
        except FileNotFoundError:
            log.warning("Audit trail file not found: %s", self._log_path)
        except Exception as e:
            log.warning("Error reading audit trail: %s", e)
        return records

    def get_records_by_action(self, action: str) -> List[Dict[str, str]]:
        """Filter audit records by action type."""
        return [r for r in self.get_all_records() if r.get("Action") == action]

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics from the audit trail.

        Returns
        -------
        dict with keys:
            total_records, total_trainings, total_screenings,
            total_reports, total_sequences_embedded,
            unique_users, first_record, last_record,
            action_counts (dict), models_trained (list).
        """
        records = self.get_all_records()
        if not records:
            return {
                "total_records": 0,
                "total_trainings": 0,
                "total_screenings": 0,
                "total_reports": 0,
                "total_sequences_embedded": 0,
                "unique_users": [],
                "first_record": None,
                "last_record": None,
                "action_counts": {},
                "models_trained": [],
            }

        action_counts: Dict[str, int] = {}
        users = set()
        models = []
        total_screened = 0
        total_embedded = 0

        for r in records:
            action = r.get("Action", "UNKNOWN")
            action_counts[action] = action_counts.get(action, 0) + 1
            users.add(r.get("User_ID", "Unknown"))

            detail = r.get("Detail", "")

            # Extract candidate count from batch predictions
            if action == ACTION_BATCH_PREDICT:
                import re
                m = re.search(r"(\d+)\s*candidates", detail)
                if m:
                    total_screened += int(m.group(1))

            # Extract sequence count from PLM embed events
            if action == ACTION_PLM_EMBED:
                import re
                m = re.search(r"(\d+)\s*sequences", detail)
                if m:
                    total_embedded += int(m.group(1))

            # Track trained models
            if action == ACTION_MODEL_TRAIN:
                models.append(r.get("Model_Version", "Unknown"))

        return {
            "total_records": len(records),
            "total_trainings": action_counts.get(ACTION_MODEL_TRAIN, 0),
            "total_screenings": action_counts.get(ACTION_BATCH_PREDICT, 0)
                               + action_counts.get(ACTION_SCREENING_RUN, 0),
            "total_reports": action_counts.get(ACTION_ECTD_REPORT, 0),
            "total_sequences_embedded": total_embedded,
            "total_candidates_screened": total_screened,
            "unique_users": sorted(users),
            "first_record": records[0].get("Timestamp", "N/A"),
            "last_record": records[-1].get("Timestamp", "N/A"),
            "action_counts": action_counts,
            "models_trained": models,
        }

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the audit trail by recalculating record hashes.

        Returns
        -------
        dict with 'valid' (bool), 'total' (int), 'corrupted' (list of indices).
        """
        records = self.get_all_records()
        corrupted = []
        for idx, r in enumerate(records):
            row_values = [
                r.get("Timestamp", ""),
                r.get("User_ID", ""),
                r.get("Action", ""),
                r.get("Detail", ""),
                r.get("Model_Checksum", ""),
                r.get("Model_Version", ""),
            ]
            expected_hash = _compute_record_hash(row_values)
            actual_hash = r.get("Record_Hash", "")
            if expected_hash != actual_hash:
                corrupted.append(idx)

        return {
            "valid": len(corrupted) == 0,
            "total": len(records),
            "corrupted": corrupted,
        }

    @property
    def log_path(self) -> str:
        """Return the absolute path to the audit trail CSV."""
        return self._log_path


# ===========================================================================
#  Module-level singleton
# ===========================================================================

_CACHED_LOGGER: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get (or create) the cached AuditLogger singleton."""
    global _CACHED_LOGGER
    if _CACHED_LOGGER is None:
        _CACHED_LOGGER = AuditLogger()
    return _CACHED_LOGGER


# ===========================================================================
#  Convenience functions (for direct import)
# ===========================================================================

def log_event(action: str, detail: str = "", **kwargs) -> Dict[str, str]:
    """Log an audit event using the singleton logger."""
    return get_audit_logger().log_event(action=action, detail=detail, **kwargs)


def log_model_training(**kwargs) -> Dict[str, str]:
    """Log a model training event using the singleton logger."""
    return get_audit_logger().log_model_training(**kwargs)


def log_batch_prediction(**kwargs) -> Dict[str, str]:
    """Log a batch prediction event using the singleton logger."""
    return get_audit_logger().log_batch_prediction(**kwargs)


def log_report_generation(**kwargs) -> Dict[str, str]:
    """Log a report generation event using the singleton logger."""
    return get_audit_logger().log_report_generation(**kwargs)


def log_plm_embedding(**kwargs) -> Dict[str, str]:
    """Log a PLM embedding batch event using the singleton logger."""
    return get_audit_logger().log_plm_embedding(**kwargs)


def log_data_import(**kwargs) -> Dict[str, str]:
    """Log a data import event using the singleton logger."""
    return get_audit_logger().log_data_import(**kwargs)


def log_factory_reset(**kwargs) -> Dict[str, str]:
    """Log a factory reset event using the singleton logger."""
    return get_audit_logger().log_factory_reset(**kwargs)


def get_summary_stats() -> Dict[str, Any]:
    """Get summary statistics from the audit trail."""
    return get_audit_logger().get_summary_stats()


def get_all_records() -> List[Dict[str, str]]:
    """Get all audit records from the trail."""
    return get_audit_logger().get_all_records()


def verify_integrity() -> Dict[str, Any]:
    """Verify audit trail integrity."""
    return get_audit_logger().verify_integrity()


# ===========================================================================
# __main__: Standalone Test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  ProtePilot — Audit Logger v1.0 Test")
    print("=" * 60)

    logger = AuditLogger(log_dir="/tmp/protepilot_audit_test")

    # Log some test events
    logger.log_model_training(
        model_type="XGBoost_WetLab",
        n_samples=137,
        metrics={"per_target": {"Agg%": {"r2": 0.85}, "Tm": {"r2": 0.91}}},
    )
    logger.log_batch_prediction(n_candidates=500, duration_sec=12.5)
    logger.log_report_generation(molecule_name="Trastuzumab")
    logger.log_plm_embedding(n_sequences=200, embed_dim=480, mode="esm2")
    logger.log_data_import(filename="jain137.csv", n_rows=137, schema="jain137")
    logger.log_factory_reset()

    # Read and verify
    records = logger.get_all_records()
    print(f"\nTotal records: {len(records)}")
    for r in records:
        print(f"  [{r['Timestamp'][:19]}] {r['Action']:20s} | {r['Detail'][:60]}")

    # Summary
    stats = logger.get_summary_stats()
    print(f"\nSummary: {stats['total_records']} records, "
          f"{stats['total_trainings']} trainings, "
          f"{stats['total_screenings']} screenings")

    # Integrity check
    integrity = logger.verify_integrity()
    print(f"\nIntegrity check: {'PASS' if integrity['valid'] else 'FAIL'} "
          f"({integrity['total']} records, {len(integrity['corrupted'])} corrupted)")

    print("\nAudit Logger v1.0 test complete")
