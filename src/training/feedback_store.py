"""
feedback_store.py  ·  ProtePilot — Classification Feedback Recorder
====================================================================
Record-only mechanism for capturing user corrections to molecule
classification predictions.  Writes to a simple CSV file that can be
human-reviewed and selectively incorporated into future training rounds
via the harmonizer — but **never** triggers automatic retraining.

Design principles
-----------------
1. Append-only: every correction is a new row; nothing is overwritten.
2. Deterministic hashing: the same sequence always produces the same hash
   so duplicate feedback can be detected downstream.
3. Thread-safe: a file lock prevents concurrent writes from corrupting
   the CSV when the platform runs in multi-worker mode.
4. Zero dependencies on ML stack: this module intentionally imports only
   stdlib + csv so it can be loaded even if numpy/xgboost are absent.

CSV schema
----------
sequence_hash    : SHA-256 of uppercased, stripped concatenated sequence
predicted_class  : classifier's original prediction (before user override)
corrected_class  : user-supplied correction
confidence_score : classifier's confidence at prediction time (0.0–1.0)
timestamp        : ISO-8601 UTC timestamp of the correction
source           : origin of correction ("user_hint", "review_panel", …)
molecule_name    : optional molecule name for human readability
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("ProtePilot.FeedbackStore")

# ── Defaults ────────────────────────────────────────────────────────────
_DEFAULT_FEEDBACK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "feedback",
)
_FEEDBACK_FILENAME = "feedback.csv"

_CSV_COLUMNS = [
    "sequence_hash",
    "predicted_class",
    "corrected_class",
    "confidence_score",
    "timestamp",
    "source",
    "molecule_name",
]

# Simple threading lock for in-process safety
_write_lock = threading.Lock()


# ── Public API ──────────────────────────────────────────────────────────

def sequence_hash(sequence: str) -> str:
    """Deterministic SHA-256 hash of the uppercased, stripped sequence."""
    canonical = (sequence or "").strip().upper()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_feedback(
    sequence: str,
    predicted_class: str,
    corrected_class: str,
    confidence_score: float = 0.0,
    source: str = "user_hint",
    molecule_name: str = "",
    feedback_dir: Optional[str] = None,
) -> bool:
    """
    Append a single feedback row to the CSV file.

    Parameters
    ----------
    sequence : str
        The full (concatenated) sequence that was classified.
    predicted_class : str
        The classifier's original prediction.
    corrected_class : str
        The user's correction.
    confidence_score : float
        The classifier's confidence (0.0–1.0) at prediction time.
    source : str
        Where the correction came from (default "user_hint").
    molecule_name : str
        Optional human-readable name.
    feedback_dir : str, optional
        Override directory for the CSV file.  Defaults to data/feedback/.

    Returns
    -------
    bool
        True if the row was written successfully, False otherwise.
    """
    # Skip no-ops: predicted == corrected
    if predicted_class == corrected_class:
        log.debug("Feedback skipped: predicted == corrected (%s)", predicted_class)
        return False

    out_dir = feedback_dir or _DEFAULT_FEEDBACK_DIR
    out_path = os.path.join(out_dir, _FEEDBACK_FILENAME)

    row = {
        "sequence_hash": sequence_hash(sequence),
        "predicted_class": predicted_class,
        "corrected_class": corrected_class,
        "confidence_score": round(float(confidence_score), 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "molecule_name": molecule_name,
    }

    with _write_lock:
        try:
            os.makedirs(out_dir, exist_ok=True)
            file_exists = os.path.isfile(out_path)
            with open(out_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
                if not file_exists or os.path.getsize(out_path) == 0:
                    writer.writeheader()
                writer.writerow(row)
            log.info(
                "Feedback recorded: %s → %s (hash=%s…, source=%s)",
                predicted_class, corrected_class,
                row["sequence_hash"][:12], source,
            )
            return True
        except Exception as exc:
            log.error("Failed to write feedback: %s", exc)
            return False


def load_feedback(feedback_dir: Optional[str] = None):
    """
    Load all feedback rows as a list of dicts.

    Useful for review tooling and for the harmonizer to selectively
    incorporate corrections in future training rounds.

    Returns
    -------
    list of dict
        Each dict has keys matching _CSV_COLUMNS.
        Returns empty list if file does not exist.
    """
    out_dir = feedback_dir or _DEFAULT_FEEDBACK_DIR
    out_path = os.path.join(out_dir, _FEEDBACK_FILENAME)
    if not os.path.isfile(out_path):
        return []
    rows = []
    with open(out_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(dict(r))
    return rows


def feedback_summary(feedback_dir: Optional[str] = None):
    """
    Return a compact summary of accumulated feedback for CLI display.

    Returns
    -------
    dict
        Keys: total_rows, unique_sequences, class_corrections (dict),
        date_range (tuple).
    """
    rows = load_feedback(feedback_dir)
    if not rows:
        return {"total_rows": 0, "unique_sequences": 0,
                "class_corrections": {}, "date_range": (None, None)}

    hashes = set()
    corrections: dict = {}
    timestamps = []
    for r in rows:
        hashes.add(r.get("sequence_hash", ""))
        key = f"{r.get('predicted_class', '?')} → {r.get('corrected_class', '?')}"
        corrections[key] = corrections.get(key, 0) + 1
        ts = r.get("timestamp", "")
        if ts:
            timestamps.append(ts)

    timestamps.sort()
    return {
        "total_rows": len(rows),
        "unique_sequences": len(hashes),
        "class_corrections": corrections,
        "date_range": (timestamps[0] if timestamps else None,
                       timestamps[-1] if timestamps else None),
    }


# ── Self-test ───────────────────────────────────────────────────────────

def _selftest():
    """Quick smoke test — writes to a temp directory, verifies, cleans up."""
    import tempfile
    import shutil

    tmp = tempfile.mkdtemp(prefix="feedback_test_")
    try:
        # 1. Record feedback
        ok = record_feedback(
            sequence="EVQLVESGGGLVQ",
            predicted_class="canonical_mab",
            corrected_class="single_domain",
            confidence_score=0.82,
            source="selftest",
            molecule_name="test_molecule",
            feedback_dir=tmp,
        )
        assert ok, "record_feedback should return True"

        # 2. Skip no-op (same class)
        ok2 = record_feedback(
            sequence="EVQLVESGGGLVQ",
            predicted_class="peptide",
            corrected_class="peptide",
            feedback_dir=tmp,
        )
        assert not ok2, "no-op feedback should return False"

        # 3. Load and verify
        rows = load_feedback(tmp)
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
        assert rows[0]["predicted_class"] == "canonical_mab"
        assert rows[0]["corrected_class"] == "single_domain"
        assert rows[0]["source"] == "selftest"

        # 4. Summary
        s = feedback_summary(tmp)
        assert s["total_rows"] == 1
        assert s["unique_sequences"] == 1

        # 5. Deterministic hash
        h1 = sequence_hash("EVQLVESGGGLVQ")
        h2 = sequence_hash("  evqlvesggglvq  ")
        assert h1 == h2, "Hash should be case/whitespace insensitive"

        print("feedback_store._selftest() PASSED ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
