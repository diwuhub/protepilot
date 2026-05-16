"""
Label Store -- Append-only JSONL storage for LabelRecords and FeedbackEvents.

Adapted from biologics-decision-engine for ProtePilot (ProtePilot).

One JSONL file per module in store_dir/:
  store_dir/developability.jsonl
  store_dir/immunogenicity.jsonl
  store_dir/feedback.jsonl

Design:
  - Append-only (no edits, no deletes) for audit trail
  - One record per line (JSONL) for streaming reads
  - Simple file-based (no database dependency)
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.label_schema import LabelRecord, FeedbackEvent


class LabelStore:
    """Append-only JSONL store for label records and feedback events."""

    def __init__(self, store_dir: str):
        """Initialize store. Creates store_dir if it doesn't exist.

        Args:
            store_dir: Directory for JSONL files (one per module).
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _module_path(self, module: str) -> Path:
        """Get JSONL file path for a module."""
        safe_name = module.replace("/", "_").replace("\\", "_")
        return self.store_dir / f"{safe_name}.jsonl"

    def _feedback_path(self) -> Path:
        return self.store_dir / "feedback.jsonl"

    # -- Rotation --

    def rotate_if_needed(self, module: str, max_size_mb: float = 10.0):
        """Archive JSONL files exceeding max_size_mb.

        Args:
            module: Module name whose JSONL file to check.
            max_size_mb: Size threshold in megabytes (default 10).
        """
        path = self._module_path(module)
        if path.exists() and path.stat().st_size > max_size_mb * 1024 * 1024:
            archive_path = path.with_suffix(
                f'.{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl.bak'
            )
            path.rename(archive_path)
            # Create fresh empty file
            path.touch()

    # -- Write --

    def save_record(self, record: LabelRecord) -> str:
        """Append a LabelRecord to the module's JSONL file.

        Args:
            record: The LabelRecord to save.

        Returns:
            The record_id of the saved record.
        """
        self.rotate_if_needed(record.module)
        path = self._module_path(record.module)
        with open(path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        return record.record_id

    def save_feedback(self, event: FeedbackEvent) -> str:
        """Append a FeedbackEvent to feedback.jsonl.

        Args:
            event: The FeedbackEvent to save.

        Returns:
            The event_id of the saved event.
        """
        path = self._feedback_path()
        with open(path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
        return event.event_id

    # -- Read --

    def get_records(self, module: str, labeled_only: bool = False) -> List[LabelRecord]:
        """Load all records for a module.

        Args:
            module: Module name.
            labeled_only: If True, only return records with ground_truth filled.

        Returns:
            List of LabelRecords.
        """
        path = self._module_path(module)
        if not path.exists():
            return []

        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = LabelRecord.from_dict(data)
                    if labeled_only and not record.is_labeled:
                        continue
                    records.append(record)
                except (json.JSONDecodeError, KeyError):
                    continue  # skip malformed lines
        return records

    def get_unlabeled(self, module: str) -> List[LabelRecord]:
        """Get records where ground_truth is None (need labeling).

        Args:
            module: Module name.

        Returns:
            List of unlabeled LabelRecords.
        """
        return [r for r in self.get_records(module) if not r.is_labeled]

    def get_training_pairs(self, module: str) -> List[Tuple[Dict, Dict]]:
        """Get (prediction, ground_truth) pairs for model training.

        Only returns records where both prediction and ground_truth are present.

        Args:
            module: Module name.

        Returns:
            List of (prediction_dict, ground_truth_dict) tuples.
        """
        labeled = self.get_records(module, labeled_only=True)
        return [(r.prediction, r.ground_truth) for r in labeled if r.ground_truth]

    def get_feedback(self) -> List[FeedbackEvent]:
        """Load all feedback events."""
        path = self._feedback_path()
        if not path.exists():
            return []

        events = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(FeedbackEvent.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    continue
        return events

    # -- Export --

    def export_training_csv(self, module: str, output_path: str) -> int:
        """Export training pairs as CSV.

        Columns: record_id, prediction_json, ground_truth_json

        Args:
            module: Module name.
            output_path: Output CSV file path.

        Returns:
            Number of rows exported.
        """
        labeled = self.get_records(module, labeled_only=True)
        if not labeled:
            return 0

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["record_id", "prediction_json", "ground_truth_json"])
            for r in labeled:
                writer.writerow([
                    r.record_id,
                    json.dumps(r.prediction),
                    json.dumps(r.ground_truth),
                ])
        return len(labeled)

    # -- Stats --

    def stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all modules in the store.

        Returns:
            Dict mapping module name to {total, labeled, unlabeled}.
        """
        result = {}

        for path in sorted(self.store_dir.glob("*.jsonl")):
            if path.name == "feedback.jsonl":
                continue
            module = path.stem
            records = self.get_records(module)
            labeled = sum(1 for r in records if r.is_labeled)
            result[module] = {
                "total": len(records),
                "labeled": labeled,
                "unlabeled": len(records) - labeled,
            }

        return result
