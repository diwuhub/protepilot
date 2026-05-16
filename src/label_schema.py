"""
Label Schema -- Core dataclasses for the labeling & training readiness layer.

Adapted from biologics-decision-engine for ProtePilot (ProtePilot).

Every scoring output in the platform becomes a LabelRecord -- a prediction
paired with an initially-empty ground_truth slot. When an expert, experiment,
or regulatory outcome fills in the ground_truth, the record becomes a
training pair that can retrain narrow models to replace heuristics.

Dataclasses:
  1. LabelRecord -- prediction + empty ground_truth (filled later)
  2. FeedbackEvent -- expert accept/reject/modify action on a record
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# =========================================================================
# 1. LabelRecord
# =========================================================================

@dataclass
class LabelRecord:
    """A prediction paired with an initially-empty ground truth slot.

    Lifecycle:
      1. Module produces a prediction -> LabelRecord created (ground_truth=None)
      2. User sees the result
      3. Expert/experiment fills ground_truth later
      4. (prediction, ground_truth) becomes a training pair

    Fields:
        record_id: Unique ID (UUID4). Auto-generated if not provided.
        module: Source module name (e.g., 'developability', 'immunogenicity').
        timestamp: When the prediction was made.
        prediction: The module's output (flexible dict -- schema varies per module).
        ground_truth: None until labeled.
        annotator: Who provided the ground truth (name or ID).
        annotation_source: How the ground truth was obtained.
        confidence_delta: How much ground truth differed from prediction (0=perfect, 1=opposite).
        metadata: Module-specific context (input parameters, versions, etc.).
    """
    module: str
    prediction: Dict[str, Any]
    record_id: str = ""
    timestamp: str = ""
    ground_truth: Optional[Dict[str, Any]] = None
    annotator: Optional[str] = None
    annotation_source: Optional[str] = None  # 'expert' | 'experiment' | 'regulatory_outcome' | 'literature'
    confidence_delta: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.record_id:
            self.record_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def is_labeled(self) -> bool:
        """True if ground_truth has been filled in."""
        return self.ground_truth is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "module": self.module,
            "timestamp": self.timestamp,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "annotator": self.annotator,
            "annotation_source": self.annotation_source,
            "confidence_delta": self.confidence_delta,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LabelRecord:
        return cls(
            record_id=data.get("record_id", ""),
            module=data["module"],
            timestamp=data.get("timestamp", ""),
            prediction=data["prediction"],
            ground_truth=data.get("ground_truth"),
            annotator=data.get("annotator"),
            annotation_source=data.get("annotation_source"),
            confidence_delta=data.get("confidence_delta"),
            metadata=data.get("metadata", {}),
        )


# =========================================================================
# 2. FeedbackEvent
# =========================================================================

@dataclass
class FeedbackEvent:
    """Expert feedback on a LabelRecord -- accept, reject, or modify.

    Links to a LabelRecord via record_id. Captures the expert's action
    and reasoning, enabling active learning workflows.

    Fields:
        event_id: Unique ID (UUID4).
        record_id: Links to the LabelRecord being reviewed.
        action: What the expert did ('accept', 'reject', 'modify').
        modified_value: If action='modify', the corrected value.
        reason: Why the expert took this action.
        source_type: Type of evidence behind the feedback.
        timestamp: When the feedback was given.
    """
    record_id: str
    action: str  # 'accept' | 'reject' | 'modify'
    event_id: str = ""
    modified_value: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    source_type: str = "expert"  # 'expert' | 'experiment' | 'regulatory' | 'literature'
    timestamp: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "record_id": self.record_id,
            "action": self.action,
            "modified_value": self.modified_value,
            "reason": self.reason,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeedbackEvent:
        return cls(
            event_id=data.get("event_id", ""),
            record_id=data["record_id"],
            action=data["action"],
            modified_value=data.get("modified_value"),
            reason=data.get("reason"),
            source_type=data.get("source_type", "expert"),
            timestamp=data.get("timestamp", ""),
        )
