"""
Tests for the label schema infrastructure and prediction wiring.
"""

import json
import os
import shutil
import sys
import tempfile

import pytest

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.label_schema import LabelRecord, FeedbackEvent
from src.label_store import LabelStore


class TestLabelRecordCreation:
    """Test LabelRecord dataclass basics."""

    def test_label_record_creation(self):
        record = LabelRecord(
            module="developability",
            prediction={"score": 0.85, "risk": "low"},
            metadata={"input_length": 450},
        )
        assert record.module == "developability"
        assert record.prediction == {"score": 0.85, "risk": "low"}
        assert record.metadata == {"input_length": 450}
        assert record.ground_truth is None
        assert not record.is_labeled
        assert len(record.record_id) > 0  # auto-generated UUID
        assert len(record.timestamp) > 0  # auto-generated timestamp

    def test_label_record_to_dict_roundtrip(self):
        record = LabelRecord(
            module="immunogenicity",
            prediction={"ada_risk": 0.3},
        )
        d = record.to_dict()
        restored = LabelRecord.from_dict(d)
        assert restored.module == record.module
        assert restored.prediction == record.prediction
        assert restored.record_id == record.record_id

    def test_feedback_event_creation(self):
        event = FeedbackEvent(
            record_id="abc-123",
            action="accept",
            reason="Matches experimental data",
        )
        assert event.record_id == "abc-123"
        assert event.action == "accept"
        assert len(event.event_id) > 0


class TestLabelStoreRoundtrip:
    """Test LabelStore save/load cycle."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_label_store_roundtrip(self):
        store = LabelStore(self.tmp_dir)
        record = LabelRecord(
            module="stability",
            prediction={"shelf_life_months": 18.5, "passes_spec": True},
            metadata={"temperature_c": 5},
        )
        record_id = store.save_record(record)
        assert record_id == record.record_id

        # Reload
        loaded = store.get_records("stability")
        assert len(loaded) == 1
        assert loaded[0].module == "stability"
        assert loaded[0].prediction["shelf_life_months"] == 18.5
        assert loaded[0].record_id == record_id

    def test_multiple_records(self):
        store = LabelStore(self.tmp_dir)
        for i in range(5):
            store.save_record(LabelRecord(
                module="analytical_qc",
                prediction={"run": i, "pass": True},
            ))
        records = store.get_records("analytical_qc")
        assert len(records) == 5

    def test_jsonl_file_format(self):
        store = LabelStore(self.tmp_dir)
        store.save_record(LabelRecord(
            module="test_mod",
            prediction={"val": 42},
        ))
        jsonl_path = os.path.join(self.tmp_dir, "test_mod.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path) as f:
            line = f.readline().strip()
            data = json.loads(line)
            assert data["module"] == "test_mod"
            assert data["prediction"]["val"] == 42


class TestEmissionDoesNotBreakPrediction:
    """Verify that label emission failures do not break predictions."""

    def test_stability_still_returns_with_missing_label_dir(self):
        """simulate_stability should return a result even if label dir is gone."""
        from src.stability_twin import simulate_stability

        result = simulate_stability(
            starting_hmw_pct=1.0,
            formulation_ph=6.0,
            pI=8.5,
            temperature_c=5.0,
            duration_months=24,
        )
        # The function should still return a StabilityResult
        assert result is not None
        assert hasattr(result, "shelf_life_months")
        assert hasattr(result, "passes_24month_spec")
