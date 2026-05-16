"""
Label Emitter -- Convenience helper for modules to emit LabelRecords.

Usage in any prediction module:
    from src.label_emitter import emit_prediction_label
    emit_prediction_label("developability", result_dict, {"input_length": 450})
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_STORE_DIR = os.path.join(_PROJECT_ROOT, "labels")


class _SafeEncoder(json.JSONEncoder):
    """Handle numpy types and other non-standard JSON types."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def _make_json_safe(obj: Any) -> Any:
    """Round-trip through JSON with safe encoder to coerce numpy types."""
    return json.loads(json.dumps(obj, cls=_SafeEncoder))


def emit_prediction_label(
    module: str,
    prediction: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    store_dir: Optional[str] = None,
) -> str:
    """Emit a LabelRecord after each prediction. Returns record_id.

    Args:
        module: Source module name (e.g., 'developability', 'immunogenicity').
        prediction: The module's output dict.
        metadata: Optional context (input params, version, etc.).
        store_dir: Override label store directory.

    Returns:
        record_id of the saved record.
    """
    if os.environ.get("PROTEPILOT_DISABLE_LABEL_EMISSION", "").lower() in {"1", "true", "yes"}:
        return "label-emission-disabled"

    from src.label_schema import LabelRecord
    from src.label_store import LabelStore

    # Coerce numpy types to native Python types for JSON serialization
    prediction = _make_json_safe(prediction)

    store = LabelStore(store_dir or _DEFAULT_STORE_DIR)
    store.rotate_if_needed(module)
    record = LabelRecord(
        module=module,
        prediction=prediction,
        metadata=metadata or {},
    )
    return store.save_record(record)
