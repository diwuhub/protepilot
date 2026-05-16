"""
regulatory_context.py  ·  ProtePilot — Regulatory Signal Bridge
=================================================================
Lightweight cross-repo bridge to the Policy-Signal Classifier from
reg-intel-biopharma. Classifies text into one of four regulatory
signal categories:

  - new_requirement : New regulatory expectation
  - relaxation      : Reduced requirement or simplified pathway
  - maintenance     : Existing standard restated
  - ambiguous       : Unclear or mixed signal

Falls back gracefully when reg-intel-biopharma is not available.

Author  : Di (ProtePilot)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

log = logging.getLogger("ProtePilot.RegulatoryContext")

# ── Lazy-loaded classifier state ──────────────────────────────────────
_model = None
_vectorizer = None
_classifier_available: Optional[bool] = None

_REG_INTEL_ROOT = os.path.normpath(
    os.environ.get(
        "REGINTEL_ROOT",
        os.path.join(os.path.dirname(__file__), "..", "..", "reg-intel-biopharma"),
    )
)


def _ensure_classifier():
    """Attempt to import and initialise the policy-signal classifier.

    Adds reg-intel-biopharma to sys.path only once, and caches the
    trained model/vectorizer so subsequent calls are instant.
    """
    global _model, _vectorizer, _classifier_available

    if _classifier_available is not None:
        return _classifier_available

    try:
        if _REG_INTEL_ROOT not in sys.path:
            sys.path.insert(0, _REG_INTEL_ROOT)

        from models.policy_signal_classifier import train_model, predict  # noqa: F401

        _model, _vectorizer = train_model()
        _classifier_available = True
        log.info("Policy-Signal Classifier loaded from %s", _REG_INTEL_ROOT)
    except Exception as exc:
        _classifier_available = False
        log.debug("reg-intel-biopharma not available: %s", exc)

    return _classifier_available


def assess_regulatory_context(claim_text: str) -> Dict[str, Any]:
    """Classify a regulatory claim into one of four signal categories.

    Parameters
    ----------
    claim_text : str
        A sentence or short paragraph describing a regulatory position
        (e.g. from FDA guidance, ICH guidelines, EMA opinions).

    Returns
    -------
    dict
        Always contains at least ``signal_class`` and ``source``.
        On success::

            {
                "signal_class": "new_requirement" | "relaxation" | "maintenance" | "ambiguous",
                "probabilities": {"new_requirement": 0.82, ...},
                "source": "policy_signal_classifier",
            }

        On failure (classifier unavailable)::

            {
                "signal_class": "unknown",
                "note": "reg-intel-biopharma not available",
                "source": "fallback",
            }
    """
    if not claim_text or not claim_text.strip():
        return {
            "signal_class": "unknown",
            "note": "empty input",
            "source": "fallback",
        }

    if not _ensure_classifier():
        return {
            "signal_class": "unknown",
            "note": "reg-intel-biopharma not available",
            "source": "fallback",
        }

    try:
        from models.policy_signal_classifier import predict

        result = predict(claim_text, model=_model, vectorizer=_vectorizer)
        return {
            "signal_class": result["prediction"],
            "probabilities": result["probabilities"],
            "source": "policy_signal_classifier",
        }
    except Exception as exc:
        log.warning("Policy-signal prediction failed: %s", exc)
        return {
            "signal_class": "unknown",
            "note": f"prediction error: {exc}",
            "source": "fallback",
        }
