"""
workspace_manager.py  ·  ProtePilot — Milestone 10
===========================================================
Enterprise Workspace Manager — Session Isolation & Persistence

Version   : 1.0
Author    : Di (ProtePilot)
Depends   : json, datetime, uuid

Purpose
------------------------------------------------------------
Provides session-based workspace isolation so each analysis run
is independent. Past sessions are stored in a JSON-backed store
and can be recalled, compared, or deleted. Each workspace holds:

  - intent        : Parsed protein parameters
  - messages       : Chat history for that run
  - results        : Developability scores, SHAP, validation plans
  - labeled_data   : Expert-corrected predictions (human-in-the-loop)
  - created_at     : Timestamp
  - display_name   : User-facing label

Storage is in-memory (st.session_state) with optional JSON export.
"""

from __future__ import annotations

import json
import logging
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("ProtePilot.WorkspaceManager")


# ===========================================================================
# 1. Workspace Data Model
# ===========================================================================

def create_workspace(
    display_name: Optional[str] = None,
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a new isolated workspace.

    Returns a workspace dict with a unique ID and empty state.
    """
    ws_id = f"ws_{uuid.uuid4().hex[:8]}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ctx_id = f"ctx_{uuid.uuid4().hex[:12]}"

    return {
        "id": ws_id,
        "context_id": ctx_id,           # Unique provenance trace — embeds in report
        "display_name": display_name or f"Run {now}",
        "created_at": now,
        "intent": intent,
        "messages": [],
        "results": {},
        "labeled_data": [],
        "dev_result": None,
        "validation_report": None,
        "ml_prediction": None,
        "export_ready": False,
        # M11+: Deep-cache for workspace history re-rendering
        "analysis_cache": None,
        # analysis_cache schema (when populated):
        # {
        #     "mode": "standard" | "bispecific",
        #     "intent": dict,
        #     "source": "fasta" | "text",
        #     "ml_override": dict | None,
        #     "source_label": str,
        #     "variants": dict | None,
        #     "dev_result": dict | None,
        #     "cqa": dict | None,
        #     "sim_summary": str | None,
        #     "sim_elapsed": float | None,
        #     "bispecific_result": dict | None,
        #     "ms_characterization": dict | None,
        # }
    }


# ===========================================================================
# 2. Workspace Store (in-memory, JSON-serializable)
# ===========================================================================

class WorkspaceStore:
    """
    Manages multiple workspaces in st.session_state.

    Usage:
        store = WorkspaceStore.from_session_state(st.session_state)
        ws = store.create_new("Adalimumab Analysis")
        store.set_active(ws["id"])
        store.save_to_session_state(st.session_state)
    """

    def __init__(self):
        self.workspaces: Dict[str, Dict[str, Any]] = {}
        self.active_id: Optional[str] = None
        self.history_order: List[str] = []  # Most recent first

    @classmethod
    def from_session_state(cls, ss: Any) -> "WorkspaceStore":
        """Load or initialize from Streamlit session_state."""
        if "workspace_store" in ss:
            return ss["workspace_store"]
        store = cls()
        ss["workspace_store"] = store
        return store

    def save_to_session_state(self, ss: Any) -> None:
        """Persist store back to session_state."""
        ss["workspace_store"] = self

    def create_new(
        self,
        display_name: Optional[str] = None,
        intent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new workspace and make it active."""
        ws = create_workspace(display_name=display_name, intent=intent)
        self.workspaces[ws["id"]] = ws
        self.history_order.insert(0, ws["id"])
        self.active_id = ws["id"]
        log.info("Created workspace %s: %s", ws["id"], ws["display_name"])
        return ws

    def get_active(self) -> Optional[Dict[str, Any]]:
        """Get the currently active workspace."""
        if self.active_id and self.active_id in self.workspaces:
            return self.workspaces[self.active_id]
        return None

    def set_active(self, ws_id: str) -> bool:
        """Switch to a different workspace."""
        if ws_id in self.workspaces:
            self.active_id = ws_id
            return True
        return False

    def delete_workspace(self, ws_id: str) -> bool:
        """Delete a workspace and switch to the next available."""
        if ws_id not in self.workspaces:
            return False

        del self.workspaces[ws_id]
        self.history_order = [h for h in self.history_order if h != ws_id]

        if self.active_id == ws_id:
            self.active_id = self.history_order[0] if self.history_order else None

        log.info("Deleted workspace %s", ws_id)
        return True

    def list_workspaces(self) -> List[Dict[str, Any]]:
        """Return workspace summaries in recent-first order."""
        result = []
        for ws_id in self.history_order:
            if ws_id in self.workspaces:
                ws = self.workspaces[ws_id]
                result.append({
                    "id": ws["id"],
                    "display_name": ws["display_name"],
                    "created_at": ws["created_at"],
                    "n_messages": len(ws["messages"]),
                    "n_labels": len(ws["labeled_data"]),
                    "has_results": bool(ws["results"]),
                    "is_active": ws["id"] == self.active_id,
                })
        return result

    def update_active_results(self, key: str, value: Any) -> None:
        """Store a result in the active workspace."""
        ws = self.get_active()
        if ws:
            ws["results"][key] = value

    def update_active_field(self, field: str, value: Any) -> None:
        """Update a top-level field in the active workspace."""
        ws = self.get_active()
        if ws:
            ws[field] = value

    def add_message_to_active(self, role: str, content: str) -> None:
        """Append a message to the active workspace."""
        ws = self.get_active()
        if ws:
            ws["messages"].append({"role": role, "content": content})

    def add_labeled_data(self, labeled_row: Dict[str, Any]) -> None:
        """Add an expert-labeled data point to the active workspace."""
        ws = self.get_active()
        if ws:
            labeled_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            labeled_row["workspace_id"] = ws["id"]
            ws["labeled_data"].append(labeled_row)
            log.info("Added labeled data to %s: %s", ws["id"], labeled_row.get("tag", ""))

    def get_all_labeled_data(self) -> List[Dict[str, Any]]:
        """Collect labeled data from ALL workspaces for retraining."""
        all_labels = []
        for ws in self.workspaces.values():
            all_labels.extend(ws.get("labeled_data", []))
        return all_labels

    @property
    def count(self) -> int:
        return len(self.workspaces)


# ===========================================================================
# 3. Report Generation
# ===========================================================================

def generate_report(
    workspace: Dict[str, Any],
    fmt: str = "markdown",
) -> str:
    """
    Export the workspace findings as a Markdown or CSV report.

    Parameters
    ----------
    workspace : Workspace dict
    fmt : "markdown" or "csv"

    Returns
    -------
    Report string (Markdown or CSV content)
    """
    if fmt == "csv":
        return _generate_csv_report(workspace)
    return _generate_markdown_report(workspace)


def _generate_markdown_report(ws: Dict[str, Any]) -> str:
    """Generate a Markdown summary report."""
    lines = [
        f"# ProtePilot — Analysis Report",
        f"",
        f"**Session:** {ws.get('display_name', 'N/A')}",
        f"**Created:** {ws.get('created_at', 'N/A')}",
        f"",
    ]

    # Intent / Protein Info
    intent = ws.get("intent")
    if intent:
        lines.extend([
            f"## Protein Parameters",
            f"",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Name | {intent.get('name', 'N/A')} |",
            f"| pI | {intent.get('pI', 'N/A')} |",
            f"| MW (kDa) | {intent.get('mw', 'N/A')} |",
            f"| Hydrophobicity | {intent.get('hydrophobicity', 'N/A')} |",
            f"| Source | {intent.get('source', 'N/A')} |",
            f"| Sequence Length | {intent.get('seq_length', 'N/A')} |",
            f"",
        ])

    # Developability
    dev = ws.get("dev_result")
    if dev and dev.get("status") == "success":
        data = dev.get("data", {})
        score = data.get("score", {})
        preds = data.get("predictions", {})
        lines.extend([
            f"## Developability Assessment",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Composite Score | {score.get('value', 'N/A')} |",
            f"| Grade | {score.get('grade', 'N/A')} |",
            f"| Aggregation Risk | {preds.get('agg_risk', 'N/A')} |",
            f"| Stability | {preds.get('stability', 'N/A')} |",
            f"| Viscosity Risk | {preds.get('viscosity_risk', 'N/A')} |",
            f"",
        ])

        # SHAP insights
        advice = data.get("advice", [])
        if advice:
            lines.append("## Actionable Insights\n")
            for a in advice:
                lines.append(f"- **{a.get('category', '')}**: {a.get('recommendation', '')}")
            lines.append("")

        # Validation Plan
        vplan = data.get("validation_plan", {})
        if vplan:
            lines.extend([
                f"## Analytical Validation Plan",
                f"",
                f"Total assays recommended: {vplan.get('total_assays', 0)}",
                f"",
            ])
            for assay in vplan.get("required_assays", []):
                lines.append(f"- **{assay.get('name', '')}** ({assay.get('priority', '')}): "
                             f"{assay.get('measures', '')}")
            lines.append("")

    # ML Prediction
    ml = ws.get("ml_prediction")
    if ml:
        lines.extend([
            f"## ML Prediction (ka/nu)",
            f"",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| ka | {ml.get('ka', 'N/A')} |",
            f"| nu | {ml.get('nu', 'N/A')} |",
            f"| Estimated RT | {ml.get('estimated_rt_min', 'N/A')} min |",
            f"",
        ])

    # Expert Labels
    labels = ws.get("labeled_data", [])
    if labels:
        lines.extend([
            f"## Expert Labels ({len(labels)} corrections)",
            f"",
            f"| Timestamp | Predicted | Actual | Tag |",
            f"|-----------|-----------|--------|-----|",
        ])
        for lb in labels:
            lines.append(
                f"| {lb.get('timestamp', '')} | "
                f"{lb.get('predicted_value', '')} | "
                f"{lb.get('actual_value', '')} | "
                f"{lb.get('tag', '')} |"
            )
        lines.append("")

    lines.extend([
        f"---",
        f"*Generated by ProtePilot v12.0 — Enterprise Biotech OS*",
    ])

    return "\n".join(lines)


def _generate_csv_report(ws: Dict[str, Any]) -> str:
    """Generate a CSV report of key findings."""
    rows = []
    header = "Category,Metric,Value"
    rows.append(header)

    intent = ws.get("intent", {})
    if intent:
        rows.append(f"Protein,Name,{intent.get('name', '')}")
        rows.append(f"Protein,pI,{intent.get('pI', '')}")
        rows.append(f"Protein,MW_kDa,{intent.get('mw', '')}")
        rows.append(f"Protein,Hydrophobicity,{intent.get('hydrophobicity', '')}")

    dev = ws.get("dev_result")
    if dev and dev.get("status") == "success":
        data = dev.get("data", {})
        score = data.get("score", {})
        preds = data.get("predictions", {})
        rows.append(f"Developability,Score,{score.get('value', '')}")
        rows.append(f"Developability,Grade,{score.get('grade', '')}")
        rows.append(f"Developability,Agg_Risk,{preds.get('agg_risk', '')}")
        rows.append(f"Developability,Stability,{preds.get('stability', '')}")
        rows.append(f"Developability,Viscosity_Risk,{preds.get('viscosity_risk', '')}")

    ml = ws.get("ml_prediction")
    if ml:
        rows.append(f"ML_Prediction,ka,{ml.get('ka', '')}")
        rows.append(f"ML_Prediction,nu,{ml.get('nu', '')}")
        rows.append(f"ML_Prediction,RT_min,{ml.get('estimated_rt_min', '')}")

    for lb in ws.get("labeled_data", []):
        rows.append(
            f"Expert_Label,{lb.get('tag', '')},"
            f"predicted={lb.get('predicted_value', '')} actual={lb.get('actual_value', '')}"
        )

    return "\n".join(rows)


# ===========================================================================
# __main__: Test
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Workspace Manager v1.0 Test")
    print("=" * 60)

    store = WorkspaceStore()

    # Create workspaces
    ws1 = store.create_new("Adalimumab Run 1")
    ws2 = store.create_new("Rituximab Batch Analysis")

    print(f"Created {store.count} workspaces")
    print(f"Active: {store.get_active()['display_name']}")

    # Switch workspace
    store.set_active(ws1["id"])
    print(f"Switched to: {store.get_active()['display_name']}")

    # Add data
    store.add_message_to_active("user", "pI 8.5 mw 150")
    store.add_message_to_active("assistant", "Processing...")
    store.update_active_field("intent", {"pI": 8.5, "mw": 150, "name": "Adalimumab"})
    store.add_labeled_data({
        "predicted_value": "15.2 min",
        "actual_value": "18.1 min",
        "tag": "RT_correction",
    })

    # List
    for ws_summary in store.list_workspaces():
        print(f"  {ws_summary['display_name']}: {ws_summary['n_messages']} msgs, "
              f"active={ws_summary['is_active']}")

    # Generate report
    report = generate_report(store.get_active())
    print(f"\nMarkdown report: {len(report)} chars")

    csv_report = generate_report(store.get_active(), fmt="csv")
    print(f"CSV report: {len(csv_report)} chars")

    # Delete
    store.delete_workspace(ws2["id"])
    print(f"After delete: {store.count} workspaces")

    # All labeled data
    all_labels = store.get_all_labeled_data()
    print(f"Total labeled data: {len(all_labels)} rows")

    print("\nWorkspace Manager test complete")
