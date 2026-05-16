"""
src/result_logger.py — Backend Result Logging
==============================================
ProtePilot — v7.3.1

Stores per-run results as JSON files for offline analysis.
Each analysis run generates a timestamped log entry containing
all key outputs from every module (sequence analysis, chromatography,
developability, PK, upstream, downstream).

Usage:
    from src.result_logger import ResultLogger
    logger = ResultLogger()
    logger.log_run(run_data)         # Store a complete run
    runs = logger.list_runs()         # List all stored runs
    data = logger.load_run(run_id)    # Load a specific run

Storage:
    logs/results/YYYY-MM-DD_HH-MM-SS_{molecule_name}.json
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

log = logging.getLogger("ProtePilot.ResultLogger")


class ResultLogger:
    """Persistent result logger for ProtePilot analysis runs."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.join(root, "logs", "results")
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def log_run(self, run_data: Dict[str, Any]) -> str:
        """
        Log a complete analysis run.

        Parameters
        ----------
        run_data : dict with keys like:
            - molecule_name: str
            - assembly: dict (chains, pI, MW, etc.)
            - chromatography: dict (ka, nu, RT, Rs, etc.)
            - developability: dict (scores, SHAP, etc.)
            - pk: dict (half_life, penalties, etc.)
            - upstream: dict (titer, VCD, etc.)
            - downstream: dict (DoE optimal, yield, etc.)
            - sanity_warnings: list of warning dicts

        Returns
        -------
        str : run_id (filename stem)
        """
        now = datetime.now()
        mol_name = run_data.get("molecule_name", "unknown")
        # Sanitize molecule name for filename
        safe_name = re.sub(r'[^\w\-]', '_', mol_name)[:40]
        run_id = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{safe_name}"

        # Add metadata
        run_data["_meta"] = {
            "run_id": run_id,
            "timestamp": now.isoformat(),
            "platform_version": "v31.0",
            "mapper_version": "v7.3.1",
        }

        filepath = os.path.join(self.base_dir, f"{run_id}.json")
        try:
            with open(filepath, "w") as f:
                json.dump(run_data, f, indent=2, default=str)
            log.info("Run logged: %s (%d bytes)", filepath, os.path.getsize(filepath))
        except Exception as e:
            log.warning("Failed to log run: %s", e)

        return run_id

    def list_runs(self, limit: int = 50) -> List[Dict[str, str]]:
        """List recent runs (newest first)."""
        files = []
        try:
            for fn in sorted(os.listdir(self.base_dir), reverse=True):
                if fn.endswith(".json"):
                    files.append({
                        "run_id": fn[:-5],
                        "filename": fn,
                        "path": os.path.join(self.base_dir, fn),
                    })
                    if len(files) >= limit:
                        break
        except OSError:
            pass
        return files

    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific run by ID."""
        filepath = os.path.join(self.base_dir, f"{run_id}.json")
        if not os.path.exists(filepath):
            # Try partial match
            for fn in os.listdir(self.base_dir):
                if run_id in fn:
                    filepath = os.path.join(self.base_dir, fn)
                    break
            else:
                return None
        try:
            with open(filepath) as f:
                return json.load(f)
        except Exception as e:
            log.warning("Failed to load run %s: %s", run_id, e)
            return None

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent run."""
        runs = self.list_runs(limit=1)
        if runs:
            return self.load_run(runs[0]["run_id"])
        return None
