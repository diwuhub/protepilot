"""
batch_processor.py  ·  ProtePilot — Milestone 4A/5
===========================================================
Industrial High-Throughput Concurrent Batch Processor

Version   : 2.0 (Milestone 5 — English-Only)
Author    : Di (ProtePilot)

Architecture
------------------------------------------------------------
                    HighThroughputOrchestrator
                           |
            +--------------+--------------+
            v              v              v
      Process Pool (N workers)
      +---------+  +---------+  +---------+
      |Worker-0 |  |Worker-1 |  |Worker-2 | ...
      | Agent   |  | Agent   |  | Agent   |
      | Manager |  | Manager |  | Manager |
      | |       |  | |       |  | |       |
      | Tool1->2|  | Tool1->2|  | Tool1->2|
      | |       |  | |       |  | |       |
      | uuid.h5 |  | uuid.h5 |  | uuid.h5 |
      +---------+  +---------+  +---------+
            |              |              |
            +--------------+--------------+
                           v
                   DataFrame Aggregation
                   batch_summary.csv

Concurrency Design Notes
------------------------------------------------------------
1. CPU-Intensive:
   CADET-Core PDE solving is pure CPU computation. Python's GIL prevents
   true CPU parallelism with threads. Therefore we use ProcessPoolExecutor
   (multiprocessing), where each process has its own Python interpreter
   and memory space.

2. HDF5 I/O Conflict Prevention:
   The underlying HDF5 C library does not support concurrent multi-process
   writes to the same file. Solution: each task uses a UUID4-generated
   unique filename:  batch_{uuid4_hex[:12]}.h5
   Even with hundreds of concurrent processes writing to data/, no file-level
   conflicts will occur.

3. Process Isolation:
   Each worker process independently instantiates PharmaAgentManager and
   CadetEngine. No shared mutable state. Engine objects, HDF5 handles,
   and numpy arrays are all process-private.

4. Serialization Safety:
   ProcessPoolExecutor serializes parameters and return values via pickle.
   All inputs (dict) and outputs (dict) are JSON-serializable primitive types,
   ensuring lossless cross-process transfer.
"""

from __future__ import annotations

import logging
import os
import time as _time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("ProtePilot.Batch")


# ===========================================================================
# Worker Function — Runs in Isolated Subprocess (module-level for pickling)
# ===========================================================================

def _worker_run_pipeline(
    intent: Dict[str, Any],
    task_id: str,
    workspace: str,
    engine_dir: str,
) -> Dict[str, Any]:
    """
    Execute full pipeline for a single molecule.

    Runs in an isolated subprocess. Each call independently instantiates
    PharmaAgentManager -> predict_physical_params -> run_chromatography_simulation.

    Parameters
    ----------
    intent     : Molecule intent dictionary (name, pI, mw, deam_sites, ox_sites, ...)
    task_id    : Globally unique task ID (UUID)
    workspace  : CADET working directory
    engine_dir : cadet-cli engine directory

    Returns
    -------
    dict : {task_id, name, status, result|error, wall_time}
    """
    import sys
    # Ensure src/ is on the Python path (subprocesses may not inherit it)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    for p in (project_root, src_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    t_start = _time.time()
    mol_name = intent.get("name", "unknown")

    try:
        from src.agents import PharmaAgentManager

        # -- UUID injection: ensure unique HDF5 filenames -------------------
        # Copy intent (subprocess has independent memory, won't affect original)
        intent_copy = dict(intent)
        intent_copy["_task_id"] = task_id  # Pass to downstream for tracking

        # -- Instantiate independent Agent Manager --------------------------
        manager = PharmaAgentManager(
            workspace=workspace,
            engine_dir=engine_dir,
        )

        # -- Make run_name unique -------------------------------------------
        # PharmaAgentManager uses f"pipeline_{name}" as run_name internally
        unique_name = f"{mol_name}_{task_id}"
        intent_copy["name"] = unique_name

        # -- Execute pipeline -----------------------------------------------
        result = manager.run_deterministic_pipeline(intent_copy)

        wall = _time.time() - t_start
        return {
            "task_id":   task_id,
            "name":      mol_name,        # Original name (without UUID)
            "status":    result["status"],
            "result":    result,
            "wall_time": round(wall, 3),
        }

    except Exception as e:
        import traceback
        wall = _time.time() - t_start
        return {
            "task_id":   task_id,
            "name":      mol_name,
            "status":    "error",
            "result":    None,
            "error":     str(e),
            "traceback": traceback.format_exc(),
            "wall_time": round(wall, 3),
        }


# ===========================================================================
# HighThroughputOrchestrator — Concurrent Batch Dispatcher
# ===========================================================================

class HighThroughputOrchestrator:
    """
    Industrial high-throughput concurrent batch processor.

    Uses ProcessPoolExecutor for multi-process CPU parallelism,
    suitable for CMC early-stage molecule developability screening.

    Parameters
    ----------
    max_workers : Maximum parallel processes, default = min(CPU cores, 8)
    workspace   : CADET data directory
    engine_dir  : cadet-cli engine directory
    output_dir  : Batch report output directory

    Usage
    -----
    ::

        orch = HighThroughputOrchestrator(max_workers=4)

        intents = [
            {"name": "mAb_A", "pI": 8.5, "mw": 148.0, "deam_sites": 1, "ox_sites": 1},
            {"name": "mAb_B", "pI": 7.9, "mw": 150.0, "deam_sites": 2, "ox_sites": 0},
            {"name": "mAb_C", "pI": 8.8, "mw": 146.0, "deam_sites": 0, "ox_sites": 2},
        ]

        report = orch.run_batch(intents)
        print(report["summary"])
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        workspace:   str = "data",
        engine_dir:  str = "engine",
        output_dir:  str = "reports",
    ):
        # CPU core detection, capped at 8 (avoid excessive parallelism / memory pressure)
        cpu_count = os.cpu_count() or 4
        self.max_workers = max_workers or min(cpu_count, 8)
        self.workspace   = workspace
        self.engine_dir  = engine_dir
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "HighThroughputOrchestrator initialized: max_workers=%d, CPUs=%d",
            self.max_workers, cpu_count,
        )

    # -- Core Batch API -----------------------------------------------------

    def run_batch(
        self,
        intents: List[Dict[str, Any]],
        timeout_per_task: int = 600,
    ) -> Dict[str, Any]:
        """
        Execute multiple molecule pipelines concurrently.

        Parameters
        ----------
        intents           : List of molecule intent dicts, each containing name, pI, mw, etc.
        timeout_per_task  : Per-task timeout (s), default 600

        Returns
        -------
        dict : {
            "status":     "completed",
            "total":      int,
            "succeeded":  int,
            "failed":     int,
            "results":    [per-molecule result dicts],
            "dataframe":  pd.DataFrame (CQA summary),
            "csv_path":   str (exported CSV path),
            "summary":    str (human-readable report),
            "wall_time":  float (total elapsed time),
        }
        """
        n_total = len(intents)
        if n_total == 0:
            return {
                "status": "completed", "total": 0, "succeeded": 0,
                "failed": 0, "results": [], "dataframe": None,
                "csv_path": None, "summary": "Empty task list", "wall_time": 0,
            }

        # -- Assign UUID to each task ---------------------------------------
        tasks = []
        for intent in intents:
            task_id = uuid.uuid4().hex[:12]
            tasks.append((intent, task_id))

        effective_workers = min(self.max_workers, n_total)
        log.info(
            "Batch processing started: %d molecules, %d parallel workers",
            n_total, effective_workers,
        )

        t_batch_start = _time.time()
        results_raw: List[Dict[str, Any]] = []
        futures_map = {}

        # -- Submit all tasks to process pool --------------------------------
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            for intent, task_id in tasks:
                future = pool.submit(
                    _worker_run_pipeline,
                    intent     = intent,
                    task_id    = task_id,
                    workspace  = self.workspace,
                    engine_dir = self.engine_dir,
                )
                futures_map[future] = (intent, task_id)

            # -- Collect results (in completion order) -----------------------
            for future in as_completed(futures_map, timeout=timeout_per_task * n_total):
                intent, task_id = futures_map[future]
                mol_name = intent.get("name", "unknown")

                try:
                    result = future.result(timeout=timeout_per_task)
                    results_raw.append(result)
                    status_icon = "OK" if result["status"] == "success" else "FAIL"
                    log.info(
                        "  [%s] %s (task=%s) — %.1fs",
                        status_icon, mol_name, task_id, result["wall_time"],
                    )
                except Exception as e:
                    results_raw.append({
                        "task_id":   task_id,
                        "name":      mol_name,
                        "status":    "error",
                        "result":    None,
                        "error":     str(e),
                        "wall_time": 0,
                    })
                    log.error("  [FAIL] %s (task=%s) — Exception: %s", mol_name, task_id, e)

        wall_total = _time.time() - t_batch_start

        # -- Result statistics -----------------------------------------------
        succeeded = [r for r in results_raw if r["status"] == "success"]
        failed    = [r for r in results_raw if r["status"] != "success"]

        # -- DataFrame aggregation -------------------------------------------
        df, csv_path = self._aggregate_to_dataframe(succeeded, wall_total)

        # -- Human-readable summary ------------------------------------------
        summary = self._build_summary(
            n_total, len(succeeded), len(failed),
            results_raw, wall_total, csv_path,
        )

        log.info("Batch complete: %d/%d succeeded, total %.1fs", len(succeeded), n_total, wall_total)

        return {
            "status":    "completed",
            "total":     n_total,
            "succeeded": len(succeeded),
            "failed":    len(failed),
            "results":   results_raw,
            "dataframe": df,
            "csv_path":  str(csv_path) if csv_path else None,
            "summary":   summary,
            "wall_time": round(wall_total, 3),
        }

    # -- Sequential Execution Mode (for debugging / single-core) -------------

    def run_batch_sequential(
        self,
        intents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Sequential execution (non-concurrent), for debugging or CI/CD environments.

        Interface identical to run_batch().
        """
        n_total = len(intents)
        t_start = _time.time()
        results_raw = []

        for i, intent in enumerate(intents):
            task_id = uuid.uuid4().hex[:12]
            mol_name = intent.get("name", "unknown")
            log.info("[%d/%d] Sequential run: %s (task=%s)", i + 1, n_total, mol_name, task_id)

            result = _worker_run_pipeline(
                intent=intent, task_id=task_id,
                workspace=self.workspace, engine_dir=self.engine_dir,
            )
            results_raw.append(result)
            status_icon = "OK" if result["status"] == "success" else "FAIL"
            log.info("  [%s] %s — %.1fs", status_icon, mol_name, result["wall_time"])

        wall_total = _time.time() - t_start
        succeeded = [r for r in results_raw if r["status"] == "success"]
        failed    = [r for r in results_raw if r["status"] != "success"]
        df, csv_path = self._aggregate_to_dataframe(succeeded, wall_total)
        summary = self._build_summary(
            n_total, len(succeeded), len(failed),
            results_raw, wall_total, csv_path,
        )

        return {
            "status": "completed", "total": n_total,
            "succeeded": len(succeeded), "failed": len(failed),
            "results": results_raw, "dataframe": df,
            "csv_path": str(csv_path) if csv_path else None,
            "summary": summary, "wall_time": round(wall_total, 3),
        }

    # -- DataFrame Aggregation -----------------------------------------------

    def _aggregate_to_dataframe(
        self,
        succeeded: List[Dict[str, Any]],
        wall_total: float,
    ):
        """
        Extract core CQA metrics from successful results and aggregate into Pandas DataFrame.

        Columns:
            Molecule, Acidic_RT_min, Main_RT_min, Basic_RT_min,
            Acidic_FWHM, Main_FWHM, Basic_FWHM,
            Rs_Acidic_Main, Rs_Main_Basic,
            Acidic_Area%, Main_Area%, Basic_Area%,
            Sim_Wall_Time_s
        """
        try:
            import pandas as pd
        except ImportError:
            log.warning("pandas not installed; skipping DataFrame aggregation")
            return None, None

        if not succeeded:
            return pd.DataFrame(), None

        rows = []
        for item in succeeded:
            name = item["name"]
            r = item.get("result")
            if not r or r.get("status") != "success":
                continue

            cqa = r.get("final_cqa")
            if not cqa:
                continue

            peaks = cqa.get("peaks", {})
            res   = cqa.get("resolution", {})
            area  = cqa.get("area_pct", {})

            row = {
                "Molecule":        name,
                "Acidic_RT_min":   peaks.get("Acidic", {}).get("rt_min", 0),
                "Main_RT_min":     peaks.get("Main",   {}).get("rt_min", 0),
                "Basic_RT_min":    peaks.get("Basic",  {}).get("rt_min", 0),
                "Acidic_FWHM_min": peaks.get("Acidic", {}).get("fwhm_min", 0),
                "Main_FWHM_min":   peaks.get("Main",   {}).get("fwhm_min", 0),
                "Basic_FWHM_min":  peaks.get("Basic",  {}).get("fwhm_min", 0),
                "Acidic_Height":   peaks.get("Acidic", {}).get("height", 0),
                "Main_Height":     peaks.get("Main",   {}).get("height", 0),
                "Basic_Height":    peaks.get("Basic",  {}).get("height", 0),
                "Rs_Acidic_Main":  res.get("Acidic_vs_Main", 0),
                "Rs_Main_Basic":   res.get("Main_vs_Basic", 0),
                "Acidic_Area_pct": area.get("Acidic", 0),
                "Main_Area_pct":   area.get("Main",   0),
                "Basic_Area_pct":  area.get("Basic",  0),
                "Sim_Wall_Time_s": item.get("wall_time", 0),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # -- Export CSV ------------------------------------------------------
        timestamp = _time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f"batch_m4a_{timestamp}.csv"
        csv_path = self.output_dir / csv_filename
        df.to_csv(csv_path, index=False, float_format="%.4f")
        log.info("CQA DataFrame exported: %s (%d rows)", csv_path, len(df))

        return df, csv_path

    # -- Human-Readable Report -----------------------------------------------

    @staticmethod
    def _build_summary(
        n_total:     int,
        n_succeeded: int,
        n_failed:    int,
        results_raw: List[Dict],
        wall_total:  float,
        csv_path:    Optional[Path],
    ) -> str:
        """Generate batch processing summary report."""
        lines = [
            "=" * 60,
            "  ProtePilot — M4A High-Throughput Batch Report",
            "=" * 60,
            f"  Total tasks:    {n_total}",
            f"  Succeeded:      {n_succeeded}",
            f"  Failed:         {n_failed}",
            f"  Total time:     {wall_total:.2f}s",
            f"  Avg per mol:    {wall_total / max(n_total, 1):.2f}s/molecule",
        ]

        if csv_path:
            lines.append(f"  CSV report:     {csv_path}")

        # -- Per-molecule overview -------------------------------------------
        lines.append("")
        lines.append(f"  {'Molecule':<25s}  {'Status':^6s}  {'Main RT':>8s}  {'Rs(A/M)':>8s}  {'Rs(M/B)':>8s}  {'Time':>6s}")
        lines.append("  " + "-" * 56)

        for r in results_raw:
            name = r["name"][:25]
            status = " OK " if r["status"] == "success" else "FAIL"
            wall = f"{r.get('wall_time', 0):.1f}s"

            if r["status"] == "success" and r.get("result", {}).get("final_cqa"):
                cqa = r["result"]["final_cqa"]
                main_rt = f"{cqa['peaks']['Main']['rt_min']:.2f}"
                rs_am   = f"{cqa['resolution']['Acidic_vs_Main']:.3f}"
                rs_mb   = f"{cqa['resolution']['Main_vs_Basic']:.3f}"
            else:
                main_rt = rs_am = rs_mb = "  --"

            lines.append(f"  {name:<25s}  {status:^6s}  {main_rt:>8s}  {rs_am:>8s}  {rs_mb:>8s}  {wall:>6s}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ===========================================================================
# __main__: Local Testing
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  [%(processName)s]  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  ProtePilot — Milestone 4A: High-Throughput Concurrency Test")
    print("=" * 60)

    # -- Simulate CMC screening: 6 candidate molecules -----------------------
    screening_panel = [
        {"name": "mAb_Candidate_A", "pI": 8.45, "mw": 148.0,
         "hydrophobicity": 0.35, "deam_sites": 1, "ox_sites": 1},
        {"name": "mAb_Candidate_B", "pI": 7.90, "mw": 150.0,
         "hydrophobicity": 0.45, "deam_sites": 2, "ox_sites": 0},
        {"name": "mAb_Candidate_C", "pI": 8.80, "mw": 146.0,
         "hydrophobicity": 0.30, "deam_sites": 0, "ox_sites": 2},
        {"name": "bispecific_D",    "pI": 7.40, "mw": 145.0,
         "hydrophobicity": 0.50, "deam_sites": 1, "ox_sites": 1},
        {"name": "mAb_Candidate_E", "pI": 8.65, "mw": 149.0,
         "hydrophobicity": 0.32, "deam_sites": 1, "ox_sites": 2},
        {"name": "mAb_Candidate_F", "pI": 8.20, "mw": 147.0,
         "hydrophobicity": 0.40, "deam_sites": 2, "ox_sites": 1},
    ]

    print(f"\nScreening panel: {len(screening_panel)} candidate molecules")
    for mol in screening_panel:
        print(f"  {mol['name']}: pI={mol['pI']}, MW={mol['mw']} kDa, "
              f"deam={mol['deam_sites']}, ox={mol['ox_sites']}")

    # -- Initialize concurrent orchestrator ----------------------------------
    orch = HighThroughputOrchestrator(max_workers=4)

    # -- Execute batch (sequential mode; switch to run_batch for concurrency) -
    print("\nRunning sequential batch processing...")
    report = orch.run_batch_sequential(screening_panel)

    # -- Output report -------------------------------------------------------
    print("\n" + report["summary"])

    if report["dataframe"] is not None and len(report["dataframe"]) > 0:
        print(f"\nDataFrame preview ({len(report['dataframe'])} rows):")
        print(report["dataframe"].to_string(index=False))

    print(f"\nMilestone 4A test complete")
