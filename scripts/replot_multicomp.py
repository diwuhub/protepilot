"""Re-render data/multicomp_chromatogram.png from an existing CADET .h5 — no simulation."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.CharacterizationAgent import (
    CharacterizationAgent,
    MultiComponentResult,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

H5 = ROOT / "data" / "multicomp_PRD-001.h5"
OUT = ROOT / "data" / "multicomp_chromatogram.png"

agent = CharacterizationAgent(workspace=ROOT / "data", engine_dir=ROOT / "engine")

ms = agent.get_ms_profile(product_name="mAb_ProductA", lot_number="LOT-2024-001")
components = agent.get_default_components(ms)
comp_names = ["Salt"] + [c.name for c in components]

time_s, outlet = agent._read_multicomp_results(H5, comp_names)

result = MultiComponentResult(
    h5_path=H5,
    time_s=time_s,
    outlet=outlet,
    components=components,
    ms_profile=ms,
    wall_time_s=0.0,
)

agent.plot_multi_component(
    result,
    save_path=OUT,
    show=False,
    title="ProtePilot — Multi-Component IEX Chromatogram (mAb_ProductA)",
)
print(f"saved → {OUT}")
