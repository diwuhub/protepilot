#!/usr/bin/env python3
"""Run the ProtePilot LLM tool-calling portfolio demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_tool_demo import format_markdown_report, load_scenario, run_demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline-safe ProtePilot LLM tool-calling demo.")
    parser.add_argument(
        "--scenario",
        default="examples/llm_demo_antibody.json",
        help="Path to a JSON scenario file.",
    )
    parser.add_argument("--question", default=None, help="Override the scenario question.")
    parser.add_argument("--provider", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--out", default=None, help="Optional output path for a Markdown report.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown.")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    result = run_demo(
        scenario,
        question=args.question,
        provider=args.provider,
        model=args.model,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    if args.json:
        payload = json.dumps(result, indent=2, sort_keys=True)
    else:
        payload = format_markdown_report(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
