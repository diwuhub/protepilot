"""
Offline-safe LLM tool-calling demo for ProtePilot.

The module shows the tool-calling pattern clearly: an LLM-style planner may
request analyses, but all scientific claims come from deterministic ProtePilot
tools. Mock mode is the default so the demo runs in CI without
API keys. Optional OpenAI mode is intentionally thin and still constrained by
the same tool allow-list.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.agents import get_tool


ALLOWED_TOOL_NAMES = {
    "analyze_molecular_physics",
    "predict_developability_risk",
    "predict_physical_params",
}


class ToolDemoError(ValueError):
    """Raised when a demo request cannot be executed safely."""


@dataclass(frozen=True)
class PlannedToolCall:
    """A tool request proposed by the LLM/mock planner."""

    name: str
    arguments: Dict[str, Any]
    reason: str


@dataclass(frozen=True)
class ToolObservation:
    """A compact, citation-friendly view of a deterministic tool result."""

    ref: str
    tool: str
    status: str
    reason: str
    evidence: Dict[str, Any]
    message: str


def load_scenario(path: str | Path) -> Dict[str, Any]:
    """Load a JSON demo scenario."""
    with Path(path).open("r", encoding="utf-8") as f:
        scenario = json.load(f)
    if not isinstance(scenario, dict):
        raise ToolDemoError("Scenario JSON must contain an object.")
    return scenario


def run_demo(
    scenario: Mapping[str, Any],
    question: Optional[str] = None,
    provider: str = "mock",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run an LLM tool-calling demo and return a JSON-serializable report.

    Parameters
    ----------
    scenario:
        Mapping with either ``candidate`` or ``candidates``. The first candidate
        is used for this focused demo.
    question:
        User-facing question. Defaults to scenario["question"].
    provider:
        ``mock`` for deterministic offline planning or ``openai`` for optional
        API-backed tool-call planning.
    """
    candidate = _select_candidate(scenario)
    user_question = question or str(scenario.get("question") or _default_question(candidate))

    if provider == "mock":
        plan = plan_mock_tool_calls(user_question, candidate)
    elif provider == "openai":
        plan = plan_openai_tool_calls(user_question, candidate, model=model, api_key=api_key)
        if not plan:
            plan = plan_mock_tool_calls(user_question, candidate)
    else:
        raise ToolDemoError(f"Unsupported provider: {provider}. Use 'mock' or 'openai'.")

    observations: List[ToolObservation] = []
    for idx, call in enumerate(plan, 1):
        raw = execute_tool_call(call)
        observations.append(
            ToolObservation(
                ref=f"T{idx}",
                tool=call.name,
                status=str(raw.get("status", "unknown")),
                reason=call.reason,
                evidence=extract_evidence(call.name, raw),
                message=str(raw.get("message", "")),
            )
        )

    answer = build_cited_answer(user_question, candidate, observations)
    return {
        "mode": provider,
        "model": model if provider != "mock" else "mock-planner-v1",
        "question": user_question,
        "candidate": {
            "name": candidate.get("name", "Unknown candidate"),
            "molecule_type": candidate.get("molecule_type", "antibody candidate"),
        },
        "tool_calls": [
            {"ref": obs.ref, "tool": obs.tool, "status": obs.status, "reason": obs.reason}
            for obs in observations
        ],
        "evidence": [
            {"ref": obs.ref, "tool": obs.tool, "evidence": obs.evidence, "message": obs.message}
            for obs in observations
        ],
        "answer": answer,
        "guardrails": [
            "Only allow-listed ProtePilot tools can execute.",
            "The narrative cites deterministic tool outputs and does not invent wet-lab validation.",
            "The demo is decision support for prioritization, not an autonomous clinical or regulatory decision.",
        ],
    }


def plan_mock_tool_calls(question: str, candidate: Mapping[str, Any]) -> List[PlannedToolCall]:
    """Create a deterministic tool plan that mimics LLM tool selection."""
    name = str(candidate.get("name", "candidate"))
    pI = float(candidate.get("pI", candidate.get("pi", 8.2)))
    mw = float(candidate.get("mw", candidate.get("mw_kda", 148.0)))
    hydrophobicity = float(candidate.get("hydrophobicity", 0.35))
    sequence = str(candidate.get("sequence", ""))
    deam_sites = int(candidate.get("deam_sites", _count_deamidation_sites(sequence) or 1))
    ox_sites = int(candidate.get("ox_sites", sequence.count("M") or 1))
    acidic = int(candidate.get("acidic_residues", _count(sequence, "DE") or 40))
    basic = int(candidate.get("basic_residues", _count(sequence, "KRH") or 50))

    calls: List[PlannedToolCall] = []
    if sequence:
        calls.append(
            PlannedToolCall(
                name="analyze_molecular_physics",
                arguments={"sequence": sequence, "ph": float(candidate.get("ph", 7.0))},
                reason="Characterize sequence-level liabilities before interpreting developability.",
            )
        )

    calls.append(
        PlannedToolCall(
            name="predict_developability_risk",
            arguments={
                "pI": pI,
                "mw": mw,
                "hydrophobicity": hydrophobicity,
                "deam_sites": deam_sites,
                "ox_sites": ox_sites,
                "acidic_residues": acidic,
                "basic_residues": basic,
                # The sequence is already analyzed by analyze_molecular_physics.
                # Keeping this blank avoids heavyweight embedding downloads during
                # a demo trace while retaining deterministic biophysical scoring.
                "sequence": "",
                "molecule_class": candidate.get("molecule_class", "canonical_mab"),
            },
            reason="Estimate aggregation, stability, viscosity risk, and recommended validation assays.",
        )
    )

    calls.append(
        PlannedToolCall(
            name="predict_physical_params",
            arguments={
                "name": name,
                "pI": pI,
                "mw": mw,
                "hydrophobicity": hydrophobicity,
                "pH_working": float(candidate.get("pH_working", 7.0)),
                "deam_sites": deam_sites,
                "ox_sites": ox_sites,
                "sequence": sequence,
            },
            reason="Translate candidate properties into chromatography-relevant SMA parameters.",
        )
    )

    return calls


def plan_openai_tool_calls(
    question: str,
    candidate: Mapping[str, Any],
    model: str,
    api_key: Optional[str],
) -> List[PlannedToolCall]:
    """Optionally ask OpenAI to choose tool calls, still constrained by the allow-list."""
    if not api_key:
        raise ToolDemoError("OpenAI provider requires an API key. Use provider='mock' for offline demo.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ToolDemoError("OpenAI package is not installed. Use provider='mock'.") from exc

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are planning ProtePilot tool calls. You may only call the provided tools. "
                    "Do not make scientific conclusions yourself."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"question": question, "candidate": dict(candidate)}, sort_keys=True),
            },
        ],
        tools=_allowed_openai_tool_schemas(),
        tool_choice="auto",
    )

    message = response.choices[0].message
    planned: List[PlannedToolCall] = []
    for tool_call in getattr(message, "tool_calls", []) or []:
        fn = tool_call.function
        args = json.loads(fn.arguments or "{}")
        planned.append(PlannedToolCall(name=fn.name, arguments=args, reason="OpenAI-selected tool call."))
    return planned


def execute_tool_call(call: PlannedToolCall) -> Dict[str, Any]:
    """Execute one allow-listed tool call."""
    if call.name not in ALLOWED_TOOL_NAMES:
        raise ToolDemoError(f"Tool '{call.name}' is not allowed in this demo.")
    spec = get_tool(call.name)
    if spec is None:
        raise ToolDemoError(f"Tool '{call.name}' is not registered.")
    previous_label_setting = os.environ.get("PROTEPILOT_DISABLE_LABEL_EMISSION")
    previous_embedding_setting = os.environ.get("PROTEPILOT_FORCE_MOCK_EMBEDDING")
    os.environ["PROTEPILOT_DISABLE_LABEL_EMISSION"] = "1"
    os.environ["PROTEPILOT_FORCE_MOCK_EMBEDDING"] = "1"
    try:
        result = spec.func(**call.arguments)
    finally:
        if previous_label_setting is None:
            os.environ.pop("PROTEPILOT_DISABLE_LABEL_EMISSION", None)
        else:
            os.environ["PROTEPILOT_DISABLE_LABEL_EMISSION"] = previous_label_setting
        if previous_embedding_setting is None:
            os.environ.pop("PROTEPILOT_FORCE_MOCK_EMBEDDING", None)
        else:
            os.environ["PROTEPILOT_FORCE_MOCK_EMBEDDING"] = previous_embedding_setting
    if not isinstance(result, dict):
        raise ToolDemoError(f"Tool '{call.name}' returned non-dict result.")
    return result


def extract_evidence(tool_name: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    """Project large tool outputs into stable, citation-friendly evidence."""
    data = result.get("data", {}) if isinstance(result, Mapping) else {}
    if not isinstance(data, Mapping):
        return {}

    if tool_name == "analyze_molecular_physics":
        return {
            "sequence_length": data.get("sequence_length"),
            "pI": data.get("pI"),
            "mw_kda": data.get("mw_kda"),
            "gravy": data.get("gravy"),
            "deamidation_sites": data.get("deamidation_sites"),
            "oxidation_sites": data.get("oxidation_sites"),
            "hydrophobicity_normalized": data.get("hydrophobicity_normalized"),
        }

    if tool_name == "predict_developability_risk":
        score = data.get("score", {}) if isinstance(data.get("score"), Mapping) else {}
        predictions = data.get("predictions", {}) if isinstance(data.get("predictions"), Mapping) else {}
        validation_plan = data.get("validation_plan", {}) if isinstance(data.get("validation_plan"), Mapping) else {}
        advice = data.get("advice", []) if isinstance(data.get("advice"), list) else []
        return {
            "score": score.get("score"),
            "grade": score.get("grade"),
            "aggregation_risk": predictions.get("agg_risk"),
            "stability": predictions.get("stability"),
            "viscosity_risk": predictions.get("viscosity_risk"),
            "prediction_mode": data.get("prediction_mode"),
            "embedding_mode": data.get("embedding_mode"),
            "total_assays": validation_plan.get("total_assays"),
            "top_advice": [item.get("message") for item in advice[:3] if isinstance(item, Mapping)],
        }

    if tool_name == "predict_physical_params":
        variants = data.get("variants", {}) if isinstance(data.get("variants"), Mapping) else {}
        main = variants.get("main", {}) if isinstance(variants.get("main"), Mapping) else {}
        return {
            "source": data.get("source"),
            "main_variant_ka": main.get("ka"),
            "main_variant_nu": main.get("nu"),
            "main_variant_sigma": main.get("sigma"),
            "kd": variants.get("kd"),
            "lambda": variants.get("lambda_"),
        }

    return dict(data)


def build_cited_answer(
    question: str,
    candidate: Mapping[str, Any],
    observations: Iterable[ToolObservation],
) -> str:
    """Build a concise, cited demo answer."""
    obs_by_tool = {obs.tool: obs for obs in observations}
    name = candidate.get("name", "the candidate")
    lines = [
        f"Question: {question}",
        "",
        f"Decision-support summary for {name}:",
    ]

    seq = obs_by_tool.get("analyze_molecular_physics")
    if seq:
        ev = seq.evidence
        lines.append(
            "- Sequence physics: "
            f"{ev.get('sequence_length')} aa, pI {ev.get('pI')}, MW {ev.get('mw_kda')} kDa, "
            f"{ev.get('deamidation_sites')} deamidation motifs and "
            f"{ev.get('oxidation_sites')} oxidation sites "
            f"[{seq.ref}:analyze_molecular_physics]."
        )

    dev = obs_by_tool.get("predict_developability_risk")
    if dev:
        ev = dev.evidence
        lines.append(
            "- Developability: "
            f"score { _fmt(ev.get('score')) } ({ev.get('grade')}), "
            f"aggregation risk { _fmt(ev.get('aggregation_risk')) }, "
            f"stability { _fmt(ev.get('stability')) }, "
            f"viscosity risk { _fmt(ev.get('viscosity_risk')) }; "
            f"validation plan contains {ev.get('total_assays')} assays "
            f"[{dev.ref}:predict_developability_risk]."
        )
        advice = ev.get("top_advice") or []
        if advice:
            lines.append(f"- Top recommended action: {advice[0]} [{dev.ref}:top_advice].")

    phys = obs_by_tool.get("predict_physical_params")
    if phys:
        ev = phys.evidence
        lines.append(
            "- Chromatography input: "
            f"main variant ka { _fmt(ev.get('main_variant_ka')) }, "
            f"nu { _fmt(ev.get('main_variant_nu')) }, "
            f"sigma { _fmt(ev.get('main_variant_sigma')) }, "
            f"source {ev.get('source')} [{phys.ref}:predict_physical_params]."
        )

    lines.extend(
        [
            "",
            "Professional interpretation:",
            "Use this as a ranked triage signal for engineering and assay planning. "
            "It does not claim wet-lab validation, clinical readiness, or regulatory approval.",
        ]
    )
    return "\n".join(lines)


def format_markdown_report(result: Mapping[str, Any]) -> str:
    """Render the JSON result as a compact Markdown demo artifact."""
    lines = [
        "# ProtePilot LLM Tool-Calling Demo",
        "",
        f"**Mode:** {result.get('mode')}  ",
        f"**Candidate:** {result.get('candidate', {}).get('name')}  ",
        f"**Question:** {result.get('question')}",
        "",
        "## Tool Trace",
    ]
    for call in result.get("tool_calls", []):
        lines.append(f"- `{call['ref']}` `{call['tool']}`: {call['reason']} Status: `{call['status']}`.")

    lines.extend(["", "## Answer", "", str(result.get("answer", "")), "", "## Guardrails"])
    for guardrail in result.get("guardrails", []):
        lines.append(f"- {guardrail}")
    return "\n".join(lines) + "\n"


def _select_candidate(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(scenario.get("candidate"), Mapping):
        return scenario["candidate"]
    candidates = scenario.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
        return candidates[0]
    raise ToolDemoError("Scenario must include candidate or candidates[0].")


def _default_question(candidate: Mapping[str, Any]) -> str:
    return f"Evaluate {candidate.get('name', 'this antibody')} for developability and CMC risk."


def _allowed_openai_tool_schemas() -> List[Dict[str, Any]]:
    schemas: List[Dict[str, Any]] = []
    for tool_name in sorted(ALLOWED_TOOL_NAMES):
        spec = get_tool(tool_name)
        if spec is None:
            continue
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
        )
    return schemas


def _count(sequence: str, alphabet: str) -> int:
    return sum(1 for aa in sequence.upper() if aa in alphabet)


def _count_deamidation_sites(sequence: str) -> int:
    sequence = sequence.upper()
    return sum(1 for idx in range(len(sequence) - 1) if sequence[idx] == "N" and sequence[idx + 1] in {"G", "S"})


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
