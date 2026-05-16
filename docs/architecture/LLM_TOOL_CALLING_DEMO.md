# ProtePilot LLM Tool-Calling Demo

This demo shows how an LLM-style copilot can orchestrate ProtePilot tools without turning the project into an unsupported chatbot.

## Problem

Biologics teams often have many antibody candidates and multiple disconnected analyses: sequence liabilities, developability risk, chromatography parameters, and validation planning. A plain LLM can summarize biology fluently, but it can also invent results. ProtePilot needs an auditable copilot pattern where the model asks for deterministic analyses and every claim traces back to tool output.

## Architecture

- The planner proposes tool calls from an allow-list.
- ProtePilot executes deterministic tools from `src.agents`.
- The answer layer summarizes only compact evidence extracted from tool outputs.
- The default provider is `mock`, so demos, tests, and screenshots run offline.
- Optional `openai` mode may be used with `OPENAI_API_KEY`, but it is still constrained by the same allow-list.

Current allow-list:

- `analyze_molecular_physics`
- `predict_developability_risk`
- `predict_physical_params`

## Run

```bash
python scripts/run_llm_tool_demo.py
python scripts/run_llm_tool_demo.py --json
python scripts/run_llm_tool_demo.py --out demo_outputs/protepilot_llm_tool_demo.md
```

Optional API-backed planning:

```bash
OPENAI_API_KEY=... python scripts/run_llm_tool_demo.py --provider openai
```

## Guardrails

- Unsupported tools are rejected before execution.
- The narrative includes citations such as `[T2:predict_developability_risk]`.
- The output explicitly states that results are decision support, not wet-lab validation or regulatory approval.
- Tests cover offline execution, citation presence, and allow-list rejection.
- Demo execution sets `PROTEPILOT_DISABLE_LABEL_EMISSION=1` and `PROTEPILOT_FORCE_MOCK_EMBEDDING=1` around tool calls to avoid modifying local label stores or downloading heavyweight embedding models.

## Demo Story

The strongest story is not "I added ChatGPT." The stronger story is: "I built a constrained scientific copilot where LLM planning is separated from deterministic evidence generation. That lets a reviewer inspect which tool produced each claim and prevents the model from silently changing scientific conclusions."
