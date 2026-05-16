"""
src/llm_copilot.py — Embedded Scientific LLM Co-Pilot
======================================================
ProtePilot — Milestone 21 · Version 1.0

Provides a context-aware conversational AI for the ProtePilot platform.
Compiles the current session state (molecule properties, developability,
upstream titer, liabilities, COGS) into a background system prompt,
then routes to either:

  1. OpenAI API (if key provided) — GPT-4o / GPT-3.5-turbo
  2. Mock response engine (graceful fallback) — references real session data

Architecture
------------
  Session State → compile_context() → system prompt
  User message + system prompt → route_to_llm() → response
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ===========================================================================
# 1. Context Compiler
# ===========================================================================

def compile_context(session_state: Dict[str, Any]) -> str:
    """
    Compile the current session state into a background context string
    for the LLM system prompt.

    Extracts key biological and process data from session_state and
    formats it as a concise scientific briefing.

    Parameters
    ----------
    session_state : Streamlit session state (or dict with same keys)

    Returns
    -------
    str : Formatted context block for the LLM system prompt
    """
    lines = ["=== ProtePilot Session Context ==="]

    # -- Active molecule --
    intent = session_state.get("last_intent") or {}
    if intent:
        lines.append(f"\n[Active Molecule]")
        lines.append(f"  Name: {intent.get('name', 'Unknown')}")
        lines.append(f"  pI: {intent.get('pI', 'N/A')}")
        lines.append(f"  MW: {intent.get('mw', 'N/A')} kDa")
        lines.append(f"  Hydrophobicity: {intent.get('hydrophobicity', 'N/A')}")
        seq = intent.get("sequence", "")
        if seq:
            lines.append(f"  Sequence length: {len(seq)} aa")
            lines.append(f"  Chains: {intent.get('num_chains', 1)}")

    # -- Developability --
    _ws = session_state.get("_workspace_store")
    _active = None
    if _ws and hasattr(_ws, "get_active"):
        _active = _ws.get_active()
    elif isinstance(_ws, dict):
        _active = _ws

    _cache = None
    if _active and isinstance(_active, dict):
        _cache = _active.get("analysis_cache")

    if _cache and isinstance(_cache, dict):
        dev = _cache.get("dev_result")
        if dev and isinstance(dev, dict):
            dev_data = dev.get("data", {})
            if dev_data:
                lines.append(f"\n[Developability Assessment]")
                lines.append(f"  Composite Score: {dev_data.get('composite_score', 'N/A')}")
                preds = dev_data.get("predictions", {})
                if preds:
                    lines.append(f"  Agg Risk: {preds.get('agg_risk', 'N/A')}")
                    lines.append(f"  Stability: {preds.get('stability', 'N/A')}")
                    lines.append(f"  Viscosity Risk: {preds.get('viscosity_risk', 'N/A')}")

    # -- Upstream --
    up_dict = session_state.get("upstream_result_dict")
    if up_dict and isinstance(up_dict, dict):
        lines.append(f"\n[Upstream Bioreactor]")
        lines.append(f"  Final Titer: {up_dict.get('final_titer', 'N/A')} g/L")
        lines.append(f"  Peak VCD: {up_dict.get('peak_vcd', 'N/A')} x10^6 cells/mL")
        lines.append(f"  Viability: {up_dict.get('viability_at_harvest', 'N/A')}%")
        lines.append(f"  Dev Penalty: {up_dict.get('dev_penalty_applied', 'N/A')}")

    # -- Downstream / DoE --
    doe_dict = session_state.get("doe_result_dict")
    if doe_dict and isinstance(doe_dict, dict):
        lines.append(f"\n[Downstream Purification DoE]")
        lines.append(f"  Optimal pH: {doe_dict.get('optimal_ph', 'N/A')}")
        lines.append(f"  Optimal Gradient: {doe_dict.get('optimal_gradient', 'N/A')} mM/min")
        lines.append(f"  Resolution: {doe_dict.get('optimal_resolution', 'N/A')}")
        lines.append(f"  Yield: {doe_dict.get('optimal_yield', 'N/A')}")

    # -- COGS --
    cogs_dict = session_state.get("cogs_result_dict")
    if cogs_dict and isinstance(cogs_dict, dict):
        lines.append(f"\n[COGS Analysis]")
        lines.append(f"  COGS: ${cogs_dict.get('cogs_per_gram', 'N/A')}/g")
        lines.append(f"  Rating: {cogs_dict.get('cost_rating', 'N/A')}")
        lines.append(f"  Batch Output: {cogs_dict.get('batch_output_g', 'N/A')} g")

    # -- Formulation --
    form_ph = session_state.get("formulation_buffer_ph")
    form_type = session_state.get("formulation_buffer_type")
    form_exc = session_state.get("formulation_excipients")
    if form_ph:
        lines.append(f"\n[Formulation]")
        lines.append(f"  Buffer pH: {form_ph}")
        lines.append(f"  Buffer Type: {form_type or 'N/A'}")
        lines.append(f"  Excipients: {', '.join(form_exc) if form_exc else 'None'}")

    # -- Glycoform --
    glyco = session_state.get("glycoform_profile")
    if glyco:
        lines.append(f"\n[Glycoform Profile]: {glyco}")

    # -- Liabilities (from intent) --
    liabilities = intent.get("liabilities")
    if liabilities and isinstance(liabilities, dict):
        flags = liabilities.get("risk_flags", [])
        if flags:
            lines.append(f"\n[Sequence Liabilities]")
            for f in flags:
                lines.append(f"  - {f}")
        lines.append(f"  Deamidation sites: {liabilities.get('deamidation_count', 0)}")
        lines.append(f"  Oxidation sites (Met): {liabilities.get('met_count', 0)}")

    if len(lines) == 1:
        lines.append("\nNo molecule or analysis data in current session.")

    return "\n".join(lines)


# ===========================================================================
# 2. System Prompt Builder
# ===========================================================================

SYSTEM_PROMPT_TEMPLATE = """You are the ProtePilot Scientific Co-Pilot, an expert in antibody \
engineering, chromatography, formulation science, upstream bioprocessing, \
and CMC (Chemistry, Manufacturing, and Controls).

You have access to the following real-time data from the current analysis session:

{context}

Instructions:
- Answer questions using the session data above as your primary reference.
- Be specific: cite actual numbers (pI, titer, COGS, etc.) from the context.
- If data is missing, say what analysis the user should run to generate it.
- Provide actionable recommendations grounded in CMC science.
- Be concise but thorough. Use scientific terminology appropriately.
- If asked about optimizations, reference the specific module (Upstream, DoE, Formulation).
"""


def build_system_prompt(session_state: Dict[str, Any]) -> str:
    """Build the full system prompt with session context."""
    context = compile_context(session_state)
    return SYSTEM_PROMPT_TEMPLATE.format(context=context)


# ===========================================================================
# 3. LLM Router (OpenAI or Mock)
# ===========================================================================

def call_openai(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """
    Call OpenAI Chat Completions API.

    Parameters
    ----------
    messages  : List of {"role": ..., "content": ...} dicts
    api_key   : OpenAI API key
    model     : Model name (default gpt-4o-mini for cost efficiency)
    max_tokens: Maximum response tokens

    Returns
    -------
    str : Assistant response text
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except ImportError:
        log.warning("openai package not installed; falling back to mock")
        return None
    except Exception as e:
        log.error("OpenAI API call failed: %s", e)
        return f"(API error: {e})"


def generate_mock_response(
    user_message: str,
    session_state: Dict[str, Any],
) -> str:
    """
    Generate a context-aware mock response that references real session data.
    Used when no OpenAI key is provided.
    """
    intent = session_state.get("last_intent") or {}
    name = intent.get("name", "your molecule")
    pI = intent.get("pI")
    mw = intent.get("mw")

    up_dict = session_state.get("upstream_result_dict") or {}
    titer = up_dict.get("final_titer")

    doe_dict = session_state.get("doe_result_dict") or {}
    opt_yield = doe_dict.get("optimal_yield")

    cogs_dict = session_state.get("cogs_result_dict") or {}
    cogs_val = cogs_dict.get("cogs_per_gram")

    msg_lower = user_message.lower()

    # Route to relevant response based on keywords
    if any(k in msg_lower for k in ["titer", "upstream", "bioreactor", "cell culture"]):
        if titer:
            return (
                f"Based on the current upstream simulation, **{name}** achieves a harvest titer "
                f"of **{titer:.2f} g/L**. "
                f"{'This is a strong titer suitable for commercial manufacturing.' if titer > 5 else 'Consider optimizing temperature shift timing or seed density to improve titer.'} "
                f"You can adjust parameters in the **Upstream Cell Culture** page."
            )
        return "No upstream simulation data yet. Run a fed-batch simulation in the **Upstream Cell Culture** page first."

    if any(k in msg_lower for k in ["cogs", "cost", "manufacturing", "commercial"]):
        if cogs_val:
            return (
                f"The current COGS for **{name}** is **${cogs_val:.2f}/g**. "
                f"{'This is commercially viable.' if cogs_val < 150 else 'This exceeds the $150/g threshold — consider improving titer or downstream yield.'} "
                f"Key levers: upstream titer (currently {titer or 'N/A'} g/L) and downstream yield (currently {opt_yield or 'N/A'})."
            )
        return "No COGS analysis yet. Run the COGS calculator in the **Manufacturability & COGS** section."

    if any(k in msg_lower for k in ["purification", "doe", "downstream", "resolution", "chromatography"]):
        if opt_yield:
            return (
                f"The DoE optimization for **{name}** identified an optimal yield of "
                f"**{opt_yield:.1%}** at pH {doe_dict.get('optimal_ph', 'N/A')} "
                f"with gradient {doe_dict.get('optimal_gradient', 'N/A')} mM/min. "
                f"Resolution = {doe_dict.get('optimal_resolution', 'N/A')}. "
                f"Check the contour plots on the **Downstream Purification** page."
            )
        return "No DoE data yet. Run the In-Silico DoE on the **Downstream Purification** page."

    if any(k in msg_lower for k in ["developability", "aggregation", "stability", "viscosity"]):
        return (
            f"For **{name}** (pI={pI or 'N/A'}, MW={mw or 'N/A'} kDa): "
            f"Run or review the developability assessment on the **Molecular Characterization** page. "
            f"The Developability Score combines aggregation risk, thermal stability, and viscosity "
            f"into a 0-1 composite metric."
        )

    if any(k in msg_lower for k in ["formulation", "buffer", "excipient", "ph"]):
        form_ph = session_state.get("formulation_buffer_ph", 6.0)
        form_type = session_state.get("formulation_buffer_type", "histidine")
        return (
            f"Current formulation: **{form_type}** buffer at pH **{form_ph}**. "
            f"For {name} (pI={pI or 'N/A'}), the buffer pH should be 1-2 units below pI "
            f"to maintain colloidal stability. Adjust settings in the sidebar Formulation panel."
        )

    if any(k in msg_lower for k in ["summary", "overview", "status", "where"]):
        parts = [f"Current session overview for **{name}**:"]
        if pI:
            parts.append(f"- pI: {pI}, MW: {mw or 'N/A'} kDa")
        if titer:
            parts.append(f"- Upstream titer: {titer:.2f} g/L")
        if opt_yield:
            parts.append(f"- Downstream yield: {opt_yield:.1%}")
        if cogs_val:
            parts.append(f"- COGS: ${cogs_val:.2f}/g")
        if len(parts) == 1:
            parts.append("No analyses run yet. Start by analyzing a sequence on the Molecular Characterization page.")
        return "\n".join(parts)

    # Generic fallback
    return (
        f"I'm the ProtePilot Co-Pilot. I can help you with questions about "
        f"**{name}**'s developability, upstream process, downstream purification, "
        f"formulation, COGS, and more. "
        f"{'Current data: pI=' + str(pI) + ', MW=' + str(mw) + ' kDa.' if pI else 'Load a molecule first on the Molecular Characterization page.'} "
        f"Try asking about titer, COGS, purification, or formulation."
    )


def route_message(
    user_message: str,
    chat_history: List[Dict[str, str]],
    session_state: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Route a user message to the appropriate LLM backend.

    Parameters
    ----------
    user_message  : The user's question
    chat_history  : Previous messages in the copilot chat
    session_state : Current Streamlit session state
    api_key       : OpenAI API key (None → mock fallback)
    model         : OpenAI model name

    Returns
    -------
    str : Assistant response
    """
    if api_key and api_key.strip().startswith("sk-"):
        # Build messages with system prompt
        system_prompt = build_system_prompt(session_state)
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent history (last 10 exchanges to stay within context)
        for msg in chat_history[-20:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})

        response = call_openai(messages, api_key, model=model)
        if response is not None:
            return response
        # Fall through to mock if OpenAI call returned None

    # Mock fallback
    return generate_mock_response(user_message, session_state)


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    passed = 0

    # Test 1: Context compilation
    mock_state = {
        "last_intent": {"name": "TestmAb", "pI": 8.5, "mw": 150.0, "sequence": "ACDE" * 100},
        "formulation_buffer_ph": 6.0,
        "formulation_buffer_type": "histidine",
        "formulation_excipients": ["trehalose"],
        "upstream_result_dict": {"final_titer": 9.05, "peak_vcd": 76.0, "viability_at_harvest": 85.0},
        "doe_result_dict": {"optimal_ph": 6.5, "optimal_gradient": 22.6, "optimal_yield": 0.40},
        "cogs_result_dict": {"cogs_per_gram": 40.43, "cost_rating": "Excellent"},
    }
    ctx = compile_context(mock_state)
    assert "TestmAb" in ctx
    assert "9.05" in ctx
    assert "40.43" in ctx
    print(f"Test 1 PASS: Context compiled ({len(ctx)} chars)")
    passed += 1

    # Test 2: System prompt
    sp = build_system_prompt(mock_state)
    assert "ProtePilot Scientific Co-Pilot" in sp
    assert "TestmAb" in sp
    print(f"Test 2 PASS: System prompt ({len(sp)} chars)")
    passed += 1

    # Test 3: Mock responses reference real data
    r1 = generate_mock_response("What is the upstream titer?", mock_state)
    assert "9.05" in r1
    print(f"Test 3 PASS: Mock titer response references real data")
    passed += 1

    # Test 4: Mock COGS response
    r2 = generate_mock_response("What are the COGS?", mock_state)
    assert "40.43" in r2
    print(f"Test 4 PASS: Mock COGS response references real data")
    passed += 1

    # Test 5: Route without API key → mock
    r3 = route_message("Tell me about purification", [], mock_state, api_key=None)
    assert len(r3) > 0
    print(f"Test 5 PASS: route_message without key → mock ({len(r3)} chars)")
    passed += 1

    print(f"\n{'='*50}")
    print(f"llm_copilot self-test: {passed}/5 passed")
    sys.exit(0 if passed == 5 else 1)
