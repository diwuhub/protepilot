import pytest

from src.llm_tool_demo import (
    PlannedToolCall,
    ToolDemoError,
    execute_tool_call,
    format_markdown_report,
    run_demo,
)


SCENARIO = {
    "question": "Should PP-test advance to validation planning?",
    "candidate": {
        "name": "PP-test",
        "molecule_type": "canonical monoclonal antibody",
        "molecule_class": "canonical_mab",
        "pI": 8.2,
        "mw": 147.9,
        "hydrophobicity": 0.4,
        "deam_sites": 1,
        "ox_sites": 2,
        "acidic_residues": 40,
        "basic_residues": 52,
        "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR",
    },
}


def test_mock_demo_runs_with_cited_tool_outputs():
    result = run_demo(SCENARIO)

    assert result["mode"] == "mock"
    assert result["candidate"]["name"] == "PP-test"
    assert len(result["tool_calls"]) == 3
    assert "predict_developability_risk" in result["answer"]
    assert "[T" in result["answer"]
    assert "does not claim wet-lab validation" in result["answer"]


def test_markdown_report_contains_trace_and_guardrails():
    result = run_demo(SCENARIO)
    report = format_markdown_report(result)

    assert "# ProtePilot LLM Tool-Calling Demo" in report
    assert "Tool Trace" in report
    assert "Only allow-listed" in report


def test_unsupported_tool_call_is_rejected():
    with pytest.raises(ToolDemoError, match="not allowed"):
        execute_tool_call(
            PlannedToolCall(
                name="delete_training_data",
                arguments={},
                reason="Should never execute.",
            )
        )
