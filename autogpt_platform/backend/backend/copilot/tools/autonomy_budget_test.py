import json

import pytest

from backend.copilot import autonomy_budget


@pytest.fixture(autouse=True)
def reset_budget():
    autonomy_budget.start_autonomy_budget(enabled=False)
    yield
    autonomy_budget.start_autonomy_budget(enabled=False)


def _mcp_result(payload: dict, *, error: bool = False):
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "isError": error,
    }


def test_flag_off_keeps_configured_rounds_and_has_no_tool_guard(monkeypatch):
    monkeypatch.setattr(autonomy_budget, "FLAGGED_AUTONOMY_MAX_TOOL_CALLS", 1)
    autonomy_budget.start_autonomy_budget(enabled=False)

    assert autonomy_budget.bounded_agent_rounds(100, enabled=False) == 100
    assert autonomy_budget.before_tool("run_agent").allowed
    assert autonomy_budget.before_tool("run_agent").allowed


def test_flag_on_caps_agent_rounds():
    assert autonomy_budget.bounded_agent_rounds(100, enabled=True) == 36
    assert autonomy_budget.bounded_agent_rounds(12, enabled=True) == 12


def test_repeated_unchanged_failure_stops_only_that_path():
    autonomy_budget.start_autonomy_budget(enabled=True)
    failed = _mcp_result(
        {"type": "error", "status": "failed", "message": "Required node failed"},
        error=True,
    )

    assert autonomy_budget.before_tool("run_agent").allowed
    autonomy_budget.after_tool("run_agent", failed)
    assert autonomy_budget.before_tool("run_agent").allowed
    autonomy_budget.after_tool("run_agent", failed)

    stopped = autonomy_budget.before_tool("run_agent")
    fallback = autonomy_budget.before_tool("write_workspace_file")

    assert stopped.allowed is False
    assert stopped.reason == "unchanged"
    assert "Do not retry" in (stopped.message or "")
    assert "degraded fallback" in (stopped.message or "")
    assert fallback.allowed is True


def test_changed_arguments_allow_a_degraded_path_with_the_same_tool():
    autonomy_budget.start_autonomy_budget(enabled=True)
    failed = _mcp_result({"type": "error", "message": "source unavailable"}, error=True)
    original = {"source": "primary"}

    assert autonomy_budget.before_tool("search", original).allowed
    autonomy_budget.after_tool("search", failed, original)
    assert autonomy_budget.before_tool("search", original).allowed
    autonomy_budget.after_tool("search", failed, original)

    assert autonomy_budget.before_tool("search", original).allowed is False
    assert autonomy_budget.before_tool("search", {"source": "fallback"}).allowed


def test_meaningful_progress_resets_unchanged_state():
    autonomy_budget.start_autonomy_budget(enabled=True)
    failed = _mcp_result({"type": "error", "message": "temporary"}, error=True)

    assert autonomy_budget.before_tool("validate_agent_graph").allowed
    autonomy_budget.after_tool("validate_agent_graph", failed)
    assert autonomy_budget.before_tool("validate_agent_graph").allowed
    autonomy_budget.after_tool(
        "validate_agent_graph",
        _mcp_result({"type": "agent_builder_validation_result", "status": "delivered"}),
    )
    assert autonomy_budget.before_tool("validate_agent_graph").allowed


def test_tool_call_and_elapsed_limits_stop_with_structured_reason(monkeypatch):
    monkeypatch.setattr(autonomy_budget, "FLAGGED_AUTONOMY_MAX_TOOL_CALLS", 2)
    autonomy_budget.start_autonomy_budget(enabled=True)
    assert autonomy_budget.before_tool("one").allowed
    assert autonomy_budget.before_tool("two").allowed
    calls_stopped = autonomy_budget.before_tool("three")
    assert calls_stopped.reason == "tool_calls"

    now = [100.0]
    monkeypatch.setattr(autonomy_budget.time, "monotonic", lambda: now[0])
    autonomy_budget.start_autonomy_budget(enabled=True)
    now[0] += autonomy_budget.FLAGGED_AUTONOMY_MAX_ELAPSED_SECONDS
    elapsed_stopped = autonomy_budget.before_tool("run_agent")
    assert elapsed_stopped.reason == "elapsed"
