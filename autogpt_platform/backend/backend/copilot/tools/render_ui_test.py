import json

import pytest

from backend.copilot.baseline.service import (
    _baseline_tool_executor,
    _BaselineStreamState,
)
from backend.copilot.capabilities import registry
from backend.copilot.capabilities.dispatch import resolve_tool_dispatch
from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.sources.static_tools import tool_entries
from backend.copilot.prompting import get_openui_supplement, get_sdk_supplement
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.sdk.tool_adapter import (
    _make_truncating_wrapper,
    _text_from_mcp_result,
    pop_pending_tool_output,
    reset_pending_tool_outputs,
    reset_tool_failure_counters,
    set_execution_context,
)
from backend.copilot.tools import TOOL_REGISTRY, get_available_tools
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import ResponseType
from backend.util.tool_call_loop import LLMToolCall

SOURCE = 'root = Workspace("Revenue", "From the uploaded report", [Metrics([Metric("Revenue", "$42,000", "Q3", "positive")])])'


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("CHAT_OPENUI_ENABLED", "true")
    monkeypatch.setattr(
        registry,
        "_registry",
        CapabilityIndex(tool_entries({"render_ui": TOOL_REGISTRY["render_ui"]}, {})),
    )
    return TOOL_REGISTRY["render_ui"]


def test_render_ui_is_discoverable_but_does_not_expand_every_turn(enabled):

    assert "render_ui" not in {
        tool["function"]["name"] for tool in get_available_tools()
    }
    call = resolve_tool_dispatch(
        "run_capability",
        {"id": "tool:render_ui", "input": {"source": SOURCE, "summary": "Revenue."}},
    )
    assert call is not None and call.tool is enabled
    assert call.name == "render_ui" and call.args["source"] == SOURCE


def test_prompt_guidance_is_opt_in_for_both_engines(monkeypatch):
    get_sdk_supplement.cache_clear()
    try:
        monkeypatch.setenv("CHAT_OPENUI_ENABLED", "false")
        assert get_openui_supplement() == ""
        assert "tool:render_ui" not in get_sdk_supplement(False)
        get_sdk_supplement.cache_clear()
        monkeypatch.setenv("CHAT_OPENUI_ENABLED", "true")
        assert "tool:render_ui" in get_openui_supplement()
        assert "tool:render_ui" in get_sdk_supplement(False)
        assert "tool:render_ui" in get_sdk_supplement(True)
    finally:
        get_sdk_supplement.cache_clear()


@pytest.mark.asyncio
async def test_render_ui_requires_a_logged_in_user(enabled):
    result = await enabled.execute(
        None, make_session("owner"), "ui-call", source=SOURCE, summary="Revenue."
    )
    assert json.loads(result.output)["type"] == "need_login"


@pytest.mark.asyncio
async def test_render_ui_limits_encoded_size_before_tool_output_truncation(enabled):
    source = 'root = Workspace("' + "🎨" * 20_000 + '", "Example", [])'
    assert len(source) < 60_000
    result = await enabled._execute(
        "owner", make_session("owner"), source=source, summary="Summary"
    )
    assert result.type == ResponseType.ERROR


@pytest.mark.asyncio
async def test_baseline_dispatch_emits_and_records_the_ui_payload(enabled):

    state = _BaselineStreamState()
    session = make_session("owner")
    await _baseline_tool_executor(
        LLMToolCall(
            id="ui-call",
            name="run_capability",
            arguments=json.dumps(
                {
                    "id": "tool:render_ui",
                    "input": {"source": SOURCE, "summary": "Revenue."},
                }
            ),
        ),
        tools=[],
        state=state,
        user_id="owner",
        session=session,
        disabled_groups=[],
        disabled_tools=frozenset(),
    )
    outputs = [
        event
        for event in state.emitted_events
        if isinstance(event, StreamToolOutputAvailable)
    ]
    assert len(outputs) == 1
    assert outputs[0].toolName == "render_ui"
    assert json.loads(outputs[0].output)["source"] == SOURCE
    records = list(state.tool_persistence.results.values())
    assert len(records) == 1 and records[0].tool_name == "render_ui"
    assert json.loads(records[0].content)["source"] == SOURCE


@pytest.mark.asyncio
async def test_sdk_dispatch_preserves_the_ui_payload_for_the_frontend(enabled):

    async def dispatcher_should_not_run(args):
        raise AssertionError("Expected dispatch to render_ui")

    args = {"source": SOURCE, "summary": "Revenue."}
    reset_pending_tool_outputs()
    reset_tool_failure_counters()
    set_execution_context("owner", make_session("owner"))
    try:
        wrapper = _make_truncating_wrapper(
            dispatcher_should_not_run, "run_capability", required_args=["id", "input"]
        )
        response = await wrapper({"id": "tool:render_ui", "input": args})
        assert json.loads(_text_from_mcp_result(response))["source"] == SOURCE
        output = pop_pending_tool_output("render_ui", args)
        assert output is not None
    finally:
        set_execution_context(None, None)
        reset_pending_tool_outputs()


@pytest.mark.asyncio
async def test_render_ui_returns_a_persistable_result(monkeypatch):
    monkeypatch.setenv("CHAT_OPENUI_ENABLED", "true")
    tool = TOOL_REGISTRY["render_ui"]
    session = make_session("owner")
    result = await tool._execute(
        user_id="owner",
        session=session,
        source=SOURCE,
        summary="Revenue was $42,000 in Q3.",
    )
    payload = json.loads(result.model_dump_json())
    assert payload["type"] == "ui_rendered"
    assert payload["source"] == SOURCE
    assert payload["message"] == "Revenue was $42,000 in Q3."
    assert payload["session_id"] == session.session_id
    assert payload["version"] == 1


@pytest.mark.asyncio
async def test_render_ui_refuses_disabled_or_another_users_session(monkeypatch):
    tool = TOOL_REGISTRY["render_ui"]
    monkeypatch.setenv("CHAT_OPENUI_ENABLED", "false")
    assert not tool.is_available
    result = await tool._execute(
        user_id="owner",
        session=make_session("owner"),
        source=SOURCE,
        summary="Revenue.",
    )
    assert result.type == ResponseType.ERROR
    monkeypatch.setenv("CHAT_OPENUI_ENABLED", "true")
    assert tool.is_available
    result = await tool._execute(
        user_id="other-user",
        session=make_session("owner"),
        source=SOURCE,
        summary="Revenue.",
    )
    assert result.type == ResponseType.ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "",
        "x" * 60_001,
        "```openui\nroot = Workspace()\n```",
        "<script>alert(1)</script>",
    ],
    ids=["empty", "oversized", "fenced", "html"],
)
async def test_render_ui_rejects_invalid_or_oversized_inputs(monkeypatch, source):
    monkeypatch.setenv("CHAT_OPENUI_ENABLED", "true")
    result = await TOOL_REGISTRY["render_ui"]._execute(
        user_id="owner",
        session=make_session("owner"),
        source=source,
        summary="Fallback.",
    )
    assert result.type == ResponseType.ERROR
