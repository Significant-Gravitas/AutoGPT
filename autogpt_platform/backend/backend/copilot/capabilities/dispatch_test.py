"""A dispatched tool call takes the path a direct tool call takes.

Deferred tools are reached as ``run_capability(id="tool:<name>")`` (#14569).
Each engine resolves that back into a call to the tool itself, so the tests
below drive a real dispatch through each engine's executor and assert that
what it records — the announce, the emitted events, the persisted row, the
failure count — names the tool that ran, never the dispatcher.
"""

import json
from unittest.mock import patch

import pytest

from backend.copilot.baseline.service import (
    _baseline_tool_executor,
    _BaselineStreamState,
)
from backend.copilot.capabilities.dispatch import resolve_tool_dispatch
from backend.copilot.capabilities.models import SKILL_TOOL
from backend.copilot.model import ChatSession
from backend.copilot.permissions import ALL_TOOL_NAMES, CopilotPermissions
from backend.copilot.response_model import (
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
)
from backend.copilot.sdk.tool_adapter import (
    _check_circuit_breaker,
    _make_truncating_wrapper,
    _text_from_mcp_result,
    pop_pending_tool_output,
    reset_pending_tool_outputs,
    reset_tool_failure_counters,
    set_execution_context,
)
from backend.copilot.tools import execute_tool
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ErrorResponse
from backend.util.tool_call_loop import LLMToolCall

# A real deferred tool, so ``resolve_entry`` finds a real registry entry; its
# implementation is stubbed because only the bookkeeping around it is on trial.
INNER = "list_schedules"
ARGS = {"x": "1"}
USER = "user-dispatch"


class _StubTool(BaseTool):
    """A real BaseTool, so ``execute``'s own bookkeeping runs for real."""

    def __init__(self, name: str, fails: bool = False) -> None:
        self._name = name
        self._fails = fails
        self.seen: dict | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "stub"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"x": {"type": "string"}}}

    async def _execute(self, user_id, session, **kwargs):
        self.seen = kwargs
        return ErrorResponse(
            message="boom" if self._fails else "ran", session_id=session.session_id
        )


def _dispatch_of(tool: BaseTool):
    """Patch the resolver's tool lookup so the dispatch lands on *tool*."""
    return patch(
        "backend.copilot.capabilities.dispatch.configured_tool", return_value=tool
    )


# ------------------------------------------------------------ the resolver


def test_resolver_yields_the_inner_call():
    for capability_id in (f"tool:{INNER}", INNER):
        call = resolve_tool_dispatch(
            "run_capability", {"id": capability_id, "input": ARGS}
        )
        assert call is not None and call.name == INNER
        assert call.args == ARGS


def test_resolver_turns_a_skill_id_into_the_read_skill_call():
    """A skill found by search is loaded the way every skill is: the
    dispatch is the ``read_skill`` call, named by the id, so the engine
    records and gates that tool.  The id wins over a ``name`` in the input."""
    tool = _StubTool(SKILL_TOOL)
    with _dispatch_of(tool):
        call = resolve_tool_dispatch(
            "run_capability",
            {"id": "skill:Triage-Tickets", "input": {"name": "other", "x": "1"}},
        )
    assert call is not None and call.tool is tool and call.name == SKILL_TOOL
    assert call.args == {"name": "triage-tickets", "x": "1"}


@pytest.mark.parametrize(
    "tool_name,args",
    [
        # validate_only describes the call; it must not run the tool.
        ("run_capability", {"id": f"tool:{INNER}", "input": {}, "validate_only": True}),
        ("run_capability", {"id": "skill:", "input": {}}),
        ("run_capability", {"id": "block:1234", "input": {}}),
        ("run_capability", {"id": "https://mcp.example/sse", "input": {}}),
        ("run_capability", {"id": "", "input": {}}),
        ("run_capability", {"id": 42}),
        ("run_capability", None),
        ("find_capability", {"query": "schedules"}),
    ],
)
def test_resolver_leaves_everything_else_alone(tool_name, args):
    assert resolve_tool_dispatch(tool_name, args) is None


@pytest.mark.parametrize("payload", ["a string", [1, 2], 42, True])
def test_resolver_refuses_an_input_that_is_not_an_object(payload):
    """Coercing a malformed ``input`` to ``{}`` would run the tool on its
    defaults, and disagree with the dispatcher, which answers "input must be an
    object" for the same call. Declining here routes it back to that answer."""
    assert (
        resolve_tool_dispatch(
            "run_capability", {"id": f"tool:{INNER}", "input": payload}
        )
        is None
    )


@pytest.mark.parametrize(
    "args", [{"id": f"tool:{INNER}", "input": {}}, {"id": f"tool:{INNER}"}]
)
def test_resolver_runs_on_defaults_when_input_is_empty_or_absent(args):
    """An absent or empty ``input`` is a legitimate "run with defaults"."""
    call = resolve_tool_dispatch("run_capability", args)
    assert call is not None and call.name == INNER and call.args == {}


# ------------------------------------------------------- the baseline engine


@pytest.mark.asyncio
async def test_baseline_dispatch_is_recorded_as_the_inner_call():
    session = ChatSession.new(USER, dry_run=False)
    state = _BaselineStreamState()
    tool = _StubTool(INNER)

    with _dispatch_of(tool):
        await _baseline_tool_executor(
            LLMToolCall(
                id="call-1",
                name="run_capability",
                arguments=json.dumps({"id": f"tool:{INNER}", "input": ARGS}),
            ),
            tools=[],
            state=state,
            user_id=USER,
            session=session,
            disabled_groups=[],
            disabled_tools=frozenset(),
        )

    assert tool.seen == ARGS
    # The announce a same-turn gate reads.
    assert session.has_tool_been_called(INNER) is True
    assert session.get_inflight_tool_call_args(INNER) == [ARGS]
    # The row that lands in history, and the result event the frontend keys on.
    recorded = list(state.tool_persistence.results.values())
    assert [r.tool_name for r in recorded] == [INNER]
    events = state.emitted_events
    inputs = [e for e in events if isinstance(e, StreamToolInputAvailable)]
    assert [(e.toolName, e.input) for e in inputs] == [(INNER, ARGS)]
    outputs = [e for e in events if isinstance(e, StreamToolOutputAvailable)]
    assert [e.toolName for e in outputs] == [INNER]


@pytest.mark.asyncio
async def test_baseline_answers_a_malformed_input_without_running_anything():
    """What the model sees for a malformed ``input``: the dispatcher's own
    validation error, under the dispatcher's name, and the target untouched."""
    session = ChatSession.new(USER, dry_run=False)
    state = _BaselineStreamState()
    tool = _StubTool(INNER)

    with _dispatch_of(tool):
        await _baseline_tool_executor(
            LLMToolCall(
                id="call-3",
                name="run_capability",
                arguments=json.dumps({"id": f"tool:{INNER}", "input": "a string"}),
            ),
            tools=[],
            state=state,
            user_id=USER,
            session=session,
            disabled_groups=[],
            disabled_tools=frozenset(),
        )

    assert tool.seen is None
    assert session.has_tool_been_called(INNER) is False
    outputs = [
        e for e in state.emitted_events if isinstance(e, StreamToolOutputAvailable)
    ]
    assert [e.toolName for e in outputs] == ["run_capability"]
    assert "input must be an object" in str(outputs[0].output)


@pytest.mark.asyncio
async def test_baseline_still_refuses_a_deferred_tool_named_directly():
    """The dispatch resolves after that refusal, so naming a deferred tool
    outright stays refused — it is the one way to reach one unguarded."""
    result = await execute_tool(
        tool_name=INNER,
        parameters={},
        user_id=USER,
        session=ChatSession.new(USER, dry_run=False),
        tool_call_id="call-2",
        disabled_groups=[],
        disabled_tools=frozenset(),
    )
    assert result.success is False
    assert "not available in this session" in result.output


# ------------------------------------------------------------ the SDK engine


@pytest.fixture
def _sdk_context():
    session = ChatSession.new(USER, dry_run=False)
    set_execution_context(USER, session)
    reset_pending_tool_outputs()
    reset_tool_failure_counters()
    yield session
    set_execution_context(None, None)


async def _never(args):
    raise AssertionError("the dispatcher's own handler must not run")


@pytest.mark.asyncio
async def test_sdk_dispatch_is_recorded_as_the_inner_call(_sdk_context):
    session = _sdk_context
    tool = _StubTool(INNER)
    wrapper = _make_truncating_wrapper(
        _never,
        "run_capability",
        input_schema={"type": "object", "properties": {"id": {}}},
        required_args=["id", "input"],
    )

    with _dispatch_of(tool):
        result = await wrapper({"id": f"tool:{INNER}", "input": ARGS})

    assert result.get("isError") is not True
    assert tool.seen == ARGS
    # The announce a same-turn gate reads.
    assert session.has_tool_been_called(INNER) is True
    assert session.get_inflight_tool_call_args(INNER) == [ARGS]
    # The output the response adapter pops is keyed by the inner call, so a
    # key written under the dispatcher would silently lose the payload.
    assert pop_pending_tool_output(INNER, ARGS) is not None


@pytest.mark.asyncio
async def test_sdk_dispatch_failure_counts_against_the_inner_tool(_sdk_context):
    """One dispatcher key per engine would let a failing tool alternate with a
    healthy one forever without ever tripping the breaker."""
    tool = _StubTool(INNER, fails=True)
    wrapper = _make_truncating_wrapper(_never, "run_capability", required_args=["id"])

    with _dispatch_of(tool), patch(
        "backend.copilot.sdk.tool_adapter.truncate",
        side_effect=lambda r, _: {**r, "isError": True},
    ):
        for _ in range(3):
            await wrapper({"id": f"tool:{INNER}", "input": ARGS})

    assert _check_circuit_breaker("run_capability", {"id": f"tool:{INNER}"}) is None
    stop = _check_circuit_breaker(INNER, ARGS)
    assert stop is not None and f"Tool '{INNER}'" in stop


@pytest.mark.asyncio
async def test_sdk_history_row_names_the_inner_call():
    """The persisted assistant row comes from the response adapter, not the
    tool handler, so it has to resolve the dispatch too or the gates read
    ``run_capability`` out of history for the rest of the session."""
    from claude_agent_sdk import AssistantMessage, ToolUseBlock

    from backend.copilot.sdk.response_adapter import SDKResponseAdapter

    adapter = SDKResponseAdapter(session_id="s-1")
    message = AssistantMessage(
        content=[
            ToolUseBlock(
                id="tu-1",
                name="mcp__copilot__run_capability",
                input={"id": f"tool:{INNER}", "input": ARGS},
            )
        ],
        model="claude-opus-5",
    )

    with _dispatch_of(_StubTool(INNER)):
        responses = adapter.convert_message(message)

    rows = [r for r in responses if isinstance(r, StreamToolInputAvailable)]
    assert [(r.toolName, r.input) for r in rows] == [(INNER, ARGS)]


@pytest.mark.asyncio
async def test_a_whitelist_grants_the_dispatcher_but_not_every_tool(_sdk_context):
    """Allowing a deferred tool implies ``run_capability``, so the dispatcher is
    reachable whenever any deferred tool is allowed. Denial then rests entirely
    on gating the RESOLVED tool: without that, one whitelisted tool would open
    every other one through the same dispatcher."""
    session = _sdk_context
    permissions = CopilotPermissions(tools=[INNER], tools_exclude=False)
    set_execution_context(USER, session, permissions=permissions)
    assert "run_capability" in permissions.effective_allowed_tools(ALL_TOOL_NAMES)

    denied = _StubTool("hire_expert")
    wrapper = _make_truncating_wrapper(_never, "run_capability", required_args=["id"])
    with _dispatch_of(denied):
        result = await wrapper({"id": "tool:hire_expert", "input": {}})

    assert result.get("isError") is True
    assert "tool_disabled" in _text_from_mcp_result(result)
    assert denied.seen is None
    assert session.has_tool_been_called("hire_expert") is False


@pytest.mark.asyncio
async def test_sdk_leaves_a_non_dispatch_call_alone(_sdk_context):
    """The wrapper's own tool keeps its name, args and handler."""
    seen = {}

    async def handler(args):
        seen.update(args)
        return {"content": [{"type": "text", "text": "ok"}], "isError": False}

    wrapper = _make_truncating_wrapper(handler, "find_capability", required_args=["q"])
    result = await wrapper({"q": "schedules"})

    assert seen == {"q": "schedules"}
    assert result.get("isError") is not True
    assert pop_pending_tool_output("find_capability", {"q": "schedules"}) is not None
