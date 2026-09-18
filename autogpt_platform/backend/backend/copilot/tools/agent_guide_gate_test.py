"""Tests for the ``require_guide_read`` gate on agent-generation tools.

The agent-building guide carries block ids, link semantics, and
AgentExecutorBlock / MCPToolBlock conventions that the agent needs before
producing agent JSON. Without the gate, agents often skip the guide to save
tokens and then produce JSON that fails validation — wasting turns on
auto-fix loops.
"""

import json

import pytest

from backend.copilot.model import ChatMessage, ChatSession

from .helpers import require_guide_read
from .models import ErrorResponse


def _session_with_messages(
    messages: list[ChatMessage],
    builder_graph_id: str | None = None,
) -> ChatSession:
    """Build a real ChatSession with the given messages.

    Uses ``ChatSession.new`` + attribute reassignment rather than
    ``MagicMock(spec=...)`` because the gate now calls
    ``session.has_tool_been_called(...)`` and a ``spec`` mock
    returns a truthy ``MagicMock`` from that call, hiding real gate
    behaviour.  A live ``ChatSession`` also correctly initialises the
    ``_inflight_tool_calls`` PrivateAttr scratch buffer used by the
    in-turn announcement path.
    """
    session = ChatSession.new(
        "test-user", dry_run=False, builder_graph_id=builder_graph_id
    )
    session.session_id = "test-session"
    session.messages = messages
    return session


def test_no_messages_gate_fires():
    session = _session_with_messages([])
    result = require_guide_read(session, "create_agent")
    assert isinstance(result, ErrorResponse)
    assert "get_agent_building_guide" in result.message
    assert "create_agent" in result.message


def test_user_message_only_gate_fires():
    session = _session_with_messages(
        [ChatMessage(role="user", content="build an agent")]
    )
    assert isinstance(require_guide_read(session, "create_agent"), ErrorResponse)


def test_assistant_without_tool_calls_gate_fires():
    session = _session_with_messages(
        [ChatMessage(role="assistant", content="sure!", tool_calls=None)]
    )
    assert isinstance(require_guide_read(session, "create_agent"), ErrorResponse)


def test_unrelated_tool_call_gate_fires():
    session = _session_with_messages(
        [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[{"function": {"name": "find_block"}}],
            )
        ]
    )
    assert isinstance(require_guide_read(session, "create_agent"), ErrorResponse)


def test_guide_called_via_openai_shape_gate_passes():
    """OpenAI/Anthropic wrap names under 'function': {'name': ...}."""
    session = _session_with_messages(
        [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {"function": {"name": "get_agent_building_guide"}},
                ],
            )
        ]
    )
    assert require_guide_read(session, "create_agent") is None


def test_guide_called_via_flat_shape_gate_passes():
    """Some callers log tool calls with a flat {'name': ...} shape."""
    session = _session_with_messages(
        [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[{"name": "get_agent_building_guide"}],
            )
        ]
    )
    assert require_guide_read(session, "create_agent") is None


def test_guide_earlier_in_history_still_passes():
    """A guide call earlier in the session keeps the gate open for subsequent
    create/edit/validate/fix calls — the agent doesn't need to re-read it."""
    session = _session_with_messages(
        [
            ChatMessage(role="user", content="build X"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[{"function": {"name": "get_agent_building_guide"}}],
            ),
            ChatMessage(role="user", content="also Y"),
            ChatMessage(role="assistant", content="working on it"),
        ]
    )
    assert require_guide_read(session, "edit_agent") is None


@pytest.mark.parametrize(
    "tool_name",
    ["create_agent", "edit_agent", "validate_agent_graph", "fix_agent_graph"],
)
def test_tool_name_surfaced_in_error(tool_name: str):
    session = _session_with_messages([])
    result = require_guide_read(session, tool_name)
    assert isinstance(result, ErrorResponse)
    assert tool_name in result.message


def test_inflight_announcement_lets_gate_pass_within_same_turn():
    """Regression for the Kimi baseline loop: the guide call is
    dispatched earlier in the SAME turn and buffered by the
    ``_baseline_tool_executor`` into the in-flight announcement set,
    but hasn't been flushed into ``session.messages`` yet.  The gate
    must see it anyway — otherwise a follow-up ``create_agent`` in the
    same turn re-fires the guard despite the guide call and the model
    loops retrying the guide."""
    session = _session_with_messages(
        [ChatMessage(role="user", content="build something")]
    )
    # Simulate _baseline_tool_executor's announce.
    session.announce_inflight_tool_call("get_agent_building_guide")
    assert require_guide_read(session, "create_agent") is None


def test_inflight_clear_restores_gate_for_next_turn():
    """End-of-turn cleanup must drop the in-flight buffer so it can't
    leak into the *next* turn's ``session.messages`` scan (e.g. a second
    session turn that should legitimately require a fresh guide call if
    ``messages`` got compressed away)."""
    session = _session_with_messages([ChatMessage(role="user", content="build")])
    session.announce_inflight_tool_call("get_agent_building_guide")
    assert require_guide_read(session, "create_agent") is None
    session.clear_inflight_tool_calls()
    # With the buffer cleared and no guide row in messages, the guard
    # fires again.
    assert isinstance(require_guide_read(session, "create_agent"), ErrorResponse)


def test_inflight_announcement_does_not_serialise_into_model_dump():
    """PrivateAttr invariant: the scratch buffer must never leak into
    ``model_dump()`` / the Redis cache payload / the DB — it's
    process-local turn state, not durable session state."""
    session = _session_with_messages([])
    session.announce_inflight_tool_call("get_agent_building_guide")
    dumped = session.model_dump()
    assert "_inflight_tool_calls" not in dumped
    assert "inflight_tool_calls" not in dumped


def test_builder_bound_session_bypasses_gate():
    """Builder-bound sessions receive the guide via <builder_context> on
    every turn, so the tool-call gate is unnecessary and only wastes a
    round-trip."""
    session = _session_with_messages(
        [ChatMessage(role="user", content="edit this agent")],
        builder_graph_id="graph-abc",
    )
    assert require_guide_read(session, "edit_agent") is None


def test_builder_bound_session_bypasses_gate_for_all_tools():
    session = _session_with_messages(
        [ChatMessage(role="user", content="build it")],
        builder_graph_id="graph-xyz",
    )
    for tool in [
        "create_agent",
        "edit_agent",
        "validate_agent_graph",
        "fix_agent_graph",
    ]:
        assert require_guide_read(session, tool) is None


def test_read_skill_default_guide_in_messages_satisfies_gate():
    """The skill registry seeds ``agent_building_guide`` as a default
    skill — calling ``read_skill(name="agent_building_guide")`` must
    satisfy the gate the same way ``get_agent_building_guide`` does so
    the legacy and new paths converge on one mechanism."""
    session = _session_with_messages(
        [
            ChatMessage(role="user", content="build it"),
            ChatMessage(
                role="assistant",
                content="loading guide",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_skill",
                            "arguments": '{"name": "agent_building_guide"}',
                        },
                    }
                ],
            ),
        ]
    )
    assert require_guide_read(session, "create_agent") is None


def test_read_skill_wrong_skill_does_not_satisfy_gate():
    """A ``read_skill`` call for a *different* skill must NOT satisfy
    the agent-building gate — argument discrimination is the whole
    point of the helper."""
    session = _session_with_messages(
        [
            ChatMessage(role="user", content="build it"),
            ChatMessage(
                role="assistant",
                content="loading something else",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_skill",
                            "arguments": '{"name": "mcp_tool_guide"}',
                        },
                    }
                ],
            ),
        ]
    )
    assert isinstance(require_guide_read(session, "create_agent"), ErrorResponse)


def test_inflight_read_skill_with_args_satisfies_gate():
    """Same-turn safety for the skill-registry path: a ``read_skill``
    dispatched earlier in the current turn (in-flight, args captured)
    must satisfy the gate before the call lands in ``session.messages``.
    Mirrors the existing ``get_agent_building_guide`` in-flight test."""
    session = _session_with_messages([ChatMessage(role="user", content="build it")])
    session.announce_inflight_tool_call("read_skill", {"name": "agent_building_guide"})
    assert require_guide_read(session, "create_agent") is None


def test_inflight_read_skill_with_wrong_args_does_not_satisfy_gate():
    """In-flight ``read_skill`` for a different skill must NOT satisfy
    the gate — argument discrimination applies to in-flight calls too."""
    session = _session_with_messages([ChatMessage(role="user", content="build it")])
    session.announce_inflight_tool_call("read_skill", {"name": "mcp_tool_guide"})
    assert isinstance(require_guide_read(session, "create_agent"), ErrorResponse)


def test_read_skill_malformed_json_args_does_not_crash():
    """A malformed ``arguments`` string from a hypothetical historical
    row must not crash the gate scan — fall through to the next
    candidate instead."""
    session = _session_with_messages(
        [
            ChatMessage(role="user", content="build it"),
            ChatMessage(
                role="assistant",
                content="oops",
                tool_calls=[
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "read_skill",
                            "arguments": "{not-valid-json",
                        },
                    }
                ],
            ),
        ]
    )
    # No valid read_skill row → gate must still fire.
    assert isinstance(require_guide_read(session, "create_agent"), ErrorResponse)


def test_announce_inflight_drops_non_dict_arguments():
    """JSON-shaped tool arguments can in principle be a list or scalar
    (``orjson.loads("[1,2]")`` → ``[1, 2]``).  Argument-discriminating
    guards do ``.get("name")`` on the captured args, which would crash
    on a list.  Drop non-dict shapes silently so guards stay safe."""
    session = _session_with_messages([])
    session.announce_inflight_tool_call("read_skill", [1, 2])
    session.announce_inflight_tool_call("read_skill", "not-a-dict")
    session.announce_inflight_tool_call("read_skill", 42)
    # None of those landed in the args buffer, so a gate looking for
    # read_skill(name="agent_building_guide") still sees nothing.
    assert session.get_inflight_tool_call_args("read_skill") == []
    # The name buffer still got hit (tools still ran), so a name-only
    # gate (``has_tool_been_called``) reports True.
    assert session.has_tool_been_called("read_skill") is True


def test_announce_inflight_captures_dict_arguments():
    """The intentional path: a proper dict arg is captured so
    argument-discriminating guards can find it."""
    session = _session_with_messages([])
    session.announce_inflight_tool_call("read_skill", {"name": "mcp_tool_guide"})
    args = session.get_inflight_tool_call_args("read_skill")
    assert args == [{"name": "mcp_tool_guide"}]


def test_guide_in_system_prompt_flag_gate_passes():
    """When this turn's system prompt carries the guide (building session),
    the gate passes without any guide tool call in history."""
    session = _session_with_messages([])
    session.guide_in_system_prompt = True
    assert require_guide_read(session, "create_agent") is None


def _enter_call_message() -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content="",
        tool_calls=[{"function": {"name": "enter_agent_building_mode"}}],
    )


def test_pending_switch_tells_model_to_end_turn(mocker):
    """Baseline turn on an SDK-capable deployment: the enter call registers
    an engine switch; further build tools must wait for the continuation."""
    mocker.patch(
        "backend.copilot.tools.helpers.chat_config",
        mocker.MagicMock(transport=mocker.MagicMock(supports_sdk=True)),
    )
    session = _session_with_messages([_enter_call_message()])
    result = require_guide_read(session, "create_agent")
    assert isinstance(result, ErrorResponse)
    assert "engine switch is pending" in result.message


def test_enter_call_satisfies_gate_on_sdk_less_deployment(mocker):
    """Without SDK support the enter tool served the guide inline — the
    gate must pass instead of stranding the model."""
    mocker.patch(
        "backend.copilot.tools.helpers.chat_config",
        mocker.MagicMock(transport=mocker.MagicMock(supports_sdk=False)),
    )
    session = _session_with_messages([_enter_call_message()])
    assert require_guide_read(session, "create_agent") is None


# --------------------------------------- the run_capability dispatcher shape
#
# Deferred tools (#14569) reach the model only as
# ``run_capability(id="tool:<name>", input={...})``, so that is the row the
# gate has to read out of history.  Every gated tool shares one matcher, so
# each case is parametrized over all four.

GATED_TOOLS = ["create_agent", "edit_agent", "fix_agent_graph", "validate_agent_graph"]


def _run_capability_message(capability_id: str, payload: dict | None = None):
    return ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_rc",
                "type": "function",
                "function": {
                    "name": "run_capability",
                    "arguments": json.dumps(
                        {"id": capability_id, "input": payload or {}}
                    ),
                },
            }
        ],
    )


@pytest.mark.parametrize("tool_name", GATED_TOOLS)
@pytest.mark.parametrize(
    "capability_id", ["tool:read_skill", "read_skill"]  # resolve_entry takes either
)
def test_read_skill_via_run_capability_satisfies_gate(tool_name, capability_id):
    """The regression #14569 shipped: the guide was loaded through the
    dispatcher, so history holds ``run_capability`` and the gate refused
    every build tool forever."""
    session = _session_with_messages(
        [
            ChatMessage(role="user", content="build it"),
            _run_capability_message(capability_id, {"name": "agent_building_guide"}),
        ]
    )
    assert require_guide_read(session, tool_name) is None


@pytest.mark.parametrize(
    "capability_id", ["tool:enter_agent_building_mode", "enter_agent_building_mode"]
)
def test_enter_via_run_capability_registers_the_engine_switch(mocker, capability_id):
    """The enter tool through the dispatcher must reach the engine-switch
    branch, not the refusal — the refusal is what looped."""
    mocker.patch(
        "backend.copilot.tools.helpers.chat_config",
        mocker.MagicMock(transport=mocker.MagicMock(supports_sdk=True)),
    )
    session = _session_with_messages([_run_capability_message(capability_id)])
    result = require_guide_read(session, "create_agent")
    assert isinstance(result, ErrorResponse)
    assert "engine switch is pending" in result.message


def test_enter_via_run_capability_satisfies_gate_on_sdk_less_deployment(mocker):
    mocker.patch(
        "backend.copilot.tools.helpers.chat_config",
        mocker.MagicMock(transport=mocker.MagicMock(supports_sdk=False)),
    )
    session = _session_with_messages(
        [_run_capability_message("tool:enter_agent_building_mode")]
    )
    assert require_guide_read(session, "create_agent") is None


@pytest.mark.parametrize(
    "capability_id,payload",
    [
        ("tool:read_skill", {"name": "mcp_tool_guide"}),  # wrong skill
        ("block:enter_agent_building_mode", {}),  # a block id, never a tool
        ("https://enter_agent_building_mode", {}),  # an MCP server URL
        ("tool:", {}),  # empty inner name
        ("", {}),  # missing id
        ("tool:find_capability", {}),  # an unrelated tool
    ],
)
def test_run_capability_rows_that_must_not_open_the_gate(capability_id, payload):
    session = _session_with_messages(
        [
            ChatMessage(role="user", content="build it"),
            _run_capability_message(capability_id, payload),
        ]
    )
    assert isinstance(require_guide_read(session, "create_agent"), ErrorResponse)


def test_run_capability_malformed_arguments_do_not_crash_the_gate():
    """A history row the gate cannot parse must fall through, not raise."""
    session = _session_with_messages(
        [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {"function": {"name": "run_capability", "arguments": "{not-json"}},
                    {"function": {"name": "run_capability", "arguments": None}},
                    {"function": None, "name": "run_capability", "arguments": "[1,2]"},
                    {"function": {"name": "run_capability", "arguments": '{"id": 42}'}},
                ],
            )
        ]
    )
    assert isinstance(require_guide_read(session, "create_agent"), ErrorResponse)
