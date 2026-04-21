"""Tests for the ``require_guide_read`` gate on agent-generation tools.

The agent-building guide carries block ids, link semantics, and
AgentExecutorBlock / MCPToolBlock conventions that the agent needs before
producing agent JSON. Without the gate, agents often skip the guide to save
tokens and then produce JSON that fails validation — wasting turns on
auto-fix loops.
"""

from unittest.mock import MagicMock

import pytest

from backend.copilot.model import ChatMessage, ChatSession

from .helpers import require_guide_read
from .models import ErrorResponse


def _session_with_messages(
    messages: list[ChatMessage],
    builder_graph_id: str | None = None,
) -> ChatSession:
    """Build a minimal ChatSession whose ``messages`` matches *messages*."""
    session = MagicMock(spec=ChatSession)
    session.session_id = "test-session"
    session.messages = messages
    session.metadata = MagicMock()
    session.metadata.builder_graph_id = builder_graph_id
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
