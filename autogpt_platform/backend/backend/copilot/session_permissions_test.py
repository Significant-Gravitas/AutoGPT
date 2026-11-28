"""What a session's own kind or binding lets it call, and that it holds."""

import pytest

from backend.copilot.context import set_execution_context
from backend.copilot.model import ONBOARDING_INTERVIEW_KIND, ChatSession
from backend.copilot.permissions import (
    CopilotPermissions,
    all_known_tool_names,
    apply_tool_permissions,
    denied_tool_names,
)
from backend.copilot.session_permissions import (
    BUILDER_BLOCKED_TOOLS,
    ONBOARDING_INTERVIEW_ALLOWED_TOOLS,
    resolve_session_permissions,
)


def _interview() -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    session.metadata.kind = ONBOARDING_INTERVIEW_KIND
    return session


def _interview_permissions() -> CopilotPermissions:
    permissions = resolve_session_permissions(_interview())
    assert permissions is not None
    return permissions


def test_an_ordinary_chat_is_unrestricted() -> None:
    assert resolve_session_permissions(None) is None
    assert resolve_session_permissions(ChatSession.new("u1", dry_run=False)) is None


def test_a_builder_session_keeps_its_blacklist() -> None:
    permissions = resolve_session_permissions(
        ChatSession.new("u1", dry_run=False, builder_graph_id="g1")
    )

    assert permissions is not None
    assert permissions.tools_exclude is True
    assert sorted(permissions.tools) == sorted(BUILDER_BLOCKED_TOOLS)


def test_an_interview_may_call_only_ask_question_and_its_end_signal() -> None:
    permissions = _interview_permissions()

    assert permissions.tools_exclude is False
    assert permissions.tools == ["ask_question", "end_interview"]
    assert permissions.effective_allowed_tools(all_known_tool_names()) == {
        "ask_question",
        "end_interview",
    }


def test_the_interview_kind_wins_over_a_builder_binding() -> None:
    """The kind is the narrower rule, so a session carrying both is an
    interview: a blacklist would hand it every tool the builder keeps."""
    session = ChatSession.new("u1", dry_run=False, builder_graph_id="g1")
    session.metadata.kind = ONBOARDING_INTERVIEW_KIND

    assert resolve_session_permissions(session) == _interview_permissions()


def test_the_allowlist_drags_in_no_dispatcher() -> None:
    """A deferred tool in a whitelist implies ``run_capability``, the way into
    every deferred tool and block. Every allowed tool must be in the eager
    core, so the interview's declared tool list stays exactly the list."""
    from backend.copilot.tools import DEFERRED_TOOL_NAMES, TOOL_REGISTRY

    for name in ONBOARDING_INTERVIEW_ALLOWED_TOOLS:
        assert name in TOOL_REGISTRY
        assert name not in DEFERRED_TOOL_NAMES
    assert "run_capability" in denied_tool_names(_interview_permissions())


def test_the_baseline_engine_declares_only_the_allowed_tools() -> None:
    from backend.copilot.baseline.service import _filter_tools_by_permissions
    from backend.copilot.tools import get_available_tools

    tools = _filter_tools_by_permissions(
        get_available_tools(), _interview_permissions()
    )

    assert [tool["function"]["name"] for tool in tools] == [
        "ask_question",
        "end_interview",
    ]


def test_the_sdk_engine_registers_only_the_allowed_tools() -> None:
    from backend.copilot.sdk.tool_adapter import MCP_TOOL_PREFIX

    allowed, disallowed = apply_tool_permissions(_interview_permissions())

    # ``read_tool_result`` is the CLI's reader for its own truncated outputs,
    # kept on every filtered turn.
    assert set(allowed) == {
        f"{MCP_TOOL_PREFIX}ask_question",
        f"{MCP_TOOL_PREFIX}end_interview",
        f"{MCP_TOOL_PREFIX}read_tool_result",
    }
    # The CLI's own built-ins go too: no sub-agents, files or shell.
    for builtin in ("Task", "Agent", "TodoWrite", "Bash", "Read", "Write"):
        assert builtin not in allowed
    assert "Task" in disallowed
    # And the MCP server never registers the rest of the registry.
    assert denied_tool_names(_interview_permissions()) == (
        all_known_tool_names() - {"ask_question", "end_interview"}
    )


@pytest.mark.parametrize("kind", ["normal", "dream", None])
def test_every_other_kind_of_session_hides_the_end_signal(kind: str | None) -> None:
    """The end signal closes the interview it is called in, so it is offered
    in an onboarding interview and hidden from every other chat."""
    from backend.copilot.tools import (
        get_available_tools,
        interview_disabled_tools,
        reachable_tool_names,
    )

    hidden = interview_disabled_tools(kind)

    assert hidden == {"end_interview"}
    declared = {
        t["function"]["name"] for t in get_available_tools(disabled_tools=hidden)
    }
    assert "end_interview" not in declared
    assert "end_interview" not in reachable_tool_names(disabled_tools=hidden)
    assert interview_disabled_tools(ONBOARDING_INTERVIEW_KIND) == frozenset()


@pytest.mark.asyncio
async def test_an_ordinary_chat_that_names_the_end_signal_is_refused() -> None:
    from backend.copilot.tools import execute_tool, interview_disabled_tools

    session = ChatSession.new("user-1", dry_run=False)
    hidden = interview_disabled_tools(session.metadata.kind)
    set_execution_context("user-1", session, hidden_tools=hidden)
    try:
        result = await execute_tool(
            "end_interview",
            {"first_task": "Chase unpaid invoices"},
            "user-1",
            session,
            "call-1",
            disabled_groups=[],
            disabled_tools=hidden,
        )
    finally:
        set_execution_context(None, None)

    assert result.success is False
    assert "tool_disabled" in str(result.output)
    assert session.metadata.onboarding_interview_stage is None


@pytest.mark.asyncio
async def test_a_tool_outside_the_allowlist_is_refused_even_when_named() -> None:
    """Hiding a tool is presentation. A model that names one anyway, from a
    replayed transcript or an injected instruction, is refused at the tool."""
    from backend.copilot.tools import execute_tool

    session = _interview()
    permissions = _interview_permissions()
    set_execution_context(
        "user-1",
        session,
        permissions=permissions,
        hidden_tools=denied_tool_names(permissions),
    )
    try:
        result = await execute_tool(
            "web_search",
            {"query": "user's home address"},
            "user-1",
            session,
            "call-1",
            disabled_groups=[],
            disabled_tools=frozenset(),
        )
    finally:
        set_execution_context(None, None)

    assert result.success is False
    assert "tool_disabled" in str(result.output)


@pytest.mark.parametrize(
    ("name", "denied"),
    [
        ("ask_question", False),
        ("add_understanding", True),
        ("memory_store", True),
        ("run_capability", True),
        ("bash_exec", True),
        ("hire_expert", True),
    ],
)
def test_the_capability_gate_follows_the_allowlist(name: str, denied: bool) -> None:
    """``run_capability`` resolves a deferred tool and runs it through this
    gate, so a dispatch to a deferred writer is refused here too."""
    from backend.copilot.tools.capability_gates import gate_denied

    session = _interview()
    set_execution_context("user-1", session, permissions=_interview_permissions())
    try:
        assert gate_denied(name) is denied
    finally:
        set_execution_context(None, None)
