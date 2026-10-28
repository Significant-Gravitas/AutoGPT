"""Expert tools remain authenticated after retiring the rollout gate."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.budget_gate import AutoContinueGate
from backend.copilot.model import ChatMessage, ChatSession, ChatSessionMetadata
from backend.copilot.response_model import StreamStart
from backend.copilot.turn_budget import TurnBudget

_SVC = "backend.copilot.sdk.service"


def _make_lock_mock():
    captured_owner: dict[str, str] = {}

    def _lock_factory(*args, **kwargs):
        captured_owner["id"] = kwargs.get("owner_id", "")
        mock_lock = MagicMock()
        mock_lock.try_acquire = AsyncMock(side_effect=lambda: captured_owner["id"])
        mock_lock.refresh = AsyncMock()
        mock_lock.release = AsyncMock()
        return mock_lock

    return _lock_factory


def _make_session() -> ChatSession:
    return ChatSession(
        session_id="test-session-id",
        user_id="test-user",
        title="Already titled",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[ChatMessage(role="user", content="hello")],
        # Stated, not defaulted: the staffing tools this file asserts on are
        # also hidden off an interactive origin, so a session that left it
        # unset would hide staffing tools for the wrong reason.
        metadata=ChatSessionMetadata(origin="interactive"),
    )


class _ToolRegistrationReached(Exception):
    pass


def _make_patches():
    """Capture registration and stop before any provider is constructed."""
    mcp_server_mock = MagicMock(side_effect=_ToolRegistrationReached)

    patches = [
        (f"{_SVC}.get_chat_session", dict(new_callable=AsyncMock)),
        (
            f"{_SVC}.upsert_chat_session",
            dict(new_callable=AsyncMock, side_effect=lambda s: s),
        ),
        (f"{_SVC}.build_skills_context", dict(new_callable=AsyncMock, return_value="")),
        (f"{_SVC}.get_redis_async", dict(new_callable=AsyncMock)),
        (f"{_SVC}.AsyncClusterLock", dict(side_effect=_make_lock_mock())),
        (f"{_SVC}._make_sdk_cwd", dict(return_value="/tmp/test-sdk-cwd")),
        ("os.makedirs", {}),
        (
            f"{_SVC}.propagate_attributes",
            dict(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())),
        ),
        (
            f"{_SVC}._build_system_prompt",
            dict(new_callable=AsyncMock, return_value=("system prompt", None)),
        ),
        (
            f"{_SVC}.ClaudeSDKClient",
            dict(
                side_effect=AssertionError("Provider construction is outside this test")
            ),
        ),
        (
            f"{_SVC}.require_verbatim_prompts_cli",
            dict(return_value="/opt/claude/claude"),
        ),
        (f"{_SVC}.create_copilot_mcp_server", dict(new=mcp_server_mock)),
        (f"{_SVC}.create_security_hooks", dict(return_value=MagicMock())),
        (f"{_SVC}.get_copilot_tool_names", dict(return_value=[])),
        (f"{_SVC}.get_sdk_disallowed_tools", dict(return_value=[])),
        (f"{_SVC}.build_sdk_env", dict(return_value={})),
        (f"{_SVC}._resolve_sdk_model", dict(return_value=None)),
        (f"{_SVC}.set_execution_context", {}),
        (
            f"{_SVC}._fetch_graphiti_context",
            {"new": AsyncMock(side_effect=lambda user_id, *_: (bool(user_id), ""))},
        ),
        (
            f"{_SVC}.inject_user_context",
            {"new_callable": AsyncMock, "return_value": "hello"},
        ),
        (f"{_SVC}._enqueue_graphiti_turn", {"new_callable": AsyncMock}),
        (f"{_SVC}.capture_chat_turn", {"new_callable": AsyncMock}),
        (f"{_SVC}.persist_and_record_usage", {"new_callable": AsyncMock}),
        (
            f"{_SVC}.config",
            dict(
                api_key="test-key",
                use_claude_code_subscription=False,
                claude_agent_use_resume=True,
                claude_agent_max_buffer_size=100_000,
                claude_agent_max_subtasks=5,
                stream_lock_ttl=60,
                active_e2b_api_key=None,
                use_e2b_sandbox=False,
                claude_agent_max_transient_retries=1,
                agent_max_turns=1000,
                sdk_step_checkin=1000,
                claude_agent_max_budget_usd=100.0,
                claude_agent_max_thinking_tokens=0,
                claude_agent_thinking_effort=None,
                claude_agent_fallback_model=None,
                claude_agent_model="claude-sonnet-4-6",
                thinking_standard_model="anthropic/claude-sonnet-4-6",
            ),
        ),
        (f"{_SVC}.get_user_tier", dict(new_callable=AsyncMock, return_value=None)),
        (
            f"{_SVC}.resolve_turn_budget",
            dict(
                new_callable=AsyncMock,
                return_value=TurnBudget(usd=100.0, limiting="task"),
            ),
        ),
        (
            f"{_SVC}.gate_auto_continue",
            dict(new_callable=AsyncMock, return_value=AutoContinueGate(proceed=True)),
        ),
        (f"{_SVC}.drain_pending_safe", dict(new_callable=AsyncMock, return_value=[])),
    ]
    return patches, mcp_server_mock


async def _run_sdk_turn(
    *,
    user_id: str | None,
    expert_id: str | None = None,
    extra_patches: list[tuple[str, dict]] | None = None,
):
    from backend.copilot.sdk.service import stream_chat_completion_sdk

    session = _make_session()
    session.expert_id = expert_id
    patches, mcp_server_mock = _make_patches()
    patches += extra_patches or []

    events = []
    with contextlib.ExitStack() as stack, pytest.raises(_ToolRegistrationReached):
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        async with asyncio.timeout(10):
            async for event in stream_chat_completion_sdk(
                session_id=session.session_id,
                message="hello",
                is_user_message=True,
                user_id=user_id,
                session=session,
            ):
                events.append(event)

    assert any(
        isinstance(e, StreamStart) for e in events
    ), f"Turn did not complete far enough to reach tool registration: {events}"
    return mcp_server_mock


class TestSdkExpertsAuthorization:
    @pytest.mark.asyncio
    async def test_anonymous_turn_hides_expert_tools(self) -> None:
        mcp_server_mock = await _run_sdk_turn(user_id=None)

        hidden = mcp_server_mock.call_args.kwargs["hidden_tool_names"]
        # Anonymous turns cannot use expert or memory tools.
        assert "memory_store" in hidden
        assert "hire_expert" in hidden
        assert "update_expert_soul" in hidden
        assert "delegate_to_expert" in hidden

    @pytest.mark.asyncio
    async def test_authenticated_plain_turn_keeps_staffing_tools(
        self,
    ) -> None:
        mcp_server_mock = await _run_sdk_turn(user_id="test-user")

        hidden = mcp_server_mock.call_args.kwargs["hidden_tool_names"]
        # Plain Otto session (no session.expert_id): loses the
        # expert-session tools, keeps the staffing ("expert_admin") tools.
        assert "update_expert_soul" in hidden
        assert "hire_expert" not in hidden
        assert "delegate_to_expert" not in hidden


class TestSdkComputerNoteWiring:
    """The plain chat's computer note rides ``get_sdk_supplement``; an expert
    session must ask for the variant without it, because its own
    ``<expert_computer>`` block arrives in the first user message."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("expert_id", [None, "expert-1"])
    async def test_main_turn_tells_the_supplement_whether_it_is_an_expert_session(
        self, expert_id: str | None
    ) -> None:
        supplement = MagicMock(return_value="")
        await _run_sdk_turn(
            user_id=None,
            expert_id=expert_id,
            extra_patches=[
                (f"{_SVC}.get_sdk_supplement", dict(new=supplement)),
                (
                    f"{_SVC}.build_expert_identity_suffix",
                    dict(new_callable=AsyncMock, return_value=""),
                ),
            ],
        )

        supplement.assert_called_once_with(
            use_e2b=False, expert_session=bool(expert_id)
        )
