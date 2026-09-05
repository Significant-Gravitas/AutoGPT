"""SDK-engine coverage for the hire-experts flag guard and tool hiding.

``stream_chat_completion_sdk`` resolves ``experts_enabled = bool(user_id)
and await is_feature_enabled(Flag.HIRE_EXPERTS, user_id, default=False)``
and feeds the result into ``expert_tool_disabled_groups`` (tested directly
in ``handoff_to_expert_test.py``) before registering tools with the MCP
server. This file proves the SDK call SITE wires that correctly:

  * an anonymous turn (``user_id=None``) must fail closed WITHOUT ever
    resolving the flag — deleting the ``bool(user_id) and`` guard would
    still (accidentally) work most of the time, but would await
    ``is_feature_enabled`` with a ``None`` user_id.
  * the resulting disabled groups actually reach ``create_copilot_mcp_server``
    as hidden tool names — the enforcement point for the SDK engine (which
    never calls ``execute_tool`` per-call; hidden tools are simply never
    registered with the MCP server).

Reuses the same mocking shape as ``retry_scenarios_test.py``'s
``TestStreamChatCompletionRetryIntegration`` (independently duplicated here
rather than imported, since these are private test helpers local to that
module) — a single-message session takes the "no --resume" short-circuit in
``_restore_cli_session_for_turn`` for free, so ``download_transcript`` /
``strip_for_upload`` need no mocks.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ResultMessage

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamStart

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
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[ChatMessage(role="user", content="hello")],
    )


def _make_client_mock():
    """A ClaudeSDKClient async-context-manager mock that streams one
    successful ``ResultMessage`` and nothing else."""
    result_msg = ResultMessage(
        subtype="success",
        result="done",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id="test-session-id",
    )

    async def _receive():
        yield result_msg

    client = MagicMock()
    client.receive_response = _receive
    client.query = AsyncMock()
    client._transport = MagicMock()
    client._transport.write = AsyncMock()

    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    return cm


def _make_patches(*, hire_experts_enabled: bool):
    """Patch list for a full (mocked) SDK turn, plus a fresh mock for
    ``create_copilot_mcp_server`` and ``is_feature_enabled`` the caller can
    read after the turn."""
    mcp_server_mock = MagicMock(return_value=MagicMock())
    is_feature_enabled_mock = AsyncMock(return_value=hire_experts_enabled)

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
            dict(side_effect=lambda *a, **kw: _make_client_mock()),
        ),
        (f"{_SVC}.create_copilot_mcp_server", dict(new=mcp_server_mock)),
        (f"{_SVC}.create_security_hooks", dict(return_value=MagicMock())),
        (f"{_SVC}.get_copilot_tool_names", dict(return_value=[])),
        (f"{_SVC}.get_sdk_disallowed_tools", dict(return_value=[])),
        (f"{_SVC}.build_sdk_env", dict(return_value={})),
        (f"{_SVC}._resolve_sdk_model", dict(return_value=None)),
        (f"{_SVC}.set_execution_context", {}),
        (f"{_SVC}.is_feature_enabled", dict(new=is_feature_enabled_mock)),
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
            f"{_SVC}._resolve_dynamic_max_budget_usd",
            dict(new_callable=AsyncMock, return_value=100.0),
        ),
        (f"{_SVC}.drain_pending_safe", dict(new_callable=AsyncMock, return_value=[])),
    ]
    return patches, mcp_server_mock, is_feature_enabled_mock


async def _run_sdk_turn(*, user_id: str | None, hire_experts_enabled: bool):
    from backend.copilot.sdk.service import stream_chat_completion_sdk

    session = _make_session()
    patches, mcp_server_mock, is_feature_enabled_mock = _make_patches(
        hire_experts_enabled=hire_experts_enabled
    )

    events = []
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
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
    return mcp_server_mock, is_feature_enabled_mock


class TestSdkExpertsFlagGuard:
    @pytest.mark.asyncio
    async def test_anonymous_turn_never_calls_the_hire_experts_flag(self) -> None:
        mcp_server_mock, is_feature_enabled_mock = await _run_sdk_turn(
            user_id=None, hire_experts_enabled=True
        )

        is_feature_enabled_mock.assert_not_awaited()
        hidden = mcp_server_mock.call_args.kwargs["hidden_tool_names"]
        # Flag off (fails closed for anonymous turns): every expert-team
        # group is hidden — staffing, expert-session, and delegation tools.
        assert "hire_expert" in hidden
        assert "update_expert_soul" in hidden
        assert "delegate_to_expert" in hidden

    @pytest.mark.asyncio
    async def test_authenticated_turn_with_flag_on_only_hides_staffing_tools(
        self,
    ) -> None:
        mcp_server_mock, is_feature_enabled_mock = await _run_sdk_turn(
            user_id="test-user", hire_experts_enabled=True
        )

        is_feature_enabled_mock.assert_awaited_once()
        hidden = mcp_server_mock.call_args.kwargs["hidden_tool_names"]
        # Plain Autopilot session (no session.expert_id): loses the
        # expert-session tools, keeps the staffing ("expert_admin") tools.
        assert "update_expert_soul" in hidden
        assert "hire_expert" not in hidden
        assert "delegate_to_expert" not in hidden
