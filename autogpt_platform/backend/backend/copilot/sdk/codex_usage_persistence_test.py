"""Codex interruption reaches the real session and cost-log accounting path."""

import asyncio
import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from pydantic import SecretStr

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamTextDelta
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.token_tracking import persist_and_record_usage
from backend.data.model import OAuth2Credentials
from backend.integrations.codex.transport import CodexCredentialLease

from .codex_usage_test import PAYLOAD, run_conversation, usage_gateway
from .retry_scenarios_test import _make_sdk_patches

_SVC = "backend.copilot.sdk.service"
_TRACKING = "backend.copilot.token_tracking"


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["failure", "cancel", "generator_close", "success"])
async def test_sdk_finally_records_subscription_usage_once(ending):
    session, lease = _session_and_lease()
    waiting = asyncio.Event()
    async with usage_gateway("success" if ending == "success" else "failed") as gateway:
        client = _client(gateway, ending, waiting, session.session_id)
        with contextlib.ExitStack() as stack:
            _stub_sdk(stack, session, gateway, client)
            persist, save, cost_log, credits, budget = _record_accounting(stack)
            stream = stream_chat_completion_sdk(
                session_id=session.session_id,
                message="hello",
                is_user_message=True,
                user_id=session.user_id,
                session=session,
                credential_lease=lease,
            )
            await _consume_turn(stream, ending, waiting)

    persist.assert_awaited_once()
    save.assert_awaited_with(session)
    cost_log.assert_called_once()
    credits.assert_not_awaited()
    budget.assert_not_awaited()
    _assert_usage(
        session, cost_log.call_args.args[0], lease, 2 if ending == "success" else 1
    )


def _session_and_lease():
    session = ChatSession(
        session_id="codex-usage-session",
        user_id="test-user",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[ChatMessage(role="user", content="hello")],
        title="Usage regression",
    )
    lease = CodexCredentialLease(
        OAuth2Credentials(
            id="codex-usage-credential",
            provider="codex",
            title="ChatGPT",
            access_token=SecretStr("test"),
            scopes=[],
        ),
        MagicMock(),
    )
    return session, lease


def _client(gateway, ending, waiting, session_id):
    async def receive():
        if ending in ("cancel", "generator_close"):
            conversation = gateway._start_conversation(PAYLOAD)
            await asyncio.wait_for(conversation.queue.get(), timeout=2)
            waiting.set()
            if ending == "cancel":
                await asyncio.Event().wait()
        else:
            await run_conversation(gateway)
            if ending == "failure":
                raise ValueError("scripted transport failure")
        yield AssistantMessage(content=[TextBlock(text="partial")], model=gateway.model)
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=session_id,
            usage={"input_tokens": 999, "output_tokens": 999},
            total_cost_usd=99,
            result="done",
        )

    client = MagicMock()
    client.query = AsyncMock()
    client.receive_response = receive
    context = AsyncMock()
    context.__aenter__.return_value = client
    context.__aexit__.return_value = None
    return context


def _stub_sdk(stack, session, gateway, client):
    for target, kwargs in _make_sdk_patches(
        session,
        original_transcript="",
        compacted_transcript=None,
        client_side_effect=lambda **_: client,
    ):
        stack.enter_context(patch(target, **kwargs))
    stack.enter_context(patch(f"{_SVC}.resolve_answered", new=_no_answered_cards))
    for name, value in (
        ("inject_user_context", None),
        ("build_session_context", ""),
        ("build_skills_update_notice", None),
        ("capture_chat_turn", None),
        ("resolve_codex_model_route", (gateway.model, None, "env")),
    ):
        stack.enter_context(
            patch(f"{_SVC}.{name}", new_callable=AsyncMock, return_value=value)
        )
    stack.enter_context(patch(f"{_SVC}.gateway_for_lease", return_value=gateway))


async def _no_answered_cards(*_, **__):
    return
    yield


def _record_accounting(stack):
    persist = stack.enter_context(
        patch(f"{_SVC}.persist_and_record_usage", wraps=persist_and_record_usage)
    )
    save = stack.enter_context(
        patch(f"{_SVC}.upsert_chat_session", new_callable=AsyncMock)
    )
    cost_log = stack.enter_context(patch(f"{_TRACKING}._schedule_cost_log"))
    credits = stack.enter_context(
        patch(f"{_TRACKING}.record_credit_billed_cost", new_callable=AsyncMock)
    )
    budget = stack.enter_context(
        patch(f"{_TRACKING}.record_cost_usage", new_callable=AsyncMock)
    )
    return persist, save, cost_log, credits, budget


async def _consume_turn(stream, ending, waiting):
    async def consume():
        async for event in stream:
            if ending == "generator_close" and isinstance(event, StreamTextDelta):
                await stream.aclose()
                break

    task = asyncio.create_task(consume())
    try:
        if ending == "cancel":
            await asyncio.wait_for(waiting.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await asyncio.wait_for(task, timeout=5)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await stream.aclose()


def _assert_usage(session, entry, lease, calls):
    assert len(session.usage) == 1
    usage = session.usage[0]
    assert usage.prompt_tokens == 60 * calls
    assert usage.completion_tokens == 20 * calls
    assert usage.cache_read_tokens == 40 * calls
    assert usage.total_tokens == 80 * calls
    assert usage.cache_creation_tokens == 0
    assert entry.input_tokens == 60 * calls
    assert entry.output_tokens == 20 * calls
    assert entry.cache_read_tokens == 40 * calls
    assert entry.provider == "codex"
    assert entry.credential_id == lease.credentials.id
    assert entry.cost_microdollars is None
    assert entry.metadata["billing_mode"] == "user_subscription"
    assert entry.metadata["execution_path"] == "codex_claude_sdk"
