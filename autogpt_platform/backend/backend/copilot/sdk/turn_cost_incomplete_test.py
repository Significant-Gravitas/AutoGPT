"""An incomplete SDK response must account for spend saved in its CLI session."""

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamError, StreamTextDelta
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.stream_heartbeat import wrap_stream_with_heartbeat
from backend.copilot.transcript import cli_session_path

from .conftest import build_test_transcript
from .retry_scenarios_test import _make_sdk_patches
from .turn_cost_test import (
    _CALL_COST,
    _CALL_USAGE,
    _KIMI,
    _SDK_CWD,
    _SESSION_ID,
    _SVC,
    _kimi_rate_card_cost,
)


def _native_cost_row(model: str, calls: int) -> str:
    return (
        json.dumps(
            {
                "type": "cost-state",
                "sessionId": _SESSION_ID,
                "totalCostUSD": calls * _CALL_COST,
                "modelUsage": {
                    model: {
                        "inputTokens": calls * _CALL_USAGE["input_tokens"],
                        "outputTokens": calls * _CALL_USAGE["output_tokens"],
                        "cacheReadInputTokens": calls
                        * _CALL_USAGE["cache_read_input_tokens"],
                        "cacheCreationInputTokens": calls
                        * _CALL_USAGE["cache_creation_input_tokens"],
                        "costUSD": calls * _CALL_COST,
                    }
                },
            }
        )
        + "\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["claude-sonnet-4-6", _KIMI])
async def test_cli_exit_without_result_records_its_spend(model, tmp_path, monkeypatch):
    session_file = _resumed_cli_session(tmp_path, monkeypatch)

    def client_factory(*args, **kwargs):
        async def receive():
            with session_file.open("a") as handle:
                handle.write(_native_cost_row(model, 2))
            yield AssistantMessage(content=[TextBlock(text="partial")], model=model)

        client = MagicMock()
        client.query = AsyncMock()
        client.receive_response = receive
        context = AsyncMock()
        context.__aenter__.return_value = client
        context.__aexit__.return_value = None
        return context

    with _patched_turn(model, client_factory) as (persist, run_turn):
        async for _ in run_turn():
            pass

    persist.assert_awaited_once()
    expected = _kimi_rate_card_cost() if model == _KIMI else _CALL_COST
    assert persist.await_args.kwargs["cost_usd"] == pytest.approx(expected)
    assert persist.await_args.kwargs["prompt_tokens"] == _CALL_USAGE["input_tokens"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["stop", "error", "close"])
async def test_a_turn_its_reader_ends_records_what_the_cli_spent(
    ending, tmp_path, monkeypatch
):
    """The executor ends a turn early two ways: ``stop`` cancels its task
    (POST /sessions/{id}/cancel), ``error`` stops reading at the first
    StreamError and closes the stream. ``close`` is a reader that closes the
    turn's own stream mid-reply."""
    model = "claude-sonnet-4-6"
    session_file = _resumed_cli_session(tmp_path, monkeypatch)

    def client_factory(*args, **kwargs):
        async def receive():
            yield AssistantMessage(content=[TextBlock(text="partial")], model=model)
            if ending == "error":
                yield ResultMessage(
                    subtype="error_during_execution",
                    duration_ms=100,
                    duration_api_ms=100,
                    is_error=True,
                    num_turns=1,
                    session_id=_SESSION_ID,
                    total_cost_usd=2 * _CALL_COST,
                    usage=dict(_CALL_USAGE),
                    result="Tool schema rejected",
                )
                return
            await asyncio.Event().wait()  # the next model call is still out

        async def close(*_):
            # The CLI saves its running total as it exits.
            with session_file.open("a") as handle:
                handle.write(_native_cost_row(model, 2))
            if ending == "error":
                await asyncio.Event().wait()  # still exiting when the reader stops

        client = MagicMock()
        client.query = AsyncMock()
        client.receive_response = receive
        context = AsyncMock()
        context.__aenter__.return_value = client
        context.__aexit__.side_effect = close
        return context

    with _patched_turn(model, client_factory) as (persist, run_turn):
        if ending == "close":
            stream = run_turn()
            async for event in stream:
                if isinstance(event, StreamTextDelta):
                    break
            await stream.aclose()
        else:
            answered = asyncio.Event()
            turn = asyncio.create_task(_read_like_the_executor(run_turn(), answered))
            await asyncio.wait_for(answered.wait(), timeout=10)
            if ending == "stop":
                turn.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(turn, timeout=10)

    persist.assert_awaited_once()
    assert persist.await_args.kwargs["cost_usd"] == pytest.approx(_CALL_COST)
    assert persist.await_args.kwargs["prompt_tokens"] == _CALL_USAGE["input_tokens"]


async def _read_like_the_executor(stream, answered: asyncio.Event) -> None:
    """``CoPilotProcessor._execute_async``: read through the heartbeat, stop at
    the first StreamError and close the stream."""
    published = wrap_stream_with_heartbeat(stream)
    try:
        async for event in published:
            if isinstance(event, StreamTextDelta):
                answered.set()
            if isinstance(event, StreamError):
                break
    finally:
        await published.aclose()


def _resumed_cli_session(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    session_file = Path(cli_session_path(_SDK_CWD, _SESSION_ID))
    session_file.parent.mkdir(parents=True)
    return session_file


@contextlib.contextmanager
def _patched_turn(model: str, client_factory):
    """A resumed turn whose chat's first turn cost one call."""
    transcript = build_test_transcript(
        [("user", "prior question"), ("assistant", "prior answer")]
    ) + _native_cost_row(model, 1)
    session = ChatSession(
        session_id=_SESSION_ID,
        user_id="test-user",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[
            ChatMessage(role="user", content="prior question"),
            ChatMessage(role="assistant", content="prior answer"),
            ChatMessage(role="user", content="hello"),
        ],
    )
    patches = _make_sdk_patches(
        session,
        original_transcript=transcript,
        compacted_transcript=None,
        client_side_effect=client_factory,
    )
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        stack.enter_context(
            patch(
                f"{_SVC}.build_skills_update_notice",
                new_callable=AsyncMock,
                return_value=None,
            )
        )
        persist = stack.enter_context(
            patch(f"{_SVC}.persist_and_record_usage", new_callable=AsyncMock)
        )

        def run_turn():
            return stream_chat_completion_sdk(
                session_id=_SESSION_ID,
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            )

        yield persist, run_turn
