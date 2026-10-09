"""Retrying a platform turn must retain spend from the failed attempt."""

import contextlib
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamError
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.transcript import cli_session_path

from .conftest import build_cli_cost_row, build_test_transcript
from .retry_scenarios_test import _make_sdk_patches
from .turn_cost_test import _CALL_COST, _SDK_CWD, _SESSION_ID, _SVC, _result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["transient", "transient-assistant", "context-result", "context-zero"]
)
async def test_retry_retains_failed_attempt_spend(
    failure, tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    session_file = Path(cli_session_path(_SDK_CWD, _SESSION_ID))
    session_file.parent.mkdir(parents=True)
    transcript = build_test_transcript(
        [("user", "prior question"), ("assistant", "prior answer")]
    ) + build_cli_cost_row(_SESSION_ID, _CALL_COST)
    compacted = build_test_transcript(
        [("user", "summary"), ("assistant", "summarized answer")]
    )
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
    launched = []

    def client_factory(*args, **kwargs):
        launched.append(kwargs["options"])
        first_process = len(launched) == 1

        async def receive():
            if first_process:
                if failure != "context-zero":
                    with session_file.open("a") as handle:
                        handle.write(build_cli_cost_row(_SESSION_ID, 2 * _CALL_COST))
                if failure == "transient":
                    raise ConnectionError("ECONNRESET: connection reset by peer")
                if failure == "transient-assistant":
                    yield AssistantMessage(
                        content=[], model="claude-sonnet-4-6", error="rate_limit"
                    )
                    return
                result = _result(0 if failure == "context-zero" else 2 * _CALL_COST)
                result.subtype = "error_during_execution"
                result.is_error = True
                result.result = "Prompt is too long"
                if failure == "context-zero":
                    result.usage = {}
                yield result
                return
            yield AssistantMessage(
                content=[TextBlock(text="ok")], model="claude-sonnet-4-6"
            )
            yield _result((2 if failure.startswith("context") else 3) * _CALL_COST)

        client = MagicMock()
        client.query = AsyncMock()
        client.receive_response = receive
        context = AsyncMock()
        context.__aenter__.return_value = client
        context.__aexit__.return_value = None
        return context

    patches = _make_sdk_patches(
        session,
        original_transcript=transcript,
        compacted_transcript=compacted,
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
        stack.enter_context(patch(f"{_SVC}._compute_transient_backoff", return_value=0))
        persist = stack.enter_context(
            patch(f"{_SVC}.persist_and_record_usage", new_callable=AsyncMock)
        )
        events = [
            event
            async for event in stream_chat_completion_sdk(
                session_id=_SESSION_ID,
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            )
        ]

    assert [options.resume for options in launched] == [
        _SESSION_ID,
        None if failure.startswith("context") else _SESSION_ID,
    ]
    assert not any(isinstance(event, StreamError) for event in events)
    persist.assert_awaited_once()
    expected = (3 if failure == "context-result" else 2) * _CALL_COST
    assert persist.await_args.kwargs["cost_usd"] == pytest.approx(expected)
    if failure.startswith("context"):
        assert persist.await_args.kwargs["prompt_tokens"] == (
            2000 if failure == "context-result" else 1000
        )
    assert "Over-charge fallback" not in caplog.text
