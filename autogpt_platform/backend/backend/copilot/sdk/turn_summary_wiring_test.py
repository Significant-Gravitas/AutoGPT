"""The turn-summary line is emitted exactly once per SDK turn, whatever its end.

Drives the real ``stream_chat_completion_sdk`` with a mocked CLI client, on a
resumed session so the first-turn context injection (which reaches the DB)
does not run, and with every Redis or database call of that path stubbed.
"""

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.conftest import build_test_transcript
from backend.copilot.sdk.retry_scenarios_test import _make_sdk_patches
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.sdk.turn_summary_model import TURN_SUMMARY_MARKER

_SVC = "backend.copilot.sdk.service"
_SUMMARY_LOGGER = "backend.copilot.sdk.turn_summary"


def _session() -> ChatSession:
    return ChatSession(
        session_id="test-session-id",
        user_id="test-user",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[
            ChatMessage(role="user", content="earlier question", sequence=0),
            ChatMessage(role="assistant", content="earlier answer", sequence=1),
            ChatMessage(role="user", content="hello", sequence=2),
        ],
    )


def _client(messages: list, query_error: BaseException | None = None) -> AsyncMock:
    async def _receive():
        for message in messages:
            yield message

    client = MagicMock()
    client.receive_response = _receive
    client.query = AsyncMock(side_effect=query_error)
    client._transport = MagicMock()
    client._transport.write = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    return cm


def _answer() -> list:
    return [
        AssistantMessage(
            content=[TextBlock(text="hi there")],
            model="claude-sonnet-4-6",
            usage={"input_tokens": 12, "cache_read_input_tokens": 300},
            message_id="msg_1",
        ),
        ResultMessage(
            subtype="success",
            result="hi there",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=1,
            session_id="test-session-id",
            total_cost_usd=0.0123,
            usage={
                "input_tokens": 12,
                "cache_read_input_tokens": 300,
                "cache_creation_input_tokens": 40,
                "output_tokens": 7,
            },
        ),
    ]


async def _run_turn(tmp_path: Path, client_factory) -> list:
    session = _session()
    transcript = build_test_transcript(
        [("user", "earlier question"), ("assistant", "earlier answer")]
    )
    patches = _make_sdk_patches(session, transcript, None, client_factory) + [
        (f"{_SVC}._make_sdk_cwd", dict(return_value=str(tmp_path))),
        (f"{_SVC}.resolve_answered", dict(new_callable=AsyncMock, return_value=[])),
        (f"{_SVC}._update_title_async", dict(new_callable=AsyncMock)),
        (
            f"{_SVC}.build_turn_budget_block",
            dict(new_callable=AsyncMock, return_value=""),
        ),
        (
            f"{_SVC}.build_skills_update_notice",
            dict(new_callable=AsyncMock, return_value=""),
        ),
        (f"{_SVC}.persist_and_record_usage", dict(new_callable=AsyncMock)),
    ]
    events = []
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        sdk_service.config.effective_transport = "direct_anthropic"
        sdk_service.config.openrouter_active = False
        with contextlib.suppress(Exception, asyncio.CancelledError):
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)
    return events


def _summaries(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        json.loads(r.getMessage().split(" ", 1)[1])
        for r in caplog.records
        if r.name == _SUMMARY_LOGGER and r.getMessage().startswith(TURN_SUMMARY_MARKER)
    ]


@pytest.mark.asyncio
async def test_a_successful_turn_logs_one_summary(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=_SUMMARY_LOGGER):
        await _run_turn(tmp_path, lambda *a, **k: _client(_answer()))

    [summary] = _summaries(caplog)
    assert summary["outcome"] == "ok"
    assert summary["session_id"] == "test-session-id"
    assert (summary["turn"], summary["first_turn"]) == (2, False)
    assert (summary["cli_launches"], summary["retries"]) == (1, 0)
    assert (summary["model_calls"], summary["tool_calls"]) == (1, 0)
    assert summary["model_answered"] == "claude-sonnet-4-6"
    assert (summary["input_tokens"], summary["cache_read_tokens"]) == (12, 300)
    assert (summary["cache_write_tokens"], summary["output_tokens"]) == (40, 7)
    assert (summary["cost_usd"], summary["cost_source"]) == (0.0123, "cli_reported")
    assert summary["transcript_download_bytes"] > 0
    assert summary["transcript_covered_rows"] == 2
    assert summary["session_rows"] == 3
    assert summary["query_path"] in {"resume", "no_resume_history"}
    assert summary["transcript_upload_skipped"] == "no_session_file"
    assert summary["system_prompt_chars"] > 0
    assert "error_class" not in summary


@pytest.mark.asyncio
async def test_a_failing_turn_logs_one_summary(tmp_path, caplog):
    def _failing(*args, **kwargs):
        return _client([], query_error=RuntimeError("authentication failed"))

    with caplog.at_level(logging.INFO, logger=_SUMMARY_LOGGER):
        await _run_turn(tmp_path, _failing)

    [summary] = _summaries(caplog)
    assert summary["outcome"] == "error"
    assert summary["error_class"]
    assert summary["cli_launches"] == 1
    assert summary["transcript_upload_skipped"] == "stream_error"


@pytest.mark.asyncio
async def test_a_cancelled_turn_logs_one_summary(tmp_path, caplog):
    def _cancelled(*args, **kwargs):
        return _client([], query_error=asyncio.CancelledError())

    with caplog.at_level(logging.INFO, logger=_SUMMARY_LOGGER):
        await _run_turn(tmp_path, _cancelled)

    [summary] = _summaries(caplog)
    assert (summary["outcome"], summary["error_class"]) == (
        "cancelled",
        "CancelledError",
    )
