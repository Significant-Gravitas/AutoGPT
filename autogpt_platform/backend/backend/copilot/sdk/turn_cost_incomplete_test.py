"""An incomplete SDK response must account for spend saved in its CLI session."""

import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk.service import stream_chat_completion_sdk
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
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    session_file = Path(cli_session_path(_SDK_CWD, _SESSION_ID))
    session_file.parent.mkdir(parents=True)
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
        async for _ in stream_chat_completion_sdk(
            session_id=_SESSION_ID,
            message="hello",
            is_user_message=True,
            user_id="test-user",
            session=session,
        ):
            pass

    persist.assert_awaited_once()
    expected = _kimi_rate_card_cost() if model == _KIMI else _CALL_COST
    assert persist.await_args.kwargs["cost_usd"] == pytest.approx(expected)
    assert persist.await_args.kwargs["prompt_tokens"] == _CALL_USAGE["input_tokens"]
