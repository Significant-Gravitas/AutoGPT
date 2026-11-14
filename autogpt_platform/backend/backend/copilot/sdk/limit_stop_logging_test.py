"""A turn the platform stops on purpose must not page anyone.

The SDK ends a turn with ``error_max_budget_usd`` or ``error_max_turns`` when a
limit we set does its job. The user is told how to carry on, and the message
must survive a refresh, but Sentry's LoggingIntegration files every ERROR record
as an issue, so these stops buried AUTOGPT-SERVER-8C4 under expected outcomes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import ResultMessage, SystemMessage

from backend.copilot.constants import COPILOT_ERROR_PREFIX
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamError
from backend.copilot.sdk.response_adapter import SDKResponseAdapter
from backend.copilot.sdk.service import _dispatch_response, _StreamAccumulator

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "subtype, code, level",
    [
        ("error_max_budget_usd", "max_budget_exhausted", logging.INFO),
        ("error_max_turns", "max_turns_exhausted", logging.INFO),
        # A turn that really failed still reports.
        ("error_during_execution", "sdk_error", logging.ERROR),
    ],
)
def test_a_limit_stop_reaches_the_user_without_an_error_log(
    subtype: str, code: str, level: int, caplog: pytest.LogCaptureFixture
):
    session = ChatSession(
        session_id="session-1",
        user_id="user-1",
        title="A titled chat",
        messages=[],
        usage=[],
        started_at=_NOW,
        updated_at=_NOW,
    )
    ctx = MagicMock(session=session, codex_gateway=None, log_prefix="[test]")
    acc = _StreamAccumulator(
        assistant_response=ChatMessage(role="assistant", content=""),
        accumulated_tool_calls=[],
    )
    adapter = SDKResponseAdapter(message_id="msg-1", session_id="session-1")
    adapter.convert_message(SystemMessage(subtype="init", data={}))
    responses = adapter.convert_message(
        ResultMessage(
            subtype=subtype,
            duration_ms=100,
            duration_api_ms=50,
            is_error=True,
            num_turns=1,
            session_id="sdk-session-1",
            result="stopped",
            usage={"output_tokens": 0},
        )
    )

    with caplog.at_level(logging.INFO, logger="backend.copilot.sdk.service"):
        for response in responses:
            _dispatch_response(response, acc, ctx, MagicMock(), False, "[test]")

    [error] = [r for r in responses if isinstance(r, StreamError)]
    assert error.code == code
    sent = [r for r in caplog.records if "Sending error to frontend" in r.message]
    assert [r.levelno for r in sent] == [level]
    # Logging it quietly must not cost the user the explanation.
    assert session.messages[-1].content == f"{COPILOT_ERROR_PREFIX} {error.errorText}"
