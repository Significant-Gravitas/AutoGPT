"""A resumed SDK turn is charged its own cost, not the CLI session's running total.

On ``--resume`` the CLI restores its session total from the session file's
``cost-state`` row, so a resumed turn's ``ResultMessage.total_cost_usd`` also
counts every earlier turn of the chat.
"""

import contextlib
import json
import logging
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.moonshot import rate_card_usd
from backend.copilot.sdk.cost_tracking import read_cli_session_usage
from backend.copilot.sdk.service import (
    _record_result_usage,
    _RetryState,
    _TokenUsage,
    stream_chat_completion_sdk,
)
from backend.copilot.transcript import cli_session_path

from .conftest import build_test_transcript
from .retry_scenarios_test import _make_sdk_patches

_SVC = "backend.copilot.sdk.service"
_SESSION_ID = "5f0c2a4e-1b7d-4c9a-9e3f-2d8b6a1c4e70"
# What ``_make_sdk_patches`` pins ``_make_sdk_cwd`` to.
_SDK_CWD = "/tmp/test-sdk-cwd"
_CALL_USAGE = {
    "input_tokens": 1000,
    "output_tokens": 100,
    "cache_read_input_tokens": 100_000,
    "cache_creation_input_tokens": 2000,
}
# _CALL_USAGE at the CLI's Sonnet rates, as the locked CLI reports it.
_CALL_COST = 0.028
_KIMI = "moonshotai/kimi-k2.6"


def _kimi_rate_card_cost() -> float:
    rates = rate_card_usd(_KIMI)
    assert rates is not None
    prompt = (
        _CALL_USAGE["input_tokens"]
        + _CALL_USAGE["cache_read_input_tokens"]
        + _CALL_USAGE["cache_creation_input_tokens"]
    )
    return (prompt * rates[0] + _CALL_USAGE["output_tokens"] * rates[1]) / 1e6


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,message_id,sink,expected",
    [
        pytest.param(
            "claude-sonnet-4-6", None, "persist", _CALL_COST, id="cli-cost-recorded"
        ),
        pytest.param(
            "claude-sonnet-4-6",
            "gen-1",
            "reconcile",
            _CALL_COST,
            id="cli-cost-as-openrouter-fallback",
        ),
        pytest.param(
            _KIMI, None, "persist", _kimi_rate_card_cost(), id="moonshot-rate-card"
        ),
    ],
)
async def test_resumed_turn_is_charged_its_own_cost(
    model, message_id, sink, expected, tmp_path, monkeypatch
):
    # The session file the turn resumes says turn 1 cost _CALL_COST, so the
    # CLI reports this identical call as twice that.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    os.makedirs(os.path.dirname(cli_session_path(_SDK_CWD, _SESSION_ID)))
    transcript = build_test_transcript(
        [("user", "prior question"), ("assistant", "prior answer")]
    )
    cost_state = {
        "type": "cost-state",
        "sessionId": _SESSION_ID,
        "totalCostUSD": _CALL_COST,
    }
    transcript = transcript.rstrip("\n") + "\n" + json.dumps(cost_state) + "\n"
    messages = [
        AssistantMessage(
            content=[TextBlock(text="ok")], model=model, message_id=message_id
        ),
        ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=100,
            is_error=False,
            num_turns=1,
            session_id=_SESSION_ID,
            total_cost_usd=2 * _CALL_COST,
            usage=dict(_CALL_USAGE),
            result="ok",
        ),
    ]
    launched = []

    def _client_factory(*args, **kwargs):
        launched.append(kwargs["options"])

        async def _receive():
            for message in messages:
                yield message

        client = MagicMock()
        client.query = AsyncMock()
        client.receive_response = _receive
        cm = AsyncMock()
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None
        return cm

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
        client_side_effect=_client_factory,
    )
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        # A resumed turn diffs the user's skills over a DB RPC.
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
        reconcile = stack.enter_context(
            patch(f"{_SVC}.record_turn_cost_from_openrouter", new_callable=AsyncMock)
        )
        async for _ in stream_chat_completion_sdk(
            session_id=_SESSION_ID,
            message="hello",
            is_user_message=True,
            user_id="test-user",
            session=session,
        ):
            pass

    assert [options.resume for options in launched] == [_SESSION_ID]
    if sink == "persist":
        reconcile.assert_not_called()
        persist.assert_awaited_once()
        assert persist.await_args.kwargs["cost_usd"] == pytest.approx(expected)
    else:
        persist.assert_not_awaited()
        reconcile.assert_called_once()
        cost = reconcile.call_args.kwargs["fallback_cost_usd"]
        assert cost == pytest.approx(expected)


def test_every_query_of_one_cli_process_is_charged():
    # A re-prompt is a second query to the same CLI process, whose total
    # then counts both queries.
    state = _retry_state()
    for running_total in (_CALL_COST, 2 * _CALL_COST):
        _record_result_usage(_result(running_total), state, "")

    assert state.usage.cost_usd == pytest.approx(2 * _CALL_COST)


@pytest.mark.parametrize(
    "models", [(_KIMI, "claude-sonnet-4-6"), ("claude-sonnet-4-6", _KIMI)]
)
def test_model_switch_preserves_the_price_of_each_result(models):
    state = _retry_state()
    for index, model in enumerate(models, start=1):
        state.observed_model = model
        _record_result_usage(_result(index * _CALL_COST), state, "")

    assert state.usage.cost_usd == pytest.approx(_CALL_COST + _kimi_rate_card_cost())


def test_reset_cli_total_does_not_erase_already_billed_spend(caplog):
    state = _retry_state()
    state.usage.cli_session_total_usd = 2 * _CALL_COST
    with caplog.at_level(logging.ERROR):
        for running_total in (3 * _CALL_COST, _CALL_COST, 2 * _CALL_COST):
            _record_result_usage(_result(running_total), state, "")

    assert state.usage.cost_usd == pytest.approx(3 * _CALL_COST)
    assert [r.levelno for r in caplog.records] == [logging.ERROR]


def test_a_total_below_the_baseline_is_charged_in_full_and_alerts(caplog):
    state = _retry_state()
    state.usage.cli_session_total_usd = 2 * _CALL_COST
    with caplog.at_level(logging.ERROR, logger="backend.copilot.sdk.cost_tracking"):
        _record_result_usage(_result(_CALL_COST), state, "")

    assert state.usage.cost_usd == pytest.approx(_CALL_COST)
    assert [r.levelno for r in caplog.records] == [logging.ERROR]


def test_an_unreadable_session_file_is_charged_in_full_and_alerts(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))  # holds no session file
    state = _retry_state()
    with caplog.at_level(logging.ERROR, logger="backend.copilot.sdk.cost_tracking"):
        state.usage.start_cli_session(read_cli_session_usage(_SDK_CWD, _SESSION_ID, ""))
        _record_result_usage(_result(2 * _CALL_COST), state, "")

    assert state.usage.cost_usd == pytest.approx(2 * _CALL_COST)
    assert [r.levelno for r in caplog.records] == [logging.ERROR]


def _retry_state() -> _RetryState:
    return _RetryState(
        options=MagicMock(model="claude-sonnet-4-6"),
        query_message="",
        compaction_stats=None,
        use_resume=False,
        resume_file=None,
        transcript_msg_count=0,
        adapter=MagicMock(),
        transcript_builder=MagicMock(),
        usage=_TokenUsage(),
    )


def _result(total_cost_usd: float) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=100,
        is_error=False,
        num_turns=1,
        session_id=_SESSION_ID,
        total_cost_usd=total_cost_usd,
        usage=dict(_CALL_USAGE),
    )
