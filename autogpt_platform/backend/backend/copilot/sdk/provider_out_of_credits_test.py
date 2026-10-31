"""The CLI reporting an empty platform provider account must not reach the chat raw.

The Claude CLI turns a provider billing refusal into a synthetic
``AssistantMessage`` whose text is the provider's own wording ("Credit
balance is too low", OpenRouter's "purchase more at openrouter.ai..."). Left
alone, the adapter streams that as the assistant's reply.
"""

import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultError, ResultMessage, TextBlock

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamError, StreamTextDelta
from backend.copilot.sdk.service import (
    _classify_final_failure,
    _FinalFailure,
    _InterruptedAttempt,
    stream_chat_completion_sdk,
)
from backend.util.llm.provider_billing import PROVIDER_UNAVAILABLE_MESSAGE

from .conftest import build_test_transcript
from .retry_scenarios_test import _make_sdk_patches

_OPENROUTER_402 = (
    'API Error: 402 {"error":{"message":"Insufficient credits. Purchase more at '
    'https://openrouter.ai/settings/credits","code":402}}'
)


def _session() -> ChatSession:
    return ChatSession(
        session_id="test-session-id",
        user_id="test-user",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[ChatMessage(role="user", content="hello")],
    )


def _result(
    text: str, usage: dict | None = None, cost_usd: float | None = None
) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        result=text,
        duration_ms=100,
        duration_api_ms=0,
        is_error=True,
        num_turns=1,
        session_id="test-session-id",
        total_cost_usd=cost_usd,
        usage=usage,
    )


async def _run_turn(messages: list):
    session = _session()
    attempts = [0]

    def _client_factory(*args, **kwargs):
        attempts[0] += 1

        async def _receive():
            for message in messages:
                if isinstance(message, BaseException):
                    raise message
                yield message

        client = MagicMock()
        client._transport = MagicMock()
        client._transport.write = AsyncMock()
        client.query = AsyncMock()
        client.receive_response = _receive
        cm = AsyncMock()
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None
        return cm

    patches = _make_sdk_patches(
        session,
        original_transcript=build_test_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        ),
        compacted_transcript=None,
        client_side_effect=_client_factory,
    )
    events = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch("backend.copilot.sdk.service.asyncio.sleep", new_callable=AsyncMock)
        )
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        async for event in stream_chat_completion_sdk(
            session_id="test-session-id",
            message="hello",
            is_user_message=True,
            user_id="test-user",
            session=session,
        ):
            events.append(event)
    return events, attempts[0], session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refusal,result_text",
    [
        pytest.param(
            AssistantMessage(
                content=[TextBlock(text="Credit balance is too low")],
                model="<synthetic>",
                error="billing_error",
            ),
            "Credit balance is too low",
            id="anthropic-billing-error",
        ),
        pytest.param(
            AssistantMessage(
                content=[TextBlock(text=_OPENROUTER_402)],
                model="<synthetic>",
                error="unknown",
            ),
            _OPENROUTER_402,
            id="openrouter-402",
        ),
    ],
)
async def test_billing_refusal_shows_platform_message(refusal, result_text):
    events, attempts, session = await _run_turn([refusal, _result(result_text)])

    streamed_text = "".join(
        e.delta for e in events if isinstance(e, StreamTextDelta)
    ).lower()
    assert "credit" not in streamed_text
    assert "openrouter" not in streamed_text

    errors = [e for e in events if isinstance(e, StreamError)]
    assert len(errors) == 1
    assert errors[0].code == "provider_unavailable"
    assert "temporarily unavailable" in errors[0].errorText
    assert "openrouter" not in errors[0].errorText.lower()
    assert attempts == 1

    # The row a reload renders carries the same message, not the raw text.
    marker = session.messages[-1].content or ""
    assert "temporarily unavailable" in marker
    assert "openrouter" not in marker.lower()


_ROUNDS_USAGE = {
    "input_tokens": 1200,
    "output_tokens": 300,
    "cache_read_input_tokens": 50,
    "cache_creation_input_tokens": 10,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [
                AssistantMessage(
                    content=[TextBlock(text="Let me look that up.")],
                    model="claude-sonnet-4-6",
                ),
                AssistantMessage(
                    content=[TextBlock(text="Credit balance is too low")],
                    model="<synthetic>",
                    error="billing_error",
                ),
                _result("Credit balance is too low", _ROUNDS_USAGE, 0.042),
            ],
            id="refusal-after-a-paid-round",
        ),
        pytest.param(
            [_result(_OPENROUTER_402, _ROUNDS_USAGE, 0.042)],
            id="refusal-is-the-result",
        ),
    ],
)
async def test_billing_refusal_still_records_the_turns_usage(messages):
    # Rounds the provider already served in this turn are billed on the
    # final ResultMessage, which must still be read after the refusal.
    with patch(
        "backend.copilot.sdk.service.persist_and_record_usage",
        new_callable=AsyncMock,
    ) as persist:
        events, _, _ = await _run_turn(messages)

    errors = [e for e in events if isinstance(e, StreamError)]
    assert [e.code for e in errors] == ["provider_unavailable"]
    persist.assert_awaited_once()
    usage = persist.await_args.kwargs
    assert usage["prompt_tokens"] == 1200
    assert usage["completion_tokens"] == 300
    assert usage["cache_read_tokens"] == 50
    assert usage["cache_creation_tokens"] == 10
    assert usage["cost_usd"] == 0.042


def _raised_402() -> ResultError:
    # How the SDK raises the CLI's error result once the CLI exits non-zero.
    return ResultError(
        f"Claude Code returned an error result: {_OPENROUTER_402}",
        data={
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": _OPENROUTER_402,
            "api_error_status": 402,
            "terminal_reason": "api_error",
        },
        exit_code=1,
    )


def test_raised_billing_error_maps_to_platform_message():
    failure = _classify_final_failure(
        _InterruptedAttempt(),
        attempts_exhausted=False,
        transient_exhausted=False,
        stream_err=_raised_402(),
    )
    assert failure == _FinalFailure(
        display_msg=PROVIDER_UNAVAILABLE_MESSAGE,
        code="provider_unavailable",
        retryable=True,
    )


def _raised_402_quoting_a_rate_limit() -> ResultError:
    # A billing refusal whose text also matches a transient pattern, so the
    # turn retries it as transient until the retries run out.
    text = f"{_OPENROUTER_402} (upstream: rate limit)"
    return ResultError(
        f"Claude Code returned an error result: {text}",
        data={
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": text,
            "api_error_status": 402,
            "terminal_reason": "api_error",
        },
        exit_code=1,
    )


@pytest.mark.parametrize(
    "attempts_exhausted,transient_exhausted",
    [
        pytest.param(False, True, id="transient-retries-exhausted"),
        pytest.param(True, False, id="context-attempts-exhausted"),
    ],
)
def test_billing_refusal_outranks_an_exhausted_retry_verdict(
    attempts_exhausted, transient_exhausted
):
    failure = _classify_final_failure(
        _InterruptedAttempt(),
        attempts_exhausted=attempts_exhausted,
        transient_exhausted=transient_exhausted,
        stream_err=_raised_402_quoting_a_rate_limit(),
    )
    assert failure == _FinalFailure(
        display_msg=PROVIDER_UNAVAILABLE_MESSAGE,
        code="provider_unavailable",
        retryable=True,
    )


def test_codex_billing_refusal_after_transient_retries_stays_transient():
    failure = _classify_final_failure(
        _InterruptedAttempt(),
        attempts_exhausted=False,
        transient_exhausted=True,
        stream_err=_raised_402_quoting_a_rate_limit(),
        platform_route=False,
    )
    assert failure is not None
    assert failure.code == "transient_api_error"


@pytest.mark.asyncio
async def test_raised_billing_refusal_retried_as_transient_ends_as_billing():
    with patch("backend.copilot.sdk.service.report_provider_out_of_credits") as report:
        events, attempts, session = await _run_turn(
            [_raised_402_quoting_a_rate_limit()]
        )

    assert attempts == 2  # the one transient retry the test config allows
    errors = [e for e in events if isinstance(e, StreamError)]
    assert [e.code for e in errors] == ["provider_unavailable"]
    assert "openrouter" not in errors[0].errorText.lower()
    report.assert_called_once()
    marker = session.messages[-1].content or ""
    assert "temporarily unavailable" in marker


def test_codex_billing_error_keeps_the_providers_wording():
    # A linked ChatGPT subscription running out is the user's own limit.
    failure = _classify_final_failure(
        _InterruptedAttempt(),
        attempts_exhausted=False,
        transient_exhausted=False,
        stream_err=_raised_402(),
        platform_route=False,
    )
    assert failure is not None
    assert failure.code == "sdk_stream_error"
    assert "openrouter.ai" in failure.display_msg


@pytest.mark.parametrize(
    "stream_err",
    [
        pytest.param(
            Exception(f"Command failed: {_OPENROUTER_402}"), id="bare-exception"
        ),
        pytest.param(
            ValueError(
                "Tool input rejected: user pasted 'visit "
                "openrouter.ai/settings/credits, requires more credits'"
            ),
            id="exception-quoting-user-text",
        ),
        pytest.param(
            ResultError(
                "Claude Code returned an error result: tool failed",
                data={
                    "subtype": "error_during_execution",
                    "is_error": True,
                    "errors": ["fetched page says: requires more credits"],
                },
            ),
            id="cli-error-result-not-about-billing",
        ),
    ],
)
def test_untyped_error_mentioning_billing_is_not_a_billing_refusal(stream_err):
    # Only a typed provider error or the CLI's own error result is judged;
    # str() of anything else can carry the user's input or a fetched page.
    failure = _classify_final_failure(
        _InterruptedAttempt(),
        attempts_exhausted=False,
        transient_exhausted=False,
        stream_err=stream_err,
    )
    assert failure is not None
    assert failure.code == "sdk_stream_error"
