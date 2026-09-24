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
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

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


async def _run_turn(refusal: AssistantMessage, result_text: str):
    session = _session()
    attempts = [0]

    def _client_factory(*args, **kwargs):
        attempts[0] += 1

        async def _receive():
            yield refusal
            yield ResultMessage(
                subtype="success",
                result=result_text,
                duration_ms=100,
                duration_api_ms=0,
                is_error=True,
                num_turns=1,
                session_id="test-session-id",
            )

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
    events, attempts, session = await _run_turn(refusal, result_text)

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


def _raised_402() -> Exception:
    return Exception(f"Command failed: {_OPENROUTER_402}")


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
