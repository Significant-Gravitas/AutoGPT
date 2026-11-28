"""Both engines put the hint on the turn's model input, and neither stores it."""

import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot import model_hints
from backend.copilot.baseline import service_cancellation_test as baseline_fixtures
from backend.copilot.model import ChatSession
from backend.copilot.model_hint_profiles import PROFILES_BY_KEY, render_hint
from backend.copilot.model_hints_test import choice, jev_result
from backend.copilot.response_model import StreamError
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.retry_scenarios_test import (
    _build_transcript,
    _make_sdk_patches,
)
from backend.copilot.tools import skill_matching_helpers_test as baseline
from backend.copilot.tools import skill_matching_integration_test as sdk_harness

baseline_io = baseline_fixtures.baseline_io
HINT = render_hint(PROFILES_BY_KEY["web_research"])
TASK = "Find the latest pricing for HubSpot's Starter plan"


@pytest.fixture
def jev(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    call = AsyncMock(return_value=jev_result(choice("web_research", 0.99)))
    monkeypatch.setattr(model_hints, "call_jev", call)
    monkeypatch.setattr(model_hints, "_api_key", "jev-key")
    return call


@pytest.fixture(params=[True, False], ids=["flag-on", "flag-off"])
def flag_on(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    monkeypatch.setattr(
        model_hints, "is_feature_enabled", AsyncMock(return_value=request.param)
    )
    return request.param


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [False, True], ids=["first-send", "context-retry"])
async def test_sdk_engine_hints_every_send_of_the_turn(
    jev: AsyncMock, flag_on: bool, retry: bool
) -> None:
    session = sdk_harness._sdk_session()
    session.messages[-1].content = TASK
    queries: list[str] = []
    patches = _make_sdk_patches(
        session,
        original_transcript=_build_transcript(
            [("user", "Earlier"), ("assistant", "Ready")]
        ),
        compacted_transcript=None,
        client_side_effect=sdk_harness._sdk_client_factory(queries, retry),
    )
    with ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        sdk_harness._patch_sdk_context(stack, resume=True)
        # Memory recall would open a real embedding client.
        stack.enter_context(
            patch.object(sdk_service, "_graphiti_enabled", return_value=False)
        )
        stack.enter_context(
            patch.object(
                sdk_service, "build_matched_skills_notice", AsyncMock(return_value="")
            )
        )
        async with asyncio.timeout(10):
            events = [
                event
                async for event in sdk_service.stream_chat_completion_sdk(
                    session_id=session.session_id,
                    message=TASK,
                    user_id="test-user",
                    session=session,
                )
            ]

    assert not [event for event in events if isinstance(event, StreamError)]
    assert len(queries) == (2 if retry else 1)
    for query in queries:
        assert (HINT + TASK in query) is flag_on
    assert jev.await_count == (1 if flag_on else 0)
    assert all("<model_hint>" not in (m.content or "") for m in session.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_turn", [False, True])
async def test_baseline_engine_hints_the_request_after_its_context(
    monkeypatch: pytest.MonkeyPatch,
    baseline_io: list[ChatSession],
    jev: AsyncMock,
    flag_on: bool,
    first_turn: bool,
) -> None:
    session = baseline.new_session(first_turn)
    provider, upload, _ = baseline.patch_provider(monkeypatch)
    monkeypatch.setattr(
        baseline.service, "build_matched_skills_notice", AsyncMock(return_value="")
    )
    prefix = baseline.patch_first_turn_context(monkeypatch) if first_turn else ""

    await baseline.run_turn(session, TASK)

    request = baseline._provider_user_texts(provider)[-1]
    assert request == prefix + (HINT if flag_on else "") + TASK
    assert jev.await_count == (1 if flag_on else 0)
    assert "<model_hint>" not in baseline._uploaded_text(upload)
    assert all("<model_hint>" not in (m.content or "") for m in session.messages)
