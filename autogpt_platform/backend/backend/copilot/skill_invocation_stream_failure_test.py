from contextlib import nullcontext
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot import stream_registry
from backend.copilot.baseline import service_unit_test as baseline_tests
from backend.copilot.executor.processor_error_code_test import _run
from backend.copilot.executor.processor_error_code_test import (
    allowance_admits_every_turn as allowance_admits_every_turn,
)
from backend.copilot.executor.processor_error_code_test import (
    no_trial_attribution as no_trial_attribution,
)
from backend.copilot.response_model import StreamError, StreamFinish
from backend.copilot.sdk import retry_scenarios_test as sdk_tests
from backend.copilot.skill_invocation import InvocableSkill
from backend.copilot.stream_registry_test import _FakeRedis
from backend.copilot.tools.skills import ParsedSkill


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["baseline", "sdk"])
async def test_command_persistence_failure_completes_the_turn_with_stream_error(engine):
    captured = []
    queries = []
    module = baseline_tests if engine == "baseline" else sdk_tests
    entrypoint = f"stream_chat_completion_{engine}"
    original = getattr(module, entrypoint)

    async def capture(*args, **kwargs):
        async for event in original(*args, **kwargs):
            captured.append(event)
            yield event

    found = InvocableSkill(
        skill=ParsedSkill(name="report", description="Report", body="Write $0"),
        is_default=True,
    )
    with (
        patch.object(module, entrypoint, new=capture),
        patch(
            "backend.copilot.skill_invocation.is_skills_feature_enabled",
            AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.skill_invocation.resolve_invocable_skill",
            AsyncMock(return_value=found),
        ),
    ):
        with pytest.raises(
            RuntimeError, match="Cannot persist the current user message"
        ) as raised:
            if engine == "baseline":
                session = baseline_tests._baseline_session("/report weekly")
                session.messages[0].sequence = None
                await baseline_tests._run_baseline_turn(
                    session, first_block=None, refresh=baseline_tests._refresh_nothing
                )
            else:
                helper = sdk_tests.TestFirstTurnWarmContextIsEphemeral()
                session = helper._first_turn("/report weekly")
                session.messages[0].sequence = None
                await helper._run(
                    session,
                    sdk_tests.TestFollowUpWarmContextCallSite._clients(queries),
                    first_block=None,
                    refresh=helper._no_refresh,
                    expect_error=True,
                )
    assert not queries
    assert "<skill_invocation>" not in (session.messages[0].content or "")
    errors = [event for event in captured if isinstance(event, StreamError)]
    assert len(errors) == (1 if engine == "sdk" else 0)
    if errors:
        assert errors[0].code == "sdk_error"

    class FailedTurn:
        def __aiter__(self):
            return self.events()

        async def events(self):
            for event in captured:
                yield event
            raise raised.value

        async def aclose(self):
            pass

    published = AsyncMock()
    completed = AsyncMock(wraps=stream_registry.mark_session_completed)
    with (
        patch.object(
            stream_registry,
            "get_redis_async",
            AsyncMock(return_value=_FakeRedis({"status": "running"})),
        ),
        patch.object(
            stream_registry, "hash_compare_and_set", AsyncMock(return_value=True)
        ),
        patch.object(stream_registry, "publish_chunk", published),
        patch(
            "backend.copilot.executor.processor.is_user_banned",
            AsyncMock(return_value=False),
        ),
    ):
        expectation = (
            pytest.raises(RuntimeError, match="Cannot persist the current user message")
            if engine == "baseline"
            else nullcontext()
        )
        with expectation:
            await _run(FailedTurn(), completed)

    events = [call.args[1] for call in published.await_args_list]
    assert any(
        isinstance(event, StreamError) and event.errorText == str(raised.value)
        for event in events
    )
    assert any(isinstance(event, StreamFinish) for event in events)
    assert completed.await_args.kwargs["error_message"] == str(raised.value)
