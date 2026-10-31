"""A turn whose sandbox start is refused under a running-box limit ends with
the refusal on both engines, and never runs its commands on the local fallback."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from backend.copilot.baseline import service as baseline_service
from backend.copilot.baseline.service import stream_chat_completion_baseline
from backend.copilot.model import ChatSession
from backend.copilot.model_router import ResolvedModel
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import StreamError, StreamStart
from backend.copilot.sdk import retry_scenarios_test as scenarios
from backend.copilot.sdk.cli_prelaunch_turn_test import _patched_turn
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.util.sandbox_limits import SandboxLimitError

_REFUSAL = SandboxLimitError("Too many sandboxes are running on your account.")
_EXPECTED = StreamError(errorText=str(_REFUSAL), code="sandbox_limit_reached")
# The first thing a turn does once its sandbox is set up: a turn that went on
# without one fails here at once instead of running on into the unmocked rest.
_WENT_ON = AssertionError("the turn went on without its sandbox")


async def test_the_sdk_engine_ends_a_refused_turn_with_the_refusal():
    svc = "backend.copilot.sdk.service"
    turns = scenarios.TestStreamChatCompletionRetryIntegration()
    launches: list[dict] = []

    def client(*_args, **kwargs):
        launches.append(kwargs)
        return turns._make_client_mock(result_message=turns._make_result_message())

    session = turns._make_session()
    events = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched_turn(client, session))
        stack.enter_context(patch(f"{svc}.config.active_e2b_api_key", "e2b-key"))
        stack.enter_context(
            patch(f"{svc}.get_or_create_sandbox", AsyncMock(side_effect=_REFUSAL))
        )
        stack.enter_context(patch(f"{svc}.workspace_volume_mounts", MagicMock()))
        stack.enter_context(patch(f"{svc}.forget_turn_ceiling", AsyncMock()))
        stack.enter_context(
            patch(f"{svc}.approval_mode_supplement", MagicMock(side_effect=_WENT_ON))
        )
        async for event in stream_chat_completion_sdk(
            session_id="test-session-id",
            message="hello",
            is_user_message=True,
            user_id="test-user",
            session=session,
        ):
            events.append(event)

    assert [e for e in events if isinstance(e, StreamError)] == [_EXPECTED]
    assert not [e for e in events if isinstance(e, StreamStart)]
    assert launches == []


async def test_the_baseline_engine_ends_a_refused_turn_and_keeps_queued_messages():
    svc = "backend.copilot.baseline.service"
    session = ChatSession.new("user-1", dry_run=False)
    queued = PendingMessage(content="and one more thing")

    with (
        patch(f"{svc}.build_expert_identity_suffix", AsyncMock(return_value="")),
        patch(f"{svc}.drain_pending_safe", AsyncMock(return_value=[queued])),
        patch(
            f"{svc}._resolve_baseline_model",
            AsyncMock(
                return_value=ResolvedModel(
                    model="anthropic/claude-sonnet-4-6", source="env"
                )
            ),
        ),
        patch(
            f"{svc}.normalize_model_for_transport",
            MagicMock(side_effect=lambda model, cfg=None: model),
        ),
        patch.object(
            type(baseline_service.config),
            "active_e2b_api_key",
            new_callable=PropertyMock,
            return_value="e2b-key",
        ),
        patch(f"{svc}.get_or_create_sandbox", AsyncMock(side_effect=_REFUSAL)),
        patch(f"{svc}.workspace_volume_mounts", MagicMock()),
        patch(f"{svc}.forget_turn_ceiling", AsyncMock()),
        patch(f"{svc}.push_pending_message", AsyncMock()) as requeue,
        patch(f"{svc}._build_system_prompt", MagicMock(side_effect=_WENT_ON)),
    ):
        events = [
            event
            async for event in stream_chat_completion_baseline(
                session_id=session.session_id,
                message=None,
                is_user_message=False,
                user_id="user-1",
                session=session,
            )
        ]

    assert events == [_EXPECTED]
    requeue.assert_awaited_once_with(session.session_id, queued)
