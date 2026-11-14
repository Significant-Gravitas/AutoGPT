"""A turn whose sandbox cannot start, refused under a running-box limit or
failed for any other reason, ends with an error on both engines, and never runs
its commands on the local fallback."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from backend.copilot.baseline import service as baseline_service
from backend.copilot.baseline.service import stream_chat_completion_baseline
from backend.copilot.constants import COPILOT_RETRYABLE_ERROR_PREFIX
from backend.copilot.model import ChatSession
from backend.copilot.model_router import ResolvedModel
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import StreamError, StreamStart
from backend.copilot.sdk import retry_scenarios_test as scenarios
from backend.copilot.sdk.cli_prelaunch_turn_test import _patched_turn
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.tools.e2b_sandbox import SandboxUnavailableError
from backend.util.sandbox_limits import SandboxLimitError

_REFUSAL = SandboxLimitError("Too many sandboxes are running on your account.")
_UNAVAILABLE = SandboxUnavailableError()
# What setup raises, and the error the turn ends with: the refusal as written,
# and any other failure (an error from E2B's API, say) as one.
_CASES = pytest.mark.parametrize(
    "raised, shown",
    [
        pytest.param(_REFUSAL, _REFUSAL, id="limit"),
        pytest.param(RuntimeError("E2B API returned 500"), _UNAVAILABLE, id="failed"),
    ],
)
# The first thing a turn does once its sandbox is set up: a turn that went on
# without one fails here at once instead of running on into the unmocked rest.
_WENT_ON = AssertionError("the turn went on without its sandbox")


@_CASES
async def test_the_sdk_engine_ends_a_turn_whose_sandbox_cannot_start(
    raised: Exception, shown: SandboxLimitError | SandboxUnavailableError
):
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
            patch(f"{svc}.get_or_create_sandbox", AsyncMock(side_effect=raised))
        )
        stack.enter_context(patch(f"{svc}.workspace_volume_mounts", MagicMock()))
        forgot = stack.enter_context(patch(f"{svc}.forget_turn_ceiling", AsyncMock()))
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

    assert [e for e in events if isinstance(e, StreamError)] == [_error(shown)]
    assert not [e for e in events if isinstance(e, StreamStart)]
    assert launches == []
    assert session.messages[-1].content == _marker(shown)
    forgot.assert_awaited_once_with("test-session-id", session.expert_id)


@_CASES
async def test_the_baseline_engine_ends_a_turn_whose_sandbox_cannot_start(
    raised: Exception, shown: SandboxLimitError | SandboxUnavailableError
):
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
        patch(f"{svc}.get_or_create_sandbox", AsyncMock(side_effect=raised)),
        patch(f"{svc}.workspace_volume_mounts", MagicMock()),
        patch(f"{svc}.forget_turn_ceiling", AsyncMock()) as forgot,
        patch(f"{svc}.restore_pending_messages", AsyncMock()) as requeue,
        patch(f"{svc}.upsert_chat_session", AsyncMock()) as saved,
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

    assert events == [_error(shown)]
    requeue.assert_awaited_once_with(session.session_id, [queued])
    saved.assert_awaited_once_with(session)
    assert session.messages[-1].content == _marker(shown)
    forgot.assert_awaited_once_with(session.session_id, session.expert_id)


def _error(shown: SandboxLimitError | SandboxUnavailableError) -> StreamError:
    return StreamError(errorText=str(shown), code=shown.code)


def _marker(shown: SandboxLimitError | SandboxUnavailableError) -> str:
    """What a reloaded chat shows in place of the turn's reply."""
    return f"{COPILOT_RETRYABLE_ERROR_PREFIX} {shown}"
