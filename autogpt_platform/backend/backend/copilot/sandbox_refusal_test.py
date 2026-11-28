"""A turn whose sandbox cannot start, refused under a running-box limit or
failed for any other reason, ends with an error on both engines, and never runs
its commands on the local fallback."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import pytest

from backend.copilot.baseline import service as baseline_service
from backend.copilot.baseline.service import stream_chat_completion_baseline
from backend.copilot.constants import COPILOT_RETRYABLE_ERROR_PREFIX
from backend.copilot.model import ChatSession, LocalExecutionTargetMetadata
from backend.copilot.model_router import ResolvedModel
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import StreamError, StreamStart
from backend.copilot.sdk import retry_scenarios_test as scenarios
from backend.copilot.sdk.cli_prelaunch_turn_test import _patched_turn
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.tool_image import ToolImage
from backend.copilot.tools.e2b_sandbox import SandboxUnavailableError
from backend.copilot.tools.local_pc_machine import MachineRootGrantRefusedError
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
    events, session, launches, forgot, _ = await _sdk_turn(
        AsyncMock(side_effect=raised)
    )

    assert [e for e in events if isinstance(e, StreamError)] == [_error(shown)]
    assert not [e for e in events if isinstance(e, StreamStart)]
    assert launches == []
    assert session.messages[-1].content == _marker(shown)
    forgot.assert_awaited_once_with("test-session-id", session.expert_id)


@pytest.mark.parametrize(
    "exit_code, image",
    [
        ("sandbox_unavailable", True),
        ("local_pc_folder_changed", True),
        ("local_pc_executor_unavailable", True),
        ("sandbox_unavailable", False),
    ],
)
async def test_an_auto_continue_ended_at_setup_requeues_a_batch_with_images(
    exit_code: str, image: bool
):
    """Its rows are already saved as text: only the queue keeps an image."""
    local_pc, setup = {
        "sandbox_unavailable": (
            False,
            AsyncMock(side_effect=RuntimeError("E2B API returned 500")),
        ),
        "local_pc_folder_changed": (
            True,
            AsyncMock(
                side_effect=MachineRootGrantRefusedError(
                    "ROOT_GRANT_INVALID", "folder replaced"
                )
            ),
        ),
        "local_pc_executor_unavailable": (True, AsyncMock(return_value=None)),
    }[exit_code]
    batch = [
        PendingMessage(
            content="Approved screenshot",
            model_image=(
                ToolImage(mime_type="image/png", data_base64="cGl4ZWxz")
                if image
                else None
            ),
        )
    ]

    events, _, _, _, requeue = await _sdk_turn(setup, batch, local_pc=local_pc)

    assert [e.code for e in events if isinstance(e, StreamError)] == [exit_code]
    assert requeue.await_args_list == (
        [call("test-session-id", batch)] if image else []
    )


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


async def _sdk_turn(
    setup: AsyncMock,
    continued_pending: list[PendingMessage] | None = None,
    *,
    local_pc: bool = False,
) -> tuple[list, ChatSession, list[dict], AsyncMock, AsyncMock]:
    """One SDK turn whose executor setup is *setup*, for the E2B sandbox or a
    Local PC chat's computer: its events, session, CLI launches, and the
    ceiling and requeue mocks."""
    svc = "backend.copilot.sdk.service"
    turns = scenarios.TestStreamChatCompletionRetryIntegration()
    launches: list[dict] = []

    def client(*_args, **kwargs):
        launches.append(kwargs)
        return turns._make_client_mock(result_message=turns._make_result_message())

    session = turns._make_session()
    if local_pc:
        session.metadata = session.metadata.model_copy(
            update={
                "execution_target": LocalExecutionTargetMetadata(
                    machine_id="machine-1",
                    directory_ref="directory-1",
                    allowed_root="/Users/ada/Projects",
                    root_fingerprint="a" * 64,
                    root_grant="grant-1",
                    revision=1,
                )
            }
        )
    events = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched_turn(client, session))
        stack.enter_context(patch(f"{svc}.config.active_e2b_api_key", "e2b-key"))
        stack.enter_context(
            patch(
                (
                    f"{svc}._setup_local_pc_executor"
                    if local_pc
                    else f"{svc}.get_or_create_sandbox"
                ),
                setup,
            )
        )
        stack.enter_context(patch(f"{svc}.workspace_volume_mounts", MagicMock()))
        forgot = stack.enter_context(patch(f"{svc}.forget_turn_ceiling", AsyncMock()))
        stack.enter_context(
            patch(f"{svc}.persist_pending_as_user_rows", AsyncMock(return_value=True))
        )
        requeue = stack.enter_context(
            patch(f"{svc}.restore_pending_messages", AsyncMock())
        )
        stack.enter_context(
            patch(f"{svc}.approval_mode_supplement", MagicMock(side_effect=_WENT_ON))
        )
        async for event in stream_chat_completion_sdk(
            session_id="test-session-id",
            message="hello",
            is_user_message=True,
            user_id="test-user",
            session=session,
            continued_pending=continued_pending,
        ):
            events.append(event)
    return events, session, launches, forgot, requeue


def _error(shown: SandboxLimitError | SandboxUnavailableError) -> StreamError:
    return StreamError(errorText=str(shown), code=shown.code)


def _marker(shown: SandboxLimitError | SandboxUnavailableError) -> str:
    """What a reloaded chat shows in place of the turn's reply."""
    return f"{COPILOT_RETRYABLE_ERROR_PREFIX} {shown}"
