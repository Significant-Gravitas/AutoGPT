"""An onboarding interview's turns on the executor: no plan allowance, no turn
once the interview is over, and always the fast path."""

import logging
import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.executor.utils import (
    CoPilotExecutionEntry,
    CoPilotLogMetadata,
    ScheduledTurnOrigin,
)
from backend.copilot.model import (
    INTERVIEW_STAGE_CLOSED,
    INTERVIEW_STAGE_HANDED_OFF,
    INTERVIEW_STAGE_WRAP_UP_ISSUED,
    ONBOARDING_INTERVIEW_KIND,
    ChatMessage,
    ChatSession,
)
from backend.copilot.onboarding_interview import (
    INTERVIEW_CLOSED,
    INTERVIEW_PLAIN_USER_TURNS_ONLY,
    INTERVIEW_TURN_CAP,
)
from backend.copilot.turn_pause import TurnPause

_PROC = "backend.copilot.executor.processor"
PAUSE = TurnPause(
    reason="weekly",
    resets_at=datetime(2026, 11, 10, tzinfo=UTC),
    checkpoint_saved=False,
    started=False,
)


@pytest.fixture(autouse=True)
def no_trial_attribution(mocker):
    store = MagicMock()
    store.get_usage_activation_state = AsyncMock(
        side_effect=lambda user_id: SimpleNamespace(
            user_id=user_id, generation=None, trial_id=None, ready=True
        )
    )
    mocker.patch(
        "backend.copilot.usage_activation.pro_activation_db", return_value=store
    )


class _EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self) -> None:
        return None


class _Turn:
    """Runs a turn with the registry, session store and engines replaced."""

    def __init__(self, session: ChatSession) -> None:
        self.session = session
        self.refusal = AsyncMock(return_value=PAUSE)
        self.complete = AsyncMock()
        self.baseline = MagicMock(return_value=MagicMock())
        self.sdk = MagicMock(return_value=MagicMock())
        self.dummy = MagicMock(return_value=MagicMock())
        self.use_sdk = AsyncMock(return_value=True)

    async def run(self, *, test_mode: bool = False, **entry_fields) -> None:
        entry = CoPilotExecutionEntry(
            session_id=self.session.session_id,
            turn_id="turn-1",
            user_id="user-1",
            message="I run a bakery.",
            **entry_fields,
        )
        with (
            patch(f"{_PROC}.refusal_for", self.refusal),
            patch(f"{_PROC}.resolve_use_sdk", self.use_sdk),
            patch(f"{_PROC}.stream_registry.publish_chunk", AsyncMock()),
            patch(f"{_PROC}.stream_registry.mark_session_completed", self.complete),
            patch(
                f"{_PROC}.stream_registry.stream_and_publish",
                MagicMock(return_value=_EmptyStream()),
            ),
            patch(
                f"{_PROC}.ChatConfig",
                return_value=MagicMock(
                    test_mode=test_mode,
                    thinking_available=True,
                    load_test_email_domain="",
                ),
            ),
            patch(f"{_PROC}.stream_chat_completion_baseline", self.baseline),
            patch(f"{_PROC}.sdk_service.stream_chat_completion_sdk", self.sdk),
            patch(f"{_PROC}.stream_chat_completion_dummy", self.dummy),
            patch(
                "backend.copilot.model.get_chat_session",
                AsyncMock(return_value=self.session),
            ),
        ):
            await CoPilotProcessor()._execute_async(
                entry,
                threading.Event(),
                MagicMock(),
                CoPilotLogMetadata(logger=logging.getLogger("test-copilot")),
            )


def _interview(*messages: ChatMessage, stage: str | None = None) -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    session.session_id = "sess-1"
    session.metadata.kind = ONBOARDING_INTERVIEW_KIND
    session.metadata.onboarding_interview_stage = stage
    session.messages = list(messages)
    return session


def _answered(count: int) -> list[ChatMessage]:
    rows: list[ChatMessage] = []
    for n in range(count):
        rows.append(ChatMessage(role="user", content=f"Answer {n}."))
        rows.append(ChatMessage(role="assistant", content=f"Reply {n}."))
    return rows


@pytest.mark.asyncio
async def test_an_interview_turn_runs_before_the_plan_step():
    """No plan, no allowance: the interview's own bounds were checked when the
    turn was admitted."""
    turn = _Turn(_interview(*_answered(2)))

    await turn.run()

    turn.refusal.assert_not_awaited()
    turn.baseline.assert_called_once()
    turn.complete.assert_awaited_once_with(
        "sess-1",
        error_message=None,
        error_code=None,
        skip_error_publish=False,
        turn_id="turn-1",
    )


@pytest.mark.asyncio
async def test_an_interview_runs_on_the_fast_path_whatever_the_engine_flags_say():
    turn = _Turn(_interview(*_answered(2), stage=INTERVIEW_STAGE_WRAP_UP_ISSUED))

    await turn.run()

    turn.baseline.assert_called_once()
    turn.sdk.assert_not_called()
    turn.use_sdk.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_ordinary_chat_still_passes_the_allowance_and_the_engine_flags():
    session = _interview(*_answered(2))
    session.metadata.kind = "normal"
    turn = _Turn(session)
    turn.refusal.return_value = None

    await turn.run()

    turn.refusal.assert_awaited_once_with("user-1")
    turn.use_sdk.assert_awaited_once()
    turn.sdk.assert_called_once()
    turn.baseline.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [INTERVIEW_STAGE_CLOSED, INTERVIEW_STAGE_HANDED_OFF])
async def test_a_closed_interview_takes_no_turn_whatever_queued_it(stage: str):
    turn = _Turn(_interview(*_answered(2), stage=stage))

    await turn.run()

    turn.baseline.assert_not_called()
    turn.refusal.assert_not_awaited()
    turn.complete.assert_awaited_once_with(
        "sess-1",
        error_message="The interview is over.",
        error_code=INTERVIEW_CLOSED,
        skip_error_publish=False,
        turn_id="turn-1",
    )


@pytest.mark.asyncio
async def test_an_interview_with_its_replies_used_up_takes_no_turn():
    turn = _Turn(_interview(*_answered(INTERVIEW_TURN_CAP)))

    await turn.run(test_mode=True)

    turn.dummy.assert_not_called()
    assert turn.complete.await_args.kwargs["error_code"] == INTERVIEW_CLOSED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry_fields",
    [
        pytest.param(
            {"scheduled": ScheduledTurnOrigin(schedule_id="sched-1")},
            id="scheduled",
        ),
        pytest.param({"unattended": True}, id="fired-by-the-scheduler"),
        pytest.param({"is_user_message": False}, id="not-a-user-message"),
        pytest.param(
            {"message_metadata": {"from_session_id": "another-session"}},
            id="sent-from-another-session",
        ),
    ],
)
async def test_an_interview_takes_only_its_owner_s_own_messages(
    entry_fields: dict,
):
    """Whatever queued it, a turn the person did not send never runs in an
    interview, and is refused before the allowance step."""
    turn = _Turn(_interview(*_answered(2)))

    await turn.run(**entry_fields)

    turn.baseline.assert_not_called()
    turn.sdk.assert_not_called()
    turn.refusal.assert_not_awaited()
    assert (
        turn.complete.await_args.kwargs["error_code"] == INTERVIEW_PLAIN_USER_TURNS_ONLY
    )


@pytest.mark.asyncio
async def test_an_ordinary_chat_still_takes_a_message_sent_from_another_session():
    session = _interview(*_answered(2))
    session.metadata.kind = "normal"
    turn = _Turn(session)
    turn.refusal.return_value = None

    await turn.run(message_metadata={"from_session_id": "another-session"})

    turn.sdk.assert_called_once()
