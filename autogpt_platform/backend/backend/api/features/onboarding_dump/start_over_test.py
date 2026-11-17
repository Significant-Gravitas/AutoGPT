"""Starting a closed onboarding interview over, from its ending screen.

Run on the hand-off rig (``voice_handoff_test``): calls and their hand-offs
on Redis, the typed interview in the chat store's stand-in, and the take on
the in-memory dump table, so these assert which take the plan step reads.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import interview, voice
from backend.api.features.onboarding_dump.interview_test import (
    StoredInterview,
    _session,
    allowance,
    jobs,
    opening,
    stored,
    tester_pick,
)
from backend.api.features.onboarding_dump.service_test import (
    DumpStore,
    dumps,
    extraction,
    personalization_admission,
    quality_gate,
    storage_mocks,
    transcribe,
)
from backend.api.features.onboarding_dump.voice_handoff_test import (
    USER,
    Interviews,
    Rig,
    _close_typed,
    _finish,
    _type,
    interviews,
    redis,
    replica,
    rig,
    typed_arm,
    voice_starts,
)
from backend.copilot.model import INTERVIEW_STAGE_CLOSED, INTERVIEW_STAGE_HANDED_OFF
from backend.copilot.onboarding_interview import INTERVIEW_CLOSED
from backend.voice_relay import locks

__all__ = [
    "allowance",
    "dumps",
    "extraction",
    "interviews",
    "jobs",
    "opening",
    "personalization_admission",
    "quality_gate",
    "redis",
    "replica",
    "rig",
    "storage_mocks",
    "stored",
    "tester_pick",
    "transcribe",
    "typed_arm",
    "voice_starts",
]

PROFILE_CARD = "02-profile-card"


@pytest.fixture(autouse=True)
def onboarding_row(mocker: MockerFixture, interviews: Interviews) -> None:
    """The onboarding row's columns a start-over clears, each only while it
    still holds the id read, as the real update does."""

    async def forget_calls(
        user_id: str, *, latest: str | None, taken: str | None
    ) -> None:
        if latest is not None and interviews.latest_call == latest:
            interviews.latest_call = None
        if taken is not None and interviews.taken == taken:
            interviews.taken = None

    async def forget_session(session_id: str) -> None:
        if interviews.typed is not None and interviews.typed.session_id == session_id:
            interviews.typed = None

    mocker.patch.object(
        interview,
        "forget_onboarding_voice_calls",
        new=AsyncMock(side_effect=forget_calls),
    )
    mocker.patch.object(
        interview,
        "forget_onboarding_interview_session",
        new=AsyncMock(side_effect=forget_session),
    )


@pytest.fixture(autouse=True)
def turn_budget(mocker: MockerFixture) -> AsyncMock:
    """The interview's turn budget, read before anything is set aside: room
    for a greeting unless a test says otherwise."""
    return mocker.patch.object(
        interview, "check_interview_turn_budget", new=AsyncMock()
    )


def _next_typed_interview(
    mocker: MockerFixture, opening: dict[str, MagicMock], interviews: Interviews
) -> StoredInterview:
    """The fresh typed interview the next start opens, with the person's
    answers in it once they type, held by the chat store's stand-in."""
    fresh = _session()
    opening["open"].return_value = fresh
    store = StoredInterview(fresh)
    mocker.patch.object(interview, "get_chat_session", new=store.get)
    mocker.patch.object(interview, "get_chat_session_metadata", new=store.get)
    mocker.patch.object(interview, "update_session_interview_stage", new=store.advance)
    mocker.patch.object(interview, "update_session_interview_ending", new=store.stamp)

    async def opened(user_id: str):
        interviews.typed = fresh
        return fresh

    opening["open"].side_effect = opened
    return store


async def _call_took_the_interview(rig: Rig) -> str:
    call_id = await rig.call()
    record = await rig.hand_off(call_id)
    assert record.interview == locks.INTERVIEW_TAKEN
    assert rig.interviews.taken == call_id
    return call_id


# ---------- without start_over: the refusals stay ----------


async def test_without_start_over_a_typed_start_after_a_calls_take_is_still_refused(
    rig: Rig, opening: dict[str, MagicMock], allowance: AsyncMock
):
    call_id = await _call_took_the_interview(rig)

    with pytest.raises(HTTPException) as refused:
        await interview.start_onboarding_interview(USER)

    assert refused.value.status_code == 409
    assert refused.value.detail == {"code": INTERVIEW_CLOSED, "call_id": call_id}
    assert rig.interviews.taken == call_id


async def test_without_start_over_a_call_after_the_typed_interview_closed_is_still_refused(
    rig: Rig, stored: StoredInterview, voice_starts: None, mocker: MockerFixture
):
    stored.session.metadata.onboarding_interview_stage = INTERVIEW_STAGE_HANDED_OFF
    rig.interviews.typed = stored.session
    mocker.patch.object(
        voice,
        "find_onboarding_interview_session",
        AsyncMock(return_value=stored.session),
    )
    mocker.patch.object(voice, "enforce_voice_call_budget", AsyncMock())

    with pytest.raises(HTTPException) as refused:
        await voice.start_voice_call(USER)

    assert refused.value.status_code == 409
    assert refused.value.detail == {"code": INTERVIEW_CLOSED}
    assert rig.interviews.typed is stored.session


# ---------- typed, after a call took the interview ----------


async def test_typing_over_a_calls_take_replaces_it_only_once_handed_off(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    dumps: DumpStore,
    mocker: MockerFixture,
):
    rig.interviews.typed = stored.session
    call_id = await _call_took_the_interview(rig)
    earlier = stored.session.session_id
    fresh = _next_typed_interview(mocker, opening, rig.interviews)

    started = await interview.start_onboarding_interview(USER, start_over=True)

    assert started.session is fresh.session
    assert rig.interviews.taken is None
    assert rig.interviews.latest_call is None
    assert rig.interviews.typed is fresh.session
    assert rig.interviews.typed.session_id != earlier
    # Until the new interview is handed off, the plan reads the call's take.
    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{call_id}"
    assert dumps.row.status == BrainDumpStatus.completed

    await _type(fresh)
    response = await _close_typed(fresh)

    assert response.status == BrainDumpStatus.transcribed
    assert dumps.row.recordingId == f"interview-{fresh.session.session_id}"
    assert dumps.row.inputMode == BrainDumpInputMode.typed


async def test_a_start_over_retires_the_earlier_typed_interview(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    dumps: DumpStore,
):
    """A typed interview closed on a confirm arm, never confirmed: nothing of
    it ran, and its hand-off can never run after the start-over."""
    stored.session.metadata.onboarding_interview_stage = INTERVIEW_STAGE_CLOSED
    stored.session.metadata.onboarding_interview_ending = PROFILE_CARD
    rig.interviews.typed = stored.session

    await interview.start_onboarding_interview(USER, start_over=True)

    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_HANDED_OFF
    )
    assert rig.interviews.typed is None
    await interview.hand_off_closed_interview(USER, stored.session.session_id)
    assert dumps.row is None


async def test_a_start_over_on_an_interview_that_is_not_closed_starts_as_usual(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
):
    rig.interviews.typed = stored.session

    await interview.start_onboarding_interview(USER, start_over=True)

    assert rig.interviews.typed is stored.session
    assert stored.stages == []


# ---------- spoken, after the typed interview was handed off ----------


async def test_a_call_over_a_handed_off_typed_interview_replaces_it_only_once_handed_off(
    rig: Rig,
    stored: StoredInterview,
    voice_starts: None,
    dumps: DumpStore,
    allowance: AsyncMock,
    mocker: MockerFixture,
):
    rig.interviews.typed = stored.session
    await _close_typed(stored)
    typed_take = f"interview-{stored.session.session_id}"
    assert dumps.row is not None and dumps.row.recordingId == typed_take
    mocker.patch.object(
        voice,
        "find_onboarding_interview_session",
        AsyncMock(side_effect=lambda user_id: rig.interviews.typed),
    )
    mocker.patch.object(voice, "enforce_voice_call_budget", AsyncMock())

    started = await voice.start_voice_call(USER, start_over=True)

    assert started.call_id
    assert rig.interviews.typed is None
    assert dumps.row.recordingId == typed_take

    call_id = await rig.call()
    record = await rig.hand_off(call_id)

    assert record.interview == locks.INTERVIEW_TAKEN
    assert dumps.row.recordingId == f"interview-{call_id}"


async def test_a_call_over_a_calls_take_starts_a_fresh_chain(
    rig: Rig, voice_starts: None, dumps: DumpStore, mocker: MockerFixture
):
    first = await _call_took_the_interview(rig)
    mocker.patch.object(voice, "enforce_voice_call_budget", AsyncMock())

    await voice.start_voice_call(USER, start_over=True)

    assert rig.interviews.taken is None
    second = await rig.call()
    assert (await rig.read(second)).prev is None
    record = await rig.hand_off(second)
    assert record.interview == locks.INTERVIEW_TAKEN
    assert rig.interviews.taken == second
    assert dumps.row is not None and dumps.row.recordingId == f"interview-{second}"
    assert (await rig.read(first)).interview == locks.INTERVIEW_TAKEN


# ---------- what a start-over never does ----------


@pytest.mark.parametrize("mode", ["typed", "voice"])
async def test_a_live_call_still_refuses_a_start_over(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    voice_starts: None,
    mocker: MockerFixture,
    mode: str,
):
    taken = await _call_took_the_interview(rig)
    await locks.take_user_lock(rig.redis, USER, "live-call")
    mocker.patch.object(voice, "enforce_voice_call_budget", AsyncMock())

    with pytest.raises(HTTPException) as refused:
        if mode == "typed":
            await interview.start_onboarding_interview(USER, start_over=True)
        else:
            await voice.start_voice_call(USER, start_over=True)

    assert refused.value.status_code == 409
    assert refused.value.detail == {
        "code": locks.CALL_IN_PROGRESS,
        "call_id": "live-call",
    }
    assert rig.interviews.taken == taken


async def test_a_typed_start_over_past_the_turn_budget_sets_nothing_aside(
    rig: Rig,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    turn_budget: AsyncMock,
):
    taken = await _call_took_the_interview(rig)
    turn_budget.side_effect = HTTPException(status_code=429, detail="later")

    with pytest.raises(HTTPException) as refused:
        await interview.start_onboarding_interview(USER, start_over=True)

    assert refused.value.status_code == 429
    assert rig.interviews.taken == taken
    opening["open"].assert_not_awaited()


async def test_a_call_start_over_past_the_call_allowance_sets_nothing_aside(
    rig: Rig, voice_starts: None, mocker: MockerFixture
):
    taken = await _call_took_the_interview(rig)
    mocker.patch.object(
        voice,
        "enforce_voice_call_budget",
        AsyncMock(side_effect=HTTPException(status_code=429, detail="later")),
    )

    with pytest.raises(HTTPException) as refused:
        await voice.start_voice_call(USER, start_over=True)

    assert refused.value.status_code == 429
    assert rig.interviews.taken == taken


async def test_a_call_start_over_is_counted_on_the_start_budget(
    rig: Rig, voice_starts: None, mocker: MockerFixture
):
    await _call_took_the_interview(rig)
    mocker.patch.object(voice, "enforce_voice_call_budget", AsyncMock())
    starts = mocker.patch.object(voice, "enforce_voice_start_budget", AsyncMock())

    await voice.start_voice_call(USER, start_over=True)

    starts.assert_awaited_once_with(USER)


async def test_a_call_start_over_with_voice_off_is_not_found_and_sets_nothing_aside(
    rig: Rig, voice_starts: None, mocker: MockerFixture
):
    taken = await _call_took_the_interview(rig)
    mocker.patch.object(
        voice, "evaluate_feature_flag", AsyncMock(return_value=(False, False))
    )

    with pytest.raises(HTTPException) as refused:
        await voice.start_voice_call(USER, start_over=True)

    assert refused.value.status_code == 404
    assert rig.interviews.taken == taken


async def test_a_start_over_that_cannot_read_the_interview_refuses_and_sets_nothing_aside(
    rig: Rig,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    mocker: MockerFixture,
):
    taken = await _call_took_the_interview(rig)
    mocker.patch.object(
        interview,
        "find_onboarding_interview_session",
        AsyncMock(side_effect=ConnectionError("db down")),
    )

    with pytest.raises(HTTPException) as refused:
        await interview.start_onboarding_interview(USER, start_over=True)

    assert refused.value.status_code == 503
    assert rig.interviews.taken == taken


async def test_a_waiting_calls_words_never_reach_the_new_interview(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    dumps: DumpStore,
):
    """A call on a confirm arm whose words still wait for the person, beside
    a typed interview that closed: starting over gives its words up, so a
    late confirm, or the sweep, leaves the dump row alone."""
    stored.session.metadata.onboarding_interview_stage = INTERVIEW_STAGE_HANDED_OFF
    rig.interviews.typed = stored.session
    waiting = await rig.call(ending=PROFILE_CARD)

    await interview.start_onboarding_interview(USER, start_over=True)

    assert await locks.interview_of(rig.redis, waiting) == locks.INTERVIEW_SUPERSEDED
    answered = await _finish(waiting)
    assert answered.error_code == interview.SUPERSEDED
    assert dumps.row is None


async def test_an_earlier_take_no_pipeline_ran_is_given_up(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    dumps: DumpStore,
):
    stored.session.metadata.onboarding_interview_stage = INTERVIEW_STAGE_HANDED_OFF
    rig.interviews.typed = stored.session
    take = f"interview-{stored.session.session_id}"
    await dumps.start_dump(USER, take, BrainDumpInputMode.typed)

    await interview.start_onboarding_interview(USER, start_over=True)

    assert dumps.row is not None and dumps.row.recordingId == take
    assert dumps.row.status == BrainDumpStatus.failed
    assert dumps.row.errorCode == interview.SUPERSEDED
