"""Starting a closed onboarding interview over, from its ending screen.

Run on the hand-off rig (``voice_handoff_test``): calls and their hand-offs
on Redis, the typed interview in the chat store's stand-in, and the take on
the in-memory dump table, so these assert which take the plan step reads.
"""

import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import BackgroundTasks, HTTPException
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import interview, rate_limit, service, voice
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
from backend.util.settings import BehaveAs
from backend.voice_relay import locks, tokens

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
def running_turns(mocker: MockerFixture) -> AsyncMock:
    """The person's turns running at once: none unless a test says
    otherwise."""
    return mocker.patch.object(
        interview, "count_running_turns", new=AsyncMock(return_value=0)
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

    started = await interview.start_onboarding_interview(USER, start_over=call_id)

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
    """A typed interview closed and not yet handed off: nothing of it ran,
    and its hand-off can never run after the start-over."""
    stored.session.metadata.onboarding_interview_stage = INTERVIEW_STAGE_CLOSED
    stored.session.metadata.onboarding_interview_ending = PROFILE_CARD
    rig.interviews.typed = stored.session

    await interview.start_onboarding_interview(
        USER, start_over=stored.session.session_id
    )

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

    await interview.start_onboarding_interview(
        USER, start_over=stored.session.session_id
    )

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

    started = await voice.start_voice_call(USER, start_over=stored.session.session_id)

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

    await voice.start_voice_call(USER, start_over=first)

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
            await interview.start_onboarding_interview(USER, start_over=taken)
        else:
            await voice.start_voice_call(USER, start_over=taken)

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
        await interview.start_onboarding_interview(USER, start_over=taken)

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
        await voice.start_voice_call(USER, start_over=taken)

    assert refused.value.status_code == 429
    assert rig.interviews.taken == taken


async def test_a_call_start_over_is_counted_on_the_start_budget(
    rig: Rig, voice_starts: None, mocker: MockerFixture
):
    taken = await _call_took_the_interview(rig)
    mocker.patch.object(voice, "enforce_voice_call_budget", AsyncMock())
    starts = mocker.patch.object(voice, "enforce_voice_start_budget", AsyncMock())

    await voice.start_voice_call(USER, start_over=taken)

    starts.assert_awaited_once_with(USER)


async def test_a_call_start_over_with_voice_off_is_not_found_and_sets_nothing_aside(
    rig: Rig, voice_starts: None, mocker: MockerFixture
):
    taken = await _call_took_the_interview(rig)
    mocker.patch.object(
        voice, "evaluate_feature_flag", AsyncMock(return_value=(False, False))
    )

    with pytest.raises(HTTPException) as refused:
        await voice.start_voice_call(USER, start_over=taken)

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
        await interview.start_onboarding_interview(USER, start_over=taken)

    assert refused.value.status_code == 503
    assert rig.interviews.taken == taken


async def test_a_waiting_calls_words_never_reach_the_new_interview(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    dumps: DumpStore,
):
    """A call whose words are not handed off yet, beside a typed interview
    that closed: starting over gives its words up, so a late finish, or the
    sweep, leaves the dump row alone."""
    stored.session.metadata.onboarding_interview_stage = INTERVIEW_STAGE_HANDED_OFF
    rig.interviews.typed = stored.session
    waiting = await rig.call(ending=PROFILE_CARD)

    await interview.start_onboarding_interview(
        USER, start_over=stored.session.session_id
    )

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

    await interview.start_onboarding_interview(
        USER, start_over=stored.session.session_id
    )

    assert dumps.row is not None and dumps.row.recordingId == take
    assert dumps.row.status == BrainDumpStatus.failed
    assert dumps.row.errorCode == interview.SUPERSEDED


# ---------- bound to the interview the person saw end ----------


async def test_a_typed_start_over_of_an_interview_no_longer_the_closed_one_starts_as_usual(
    rig: Rig, opening: dict[str, MagicMock], allowance: AsyncMock
):
    taken = await _call_took_the_interview(rig)

    with pytest.raises(HTTPException) as refused:
        await interview.start_onboarding_interview(USER, start_over="an-earlier-call")

    assert refused.value.status_code == 409
    assert refused.value.detail == {"code": INTERVIEW_CLOSED, "call_id": taken}
    assert rig.interviews.taken == taken
    opening["open"].assert_not_awaited()


async def test_a_call_start_over_of_an_interview_no_longer_the_closed_one_starts_as_usual(
    rig: Rig, voice_starts: None, mocker: MockerFixture
):
    taken = await _call_took_the_interview(rig)
    mocker.patch.object(voice, "enforce_voice_call_budget", AsyncMock())

    with pytest.raises(HTTPException) as refused:
        await voice.start_voice_call(USER, start_over="an-earlier-call")

    assert refused.value.status_code == 409
    assert refused.value.detail == {"code": INTERVIEW_CLOSED, "call_id": taken}
    assert rig.interviews.taken == taken


# ---------- every refusal comes before anything is set aside ----------


async def test_a_typed_start_over_at_the_running_turn_cap_sets_nothing_aside(
    rig: Rig,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    running_turns: AsyncMock,
):
    taken = await _call_took_the_interview(rig)
    running_turns.return_value = 10_000

    with pytest.raises(HTTPException) as refused:
        await interview.start_onboarding_interview(USER, start_over=taken)

    assert refused.value.status_code == 429
    assert rig.interviews.taken == taken
    opening["open"].assert_not_awaited()


async def test_a_call_start_over_whose_room_cannot_be_registered_sets_nothing_aside(
    rig: Rig, voice_starts: None, mocker: MockerFixture
):
    taken = await _call_took_the_interview(rig)
    mocker.patch.object(voice, "enforce_voice_call_budget", AsyncMock())
    mocker.patch.object(
        voice.locks,
        "register_room",
        AsyncMock(side_effect=HTTPException(status_code=503, detail="later")),
    )

    with pytest.raises(HTTPException) as refused:
        await voice.start_voice_call(USER, start_over=taken)

    assert refused.value.status_code == 503
    assert rig.interviews.taken == taken


# ---------- a call taking the interview meanwhile ----------


def _closed_not_handed_off(rig: Rig, stored: StoredInterview) -> None:
    stored.session.metadata.onboarding_interview_stage = INTERVIEW_STAGE_CLOSED
    stored.session.metadata.onboarding_interview_ending = PROFILE_CARD
    rig.interviews.typed = stored.session


async def test_a_call_that_took_the_interview_after_the_read_refuses_the_start_over(
    rig: Rig, stored: StoredInterview
):
    """The read saw the call's words unclaimed; the call's hand-off claimed
    the interview before the start-over did. The claim that stands decides,
    and nothing is set aside."""
    _closed_not_handed_off(rig, stored)
    waiting = await rig.call(ending=PROFILE_CARD)
    plan = await interview.read_start_over(USER, stored.session.session_id)
    assert plan is not None and plan.unclaimed_calls == (waiting,)

    assert (
        await locks.claim_interview(rig.redis, waiting, locks.INTERVIEW_TAKEN)
        == locks.INTERVIEW_TAKEN
    )
    with pytest.raises(HTTPException) as refused:
        await interview.apply_start_over(USER, plan)

    assert refused.value.status_code == 409
    assert refused.value.detail == {"code": INTERVIEW_CLOSED, "call_id": waiting}
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_CLOSED
    )
    assert rig.interviews.typed is stored.session
    assert rig.interviews.latest_call == waiting
    assert await locks.interview_of(rig.redis, waiting) == locks.INTERVIEW_TAKEN


async def test_a_call_whose_words_are_taking_the_interview_refuses_the_start_over(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
):
    """A call's hand-off claimed the interview and has not yet written it on
    the onboarding row: the start-over is refused, so the row write that
    follows can never land on the new interview."""
    _closed_not_handed_off(rig, stored)
    taking = await rig.call(ending=PROFILE_CARD)
    await locks.claim_interview(rig.redis, taking, locks.INTERVIEW_TAKEN)
    assert (await rig.read(taking)).hand_off is None
    assert rig.interviews.taken is None

    with pytest.raises(HTTPException) as refused:
        await interview.start_onboarding_interview(
            USER, start_over=stored.session.session_id
        )

    assert refused.value.status_code == 409
    assert refused.value.detail == {"code": INTERVIEW_CLOSED, "call_id": taking}
    assert rig.interviews.typed is stored.session
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_CLOSED
    )
    opening["open"].assert_not_awaited()


# ---------- a call not yet settled ----------


async def _call_still_running(rig: Rig, status: str | None = locks.LIVE) -> str:
    """A call whose record still says it starts or runs, with no lock or
    place in the live count behind it: its relay went away and the sweep has
    not settled it yet."""
    call_id = tokens.new_call_id()
    fields: dict[str, object] = {
        "uid": USER,
        "started": time.time() - 60,
        "prev": rig.interviews.latest_call,
    }
    if status is not None:
        fields["status"] = status
    await locks.write_call_record(rig.redis, call_id, fields)
    rig.interviews.latest_call = call_id
    return call_id


@pytest.mark.parametrize("status", [None, locks.CONNECTING, locks.LIVE])
async def test_a_start_over_while_a_call_is_not_settled_is_refused_until_it_settles(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    status: str | None,
):
    _closed_not_handed_off(rig, stored)
    running = await _call_still_running(rig, status)

    with pytest.raises(HTTPException) as refused:
        await interview.start_onboarding_interview(
            USER, start_over=stored.session.session_id
        )

    assert refused.value.status_code == 503
    assert rig.interviews.typed is stored.session
    assert rig.interviews.latest_call == running
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_CLOSED
    )
    assert await locks.interview_of(rig.redis, running) is None
    opening["open"].assert_not_awaited()

    await rig.no_words(running, "no_usable_speech")
    await interview.start_onboarding_interview(
        USER, start_over=stored.session.session_id
    )

    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_HANDED_OFF
    )
    assert rig.interviews.latest_call is None
    assert await locks.interview_of(rig.redis, running) == locks.INTERVIEW_SUPERSEDED


async def test_applying_a_plan_that_names_a_call_not_settled_sets_nothing_aside(
    rig: Rig, stored: StoredInterview
):
    _closed_not_handed_off(rig, stored)
    settled = await rig.call(ending=PROFILE_CARD)
    running = await _call_still_running(rig)
    plan = interview.StartOver(
        session_id=stored.session.session_id,
        latest_call=running,
        unclaimed_calls=(running, settled),
    )

    with pytest.raises(HTTPException) as refused:
        await interview.apply_start_over(USER, plan)

    assert refused.value.status_code == 503
    assert await locks.interview_of(rig.redis, settled) is None
    assert await locks.interview_of(rig.redis, running) is None
    assert rig.interviews.typed is stored.session
    assert rig.interviews.latest_call == running


# ---------- the retired take is recorded before anything is set aside ----------


async def test_a_start_over_whose_retired_take_cannot_be_recorded_sets_nothing_aside_and_a_retry_goes_ahead(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    dumps: DumpStore,
    jobs: dict[str, AsyncMock],
    mocker: MockerFixture,
):
    rig.interviews.typed = stored.session
    await _close_typed(stored)
    earlier = f"interview-{stored.session.session_id}"
    assert dumps.row is not None and dumps.row.recordingId == earlier
    dumps.row.status = BrainDumpStatus.extracting
    fresh = _next_typed_interview(mocker, opening, rig.interviews)
    unwritable = mocker.patch.object(
        rig.redis, "set", AsyncMock(side_effect=ConnectionError("redis down"))
    )

    with pytest.raises(HTTPException) as refused:
        await interview.start_onboarding_interview(
            USER, start_over=stored.session.session_id
        )

    assert refused.value.status_code == 503
    assert rig.interviews.typed is stored.session
    assert dumps.row.recordingId == earlier
    assert dumps.row.status == BrainDumpStatus.extracting
    opening["open"].assert_not_awaited()

    mocker.stop(unwritable)
    await interview.start_onboarding_interview(
        USER, start_over=stored.session.session_id
    )

    assert rig.interviews.typed is fresh.session
    assert await rig.redis.get(interview.retired_take_key(USER)) == earlier
    await _type(fresh)
    await _close_typed(fresh)
    assert dumps.row.recordingId == f"interview-{fresh.session.session_id}"
    assert dumps.row.status == BrainDumpStatus.completed


async def test_a_start_over_after_a_corrected_interview_retires_its_take(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    dumps: DumpStore,
    jobs: dict[str, AsyncMock],
    mocker: MockerFixture,
):
    """A profile card's edits re-ran the take under its own id; starting over
    retires that take, and the new interview's take replaces it."""
    rig.interviews.typed = stored.session
    await _close_typed(stored)
    earlier = f"interview-{stored.session.session_id}"
    await _close_typed(stored, corrections="Also: we do snow removal")
    assert dumps.row is not None and dumps.row.recordingId == earlier
    assert dumps.row.transcript.endswith("Also: we do snow removal")
    # The corrected take is still in its second run.
    dumps.row.status = BrainDumpStatus.extracting
    fresh = _next_typed_interview(mocker, opening, rig.interviews)

    await interview.start_onboarding_interview(
        USER, start_over=stored.session.session_id
    )

    assert rig.interviews.typed is fresh.session
    assert await rig.redis.get(interview.retired_take_key(USER)) == earlier
    await _type(fresh)
    await _close_typed(fresh)
    assert dumps.row.recordingId == f"interview-{fresh.session.session_id}"
    assert dumps.row.status == BrainDumpStatus.completed


async def test_a_start_over_retires_the_take_on_the_row_when_it_is_applied(
    rig: Rig, stored: StoredInterview, dumps: DumpStore
):
    """A take that lands between the read of a start-over and its writes,
    such as the closed interview's own hand-off finishing, is the one
    retired."""
    rig.interviews.typed = stored.session
    stored.session.metadata.onboarding_interview_stage = INTERVIEW_STAGE_HANDED_OFF
    plan = await interview.read_start_over(USER, stored.session.session_id)
    assert plan is not None
    landed = f"interview-{stored.session.session_id}"
    await dumps.start_dump(USER, landed, BrainDumpInputMode.typed)

    await interview.apply_start_over(USER, plan)

    assert await rig.redis.get(interview.retired_take_key(USER)) == landed
    assert dumps.row is not None and dumps.row.status == BrainDumpStatus.failed
    assert dumps.row.errorCode == interview.SUPERSEDED


# ---------- the new take, never the old one ----------


@pytest.mark.parametrize(
    "status", [BrainDumpStatus.transcribed, BrainDumpStatus.extracting]
)
async def test_a_new_typed_interview_replaces_a_retired_take_still_in_the_pipeline(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    dumps: DumpStore,
    jobs: dict[str, AsyncMock],
    mocker: MockerFixture,
    status: BrainDumpStatus,
):
    rig.interviews.typed = stored.session
    await _close_typed(stored)
    earlier = f"interview-{stored.session.session_id}"
    assert dumps.row is not None and dumps.row.recordingId == earlier
    dumps.row.status = status
    fresh = _next_typed_interview(mocker, opening, rig.interviews)

    await interview.start_onboarding_interview(
        USER, start_over=stored.session.session_id
    )
    assert dumps.row.recordingId == earlier

    await _type(fresh)
    response = await _close_typed(fresh)

    assert response.status == BrainDumpStatus.transcribed
    assert dumps.row.recordingId == f"interview-{fresh.session.session_id}"
    assert dumps.row.status == BrainDumpStatus.completed
    assert fresh.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_HANDED_OFF
    )


async def test_a_call_replaces_a_retired_take_still_in_the_pipeline(
    rig: Rig,
    stored: StoredInterview,
    voice_starts: None,
    dumps: DumpStore,
    allowance: AsyncMock,
    mocker: MockerFixture,
):
    rig.interviews.typed = stored.session
    await _close_typed(stored)
    assert dumps.row is not None
    dumps.row.status = BrainDumpStatus.extracting
    mocker.patch.object(
        voice,
        "find_onboarding_interview_session",
        AsyncMock(side_effect=lambda user_id: rig.interviews.typed),
    )
    mocker.patch.object(voice, "enforce_voice_call_budget", AsyncMock())

    await voice.start_voice_call(USER, start_over=stored.session.session_id)
    call_id = await rig.call()
    record = await rig.hand_off(call_id)

    assert record.interview == locks.INTERVIEW_TAKEN
    assert dumps.row.recordingId == f"interview-{call_id}"


async def test_a_take_given_up_while_its_quality_check_runs_is_never_processed(
    dumps: DumpStore,
    quality_gate: AsyncMock,
    extraction: dict[str, AsyncMock],
    jobs: dict[str, AsyncMock],
):
    take = "interview-earlier"

    async def given_up_meanwhile(*args, **kwargs):
        assert await service.give_up_unrun_take(USER, take, interview.SUPERSEDED)
        return None

    quality_gate.side_effect = given_up_meanwhile
    background = BackgroundTasks()

    await service.finalize_typed_take(
        USER, take, "I run a bakery and chase supplier invoices.", background
    )
    await background()

    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.failed
    assert dumps.row.errorCode == interview.SUPERSEDED
    assert BrainDumpStatus.extracting not in dumps.statuses
    extraction["upsert_business_understanding"].assert_not_awaited()
    jobs["greeting"].assert_not_awaited()


# ---------- a skip never replaces a closed interview's take ----------


async def test_a_skip_over_a_closed_typed_interviews_take_is_refused_and_the_take_stays(
    rig: Rig,
    stored: StoredInterview,
    dumps: DumpStore,
    allowance: AsyncMock,
    jobs: dict[str, AsyncMock],
):
    rig.interviews.typed = stored.session
    await _close_typed(stored)
    assert dumps.row is not None
    kept = (dumps.row.recordingId, dumps.row.status, dumps.row.inputMode)

    with pytest.raises(HTTPException) as refused:
        await interview.refuse_a_skip_over_a_finished_take(USER)

    assert refused.value.status_code == 409
    assert refused.value.detail == {"code": INTERVIEW_CLOSED}
    assert (dumps.row.recordingId, dumps.row.status, dumps.row.inputMode) == kept


async def test_a_skip_over_a_calls_take_is_refused_naming_the_call(
    rig: Rig, dumps: DumpStore
):
    taken = await _call_took_the_interview(rig)

    with pytest.raises(HTTPException) as refused:
        await interview.refuse_a_skip_over_a_finished_take(USER)

    assert refused.value.status_code == 409
    assert refused.value.detail == {"code": INTERVIEW_CLOSED, "call_id": taken}
    assert dumps.row is not None and dumps.row.recordingId == f"interview-{taken}"


async def test_a_skip_once_the_interview_was_started_over_is_not_refused(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    dumps: DumpStore,
    jobs: dict[str, AsyncMock],
    mocker: MockerFixture,
):
    rig.interviews.typed = stored.session
    await _close_typed(stored)
    _next_typed_interview(mocker, opening, rig.interviews)
    await interview.start_onboarding_interview(
        USER, start_over=stored.session.session_id
    )

    await interview.refuse_a_skip_over_a_finished_take(USER)


async def test_a_skip_with_an_unreadable_interview_is_refused(
    rig: Rig, dumps: DumpStore, mocker: MockerFixture
):
    await _call_took_the_interview(rig)
    mocker.patch.object(
        interview,
        "call_that_took_the_interview",
        AsyncMock(side_effect=ConnectionError("db down")),
    )

    with pytest.raises(HTTPException) as refused:
        await interview.refuse_a_skip_over_a_finished_take(USER)

    assert refused.value.status_code == 503


# ---------- the person's turn budget spans their interviews ----------


class _MidHour(datetime):
    """The rate limiter's clock, held halfway through an hour."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 9, 12, 30, tzinfo=tz or UTC)


async def test_a_fresh_interview_after_a_start_over_spends_the_same_turn_budget(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    mocker: MockerFixture,
    set_platform_payment,
):
    mocker.patch.object(rate_limit.settings.config, "behave_as", BehaveAs.CLOUD)
    set_platform_payment(True)
    mocker.patch.object(rate_limit, "datetime", _MidHour)
    mocker.patch.object(
        rate_limit, "get_redis_async", new=AsyncMock(return_value=rig.redis)
    )
    mocker.patch.object(
        interview,
        "check_interview_turn_budget",
        new=rate_limit.check_interview_turn_budget,
    )
    stored.session.metadata.onboarding_interview_stage = INTERVIEW_STAGE_HANDED_OFF
    rig.interviews.typed = stored.session
    for _ in range(rate_limit.INTERVIEW_HOURLY_TURNS - 1):
        await rate_limit.enforce_interview_turn_budget(USER)

    async def opened(user_id: str):
        rig.interviews.typed = opening["session"]
        return opening["session"]

    opening["open"].side_effect = opened

    started = await interview.start_onboarding_interview(
        USER, start_over=stored.session.session_id
    )

    assert started.turn_id == "turn-1"
    with pytest.raises(HTTPException) as refused:
        await interview.enforce_onboarding_interview_allowance(USER)
    assert refused.value.status_code == 429
