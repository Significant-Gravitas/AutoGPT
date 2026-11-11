"""The REST API's half of a call's post-call work: the hand-off of its
transcript of record to the person's interview, what starts it, and which
take owns the interview.

The relay leaves a call ``transcribed``; these start from there. The
hand-off runs for real into the onboarding pipeline over the in-memory dump
table the pipeline tests use, so these assert what reaches the take.
"""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from fastapi import BackgroundTasks, HTTPException
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import (
    interview,
    quality,
    service,
    voice,
    voice_handoff,
)
from backend.api.features.onboarding_dump.interview_endings import InterviewEnding
from backend.api.features.onboarding_dump.interview_test import (
    StoredInterview,
    allowance,
    jobs,
    opening,
    stored,
)
from backend.api.features.onboarding_dump.models import (
    MAX_CORRECTION_CHARS,
    FinalizeResponse,
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
from backend.copilot.model import (
    INTERVIEW_STAGE_CLOSED,
    INTERVIEW_STAGE_HANDED_OFF,
    ONBOARDING_INTERVIEW_KIND,
    ChatSession,
)
from backend.copilot.onboarding_interview import INTERVIEW_CLOSED
from backend.voice_relay import locks, tokens
from backend.voice_relay.conftest import MemoryRecordings, ScriptedTranscriber
from backend.voice_relay.locks import TurnMark
from backend.voice_relay.postcall import PostCall
from backend.voice_relay.postcall_test import answer, recording_of
from backend.voice_relay.record import wav_of

__all__ = [
    "allowance",
    "dumps",
    "extraction",
    "jobs",
    "opening",
    "personalization_admission",
    "quality_gate",
    "storage_mocks",
    "stored",
    "transcribe",
]

USER = "user-1"
SAID = ["I run a bakery with five staff.", "Answering order emails, mostly."]
LIVEKIT_SECRET = "r" * 48  # pragma: allowlist secret


@pytest.fixture
async def redis():
    client = FakeRedis(server=FakeServer(), decode_responses=True)
    yield client
    await client.aclose()


class Interviews:
    """What the person's onboarding row and chat store hold of their
    interviews."""

    def __init__(self) -> None:
        self.latest_call: str | None = None
        # The call that took the interview, as the onboarding row keeps it.
        self.taken: str | None = None
        self.typed: ChatSession | None = None


@pytest.fixture(autouse=True)
def interviews(mocker: MockerFixture, redis) -> Interviews:
    state = Interviews()

    async def latest_call(user_id: str) -> str | None:
        return state.latest_call

    async def typed(user_id: str) -> ChatSession | None:
        return state.typed

    async def taken(user_id: str) -> str | None:
        return state.taken

    async def take(user_id: str, call_id: str) -> None:
        state.taken = call_id

    mocker.patch.object(
        interview,
        "get_onboarding_voice_interview_call_id",
        new=AsyncMock(side_effect=taken),
    )
    mocker.patch.object(
        voice_handoff,
        "record_onboarding_voice_interview_call",
        new=AsyncMock(side_effect=take),
    )
    for module in (interview, voice):
        mocker.patch.object(
            module,
            "get_onboarding_voice_call_id",
            new=AsyncMock(side_effect=latest_call),
        )
    mocker.patch.object(
        interview,
        "find_onboarding_interview_session",
        new=AsyncMock(side_effect=typed),
    )
    for module in (interview, voice, voice_handoff):
        mocker.patch.object(module, "get_redis_async", AsyncMock(return_value=redis))
    return state


@pytest.fixture(autouse=True)
def typed_arm(mocker: MockerFixture) -> AsyncMock:
    """The person's arm of the typed ending experiment: none. A call's
    hand-off reads no flag: its arm comes with the call's record."""

    async def arm(user_id: str, mode: str) -> str | None:
        assert mode == "text", "a call's hand-off read a flag"
        return None

    return mocker.patch.object(
        interview, "assigned_ending", new=AsyncMock(side_effect=arm)
    )


@pytest.fixture(autouse=True)
def replica(monkeypatch) -> None:
    """A replica serving, with no hand-off of its own running yet."""
    monkeypatch.setattr(voice_handoff, "_stopping", False)
    monkeypatch.setattr(voice_handoff, "_running", {})


class Rig:
    def __init__(self, redis, interviews: Interviews) -> None:
        self.redis = redis
        self.interviews = interviews

    async def call(
        self,
        said: str | None = SAID[0],
        *,
        end_reason: str = "finished",
        ending: str | None = None,
        first_task: str | None = None,
        transcribed_at: float | None = None,
        went_live: bool = True,
    ) -> str:
        """A call as the relay leaves it: ``transcribed`` with *said* as its
        transcript of record, pending its hand-off, the person's latest, and
        naming the call it replaced there. With *said* None, its transcript
        is still being made; with *went_live* False, its room or its model
        session never opened."""
        call_id = tokens.new_call_id()
        await locks.write_call_record(
            self.redis,
            call_id,
            {
                "uid": USER,
                "status": locks.TRANSCRIBING,
                "started": time.time() - 600,
                "connect_s": 1.4 if went_live else None,
                "end_reason": end_reason,
                "ending": ending,
                "first_task": first_task,
                "prev": self.interviews.latest_call,
            },
        )
        await locks.add_postcall_pending(self.redis, call_id)
        self.interviews.latest_call = call_id
        if said is not None:
            await self.transcribed(call_id, said, at=transcribed_at)
        return call_id

    async def transcribed(
        self, call_id: str, said: str = SAID[0], *, at: float | None = None
    ) -> None:
        """The relay's transcript of the call lands."""
        await locks.store_transcript(
            self.redis, call_id, said, uid=USER, from_upload=False, now=at
        )

    async def no_words(self, call_id: str, code: str) -> None:
        """The call settles without words: the relay, or the hand-off,
        records *code*."""
        await locks.record_answer(
            self.redis,
            call_id,
            uid=USER,
            status=locks.FAILED,
            answer=voice_handoff.failed_hand_off(code),
        )
        await locks.remove_postcall_pending(self.redis, call_id)

    async def backoff_passed(self, call_id: str) -> None:
        """The wait after the call's last failed hand-off attempt is over."""
        await locks.write_call_record(
            self.redis, call_id, {"handoff_retry_at": time.time() - 1}
        )

    async def hand_off(
        self, call_id: str, *, confirmed: bool = False
    ) -> locks.CallRecord:
        """Hand the call off as a status read or the sweep does."""
        await voice_handoff.hand_off_owed(call_id, trigger="test", confirmed=confirmed)
        return await self.read(call_id)

    async def read(self, call_id: str) -> locks.CallRecord:
        record = await locks.read_call(self.redis, call_id)
        assert record is not None
        return record


@pytest.fixture
def rig(redis, interviews: Interviews) -> Rig:
    return Rig(redis, interviews)


async def _finish(call_id: str, **kwargs):
    background = BackgroundTasks()
    answered = await voice.finish_call(USER, call_id, background, **kwargs)
    await background()
    return answered


async def _type(stored: StoredInterview) -> None:
    """A typed message of the person's, admitted and then dispatched, as the
    chat routes take it."""
    await interview.admit_onboarding_interview_turn(stored.session, USER)
    await interview.typed_message_accepted(USER)


def _accepted() -> interview.InterviewClosure:
    return interview.InterviewClosure(
        ending=None,
        hand_off=FinalizeResponse(
            status=BrainDumpStatus.transcribed, input_mode=BrainDumpInputMode.typed
        ),
    )


class Judge:
    """The quality gate, holding its first ruling until released: a hand-off
    inside the judge, for a test to cut or race."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.judged = 0

    async def rule(self, *args, **kwargs) -> None:
        """Pass the words, once released; counts the rulings made."""
        if not self.entered.is_set():
            self.entered.set()
            await self.release.wait()
        self.judged += 1


# ---------- what starts a hand-off ----------


async def test_a_calls_take_reaches_the_pipeline_as_spoken_words(
    rig: Rig,
    dumps: DumpStore,
    quality_gate: AsyncMock,
    extraction: dict[str, AsyncMock],
):
    """The row, the judge's prompt, the profile's note and the call's answer
    all say ``voice``: the words were said, not typed. The answer counts the
    words it judged."""
    call_id = await rig.call()

    record = await rig.hand_off(call_id)

    assert dumps.row is not None and dumps.row.inputMode == BrainDumpInputMode.voice
    assert quality_gate.await_args is not None
    assert quality_gate.await_args.kwargs["input_mode"] == BrainDumpInputMode.voice
    upsert = extraction["upsert_business_understanding"]
    assert upsert.await_args is not None
    _, understanding = upsert.await_args.args
    assert understanding.additional_notes == (
        f"Onboarding brain dump (spoken): {SAID[0]}"
    )
    assert record.hand_off is not None
    assert record.hand_off.input_mode == BrainDumpInputMode.voice
    assert record.hand_off.transcript_words == quality.count_words(SAID[0])


async def test_a_status_read_of_an_owed_call_answers_handing_off_and_hands_it_off(
    rig: Rig, dumps: DumpStore
):
    call_id = await rig.call()

    read = await voice.call_status(USER, call_id)

    assert read.status == "handing_off"
    running = voice_handoff.settle_if_owed(await rig.read(call_id))
    assert running is not None
    await asyncio.wait_for(running, timeout=5)
    read = await voice.call_status(USER, call_id)
    assert read.status == locks.HANDED_OFF
    assert read.interview == locks.INTERVIEW_TAKEN
    assert dumps.row is not None and dumps.row.recordingId == f"interview-{call_id}"
    assert await locks.postcall_pending(rig.redis) == []


async def test_a_second_poll_finds_the_call_held_while_the_first_hands_it_off(
    rig: Rig, quality_gate: AsyncMock
):
    """The first hand-off is inside the judge: a second replica's read, and
    the sweep, find the call held, and the words are judged once."""
    judge = Judge()
    quality_gate.side_effect = judge.rule
    call_id = await rig.call()

    first = voice_handoff.settle_if_owed(await rig.read(call_id))
    assert first is not None
    # A second poll in this replica shares the running hand-off.
    assert voice_handoff.settle_if_owed(await rig.read(call_id)) is first
    await asyncio.wait_for(judge.entered.wait(), timeout=5)

    second = await voice_handoff.hand_off_owed(call_id, trigger="status")
    swept = await voice_handoff.sweep()
    judge.release.set()

    assert second == voice_handoff.HELD
    assert swept == []
    assert await asyncio.wait_for(first, timeout=5) == locks.HANDED_OFF
    assert judge.judged == 1


async def test_a_status_read_of_a_confirm_arms_call_starts_nothing(
    rig: Rig, dumps: DumpStore
):
    call_id = await rig.call(ending=InterviewEnding.PROFILE_CARD.value)

    read = await voice.call_status(USER, call_id)

    assert read.status == locks.TRANSCRIBED
    assert voice_handoff.settle_if_owed(await rig.read(call_id)) is None
    assert dumps.row is None


async def test_the_sweep_hands_off_an_owed_call_nobody_polled(
    rig: Rig, dumps: DumpStore
):
    call_id = await rig.call()

    assert await voice_handoff.sweep() == [call_id]

    assert (await rig.read(call_id)).status == locks.HANDED_OFF
    assert dumps.row is not None and dumps.row.recordingId == f"interview-{call_id}"


async def test_the_sweep_hands_off_a_confirm_arms_call_left_unconfirmed_an_hour(
    rig: Rig,
):
    arm = InterviewEnding.FIRST_TASK.value
    waiting = await rig.call(ending=arm, transcribed_at=time.time() - 600)
    await rig.hand_off(waiting)
    left = await rig.call(ending=arm, transcribed_at=time.time() - 3601)

    assert await voice_handoff.sweep() == [left]

    record = await rig.read(left)
    assert record.status == locks.HANDED_OFF
    # The call ran to its close, so its take is the interview's.
    assert record.interview == locks.INTERVIEW_TAKEN
    assert (await rig.read(waiting)).status == locks.TRANSCRIBED


async def test_an_unconfirmed_cut_off_call_handed_off_after_an_hour_takes_nothing(
    rig: Rig, dumps: DumpStore
):
    """The hour rule hands the words off as a confirm would, but it is not
    the person's own finish: a cut-off call still leaves the interview
    open."""
    call_id = await rig.call(
        end_reason="client_gone",
        ending=InterviewEnding.FIRST_TASK.value,
        transcribed_at=time.time() - 3601,
    )

    assert await voice_handoff.sweep() == [call_id]

    record = await rig.read(call_id)
    assert record.status == locks.HANDED_OFF
    assert record.accepted
    assert record.interview is None
    assert rig.interviews.taken is None
    assert dumps.row is not None and dumps.row.recordingId == f"interview-{call_id}"


async def test_a_hand_off_that_raises_is_tried_again_then_given_up(
    rig: Rig, mocker: MockerFixture, dumps: DumpStore
):
    """The finish route's "Try again" hands the kept words off again."""
    close = mocker.patch.object(
        voice_handoff,
        "close_voice_interview",
        new=AsyncMock(side_effect=RuntimeError("the pipeline fell over")),
    )
    call_id = await rig.call()

    outcomes = []
    for _ in range(3):
        outcomes.append(await voice_handoff.hand_off_owed(call_id, trigger="sweep"))
        await rig.backoff_passed(call_id)

    assert outcomes == [
        voice_handoff.RETRY,
        voice_handoff.RETRY,
        voice_handoff.GAVE_UP,
    ]
    record = await rig.read(call_id)
    assert record.status == locks.FAILED
    assert record.handoff_attempts == 3
    assert record.hand_off is not None
    assert record.hand_off.error_code == voice_handoff.PERSONALIZATION_UNAVAILABLE
    assert await locks.postcall_pending(rig.redis) == []
    assert await rig.redis.get(locks.postcall_claim_key(call_id)) is None

    mocker.stop(close)
    again = await _finish(call_id)

    assert again.status == BrainDumpStatus.transcribed
    assert dumps.row is not None and dumps.row.transcript == SAID[0]


async def test_a_failed_attempt_is_tried_again_only_once_its_backoff_passed(
    rig: Rig, mocker: MockerFixture
):
    close = mocker.patch.object(
        voice_handoff,
        "close_voice_interview",
        new=AsyncMock(side_effect=[RuntimeError("once"), _accepted()]),
    )
    call_id = await rig.call()

    assert await voice_handoff.hand_off_owed(call_id, trigger="status") == (
        voice_handoff.RETRY
    )
    record = await rig.read(call_id)
    assert record.handoff_retry_at is not None
    assert record.handoff_retry_at >= time.time() + voice_handoff.HANDOFF_RETRY_S - 5

    # Polls and the sweep start nothing before it is due.
    assert voice_handoff.settle_if_owed(record) is None
    assert await voice_handoff.sweep() == []
    assert await voice_handoff.hand_off_owed(call_id, trigger="status") == (
        voice_handoff.SETTLED
    )
    assert close.await_count == 1
    # The status still says the hand-off is under way.
    assert (await voice.call_status(USER, call_id)).status == "handing_off"

    await rig.backoff_passed(call_id)
    running = voice_handoff.settle_if_owed(await rig.read(call_id))
    assert running is not None
    assert await asyncio.wait_for(running, timeout=5) == locks.HANDED_OFF
    assert close.await_count == 2


async def test_a_refusal_for_now_is_recorded_for_the_finish_route_not_the_sweep(
    rig: Rig, personalization_admission: AsyncMock
):
    personalization_admission.side_effect = HTTPException(
        status_code=429, headers={"Retry-After": "60"}
    )
    call_id = await rig.call()

    record = await rig.hand_off(call_id)

    assert record.status == locks.FAILED
    assert record.hand_off is not None
    assert record.hand_off.error_code == voice_handoff.RATE_LIMITED
    assert record.transcript_of_record == SAID[0]
    assert await locks.postcall_pending(rig.redis) == []
    assert await voice_handoff.sweep() == []


async def test_a_settled_call_leaves_the_pending_set_and_a_waiting_one_stays(
    rig: Rig,
):
    settled = await rig.call()
    waiting = await rig.call(ending=InterviewEnding.PROFILE_CARD.value)

    await voice_handoff.record_hand_off(rig.redis, settled, _accepted(), uid=USER)
    await voice_handoff.record_hand_off(
        rig.redis,
        waiting,
        interview.InterviewClosure(
            ending=InterviewEnding.PROFILE_CARD.value, hand_off=None
        ),
        uid=USER,
    )

    assert await locks.postcall_pending(rig.redis) == [waiting]


async def test_an_answer_written_after_the_record_expired_names_its_owner(
    rig: Rig, mocker: MockerFixture
):
    call_id = await rig.call()
    judge = voice_handoff.close_voice_interview

    async def judged_then_expired(*args, **kwargs):
        closure = await judge(*args, **kwargs)
        await rig.redis.delete(locks.call_record_key(call_id))
        return closure

    mocker.patch.object(voice_handoff, "close_voice_interview", new=judged_then_expired)

    record = await rig.hand_off(call_id)

    assert record.uid == USER
    assert record.status == locks.HANDED_OFF


async def test_a_call_whose_record_names_no_owner_is_owed_nothing(rig: Rig):
    call_id = await rig.call()
    await rig.redis.hdel(locks.call_record_key(call_id), "uid")
    record = await rig.read(call_id)

    assert not voice_handoff.owed(record)
    assert voice_handoff.settle_if_owed(record) is None
    assert await voice_handoff.sweep() == []
    assert (await rig.read(call_id)).hand_off is None


async def test_a_hand_off_cut_inside_the_judge_is_judged_again_on_its_retry(
    rig: Rig, quality_gate: AsyncMock, jobs: dict[str, AsyncMock]
):
    """A REST pod stopped while the judge weighed the call's words: the row
    holds the take, claimed and never judged. The retry judges the words
    rather than reading the row as a ruling, and the pipeline's jobs run
    once."""
    judge = Judge()
    quality_gate.side_effect = judge.rule
    call_id = await rig.call()
    cut = asyncio.create_task(voice_handoff.hand_off_owed(call_id, trigger="status"))
    await asyncio.wait_for(judge.entered.wait(), timeout=5)

    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut

    record = await rig.read(call_id)
    assert record.hand_off is None
    assert await rig.redis.get(locks.postcall_claim_key(call_id)) is None

    judge.release.set()
    assert await voice_handoff.hand_off_owed(call_id, trigger="status") == (
        locks.HANDED_OFF
    )

    assert judge.judged == 1
    jobs["greeting"].assert_awaited_once_with(SAID[0])


async def test_a_hand_off_whose_answer_was_lost_after_its_jobs_ran_runs_none_again(
    rig: Rig, dumps: DumpStore, quality_gate: AsyncMock, jobs: dict[str, AsyncMock]
):
    """The pipeline judged the words and ran its jobs, and the answer on the
    call was lost: the hand-off runs again, and reads the row's ruling."""
    call_id = await rig.call()
    await rig.hand_off(call_id)
    await rig.redis.hdel(locks.call_record_key(call_id), "hand_off", "interview")
    await locks.write_call_record(rig.redis, call_id, {"status": locks.TRANSCRIBED})
    await locks.add_postcall_pending(rig.redis, call_id)

    record = await rig.hand_off(call_id)

    assert record.status == locks.HANDED_OFF
    assert record.took_interview
    assert quality_gate.await_count == 1
    jobs["greeting"].assert_awaited_once()
    assert dumps.row is not None and dumps.row.recordingId == f"interview-{call_id}"


async def test_a_hand_off_whose_claim_was_lost_writes_nothing(
    rig: Rig, mocker: MockerFixture, jobs: dict[str, AsyncMock]
):
    call_id = await rig.call()
    judge = voice_handoff.close_voice_interview

    async def judged_then_lost(*args, **kwargs):
        closure = await judge(*args, **kwargs)
        await rig.redis.delete(locks.postcall_claim_key(call_id))
        return closure

    mocker.patch.object(voice_handoff, "close_voice_interview", new=judged_then_lost)

    outcome = await voice_handoff.hand_off_owed(call_id, trigger="status")

    assert outcome == voice_handoff.LOST_CLAIM
    record = await rig.read(call_id)
    assert (record.hand_off, record.interview) == (None, None)
    assert rig.interviews.taken is None
    jobs["greeting"].assert_not_awaited()
    # Still pending, for the hand-off that holds the call now.
    assert await locks.postcall_pending(rig.redis) == [call_id]


async def test_a_long_hand_off_renews_its_claim(rig: Rig, mocker: MockerFixture):
    mocker.patch.object(voice_handoff, "HANDOFF_CLAIM_TTL_S", 1)
    mocker.patch.object(voice_handoff, "HANDOFF_RENEW_S", 0.2)
    call_id = await rig.call()
    judge = voice_handoff.close_voice_interview

    async def slow(*args, **kwargs):
        await asyncio.sleep(1.6)
        return await judge(*args, **kwargs)

    mocker.patch.object(voice_handoff, "close_voice_interview", new=slow)

    assert await voice_handoff.hand_off_owed(call_id, trigger="status") == (
        locks.HANDED_OFF
    )


async def test_the_sweep_runs_only_where_this_deployment_has_a_relay(
    mocker: MockerFixture, monkeypatch
):
    sweep = mocker.patch.object(voice_handoff, "sweep_forever", new=AsyncMock())
    config = voice_handoff.settings.config

    monkeypatch.setattr(config, "voice_relay_public_url", "")
    async with voice_handoff.sweeping():
        await asyncio.sleep(0)
    sweep.assert_not_called()

    monkeypatch.setattr(config, "voice_relay_public_url", "wss://api.example.com/voice")
    async with voice_handoff.sweeping():
        await asyncio.sleep(0)
    sweep.assert_awaited_once()


async def test_shutdown_waits_for_a_running_hand_off_and_its_jobs(
    rig: Rig, jobs: dict[str, AsyncMock], monkeypatch
):
    monkeypatch.setattr(voice_handoff.settings.config, "voice_relay_public_url", "")
    greeted = asyncio.Event()

    async def slow_greeting(*args, **kwargs):
        await asyncio.sleep(0.2)
        greeted.set()
        return ("Hi there.", [])

    jobs["greeting"].side_effect = slow_greeting
    call_id = await rig.call()

    async with voice_handoff.sweeping():
        running = voice_handoff.settle_if_owed(await rig.read(call_id))
        assert running is not None

    assert running.done() and not running.cancelled()
    assert greeted.is_set()
    # No hand-off starts once the replica is stopping.
    later = await rig.call(SAID[1])
    assert voice_handoff.settle_if_owed(await rig.read(later)) is None


async def test_shutdown_cancels_a_hand_off_that_outlasts_its_wait(
    rig: Rig, quality_gate: AsyncMock, monkeypatch
):
    monkeypatch.setattr(voice_handoff.settings.config, "voice_relay_public_url", "")
    monkeypatch.setattr(voice_handoff, "HANDOFF_SHUTDOWN_S", 0.1)
    judge = Judge()
    quality_gate.side_effect = judge.rule
    call_id = await rig.call()

    async with voice_handoff.sweeping():
        running = voice_handoff.settle_if_owed(await rig.read(call_id))
        assert running is not None
        await asyncio.wait_for(judge.entered.wait(), timeout=5)

    assert running.cancelled()
    # It gave its claim up, for the next replica's read or sweep.
    assert await rig.redis.get(locks.postcall_claim_key(call_id)) is None
    assert await locks.postcall_pending(rig.redis) == [call_id]


async def test_a_call_whose_hand_off_answered_is_owed_nothing(rig: Rig):
    call_id = await rig.call()
    await rig.hand_off(call_id)

    assert (
        await voice_handoff.hand_off_owed(call_id, trigger="status")
        == voice_handoff.SETTLED
    )


async def test_a_call_ended_on_the_relay_is_handed_off_by_a_status_read(
    redis, rig: Rig, dumps: DumpStore
):
    """Both halves: the relay makes the transcript, then the step's first
    status read in the REST API hands it off."""
    pcm, marks = recording_of(answer(), answer())
    recordings = MemoryRecordings()
    call_id = tokens.new_call_id()
    await locks.write_call_record(
        redis,
        call_id,
        {
            "uid": USER,
            "status": "ended",
            "end_reason": "finished",
            "turns": [
                TurnMark(index=index, at=at).model_dump()
                for index, at in enumerate(marks)
            ],
            "audio_path": await recordings.store(
                USER, wav_of(pcm), f"brain-dump-voice-{call_id}.wav"
            ),
        },
    )
    await locks.add_postcall_pending(redis, call_id)
    rig.interviews.latest_call = call_id

    async def get_redis():
        return redis

    relay = PostCall(
        redis=get_redis,
        recordings=recordings,
        transcriber=ScriptedTranscriber(SAID),
        first_name=AsyncMock(return_value="Dana"),
        live_model="gemini-3.8-live",
        pod="relay-pod",
    )
    assert await locks.claim_postcall(redis, call_id, relay.pod)
    await relay.run(call_id, pcm=pcm)
    assert (await voice.call_status(USER, call_id)).status == "handing_off"

    running = voice_handoff.settle_if_owed(await rig.read(call_id))
    assert running is not None
    await asyncio.wait_for(running, timeout=5)

    read = await voice.call_status(USER, call_id)
    assert read.status == locks.HANDED_OFF
    assert rig.interviews.taken == call_id
    assert dumps.row is not None and dumps.row.transcript == "\n".join(SAID)
    assert recordings.objects == {}


# ---------- the hand-off's answer ----------


async def test_an_accepted_take_hands_the_persons_words_off_and_takes_the_interview(
    rig: Rig, dumps: DumpStore, jobs: dict[str, AsyncMock]
):
    call_id = await rig.call("\n".join(SAID))

    record = await rig.hand_off(call_id)

    assert record.status == locks.HANDED_OFF
    assert record.hand_off is not None
    assert record.hand_off.status == BrainDumpStatus.transcribed
    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{call_id}"
    assert dumps.row.transcript == "\n".join(SAID)
    # The pipeline's own jobs ran on the same words.
    jobs["greeting"].assert_awaited_once_with("\n".join(SAID))
    assert await locks.postcall_pending(rig.redis) == []
    assert await interview.call_that_took_the_interview(USER) == call_id


async def test_the_hand_off_is_told_why_the_call_ended_and_the_task_it_agreed(
    rig: Rig, mocker: MockerFixture
):
    close = mocker.patch.object(
        voice_handoff,
        "close_voice_interview",
        new=AsyncMock(
            return_value=interview.InterviewClosure(ending=None, hand_off=None)
        ),
    )
    call_id = await rig.call(first_task="Answer order emails")

    await rig.hand_off(call_id)

    close.assert_awaited_once()
    assert close.await_args.args[:3] == (USER, call_id, SAID[0])
    assert close.await_args.kwargs["reason"] == "confirmed"
    assert close.await_args.kwargs["first_task"] == "Answer order emails"


async def test_a_call_where_nothing_was_said_leaves_the_interview_open_for_another(
    rig: Rig, dumps: DumpStore
):
    first = await rig.call("")

    record = await rig.hand_off(first)

    assert record.status == locks.FAILED
    assert record.hand_off is not None
    assert record.hand_off.error_code == quality.NO_USABLE_SPEECH
    assert record.hand_off.input_mode == BrainDumpInputMode.voice
    assert dumps.row is None
    assert await locks.postcall_pending(rig.redis) == []
    assert await interview.call_that_took_the_interview(USER) is None

    second = await rig.call()
    record = await rig.hand_off(second)

    assert record.status == locks.HANDED_OFF
    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{second}"


async def test_a_verdict_on_the_words_fails_the_call_and_leaves_the_interview_open(
    rig: Rig, quality_gate: AsyncMock
):
    quality_gate.return_value = quality.QualityRejection(
        error_code=quality.INSUFFICIENT_CONTENT, reason="judge_unusable"
    )
    call_id = await rig.call("idk")

    record = await rig.hand_off(call_id)

    assert record.status == locks.FAILED
    assert record.hand_off is not None
    assert record.hand_off.error_code == quality.INSUFFICIENT_CONTENT
    assert await interview.call_that_took_the_interview(USER) is None


async def test_a_take_the_judge_never_ruled_on_is_handed_off_again_by_the_finish_route(
    rig: Rig, quality_gate: AsyncMock, dumps: DumpStore
):
    quality_gate.side_effect = [
        quality.QualityRejection(
            error_code=quality.QUALITY_CHECK_UNAVAILABLE, reason="judge_error"
        ),
        None,
    ]
    call_id = await rig.call()

    record = await rig.hand_off(call_id)

    assert record.status == locks.FAILED
    assert record.hand_off is not None
    assert record.hand_off.error_code == quality.QUALITY_CHECK_UNAVAILABLE
    # The words stay on the call for the retry.
    assert record.transcript_of_record == SAID[0]
    assert await interview.call_that_took_the_interview(USER) is None

    again = await _finish(call_id)

    assert again.status == BrainDumpStatus.transcribed
    assert quality_gate.await_count == 2
    record = await rig.read(call_id)
    assert record.status == locks.HANDED_OFF
    # The person's finish makes the take the interview's.
    assert record.took_interview
    assert rig.interviews.taken == call_id
    assert dumps.row is not None and dumps.row.transcript == SAID[0]


async def test_a_confirm_arm_waits_and_the_finish_route_completes_the_hand_off(
    rig: Rig, dumps: DumpStore
):
    """The arm is the one the call carries from its start; the hand-off reads
    no flag (``typed_arm``)."""
    call_id = await rig.call(ending=InterviewEnding.PROFILE_CARD.value)

    assert (await rig.hand_off(call_id)).status == locks.TRANSCRIBED
    assert dumps.row is None

    confirmed = await _finish(call_id, corrections="Eight staff, not five.")

    assert confirmed.status == BrainDumpStatus.transcribed
    assert dumps.row is not None
    assert dumps.row.transcript == f"{SAID[0]}\nEight staff, not five."
    record = await rig.read(call_id)
    assert record.status == locks.HANDED_OFF
    # The person's finish makes the take the interview's.
    assert record.took_interview
    assert rig.interviews.taken == call_id
    assert await locks.postcall_pending(rig.redis) == []


async def test_a_call_the_person_has_called_again_since_is_superseded(
    rig: Rig, dumps: DumpStore
):
    """The later call has words of record: its take is the newer one."""
    call_id = await rig.call()
    await rig.call(SAID[1])

    record = await rig.hand_off(call_id)

    assert record.status == locks.HANDED_OFF
    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    # The step reads the field first: it says so too.
    assert record.interview == locks.INTERVIEW_SUPERSEDED
    assert dumps.row is None


async def test_a_call_after_the_typed_interview_ended_is_superseded(
    rig: Rig, dumps: DumpStore
):
    typed = ChatSession.new(USER, dry_run=False)
    typed.metadata.kind = ONBOARDING_INTERVIEW_KIND
    typed.metadata.onboarding_interview_stage = INTERVIEW_STAGE_HANDED_OFF
    rig.interviews.typed = typed
    call_id = await rig.call()

    record = await rig.hand_off(call_id)

    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    assert record.interview == locks.INTERVIEW_SUPERSEDED
    assert dumps.row is None


async def test_a_call_whose_recording_was_lost_is_answered_as_it_stands(rig: Rig):
    """The relay recorded ``recording_unavailable``: no words to hand off."""
    call_id = await rig.call(None)
    await rig.no_words(call_id, locks.RECORDING_UNAVAILABLE)

    answered = await _finish(call_id)

    assert answered.error_code == locks.RECORDING_UNAVAILABLE
    assert (await voice.call_status(USER, call_id)).status == locks.FAILED


# ---------- which take owns the interview ----------

RAN_TO_ITS_CLOSE = (
    "finished",
    "wall_clock",
    "turn_cap",
    "spend_cap",
    "scope",
    "user_hangup",
)
CUT_OFF = (
    "client_gone",
    "idle",
    "vendor_error",
    "server_restart",
    "superseded",
    "crashed",
)


def test_the_ends_that_run_a_call_to_its_close_are_these_six():
    assert voice_handoff.RAN_TO_ITS_CLOSE == frozenset(RAN_TO_ITS_CLOSE)
    assert not any(voice_handoff.ran_to_its_close(end) for end in CUT_OFF)


async def _typed_refusals(stored: StoredInterview) -> list[tuple[int, object]]:
    """How the typed start, a typed turn and the typed close are refused."""
    refusals = []
    for attempt in (
        lambda: interview.start_onboarding_interview(USER),
        lambda: interview.admit_onboarding_interview_turn(stored.session, USER),
        lambda: interview.finalize_onboarding_interview(
            USER, stored.session.session_id, BackgroundTasks()
        ),
    ):
        with pytest.raises(HTTPException) as refused:
            await attempt()
        refusals.append((refused.value.status_code, refused.value.detail))
    return refusals


async def _close_typed(stored: StoredInterview, *, corrections: str | None = None):
    background = BackgroundTasks()
    response = await interview.finalize_onboarding_interview(
        USER, stored.session.session_id, background, corrections=corrections
    )
    await background()
    return response


async def _started(prefix: str) -> None:
    """Wait for the background tasks named *prefix* started so far."""
    await asyncio.wait_for(
        asyncio.gather(
            *(
                task
                for task in asyncio.all_tasks()
                if task.get_name().startswith(prefix)
                and task is not asyncio.current_task()
            )
        ),
        timeout=5,
    )


async def _typed_take_handed_off() -> None:
    """Wait for the typed hand-offs owed and started so far."""
    await _started("interview-owed-hand-off")


@pytest.mark.parametrize("end_reason", RAN_TO_ITS_CLOSE)
async def test_an_accepted_call_that_ran_to_its_close_takes_the_interview(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    end_reason: str,
):
    rig.interviews.typed = stored.session
    call_id = await rig.call(end_reason=end_reason)

    record = await rig.hand_off(call_id)

    assert record.status == locks.HANDED_OFF
    assert record.interview == locks.INTERVIEW_TAKEN
    assert rig.interviews.taken == call_id
    closed = (409, {"code": INTERVIEW_CLOSED, "call_id": call_id})
    assert await _typed_refusals(stored) == [closed] * 3


@pytest.mark.parametrize("end_reason", CUT_OFF)
async def test_an_accepted_call_that_was_cut_off_leaves_the_interview_open(
    rig: Rig,
    dumps: DumpStore,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    end_reason: str,
):
    call_id = await rig.call(end_reason=end_reason)

    record = await rig.hand_off(call_id)

    assert record.status == locks.HANDED_OFF
    assert record.accepted
    assert record.interview is None
    assert rig.interviews.taken is None
    # Handed off all the same, so the person has a Home if they never return.
    assert dumps.row is not None and dumps.row.recordingId == f"interview-{call_id}"
    started = await interview.start_onboarding_interview(USER)
    assert started.turn_id == "turn-1"


async def test_the_typed_interview_replaces_a_cut_off_calls_take(
    rig: Rig, stored: StoredInterview, allowance: AsyncMock, dumps: DumpStore
):
    rig.interviews.typed = stored.session
    call_id = await rig.call(end_reason="client_gone")
    await rig.hand_off(call_id)

    await _type(stored)
    response = await _close_typed(stored)

    assert response.status == BrainDumpStatus.transcribed
    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{stored.session.session_id}"
    assert rig.interviews.taken is None


async def test_a_second_call_replaces_a_cut_off_calls_take(rig: Rig, dumps: DumpStore):
    cut_off = await rig.call(end_reason="idle")
    await rig.hand_off(cut_off)

    second = await rig.call(SAID[1], end_reason="finished")
    record = await rig.hand_off(second)

    assert record.interview == locks.INTERVIEW_TAKEN
    assert rig.interviews.taken == second
    assert dumps.row is not None and dumps.row.recordingId == f"interview-{second}"


async def test_the_persons_finish_of_a_cut_off_call_makes_its_take_the_interviews(
    rig: Rig, quality_gate: AsyncMock
):
    call_id = await rig.call(end_reason="vendor_error")
    await rig.hand_off(call_id)

    answered = await _finish(call_id)

    # The take as it stands: its pipeline jobs ran with the call's hand-off.
    assert answered.status == BrainDumpStatus.completed
    record = await rig.read(call_id)
    assert record.interview == locks.INTERVIEW_TAKEN
    assert rig.interviews.taken == call_id
    # The take was the call's already: its words were judged once.
    assert quality_gate.await_count == 1


async def test_which_call_took_the_interview_outlives_the_calls_record(
    rig: Rig, opening: dict[str, MagicMock], allowance: AsyncMock
):
    call_id = await rig.call()
    await rig.hand_off(call_id)
    await rig.redis.delete(locks.call_record_key(call_id))

    with pytest.raises(HTTPException) as refused:
        await interview.start_onboarding_interview(USER)

    assert refused.value.detail == {"code": INTERVIEW_CLOSED, "call_id": call_id}


async def test_a_cut_off_call_whose_record_expired_leaves_typing_open(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
):
    call_id = await rig.call(end_reason="client_gone")
    await rig.hand_off(call_id)
    await rig.redis.delete(locks.call_record_key(call_id))

    started = await interview.start_onboarding_interview(USER)
    await _type(stored)

    assert started.turn_id == "turn-1"
    assert not await rig.redis.exists(locks.call_record_key(call_id))


# ---------- a hand-off that lands after the person started typing ----------


async def test_typing_after_a_call_ended_supersedes_its_late_hand_off(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
    dumps: DumpStore,
    quality_gate: AsyncMock,
):
    """The call ran to its close, and its transcript is slow: the person
    types instead, and what they type owns the interview."""
    rig.interviews.typed = stored.session
    call_id = await rig.call(None)

    started = await interview.start_onboarding_interview(USER)
    assert started.turn_id == "turn-1"
    # The typed start claims nothing, and nor does a message before it is
    # dispatched.
    assert await locks.interview_of(rig.redis, call_id) is None
    await interview.admit_onboarding_interview_turn(stored.session, USER)
    assert await locks.interview_of(rig.redis, call_id) is None

    await interview.typed_message_accepted(USER)
    assert await locks.interview_of(rig.redis, call_id) == locks.INTERVIEW_SUPERSEDED

    await rig.transcribed(call_id)
    record = await rig.hand_off(call_id)

    assert record.status == locks.HANDED_OFF
    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    assert rig.interviews.taken is None
    assert dumps.row is None
    quality_gate.assert_not_awaited()

    response = await _close_typed(stored)

    assert response.status == BrainDumpStatus.transcribed
    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{stored.session.session_id}"


async def test_a_call_whose_words_landed_before_any_typing_takes_the_interview(
    rig: Rig,
    stored: StoredInterview,
    opening: dict[str, MagicMock],
    allowance: AsyncMock,
):
    """The person opened the typed interview and looked at the greeting, but
    typed nothing before the call's words landed."""
    rig.interviews.typed = stored.session
    call_id = await rig.call(None)
    await interview.start_onboarding_interview(USER)

    await rig.transcribed(call_id)
    record = await rig.hand_off(call_id)

    assert record.interview == locks.INTERVIEW_TAKEN
    with pytest.raises(HTTPException) as refused:
        await interview.admit_onboarding_interview_turn(stored.session, USER)
    assert refused.value.status_code == 409
    assert refused.value.detail == {"code": INTERVIEW_CLOSED, "call_id": call_id}


async def test_typing_that_lands_while_the_pipeline_judges_the_call_wins(
    rig: Rig,
    stored: StoredInterview,
    allowance: AsyncMock,
    mocker: MockerFixture,
    dumps: DumpStore,
    jobs: dict[str, AsyncMock],
):
    """The pipeline accepts the call's words, and the person's message lands
    before the call claims the interview: the message's claim stands, the
    call's take gives up the row with none of its jobs run, and the typed
    take owns the interview."""
    rig.interviews.typed = stored.session
    call_id = await rig.call()
    judge = voice_handoff.close_voice_interview

    async def judged_then_typed(*args, **kwargs):
        closure = await judge(*args, **kwargs)
        await _type(stored)
        return closure

    mocker.patch.object(voice_handoff, "close_voice_interview", new=judged_then_typed)

    record = await rig.hand_off(call_id)

    assert record.status == locks.HANDED_OFF
    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    assert record.interview == locks.INTERVIEW_SUPERSEDED
    assert rig.interviews.taken is None
    jobs["greeting"].assert_not_awaited()
    assert dumps.row is not None
    assert (dumps.row.recordingId, dumps.row.status, dumps.row.errorCode) == (
        f"interview-{call_id}",
        BrainDumpStatus.failed,
        interview.SUPERSEDED,
    )

    response = await _close_typed(stored)

    assert response.status == BrainDumpStatus.transcribed
    assert dumps.row.recordingId == f"interview-{stored.session.session_id}"


async def test_a_typed_finish_while_the_pipeline_judges_the_call_hands_the_typed_take_off(
    rig: Rig,
    stored: StoredInterview,
    mocker: MockerFixture,
    dumps: DumpStore,
):
    """The person finishes the typed interview, with corrections, while the
    pipeline judges the call. The finish wins the interview, but the row
    still holds the call's take, so it answers ``superseded`` and its
    hand-off is owed. The call's hand-off then gives the row up and starts
    the owed hand-off, which lands the typed take, corrections and all, with
    no further request."""
    rig.interviews.typed = stored.session
    call_id = await rig.call()
    judge = voice_handoff.close_voice_interview
    finished = []

    async def judged_then_finished(*args, **kwargs):
        closure = await judge(*args, **kwargs)
        finished.append(
            await _close_typed(stored, corrections="Eight staff, not five.")
        )
        return closure

    mocker.patch.object(
        voice_handoff, "close_voice_interview", new=judged_then_finished
    )

    record = await rig.hand_off(call_id)
    await _typed_take_handed_off()

    assert finished[0].error_code == interview.SUPERSEDED
    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    assert record.interview == locks.INTERVIEW_SUPERSEDED
    assert rig.interviews.taken is None
    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{stored.session.session_id}"
    assert dumps.row.status != BrainDumpStatus.failed
    assert dumps.row.transcript is not None
    assert dumps.row.transcript.endswith("Eight staff, not five.")
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_HANDED_OFF
    )
    assert not await rig.redis.exists(interview.owed_hand_off_key(USER))


async def test_the_sweep_hands_off_a_typed_interview_left_owed(
    rig: Rig,
    stored: StoredInterview,
    mocker: MockerFixture,
    dumps: DumpStore,
):
    """The call's hand-off gave the row up, and the owed typed hand-off it
    starts never ran: the REST API's sweep hands it off. The owed hand-off
    keeps the corrections as the hand-off would, and lapses a day after it
    was owed."""
    rig.interviews.typed = stored.session
    call_id = await rig.call()
    judge = voice_handoff.close_voice_interview
    kept = "x" * MAX_CORRECTION_CHARS

    async def judged_then_finished(*args, **kwargs):
        closure = await judge(*args, **kwargs)
        await _close_typed(stored, corrections=f"  {kept}{'y' * 50}  ")
        return closure

    mocker.patch.object(
        voice_handoff, "close_voice_interview", new=judged_then_finished
    )
    lost = mocker.patch.object(voice_handoff, "start_owed_typed_hand_off")

    await rig.hand_off(call_id)

    lost.assert_called_once_with(USER)
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_CLOSED
    )
    key = interview.owed_hand_off_key(USER)
    owed = interview.OwedHandOff.model_validate_json(await rig.redis.get(key))
    assert (owed.session_id, owed.corrections) == (stored.session.session_id, kept)
    assert 0 < await rig.redis.ttl(key) <= interview.OWED_HAND_OFF_TTL_S
    assert await rig.redis.zscore(interview.OWED_HAND_OFFS_KEY, USER) is not None

    await voice_handoff.sweep()

    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{stored.session.session_id}"
    assert dumps.row.transcript is not None and dumps.row.transcript.endswith(kept)
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_HANDED_OFF
    )
    assert not await rig.redis.exists(key)
    assert await rig.redis.zscore(interview.OWED_HAND_OFFS_KEY, USER) is None


async def test_an_owed_hand_off_that_lapsed_leaves_the_sweeps_index(rig: Rig):
    """Its entry expired with its day, or its index entry outlived it."""
    await rig.redis.zadd(
        interview.OWED_HAND_OFFS_KEY,
        {USER: time.time() - interview.OWED_HAND_OFF_TTL_S - 1, "user-2": time.time()},
    )

    await voice_handoff.sweep()

    assert await rig.redis.zcard(interview.OWED_HAND_OFFS_KEY) == 0


async def test_a_typed_interview_waiting_for_its_confirm_is_not_handed_off_for_it(
    rig: Rig,
    stored: StoredInterview,
    typed_arm: AsyncMock,
    quality_gate: AsyncMock,
    dumps: DumpStore,
):
    """While the pipeline judges the call, the person types, and Otto's
    goodbye closes the typed interview on a confirm arm. The call's take
    gives the row up, but the typed interview owes no hand-off until the
    person confirms, so their corrections reach it."""
    typed_arm.side_effect = None
    typed_arm.return_value = InterviewEnding.PROFILE_CARD.value
    rig.interviews.typed = stored.session
    judge = Judge()
    quality_gate.side_effect = judge.rule
    call_id = await rig.call()
    hand_off = asyncio.create_task(
        voice_handoff.hand_off_owed(call_id, trigger="status")
    )
    await asyncio.wait_for(judge.entered.wait(), timeout=5)

    await _type(stored)
    closed = await interview.close_onboarding_interview(
        USER, stored.session.session_id, reason="confirmed"
    )
    judge.release.set()
    await asyncio.wait_for(hand_off, timeout=5)
    await _typed_take_handed_off()

    assert closed.hand_off is None
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_CLOSED
    )
    assert dumps.row is not None
    assert (dumps.row.recordingId, dumps.row.status) == (
        f"interview-{call_id}",
        BrainDumpStatus.failed,
    )

    confirmed = await _close_typed(stored, corrections="Eight staff, not five.")

    assert confirmed.status == BrainDumpStatus.transcribed
    assert dumps.row.recordingId == f"interview-{stored.session.session_id}"
    assert dumps.row.transcript is not None
    assert dumps.row.transcript.endswith("Eight staff, not five.")


@pytest.mark.parametrize("first", ["typing", "hand_off"])
async def test_a_hand_off_and_a_message_at_once_leave_exactly_one_take(
    rig: Rig,
    stored: StoredInterview,
    allowance: AsyncMock,
    mocker: MockerFixture,
    quality_gate: AsyncMock,
    dumps: DumpStore,
    jobs: dict[str, AsyncMock],
    first: str,
):
    """The person's message is admitted while the call's hand-off judges its
    words, and the message's dispatch and the hand-off then claim the
    interview on the call's record: whichever claim lands first stands, and
    the other side's answer says so. The words are judged once either way."""
    rig.interviews.typed = stored.session
    judge = Judge()
    quality_gate.side_effect = judge.rule
    call_id = await rig.call()
    await interview.admit_onboarding_interview_turn(stored.session, USER)
    hand_off = asyncio.create_task(
        voice_handoff.hand_off_owed(call_id, trigger="status")
    )
    await asyncio.wait_for(judge.entered.wait(), timeout=5)

    if first == "typing":
        await interview.typed_message_accepted(USER)
        judge.release.set()
        await asyncio.wait_for(hand_off, timeout=5)
    else:
        claim = locks.claim_interview
        taken = asyncio.Event()

        async def after_the_take(redis, call: str, standing):
            if standing == locks.INTERVIEW_SUPERSEDED:
                await taken.wait()
            result = await claim(redis, call, standing)
            if standing == locks.INTERVIEW_TAKEN:
                taken.set()
            return result

        mocker.patch.object(locks, "claim_interview", new=after_the_take)
        judge.release.set()
        await asyncio.wait_for(
            asyncio.gather(hand_off, interview.typed_message_accepted(USER)),
            timeout=5,
        )
    await _typed_take_handed_off()

    record = await rig.read(call_id)
    assert judge.judged == 1
    assert record.hand_off is not None
    assert dumps.row is not None
    if first == "hand_off":
        assert record.interview == locks.INTERVIEW_TAKEN
        assert record.accepted
        assert rig.interviews.taken == call_id
        jobs["greeting"].assert_awaited_once()
        with pytest.raises(HTTPException) as refused:
            await interview.admit_onboarding_interview_turn(stored.session, USER)
        assert refused.value.detail == {"code": INTERVIEW_CLOSED, "call_id": call_id}
    else:
        assert record.interview == locks.INTERVIEW_SUPERSEDED
        assert record.hand_off.error_code == interview.SUPERSEDED
        assert rig.interviews.taken is None
        jobs["greeting"].assert_not_awaited()
        # The call's take gave the row up.
        assert (dumps.row.recordingId, dumps.row.status, dumps.row.errorCode) == (
            f"interview-{call_id}",
            BrainDumpStatus.failed,
            interview.SUPERSEDED,
        )
        await interview.admit_onboarding_interview_turn(stored.session, USER)


async def test_turns_typed_before_the_call_do_not_supersede_it(
    rig: Rig, stored: StoredInterview
):
    rig.interviews.typed = stored.session
    await _type(stored)
    call_id = await rig.call()

    record = await rig.hand_off(call_id)

    assert record.interview == locks.INTERVIEW_TAKEN
    assert rig.interviews.taken == call_id


async def test_typing_inside_an_earlier_calls_judge_past_a_no_words_call_wins(
    rig: Rig,
    stored: StoredInterview,
    allowance: AsyncMock,
    quality_gate: AsyncMock,
    dumps: DumpStore,
    jobs: dict[str, AsyncMock],
):
    """The later call heard nothing, so the earlier call's hand-off runs, and
    the person's message is dispatched while its words are judged. Typing's
    claim reaches the earlier call's record past the no-words one, where the
    hand-off claims too, so exactly one take stands: the typed one."""
    rig.interviews.typed = stored.session
    judge = Judge()
    quality_gate.side_effect = judge.rule
    earlier = await rig.call()
    later = await rig.call(None)
    await rig.no_words(later, quality.NO_USABLE_SPEECH)
    await interview.admit_onboarding_interview_turn(stored.session, USER)
    hand_off = asyncio.create_task(
        voice_handoff.hand_off_owed(earlier, trigger="settle")
    )
    await asyncio.wait_for(judge.entered.wait(), timeout=5)

    await interview.typed_message_accepted(USER)
    judge.release.set()
    await asyncio.wait_for(hand_off, timeout=5)
    await _typed_take_handed_off()

    record = await rig.read(earlier)
    assert record.interview == locks.INTERVIEW_SUPERSEDED
    assert (await rig.read(later)).interview == locks.INTERVIEW_SUPERSEDED
    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    assert rig.interviews.taken is None
    jobs["greeting"].assert_not_awaited()

    await interview.admit_onboarding_interview_turn(stored.session, USER)
    response = await _close_typed(stored)

    assert response.status == BrainDumpStatus.transcribed
    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{stored.session.session_id}"


async def test_a_message_after_an_earlier_call_took_the_interview_takes_nothing(
    rig: Rig, stored: StoredInterview, allowance: AsyncMock
):
    """Behind a later call that heard nothing, the earlier call's words took
    the interview a moment before the person's message was dispatched: the
    message claims nothing, the no-words call included, and the next one is
    refused naming the earlier call, before its onboarding row is written
    as after."""
    rig.interviews.typed = stored.session
    earlier = await rig.call()
    later = await rig.call(None)
    await rig.no_words(later, quality.NO_USABLE_SPEECH)
    await interview.admit_onboarding_interview_turn(stored.session, USER)
    assert (await rig.hand_off(earlier)).interview == locks.INTERVIEW_TAKEN

    await interview.typed_message_accepted(USER)

    assert await locks.interview_of(rig.redis, earlier) == locks.INTERVIEW_TAKEN
    assert await locks.interview_of(rig.redis, later) is None
    for row_written in (False, True):
        rig.interviews.taken = earlier if row_written else None
        with pytest.raises(HTTPException) as refused:
            await interview.admit_onboarding_interview_turn(stored.session, USER)
        assert refused.value.detail == {"code": INTERVIEW_CLOSED, "call_id": earlier}


async def _cut_inside_the_judge(call_id: str, judge: Judge) -> None:
    """A hand-off of *call_id* stopped inside the quality gate: the call's
    take holds the row, claimed and never judged, and no answer is recorded."""
    cut = asyncio.create_task(voice_handoff.hand_off_owed(call_id, trigger="status"))
    await asyncio.wait_for(judge.entered.wait(), timeout=5)
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut


async def test_typing_claimed_on_a_later_call_beats_an_earlier_calls_take_on_the_row(
    rig: Rig,
    stored: StoredInterview,
    allowance: AsyncMock,
    quality_gate: AsyncMock,
    dumps: DumpStore,
):
    """A stopped hand-off left the earlier call's take on the row, with no
    claim on its record, and the person typed after a later call with words,
    so typing's claim lands on the later call. The typed finish still reads
    the earlier take as beaten, waits for the row, and lands once the
    earlier call's retry gives it up."""
    rig.interviews.typed = stored.session
    judge = Judge()
    quality_gate.side_effect = judge.rule
    earlier = await rig.call()
    await _cut_inside_the_judge(earlier, judge)
    later = await rig.call(SAID[1])
    await _type(stored)
    assert await locks.interview_of(rig.redis, later) == locks.INTERVIEW_SUPERSEDED
    assert await locks.interview_of(rig.redis, earlier) is None

    finished = await _close_typed(stored)

    assert finished.error_code == interview.SUPERSEDED
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_CLOSED
    )

    record = await rig.hand_off(earlier)
    await _typed_take_handed_off()

    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{stored.session.session_id}"
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_HANDED_OFF
    )


async def test_finishing_a_call_that_heard_nothing_with_corrections_hands_nothing_off(
    rig: Rig, dumps: DumpStore, quality_gate: AsyncMock
):
    """Corrections alone are no take: the call heard nothing, so there is
    nothing to confirm or correct."""
    call_id = await rig.call("", ending=InterviewEnding.PROFILE_CARD.value)

    answered = await _finish(call_id, corrections="Eight staff, not five.")

    assert answered.error_code == quality.NO_USABLE_SPEECH
    assert dumps.row is None
    quality_gate.assert_not_awaited()


async def test_a_take_left_unjudged_on_the_row_is_given_up_once_typing_beat_its_call(
    rig: Rig,
    stored: StoredInterview,
    allowance: AsyncMock,
    quality_gate: AsyncMock,
    dumps: DumpStore,
):
    """A hand-off stopped inside the judge left the call's take on the row,
    and the person typed and finished before its retry: their finish answers
    ``superseded`` while the take holds the row. The retry answers
    ``superseded`` without the pipeline and gives the row up, and the typed
    take lands with no further request."""
    rig.interviews.typed = stored.session
    judge = Judge()
    quality_gate.side_effect = judge.rule
    call_id = await rig.call()
    await _cut_inside_the_judge(call_id, judge)
    assert dumps.row is not None
    assert (dumps.row.recordingId, dumps.row.status) == (
        f"interview-{call_id}",
        BrainDumpStatus.transcribed,
    )

    await _type(stored)
    finished = await _close_typed(stored)
    assert finished.error_code == interview.SUPERSEDED
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_CLOSED
    )

    record = await rig.hand_off(call_id)
    await _typed_take_handed_off()

    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    assert dumps.row.recordingId == f"interview-{stored.session.session_id}"
    assert dumps.row.status != BrainDumpStatus.failed
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_HANDED_OFF
    )


async def test_a_typed_finish_frees_a_row_a_call_beaten_by_typing_left_after_answering(
    rig: Rig,
    stored: StoredInterview,
    allowance: AsyncMock,
    quality_gate: AsyncMock,
    dumps: DumpStore,
):
    """The call's hand-off has answered, yet its take still holds the row
    unjudged: no hand-off of the call is left to give it up, so the typed
    finish gives it up itself, while no pipeline has run the take, and the
    typed take lands with no further request."""
    rig.interviews.typed = stored.session
    judge = Judge()
    quality_gate.side_effect = judge.rule
    call_id = await rig.call()
    await _cut_inside_the_judge(call_id, judge)
    await _type(stored)
    await locks.record_answer(
        rig.redis,
        call_id,
        uid=USER,
        status=locks.HANDED_OFF,
        answer=interview.superseded_take(),
    )

    finished = await _close_typed(stored)
    await _typed_take_handed_off()

    assert finished.error_code == interview.SUPERSEDED
    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{stored.session.session_id}"
    assert dumps.row.status != BrainDumpStatus.failed
    assert stored.session.metadata.onboarding_interview_stage == (
        INTERVIEW_STAGE_HANDED_OFF
    )


async def test_a_hand_off_that_loses_its_claim_at_its_last_check_runs_no_jobs(
    rig: Rig,
    mocker: MockerFixture,
    quality_gate: AsyncMock,
    jobs: dict[str, AsyncMock],
):
    """The claim lapses between the interview's claim and the answer: the
    hand-off records no answer and runs none of its pipeline jobs, and the
    next hand-off judges the words again and runs them once."""
    call_id = await rig.call()
    claim = locks.claim_interview
    lapsed: list[str] = []

    async def claimed_then_lapsed(redis, call: str, standing):
        result = await claim(redis, call, standing)
        if not lapsed:
            lapsed.append(call)
            await redis.delete(locks.postcall_claim_key(call))
        return result

    mocker.patch.object(locks, "claim_interview", new=claimed_then_lapsed)

    assert await voice_handoff.hand_off_owed(call_id, trigger="status") == (
        voice_handoff.LOST_CLAIM
    )

    assert (await rig.read(call_id)).hand_off is None
    jobs["greeting"].assert_not_awaited()
    assert call_id in await locks.postcall_pending(rig.redis)

    assert (await rig.hand_off(call_id)).status == locks.HANDED_OFF
    assert quality_gate.await_count == 2
    jobs["greeting"].assert_awaited_once_with(SAID[0])


async def test_a_record_on_the_row_that_is_not_the_persons_ends_the_walk(rig: Rig):
    """The row's latest call is a record of someone else's, still being
    transcribed: no call of the person's, so the earlier call's hand-off
    does not wait on it."""
    earlier = await rig.call()
    other = tokens.new_call_id()
    await locks.write_call_record(
        rig.redis, other, {"uid": "someone-else", "status": locks.TRANSCRIBING}
    )
    rig.interviews.latest_call = other

    record = await rig.hand_off(earlier)

    assert record.status == locks.HANDED_OFF
    assert record.interview == locks.INTERVIEW_TAKEN


async def test_try_again_after_a_cut_retry_judges_the_take_it_left_on_the_row(
    rig: Rig,
    quality_gate: AsyncMock,
    dumps: DumpStore,
    jobs: dict[str, AsyncMock],
):
    """The call's hand-off answered ``quality_check_unavailable``, and a "Try
    again" stopped inside the judge, leaving the take on the row unjudged.
    The next "Try again" judges it rather than reading the row as accepted,
    and the pipeline's jobs run once."""
    rulings = 0
    cut_here = asyncio.Event()

    async def gate(*args, **kwargs):
        nonlocal rulings
        rulings += 1
        if rulings == 1:
            return quality.QualityRejection(
                error_code=quality.QUALITY_CHECK_UNAVAILABLE, reason="judge_error"
            )
        if rulings == 2:
            cut_here.set()
            await asyncio.Event().wait()
        return None

    quality_gate.side_effect = gate
    call_id = await rig.call()
    answered = (await rig.hand_off(call_id)).hand_off
    assert answered is not None
    assert answered.error_code == quality.QUALITY_CHECK_UNAVAILABLE
    cut = asyncio.create_task(_finish(call_id))
    await asyncio.wait_for(cut_here.wait(), timeout=5)
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut
    assert dumps.row is not None
    assert dumps.row.status == BrainDumpStatus.transcribed

    again = await _finish(call_id)

    assert again.status == BrainDumpStatus.transcribed
    assert rulings == 3
    jobs["greeting"].assert_awaited_once_with(SAID[0])
    assert (await rig.read(call_id)).took_interview


def _note_settles(mocker: MockerFixture) -> list[asyncio.Task | None]:
    """The hand-offs started for a take a no-words call settled."""
    settle = voice_handoff.settle_if_owed
    settled: list[asyncio.Task | None] = []

    def settle_noted(record: locks.CallRecord, *, trigger: str = "status"):
        running = settle(record, trigger=trigger)
        if trigger == "settle":
            settled.append(running)
        return running

    mocker.patch.object(voice_handoff, "settle_if_owed", new=settle_noted)
    return settled


async def test_a_no_words_answer_settles_the_take_past_other_no_words_calls(
    rig: Rig, mocker: MockerFixture
):
    earlier = await rig.call()
    middle = await rig.call(None)
    await rig.no_words(middle, locks.TRANSCRIPTION_UNAVAILABLE)
    later = await rig.call("")
    settled = _note_settles(mocker)

    answer = (await rig.hand_off(later)).hand_off

    assert answer is not None and answer.error_code == quality.NO_USABLE_SPEECH
    assert len(settled) == 1 and settled[0] is not None
    assert await asyncio.wait_for(settled[0], timeout=5) == locks.HANDED_OFF
    assert rig.interviews.taken == earlier


async def test_a_status_read_of_a_no_words_call_settles_the_take_before_it(
    rig: Rig, mocker: MockerFixture
):
    """The relay recorded the later call's answer, which settles nothing on
    its own: the step's read of that call does."""
    earlier = await rig.call()
    later = await rig.call(None)
    assert await voice_handoff.hand_off_owed(earlier, trigger="status") == (
        voice_handoff.WAITING
    )
    await rig.no_words(later, locks.TRANSCRIPTION_UNAVAILABLE)
    settled = _note_settles(mocker)

    read = await voice.call_status(USER, later)
    await _started("voice-settle-before")

    assert read.status == locks.FAILED
    assert len(settled) == 1 and settled[0] is not None
    assert await asyncio.wait_for(settled[0], timeout=5) == locks.HANDED_OFF
    assert rig.interviews.taken == earlier


async def test_a_settle_that_fails_never_fails_the_answer(
    rig: Rig, mocker: MockerFixture
):
    earlier = await rig.call()
    later = await rig.call("")
    mocker.patch.object(
        voice_handoff, "settle_if_owed", side_effect=ConnectionError("redis")
    )

    record = await rig.hand_off(later)

    assert record.status == locks.FAILED
    assert record.hand_off is not None
    assert record.hand_off.error_code == quality.NO_USABLE_SPEECH
    assert earlier in await locks.postcall_pending(rig.redis)


async def _until(done, what: str) -> None:
    for _ in range(500):
        if done():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"never {what}")


@pytest.mark.parametrize("held_as", ["transcribed", "extracting"])
@pytest.mark.parametrize(
    ("end_reason", "claim"),
    [("client_gone", None), ("finished", locks.INTERVIEW_TAKEN)],
    ids=["cut-off", "ran-to-its-close"],
)
async def test_a_later_calls_take_displaces_an_earlier_calls_take_in_the_pipeline(
    rig: Rig,
    mocker: MockerFixture,
    dumps: DumpStore,
    jobs: dict[str, AsyncMock],
    extraction: dict[str, AsyncMock],
    quality_gate: AsyncMock,
    held_as: str,
    end_reason: str,
    claim: str | None,
):
    """A cut-off call's accepted take holds the row while its pipeline runs,
    and the person called again. The later call's take claims the row from
    it, is never answered ``superseded``, claims nothing for it, and its
    pipeline runs; the earlier take's jobs write nothing more, and its call's
    record is left as it was until the person finishes that call."""
    release = asyncio.Event()
    held = asyncio.Event()

    async def greeting_of(transcript: str, *args, **kwargs):
        return (f"Greeting for {transcript}", [])

    jobs["greeting"].side_effect = greeting_of
    earlier = await rig.call(SAID[0], end_reason="client_gone")
    earlier_take = f"interview-{earlier}"

    async def hold_the_earlier(*args, **kwargs) -> None:
        if any(SAID[0] in str(arg) for arg in (*args, *kwargs.values())):
            held.set()
            await release.wait()

    if held_as == "transcribed":
        run_jobs = service._run_background_jobs

        async def jobs_held(user_id: str, recording_id: str, *args):
            if recording_id == earlier_take:
                held.set()
                await release.wait()
            await run_jobs(user_id, recording_id, *args)

        mocker.patch.object(service, "_run_background_jobs", new=jobs_held)
    else:
        team = jobs["team"].return_value
        understanding = extraction["extract_business_understanding"].return_value

        async def team_held(*args, **kwargs):
            await hold_the_earlier(*args, **kwargs)
            return team

        async def understanding_held(*args, **kwargs):
            await hold_the_earlier(*args, **kwargs)
            return understanding

        jobs["team"].side_effect = team_held
        extraction["extract_business_understanding"].side_effect = understanding_held

    first = asyncio.create_task(voice_handoff.hand_off_owed(earlier, trigger="status"))
    await asyncio.wait_for(held.wait(), timeout=5)
    if held_as == "extracting":
        await _until(
            lambda: dumps.row is not None and dumps.row.greeting is not None,
            "stored the earlier greeting",
        )
    assert dumps.row is not None
    assert (dumps.row.recordingId, dumps.row.status) == (
        earlier_take,
        BrainDumpStatus(held_as),
    )

    later = await rig.call(SAID[1], end_reason=end_reason)
    record = await rig.hand_off(later)
    release.set()
    await asyncio.wait_for(first, timeout=5)

    # The later call's take: on the row, its pipeline run.
    assert record.status == locks.HANDED_OFF
    assert record.accepted
    assert record.hand_off is not None and record.hand_off.error_code is None
    assert record.interview == claim
    row = dumps.row
    assert (row.recordingId, row.transcript, row.greeting, row.status) == (
        f"interview-{later}",
        SAID[1],
        f"Greeting for {SAID[1]}",
        BrainDumpStatus.completed,
    )
    # The earlier take's jobs wrote nothing more: no model ran for a take
    # whose jobs had not started, and none of its later results was kept.
    greeted = [call.args[0] for call in jobs["greeting"].await_args_list]
    assert greeted == ([SAID[1]] if held_as == "transcribed" else SAID)
    assert extraction["upsert_business_understanding"].await_count == 1
    # The earlier call's record is as it was.
    before = await rig.read(earlier)
    assert (before.status, before.accepted, before.interview) == (
        locks.HANDED_OFF,
        True,
        None,
    )

    gate_runs = quality_gate.await_count
    answered = await _finish(earlier)

    assert answered.error_code == interview.SUPERSEDED
    assert quality_gate.await_count == gate_runs
    assert (await rig.read(earlier)).interview == locks.INTERVIEW_SUPERSEDED
    assert dumps.row.recordingId == f"interview-{later}"


@pytest.mark.parametrize("held", ["typed-interview", "brain-dump"])
async def test_a_calls_take_never_displaces_a_typed_interviews_or_a_brain_dumps(
    rig: Rig, dumps: DumpStore, quality_gate: AsyncMock, held: str
):
    """Nor does the row's ``superseded`` claim anything: it is recorded as
    the call's answer."""
    take = f"interview-{uuid.uuid4()}" if held == "typed-interview" else "brain-dump-1"
    row = await dumps.start_dump(
        USER,
        take,
        (
            BrainDumpInputMode.typed
            if held == "typed-interview"
            else BrainDumpInputMode.voice
        ),
    )
    row.status = BrainDumpStatus.extracting
    call_id = await rig.call()

    record = await rig.hand_off(call_id)

    assert record.status == locks.HANDED_OFF
    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    assert record.hand_off.input_mode == BrainDumpInputMode.voice
    assert record.interview is None
    quality_gate.assert_not_awaited()
    assert dumps.row is not None and dumps.row.recordingId == take


async def test_the_sweeps_retry_of_a_superseded_call_runs_no_pipeline(
    rig: Rig, quality_gate: AsyncMock, dumps: DumpStore
):
    call_id = await rig.call()
    await locks.claim_interview(rig.redis, call_id, locks.INTERVIEW_SUPERSEDED)

    assert await voice_handoff.sweep() == [call_id]

    record = await rig.read(call_id)
    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    quality_gate.assert_not_awaited()
    assert dumps.row is None


async def test_finishing_a_call_the_persons_typing_superseded_runs_no_pipeline(
    rig: Rig, quality_gate: AsyncMock, dumps: DumpStore
):
    call_id = await rig.call(ending=InterviewEnding.FIRST_TASK.value)
    await locks.claim_interview(rig.redis, call_id, locks.INTERVIEW_SUPERSEDED)

    answered = await _finish(call_id)

    assert answered.error_code == interview.SUPERSEDED
    quality_gate.assert_not_awaited()
    assert dumps.row is None


async def test_a_second_finish_finds_the_first_ones_verdict_and_judges_nothing(
    rig: Rig, quality_gate: AsyncMock, mocker: MockerFixture
):
    """Two clicks of "Try again": the second read the call before the first
    held it, then reads it again once it holds it."""
    quality_gate.side_effect = [
        quality.QualityRejection(
            error_code=quality.QUALITY_CHECK_UNAVAILABLE, reason="judge_error"
        ),
        quality.QualityRejection(
            error_code=quality.INSUFFICIENT_CONTENT, reason="judge_unusable"
        ),
    ]
    call_id = await rig.call()
    await rig.hand_off(call_id)
    read_before = await rig.read(call_id)

    first = await _finish(call_id)
    assert first.error_code == quality.INSUFFICIENT_CONTENT

    real_read = voice._their_call
    reads: list[str] = []

    async def stale_then_real(redis, user_id: str, call: str):
        reads.append(call)
        if len(reads) == 1:
            return read_before
        return await real_read(redis, user_id, call)

    mocker.patch.object(voice, "_their_call", new=stale_then_real)
    second = await _finish(call_id)

    assert second == first
    assert quality_gate.await_count == 2
    assert len(reads) == 2


async def test_a_call_ended_from_another_device_leaves_room_for_a_new_call(
    rig: Rig, mocker: MockerFixture, monkeypatch
):
    """ "End that call" cuts the call off: its take never closes the
    interview, so the person can call again."""
    call_id = await rig.call(end_reason="superseded")
    record = await rig.hand_off(call_id)
    assert record.status == locks.HANDED_OFF and record.interview is None

    monkeypatch.setattr(
        voice.settings.config, "voice_relay_public_url", "wss://api.example.com/voice"
    )
    monkeypatch.setattr(voice.settings.config, "voice_interview_max_concurrent", 25)
    monkeypatch.setattr(voice.settings.config, "voice_interview_daily_budget_usd", 25.0)
    monkeypatch.setattr(tokens.settings.secrets, "voice_relay_signing_key", "k" * 48)
    monkeypatch.setattr(
        voice.settings.config, "livekit_url", "wss://agpt-test.livekit.cloud"
    )
    monkeypatch.setattr(voice.settings.secrets, "livekit_api_key", "APItest")
    monkeypatch.setattr(voice.settings.secrets, "livekit_api_secret", LIVEKIT_SECRET)
    await locks.mark_rooms_swept(rig.redis, time.time())
    mocker.patch.object(
        voice, "evaluate_feature_flag", AsyncMock(return_value=(True, True))
    )
    mocker.patch.object(
        voice, "find_onboarding_interview_session", AsyncMock(return_value=None)
    )
    mocker.patch.object(voice, "enforce_voice_call_budget", AsyncMock())
    mocker.patch.object(voice, "assigned_ending", AsyncMock(return_value=None))

    started = await voice.start_voice_call(USER)

    assert started.call_id != call_id


# ---------- a later call that produced no take ----------

NO_WORDS_ANSWERS = (
    quality.NO_USABLE_SPEECH,
    locks.TRANSCRIPTION_UNAVAILABLE,
    locks.RECORDING_UNAVAILABLE,
)


@pytest.mark.parametrize("later", ["connecting", "live", "ended", "transcribing"])
async def test_a_hand_off_waits_while_a_later_call_could_still_become_a_take(
    rig: Rig, quality_gate: AsyncMock, later: str
):
    earlier = await rig.call()
    second = await rig.call(None)
    await locks.write_call_record(rig.redis, second, {"status": later})

    assert await voice_handoff.hand_off_owed(earlier, trigger="sweep") == (
        voice_handoff.WAITING
    )

    record = await rig.read(earlier)
    assert (record.status, record.hand_off, record.handoff_attempts) == (
        locks.TRANSCRIBED,
        None,
        0,
    )
    assert record.handoff_retry_at is None
    quality_gate.assert_not_awaited()
    assert earlier in await locks.postcall_pending(rig.redis)

    # The next read asks again, and it still waits.
    assert (await voice.call_status(USER, earlier)).status == "handing_off"
    running = voice_handoff.settle_if_owed(await rig.read(earlier))
    assert running is not None
    assert await asyncio.wait_for(running, timeout=5) == voice_handoff.WAITING


@pytest.mark.parametrize("settled", ["empty-transcript", *NO_WORDS_ANSWERS])
async def test_a_later_call_that_settled_without_words_is_passed_over(
    rig: Rig, dumps: DumpStore, settled: str
):
    earlier = await rig.call()
    later = await rig.call(None)
    if settled == "empty-transcript":
        await rig.transcribed(later, "")
    else:
        await rig.no_words(later, settled)

    record = await rig.hand_off(earlier)

    # A complete call's take takes the interview, and the row names it.
    assert record.status == locks.HANDED_OFF
    assert record.interview == locks.INTERVIEW_TAKEN
    assert rig.interviews.taken == earlier
    assert dumps.row is not None and dumps.row.recordingId == f"interview-{earlier}"


async def test_a_cut_off_call_before_a_no_words_call_leaves_the_interview_open(
    rig: Rig,
):
    """Passed over, the later call changes nothing about the earlier one: a
    cut-off call's take is handed off and leaves the interview open."""
    earlier = await rig.call(end_reason="client_gone")
    later = await rig.call(None)
    await rig.no_words(later, quality.NO_USABLE_SPEECH)

    record = await rig.hand_off(earlier)

    assert record.status == locks.HANDED_OFF
    assert record.accepted and record.interview is None
    assert rig.interviews.taken is None


async def test_a_no_words_answer_settles_the_call_before_it_at_once(
    rig: Rig, dumps: DumpStore, mocker: MockerFixture
):
    earlier = await rig.call()
    later = await rig.call(None)
    assert await voice_handoff.hand_off_owed(earlier, trigger="status") == (
        voice_handoff.WAITING
    )
    settle = voice_handoff.settle_if_owed
    settled: list[asyncio.Task | None] = []

    def settle_noted(record: locks.CallRecord, *, trigger: str = "status"):
        running = settle(record, trigger=trigger)
        if trigger == "settle":
            settled.append(running)
        return running

    mocker.patch.object(voice_handoff, "settle_if_owed", new=settle_noted)

    # The later call's words land empty, and its hand-off answers without
    # them: the earlier call's hand-off starts at once, with no read or sweep.
    await rig.transcribed(later, "")
    assert (await rig.hand_off(later)).hand_off is not None

    assert len(settled) == 1 and settled[0] is not None
    assert await asyncio.wait_for(settled[0], timeout=5) == locks.HANDED_OFF
    assert dumps.row is not None and dumps.row.recordingId == f"interview-{earlier}"
    assert rig.interviews.taken == earlier


async def test_a_later_call_judged_insufficient_supersedes_the_earlier(
    rig: Rig, quality_gate: AsyncMock
):
    """Words judged thin are still words, as a thin typed message is."""
    quality_gate.return_value = quality.QualityRejection(
        error_code=quality.INSUFFICIENT_CONTENT, reason="judge_unusable"
    )
    earlier = await rig.call(ending=InterviewEnding.PROFILE_CARD.value)
    later = await rig.call("idk")
    assert (await rig.hand_off(later)).hand_off is not None

    answered = await _finish(earlier)

    assert answered.error_code == interview.SUPERSEDED


async def test_typing_after_a_no_words_call_supersedes_the_call_before_it(
    rig: Rig, stored: StoredInterview, allowance: AsyncMock
):
    rig.interviews.typed = stored.session
    earlier = await rig.call()
    later = await rig.call(None)
    await rig.no_words(later, quality.NO_USABLE_SPEECH)

    await _type(stored)

    # Claimed on the take, the earlier call, and on the no-words call after it.
    assert await locks.interview_of(rig.redis, later) == locks.INTERVIEW_SUPERSEDED
    assert await locks.interview_of(rig.redis, earlier) == (locks.INTERVIEW_SUPERSEDED)
    record = await rig.hand_off(earlier)
    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED


async def test_no_words_calls_in_a_row_are_all_passed_over(rig: Rig):
    earlier = await rig.call()
    await rig.call("")
    later = await rig.call(None)
    await rig.no_words(later, locks.TRANSCRIPTION_UNAVAILABLE)

    record = await rig.hand_off(earlier)

    assert record.interview == locks.INTERVIEW_TAKEN


async def test_a_later_call_that_died_before_its_record_is_passed_over(rig: Rig):
    """The row names it, but it has no record and holds no lock: it never
    began."""
    earlier = await rig.call()
    rig.interviews.latest_call = tokens.new_call_id()

    record = await rig.hand_off(earlier)

    assert record.interview == locks.INTERVIEW_TAKEN


async def test_a_later_call_still_starting_holds_the_hand_off(rig: Rig):
    earlier = await rig.call()
    starting = tokens.new_call_id()
    rig.interviews.latest_call = starting
    await locks.take_user_lock(rig.redis, USER, starting)

    assert await voice_handoff.hand_off_owed(earlier, trigger="status") == (
        voice_handoff.WAITING
    )


async def test_finishing_an_earlier_call_waits_while_a_later_one_is_unsettled(
    rig: Rig, quality_gate: AsyncMock
):
    earlier = await rig.call(ending=InterviewEnding.PROFILE_CARD.value)
    await rig.call(None)

    with pytest.raises(HTTPException) as refused:
        await _finish(earlier)

    assert refused.value.status_code == 409
    assert refused.value.detail == {"code": voice.CALL_NOT_READY}
    quality_gate.assert_not_awaited()
    assert await rig.redis.get(locks.postcall_claim_key(earlier)) is None


async def test_a_reload_looks_past_a_no_words_call_to_the_one_before(rig: Rig):
    earlier = await rig.call(ending=InterviewEnding.PROFILE_CARD.value)
    later = await rig.call(None)
    await rig.no_words(later, quality.NO_USABLE_SPEECH)

    reloaded = await voice.latest_call(USER)

    assert reloaded.call_id == earlier
    assert reloaded.status == locks.TRANSCRIBED
    # The no-words call itself still answers as it stands.
    read = await voice.call_status(USER, later)
    assert read.hand_off is not None
    assert read.hand_off.error_code == quality.NO_USABLE_SPEECH


async def test_a_reload_answers_a_no_words_call_with_nothing_behind_it(rig: Rig):
    only = await rig.call(None)
    await rig.no_words(only, quality.NO_USABLE_SPEECH)

    reloaded = await voice.latest_call(USER)

    assert reloaded.call_id == only
    assert reloaded.status == locks.FAILED


async def test_a_reload_passes_over_a_call_that_never_went_live(rig: Rig):
    """Even while its post-call work still runs: it has nothing of the
    person's to settle."""
    earlier = await rig.call(ending=InterviewEnding.PROFILE_CARD.value)
    await rig.call(None, end_reason="room_error", went_live=False)

    reloaded = await voice.latest_call(USER)

    assert reloaded.call_id == earlier
    assert reloaded.status == locks.TRANSCRIBED


async def test_a_reload_finds_no_call_when_none_went_live(rig: Rig):
    """A start that failed leaves nothing to come back to: the step opens
    as if no call had been made, not on a call with no words."""
    await rig.call("", end_reason="vendor_error", went_live=False)
    await rig.call("", end_reason="room_error", went_live=False)

    with pytest.raises(HTTPException) as missing:
        await voice.latest_call(USER)

    assert missing.value.status_code == 404
    assert missing.value.detail == voice.NO_SUCH_CALL


async def test_a_reload_answers_a_no_words_call_in_front_of_one_that_never_went_live(
    rig: Rig,
):
    await rig.call("", end_reason="room_error", went_live=False)
    no_words = await rig.call(None)
    await rig.no_words(no_words, quality.NO_USABLE_SPEECH)

    reloaded = await voice.latest_call(USER)

    assert reloaded.call_id == no_words
    assert reloaded.status == locks.FAILED


async def test_a_reload_answers_a_call_still_starting(rig: Rig):
    """A call that has not gone live yet may still: it is answered as it
    stands."""
    await rig.call(ending=InterviewEnding.PROFILE_CARD.value)
    starting = tokens.new_call_id()
    await locks.write_call_record(
        rig.redis,
        starting,
        {
            "uid": USER,
            "status": locks.CONNECTING,
            "started": time.time(),
            "prev": rig.interviews.latest_call,
        },
    )
    rig.interviews.latest_call = starting

    reloaded = await voice.latest_call(USER)

    assert reloaded.call_id == starting
    assert reloaded.status == locks.CONNECTING
