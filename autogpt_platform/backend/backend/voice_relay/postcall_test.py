"""The post-call job: the transcript of record, its hand-off, and the sweep
that finishes what a dead pod left.

The hand-off runs for real into the onboarding pipeline over the in-memory
dump table the pipeline tests use, so these assert what reaches the take.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException
from google.genai import types
from prisma.enums import BrainDumpStatus
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import interview, quality, voice
from backend.api.features.onboarding_dump.interview_endings import InterviewEnding
from backend.api.features.onboarding_dump.interview_test import jobs
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
    INTERVIEW_STAGE_HANDED_OFF,
    ONBOARDING_INTERVIEW_KIND,
    ChatSession,
)
from backend.voice_relay import handoff, locks, postcall, tokens
from backend.voice_relay.config import IN_RATE, TRANSCRIBE_MODEL
from backend.voice_relay.conftest import MemoryRecordings, ScriptedTranscriber
from backend.voice_relay.locks import TurnMark
from backend.voice_relay.metering import Row
from backend.voice_relay.postcall import (
    GeminiTranscriber,
    PostCall,
    transcript_of_record,
    transcription_text,
    user_turn_clips,
)
from backend.voice_relay.record import wav_of

__all__ = [
    "dumps",
    "extraction",
    "jobs",
    "personalization_admission",
    "quality_gate",
    "storage_mocks",
    "transcribe",
]

USER = "user-1"
LIVE = "gemini-3.8-live"
SAID = ["I run a bakery with five staff.", "Answering order emails, mostly."]


def tone(seconds: float, amplitude: float = 8000) -> bytes:
    samples = np.arange(int(IN_RATE * seconds))
    wave = np.sin(samples * 2 * np.pi * 300 / IN_RATE) * amplitude
    return wave.astype(np.int16).tobytes()


def quiet(seconds: float) -> bytes:
    return bytes(int(IN_RATE * seconds) * 2)


def answer(speech_s: float = 0.6) -> bytes:
    """One answer of the person's: the moment they take to start, speech, then
    the pause Otto answers in."""
    return quiet(0.4) + tone(speech_s) + quiet(0.8)


def recording_of(*answers: bytes) -> tuple[bytes, list[int]]:
    """The person's recording, and where Otto's turns ended in it: Otto greets
    before they speak, then answers each answer."""
    pcm, marks = b"", [0]
    for spoken in answers:
        pcm += spoken
        marks.append(len(pcm) // 2)
    return pcm, marks


class Interviews:
    """What the person's onboarding row and chat store hold of their
    interviews."""

    def __init__(self) -> None:
        self.latest_call: str | None = None
        self.typed: ChatSession | None = None


@pytest.fixture(autouse=True)
def interviews(mocker: MockerFixture, redis) -> Interviews:
    state = Interviews()

    async def latest_call(user_id: str) -> str | None:
        return state.latest_call

    async def typed(user_id: str) -> ChatSession | None:
        return state.typed

    mocker.patch.object(
        interview,
        "get_onboarding_voice_call_id",
        new=AsyncMock(side_effect=latest_call),
    )
    mocker.patch.object(
        interview,
        "find_onboarding_interview_session",
        new=AsyncMock(side_effect=typed),
    )
    mocker.patch.object(interview, "get_redis_async", AsyncMock(return_value=redis))
    mocker.patch.object(voice, "get_redis_async", AsyncMock(return_value=redis))
    return state


@pytest.fixture(autouse=True)
def arm(mocker: MockerFixture) -> AsyncMock:
    """The person's arm of the voice ending experiment: none unless a test
    says otherwise."""
    return mocker.patch.object(
        interview, "assigned_ending", new=AsyncMock(return_value=None)
    )


class Rig:
    def __init__(self, redis, interviews: Interviews) -> None:
        self.redis = redis
        self.interviews = interviews
        self.recordings = MemoryRecordings()
        self.rows: list[Row] = []

    def postcall(
        self,
        transcriber,
        *,
        pod: str = "pod-a",
        attempts: int = 3,
        first_name: str | None = "Dana",
    ) -> PostCall:
        async def get_redis():
            return self.redis

        async def name(user_id: str) -> str | None:
            return first_name

        async def sink(row: Row) -> None:
            self.rows.append(row)

        return PostCall(
            redis=get_redis,
            recordings=self.recordings,
            transcriber=transcriber,
            first_name=name,
            live_model=LIVE,
            pod=pod,
            cost_sink=sink,
            attempts=attempts,
        )

    async def call(
        self,
        pcm: bytes,
        marks: list[int],
        *,
        status: str = "ended",
        end_reason: str | None = "finished",
        first_task: str | None = None,
        uploaded: bool = True,
    ) -> str:
        """A call as the relay leaves it on its record: pending its job, the
        person's latest, its recording uploaded."""
        call_id = tokens.new_call_id()
        fields: dict[str, object] = {
            "status": status,
            "uid": USER,
            "leg": "websocket",
            "end_reason": end_reason,
            "first_task": first_task,
            "turns": [
                TurnMark(index=index, at=at).model_dump()
                for index, at in enumerate(marks)
            ],
        }
        if uploaded and pcm:
            fields["audio_path"] = await self.recordings.store(
                USER, wav_of(pcm), f"brain-dump-voice-{call_id}.wav"
            )
        await locks.write_call_record(self.redis, call_id, fields)
        await locks.add_postcall_pending(self.redis, call_id)
        self.interviews.latest_call = call_id
        return call_id

    async def run(self, job: PostCall, call_id: str, **kwargs) -> locks.CallRecord:
        """Run *call_id*'s job as the pod that ran the call does."""
        assert await locks.claim_postcall(self.redis, call_id, job.pod)
        await job.run(call_id, **kwargs)
        record = await locks.read_call(self.redis, call_id)
        assert record is not None
        return record


@pytest.fixture
def rig(redis, interviews: Interviews) -> Rig:
    return Rig(redis, interviews)


# ---------- the person's turns ----------


def test_speech_goes_to_the_turn_it_falls_in():
    pcm, marks = recording_of(answer(0.6), quiet(1.0), answer(0.9))

    clips = user_turn_clips(pcm, marks)

    assert len(clips) == 4
    assert len(clips[0]) / 2 / IN_RATE == pytest.approx(0.6 + 0.4, abs=0.05)
    assert clips[1] == b""
    assert len(clips[2]) / 2 / IN_RATE == pytest.approx(0.9 + 0.4, abs=0.05)
    assert clips[3] == b""


def test_speech_over_the_greeting_joins_the_answer_to_it():
    """Talking over Otto's greeting, then answering it, is one turn."""
    pcm = answer(0.5) + answer(0.5)
    greeting_ended = int(1.0 * IN_RATE)

    clips = user_turn_clips(pcm, [greeting_ended, len(pcm) // 2])

    # Both stretches of speech, padded, with the silence between them.
    assert len(clips[0]) / 2 / IN_RATE == pytest.approx(0.9 + 0.3 + 0.9, abs=0.05)
    assert clips[1] == b""


def test_an_answer_that_starts_as_ottos_turn_ends_answers_that_turn():
    """The padding kept around speech reaches back past Otto's turn; the
    speech itself does not."""
    first = tone(0.6) + quiet(0.8)
    second = tone(0.6) + quiet(0.8)
    pcm = first + second

    clips = user_turn_clips(pcm, [0, len(first) // 2, len(pcm) // 2])

    assert len(clips[0]) / 2 / IN_RATE == pytest.approx(0.6 + 0.2, abs=0.05)
    assert len(clips[1]) / 2 / IN_RATE == pytest.approx(0.6 + 0.4, abs=0.05)
    assert clips[2] == b""


def test_a_call_with_no_turn_marked_is_one_turn():
    pcm = answer(0.5) + answer(0.5)
    [clip] = user_turn_clips(pcm, [])
    assert clip


def test_a_click_is_no_speech():
    assert user_turn_clips(quiet(0.5) + tone(0.05) + quiet(0.5), [0]) == [b""]


# ---------- transcribing ----------


@pytest.mark.parametrize(
    "response",
    [
        types.GenerateContentResponse(candidates=[]),
        types.GenerateContentResponse(candidates=None),
        types.GenerateContentResponse(candidates=[types.Candidate(content=None)]),
        types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=None))]
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        parts=[
                            types.Part(text="  "),
                            types.Part(text="Thinking it over.", thought=True),
                        ]
                    )
                )
            ]
        ),
    ],
    ids=["no-candidates", "candidates-none", "no-content", "no-parts", "no-text"],
)
def test_a_response_with_no_candidates_or_no_text_is_an_empty_turn(response):
    assert transcription_text(response) == ""


def test_the_transcription_is_read_from_its_transcription_parts():
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[
                        types.Part(
                            audio_transcription=types.Transcription(text="Hola, ")
                        ),
                        types.Part(text="tengo una panadería."),
                    ]
                )
            )
        ]
    )
    assert transcription_text(response) == "Hola, tengo una panadería."


def _gemini(response: types.GenerateContentResponse) -> tuple[MagicMock, AsyncMock]:
    client = MagicMock()
    generate = AsyncMock(return_value=response)
    client.aio.models.generate_content = generate
    return client, generate


@pytest.mark.parametrize(
    "first_name,prompt",
    [
        ("Dana", "Transcribe the speech verbatim. The speaker's name is Dana."),
        (None, "Transcribe the speech verbatim."),
    ],
    ids=["named", "no-name"],
)
async def test_the_transcription_prompt_names_the_speaker(first_name, prompt):
    client, generate = _gemini(
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(parts=[types.Part(text="I bake bread.")])
                )
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=40, candidates_token_count=5
            ),
        )
    )

    result = await GeminiTranscriber(client)(wav_of(tone(0.5)), first_name=first_name)

    assert result.text == "I bake bread."
    assert (result.tokens_in, result.tokens_out) == (40, 5)
    request = generate.await_args.kwargs
    assert request["model"] == TRANSCRIBE_MODEL
    audio, text = request["contents"]
    assert audio.inline_data.mime_type == "audio/wav"
    assert text == prompt


def test_server_tags_never_reach_the_transcript_of_record():
    texts = [
        "<user_context>\nPlan: Enterprise\n</user_context> I run a bakery.",
        "",
        "Answering\norder emails.",
    ]
    assert transcript_of_record(texts) == "I run a bakery.\nAnswering order emails."


# ---------- the job ----------


async def test_an_accepted_take_hands_the_persons_words_off_and_closes_the_interview(
    rig: Rig, dumps: DumpStore, jobs: dict[str, AsyncMock]
):
    pcm, marks = recording_of(answer(), answer())
    call_id = await rig.call(pcm, marks)
    transcriber = ScriptedTranscriber(SAID)

    record = await rig.run(rig.postcall(transcriber), call_id, pcm=pcm)

    assert record.status == locks.HANDED_OFF
    assert record.hand_off is not None
    assert record.hand_off.status == BrainDumpStatus.transcribed
    assert record.transcript_of_record == "\n".join(SAID)
    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{call_id}"
    assert dumps.row.transcript == "\n".join(SAID)
    # The pipeline's own jobs ran on the same words.
    jobs["greeting"].assert_awaited_once_with("\n".join(SAID))
    assert await locks.postcall_pending(rig.redis) == []
    assert await interview.call_that_took_the_interview(USER) == call_id


async def test_the_transcription_names_the_speaker_from_the_account(rig: Rig):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    transcriber = ScriptedTranscriber(SAID[:1])

    await rig.run(rig.postcall(transcriber, first_name="Dana"), call_id, pcm=pcm)

    assert [name for _, name in transcriber.calls] == ["Dana"]


async def test_the_hand_off_is_told_why_the_call_ended_and_the_task_it_agreed(
    rig: Rig, mocker: MockerFixture
):
    close = mocker.patch.object(
        handoff,
        "close_voice_interview",
        new=AsyncMock(
            return_value=interview.InterviewClosure(ending=None, hand_off=None)
        ),
    )
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks, first_task="Answer order emails")

    await rig.run(rig.postcall(ScriptedTranscriber(SAID[:1])), call_id, pcm=pcm)

    close.assert_awaited_once()
    assert close.await_args.args[:3] == (USER, call_id, SAID[0])
    assert close.await_args.kwargs["reason"] == "confirmed"
    assert close.await_args.kwargs["first_task"] == "Answer order emails"


async def test_the_recording_is_deleted_once_the_transcript_is_stored(rig: Rig):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)

    record = await rig.run(rig.postcall(ScriptedTranscriber(SAID[:1])), call_id)

    assert rig.recordings.objects == {}
    assert record.audio_path is None
    assert record.transcript_of_record == SAID[0]


async def test_each_transcribed_turn_is_on_the_calls_bill(rig: Rig):
    pcm, marks = recording_of(answer(), answer())
    call_id = await rig.call(pcm, marks)

    await rig.run(rig.postcall(ScriptedTranscriber(SAID)), call_id, pcm=pcm)

    billed = [(row.model, row.meter, row.tokens) for row in rig.rows]
    assert billed.count((TRANSCRIBE_MODEL, "audio_in", 100)) == 2
    assert billed.count((TRANSCRIBE_MODEL, "text_out", 10)) == 2


async def test_a_near_silent_clip_is_an_empty_turn_and_the_job_completes(
    rig: Rig, dumps: DumpStore
):
    """A 0.7 s clip barely above the speech threshold comes back from the
    transcription model with no candidates at all."""
    pcm, marks = recording_of(tone(0.7, amplitude=300) + quiet(0.8))
    call_id = await rig.call(pcm, marks)
    client, generate = _gemini(
        types.GenerateContentResponse(
            candidates=[],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=18
            ),
        )
    )

    record = await rig.run(rig.postcall(GeminiTranscriber(client)), call_id, pcm=pcm)

    generate.assert_awaited_once()
    assert record.status == locks.FAILED
    assert record.hand_off is not None
    assert record.hand_off.error_code == quality.NO_USABLE_SPEECH
    assert record.transcript_of_record == ""
    assert dumps.row is None
    assert await locks.postcall_pending(rig.redis) == []


async def test_a_call_where_nothing_was_heard_leaves_the_interview_open_for_another(
    rig: Rig, dumps: DumpStore
):
    silent, silent_marks = recording_of(quiet(2.0))
    first = await rig.call(silent, silent_marks)
    job = rig.postcall(ScriptedTranscriber(SAID[:1]))

    record = await rig.run(job, first, pcm=silent)

    assert record.status == locks.FAILED
    assert record.hand_off is not None
    assert record.hand_off.error_code == quality.NO_USABLE_SPEECH
    assert await interview.call_that_took_the_interview(USER) is None

    pcm, marks = recording_of(answer())
    second = await rig.call(pcm, marks)
    record = await rig.run(job, second, pcm=pcm)

    assert record.status == locks.HANDED_OFF
    assert dumps.row is not None
    assert dumps.row.recordingId == f"interview-{second}"


async def test_a_verdict_on_the_words_fails_the_call_and_leaves_the_interview_open(
    rig: Rig, quality_gate: AsyncMock
):
    quality_gate.return_value = quality.QualityRejection(
        error_code=quality.INSUFFICIENT_CONTENT, reason="judge_unusable"
    )
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)

    record = await rig.run(rig.postcall(ScriptedTranscriber(["idk"])), call_id, pcm=pcm)

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
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)

    record = await rig.run(
        rig.postcall(ScriptedTranscriber(SAID[:1])), call_id, pcm=pcm
    )

    assert record.status == locks.FAILED
    assert record.hand_off is not None
    assert record.hand_off.error_code == quality.QUALITY_CHECK_UNAVAILABLE
    # The words stay on the call for the retry.
    assert record.transcript_of_record == SAID[0]
    assert await interview.call_that_took_the_interview(USER) is None

    background = BackgroundTasks()
    again = await voice.finish_call(USER, call_id, background)
    await background()

    assert again.status == BrainDumpStatus.transcribed
    assert quality_gate.await_count == 2
    record = await locks.read_call(rig.redis, call_id)
    assert record is not None and record.status == locks.HANDED_OFF
    assert dumps.row is not None and dumps.row.transcript == SAID[0]


async def test_the_pipeline_refusing_for_now_is_recorded_for_the_finish_route(
    rig: Rig, personalization_admission: AsyncMock
):
    personalization_admission.side_effect = HTTPException(
        status_code=429, headers={"Retry-After": "60"}
    )
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)

    record = await rig.run(
        rig.postcall(ScriptedTranscriber(SAID[:1])), call_id, pcm=pcm
    )

    assert record.status == locks.FAILED
    assert record.hand_off is not None
    assert record.hand_off.error_code == "rate_limited"
    assert record.transcript_of_record == SAID[0]
    assert await locks.postcall_pending(rig.redis) == []


async def test_a_confirm_arm_waits_and_the_finish_route_completes_the_hand_off(
    rig: Rig, arm: AsyncMock, dumps: DumpStore
):
    arm.return_value = InterviewEnding.PROFILE_CARD.value
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)

    record = await rig.run(
        rig.postcall(ScriptedTranscriber(SAID[:1])), call_id, pcm=pcm
    )

    assert record.status == locks.TRANSCRIBED
    assert record.hand_off is None
    assert dumps.row is None

    background = BackgroundTasks()
    confirmed = await voice.finish_call(
        USER, call_id, background, corrections="Eight staff, not five."
    )
    await background()

    assert confirmed.status == BrainDumpStatus.transcribed
    assert dumps.row is not None
    assert dumps.row.transcript == f"{SAID[0]}\nEight staff, not five."
    record = await locks.read_call(rig.redis, call_id)
    assert record is not None and record.status == locks.HANDED_OFF


async def test_a_call_the_person_has_called_again_since_is_superseded(
    rig: Rig, dumps: DumpStore
):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    rig.interviews.latest_call = tokens.new_call_id()

    record = await rig.run(
        rig.postcall(ScriptedTranscriber(SAID[:1])), call_id, pcm=pcm
    )

    assert record.status == locks.HANDED_OFF
    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    assert dumps.row is None


async def test_a_call_after_the_typed_interview_ended_is_superseded(
    rig: Rig, dumps: DumpStore
):
    typed = ChatSession.new(USER, dry_run=False)
    typed.metadata.kind = ONBOARDING_INTERVIEW_KIND
    typed.metadata.onboarding_interview_stage = INTERVIEW_STAGE_HANDED_OFF
    rig.interviews.typed = typed
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)

    record = await rig.run(
        rig.postcall(ScriptedTranscriber(SAID[:1])), call_id, pcm=pcm
    )

    assert record.hand_off is not None
    assert record.hand_off.error_code == interview.SUPERSEDED
    assert dumps.row is None


async def test_a_job_that_fails_is_left_for_another_attempt_then_records_the_failure(
    rig: Rig, mocker: MockerFixture
):
    mocker.patch.object(postcall, "RETRY_BACKOFF_S", 0.0)
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    down = ConnectionError("transcription is down")
    job = rig.postcall(ScriptedTranscriber([down] * 6), attempts=2)

    first = await rig.run(job, call_id, pcm=pcm)

    assert first.status == locks.TRANSCRIBING
    assert await locks.postcall_pending(rig.redis) == [call_id]
    # The claim is given up, so the next sweep tries again.
    assert await rig.redis.get(locks.postcall_claim_key(call_id)) is None

    second = await rig.run(job, call_id, pcm=pcm)

    assert second.status == locks.FAILED
    assert second.hand_off is not None
    assert second.hand_off.error_code == postcall.TRANSCRIPTION_UNAVAILABLE
    assert await locks.postcall_pending(rig.redis) == []
    # The recording stays for the retention sweep.
    assert second.audio_path in rig.recordings.objects


async def test_a_turn_whose_transcription_fails_once_is_tried_again(
    rig: Rig, mocker: MockerFixture
):
    mocker.patch.object(postcall, "RETRY_BACKOFF_S", 0.0)
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    transcriber = ScriptedTranscriber([TimeoutError(), SAID[0]])

    record = await rig.run(rig.postcall(transcriber), call_id, pcm=pcm)

    assert len(transcriber.calls) == 2
    assert record.transcript_of_record == SAID[0]


async def test_a_call_whose_answer_was_recorded_is_only_cleared(rig: Rig):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    await locks.write_call_record(rig.redis, call_id, {"status": locks.HANDED_OFF})
    transcriber = ScriptedTranscriber(SAID[:1])

    await rig.run(rig.postcall(transcriber), call_id)

    assert transcriber.calls == []
    assert await locks.postcall_pending(rig.redis) == []


# ---------- the sweep ----------


async def test_a_crashed_call_is_finished_once_by_one_of_two_pods(
    rig: Rig, dumps: DumpStore
):
    """A pod died mid-call: its call never ended, and its recording was last
    uploaded at a checkpoint."""
    pcm, marks = recording_of(answer(), answer())
    call_id = await rig.call(pcm, marks, status="live", end_reason=None)
    await locks.register_live_call(rig.redis, call_id, time.time() - 600)
    transcriber = ScriptedTranscriber(SAID)
    pods = [
        rig.postcall(transcriber, pod="pod-a"),
        rig.postcall(transcriber, pod="pod-b"),
    ]

    started = await asyncio.gather(*(pod.sweep() for pod in pods))
    await asyncio.gather(*(job for pod in pods for job in pod.jobs.values()))

    assert sorted(len(calls) for calls in started) == [0, 1]
    assert len(transcriber.calls) == 2
    record = await locks.read_call(rig.redis, call_id)
    assert record is not None
    assert record.status == locks.HANDED_OFF
    assert record.end_reason == postcall.CRASHED
    assert dumps.row is not None and dumps.row.transcript == "\n".join(SAID)
    assert await locks.postcall_pending(rig.redis) == []
    assert rig.recordings.objects == {}


async def test_a_live_call_is_left_to_the_pod_running_it(rig: Rig):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks, status="live", end_reason=None)
    await locks.register_live_call(rig.redis, call_id, time.time())

    assert await rig.postcall(ScriptedTranscriber()).sweep() == []
    assert await locks.postcall_pending(rig.redis) == [call_id]


async def test_a_job_another_pod_holds_is_taken_up_once_its_claim_lapses(rig: Rig):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    await locks.claim_postcall(rig.redis, call_id, "pod-that-died")
    job = rig.postcall(ScriptedTranscriber(SAID[:1]))

    assert await job.sweep() == []

    await rig.redis.delete(locks.postcall_claim_key(call_id))
    assert await job.sweep() == [call_id]
    await asyncio.gather(*job.jobs.values())
    record = await locks.read_call(rig.redis, call_id)
    assert record is not None and record.status == locks.HANDED_OFF


async def test_a_pending_call_whose_record_expired_is_dropped(rig: Rig):
    await locks.add_postcall_pending(rig.redis, "f" * 32)
    job = rig.postcall(ScriptedTranscriber())

    assert await job.sweep() == ["f" * 32]
    await asyncio.gather(*job.jobs.values())

    assert await locks.postcall_pending(rig.redis) == []


async def test_a_draining_process_takes_no_other_pods_jobs(rig: Rig):
    pcm, marks = recording_of(answer())
    await rig.call(pcm, marks)
    job = rig.postcall(ScriptedTranscriber())
    job.stopped = True

    assert await job.sweep() == []
