"""The relay's half of a call's post-call work: its transcript of record,
and the sweep that makes the transcript of a call whose pod died.

No onboarding pipeline runs here: the job stops at the stored transcript,
which the REST API hands off (``onboarding_dump.voice_handoff``).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from google.genai import types
from pytest_mock import MockerFixture

from backend.voice_relay import locks, postcall, tokens
from backend.voice_relay.config import (
    CAPS,
    IN_RATE,
    POSTCALL_CLAIM_TTL_S,
    POSTCALL_JOB_TIMEOUT_S,
    TRANSCRIBE_MODEL,
    Timing,
    transcribe_timeout_s,
)
from backend.voice_relay.conftest import MemoryRecordings, ScriptedTranscriber
from backend.voice_relay.locks import TurnMark
from backend.voice_relay.metering import POSTCALLS, Row
from backend.voice_relay.postcall import (
    GeminiTranscriber,
    PostCall,
    Transcription,
    transcript_of_record,
    transcription_text,
    user_turn_clips,
)
from backend.voice_relay.record import CallRecording, wav_of

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


class Rig:
    def __init__(self, redis) -> None:
        self.redis = redis
        self.recordings = MemoryRecordings()
        self.rows: list[Row] = []

    def postcall(
        self,
        transcriber,
        *,
        pod: str = "pod-a",
        attempts: int = 3,
        first_name: str | None = "Dana",
        max_jobs: int = 4,
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
            max_jobs=max_jobs,
        )

    async def call(
        self,
        pcm: bytes,
        marks: list[int],
        *,
        status: str = "ended",
        end_reason: str | None = "finished",
        uploaded: bool = True,
        started: float | None = None,
    ) -> str:
        """A call as the relay leaves it on its record: pending, its recording
        uploaded. It started an hour ago unless *started* says otherwise."""
        call_id = tokens.new_call_id()
        fields: dict[str, object] = {
            "status": status,
            "uid": USER,
            "leg": "websocket",
            "started": time.time() - 3600 if started is None else started,
            "end_reason": end_reason,
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
        return call_id

    async def run(self, job: PostCall, call_id: str, **kwargs) -> locks.CallRecord:
        """Run *call_id*'s job as the pod that ran the call does."""
        assert await locks.claim_postcall(self.redis, call_id, job.pod)
        await job.run(call_id, **kwargs)
        record = await locks.read_call(self.redis, call_id)
        assert record is not None
        return record


@pytest.fixture
def rig(redis) -> Rig:
    return Rig(redis)


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


async def test_the_job_stores_the_transcript_and_leaves_the_call_for_its_hand_off(
    rig: Rig,
):
    pcm, marks = recording_of(answer(), answer())
    call_id = await rig.call(pcm, marks)

    record = await rig.run(rig.postcall(ScriptedTranscriber(SAID)), call_id, pcm=pcm)

    assert record.status == locks.TRANSCRIBED
    assert record.transcript_of_record == "\n".join(SAID)
    assert record.transcribed_at is not None
    assert record.hand_off is None
    assert record.audio_path is None
    assert rig.recordings.objects == {}
    # The claim is given up, and the call is the REST API's to hand off.
    assert await rig.redis.get(locks.postcall_claim_key(call_id)) is None
    assert await locks.postcall_pending(rig.redis) == [call_id]


async def test_the_claim_is_given_up_before_the_recording_is_deleted(
    rig: Rig, mocker: MockerFixture
):
    """The call is the REST API's to hand off once its transcript is stored,
    so a slow deletion holds no hand-off back."""
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    delete = rig.recordings.delete
    claims_while_deleting: list[str | None] = []

    async def noting_the_claim(path: str) -> None:
        claims_while_deleting.append(
            await rig.redis.get(locks.postcall_claim_key(call_id))
        )
        await delete(path)

    mocker.patch.object(rig.recordings, "delete", new=noting_the_claim)

    record = await rig.run(rig.postcall(ScriptedTranscriber(SAID)), call_id, pcm=pcm)

    assert record.status == locks.TRANSCRIBED
    assert claims_while_deleting == [None]
    assert rig.recordings.objects == {}


async def test_the_transcription_names_the_speaker_from_the_account(rig: Rig):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    transcriber = ScriptedTranscriber(SAID[:1])

    await rig.run(rig.postcall(transcriber, first_name="Dana"), call_id, pcm=pcm)

    assert [name for _, name in transcriber.calls] == ["Dana"]


async def test_each_transcribed_turn_is_on_the_calls_bill(rig: Rig):
    pcm, marks = recording_of(answer(), answer())
    call_id = await rig.call(pcm, marks)

    await rig.run(rig.postcall(ScriptedTranscriber(SAID)), call_id, pcm=pcm)

    billed = [(row.model, row.meter, row.tokens) for row in rig.rows]
    assert billed.count((TRANSCRIBE_MODEL, "audio_in", 100)) == 2
    assert billed.count((TRANSCRIBE_MODEL, "text_out", 10)) == 2


async def test_a_near_silent_clip_is_an_empty_turn_and_the_job_completes(rig: Rig):
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
    # The verdict on nothing said is the hand-off's to give.
    assert record.status == locks.TRANSCRIBED
    assert record.transcript_of_record == ""
    assert record.hand_off is None
    assert await locks.postcall_pending(rig.redis) == [call_id]


async def test_a_call_where_nothing_was_heard_stores_an_empty_transcript(rig: Rig):
    silent, marks = recording_of(quiet(2.0))
    call_id = await rig.call(silent, marks)
    transcriber = ScriptedTranscriber(SAID[:1])

    record = await rig.run(rig.postcall(transcriber), call_id, pcm=silent)

    assert transcriber.calls == []
    assert record.status == locks.TRANSCRIBED
    assert record.transcript_of_record == ""


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
    # The last upload stays for the next attempt.
    assert first.audio_path in rig.recordings.objects

    second = await rig.run(job, call_id, pcm=pcm)

    assert second.status == locks.FAILED
    assert second.hand_off is not None
    assert second.hand_off.error_code == postcall.TRANSCRIPTION_UNAVAILABLE
    assert await locks.postcall_pending(rig.redis) == []
    # Nothing reads the recording again, so it goes.
    assert rig.recordings.objects == {}


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


async def test_a_call_whose_hand_off_answered_is_only_cleared(rig: Rig):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    await locks.write_call_record(
        rig.redis,
        call_id,
        {
            "status": locks.HANDED_OFF,
            "hand_off": {"status": "transcribed", "input_mode": "typed"},
        },
    )
    transcriber = ScriptedTranscriber(SAID[:1])

    await rig.run(rig.postcall(transcriber), call_id, pcm=pcm)

    assert transcriber.calls == []
    assert await locks.postcall_pending(rig.redis) == []
    assert rig.recordings.objects == {}


# ---------- the sweep ----------


async def test_a_crashed_call_is_transcribed_once_by_one_of_two_pods(rig: Rig):
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
    assert record.status == locks.TRANSCRIBED
    assert record.end_reason == postcall.CRASHED
    assert record.transcript_of_record == "\n".join(SAID)
    assert record.transcribed_from_upload
    assert rig.recordings.objects == {}
    # Its hand-off is the REST API's.
    assert await locks.postcall_pending(rig.redis) == [call_id]


async def test_the_sweep_leaves_a_call_whose_transcript_is_made(rig: Rig):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    await locks.store_transcript(
        rig.redis, call_id, SAID[0], uid=USER, from_upload=False
    )

    assert await rig.postcall(ScriptedTranscriber()).sweep() == []
    assert await locks.postcall_pending(rig.redis) == [call_id]


async def test_a_live_call_is_left_to_the_pod_running_it(rig: Rig):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks, status="live", end_reason=None)
    await locks.register_live_call(rig.redis, call_id, time.time())

    assert await rig.postcall(ScriptedTranscriber()).sweep() == []
    assert await locks.postcall_pending(rig.redis) == [call_id]


async def test_the_sweep_leaves_a_call_that_may_still_run_to_its_pod(rig: Rig):
    """Out of the live count, as a moment's Redis trouble can leave a call,
    but started too recently to have ended."""
    pcm, marks = recording_of(answer())
    call_id = await rig.call(
        pcm, marks, status="live", end_reason=None, started=time.time()
    )

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
    assert record is not None and record.status == locks.TRANSCRIBED


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


async def test_the_sweep_takes_only_as_many_jobs_as_fit(rig: Rig):
    pcm, marks = recording_of(answer())
    calls = [await rig.call(pcm, marks) for _ in range(3)]
    go_on = asyncio.Event()

    class Held:
        async def __call__(self, wav: bytes, *, first_name: str | None):
            await go_on.wait()
            return Transcription(text=SAID[0])

    job = rig.postcall(Held(), max_jobs=2)

    first = await job.sweep()

    assert len(first) == 2
    assert await job.sweep() == []
    go_on.set()
    await asyncio.gather(*job.jobs.values())
    assert await job.sweep() == [call for call in calls if call not in first]
    await asyncio.gather(*job.jobs.values())
    for call_id in calls:
        record = await locks.read_call(rig.redis, call_id)
        assert record is not None and record.status == locks.TRANSCRIBED


# ---------- one job, once in effect ----------


def test_a_claim_outlives_the_longest_call_and_its_job():
    timing = Timing()
    longest_call = (
        CAPS.max_call_s
        + timing.vendor_connect_timeout_s
        + timing.wrap_grace_s
        + timing.finish_grace_s
    )
    assert POSTCALL_CLAIM_TTL_S > longest_call + POSTCALL_JOB_TIMEOUT_S


def test_a_turn_is_given_a_few_times_its_length_to_transcribe():
    assert transcribe_timeout_s(2.0) == pytest.approx(14.0)
    assert transcribe_timeout_s(30.0) == pytest.approx(70.0)


async def test_a_stalled_transcription_is_given_up_after_its_turns_share(
    rig: Rig, mocker: MockerFixture
):
    mocker.patch.object(postcall, "RETRY_BACKOFF_S", 0.0)
    mocker.patch.object(postcall, "transcribe_timeout_s", lambda audio_s: 0.05)

    class Stalled:
        calls = 0

        async def __call__(self, wav: bytes, *, first_name: str | None):
            Stalled.calls += 1
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)

    record = await rig.run(rig.postcall(Stalled()), call_id, pcm=pcm)

    assert Stalled.calls == 3
    assert record.status == locks.TRANSCRIBING
    assert await locks.postcall_pending(rig.redis) == [call_id]


async def test_a_job_whose_claim_lapsed_under_it_writes_nothing(rig: Rig):
    """The claim lapses while the first job transcribes, and another pod's
    sweep takes the call up and finishes it. The first job then stops: the
    call's transcript is the second job's."""
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    transcribing = asyncio.Event()
    go_on = asyncio.Event()

    class Held:
        async def __call__(self, wav: bytes, *, first_name: str | None):
            transcribing.set()
            await go_on.wait()
            return Transcription(text="Words the first job heard.")

    first = rig.postcall(Held(), pod="pod-a")
    assert await locks.claim_postcall(rig.redis, call_id, "pod-a")
    running = asyncio.create_task(first.run(call_id, pcm=pcm))
    await asyncio.wait_for(transcribing.wait(), timeout=5)

    await rig.redis.delete(locks.postcall_claim_key(call_id))
    second = rig.postcall(ScriptedTranscriber(SAID[:1]), pod="pod-b")
    assert await second.sweep() == [call_id]
    await asyncio.gather(*second.jobs.values())
    go_on.set()
    await asyncio.wait_for(running, timeout=5)

    record = await locks.read_call(rig.redis, call_id)
    assert record is not None
    assert record.status == locks.TRANSCRIBED
    assert record.transcript_of_record == SAID[0]
    assert await locks.postcall_pending(rig.redis) == [call_id]


async def test_a_job_that_cannot_reach_redis_still_deletes_its_spool(
    rig: Rig, tmp_path
):
    call_id = tokens.new_call_id()
    recording = CallRecording(
        call_id=call_id,
        user_id=USER,
        spool_dir=tmp_path,
        store=rig.recordings,
        call_store=locks.CallStore(rig.redis, call_id=call_id, user_id=USER),
    )
    recording.append(tone(0.2))
    assert recording.spool.exists()

    async def down():
        raise ConnectionError("redis is down")

    job = PostCall(
        redis=down,
        recordings=rig.recordings,
        transcriber=ScriptedTranscriber(),
        first_name=AsyncMock(return_value=None),
        live_model=LIVE,
        pod="pod-a",
    )
    await job.run(call_id, recording=recording, pcm=tone(0.2))

    assert not recording.spool.exists()


async def test_a_record_that_names_no_owner_is_never_transcribed(rig: Rig):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks)
    await rig.redis.hdel(locks.call_record_key(call_id), "uid")
    transcriber = ScriptedTranscriber(SAID[:1])

    record = await rig.run(rig.postcall(transcriber), call_id, pcm=pcm)

    assert transcriber.calls == []
    assert record.transcript_of_record is None
    assert await locks.postcall_pending(rig.redis) == []
    assert rig.recordings.objects == {}


# ---------- the recording ----------


async def test_the_whole_recording_is_transcribed_again_over_one_from_an_upload(
    rig: Rig,
):
    """The sweep made the transcript from an upload that stopped short; the
    pod that ran the call holds all of it."""
    pcm, marks = recording_of(answer(), answer())
    call_id = await rig.call(pcm, marks)
    await locks.store_transcript(
        rig.redis, call_id, SAID[0], uid=USER, from_upload=True
    )

    record = await rig.run(rig.postcall(ScriptedTranscriber(SAID)), call_id, pcm=pcm)

    assert record.status == locks.TRANSCRIBED
    assert record.transcript_of_record == "\n".join(SAID)
    assert not record.transcribed_from_upload


async def test_a_transcript_whose_hand_off_answered_is_kept_over_the_recording(
    rig: Rig,
):
    pcm, marks = recording_of(answer(), answer())
    call_id = await rig.call(pcm, marks)
    await locks.store_transcript(
        rig.redis, call_id, SAID[0], uid=USER, from_upload=True
    )
    await locks.write_call_record(
        rig.redis,
        call_id,
        {
            "status": locks.HANDED_OFF,
            "hand_off": {"status": "transcribed", "input_mode": "typed"},
        },
    )
    transcriber = ScriptedTranscriber(SAID)

    record = await rig.run(rig.postcall(transcriber), call_id, pcm=pcm)

    assert transcriber.calls == []
    assert record.transcript_of_record == SAID[0]
    assert record.status == locks.HANDED_OFF


async def test_a_job_for_a_call_already_transcribed_still_deletes_its_recording(
    rig: Rig, tmp_path
):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks, uploaded=False)
    await locks.store_transcript(
        rig.redis, call_id, SAID[0], uid=USER, from_upload=False
    )
    recording = CallRecording(
        call_id=call_id,
        user_id=USER,
        spool_dir=tmp_path,
        store=rig.recordings,
        call_store=locks.CallStore(rig.redis, call_id=call_id, user_id=USER),
    )
    recording.append(pcm)
    assert await recording.checkpoint()
    assert rig.recordings.objects

    await rig.run(rig.postcall(ScriptedTranscriber()), call_id, recording=recording)

    assert rig.recordings.objects == {}
    assert not recording.spool.exists()


def _outcomes(outcome: str) -> float:
    return POSTCALLS.labels(outcome=outcome)._value.get()


async def test_a_call_whose_recording_was_lost_says_so(rig: Rig):
    """The relay took the person's audio, and no recording of it is left: no
    verdict on their words, so another call or typing takes them."""
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks, uploaded=False)
    transcriber = ScriptedTranscriber(SAID[:1])
    before = _outcomes(postcall.RECORDING_UNAVAILABLE)

    record = await rig.run(rig.postcall(transcriber), call_id, pcm=b"")

    assert record.status == locks.FAILED
    assert record.hand_off is not None
    assert record.hand_off.error_code == postcall.RECORDING_UNAVAILABLE
    assert record.transcript_of_record is None
    assert transcriber.calls == []
    assert _outcomes(postcall.RECORDING_UNAVAILABLE) == before + 1
    assert await locks.postcall_pending(rig.redis) == []


async def test_a_crashed_call_with_no_upload_left_says_its_recording_was_lost(
    rig: Rig,
):
    pcm, marks = recording_of(answer())
    call_id = await rig.call(pcm, marks, status="live", end_reason=None, uploaded=False)
    job = rig.postcall(ScriptedTranscriber())

    assert await job.sweep() == [call_id]
    await asyncio.gather(*job.jobs.values())

    record = await locks.read_call(rig.redis, call_id)
    assert record is not None and record.hand_off is not None
    assert record.hand_off.error_code == postcall.RECORDING_UNAVAILABLE


async def test_a_call_that_took_none_of_the_persons_audio_has_an_empty_transcript(
    rig: Rig,
):
    pcm, marks = recording_of()
    call_id = await rig.call(pcm, marks, uploaded=False)

    record = await rig.run(rig.postcall(ScriptedTranscriber()), call_id, pcm=b"")

    assert record.status == locks.TRANSCRIBED
    assert record.transcript_of_record == ""
