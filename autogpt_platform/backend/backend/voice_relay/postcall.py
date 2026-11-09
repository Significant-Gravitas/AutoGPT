"""After a call: the transcript of record, and its hand-off to the interview.

The live model's captions are what it heard as it went; the transcript of
record is made after the call from the person's own recording, and it is
what the onboarding pipeline reads. A call's post-call job:

1. cuts the recording into speech (``vad.segments``) and groups it into the
   person's turns by where it falls between Otto's turns, which the call
   marked on its record as they ended (:func:`user_turn_clips`);
2. transcribes each turn, four at a time, naming the speaker
   (:class:`GeminiTranscriber`). A response with no candidates or no text is
   an empty turn, never an error: a near-silent clip comes back that way;
3. joins the turns one per line without the server's tags
   (:func:`transcript_of_record`) and stores the transcript on the call's
   record in the same write that forgets the recording, which is then
   deleted;
4. hands the transcript to the person's interview and records the answer on
   the call (``handoff``): ``handed_off`` once it closes the interview,
   ``failed`` with the pipeline's answer while the interview stays open, or
   ``transcribed`` while a confirm arm waits for the person.

Every call is pending from the moment it starts (``voice:postcall:pending``)
until its job records an answer, and each job runs under a claim
(``voice:postcall:claim:<call id>``), so it runs once across pods. The pod a
call ran on claims its job before the call stops counting as live. Every pod
sweeps the pending calls when it starts and every ``POSTCALL_SWEEP_S``, and
takes any no pod is running: a call whose pod died during the call or its
job, or a job that failed and waits for another attempt. A call whose pod
died is finished from its last uploaded recording, as ``crashed``.
"""

import asyncio
import bisect
import logging
import time
import traceback
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from fastapi import BackgroundTasks, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict

from backend.api.features.onboarding_dump.interview import InterviewClosure
from backend.copilot.service import strip_server_injected_tags
from backend.data.redis_client import AsyncRedisClient
from backend.util.gcs_utils import is_not_found_error
from backend.voice_relay import locks, vad
from backend.voice_relay.config import (
    IN_RATE,
    POSTCALL_ATTEMPTS,
    POSTCALL_SWEEP_S,
    TRANSCRIBE_ATTEMPTS,
    TRANSCRIBE_CONCURRENCY,
    TRANSCRIBE_MODEL,
    TRANSCRIBE_TIMEOUT_S,
)
from backend.voice_relay.handoff import (
    PERSONALIZATION_UNAVAILABLE,
    call_status_after,
    failed_hand_off,
    hand_off_call,
    record_hand_off,
    refused_hand_off,
)
from backend.voice_relay.locks import CallRecord
from backend.voice_relay.metering import (
    POSTCALL_SECONDS,
    POSTCALLS,
    CallMeter,
    CostSink,
)
from backend.voice_relay.record import (
    CallRecording,
    RecordingStore,
    delete_quietly,
    pcm_of,
    wav_of,
)

logger = logging.getLogger(__name__)

# The end reason of a call whose pod died before it could end it.
CRASHED = "crashed"

# What a call records when its post-call job gives up before a transcript was
# made. One that gives up after records ``PERSONALIZATION_UNAVAILABLE``, which
# the call's finish route can hand off again.
TRANSCRIPTION_UNAVAILABLE = "transcription_unavailable"

# Audio kept on each side of a segment of speech, and the silence between
# the segments of one turn's clip.
SEGMENT_PAD_S = 0.2
SEGMENT_GAP_S = 0.3
# Between a turn's transcription attempts, times the attempt number.
RETRY_BACKOFF_S = 1.0
# How long a Redis operation of the job's bookkeeping may take.
REDIS_TIMEOUT_S = 2.0

# --- the transcript of record ----------------------------------------------


class Transcription(BaseModel):
    """One turn's transcription, and what it cost."""

    model_config = ConfigDict(frozen=True)

    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class Transcriber(Protocol):
    async def __call__(self, wav: bytes, *, first_name: str | None) -> Transcription:
        """Transcribe one turn of the person's, given as a WAV file."""
        ...


def transcription_prompt(first_name: str | None) -> str:
    prompt = "Transcribe the speech verbatim."
    return f"{prompt} The speaker's name is {first_name}." if first_name else prompt


def transcription_text(response: types.GenerateContentResponse) -> str:
    """The text of a transcription, or ``""`` when the response has no
    candidates or no text in them, as for a near-silent clip.

    The transcription model answers in ``audio_transcription`` parts, which
    ``response.text`` skips; a plain text part is read too, and a thought
    is not.
    """
    candidates = response.candidates or []
    content = candidates[0].content if candidates else None
    pieces: list[str] = []
    for part in (content.parts if content else None) or []:
        if part.audio_transcription is not None and part.audio_transcription.text:
            pieces.append(part.audio_transcription.text)
        elif part.text and not part.thought:
            pieces.append(part.text)
    return " ".join(piece.strip() for piece in pieces if piece.strip())


class GeminiTranscriber:
    def __init__(self, client: genai.Client, *, model: str = TRANSCRIBE_MODEL) -> None:
        self._client = client
        self._model = model

    async def __call__(self, wav: bytes, *, first_name: str | None) -> Transcription:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                transcription_prompt(first_name),
            ],
        )
        usage = response.usage_metadata
        return Transcription(
            text=transcription_text(response),
            tokens_in=(usage.prompt_token_count or 0) if usage else 0,
            tokens_out=(
                (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
                if usage
                else 0
            ),
        )


def user_turn_clips(pcm: bytes, boundaries: Sequence[int]) -> list[bytes]:
    """The person's speech in a call's recording, one clip per turn of theirs.

    *boundaries* are where Otto's turns ended, in samples of the recording.
    Each segment of speech goes to the stretch its speech starts in, from the
    end of one of Otto's turns to the end of the next, so a turn of the
    person's is what they said between two of Otto's. What they said before
    Otto's first turn ended joins their answer to it, and what they said after
    the last joins their answer to that. Each segment is padded on both sides
    for the transcription, and a turn's segments are joined with a short
    silence; a turn with no speech is an empty clip.
    """
    slots = max(1, len(boundaries))
    grouped: list[list[tuple[int, int]]] = [[] for _ in range(slots)]
    pad = int(SEGMENT_PAD_S * IN_RATE)
    total = len(pcm) // 2
    for start, end in vad.segments(pcm, pad_s=0.0):
        slot = min(max(bisect.bisect_right(boundaries, start) - 1, 0), slots - 1)
        grouped[slot].append((max(0, start - pad), min(total, end + pad)))
    gap = bytes(int(SEGMENT_GAP_S * IN_RATE) * 2)
    return [
        gap.join(pcm[start * 2 : end * 2] for start, end in segments)
        for segments in grouped
    ]


def transcript_of_record(texts: Iterable[str]) -> str:
    """The person's words from a call, one turn per line.

    They reach the onboarding pipeline as the person's own words, so the
    server's tags are taken out of them first, as the typed interview's are:
    nothing said on a call can pass for a block the server writes. Empty
    turns leave no line.
    """
    lines = (" ".join(strip_server_injected_tags(text).split()) for text in texts)
    return "\n".join(line for line in lines if line)


async def transcribe_turns(
    transcriber: Transcriber,
    clips: Sequence[bytes],
    *,
    first_name: str | None,
    meter: CallMeter,
    call_id: str,
) -> list[str]:
    """Each clip's text, *TRANSCRIBE_CONCURRENCY* at a time; an empty clip is
    not sent. Each transcription is metered on the call's bill. Raises when a
    turn still fails after its attempts, so no turn is ever dropped from a
    transcript that gets handed off."""
    semaphore = asyncio.Semaphore(TRANSCRIBE_CONCURRENCY)

    async def one(clip: bytes) -> str:
        if not clip:
            return ""
        async with semaphore:
            transcription = await _transcribe(
                transcriber, wav_of(clip), first_name, call_id
            )
        await meter.transcription_usage(
            tokens_in=transcription.tokens_in, tokens_out=transcription.tokens_out
        )
        return transcription.text

    async with asyncio.TaskGroup() as group:
        texts = [group.create_task(one(clip)) for clip in clips]
    return [task.result() for task in texts]


async def _transcribe(
    transcriber: Transcriber, wav: bytes, first_name: str | None, call_id: str
) -> Transcription:
    for attempt in range(1, TRANSCRIBE_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(
                transcriber(wav, first_name=first_name), timeout=TRANSCRIBE_TIMEOUT_S
            )
        except Exception as error:
            if attempt == TRANSCRIBE_ATTEMPTS:
                raise
            logger.info(
                f"Voice call {call_id}: a turn's transcription failed "
                f"({type(error).__name__}), trying again"
            )
        await asyncio.sleep(RETRY_BACKOFF_S * attempt)
    raise AssertionError("unreachable")


# --- the job and the sweep -------------------------------------------------


HandOff = Callable[
    [AsyncRedisClient, CallRecord, BackgroundTasks], Awaitable[InterviewClosure]
]


class PostCall:
    """The post-call jobs of one relay process, and its sweep for jobs no
    process is running.

    *pod* names this process in the claims; a call that ran here claims its
    job with the same name (``locks.CallStore``). *hand_off* hands a call's
    transcript to the interview and records the answer (:func:`hand_off_call`).
    """

    def __init__(
        self,
        *,
        redis: Callable[[], Awaitable[AsyncRedisClient]],
        recordings: RecordingStore,
        transcriber: Transcriber,
        first_name: Callable[[str], Awaitable[str | None]],
        live_model: str,
        pod: str,
        cost_sink: CostSink | None = None,
        hand_off: HandOff = hand_off_call,
        sweep_s: float = POSTCALL_SWEEP_S,
        attempts: int = POSTCALL_ATTEMPTS,
    ) -> None:
        self._redis = redis
        self.recordings = recordings
        self.transcriber = transcriber
        self._first_name = first_name
        self.live_model = live_model
        self.pod = pod
        self.cost_sink = cost_sink
        self.hand_off = hand_off
        self.sweep_s = sweep_s
        self.attempts = attempts
        # The jobs running in this process, by call.
        self.jobs: dict[str, asyncio.Task] = {}
        # Set while the process drains: it takes no more jobs.
        self.stopped = False

    def after_call(
        self,
        *,
        call_id: str,
        recording: CallRecording | None,
        claimed: bool,
        first_name: str | None,
    ) -> None:
        """Finish a call this process ran: upload the rest of its recording,
        then run its job if this process claimed it. A job this process
        could not claim is the sweep's."""
        self._start(call_id, self._after_call(call_id, recording, claimed, first_name))

    async def _after_call(
        self,
        call_id: str,
        recording: CallRecording | None,
        claimed: bool,
        first_name: str | None,
    ) -> None:
        pcm = await recording.close() if recording is not None else None
        if claimed:
            await self.run(call_id, recording=recording, pcm=pcm, first_name=first_name)
        elif recording is not None:
            await recording.discard_spool()

    async def sweep(self) -> list[str]:
        """Start the job of every pending call no process is running and
        whose call is no longer live. Returns the calls it started."""
        if self.stopped:
            return []
        redis = await self._redis()
        now = time.time()
        started: list[str] = []
        for call_id in await locks.postcall_pending(redis):
            if call_id in self.jobs or await locks.is_live(redis, call_id, now):
                continue
            if not await locks.claim_postcall(redis, call_id, self.pod):
                continue
            logger.info(f"Voice call {call_id}: post-call job taken up by the sweep")
            self._start(call_id, self.run(call_id))
            started.append(call_id)
        return started

    async def sweep_forever(self) -> None:
        """Sweep now, then every *sweep_s*, until the process drains."""
        while not self.stopped:
            try:
                await self.sweep()
            except Exception:
                logger.warning("Voice post-call sweep failed", exc_info=True)
            await asyncio.sleep(self.sweep_s)

    async def run(
        self,
        call_id: str,
        *,
        recording: CallRecording | None = None,
        pcm: bytes | None = None,
        first_name: str | None = None,
    ) -> None:
        """Run the job of a call this process has claimed, then give the
        claim up.

        *pcm* is the call's recording when this process ran the call; the
        sweep's jobs read the last upload instead. A job that fails is left
        pending for another attempt, until *attempts* have failed and the
        call records the failure. The pipeline's own jobs run once the answer
        is recorded.
        """
        started = time.monotonic()
        background = BackgroundTasks()
        redis: AsyncRedisClient | None = None
        outcome = "retry"
        try:
            redis = await self._redis()
            outcome = await self._run(
                redis, call_id, recording, pcm, first_name, background
            )
        except Exception as error:
            logger.error(
                f"Voice call {call_id}: post-call job failed: {failure_of(error)}"
            )
            if redis is not None:
                outcome = await self._after_failure(redis, call_id, recording)
        finally:
            if redis is not None:
                await self._bounded(
                    "post-call release",
                    locks.release_postcall(redis, call_id, self.pod),
                )
        POSTCALLS.labels(outcome=outcome).inc()
        POSTCALL_SECONDS.observe(time.monotonic() - started)
        if background.tasks:
            try:
                await background()
            except Exception as error:
                logger.error(
                    f"Voice call {call_id}: pipeline jobs failed: {failure_of(error)}"
                )

    async def _run(
        self,
        redis: AsyncRedisClient,
        call_id: str,
        recording: CallRecording | None,
        pcm: bytes | None,
        first_name: str | None,
        background: BackgroundTasks,
    ) -> str:
        record = await locks.read_call(redis, call_id)
        if record is None:
            # Expired with the call's record: nothing is left to hand off.
            await locks.remove_postcall_pending(redis, call_id)
            if recording is not None:
                await recording.discard_spool()
            return "expired"
        if record.status in (locks.TRANSCRIBED, locks.HANDED_OFF, locks.FAILED):
            # An earlier attempt recorded its answer and stopped short of
            # clearing the call.
            await locks.remove_postcall_pending(redis, call_id)
            return record.status
        fields: dict[str, object] = {
            "status": locks.TRANSCRIBING,
            "postcall_attempts": record.postcall_attempts + 1,
        }
        if pcm is None:
            fields["recovered"] = True
        if record.status in (locks.CONNECTING, locks.LIVE):
            fields["end_reason"] = record.end_reason or CRASHED
        await locks.write_call_record(redis, call_id, fields)
        record = record.model_copy(
            update={"end_reason": fields.get("end_reason", record.end_reason)}
        )
        if record.transcript_of_record is None:
            transcript = await self._transcribe(
                redis, record, pcm=pcm, first_name=first_name
            )
            # Stored in the same write that forgets the recording, which is
            # then deleted: the recording outlives its transcript nowhere.
            await locks.forget_call_fields(
                redis, call_id, {"transcript_of_record": transcript}, "audio_path"
            )
            if recording is not None:
                await recording.discard()
            elif record.audio_path:
                await delete_quietly(self.recordings, record.audio_path, call_id)
            record = record.model_copy(update={"transcript_of_record": transcript})
        try:
            status = call_status_after(await self.hand_off(redis, record, background))
        except HTTPException as refusal:
            if refused_hand_off(refusal) is None:
                raise
            status = locks.FAILED
        await locks.remove_postcall_pending(redis, call_id)
        logger.info(f"Voice call {call_id}: {status} after ending {record.end_reason}")
        return status

    async def _transcribe(
        self,
        redis: AsyncRedisClient,
        record: CallRecord,
        *,
        pcm: bytes | None,
        first_name: str | None,
    ) -> str:
        if pcm is None:
            pcm = await self._last_upload(record)
        clips = user_turn_clips(pcm, [mark.at for mark in record.turns])
        if not any(clips):
            return ""
        if first_name is None:
            first_name = await self._first_name(record.uid)
        meter = CallMeter(
            user_id=record.uid,
            call_id=record.call_id,
            leg=record.leg,
            model=self.live_model,
            redis=redis,
            sink=self.cost_sink,
        )
        texts = await transcribe_turns(
            self.transcriber,
            clips,
            first_name=first_name,
            meter=meter,
            call_id=record.call_id,
        )
        await meter.flush()
        return transcript_of_record(texts)

    async def _last_upload(self, record: CallRecord) -> bytes:
        """The call's recording as last uploaded; empty when it has none or
        what it has cannot be read as one."""
        if not record.audio_path:
            return b""
        try:
            wav = await self.recordings.fetch(record.audio_path, record.uid)
        except Exception as error:
            if is_not_found_error(error):
                logger.warning(f"Voice call {record.call_id}: recording gone")
                return b""
            raise
        try:
            return pcm_of(wav)
        except ValueError:
            logger.warning(f"Voice call {record.call_id}: recording unreadable")
            return b""

    async def _after_failure(
        self,
        redis: AsyncRedisClient,
        call_id: str,
        recording: CallRecording | None,
    ) -> str:
        """Leave a failed job for another attempt, or record the failure on
        the call once it has had its attempts. Its recording stays for the
        retention sweep."""
        if recording is not None:
            await recording.discard_spool()
        record = await self._bounded("record read", locks.read_call(redis, call_id))
        if record is None or record.postcall_attempts < self.attempts:
            return "retry"
        code = (
            PERSONALIZATION_UNAVAILABLE
            if record.transcript_of_record is not None
            else TRANSCRIPTION_UNAVAILABLE
        )
        given_up = InterviewClosure(ending=None, hand_off=failed_hand_off(code))
        await self._bounded("failure record", record_hand_off(redis, call_id, given_up))
        await self._bounded(
            "pending removal", locks.remove_postcall_pending(redis, call_id)
        )
        return "gave_up"

    def _start(self, call_id: str, job: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(job, name=f"voice-postcall-{call_id[:12]}")
        self.jobs[call_id] = task

        def done(finished: asyncio.Task) -> None:
            if self.jobs.get(call_id) is finished:
                del self.jobs[call_id]

        task.add_done_callback(done)

    async def _bounded(self, what: str, operation: Awaitable[Any]) -> Any:
        try:
            return await asyncio.wait_for(operation, timeout=REDIS_TIMEOUT_S)
        except Exception as error:
            logger.warning(f"Voice post-call: {what} failed: {failure_of(error)}")
            return None


def failure_of(error: BaseException) -> str:
    """An error as the post-call job logs it: its type and where it was
    raised. Never its message or its frames, which can hold the person's
    words."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    frames = traceback.extract_tb(error.__traceback__)
    where = (
        f" at {Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else ""
    )
    return f"{type(error).__name__}{where}"
