"""After a call: the transcript of record, which the REST API hands off.

The live model's captions are what it heard as it went; the transcript of
record is made after the call from the person's own recording, and it is
what the onboarding pipeline reads. A call's transcript job:

1. cuts the recording into speech (``vad.segments``) and groups it into the
   person's turns by where it falls between Otto's turns, which the call
   marked on its record as they ended (:func:`user_turn_clips`);
2. transcribes each turn, four at a time, naming the speaker
   (:class:`GeminiTranscriber`). A response with no candidates or no text is
   an empty turn, never an error: a near-silent clip comes back that way;
3. joins the turns one per line without the server's tags
   (:func:`transcript_of_record`) and stores the transcript on the call's
   record, ``transcribed``, in the same write that forgets the recording,
   which is then deleted.

Then the relay is done with the call. It stays in ``voice:postcall:pending``,
and the REST API hands its transcript to the person's interview
(``onboarding_dump.voice_handoff``): the relay runs none of the onboarding
pipeline and imports none of it.

Each job runs under the call's claim (``voice:postcall:claim:<call id>``), so
it runs once across pods. The pod a call runs on claims it as the call starts
and holds it with the call's refresh; the claim outlives the call and its
job, and the job gives it up once the transcript is stored. Every pod sweeps
the pending calls when it starts and every ``POSTCALL_SWEEP_S`` and takes any
with no transcript that no process holds, as many as fit beside its own jobs:
a call whose pod died during the call or its job, or a job that failed and
waits for another attempt. A call whose pod died is transcribed from its last
upload, as ``crashed``. A job ends within ``POSTCALL_JOB_TIMEOUT_S`` and
checks it still holds its claim before each write, so it runs once in effect
even when a claim lapses under it. The pod that ran a call, holding its whole
recording, makes the transcript again over one made from a short upload,
until the call's hand-off answers.

Whatever a job settles, what is left of the call's recording goes with it;
only a job left for another attempt keeps the last upload, for that attempt
to read. A job that gives up records ``transcription_unavailable``, and a
call whose recording is lost after the relay took the person's audio records
``recording_unavailable``: no verdict on their words, which the person can
say again on another call or type.
"""

import asyncio
import bisect
import logging
import time
import traceback
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from google import genai
from google.genai import types
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from pydantic import BaseModel, ConfigDict

from backend.api.features.onboarding_dump.models import FinalizeResponse
from backend.copilot.service import strip_server_injected_tags
from backend.data.redis_client import AsyncRedisClient
from backend.util.gcs_utils import is_not_found_error
from backend.voice_relay import locks, vad
from backend.voice_relay.config import (
    CAPS,
    DEFAULT_DRAIN_S,
    IN_RATE,
    LIVE_CALL_STALE_S,
    POSTCALL_ATTEMPTS,
    POSTCALL_JOB_TIMEOUT_S,
    POSTCALL_MAX_JOBS,
    POSTCALL_SWEEP_S,
    TRANSCRIBE_ATTEMPTS,
    TRANSCRIBE_CONCURRENCY,
    TRANSCRIBE_MODEL,
    transcribe_timeout_s,
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

# What a call records when its transcript job gives up.
TRANSCRIPTION_UNAVAILABLE = "transcription_unavailable"
# What a call records when the relay took the person's audio and no recording
# of it is left: no ruling on their words.
RECORDING_UNAVAILABLE = "recording_unavailable"

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
                transcriber,
                wav_of(clip),
                first_name,
                call_id,
                timeout_s=transcribe_timeout_s(len(clip) / 2 / IN_RATE),
            )
        await meter.transcription_usage(
            tokens_in=transcription.tokens_in, tokens_out=transcription.tokens_out
        )
        return transcription.text

    async with asyncio.TaskGroup() as group:
        texts = [group.create_task(one(clip)) for clip in clips]
    return [task.result() for task in texts]


async def _transcribe(
    transcriber: Transcriber,
    wav: bytes,
    first_name: str | None,
    call_id: str,
    *,
    timeout_s: float,
) -> Transcription:
    for attempt in range(1, TRANSCRIBE_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(
                transcriber(wav, first_name=first_name), timeout=timeout_s
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

# The job's outcomes besides the statuses it can leave: left for another
# attempt, given up after its attempts, the call's record gone or naming no
# owner, the claim lost to another holder, or the call answered already.
RETRY = "retry"
GAVE_UP = "gave_up"
EXPIRED = "expired"
UNOWNED = "unowned"
LOST_CLAIM = "lost_claim"
ANSWERED = "answered"


class LostClaim(Exception):
    """This process no longer holds the job's claim: it lapsed, or another
    holder took it. The job stops without writing."""


def _failed(code: str) -> FinalizeResponse:
    """The answer a call records when no transcript of it could be made."""
    return FinalizeResponse(
        status=BrainDumpStatus.failed,
        input_mode=BrainDumpInputMode.typed,
        error_code=code,
    )


class PostCall:
    """The transcript jobs of one relay process, and its sweep for the calls
    no process is transcribing.

    *pod* names this process in the claims; a call that ran here claims its
    job with the same name (``locks.CallStore``). The sweep takes no more jobs
    than *max_jobs* beside those running, and leaves a call whose record says
    it runs to its pod for *running_window_s* from its start: the cap, a
    drain, and the time a dead pod's call takes to stop counting as live.
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
        sweep_s: float = POSTCALL_SWEEP_S,
        attempts: int = POSTCALL_ATTEMPTS,
        max_jobs: int = POSTCALL_MAX_JOBS,
        running_window_s: float = (
            CAPS.max_call_s + DEFAULT_DRAIN_S + LIVE_CALL_STALE_S
        ),
    ) -> None:
        self._redis = redis
        self.recordings = recordings
        self.transcriber = transcriber
        self._first_name = first_name
        self.live_model = live_model
        self.pod = pod
        self.cost_sink = cost_sink
        self.sweep_s = sweep_s
        self.attempts = attempts
        self.max_jobs = max_jobs
        self.running_window_s = running_window_s
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
        """Start the job of every pending call with no transcript that no
        process is running and whose call no longer runs, as many as fit
        beside this process's own jobs. Returns the calls it started. A call
        whose transcript is made is the REST API's to hand off."""
        if self.stopped:
            return []
        redis = await self._redis()
        now = time.time()
        started: list[str] = []
        for call_id in await locks.postcall_pending(redis):
            if len(self.jobs) >= self.max_jobs:
                break
            if call_id in self.jobs or await locks.is_live(redis, call_id, now):
                continue
            record = await locks.read_call(redis, call_id)
            if record is not None and (
                record.transcript_of_record is not None
                or record.hand_off is not None
                or self._may_still_run(record, now)
            ):
                continue
            if not await locks.claim_postcall(redis, call_id, self.pod):
                continue
            logger.info(f"Voice call {call_id}: transcript job taken up by the sweep")
            self._start(call_id, self.run(call_id))
            started.append(call_id)
        return started

    def _may_still_run(self, record: CallRecord, now: float) -> bool:
        """Whether *record* says its call runs and it started recently enough
        that its pod may still be running it, out of the live count only for
        a moment's Redis trouble."""
        return (
            record.status in locks.RUNNING
            and record.started is not None
            and now - record.started < self.running_window_s
        )

    async def sweep_forever(self) -> None:
        """Sweep now, then every *sweep_s*, until the process drains."""
        while not self.stopped:
            try:
                await self.sweep()
            except Exception:
                logger.warning("Voice transcript sweep failed", exc_info=True)
            await asyncio.sleep(self.sweep_s)

    async def settle(self, wait_s: float) -> None:
        """Take no more jobs, give the running ones *wait_s* to finish, and
        cancel the rest. Each gives up its claim as it stops, for another
        pod's sweep."""
        self.stopped = True
        jobs = list(self.jobs.values())
        if not jobs:
            return
        _, running = await asyncio.wait(jobs, timeout=wait_s)
        for job in running:
            job.cancel()
        await asyncio.gather(*running, return_exceptions=True)

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
        sweep's jobs read the last upload instead. The job ends within
        ``POSTCALL_JOB_TIMEOUT_S``, well inside its claim, and stops without
        writing once it no longer holds the claim. A job that fails is left
        pending for another attempt, until *attempts* have failed and the
        call records the failure. The call's spool is deleted however the job
        ends.
        """
        started = time.monotonic()
        redis: AsyncRedisClient | None = None
        outcome = RETRY
        try:
            redis = await self._redis()
            async with asyncio.timeout(POSTCALL_JOB_TIMEOUT_S):
                outcome = await self._run(redis, call_id, recording, pcm, first_name)
        except LostClaim:
            logger.warning(f"Voice call {call_id}: post-call claim lost; job stopped")
            outcome = LOST_CLAIM
        except Exception as error:
            logger.error(
                f"Voice call {call_id}: transcript job failed: {failure_of(error)}"
            )
            if redis is not None:
                outcome = await self._after_failure(redis, call_id)
        finally:
            if recording is not None:
                await recording.discard_spool()
            if redis is not None:
                await self._bounded(
                    "post-call release",
                    locks.release_postcall(redis, call_id, self.pod),
                )
        POSTCALLS.labels(outcome=outcome).inc()
        POSTCALL_SECONDS.observe(time.monotonic() - started)

    async def _run(
        self,
        redis: AsyncRedisClient,
        call_id: str,
        recording: CallRecording | None,
        pcm: bytes | None,
        first_name: str | None,
    ) -> str:
        await self._hold(redis, call_id)
        record = await locks.read_call(redis, call_id)
        if record is None:
            # Expired with the call's record: nothing is left to transcribe.
            await locks.remove_postcall_pending(redis, call_id)
            await self._discard_recording(recording, None)
            return EXPIRED
        if not record.uid:
            logger.warning(f"Voice call {call_id}: its record names no owner")
            await locks.remove_postcall_pending(redis, call_id)
            await self._discard_recording(recording, record)
            return UNOWNED
        if record.hand_off is not None:
            # Its hand-off answered: nothing of the relay's is left.
            await locks.remove_postcall_pending(redis, call_id)
            await self._discard_recording(recording, record)
            return ANSWERED
        # The pod that ran the call holds all of it, and a transcript made
        # from an upload can stop short of it: made again.
        if record.transcript_of_record is not None and not (
            pcm and record.transcribed_from_upload
        ):
            await self._discard_recording(recording, record)
            return locks.TRANSCRIBED
        fields: dict[str, object] = {
            "uid": record.uid,
            "status": locks.TRANSCRIBING,
            "postcall_attempts": record.postcall_attempts + 1,
        }
        if pcm is None:
            fields["recovered"] = True
        if record.status in locks.RUNNING:
            fields["end_reason"] = record.end_reason or CRASHED
        await locks.write_call_record(redis, call_id, fields)
        record = record.model_copy(
            update={"end_reason": fields.get("end_reason", record.end_reason)}
        )
        made = await self._make_transcript(redis, record, recording, pcm, first_name)
        if made is None:
            return RECORDING_UNAVAILABLE
        await self._discard_recording(recording, made)
        logger.info(f"Voice call {call_id}: transcribed after ending {made.end_reason}")
        return locks.TRANSCRIBED

    async def _make_transcript(
        self,
        redis: AsyncRedisClient,
        record: CallRecord,
        recording: CallRecording | None,
        pcm: bytes | None,
        first_name: str | None,
    ) -> CallRecord | None:
        """Make the call's transcript of record and store it, in the write
        that forgets the recording. Returns the record as it then stands, or
        None once the call records that its recording was lost."""
        from_upload = not pcm
        if from_upload:
            pcm = await self._last_upload(record)
        if not pcm and record.heard:
            logger.warning(f"Voice call {record.call_id}: recording lost")
            await self._hold(redis, record.call_id)
            await locks.record_answer(
                redis,
                record.call_id,
                uid=record.uid,
                status=locks.FAILED,
                answer=_failed(RECORDING_UNAVAILABLE),
            )
            await locks.remove_postcall_pending(redis, record.call_id)
            await self._discard_recording(recording, record)
            return None
        transcript = await self._transcribe(
            redis, record, pcm=pcm or b"", first_name=first_name
        )
        await self._hold(redis, record.call_id)
        stored = await locks.store_transcript(
            redis,
            record.call_id,
            transcript,
            uid=record.uid,
            from_upload=from_upload,
        )
        return record.model_copy(
            update={
                "status": locks.TRANSCRIBED,
                "transcript_of_record": stored,
                "transcribed_from_upload": (
                    from_upload
                    if stored == transcript
                    else record.transcribed_from_upload
                ),
            }
        )

    async def _transcribe(
        self,
        redis: AsyncRedisClient,
        record: CallRecord,
        *,
        pcm: bytes,
        first_name: str | None,
    ) -> str:
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

    async def _hold(self, redis: AsyncRedisClient, call_id: str) -> None:
        """Give this job's claim its full TTL again; :class:`LostClaim` once
        another holds it, or it lapsed."""
        if not await locks.hold_postcall(redis, call_id, self.pod):
            raise LostClaim

    async def _discard_recording(
        self, recording: CallRecording | None, record: CallRecord | None
    ) -> None:
        """Delete what is left of the call's recording: its transcript is
        stored, or nothing will read it again."""
        left = (
            {record.audio_path} if record is not None and record.audio_path else set()
        )
        if recording is not None:
            left -= set(recording.uploads)
            await recording.discard()
        for path in left:
            await delete_quietly(
                self.recordings, path, record.call_id if record else ""
            )

    async def _after_failure(self, redis: AsyncRedisClient, call_id: str) -> str:
        """Leave a failed job for another attempt, keeping the call's last
        upload for it, or record the failure on the call once it has had its
        attempts, and delete what is left of its recording."""
        record = await self._bounded("record read", locks.read_call(redis, call_id))
        if record is None or record.postcall_attempts < self.attempts:
            return RETRY
        if not await self._bounded(
            "post-call hold", locks.hold_postcall(redis, call_id, self.pod)
        ):
            return LOST_CLAIM
        await self._bounded(
            "failure record",
            locks.record_answer(
                redis,
                call_id,
                uid=record.uid,
                status=locks.FAILED,
                answer=_failed(TRANSCRIPTION_UNAVAILABLE),
            ),
        )
        await self._bounded(
            "pending removal", locks.remove_postcall_pending(redis, call_id)
        )
        await self._discard_recording(None, record)
        return GAVE_UP

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
