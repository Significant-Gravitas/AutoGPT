"""A voice call's hand-off, in the REST API: its transcript of record to the
person's interview, and the answer on the call's record.

The relay stops at the transcript (``voice_relay.postcall``): it leaves the
call ``transcribed``, with its transcript of record and no answer, pending in
``voice:postcall:pending``. The onboarding pipeline runs here, in the one
process that holds its keys and its imports. A call is owed a hand-off
(:func:`owed`) once its transcript is made and its arm of the ending
experiment hands off as the call closes; a confirm arm's call waits for the
person's finish.

Three things hand a call off, all under the call's one claim
(``voice:postcall:claim:<call id>``), and each re-reads the record once it
holds the claim, so a call is handed off once:

- the step's status reads (``voice.call_status``), which start the hand-off
  of an owed call in the background (:func:`settle_if_owed`), as the typed
  start hands off an interview that is over, and answer ``handing_off``
  meanwhile;
- the person's finish (``voice.finish_call``), for a confirm or a retry;
- a sweep in each replica (:func:`sweeping`), every ``HANDOFF_SWEEP_S``, for
  an owed call nobody is polling, and for a confirm arm's call left
  unconfirmed for ``UNCONFIRMED_HANDOFF_S``, handed off as if confirmed.

A hand-off that raises leaves the call pending for the next read or sweep,
until ``HANDOFF_ATTEMPTS`` have failed and the call records
``personalization_unavailable``, which the finish route hands off again. The
pipeline refusing for now (429 or 503) is recorded as it answers, for the
finish route.

The call's status follows the answer: ``handed_off`` once the pipeline took
the words or another take owns the interview, ``failed`` with the answer
while the interview stays open, or ``transcribed`` while a confirm arm waits
for the person. A call whose words the pipeline accepted takes the person's
interview when it ran to its close (:data:`RAN_TO_ITS_CLOSE`), or when the
person confirmed it. A call that was cut off is handed off all the same, so
the person has a Home if they never come back, but leaves the interview
open: a later call, or the typed interview, replaces its take as any re-take
does.

The take the person made last owns the interview. A call takes it by claiming
``taken`` on its record, set-if-absent; the person's first typed message
after the call ended claims ``superseded`` there the same way
(``interview.supersede_the_latest_call``), so of the two exactly one wins. A
hand-off whose claim lost is answered ``superseded``, as one a later call beat
is. The call that took the interview is then written on the person's
onboarding row, which outlives the record.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, HTTPException
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prometheus_client import Counter, Histogram

from backend.api.features.onboarding_dump.interview import (
    InterviewClosure,
    close_voice_interview,
    superseded_take,
    voice_hand_off_closes_interview,
)
from backend.api.features.onboarding_dump.interview_endings import hands_off_on_close
from backend.api.features.onboarding_dump.models import FinalizeResponse
from backend.copilot.onboarding_interview import InterviewEndReason
from backend.data.onboarding import record_onboarding_voice_interview_call
from backend.data.redis_client import AsyncRedisClient, get_redis_async
from backend.util.background import spawn_background_task
from backend.util.settings import Settings
from backend.voice_relay import locks
from backend.voice_relay.locks import CallRecord, CallStatus

logger = logging.getLogger(__name__)
settings = Settings()

# How a call ends when it runs to its close: Otto's goodbye, the server's caps,
# the scope rule, or the person's Hang up. Every other end cut the call off:
# the connection, silence, the vendor, a pod, or "End that call" from another
# device.
RAN_TO_ITS_CLOSE = frozenset(
    {"finished", "wall_clock", "turn_cap", "spend_cap", "scope", "user_hangup"}
)

# The pipeline's refusals to take a hand-off for now, as a call records them:
# over the person's personalization budget, or the budget cannot be read. The
# second is also what a call records once its hand-off has failed
# ``HANDOFF_ATTEMPTS`` times.
RATE_LIMITED = "rate_limited"
PERSONALIZATION_UNAVAILABLE = "personalization_unavailable"

# A hand-off holds the call's claim this long: the pipeline's admission of the
# words and its quality judge.
HANDOFF_CLAIM_TTL_S = 120
HANDOFF_ATTEMPTS = 3
# Every replica sweeps for owed calls this often.
HANDOFF_SWEEP_S = 300.0
# A confirm arm's call left unconfirmed this long after its transcript was
# made is handed off as if confirmed.
UNCONFIRMED_HANDOFF_S = 3600.0

# A hand-off's outcomes besides the call statuses it can leave: left for
# another attempt, given up after its attempts, owed nothing once read under
# the claim, or another hand-off holding the call.
RETRY = "retry"
GAVE_UP = "gave_up"
SETTLED = "settled"
HELD = "held"

HANDOFFS = Counter(
    "voice_interview_handoffs_total",
    "Voice call hand-offs, by how they ended (handed_off, failed, transcribed, "
    "retry, gave_up, settled: nothing was owed under the claim, held: another "
    "hand-off held the call) and what started them (status, sweep, "
    "unconfirmed)",
    ["outcome", "trigger"],
)
HANDOFF_SECONDS = Histogram(
    "voice_interview_handoff_seconds",
    "From a hand-off's claim to its answer on the call",
    buckets=(0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60, 120),
)

# The interview's reasons for the relay's: why the call ended, as the
# interview words it. The server ending a call it could not keep counts as
# the cap.
_INTERVIEW_REASONS: dict[str, InterviewEndReason] = {
    "finished": "confirmed",
    "wall_clock": "cap",
    "turn_cap": "cap",
    "spend_cap": "cap",
    "idle": "idle",
    "scope": "scope",
    "user_hangup": "user",
    "superseded": "user",
    "client_gone": "user",
}

# This replica's hand-offs started by status reads, by call, so a poll starts
# none beside one already running here.
_running: dict[str, asyncio.Task] = {}


def interview_reason(end_reason: str | None) -> InterviewEndReason:
    return _INTERVIEW_REASONS.get(end_reason or "", "cap")


def ran_to_its_close(end_reason: str | None) -> bool:
    """Whether a call that ended with *end_reason* ran to its close, rather
    than being cut off. An end this code does not know counts as cut off."""
    return end_reason in RAN_TO_ITS_CLOSE


def refused_hand_off(refusal: HTTPException) -> FinalizeResponse | None:
    """The pipeline's refusal to take a hand-off now, as the answer a call
    records. None for any other error."""
    code = {429: RATE_LIMITED, 503: PERSONALIZATION_UNAVAILABLE}.get(
        refusal.status_code
    )
    if code is None:
        return None
    return failed_hand_off(code)


def failed_hand_off(code: str) -> FinalizeResponse:
    return FinalizeResponse(
        status=BrainDumpStatus.failed,
        input_mode=BrainDumpInputMode.typed,
        error_code=code,
    )


def call_status_after(closure: InterviewClosure) -> CallStatus:
    """The call's status once its hand-off has answered."""
    if closure.hand_off is None:
        return locks.TRANSCRIBED
    if voice_hand_off_closes_interview(closure.hand_off):
        return locks.HANDED_OFF
    return locks.FAILED


async def record_hand_off(
    redis: AsyncRedisClient, call_id: str, closure: InterviewClosure
) -> CallStatus:
    """Record a hand-off's answer on the call. Returns the call's status. A
    call answered ``handed_off`` or ``failed`` leaves the pending set; one
    still waiting for a confirm stays in it."""
    status = call_status_after(closure)
    hand_off = closure.hand_off
    await locks.forget_call_fields(
        redis,
        call_id,
        {
            "status": status,
            "ending": closure.ending,
            "hand_off": hand_off.model_dump(mode="json") if hand_off else None,
        },
        *(() if hand_off else ("hand_off",)),
    )
    if status in (locks.HANDED_OFF, locks.FAILED):
        await locks.remove_postcall_pending(redis, call_id)
    return status


def takes_the_interview(
    closure: InterviewClosure, record: CallRecord, *, confirmed: bool
) -> bool:
    """Whether a hand-off's answer makes the call the interview's take: the
    pipeline accepted its words, and the call ran to its close or the person
    confirmed it."""
    hand_off = closure.hand_off
    accepted = hand_off is not None and hand_off.status != BrainDumpStatus.failed
    return accepted and (confirmed or ran_to_its_close(record.end_reason))


async def hand_off_call(
    redis: AsyncRedisClient,
    record: CallRecord,
    background_tasks: BackgroundTasks,
    *,
    confirmed: bool = False,
    corrections: str | None = None,
) -> InterviewClosure:
    """Hand a call's stored transcript of record to the person's interview,
    and record the answer on the call. The pipeline refusing it for now (429
    or 503) is recorded as a failed answer, then raised.

    The person's arm of the ending experiment is the one their call carries
    (``CallRecord.ending``). *confirmed* is the person's own finish of the
    call, which makes an accepted take the interview's whatever ended the
    call. An accepted take that may take the interview claims it on the
    call's record first (:func:`locks.claim_interview`): a claim that wins is
    written on the person's onboarding row, and one the person's typing beat
    answers ``superseded``.
    """
    try:
        closure = await close_voice_interview(
            record.uid,
            record.call_id,
            record.transcript_of_record or "",
            background_tasks,
            reason=interview_reason(record.end_reason),
            ending=record.ending,
            confirmed=confirmed,
            corrections=corrections,
            first_task=record.first_task,
        )
    except HTTPException as refusal:
        answer = refused_hand_off(refusal)
        if answer is not None:
            await record_hand_off(
                redis,
                record.call_id,
                InterviewClosure(ending=None, hand_off=answer),
            )
        raise
    if takes_the_interview(closure, record, confirmed=confirmed):
        standing = await locks.claim_interview(
            redis, record.call_id, locks.INTERVIEW_TAKEN
        )
        if standing == locks.INTERVIEW_TAKEN:
            await record_onboarding_voice_interview_call(record.uid, record.call_id)
        else:
            logger.info(
                f"Voice call {record.call_id}: its take did not take the "
                f"interview ({standing or 'no record'})"
            )
            closure = InterviewClosure(
                ending=closure.ending, hand_off=superseded_take()
            )
    await record_hand_off(redis, record.call_id, closure)
    return closure


def owed(record: CallRecord) -> bool:
    """Whether *record*'s call is owed a hand-off: its transcript of record is
    made, no answer is recorded, and its arm hands off as the call closes."""
    return (
        record.status == locks.TRANSCRIBED
        and record.transcript_of_record is not None
        and record.hand_off is None
        and hands_off_on_close(record.ending)
    )


def unconfirmed_too_long(record: CallRecord, now: float) -> bool:
    """Whether *record*'s call is a confirm arm's call the person left
    unconfirmed for ``UNCONFIRMED_HANDOFF_S`` since its transcript was made."""
    return (
        record.status == locks.TRANSCRIBED
        and record.transcript_of_record is not None
        and record.hand_off is None
        and not hands_off_on_close(record.ending)
        and record.transcribed_at is not None
        and now - record.transcribed_at >= UNCONFIRMED_HANDOFF_S
    )


def settle_if_owed(record: CallRecord) -> asyncio.Task | None:
    """Start the hand-off of *record*'s call in the background when it is
    owed one. Returns the hand-off running for the call in this replica, or
    None when none is owed."""
    if not owed(record):
        return None
    call_id = record.call_id
    running = _running.get(call_id)
    if running is not None and not running.done():
        return running
    task = spawn_background_task(
        hand_off_owed(call_id, trigger="status"),
        name=f"voice-handoff-{call_id[:12]}",
    )
    _running[call_id] = task

    def done(finished: asyncio.Task) -> None:
        if _running.get(call_id) is finished:
            del _running[call_id]

    task.add_done_callback(done)
    return task


async def hand_off_owed(call_id: str, *, trigger: str, confirmed: bool = False) -> str:
    """Hand off *call_id*'s call under its claim if it is still owed a
    hand-off, then run the pipeline's jobs for it. Returns how it ended.

    The record is read again once the claim is held, so two polls, two
    replicas, or a poll and the sweep run one hand-off between them.
    *confirmed* hands off a confirm arm's call left unconfirmed for
    ``UNCONFIRMED_HANDOFF_S``, as the person's finish would.
    """
    redis = await get_redis_async()
    holder = f"handoff:{uuid.uuid4().hex}"
    if not await locks.claim_postcall(
        redis, call_id, holder, ttl_s=HANDOFF_CLAIM_TTL_S
    ):
        HANDOFFS.labels(outcome=HELD, trigger=trigger).inc()
        return HELD
    started = time.monotonic()
    background = BackgroundTasks()
    try:
        outcome = await _hand_off_held(redis, call_id, background, confirmed=confirmed)
    finally:
        try:
            await locks.release_postcall(redis, call_id, holder)
        except Exception:
            logger.warning(f"Voice call {call_id}: hand-off claim not released")
    HANDOFFS.labels(outcome=outcome, trigger=trigger).inc()
    HANDOFF_SECONDS.observe(time.monotonic() - started)
    if background.tasks:
        try:
            await background()
        except Exception as error:
            logger.error(
                f"Voice call {call_id}: pipeline jobs failed: {type(error).__name__}"
            )
    return outcome


async def _hand_off_held(
    redis: AsyncRedisClient,
    call_id: str,
    background: BackgroundTasks,
    *,
    confirmed: bool,
) -> str:
    record = await locks.read_call(redis, call_id)
    if record is None or not (
        owed(record) or (confirmed and unconfirmed_too_long(record, time.time()))
    ):
        return SETTLED
    try:
        closure = await hand_off_call(redis, record, background, confirmed=confirmed)
    except HTTPException as refusal:
        if refused_hand_off(refusal) is not None:
            # Recorded as the pipeline answered, for the finish route.
            return locks.FAILED
        return await _failed_attempt(redis, record, refusal)
    except Exception as error:
        return await _failed_attempt(redis, record, error)
    logger.info(f"Voice call {call_id}: handed off after ending {record.end_reason}")
    return call_status_after(closure)


async def _failed_attempt(
    redis: AsyncRedisClient, record: CallRecord, error: Exception
) -> str:
    """Count a hand-off that raised. The call stays pending for the next
    read or sweep, until its last attempt records
    ``personalization_unavailable``."""
    logger.error(
        f"Voice call {record.call_id}: hand-off failed: {type(error).__name__}"
    )
    attempts = record.handoff_attempts + 1
    await locks.write_call_record(
        redis, record.call_id, {"uid": record.uid, "handoff_attempts": attempts}
    )
    if attempts < HANDOFF_ATTEMPTS:
        return RETRY
    await record_hand_off(
        redis,
        record.call_id,
        InterviewClosure(
            ending=None, hand_off=failed_hand_off(PERSONALIZATION_UNAVAILABLE)
        ),
    )
    return GAVE_UP


async def sweep() -> list[str]:
    """Hand off, one at a time, every pending call owed a hand-off, and every
    confirm arm's call left unconfirmed for ``UNCONFIRMED_HANDOFF_S``.
    Returns the calls this sweep handed off."""
    redis = await get_redis_async()
    now = time.time()
    handed: list[str] = []
    for call_id in await locks.postcall_pending(redis):
        record = await locks.read_call(redis, call_id)
        if record is None:
            continue
        if owed(record):
            outcome = await hand_off_owed(call_id, trigger="sweep")
        elif unconfirmed_too_long(record, now):
            outcome = await hand_off_owed(
                call_id, trigger="unconfirmed", confirmed=True
            )
        else:
            continue
        if outcome not in (HELD, SETTLED):
            handed.append(call_id)
    return handed


async def sweep_forever(sweep_s: float = HANDOFF_SWEEP_S) -> None:
    """Sweep now, then every *sweep_s*."""
    while True:
        try:
            await sweep()
        except Exception:
            logger.warning("Voice hand-off sweep failed", exc_info=True)
        await asyncio.sleep(sweep_s)


@asynccontextmanager
async def sweeping() -> AsyncIterator[None]:
    """Run the hand-off sweep while the REST API serves, where this
    deployment has a relay (``VOICE_RELAY_PUBLIC_URL``), and stop it before
    the API's services close. A hand-off it cuts short leaves its call
    pending, for the next replica's read or sweep once its claim lapses."""
    if not settings.config.voice_relay_public_url.strip():
        yield
        return
    task = asyncio.create_task(sweep_forever(), name="voice-handoff-sweep")
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
