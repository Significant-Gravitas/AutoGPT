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

A hand-off holds the claim for ``HANDOFF_CLAIM_TTL_S`` at a time, renews it
while it runs, and writes nothing once it has lost it (:class:`HandOffClaim`).
A hand-off that raises leaves the call pending, its next attempt due after a
backoff, until ``HANDOFF_ATTEMPTS`` have failed and the call records
``personalization_unavailable``, which the finish route hands off again. The
pipeline refusing for now (429 or 503) is recorded as it answers, for the
finish route. A take an earlier hand-off claimed on the dump row and never
answered is judged again, never read as a ruling. A take is words of record:
while a later call of the person's could still become one, an earlier call's
hand-off waits, writing nothing and counting no attempt, and a later call
that settled without words is passed over (``interview._call_standing``). As
a replica shuts down it starts no hand-off and gives the running ones a while
to finish, their pipeline jobs included (:func:`settle`).

The call's status follows the answer: ``handed_off`` once the pipeline took
the words or another take owns the interview, ``failed`` with the answer
while the interview stays open, or ``transcribed`` while a confirm arm waits
for the person. A call whose words the pipeline accepted takes the person's
interview when it ran to its close (:data:`RAN_TO_ITS_CLOSE`), or when the
person confirmed it. A call that was cut off, or that the person hung up, is
handed off all the same, so the person has a Home if they never come back,
but leaves the interview open: the person's finish of it takes the
interview, and a later call, or the typed interview, replaces its take as any
re-take does.

The take the person made last owns the interview. A call takes it by claiming
``taken`` on its record, set-if-absent; the person's first typed message, or
typed finish, after the call ended claims ``superseded`` there the same way
(``interview.supersede_the_latest_call``), so of the two exactly one wins. A
hand-off whose claim lost is answered ``superseded``, as one a later call beat
is, and gives up the dump row its words took, with the pipeline jobs it
queued, so the typed take can claim the row. The call that took the
interview is then written on the person's onboarding row, which outlives the
record.
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

from backend.api.features.onboarding_dump import service
from backend.api.features.onboarding_dump.interview import (
    SUPERSEDED,
    InterviewClosure,
    LaterCallUnsettled,
    call_words,
    close_voice_interview,
    hand_off_owed_typed_interviews,
    interview_recording_id,
    start_owed_typed_hand_off,
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
# or the scope rule. Every other end leaves the interview open: the person's
# Hang up, and every end that cut the call off: the connection, silence, the
# vendor, a pod, or "End that call" from another device.
RAN_TO_ITS_CLOSE = frozenset(
    {"finished", "wall_clock", "turn_cap", "spend_cap", "scope"}
)

# The pipeline's refusals to take a hand-off for now, as a call records them:
# over the person's personalization budget, or the budget cannot be read. The
# second is also what a call records once its hand-off has failed
# ``HANDOFF_ATTEMPTS`` times.
RATE_LIMITED = "rate_limited"
PERSONALIZATION_UNAVAILABLE = "personalization_unavailable"

# A hand-off holds the call's claim this long at a time, and renews it this
# often while it runs, so a claim outlives only a hand-off that stopped.
HANDOFF_CLAIM_TTL_S = 120
HANDOFF_RENEW_S = 30.0
HANDOFF_ATTEMPTS = 3
# A failed attempt waits this long for the next, doubling with each.
HANDOFF_RETRY_S = 30.0
# Every replica sweeps for owed calls this often.
HANDOFF_SWEEP_S = 300.0
# At shutdown, how long the hand-offs still running get to finish, their
# pipeline jobs included, before they are cancelled.
HANDOFF_SHUTDOWN_S = 15.0
# A confirm arm's call left unconfirmed this long after its transcript was
# made is handed off as if confirmed.
UNCONFIRMED_HANDOFF_S = 3600.0

# A hand-off's outcomes besides the call statuses it can leave: left for
# another attempt, given up after its attempts, owed nothing once read under
# the claim, another hand-off holding the call, the claim lost mid-way, or a
# later call of the person's still to settle.
RETRY = "retry"
GAVE_UP = "gave_up"
SETTLED = "settled"
HELD = "held"
LOST_CLAIM = "lost_claim"
WAITING = "waiting"

HANDOFFS = Counter(
    "voice_interview_handoffs_total",
    "Voice call hand-offs, by how they ended (handed_off, failed, transcribed, "
    "retry, gave_up, settled: nothing was owed under the claim, held: another "
    "hand-off held the call, lost_claim: the claim lapsed mid-way, waiting: a "
    "later call was still to settle) and what started them (status, sweep, "
    "unconfirmed, settle: the later call settled without words)",
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
# none beside one already running here, and shutdown waits for them.
_running: dict[str, asyncio.Task] = {}
# Set while the replica shuts down: no hand-off starts.
_stopping = False


def interview_reason(end_reason: str | None) -> InterviewEndReason:
    return _INTERVIEW_REASONS.get(end_reason or "", "cap")


def ran_to_its_close(end_reason: str | None) -> bool:
    """Whether a call that ended with *end_reason* ran to its close, rather
    than being hung up or cut off. An end this code does not know counts as
    cut off."""
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
    """A call's hand-off that failed with *code*: an answer about spoken
    words."""
    return FinalizeResponse(
        status=BrainDumpStatus.failed,
        input_mode=BrainDumpInputMode.voice,
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
    redis: AsyncRedisClient, call_id: str, closure: InterviewClosure, *, uid: str
) -> CallStatus:
    """Record a hand-off's answer on the call of *uid*'s, naming them. Returns
    the call's status. A call answered ``handed_off`` or ``failed`` leaves the
    pending set; one still waiting for a confirm stays in it. A call answered
    without words of record is no take, so the take before it, waiting on
    it, is handed off now (:func:`settle_the_take_before`)."""
    status = call_status_after(closure)
    hand_off = closure.hand_off
    await locks.forget_call_fields(
        redis,
        call_id,
        {
            "uid": uid,
            "status": status,
            "ending": closure.ending,
            "hand_off": hand_off.model_dump(mode="json") if hand_off else None,
        },
        *(() if hand_off else ("hand_off",)),
    )
    if status in (locks.HANDED_OFF, locks.FAILED):
        await locks.remove_postcall_pending(redis, call_id)
    if hand_off is not None:
        record = await _read_quietly(redis, call_id)
        if record is not None and call_words(record) == "no_words":
            await settle_the_take_before(redis, record)
    return status


async def settle_the_take_before(
    redis: AsyncRedisClient, record: CallRecord
) -> asyncio.Task | None:
    """Start the hand-off of the person's take before *record*'s call, which
    settled without words: the first call back along each record's ``prev``
    past every other that settled without words, when it is owed one
    (:func:`settle_if_owed`). It waited on the calls after it. Best effort:
    a failure is logged, and the next status read or sweep asks again.
    Returns the hand-off started, or None."""
    try:
        seen = {record.call_id}
        current = record.prev
        while current is not None and current not in seen:
            seen.add(current)
            earlier = await locks.read_call(redis, current)
            if earlier is None or earlier.uid != record.uid:
                return None
            if call_words(earlier) != "no_words":
                return settle_if_owed(earlier, trigger="settle")
            current = earlier.prev
    except Exception as error:
        logger.warning(
            f"Voice call {record.call_id}: the take before it not settled: "
            f"{type(error).__name__}"
        )
    return None


async def _read_quietly(redis: AsyncRedisClient, call_id: str) -> CallRecord | None:
    try:
        return await locks.read_call(redis, call_id)
    except Exception as error:
        logger.warning(f"Voice call {call_id}: record unread: {type(error).__name__}")
        return None


def takes_the_interview(
    closure: InterviewClosure, record: CallRecord, *, by_person: bool
) -> bool:
    """Whether a hand-off's answer makes the call the interview's take: the
    pipeline accepted its words, and the call ran to its close or the person
    finished it themselves (*by_person*)."""
    return _accepted(closure) and (by_person or ran_to_its_close(record.end_reason))


class LostClaim(Exception):
    """The hand-off no longer holds its call's claim: it lapsed, or another
    hand-off took it. The hand-off stops without writing."""


class HandOffClaim:
    """A hand-off's hold on its call (``voice:postcall:claim:<call id>``):
    renewed every ``HANDOFF_RENEW_S`` while the hand-off runs
    (:meth:`renewed`), and checked before each write the hand-off makes
    (:meth:`check`), so a hand-off whose claim was lost writes nothing."""

    def __init__(self, redis: AsyncRedisClient, call_id: str, holder: str) -> None:
        self.redis = redis
        self.call_id = call_id
        self.holder = holder

    async def check(self) -> None:
        """Renew the claim; :class:`LostClaim` once it is not this one's."""
        if not await locks.hold_postcall(
            self.redis, self.call_id, self.holder, ttl_s=HANDOFF_CLAIM_TTL_S
        ):
            raise LostClaim

    @asynccontextmanager
    async def renewed(self) -> AsyncIterator[None]:
        renewal = asyncio.create_task(
            self._renew(), name=f"voice-handoff-claim-{self.call_id[:12]}"
        )
        try:
            yield
        finally:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)

    async def _renew(self) -> None:
        while True:
            await asyncio.sleep(HANDOFF_RENEW_S)
            try:
                await self.check()
            except LostClaim:
                logger.warning(f"Voice call {self.call_id}: hand-off claim lost")
                return
            except Exception as error:
                logger.warning(
                    f"Voice call {self.call_id}: hand-off claim not renewed: "
                    f"{type(error).__name__}"
                )


async def hand_off_call(
    redis: AsyncRedisClient,
    record: CallRecord,
    background_tasks: BackgroundTasks,
    *,
    claim: HandOffClaim,
    confirmed: bool = False,
    by_person: bool = False,
    corrections: str | None = None,
) -> InterviewClosure:
    """Hand a call's stored transcript of record to the person's interview,
    and record the answer on the call. The pipeline refusing it for now (429
    or 503) is recorded as a failed answer, then raised.

    The person's arm of the ending experiment is the one their call carries
    (``CallRecord.ending``). *confirmed* hands the words off whatever the arm,
    as a confirm does; *by_person* is the person's own finish of the call,
    which makes an accepted take the interview's whatever ended the call.
    An accepted take that may take the interview claims it on the call's
    record first (:func:`locks.claim_interview`): a claim that wins is written
    on the person's onboarding row, and one the person's typing beat answers
    ``superseded`` and gives up the row its words took
    (:func:`_give_the_row_up`). The pipeline's jobs for the take reach
    *background_tasks* only when the take stands. Every write is made under
    *claim*, and none once it is lost (:class:`LostClaim`).

    A call another take outranks (``interview._call_standing``) claims
    ``superseded`` on its record; a ``superseded`` the dump row gave is the
    call's answer and claims nothing. A call's take dropped for a newer take
    gives up the dump row while no pipeline has run it
    (:func:`_give_the_row_up`), and once that frees the row the typed
    interview's owed hand-off starts
    (``interview.hand_off_owed_typed_interview``).
    """
    own = BackgroundTasks()
    try:
        closure = await close_voice_interview(
            record.uid,
            record.call_id,
            record.transcript_of_record or "",
            own,
            reason=interview_reason(record.end_reason),
            ending=record.ending,
            confirmed=confirmed,
            corrections=corrections,
            first_task=record.first_task,
            # A failure the finish hands off again accepted nothing, so a take
            # the row holds unjudged is judged again.
            unanswered=record.hand_off is None
            or record.hand_off.status == BrainDumpStatus.failed,
        )
    except HTTPException as refusal:
        answer = refused_hand_off(refusal)
        if answer is not None:
            await claim.check()
            await record_hand_off(
                redis,
                record.call_id,
                InterviewClosure(ending=None, hand_off=answer),
                uid=record.uid,
            )
        raise
    await claim.check()
    freed = False
    if _accepted(closure):
        if takes_the_interview(closure, record, by_person=by_person):
            standing = await locks.claim_interview(
                redis, record.call_id, locks.INTERVIEW_TAKEN
            )
        else:
            standing = await locks.interview_of(redis, record.call_id)
        if standing == locks.INTERVIEW_TAKEN:
            await record_onboarding_voice_interview_call(record.uid, record.call_id)
        elif standing == locks.INTERVIEW_SUPERSEDED:
            logger.info(
                f"Voice call {record.call_id}: the person's typing since "
                f"owns the interview"
            )
            freed = await _give_the_row_up(record)
            closure = InterviewClosure(
                ending=closure.ending,
                hand_off=superseded_take(BrainDumpInputMode.voice),
            )
    elif closure.outranked:
        # Another take owns the interview: the record says so too, for the
        # step that reads it first. A take an earlier hand-off of this call
        # left on the row gives it up. A ``superseded`` the row gave claims
        # nothing: it is recorded as the call's answer.
        await locks.claim_interview(redis, record.call_id, locks.INTERVIEW_SUPERSEDED)
        freed = await _give_the_row_up(record)
    await claim.check()
    if not _superseded(closure):
        background_tasks.tasks.extend(own.tasks)
    await record_hand_off(redis, record.call_id, closure, uid=record.uid)
    if freed:
        start_owed_typed_hand_off(record.uid)
    return closure


def _accepted(closure: InterviewClosure) -> bool:
    hand_off = closure.hand_off
    return hand_off is not None and hand_off.status != BrainDumpStatus.failed


def _superseded(closure: InterviewClosure) -> bool:
    hand_off = closure.hand_off
    return hand_off is not None and hand_off.error_code == SUPERSEDED


async def _give_the_row_up(record: CallRecord) -> bool:
    """Give up the dump row the call's take holds, for a newer take: the
    take is marked failed ``superseded`` while no pipeline has run it
    (:func:`service.give_up_unrun_take`), so the newer take can claim the
    row; its pipeline jobs never run. True when that freed the row."""
    return await service.give_up_unrun_take(
        record.uid, interview_recording_id(record.call_id), SUPERSEDED
    )


def owed(record: CallRecord) -> bool:
    """Whether *record*'s call is owed a hand-off: it names its owner, its
    transcript of record is made, no answer is recorded, and its arm hands
    off as the call closes."""
    return (
        bool(record.uid)
        and record.status == locks.TRANSCRIBED
        and record.transcript_of_record is not None
        and record.hand_off is None
        and hands_off_on_close(record.ending)
    )


def unconfirmed_too_long(record: CallRecord, now: float) -> bool:
    """Whether *record*'s call, naming its owner, is a confirm arm's call the
    person left unconfirmed for ``UNCONFIRMED_HANDOFF_S`` since its
    transcript was made."""
    return (
        bool(record.uid)
        and record.status == locks.TRANSCRIBED
        and record.transcript_of_record is not None
        and record.hand_off is None
        and not hands_off_on_close(record.ending)
        and record.transcribed_at is not None
        and now - record.transcribed_at >= UNCONFIRMED_HANDOFF_S
    )


def retry_due(record: CallRecord, now: float) -> bool:
    """Whether a hand-off of *record*'s call may start now: no earlier
    attempt failed, or its backoff has passed."""
    return record.handoff_retry_at is None or now >= record.handoff_retry_at


def settle_if_owed(
    record: CallRecord, *, trigger: str = "status"
) -> asyncio.Task | None:
    """Start the hand-off of *record*'s call in the background when it is
    owed one and may start now. Returns the hand-off running for the call in
    this replica, or None when none runs."""
    call_id = record.call_id
    running = _running.get(call_id)
    if running is not None and not running.done():
        return running
    if _stopping or not owed(record) or not retry_due(record, time.time()):
        return None
    task = spawn_background_task(
        hand_off_owed(call_id, trigger=trigger),
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
    hand-off and its backoff has passed, then run the pipeline's jobs for
    it. Returns how it ended.

    The record is read again once the claim is held, so two polls, two
    replicas, or a poll and the sweep run one hand-off between them.
    *confirmed* hands off a confirm arm's call left unconfirmed for
    ``UNCONFIRMED_HANDOFF_S``, as a confirm would; it is not the person's
    own finish, so it takes the interview only for a call that ran to its
    close.
    """
    redis = await get_redis_async()
    holder = f"handoff:{uuid.uuid4().hex}"
    try:
        claimed = await locks.claim_postcall(
            redis, call_id, holder, ttl_s=HANDOFF_CLAIM_TTL_S
        )
    except Exception:
        # The claim may have been written before its reply was lost.
        await release_quietly(redis, call_id, holder)
        raise
    if not claimed:
        HANDOFFS.labels(outcome=HELD, trigger=trigger).inc()
        return HELD
    started = time.monotonic()
    background = BackgroundTasks()
    claim = HandOffClaim(redis, call_id, holder)
    try:
        async with claim.renewed():
            outcome = await _hand_off_held(
                redis, call_id, background, claim=claim, confirmed=confirmed
            )
    finally:
        await release_quietly(redis, call_id, holder)
    HANDOFFS.labels(outcome=outcome, trigger=trigger).inc()
    HANDOFF_SECONDS.observe(time.monotonic() - started)
    # A hand-off that lost its claim leaves the call to the one holding it.
    if background.tasks and outcome != LOST_CLAIM:
        try:
            await background()
        except Exception as error:
            logger.error(
                f"Voice call {call_id}: pipeline jobs failed: {type(error).__name__}"
            )
    return outcome


async def release_quietly(redis: AsyncRedisClient, call_id: str, holder: str) -> None:
    """Give up *holder*'s claim on the call, if it still holds it; a failure
    is logged, and the claim lapses on its own."""
    try:
        await asyncio.wait_for(
            locks.release_postcall(redis, call_id, holder),
            timeout=locks.REDIS_TIMEOUT_S,
        )
    except Exception:
        logger.warning(f"Voice call {call_id}: hand-off claim not released")


async def _hand_off_held(
    redis: AsyncRedisClient,
    call_id: str,
    background: BackgroundTasks,
    *,
    claim: HandOffClaim,
    confirmed: bool,
) -> str:
    record = await locks.read_call(redis, call_id)
    now = time.time()
    if (
        record is None
        or not (owed(record) or (confirmed and unconfirmed_too_long(record, now)))
        or not retry_due(record, now)
    ):
        return SETTLED
    try:
        closure = await hand_off_call(
            redis, record, background, claim=claim, confirmed=confirmed
        )
    except LostClaim:
        logger.warning(f"Voice call {call_id}: hand-off claim lost; nothing written")
        return LOST_CLAIM
    except LaterCallUnsettled:
        # Asked again on the next read or sweep, with no attempt counted.
        return WAITING
    except HTTPException as refusal:
        if refused_hand_off(refusal) is not None:
            # Recorded as the pipeline answered, for the finish route.
            return locks.FAILED
        return await _failed_attempt(redis, record, refusal, claim=claim)
    except Exception as error:
        return await _failed_attempt(redis, record, error, claim=claim)
    logger.info(f"Voice call {call_id}: handed off after ending {record.end_reason}")
    return call_status_after(closure)


async def _failed_attempt(
    redis: AsyncRedisClient,
    record: CallRecord,
    error: Exception,
    *,
    claim: HandOffClaim,
) -> str:
    """Count a hand-off that raised. The call stays pending, its next attempt
    due after a backoff, until its last attempt records
    ``personalization_unavailable``."""
    logger.error(
        f"Voice call {record.call_id}: hand-off failed: {type(error).__name__}"
    )
    try:
        await claim.check()
    except LostClaim:
        return LOST_CLAIM
    attempts = record.handoff_attempts + 1
    await locks.write_call_record(
        redis,
        record.call_id,
        {
            "uid": record.uid,
            "handoff_attempts": attempts,
            "handoff_retry_at": round(
                time.time() + HANDOFF_RETRY_S * 2 ** (attempts - 1), 3
            ),
        },
    )
    if attempts < HANDOFF_ATTEMPTS:
        return RETRY
    await record_hand_off(
        redis,
        record.call_id,
        InterviewClosure(
            ending=None, hand_off=failed_hand_off(PERSONALIZATION_UNAVAILABLE)
        ),
        uid=record.uid,
    )
    return GAVE_UP


async def sweep(stop: asyncio.Event | None = None) -> list[str]:
    """Hand off, one at a time, every pending call owed a hand-off whose
    backoff has passed, and every confirm arm's call left unconfirmed for
    ``UNCONFIRMED_HANDOFF_S``, then every typed interview whose hand-off is
    owed (``interview.hand_off_owed_typed_interviews``); *stop* ends it
    between them. Returns the calls this sweep handed off."""
    redis = await get_redis_async()
    now = time.time()
    handed: list[str] = []
    for call_id in await locks.postcall_pending(redis):
        if stop is not None and stop.is_set():
            break
        record = await locks.read_call(redis, call_id)
        if record is None or not retry_due(record, now):
            continue
        if owed(record):
            outcome = await hand_off_owed(call_id, trigger="sweep")
        elif unconfirmed_too_long(record, now):
            outcome = await hand_off_owed(
                call_id, trigger="unconfirmed", confirmed=True
            )
        else:
            continue
        if outcome not in (HELD, SETTLED, WAITING):
            handed.append(call_id)
    if stop is None or not stop.is_set():
        try:
            await hand_off_owed_typed_interviews(stop)
        except Exception:
            logger.warning("Owed typed hand-offs not swept", exc_info=True)
    return handed


async def sweep_forever(stop: asyncio.Event, sweep_s: float = HANDOFF_SWEEP_S) -> None:
    """Sweep now, then every *sweep_s*, until *stop* is set."""
    while not stop.is_set():
        try:
            await sweep(stop)
        except Exception:
            logger.warning("Voice hand-off sweep failed", exc_info=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=sweep_s)
        except TimeoutError:
            pass


@asynccontextmanager
async def sweeping() -> AsyncIterator[None]:
    """Run the hand-off sweep while the REST API serves, and settle this
    replica's hand-offs before the API's services close (:func:`settle`).

    The sweep runs wherever ``VOICE_RELAY_PUBLIC_URL`` is not blank. Its
    default names a local relay, so only a deployment that sets it blank goes
    without, and then sweeps neither its calls nor its owed typed interviews
    (:func:`sweep`)."""
    global _stopping
    _stopping = False
    stop = asyncio.Event()
    sweeper = (
        asyncio.create_task(sweep_forever(stop), name="voice-handoff-sweep")
        if settings.config.voice_relay_public_url.strip()
        else None
    )
    try:
        yield
    finally:
        _stopping = True
        stop.set()
        await settle(sweeper, HANDOFF_SHUTDOWN_S)


async def settle(sweeper: asyncio.Task | None, wait_s: float) -> None:
    """Start no hand-off, give the sweep's current one and every hand-off a
    status read started *wait_s* to finish, their pipeline jobs included, and
    cancel the rest. A hand-off cancelled gives up its claim, and its call
    stays pending for the next replica's read or sweep."""
    running = [
        task
        for task in (sweeper, *_running.values())
        if task is not None and not task.done()
    ]
    if not running:
        return
    _, unfinished = await asyncio.wait(running, timeout=wait_s)
    for task in unfinished:
        task.cancel()
    await asyncio.gather(*unfinished, return_exceptions=True)
