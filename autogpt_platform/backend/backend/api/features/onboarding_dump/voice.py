"""Starting a voice call for the onboarding interview, and the call after.

The REST API decides who may call; the relay (``backend.voice_relay``) runs
the call. A start answers with the relay's socket, a signed, single-use start
token naming the person and the call, which the relay trusts without a
platform session, and the call's LiveKit room, which carries its audio, with
the browser's token for it. In order, it checks:

- the voice flag: 404 when it is off for the person or cannot be read;
- that this deployment can carry a call: a relay to call, the key its start
  tokens are signed with, and a LiveKit project; 503 when it cannot;
- that the person's interview is not over: 409. A person has one interview,
  typed or spoken, found on their onboarding row: it is over once the typed
  interview is, or once a call's words became its take (naming that call);
- that the person has no call live already: 409 naming the live call. A
  call whose pod died, or that ended without freeing the person's lock,
  holds nobody back once it no longer runs (``locks.call_is_live``);
- that a relay sweeps the call rooms lately (``locks.rooms_swept``), so the
  room this start opens is closed in the end, even if its call never starts:
  503 with ``Retry-After``;
- the platform's daily voice budget and its live-call ceiling: 503 with
  ``Retry-After``;
- the person's own call allowance, on every tier: 429 with ``Retry-After``.
  Read, not written: the relay counts a call once its model session opens,
  so a start that never got that far costs the person nothing.

None of them costs the person anything. Then it registers the call's room
(``locks.register_room``), a 503 when it cannot, and mints the browser's
token for it (``rooms.person_token``); the relay closes the room
(``room_reaper``). It reads the person's arm of the voice ending experiment,
which stores it on first read (``experiments.assign``), and the start token
carries the arm to the relay, which reads no flags. The relay records the
call on the person's onboarding row once the call starts, and its record in
Redis for a day, with the arm on it for the hand-off, and makes its
transcript of record once it ends. Then the person's own routes read the
call's state (:func:`call_status`, :func:`latest_call`, which name where a
live call is rejoined), which also starts the call's hand-off here once it is
owed one (``voice_handoff``), confirm its hand-off or hand it off again
(:func:`finish_call`), and end a call of theirs live elsewhere, or one whose
start they gave up on (:func:`end_call`). Another person's call is 404 on
every one of them.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import TypeVar

from fastapi import BackgroundTasks, HTTPException
from prisma.enums import BrainDumpInputMode
from prometheus_client import Counter

from backend.api.features.onboarding_dump import quality
from backend.api.features.onboarding_dump.interview import (
    LaterCallUnsettled,
    call_that_took_the_interview,
    call_words,
    superseded_take,
)
from backend.api.features.onboarding_dump.interview_endings import (
    MODE_ENDINGS,
    assigned_ending,
)
from backend.api.features.onboarding_dump.models import (
    FinalizeResponse,
    VoiceCallCaps,
    VoiceCallRoom,
    VoiceCallStartResponse,
    VoiceCallStatusResponse,
)
from backend.api.features.onboarding_dump.rate_limit import enforce_voice_call_budget
from backend.api.features.onboarding_dump.voice_handoff import (
    HANDOFF_CLAIM_TTL_S,
    HandOffClaim,
    LostClaim,
    hand_off_call,
    owed,
    settle_if_owed,
    settle_the_take_before,
)
from backend.copilot.onboarding_interview import (
    INTERVIEW_CLOSED,
    find_onboarding_interview_session,
    interview_is_over,
)
from backend.data.onboarding import get_onboarding_voice_call_id
from backend.data.redis_client import AsyncRedisClient, get_redis_async
from backend.util.background import spawn_background_task
from backend.util.feature_flag import Flag, evaluate_feature_flag
from backend.util.settings import AppEnvironment, Settings
from backend.voice_relay import locks, rooms, tokens
from backend.voice_relay.config import CAPS, TOKEN_TTL_S
from backend.voice_relay.locks import CallRecord

logger = logging.getLogger(__name__)
settings = Settings()

T = TypeVar("T")

REDIS_TIMEOUT_S = 0.25
# How long to wait before trying again: while the platform is at its live-call
# ceiling or no relay sweeps the rooms, and while voice cannot be served at
# all.
BUSY_RETRY_AFTER_S = 30
UNAVAILABLE_RETRY_AFTER_S = 300

CALL_IN_PROGRESS = locks.CALL_IN_PROGRESS
# Why a finish is refused: the call's words are not ready for a hand-off (the
# call is live, or its transcript is still being made), or another finish of
# the same call is handing them off.
CALL_NOT_READY = "call_not_ready"
NO_SUCH_CALL = "No such voice call"
# Answers on the words, which another hand-off of the same words would only
# repeat: the person calls again or types instead.
_VERDICTS = frozenset({quality.NO_USABLE_SPEECH, quality.INSUFFICIENT_CONTENT})

ABANDONED_STARTS = Counter(
    "voice_interview_abandoned_starts_total",
    "Voice call starts ended by the browser before the call started: its "
    "join of the call's room failed or took too long, or the person chose "
    "to type instead while it joined",
)


async def start_voice_call(user_id: str) -> VoiceCallStartResponse:
    """Start a voice call for *user_id*'s onboarding interview.

    Raises:
        HTTPException: 404, 409, 429 or 503, as the module describes.
    """
    if not await _voice_enabled(user_id):
        raise HTTPException(status_code=404, detail="Feature not available")
    relay_url = relay_socket_url()
    project = rooms.livekit_project(settings)
    if relay_url is None or project is None or not tokens.is_configured():
        raise _unavailable(
            "Voice calls are not available right now.", UNAVAILABLE_RETRY_AFTER_S
        )
    await _refuse_if_interview_over(user_id)
    redis = await _bounded(get_redis_async())
    live_call = await _bounded(locks.live_call_holding(redis, user_id, time.time()))
    if live_call:
        raise HTTPException(
            status_code=409, detail={"code": CALL_IN_PROGRESS, "call_id": live_call}
        )
    if not await _bounded(locks.rooms_swept(redis)):
        logger.warning("No relay sweeps the voice call rooms, so no call starts")
        raise _unavailable(
            "Voice calls are temporarily unavailable. Please retry shortly.",
            BUSY_RETRY_AFTER_S,
        )
    await _refuse_past_platform_limits(redis)
    await enforce_voice_call_budget(user_id)
    call_id = tokens.new_call_id()
    ending = _voice_arm(await assigned_ending(user_id, "voice"))
    # Registered before its token is minted, so every room a token can open
    # is closed in the end.
    await _bounded(locks.register_room(redis, call_id, user_id, time.time()))
    return VoiceCallStartResponse(
        call_id=call_id,
        relay_url=relay_url,
        token=tokens.mint_start(user_id, call_id, ending=ending),
        expires_in=TOKEN_TTL_S,
        room=VoiceCallRoom(
            url=project.url,
            token=rooms.person_token(user_id=user_id, call_id=call_id, project=project),
        ),
        caps=VoiceCallCaps.model_validate(CAPS.as_message()),
    )


def voice_configured() -> bool:
    """Whether this deployment can carry a voice call: a relay to call, the
    key its start tokens are signed with, and a LiveKit project."""
    return (
        relay_socket_url() is not None
        and tokens.is_configured()
        and rooms.livekit_project(settings) is not None
    )


async def voice_available(user_id: str) -> bool:
    """Whether the step may offer *user_id* a call: this deployment can carry
    one, with a Gemini key for its relay, and voice is on for them."""
    if not voice_configured() or not settings.secrets.gemini_api_key:
        return False
    return await _voice_enabled(user_id)


def _voice_arm(ending: str | None) -> str | None:
    """*ending*, when it is an arm the voice mode offers; None otherwise, so
    the call ends as an unenrolled one does."""
    if ending in {arm.value for arm in MODE_ENDINGS["voice"]}:
        return ending
    if ending is not None:
        logger.warning(f"Voice call: ending {ending!r} is no voice arm")
    return None


def relay_socket_url() -> str | None:
    """The relay's call socket as the browser reaches it, or None when this
    deployment has none. Plain ``ws://`` only for a local install."""
    base = settings.config.voice_relay_public_url.strip().rstrip("/")
    if base.startswith("wss://"):
        return f"{base}/call"
    if base.startswith("ws://") and settings.config.app_env == AppEnvironment.LOCAL:
        return f"{base}/call"
    return None


async def _voice_enabled(user_id: str) -> bool:
    try:
        enabled, _ = await evaluate_feature_flag(
            Flag.ONBOARDING_INTERVIEW_VOICE, user_id, default=False
        )
    except Exception:
        logger.warning("Voice interview flag could not be read", exc_info=True)
        return False
    return enabled


async def _refuse_if_interview_over(user_id: str) -> None:
    try:
        interview = await find_onboarding_interview_session(user_id)
        typed_over = interview is not None and interview_is_over(interview)
        taken_by = None if typed_over else await call_that_took_the_interview(user_id)
    except Exception as exc:
        logger.warning(
            f"Voice call: the interview could not be read: {type(exc).__name__}"
        )
        raise _unavailable(
            "Voice calls are temporarily unavailable. Please retry shortly.",
            BUSY_RETRY_AFTER_S,
        ) from exc
    if typed_over:
        raise HTTPException(status_code=409, detail={"code": INTERVIEW_CLOSED})
    if taken_by:
        raise HTTPException(
            status_code=409, detail={"code": INTERVIEW_CLOSED, "call_id": taken_by}
        )


async def _refuse_past_platform_limits(redis: AsyncRedisClient) -> None:
    now = datetime.now(UTC)
    spent = await _bounded(locks.daily_spend_usd(redis, now))
    if spent >= settings.config.voice_interview_daily_budget_usd:
        raise _unavailable(
            "Voice calls are paused until tomorrow. You can type instead.",
            locks.seconds_until_tomorrow(now),
        )
    live = await _bounded(locks.live_call_count(redis, now.timestamp()))
    if live >= settings.config.voice_interview_max_concurrent:
        raise _unavailable(
            "All voice lines are busy. Please try again shortly.", BUSY_RETRY_AFTER_S
        )


async def _bounded(operation: Awaitable[T]) -> T:
    """*operation*, or a 503 when Redis cannot answer in time: an allowance or
    a limit nobody could read never admits a call."""
    try:
        return await asyncio.wait_for(operation, timeout=REDIS_TIMEOUT_S)
    except Exception as exc:
        logger.warning(f"Voice call state unavailable: {type(exc).__name__}")
        raise _unavailable(
            "Voice calls are temporarily unavailable. Please retry shortly.",
            BUSY_RETRY_AFTER_S,
        ) from exc


def _unavailable(message: str, retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=503, detail=message, headers={"Retry-After": str(retry_after)}
    )


# --- the call after it starts ----------------------------------------------


async def call_status(user_id: str, call_id: str) -> VoiceCallStatusResponse:
    """*call_id* as it stands, when it is *user_id*'s. A call whose
    transcript is made and whose hand-off is owed has the hand-off started
    in the background, and reads ``handing_off`` until it answers. A call
    that settled without words, the relay's answers included, starts the
    hand-off of the take before it the same way
    (:func:`settle_the_take_before`).

    Raises:
        HTTPException: 404 for a call that is not theirs or has expired, 503
            while its record cannot be read.
    """
    redis = await _bounded(get_redis_async())
    record = await _their_call(redis, user_id, call_id)
    settle_if_owed(record)
    if call_words(record) == "no_words":
        spawn_background_task(
            settle_the_take_before(redis, record),
            name=f"voice-settle-before-{call_id[:12]}",
        )
    return _status_of(record)


async def latest_call(user_id: str) -> VoiceCallStatusResponse:
    """*user_id*'s latest call as it stands, for a step that reloads.

    A call that settled without words of record is no take, so the read
    looks back past it along ``prev`` to the latest call that has words or
    is still to settle, and answers that one; with nothing behind it, the
    no-words call itself. A call that never went live (its room or its model
    session did not open) was never the person's, so the read passes over it
    and never answers it.

    Raises:
        HTTPException: 404 when they have none, or only calls that never went
            live, or its record has expired; 503 while it cannot be read.
    """
    try:
        call_id = await get_onboarding_voice_call_id(user_id)
    except Exception as exc:
        logger.warning(f"Voice call: the latest call unreadable: {type(exc).__name__}")
        raise _unavailable(
            "Voice calls are temporarily unavailable. Please retry shortly.",
            BUSY_RETRY_AFTER_S,
        ) from exc
    if call_id is None:
        raise HTTPException(status_code=404, detail=NO_SUCH_CALL)
    redis = await _bounded(get_redis_async())
    record = await _their_call(redis, user_id, call_id)
    answered = await _call_to_answer(redis, user_id, record)
    if answered is None:
        raise HTTPException(status_code=404, detail=NO_SUCH_CALL)
    settle_if_owed(answered)
    return _status_of(answered)


async def _call_to_answer(
    redis: AsyncRedisClient, user_id: str, record: CallRecord
) -> CallRecord | None:
    """The call a reload answers, read from *record* back along ``prev``:
    the latest call that went live and has words or is still to settle, else
    the last no-words call the read reached, else None."""
    seen = {record.call_id}
    reached = None if record.never_went_live else record
    while (
        record.never_went_live or call_words(record) == "no_words"
    ) and record.prev not in (None, *seen):
        earlier = await _bounded(locks.read_call(redis, record.prev or ""))
        if earlier is None or earlier.uid != user_id:
            break
        seen.add(earlier.call_id)
        record = earlier
        if not record.never_went_live:
            reached = record
    return reached


async def finish_call(
    user_id: str,
    call_id: str,
    background_tasks: BackgroundTasks,
    *,
    corrections: str | None = None,
) -> FinalizeResponse:
    """The person's confirm of their call, or their retry of its hand-off.

    A call that took the interview or that another take beat, one whose words
    the pipeline ruled on, or one whose transcript could not be made is
    answered as it stands, so a second click costs nothing; so is a call the
    person's typing superseded, answered ``superseded`` without the pipeline.
    A call waiting for its confirm, one whose hand-off the pipeline never
    ruled on, or one cut off with its words accepted hands its transcript of
    record off now, with *corrections* after it as the person's own words,
    and its accepted take becomes the interview's. The record is read again
    once this finish holds the call, so a finish that ran meanwhile is
    answered, never repeated. The pipeline's jobs run after the response.

    Raises:
        HTTPException: 404 for a call that is not theirs; 409
            ``call_not_ready`` while its words are not ready or another finish
            of it is running; 429 and 503 as the pipeline refuses a hand-off,
            or 503 while the call cannot be read.
    """
    redis = await _bounded(get_redis_async())
    record = await _their_call(redis, user_id, call_id)
    answered = _answer_as_it_stands(record)
    if answered is not None:
        return answered
    _refuse_unless_ready(record)
    holder = f"finish:{uuid.uuid4().hex}"
    try:
        claimed = await _bounded(
            locks.claim_postcall(redis, call_id, holder, ttl_s=HANDOFF_CLAIM_TTL_S)
        )
    except HTTPException:
        # The claim may have been written before the wait for its reply gave
        # up; left, it would hold every finish back until it lapsed.
        await _quietly(locks.release_postcall(redis, call_id, holder))
        raise
    if not claimed:
        raise HTTPException(status_code=409, detail={"code": CALL_NOT_READY})
    claim = HandOffClaim(redis, call_id, holder)
    try:
        # Another finish, or the call's job, may have answered since the read
        # above, and the words are judged once.
        record = await _their_call(redis, user_id, call_id)
        answered = _answer_as_it_stands(record)
        if answered is not None:
            return answered
        _refuse_unless_ready(record)
        async with claim.renewed():
            closure = await hand_off_call(
                redis,
                record,
                background_tasks,
                claim=claim,
                confirmed=True,
                by_person=True,
                corrections=corrections,
            )
    except (LostClaim, LaterCallUnsettled):
        # Another hand-off took the call over, or a later call of the
        # person's could still become a take: the words are not ready.
        raise HTTPException(status_code=409, detail={"code": CALL_NOT_READY}) from None
    finally:
        await _quietly(locks.release_postcall(redis, call_id, holder))
    if closure.hand_off is None:
        raise RuntimeError(f"Voice call {call_id} was not handed off")
    return closure.hand_off


async def end_call(user_id: str, call_id: str) -> None:
    """Ask the pod running *user_id*'s call *call_id* to end it, as
    ``superseded``: the person can no longer reach it from the device it is
    on. A call that has ended already is left as it is. A call that no longer
    runs (its pod died) frees the person's lock here, so they can call again
    or type at once. A start the browser ended before its call began (its
    room join failed, or the person chose to type meanwhile) has its room
    closed by the next sweep, and is counted.

    Raises:
        HTTPException: 404 for a call that is not theirs, 503 while it cannot
            be read.
    """
    redis = await _bounded(get_redis_async())
    record = await _bounded(locks.read_call(redis, call_id))
    theirs = record is not None and record.uid == user_id
    if not theirs:
        # A call whose record was lost is still theirs while it holds their
        # live-call lock.
        theirs = await _bounded(locks.live_call_of(redis, user_id)) == call_id
    abandoned = False
    if not theirs and record is None:
        # A call that never started is theirs while its room is registered
        # to them.
        abandoned = await _bounded(locks.room_owner(redis, call_id)) == user_id
        theirs = abandoned
    if not theirs:
        raise HTTPException(status_code=404, detail=NO_SUCH_CALL)
    await _bounded(locks.request_end(redis, call_id))
    if abandoned:
        ABANDONED_STARTS.inc()
        return
    if not await _bounded(locks.call_is_live(redis, call_id, time.time())):
        await _bounded(locks.release_user_lock(redis, user_id, call_id))


def _answer_as_it_stands(record: CallRecord) -> FinalizeResponse | None:
    """The answer a finish of *record* only repeats, or None when the finish
    has words to hand off."""
    if record.interview == locks.INTERVIEW_SUPERSEDED:
        return superseded_take(BrainDumpInputMode.voice)
    stored = record.hand_off
    if stored is None:
        return None
    if record.transcript_of_record is None or stored.error_code in _VERDICTS:
        return stored
    if record.status == locks.HANDED_OFF and (
        # A cut-off call's accepted take waits for the person's finish.
        record.took_interview
        or not record.accepted
    ):
        return stored
    return None


def _refuse_unless_ready(record: CallRecord) -> None:
    if record.status not in (locks.TRANSCRIBED, locks.FAILED, locks.HANDED_OFF) or (
        record.transcript_of_record is None
    ):
        raise HTTPException(status_code=409, detail={"code": CALL_NOT_READY})


async def _their_call(
    redis: AsyncRedisClient, user_id: str, call_id: str
) -> CallRecord:
    record = await _bounded(locks.read_call(redis, call_id))
    if record is None or record.uid != user_id:
        raise HTTPException(status_code=404, detail=NO_SUCH_CALL)
    return record


def _status_of(record: CallRecord) -> VoiceCallStatusResponse:
    # Written with the call's first record, so a record without one is one
    # whose fields could not be read: nothing is known to be running.
    status = "handing_off" if owed(record) else record.status or locks.ENDED
    running = status in (locks.CONNECTING, locks.LIVE)
    return VoiceCallStatusResponse(
        call_id=record.call_id,
        status=status,
        interview=record.interview,
        end_reason=record.end_reason,
        relay_url=relay_socket_url() if running else None,
        livekit_url=_livekit_url() if running else None,
        hand_off=record.hand_off,
    )


def _livekit_url() -> str | None:
    project = rooms.livekit_project(settings)
    return project.url if project is not None else None


async def _quietly(operation: Awaitable[object]) -> None:
    try:
        await asyncio.wait_for(operation, timeout=REDIS_TIMEOUT_S)
    except Exception as exc:
        logger.warning(f"Voice call state not written: {type(exc).__name__}")
