"""Starting a voice call for the onboarding interview, and the call after.

The REST API decides who may call; the relay (``backend.voice_relay``) runs
the call. A start answers with the relay's socket and a signed, single-use
start token naming the person, the call and the leg its audio takes, which
the relay trusts without a platform session. In order, it checks:

- the voice flag: 404 when it is off for the person or cannot be read;
- that this deployment has a relay to call: 503 when it has none;
- that the person's interview is not over: 409. A person has one interview,
  typed or spoken, found on their onboarding row: it is over once the typed
  interview is, or once a call's words became its take (naming that call);
- that the person has no call live already: 409 naming the live call;
- the platform's daily voice budget and its live-call ceiling: 503 with
  ``Retry-After``;
- the person's own call allowance, on every tier: 429 with ``Retry-After``.

The relay records the call on the person's onboarding row once the call
starts, and its record in Redis for a day. Once it ends, the person's own
routes read the call's state (:func:`call_status`, :func:`latest_call`),
confirm its hand-off or hand it off again (:func:`finish_call`), and end a
call of theirs live elsewhere (:func:`end_call`). Another person's call is
404 on every one of them.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import TypeVar

from fastapi import BackgroundTasks, HTTPException

from backend.api.features.onboarding_dump import quality
from backend.api.features.onboarding_dump.interview import call_that_took_the_interview
from backend.api.features.onboarding_dump.models import (
    FinalizeResponse,
    VoiceCallCaps,
    VoiceCallLeg,
    VoiceCallStartResponse,
    VoiceCallStatusResponse,
)
from backend.api.features.onboarding_dump.rate_limit import enforce_voice_call_budget
from backend.copilot.onboarding_interview import (
    INTERVIEW_CLOSED,
    find_onboarding_interview_session,
    interview_is_over,
)
from backend.data.onboarding import get_onboarding_voice_call_id
from backend.data.redis_client import AsyncRedisClient, get_redis_async
from backend.util.feature_flag import (
    Flag,
    evaluate_feature_flag,
    get_feature_flag_value,
)
from backend.util.settings import AppEnvironment, Settings
from backend.voice_relay import locks, tokens
from backend.voice_relay.config import CAPS, TOKEN_TTL_S
from backend.voice_relay.handoff import hand_off_call
from backend.voice_relay.locks import CallRecord

logger = logging.getLogger(__name__)
settings = Settings()

T = TypeVar("T")

REDIS_TIMEOUT_S = 0.25
# How long to wait before trying again: while the platform is at its live-call
# ceiling, and while voice cannot be served at all.
BUSY_RETRY_AFTER_S = 30
UNAVAILABLE_RETRY_AFTER_S = 300

CALL_IN_PROGRESS = locks.CALL_IN_PROGRESS
# Why a finish is refused: the call's words are not ready for a hand-off (the
# call is live, or its transcript is still being made), or another finish of
# the same call is handing them off.
CALL_NOT_READY = "call_not_ready"
NO_SUCH_CALL = "No such voice call"
# How long a finish holds the call while it hands the words off.
FINISH_CLAIM_TTL_S = 120
# Answers on the words, which another hand-off of the same words would only
# repeat: the person calls again or types instead.
_VERDICTS = frozenset({quality.NO_USABLE_SPEECH, quality.INSUFFICIENT_CONTENT})

# The legs this deployment serves, by the edge flag's name for each. A leg the
# flag names that is not served falls back to the control socket, which every
# browser can use.
DEFAULT_LEG = VoiceCallLeg(kind="websocket")
SERVED_LEGS: dict[str, VoiceCallLeg] = {DEFAULT_LEG.kind: DEFAULT_LEG}


async def start_voice_call(user_id: str) -> VoiceCallStartResponse:
    """Start a voice call for *user_id*'s onboarding interview.

    Raises:
        HTTPException: 404, 409, 429 or 503, as the module describes.
    """
    if not await _voice_enabled(user_id):
        raise HTTPException(status_code=404, detail="Feature not available")
    relay_url = relay_socket_url()
    if relay_url is None or not tokens.is_configured():
        raise _unavailable(
            "Voice calls are not available right now.", UNAVAILABLE_RETRY_AFTER_S
        )
    await _refuse_if_interview_over(user_id)
    redis = await _bounded(get_redis_async())
    live_call = await _bounded(locks.live_call_of(redis, user_id))
    if live_call:
        raise HTTPException(
            status_code=409, detail={"code": CALL_IN_PROGRESS, "call_id": live_call}
        )
    await _refuse_past_platform_limits(redis)
    await enforce_voice_call_budget(user_id)
    leg = await _choose_leg(user_id)
    call_id = tokens.new_call_id()
    return VoiceCallStartResponse(
        call_id=call_id,
        relay_url=relay_url,
        token=tokens.mint_start(user_id, call_id, leg.kind),
        expires_in=TOKEN_TTL_S,
        leg=leg,
        caps=VoiceCallCaps.model_validate(CAPS.as_message()),
    )


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


async def _choose_leg(user_id: str) -> VoiceCallLeg:
    try:
        wanted = await get_feature_flag_value(
            Flag.ONBOARDING_INTERVIEW_VOICE_EDGE.value, user_id, default="livekit"
        )
    except Exception:
        logger.warning("Voice interview edge flag could not be read", exc_info=True)
        return DEFAULT_LEG
    return (
        SERVED_LEGS.get(wanted, DEFAULT_LEG) if isinstance(wanted, str) else DEFAULT_LEG
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
    """*call_id* as it stands, when it is *user_id*'s.

    Raises:
        HTTPException: 404 for a call that is not theirs or has expired, 503
            while its record cannot be read.
    """
    redis = await _bounded(get_redis_async())
    return _status_of(await _their_call(redis, user_id, call_id))


async def latest_call(user_id: str) -> VoiceCallStatusResponse:
    """*user_id*'s latest call as it stands, for a step that reloads.

    Raises:
        HTTPException: 404 when they have none or its record has expired, 503
            while it cannot be read.
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
    return await call_status(user_id, call_id)


async def finish_call(
    user_id: str,
    call_id: str,
    background_tasks: BackgroundTasks,
    *,
    corrections: str | None = None,
) -> FinalizeResponse:
    """The person's confirm of their call, or their retry of its hand-off.

    A call whose hand-off closed the interview, whose words the pipeline ruled
    on, or whose transcript could not be made is answered as it stands, so a
    second click costs nothing. A call waiting for its confirm, or one whose
    hand-off the pipeline never ruled on, hands its transcript of record off
    now, with *corrections* after it as the person's own words. The
    pipeline's jobs run after the response.

    Raises:
        HTTPException: 404 for a call that is not theirs; 409
            ``call_not_ready`` while its words are not ready or another finish
            of it is running; 429 and 503 as the pipeline refuses a hand-off,
            or 503 while the call cannot be read.
    """
    redis = await _bounded(get_redis_async())
    record = await _their_call(redis, user_id, call_id)
    stored = record.hand_off
    if stored is not None and (
        record.status == locks.HANDED_OFF
        or stored.error_code in _VERDICTS
        or record.transcript_of_record is None
    ):
        return stored
    if record.status not in (locks.TRANSCRIBED, locks.FAILED) or (
        record.transcript_of_record is None
    ):
        raise HTTPException(status_code=409, detail={"code": CALL_NOT_READY})
    holder = f"finish:{uuid.uuid4().hex}"
    if not await _bounded(
        locks.claim_postcall(redis, call_id, holder, ttl_s=FINISH_CLAIM_TTL_S)
    ):
        raise HTTPException(status_code=409, detail={"code": CALL_NOT_READY})
    try:
        closure = await hand_off_call(
            redis,
            record,
            background_tasks,
            confirmed=True,
            corrections=corrections,
        )
    finally:
        await _quietly(locks.release_postcall(redis, call_id, holder))
    if closure.hand_off is None:
        raise RuntimeError(f"Voice call {call_id} was not handed off")
    return closure.hand_off


async def end_call(user_id: str, call_id: str) -> None:
    """Ask the pod running *user_id*'s call *call_id* to end it, as
    ``superseded``: the person can no longer reach it from the device it is
    on. A call that has ended already is left as it is.

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
    if not theirs:
        raise HTTPException(status_code=404, detail=NO_SUCH_CALL)
    await _bounded(locks.request_end(redis, call_id))


async def _their_call(
    redis: AsyncRedisClient, user_id: str, call_id: str
) -> CallRecord:
    record = await _bounded(locks.read_call(redis, call_id))
    if record is None or record.uid != user_id:
        raise HTTPException(status_code=404, detail=NO_SUCH_CALL)
    return record


def _status_of(record: CallRecord) -> VoiceCallStatusResponse:
    return VoiceCallStatusResponse(
        call_id=record.call_id,
        # Written with the call's first record, so a record without one is
        # one whose fields could not be read: nothing is known to be running.
        status=record.status or locks.ENDED,
        end_reason=record.end_reason,
        leg="livekit" if record.leg == "livekit" else "websocket",
        hand_off=record.hand_off,
    )


async def _quietly(operation: Awaitable[object]) -> None:
    try:
        await asyncio.wait_for(operation, timeout=REDIS_TIMEOUT_S)
    except Exception as exc:
        logger.warning(f"Voice call state not written: {type(exc).__name__}")
