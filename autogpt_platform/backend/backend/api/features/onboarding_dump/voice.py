"""Starting a voice call for the onboarding interview.

The REST API decides who may call; the relay (``backend.voice_relay``) runs
the call. A start answers with the relay's socket and a signed, single-use
start token naming the person, the call and the leg its audio takes, which
the relay trusts without a platform session. In order, it checks:

- the voice flag: 404 when it is off for the person or cannot be read;
- that this deployment has a relay to call: 503 when it has none;
- that the person's interview is not over: 409;
- that the person has no call live already: 409 naming the live call;
- the platform's daily voice budget and its live-call ceiling: 503 with
  ``Retry-After``;
- the person's own call allowance: 429 with ``Retry-After``.
"""

import asyncio
import logging
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import TypeVar

from fastapi import HTTPException

from backend.api.features.onboarding_dump.models import (
    VoiceCallCaps,
    VoiceCallLeg,
    VoiceCallStartResponse,
)
from backend.data.redis_client import AsyncRedisClient, get_redis_async
from backend.util.feature_flag import (
    Flag,
    evaluate_feature_flag,
    get_feature_flag_value,
)
from backend.util.settings import AppEnvironment, Settings
from backend.voice_relay import locks, tokens
from backend.voice_relay.config import CAPS, TOKEN_TTL_S

logger = logging.getLogger(__name__)
settings = Settings()

T = TypeVar("T")

REDIS_TIMEOUT_S = 0.25
# How long to wait before trying again: while the platform is at its live-call
# ceiling, and while voice cannot be served at all.
BUSY_RETRY_AFTER_S = 30
UNAVAILABLE_RETRY_AFTER_S = 300

CALL_IN_PROGRESS = "call_in_progress"

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
    redis = await _bounded(get_redis_async())
    live_call = await _bounded(locks.live_call_of(redis, user_id))
    if live_call:
        raise HTTPException(
            status_code=409, detail={"code": CALL_IN_PROGRESS, "call_id": live_call}
        )
    await _refuse_past_platform_limits(redis)
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
