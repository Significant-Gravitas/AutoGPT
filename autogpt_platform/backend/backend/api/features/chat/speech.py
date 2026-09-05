"""Speech endpoint for AutoPilot voice mode.

Proxies OpenAI text-to-speech so the platform key never reaches the browser
and every spoken chunk is metered against the caller's AutoPilot plan.
"""

import logging
from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Response, Security
from pydantic import BaseModel, Field

from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    check_rate_limit,
    enforce_payment_paywall,
    get_global_rate_limits,
)
from backend.copilot.speech import (
    AUDIO_MEDIA_TYPE,
    MAX_SPEECH_CHARS,
    SpeechUnavailable,
    synthesize_speech,
)
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

config = ChatConfig()

router = APIRouter(tags=["chat", "speech"])

MAX_SESSION_ID_CHARS = 128


class SpeechRequest(BaseModel):
    text: str = Field(max_length=MAX_SPEECH_CHARS)
    session_id: str | None = Field(
        default=None,
        max_length=MAX_SESSION_ID_CHARS,
        description="Attributes the cost to a chat session, as a text turn is.",
    )
    voice: str | None = Field(
        default=None, description="Overrides the configured default voice."
    )


@router.post(
    "/speech",
    responses={200: {"content": {AUDIO_MEDIA_TYPE: {}}}},
    response_class=Response,
)
async def synthesize(
    request: SpeechRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> Response:
    """Return *text* as MP3 audio, charging the caller for the synthesis."""
    if not await is_feature_enabled(Flag.COPILOT_VOICE_MODE, user_id):
        raise HTTPException(status_code=404, detail="Voice mode is not enabled")

    await _enforce_spend_allowance(user_id)

    try:
        audio = await synthesize_speech(
            user_id=user_id,
            text=request.text,
            session_id=request.session_id,
            voice=request.voice,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except SpeechUnavailable:
        logger.error("Voice mode is enabled but no OpenAI key is configured")
        raise HTTPException(status_code=503, detail="Speech is unavailable")

    return Response(content=audio, media_type=AUDIO_MEDIA_TYPE)


async def _enforce_spend_allowance(user_id: str) -> None:
    """The pre-flight a chat turn runs, before this route spends the same budget.

    Metering after the fact only slows the *next* turn down; without this a
    caller past their cap keeps synthesising and the platform pays.
    """
    await enforce_payment_paywall(user_id)

    try:
        daily_limit, weekly_limit, _ = await get_global_rate_limits(
            user_id,
            config.daily_cost_limit_microdollars,
            config.weekly_cost_limit_microdollars,
        )
        await check_rate_limit(
            user_id=user_id,
            daily_cost_limit=daily_limit,
            weekly_cost_limit=weekly_limit,
        )
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RateLimitUnavailable as exc:
        # Fail closed, as the chat route does: a Redis brown-out cannot prove
        # the caller is under their cap.
        raise HTTPException(
            status_code=503,
            detail="Rate limit service degraded, retry shortly",
            headers={"Retry-After": "30"},
        ) from exc
