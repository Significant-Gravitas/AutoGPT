"""Speech endpoint for AutoPilot voice mode.

Proxies OpenAI text-to-speech so the platform key never reaches the browser
and every spoken chunk is metered against the caller's AutoPilot plan.
"""

import logging
from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Response, Security
from pydantic import BaseModel, Field

from backend.copilot.speech import (
    AUDIO_MEDIA_TYPE,
    MAX_SPEECH_CHARS,
    SpeechUnavailable,
    synthesize_speech,
)
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat", "speech"])


class SpeechRequest(BaseModel):
    text: str = Field(max_length=MAX_SPEECH_CHARS)
    session_id: str | None = Field(
        default=None,
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
