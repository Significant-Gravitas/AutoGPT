"""What the person's typed interview has heard so far, for its ending
screens.

Read with the shared extraction (``onboarding_interview_details``) on the
platform's Gemini key, on the same lite model the voice relay reads a call's
with, and kept per interview by the number of the person's turns: a read
for a count already read answers from the cache, so the step can ask after
every turn and the model is asked at most once per turn. The interview's
turns are capped, which bounds what one interview can spend here.
"""

import asyncio
import json
import logging

from google.genai import Client

from backend.api.features.onboarding_dump.interview import interview_transcript
from backend.copilot.expert_kickoff import is_hidden_chat_message
from backend.copilot.model import ChatSession
from backend.copilot.onboarding_interview_details import (
    InterviewDetails,
    extract_details,
)
from backend.data.redis_client import get_redis_async
from backend.util.settings import Settings
from backend.voice_relay.config import DETAILS_MODEL

logger = logging.getLogger(__name__)
settings = Settings()

_CACHE_KEY = "onboarding:interview:details:{session_id}"
_CACHE_TTL_S = 24 * 3600
READ_TIMEOUT_S = 8.0


def interview_lines(session: ChatSession) -> list[tuple[str, str]]:
    """The interview as the extraction reads it: the person's own words, as
    the hand-off takes them, between Otto's replies."""
    own = interview_transcript(session.messages).splitlines()
    lines: list[tuple[str, str]] = []
    for message in session.messages:
        if is_hidden_chat_message(message.metadata):
            continue
        if message.role == "assistant" and (message.content or "").strip():
            lines.append(("otto", message.content or ""))
        elif message.role == "user" and own:
            lines.append(("user", own.pop(0)))
    return lines


async def read_interview_details(session: ChatSession) -> InterviewDetails | None:
    """The details of the interview *session* as it stands; None before the
    person has said anything, without a Gemini key, or when the read fails."""
    lines = interview_lines(session)
    turns = sum(1 for who, _ in lines if who == "user")
    if turns == 0 or not settings.secrets.gemini_api_key:
        return None
    key = _CACHE_KEY.format(session_id=session.session_id)
    redis = await get_redis_async()
    try:
        cached = await redis.get(key)
        if cached:
            kept = json.loads(cached)
            if kept.get("turns") == turns:
                return InterviewDetails.model_validate(kept["details"])
    except Exception:
        logger.warning("Interview details cache unreadable", exc_info=True)
    details = await _read(lines)
    if details is None:
        return None
    try:
        await redis.set(
            key,
            json.dumps({"turns": turns, "details": details.model_dump(mode="json")}),
            ex=_CACHE_TTL_S,
        )
    except Exception:
        logger.warning("Interview details not cached", exc_info=True)
    return details


async def _read(lines: list[tuple[str, str]]) -> InterviewDetails | None:
    client = Client(api_key=settings.secrets.gemini_api_key)
    try:
        read = await asyncio.wait_for(
            extract_details(client, model=DETAILS_MODEL, lines=lines),
            timeout=READ_TIMEOUT_S,
        )
    except Exception:
        logger.warning("Interview details not read", exc_info=True)
        return None
    return read.details
