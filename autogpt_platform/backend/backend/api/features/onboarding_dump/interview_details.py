"""What the person's typed interview has heard so far, for its ending
screens.

Read with the shared extraction (``onboarding_interview_details``) on the
platform's Gemini key, on the same lite model the voice relay reads a call's
with. Each read is kept per interview and per count of the person's turns,
an empty one too, and the model is asked at most ``MAX_READS_PER_TURN`` times
for one count however many reads arrive at once: a counter per count of
turns is taken before each model call. The interview's turns are capped, so
one interview can ask the model at most that many times per turn. Each
call's tokens go to the platform cost log, on the platform's account.
"""

import asyncio
import json
import logging

from google.genai import Client

from backend.api.features.onboarding_dump.interview import interview_transcript
from backend.copilot.expert_kickoff import is_hidden_chat_message
from backend.copilot.model import ChatSession
from backend.copilot.onboarding_interview_details import (
    DetailsRead,
    InterviewDetails,
    extract_details,
)
from backend.copilot.token_tracking import persist_and_record_usage
from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import incr_with_ttl
from backend.util.settings import Settings
from backend.voice_relay.config import DETAILS_MODEL, price_of

logger = logging.getLogger(__name__)
settings = Settings()

_CACHE_KEY = "onboarding:interview:details:{session_id}:{turns}"
_READS_KEY = "onboarding:interview:details_reads:{session_id}:{turns}"
_CACHE_TTL_S = 24 * 3600
# An empty read is kept for a while, so a turn whose read came back empty is
# not read again on every request.
_EMPTY_TTL_S = 10 * 60
MAX_READS_PER_TURN = 2
READ_TIMEOUT_S = 8.0
BLOCK_NAME = "onboarding:interview_details"
COST_SOURCE = "onboarding_interview_details"


def interview_lines(session: ChatSession) -> list[tuple[str, str]]:
    """The interview as the extraction reads it: each of the person's
    messages whole, in their own words as the hand-off takes them, between
    Otto's replies."""
    lines: list[tuple[str, str]] = []
    for message in session.messages:
        if is_hidden_chat_message(message.metadata):
            continue
        if message.role == "assistant" and (message.content or "").strip():
            lines.append(("otto", message.content or ""))
        elif message.role == "user":
            own = interview_transcript([message]).strip()
            if own:
                lines.append(("user", own))
    return lines


async def read_interview_details(session: ChatSession) -> InterviewDetails | None:
    """The details of the interview *session* as it stands; None before the
    person has said anything, without a Gemini key, when the read comes back
    empty, and once this turn's reads are used up with none kept."""
    lines = interview_lines(session)
    turns = sum(1 for who, _ in lines if who == "user")
    if turns == 0 or not settings.secrets.gemini_api_key:
        return None
    keys = {"session_id": session.session_id, "turns": turns}
    cache_key = _CACHE_KEY.format(**keys)
    redis = await get_redis_async()
    try:
        cached = await redis.get(cache_key)
    except Exception as exc:
        logger.warning(f"Interview details cache unreadable: {type(exc).__name__}")
        return None
    if cached is not None:
        return _kept(cached)
    try:
        reads = await incr_with_ttl(redis, _READS_KEY.format(**keys), _CACHE_TTL_S)
    except Exception as exc:
        logger.warning(f"Interview details reads uncounted: {type(exc).__name__}")
        return None
    if reads > MAX_READS_PER_TURN:
        return None
    details = await _read(session, lines)
    try:
        await redis.set(
            cache_key,
            json.dumps(details.model_dump(mode="json") if details else None),
            ex=_CACHE_TTL_S if details else _EMPTY_TTL_S,
            # An empty read never takes the place of one already kept.
            nx=details is None,
        )
    except Exception as exc:
        logger.warning(f"Interview details not cached: {type(exc).__name__}")
    return details


def _kept(cached: str | bytes) -> InterviewDetails | None:
    try:
        value = json.loads(cached)
        return None if value is None else InterviewDetails.model_validate(value)
    except Exception:
        logger.warning("Interview details kept in the cache could not be read")
        return None


async def _read(
    session: ChatSession, lines: list[tuple[str, str]]
) -> InterviewDetails | None:
    client = Client(api_key=settings.secrets.gemini_api_key)
    try:
        read = await asyncio.wait_for(
            extract_details(client, model=DETAILS_MODEL, lines=lines),
            timeout=READ_TIMEOUT_S,
        )
    except Exception as exc:
        logger.warning(f"Interview details not read: {type(exc).__name__}")
        return None
    await _log_cost(session, read)
    return read.details


async def _log_cost(session: ChatSession, read: DetailsRead) -> None:
    """The read's tokens on the platform cost log, on the platform's
    account: the interview comes before any plan."""
    price = price_of(DETAILS_MODEL)
    usd = read.tokens_in * price.per_token(
        "text_in"
    ) + read.tokens_out * price.per_token("text_out")
    try:
        await persist_and_record_usage(
            session=None,
            user_id=session.user_id,
            prompt_tokens=read.tokens_in,
            completion_tokens=read.tokens_out,
            log_prefix="[InterviewDetails]",
            cost_usd=usd,
            model=DETAILS_MODEL,
            provider="google",
            block_name_override=BLOCK_NAME,
            extra_metadata={
                "source": COST_SOURCE,
                "session_id": session.session_id,
            },
            charge_plan=False,
        )
    except Exception as exc:
        logger.warning(f"Interview details cost not logged: {type(exc).__name__}")
