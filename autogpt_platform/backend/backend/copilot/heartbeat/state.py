"""Redis markers the heartbeat keeps between runs.

Kept apart from the runner so the ``heartbeat_respond`` tool can record its
answer without importing the runner (and through it the executor).
"""

import hashlib
import json
import logging
from datetime import UTC, datetime

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# ``ChatSessionMetadata.kind`` of a heartbeat's own session; listings hide it
# like a dream session, and ``heartbeat_respond`` only answers in one.
HEARTBEAT_SESSION_KIND = "heartbeat"
HEARTBEAT_SESSION_TITLE = "Heartbeat"

_PREFIX = "copilot:heartbeat:"
# An alert identical to the last one is held back for this long.
DEDUPE_TTL_SECONDS = 24 * 60 * 60
# Long enough to cover any interval; a lost marker only costs one extra beat.
_LAST_RUN_TTL_SECONDS = 7 * 24 * 60 * 60
_RESPONSE_TTL_SECONDS = 60 * 60


def last_run_key(user_id: str) -> str:
    return f"{_PREFIX}last_run:{user_id}"


def last_alert_key(user_id: str) -> str:
    return f"{_PREFIX}last_alert:{user_id}"


def response_key(session_id: str) -> str:
    return f"{_PREFIX}response:{session_id}"


async def get_last_run(user_id: str) -> datetime | None:
    try:
        redis = await get_redis_async()
        raw = await redis.get(last_run_key(user_id))
    except Exception:
        logger.warning("Heartbeat could not read its last run", exc_info=True)
        return None
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(_text(raw))
    except ValueError:
        return None


async def set_last_run(user_id: str, when: datetime | None = None) -> None:
    when = when or datetime.now(UTC)
    try:
        redis = await get_redis_async()
        await redis.setex(
            last_run_key(user_id), _LAST_RUN_TTL_SECONDS, when.isoformat()
        )
    except Exception:
        logger.warning("Heartbeat could not record its run", exc_info=True)


async def clear_last_run(user_id: str) -> None:
    """Forget the last beat, so the next one runs whatever changed: a new
    checklist is itself a change the signal cannot see."""
    try:
        redis = await get_redis_async()
        await redis.delete(last_run_key(user_id))
    except Exception:
        logger.warning("Heartbeat could not clear its last run", exc_info=True)


async def claim_manual_run(user_id: str, cooldown_seconds: int) -> bool:
    """One "run now" per cooldown per user: each run is a model call."""
    try:
        redis = await get_redis_async()
        return bool(
            await redis.set(
                f"{_PREFIX}manual:{user_id}", "1", nx=True, ex=cooldown_seconds
            )
        )
    except Exception:
        logger.warning("Heartbeat could not claim a manual run", exc_info=True)
        return False


def alert_fingerprint(text: str) -> str:
    """Case, spacing and punctuation-insensitive, so a reworded copy of the
    same alert still counts as a repeat."""
    normalized = " ".join(
        "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).split()
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


async def is_repeat_alert(user_id: str, text: str) -> bool:
    """Whether this alert matches the last one sent within the dedupe window.

    An unreadable store sends: a repeat is a nuisance, a lost alert is not
    recoverable.
    """
    try:
        redis = await get_redis_async()
        raw = await redis.get(last_alert_key(user_id))
    except Exception:
        logger.warning("Heartbeat could not read its last alert", exc_info=True)
        return False
    return raw is not None and _text(raw) == alert_fingerprint(text)


async def remember_alert(user_id: str, text: str) -> None:
    try:
        redis = await get_redis_async()
        await redis.setex(
            last_alert_key(user_id), DEDUPE_TTL_SECONDS, alert_fingerprint(text)
        )
    except Exception:
        logger.warning("Heartbeat could not record its alert", exc_info=True)


async def record_response(session_id: str, notify: bool, text: str) -> bool:
    """What ``heartbeat_respond`` answered in this heartbeat session."""
    try:
        redis = await get_redis_async()
        await redis.setex(
            response_key(session_id),
            _RESPONSE_TTL_SECONDS,
            json.dumps({"notify": notify, "notification_text": text}),
        )
        return True
    except Exception:
        logger.warning("Heartbeat could not record a response", exc_info=True)
        return False


async def read_response(session_id: str) -> dict | None:
    try:
        redis = await get_redis_async()
        raw = await redis.get(response_key(session_id))
    except Exception:
        logger.warning("Heartbeat could not read its response", exc_info=True)
        return None
    if raw is None:
        return None
    try:
        parsed = json.loads(_text(raw))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _text(raw: bytes | str) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)
