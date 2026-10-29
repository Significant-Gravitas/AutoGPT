"""How long a plain chat keeps its sandbox, and telling a turn when it did not.

A plain chat's box is paused between turns and found again through the id
cached in Redis.  When that box is gone (E2B no longer has it, or it ran a
superseded image and was retired) the turn gets a new one: ``~/workspace``
carries over as a volume, and every other file, installed tool and browser
sign-in does not.  Without a word the model only learns that from a failed
command, so the turn that got the new box is told once.
"""

import logging

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# How long a plain chat's sandbox id is kept after its last turn.  Paused boxes
# stay on E2B until something kills them, so a shorter key only orphans a box
# that still exists: the next turn can no longer find it and starts afresh.
SESSION_SANDBOX_TTL = 30 * 24 * 3600

# Opening the desktop can replace a box before the next chat turn. Keep the
# unread notice as long as the session can find that box.
REPLACED_FLAG_TTL = SESSION_SANDBOX_TTL

_REPLACED_KEY_PREFIX = "copilot:e2b:replaced:"
SANDBOX_STATUS_TAG = "sandbox_status"

SANDBOX_REPLACED_NOTICE = (
    f"<{SANDBOX_STATUS_TAG}>\n"
    "Your sandbox was replaced since your last turn. Only `~/workspace` (and "
    "`~/shared` on an expert's computer) carried over; other files, installed "
    "tools and browser sign-ins are gone. Check before relying on them, and "
    "set up again what the task needs.\n"
    f"</{SANDBOX_STATUS_TAG}>\n\n"
)


def sandbox_replaced_key(session_id: str) -> str:
    """Redis flag for the next turn after a gone box was replaced."""
    return f"{_REPLACED_KEY_PREFIX}{session_id}"


async def build_sandbox_replaced_notice(session_id: str) -> str:
    """The ``<sandbox_status>`` block when this turn's box replaced a gone one.

    Read once: the flag is cleared as it is read, so only the next turn is
    told. Never fails a turn; any error means no notice.
    """
    try:
        redis = await get_redis_async()
        if await redis.getdel(sandbox_replaced_key(session_id)):
            return SANDBOX_REPLACED_NOTICE
    except Exception:
        logger.warning("[E2B] sandbox-replaced flag unreadable", exc_info=True)
    return ""
