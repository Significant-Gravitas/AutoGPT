"""Redis-backed storage for in-flight ask_question choice buttons.

A native choice button/select (Discord component, Slack block action,
Telegram inline keyboard, Teams Adaptive Card action) round-trips a short
opaque token rather than the option text itself -- Telegram's
``callback_data`` caps at 64 bytes, too tight for arbitrary option text, and
carrying every platform's payload the same way keeps the click handlers
uniform. The click handler looks the real text up here by (platform, token,
index).
"""

import json
import uuid

from backend.data.redis_client import get_redis_async

CHOICE_TTL = 3600  # 1 hour -- long enough to answer, short enough that a
# stale button reliably reports "expired" instead of silently misfiring.


def _key(platform: str, token: str) -> str:
    return f"copilot-bot:choice:{platform}:{token}"


async def store_choice(platform: str, options: list[str]) -> str:
    """Persist a question's options, returning an opaque token that native
    button/select payloads can carry within their size limits."""
    token = uuid.uuid4().hex[:12]
    redis = await get_redis_async()
    await redis.set(_key(platform, token), json.dumps(options), ex=CHOICE_TTL)
    return token


async def resolve_choice(platform: str, token: str, index: int) -> str | None:
    """Atomically fetch-and-clear the option text for ``token``/``index``.

    Single-use by construction (GETDEL, one round trip): a double-click or a
    platform-level delivery retry racing the same click twice must not both
    see a valid token and continue the paused turn twice (each continuation
    is a billable AutoPilot turn) -- the second caller gets ``None``, same
    as an expired or forged payload.
    """
    redis = await get_redis_async()
    raw = await redis.getdel(_key(platform, token))
    if not raw:
        return None
    options = json.loads(raw)
    if not (0 <= index < len(options)):
        return None
    return options[index]


async def clear_choice(platform: str, token: str) -> None:
    """Invalidate a token that was never resolved (e.g. the native send
    itself failed) -- ``resolve_choice`` already clears on a successful
    resolve, so this is not needed on that path."""
    redis = await get_redis_async()
    await redis.delete(_key(platform, token))
