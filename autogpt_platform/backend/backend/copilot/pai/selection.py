"""Whether a turn runs on the pai engine.

Off by default. Two switches turn it on:

* ``COPILOT_ENGINE=pai`` (or ``CHAT_ENGINE=pai``) in the environment — the
  whole deployment, for dev stacks and canaries.
* A per-session opt-in kept in Redis (:func:`set_session_engine`), so one chat
  can be moved onto the engine without touching anyone else's.

Import-light on purpose: the API layer and the processor ask this before any
engine module is loaded.
"""

import asyncio
import logging
import os
from typing import Literal

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

PAI_ENGINE = "pai"
ENGINE_ENV_VARS = ("COPILOT_ENGINE", "CHAT_ENGINE")
# A month: an opt-in outlives any one conversation's activity window, and an
# expired key only falls back to the default engines.
_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60
_SESSION_KEY = "copilot:engine:session:"
# The opt-in is read on every turn; a slow Redis must not hold the turn.
_READ_TIMEOUT_S = 0.5

EngineName = Literal["pai"]


def env_engine() -> str | None:
    """The engine the environment pins every turn to, lower-cased, or None."""
    for name in ENGINE_ENV_VARS:
        value = (os.environ.get(name) or "").strip().lower()
        if value:
            return value
    return None


async def session_engine(session_id: str | None) -> str | None:
    """The engine this session opted into, or None (also on any Redis error)."""
    if not session_id:
        return None
    try:
        value = await asyncio.wait_for(
            _read(_SESSION_KEY + session_id), timeout=_READ_TIMEOUT_S
        )
    except Exception:
        logger.warning(f"[PAI] Could not read the engine opt-in for {session_id}")
        return None
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip().lower() or None


async def _read(key: str) -> object:
    redis = await get_redis_async()
    return await redis.get(key)


async def set_session_engine(session_id: str, engine: EngineName | None) -> None:
    """Opt *session_id* into *engine*; ``None`` clears the opt-in."""
    redis = await get_redis_async()
    key = _SESSION_KEY + session_id
    if engine is None:
        await redis.delete(key)
        return
    await redis.set(key, engine, ex=_SESSION_TTL_SECONDS)


async def pai_engine_enabled(session_id: str | None) -> bool:
    """True when this turn should run on the pai engine."""
    if env_engine() == PAI_ENGINE:
        return True
    return await session_engine(session_id) == PAI_ENGINE
