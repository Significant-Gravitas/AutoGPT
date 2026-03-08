"""E2B sandbox lifecycle for CoPilot: persistent cloud execution.

Each session gets a long-lived E2B cloud sandbox.  ``bash_exec`` runs commands
directly on the sandbox via ``sandbox.commands.run()``.  SDK file tools
(read_file/write_file/edit_file/glob/grep) route to the sandbox's
``/home/user`` directory via E2B's HTTP-based filesystem API — all tools
share a single coherent filesystem with no local sync required.

Lifecycle
---------
1. **Turn start** – connect to the existing sandbox (sandbox_id in Redis) or
   create a new one via ``get_or_create_sandbox()``.
   ``connect()`` in e2b v2 auto-resumes paused sandboxes.
2. **Execution** – ``bash_exec`` and MCP file tools operate directly on the
   sandbox's ``/home/user`` filesystem.
3. **Turn end** – the sandbox is paused via ``pause_sandbox()`` (fire-and-forget)
   so idle time between turns costs nothing.  Paused sandboxes have no compute
   cost.
4. **Session delete** – ``kill_sandbox()`` fully terminates the sandbox.

Cost control
------------
Sandboxes are created with ``lifecycle={"on_timeout": "pause"}`` so they are
automatically paused (not killed) if they somehow remain running past the
timeout.  The explicit per-turn ``pause_sandbox()`` call is the primary
mechanism; the lifecycle setting is a safety net.  Paused sandboxes are free.
Long-lived paused sandboxes are cleaned up by E2B project-level settings
("paused sandbox lifetime") — no scheduler needed on our side.

The sandbox_id is stored in Redis with a TTL.  Redis is also used for the
short-lived creation lock that prevents two concurrent requests from creating
two sandboxes for the same session.
"""

import asyncio
import contextlib
import logging
from typing import Callable, Coroutine

from e2b import AsyncSandbox

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_CREATION_LOCK_PREFIX = "copilot:e2b:creating:"
_SANDBOX_ID_PREFIX = "copilot:e2b:sandbox:"
E2B_WORKDIR = "/home/user"
_CREATION_LOCK_TTL = 60  # seconds
_MAX_WAIT_ATTEMPTS = 20  # 20 * 0.5s = 10s max wait

# How long the sandbox may run continuously before e2b auto-pauses it (safety
# net; per-turn explicit pause is the primary mechanism).
# Keep this short so a missed pause does not run up compute costs.
_E2B_PAUSE_TIMEOUT = 3600  # 1 hour in seconds

# Redis TTL for the sandbox_id key — long enough to survive across sessions.
# After this period the key expires and a fresh sandbox is created on next use.
_SANDBOX_ID_TTL = 30 * 24 * 3600  # 30 days


def _sandbox_key(session_id: str) -> str:
    return f"{_SANDBOX_ID_PREFIX}{session_id}"


async def _get_stored_sandbox_id(session_id: str) -> str | None:
    redis = await get_redis_async()
    raw = await redis.get(_sandbox_key(session_id))
    return raw.decode() if isinstance(raw, bytes) else raw


async def _set_stored_sandbox_id(session_id: str, sandbox_id: str) -> None:
    redis = await get_redis_async()
    await redis.set(_sandbox_key(session_id), sandbox_id, ex=_SANDBOX_ID_TTL)


async def _clear_stored_sandbox_id(session_id: str) -> None:
    redis = await get_redis_async()
    await redis.delete(_sandbox_key(session_id))


async def _try_reconnect(
    sandbox_id: str, session_id: str, api_key: str
) -> "AsyncSandbox | None":
    """Try to reconnect to an existing sandbox. Returns None on failure."""
    try:
        sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
        if await sandbox.is_running():
            return sandbox
    except Exception as exc:
        logger.warning("[E2B] Reconnect to %.12s failed: %s", sandbox_id, exc)

    # Stale — clear the sandbox_id from Redis so a new one can be created.
    await _clear_stored_sandbox_id(session_id)
    return None


async def get_or_create_sandbox(
    session_id: str,
    api_key: str,
    template: str = "base",
    pause_timeout: int = _E2B_PAUSE_TIMEOUT,
) -> AsyncSandbox:
    """Return the existing E2B sandbox for *session_id* or create a new one.

    The sandbox_id is stored in Redis so the same sandbox is reused across
    turns and service restarts.  Concurrent calls for the same session are
    serialised via a Redis ``SET NX`` creation lock.

    *pause_timeout* controls how long the e2b sandbox may run continuously
    before the ``on_timeout: kill`` lifecycle rule fires (default: 4 h).
    """
    redis = await get_redis_async()
    lock_key = f"{_CREATION_LOCK_PREFIX}{session_id}"

    # 1. Try reconnecting to an existing sandbox.
    stored_id = await _get_stored_sandbox_id(session_id)
    if stored_id:
        sandbox = await _try_reconnect(stored_id, session_id, api_key)
        if sandbox:
            logger.info(
                "[E2B] Reconnected to %.12s for session %.12s",
                stored_id,
                session_id,
            )
            return sandbox

    # 2. Claim creation lock. If another request holds it, wait for the result.
    claimed = await redis.set(lock_key, "1", nx=True, ex=_CREATION_LOCK_TTL)
    if not claimed:
        for _ in range(_MAX_WAIT_ATTEMPTS):
            await asyncio.sleep(0.5)
            new_id = await _get_stored_sandbox_id(session_id)
            if new_id:
                sandbox = await _try_reconnect(new_id, session_id, api_key)
                if sandbox:
                    return sandbox
                break  # Stale sandbox cleared — fall through to create

        # Try to claim creation lock again after waiting.
        claimed = await redis.set(lock_key, "1", nx=True, ex=_CREATION_LOCK_TTL)
        if not claimed:
            # Another process may have created a sandbox — try to use it.
            new_id = await _get_stored_sandbox_id(session_id)
            if new_id:
                sandbox = await _try_reconnect(new_id, session_id, api_key)
                if sandbox:
                    return sandbox
            raise RuntimeError(
                f"Could not acquire E2B creation lock for session {session_id[:12]}"
            )

    # 3. Create a new sandbox.
    try:
        sandbox = await AsyncSandbox.create(
            template=template,
            api_key=api_key,
            timeout=pause_timeout,
            lifecycle={"on_timeout": "pause"},
        )
        try:
            await _set_stored_sandbox_id(session_id, sandbox.sandbox_id)
        except Exception:
            # Redis save failed — kill the sandbox to avoid leaking it.
            with contextlib.suppress(Exception):
                await sandbox.kill()
            raise
    finally:
        # Always release the creation lock so other callers can proceed.
        await redis.delete(lock_key)
    logger.info(
        "[E2B] Created sandbox %.12s for session %.12s",
        sandbox.sandbox_id,
        session_id,
    )
    return sandbox


async def _act_on_sandbox(
    session_id: str,
    api_key: str,
    action: str,
    fn: Callable[[AsyncSandbox], Coroutine],
    *,
    clear_stored_id: bool = False,
) -> bool:
    """Connect to the sandbox for *session_id* and run *fn* on it.

    Shared by ``pause_sandbox`` and ``kill_sandbox``.  Returns ``True`` on
    success, ``False`` when no sandbox is found or the action fails.
    If *clear_stored_id* is ``True``, the sandbox_id is removed from Redis
    only after the action succeeds so a failed kill can be retried.
    """
    sandbox_id = await _get_stored_sandbox_id(session_id)
    if not sandbox_id:
        return False

    async def _connect_and_act():
        sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
        await fn(sandbox)

    try:
        await asyncio.wait_for(_connect_and_act(), timeout=10)
        if clear_stored_id:
            await _clear_stored_sandbox_id(session_id)
        logger.info(
            "[E2B] %s sandbox %.12s for session %.12s",
            action.capitalize(),
            sandbox_id,
            session_id,
        )
        return True
    except Exception as exc:
        logger.warning(
            "[E2B] Failed to %s sandbox %.12s for session %.12s: %s",
            action,
            sandbox_id,
            session_id,
            exc,
        )
        return False


async def pause_sandbox(session_id: str, api_key: str) -> bool:
    """Pause the E2B sandbox for *session_id* to stop billing between turns.

    Paused sandboxes cost nothing and are resumed automatically by
    ``get_or_create_sandbox()`` on the next turn (via ``AsyncSandbox.connect()``).
    The sandbox_id is kept in Redis so reconnection works seamlessly.

    Returns ``True`` if the sandbox was found and paused, ``False`` otherwise.
    Safe to call even when no sandbox exists for the session.
    """
    return await _act_on_sandbox(session_id, api_key, "pause", lambda sb: sb.pause())


async def kill_sandbox(
    session_id: str,
    api_key: str,
) -> bool:
    """Kill the E2B sandbox for *session_id* and clear its Redis entry.

    Returns ``True`` if a sandbox was found and killed, ``False`` otherwise.
    Safe to call even when no sandbox exists for the session.
    """
    return await _act_on_sandbox(
        session_id,
        api_key,
        "kill",
        lambda sb: sb.kill(),
        clear_stored_id=True,
    )
