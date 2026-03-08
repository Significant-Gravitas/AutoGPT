"""E2B sandbox lifecycle for CoPilot: persistent cloud execution.

Each session gets a long-lived E2B cloud sandbox.  ``bash_exec`` runs commands
directly on the sandbox via ``sandbox.commands.run()``.  SDK file tools
(read_file/write_file/edit_file/glob/grep) route to the sandbox's
``/home/user`` directory via E2B's HTTP-based filesystem API — all tools
share a single coherent filesystem with no local sync required.

Lifecycle
---------
1. **Turn start** – connect to the existing sandbox (sandbox_id in Redis) or
   create a new one via ``get_or_create_sandbox()``.  ``connect()`` in e2b v2
   auto-resumes paused sandboxes.
2. **Execution** – ``bash_exec`` and MCP file tools operate directly on the
   sandbox's ``/home/user`` filesystem.
3. **Turn end** – the sandbox is paused via ``pause_sandbox()`` so idle time
   between turns costs nothing.
4. **Session delete** – ``kill_sandbox()`` fully terminates the sandbox.

Cost control
------------
Sandboxes are created with ``lifecycle={"on_timeout": "pause"}`` so they
auto-pause (rather than terminate) if they somehow remain running past the
timeout.  The explicit per-turn ``pause_sandbox()`` call is the primary
mechanism; the lifecycle setting is a safety net.
"""

import asyncio
import logging

from e2b import AsyncSandbox

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_SANDBOX_REDIS_PREFIX = "copilot:e2b:sandbox:"
E2B_WORKDIR = "/home/user"
_CREATING = "__creating__"
_CREATION_LOCK_TTL = 60
_MAX_WAIT_ATTEMPTS = 20  # 20 * 0.5s = 10s max wait


async def _try_reconnect(
    sandbox_id: str, api_key: str, redis_key: str, timeout: int
) -> "AsyncSandbox | None":
    """Try to reconnect to an existing sandbox. Returns None on failure."""
    try:
        sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
        if await sandbox.is_running():
            redis = await get_redis_async()
            await redis.expire(redis_key, timeout)
            return sandbox
    except Exception as exc:
        logger.warning("[E2B] Reconnect to %.12s failed: %s", sandbox_id, exc)

    # Stale — clear Redis so a new sandbox can be created.
    redis = await get_redis_async()
    await redis.delete(redis_key)
    return None


async def get_or_create_sandbox(
    session_id: str,
    api_key: str,
    template: str = "base",
    timeout: int = 43200,
) -> AsyncSandbox:
    """Return the existing E2B sandbox for *session_id* or create a new one.

    The sandbox_id is persisted in Redis so the same sandbox is reused
    across turns. Concurrent calls for the same session are serialised
    via a Redis ``SET NX`` creation lock.
    """
    redis = await get_redis_async()
    redis_key = f"{_SANDBOX_REDIS_PREFIX}{session_id}"

    # 1. Try reconnecting to an existing sandbox.
    raw = await redis.get(redis_key)
    if raw:
        sandbox_id = raw if isinstance(raw, str) else raw.decode()
        if sandbox_id != _CREATING:
            sandbox = await _try_reconnect(sandbox_id, api_key, redis_key, timeout)
            if sandbox:
                logger.info(
                    "[E2B] Reconnected to %.12s for session %.12s",
                    sandbox_id,
                    session_id,
                )
                return sandbox

    # 2. Claim creation lock. If another request holds it, wait for the result.
    claimed = await redis.set(redis_key, _CREATING, nx=True, ex=_CREATION_LOCK_TTL)
    if not claimed:
        for _ in range(_MAX_WAIT_ATTEMPTS):
            await asyncio.sleep(0.5)
            raw = await redis.get(redis_key)
            if not raw:
                break  # Lock expired — fall through to retry creation
            sandbox_id = raw if isinstance(raw, str) else raw.decode()
            if sandbox_id != _CREATING:
                sandbox = await _try_reconnect(sandbox_id, api_key, redis_key, timeout)
                if sandbox:
                    return sandbox
                break  # Stale sandbox cleared — fall through to create

        # Try to claim creation lock again after waiting.
        claimed = await redis.set(redis_key, _CREATING, nx=True, ex=_CREATION_LOCK_TTL)
        if not claimed:
            # Another process may have created a sandbox — try to use it.
            raw = await redis.get(redis_key)
            if raw:
                sandbox_id = raw if isinstance(raw, str) else raw.decode()
                if sandbox_id != _CREATING:
                    sandbox = await _try_reconnect(
                        sandbox_id, api_key, redis_key, timeout
                    )
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
            timeout=timeout,
            lifecycle={"on_timeout": "pause"},
        )
    except Exception:
        await redis.delete(redis_key)
        raise

    await redis.setex(redis_key, timeout, sandbox.sandbox_id)
    logger.info(
        "[E2B] Created sandbox %.12s for session %.12s",
        sandbox.sandbox_id,
        session_id,
    )
    return sandbox


async def pause_sandbox(session_id: str, api_key: str) -> bool:
    """Pause the E2B sandbox for *session_id* to stop billing between turns.

    Paused sandboxes cost nothing and are resumed automatically by
    ``get_or_create_sandbox()`` on the next turn (via ``AsyncSandbox.connect()``).

    Returns ``True`` if the sandbox was found and paused, ``False`` otherwise.
    Safe to call even when no sandbox exists for the session.
    """
    redis = await get_redis_async()
    redis_key = f"{_SANDBOX_REDIS_PREFIX}{session_id}"
    raw = await redis.get(redis_key)
    if not raw:
        return False

    sandbox_id = raw if isinstance(raw, str) else raw.decode()
    if sandbox_id == _CREATING:
        return False

    try:

        async def _connect_and_pause():
            sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
            await sandbox.pause()

        await asyncio.wait_for(_connect_and_pause(), timeout=10)
        logger.info(
            "[E2B] Paused sandbox %.12s for session %.12s",
            sandbox_id,
            session_id,
        )
        return True
    except Exception as exc:
        logger.warning(
            "[E2B] Failed to pause sandbox %.12s for session %.12s: %s",
            sandbox_id,
            session_id,
            exc,
        )
        return False


async def kill_sandbox(session_id: str, api_key: str) -> bool:
    """Kill the E2B sandbox for *session_id* and clean up its Redis entry.

    Returns ``True`` if a sandbox was found and killed, ``False`` otherwise.
    Safe to call even when no sandbox exists for the session.
    """
    redis = await get_redis_async()
    redis_key = f"{_SANDBOX_REDIS_PREFIX}{session_id}"
    raw = await redis.get(redis_key)
    if not raw:
        return False

    sandbox_id = raw if isinstance(raw, str) else raw.decode()
    await redis.delete(redis_key)

    if sandbox_id == _CREATING:
        return False

    try:

        async def _connect_and_kill():
            sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
            await sandbox.kill()

        await asyncio.wait_for(_connect_and_kill(), timeout=10)
        logger.info(
            "[E2B] Killed sandbox %.12s for session %.12s",
            sandbox_id,
            session_id,
        )
        return True
    except Exception as exc:
        logger.warning(
            "[E2B] Failed to kill sandbox %.12s for session %.12s: %s",
            sandbox_id,
            session_id,
            exc,
        )
        return False
