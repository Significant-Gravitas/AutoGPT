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
2. **Execution** – ``bash_exec`` and MCP file tools operate directly on the
   sandbox's ``/home/user`` filesystem.
3. **Session expiry** – E2B sandbox is killed by its own timeout (session_ttl).

Infrastructure requirements
---------------------------
- ``E2B_API_KEY`` / ``CHAT_E2B_API_KEY`` environment variable or explicit config.
- Outbound HTTPS (port 443) from the container — already open for API calls.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from backend.data.redis_client import get_redis_async

if TYPE_CHECKING:
    from e2b import AsyncSandbox

logger = logging.getLogger(__name__)

# Redis key prefix for sandbox_id persistence
_SANDBOX_REDIS_PREFIX = "copilot:e2b:sandbox:"

# Working directory inside E2B sandboxes
_E2B_WORKDIR = "/home/user"

# Placeholder written to Redis while a sandbox is being created.
# Prevents concurrent requests from spawning duplicate sandboxes.
_SANDBOX_CREATING = "__creating__"

# How long (seconds) the creation placeholder lives in Redis.
# Must comfortably exceed the time it takes AsyncSandbox.create() to return.
_CREATION_LOCK_TTL = 60


# ---------------------------------------------------------------------------
# Sandbox lifecycle
# ---------------------------------------------------------------------------


async def get_or_create_sandbox(
    session_id: str,
    api_key: str,
    template: str = "base",
    timeout: int = 43200,
) -> "AsyncSandbox":
    """Return the existing E2B sandbox for *session_id* or create a new one.

    The sandbox_id is persisted in Redis so the same sandbox is reused
    across turns even when requests land on different pods.

    Concurrent calls for the same *session_id* are safe: a Redis ``SET NX``
    creation lock ensures only one caller creates the sandbox; the others
    wait and then reconnect to the sandbox that was created.

    Args:
        session_id: CoPilot session ID (used as Redis key suffix).
        api_key: E2B API key.
        template: E2B sandbox template name.
        timeout: Sandbox keepalive timeout in seconds.

    Returns:
        A connected :class:`e2b.AsyncSandbox` instance.
    """
    from e2b import AsyncSandbox

    redis = await get_redis_async()
    redis_key = f"{_SANDBOX_REDIS_PREFIX}{session_id}"

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Try to reconnect to an existing, live sandbox
    # ──────────────────────────────────────────────────────────────────────
    raw = await redis.get(redis_key)
    if raw:
        sandbox_id = raw if isinstance(raw, str) else raw.decode()
        if sandbox_id != _SANDBOX_CREATING:
            try:
                sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
                if await sandbox.is_running():
                    await redis.expire(redis_key, timeout)
                    logger.info(
                        "[E2B] Reconnected to sandbox %.12s for session %.12s",
                        sandbox_id,
                        session_id,
                    )
                    return sandbox
                logger.info(
                    "[E2B] Sandbox %.12s expired, creating new for session %.12s",
                    sandbox_id,
                    session_id,
                )
            except Exception as exc:
                logger.warning(
                    "[E2B] Reconnect to sandbox %.12s failed: %s", sandbox_id, exc
                )

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Atomically claim sandbox creation to prevent duplicate spawns
    # ──────────────────────────────────────────────────────────────────────
    claimed = await redis.set(
        redis_key, _SANDBOX_CREATING, nx=True, ex=_CREATION_LOCK_TTL
    )
    if not claimed:
        # Another coroutine holds the creation lock.  Poll until it resolves.
        poll_intervals = int(_CREATION_LOCK_TTL / 0.5)
        for _ in range(poll_intervals):
            await asyncio.sleep(0.5)
            raw = await redis.get(redis_key)
            if not raw:
                # Placeholder expired before the sandbox was registered.
                break
            sandbox_id = raw if isinstance(raw, str) else raw.decode()
            if sandbox_id != _SANDBOX_CREATING:
                # The other coroutine finished; connect to their sandbox.
                try:
                    sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
                    if await sandbox.is_running():
                        await redis.expire(redis_key, timeout)
                        logger.info(
                            "[E2B] Joined concurrently created sandbox %.12s "
                            "for session %.12s",
                            sandbox_id,
                            session_id,
                        )
                        return sandbox
                except Exception as exc:
                    logger.warning(
                        "[E2B] Join concurrently created sandbox %.12s failed: %s",
                        sandbox_id,
                        exc,
                    )
                break  # fall through to create our own
        # Re-claim the creation lock atomically to avoid overwriting a valid
        # sandbox ID that another request may have just registered.
        reclaimed = await redis.set(
            redis_key, _SANDBOX_CREATING, nx=True, ex=_CREATION_LOCK_TTL
        )
        if not reclaimed:
            # Another creator won the race — re-read and try to connect.
            raw = await redis.get(redis_key)
            if raw:
                sid = raw if isinstance(raw, str) else raw.decode()
                if sid != _SANDBOX_CREATING:
                    try:
                        sandbox = await AsyncSandbox.connect(sid, api_key=api_key)
                        if await sandbox.is_running():
                            await redis.expire(redis_key, timeout)
                            return sandbox
                    except Exception:
                        pass
            # Cannot acquire lock and cannot connect — another request owns
            # the creation slot.  Raise rather than creating a duplicate
            # sandbox without a lock.
            raise RuntimeError(
                f"Could not acquire E2B sandbox creation lock for session "
                f"{session_id[:12]}; another request is creating the sandbox."
            )

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Create a new sandbox and register it
    # ──────────────────────────────────────────────────────────────────────
    try:
        sandbox = await AsyncSandbox.create(
            template=template,
            api_key=api_key,
            timeout=timeout,
        )
    except Exception:
        # Release the creation lock so other waiters can retry.
        await redis.delete(redis_key)
        raise

    await redis.setex(redis_key, timeout, sandbox.sandbox_id)
    logger.info(
        "[E2B] Created new sandbox %.12s for session %.12s",
        sandbox.sandbox_id,
        session_id,
    )
    return sandbox
