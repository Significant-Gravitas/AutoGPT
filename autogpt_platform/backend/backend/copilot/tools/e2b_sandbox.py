"""E2B sandbox lifecycle for CoPilot: persistent cloud execution.

Each session gets a long-lived E2B cloud sandbox.  ``bash_exec`` runs commands
directly on the sandbox via ``sandbox.commands.run()``.  SDK file tools
(read_file/write_file/edit_file/glob/grep) route to the sandbox's
``/home/user`` directory via E2B's HTTP-based filesystem API — all tools
share a single coherent filesystem with no local sync required.

Lifecycle
---------
1. **Turn start** – connect to the existing sandbox (sandbox_id in
   ``ChatSession.metadata``) or create a new one via ``get_or_create_sandbox()``.
   ``connect()`` in e2b v2 auto-resumes paused sandboxes.
2. **Execution** – ``bash_exec`` and MCP file tools operate directly on the
   sandbox's ``/home/user`` filesystem.
3. **Turn end** – the sandbox is paused via ``pause_sandbox()`` so idle time
   between turns costs nothing.  Paused sandboxes have no compute cost.
4. **Session delete** – ``kill_sandbox()`` fully terminates the sandbox.

Cost control
------------
Sandboxes are created with ``lifecycle={"on_timeout": "pause"}`` so they
auto-pause (rather than terminate) if they somehow remain running past the
timeout.  The explicit per-turn ``pause_sandbox()`` call is the primary
mechanism; the lifecycle setting is a safety net.

The sandbox_id is stored in ``ChatSession.metadata`` (DB) via
``ChatSessionMetadata.e2b_sandbox_id``.  Redis is only used for the
short-lived creation lock that prevents two concurrent requests from creating
two sandboxes for the same session.
"""

import asyncio
import contextlib
import logging
from typing import Callable, Coroutine

from e2b import AsyncSandbox

from backend.copilot.db import get_session_metadata, update_session_metadata
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_CREATION_LOCK_PREFIX = "copilot:e2b:creating:"
E2B_WORKDIR = "/home/user"
_CREATION_LOCK_TTL = 60
_MAX_WAIT_ATTEMPTS = 20  # 20 * 0.5s = 10s max wait

# How long the sandbox may run continuously before e2b auto-pauses it (safety
# net; per-turn explicit pause is the primary mechanism).
_E2B_PAUSE_TIMEOUT = 14400  # 4 hours in seconds

# Sessions not updated within this window are considered abandoned; the hourly
# cleanup job will kill their paused sandboxes.  Paused sandboxes are free,
# so a generous window avoids disrupting long-lived but occasionally active
# sessions.
_E2B_KILL_TIMEOUT = 7 * 24 * 3600  # 7 days in seconds


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

    # Stale — clear the sandbox_id from metadata so a new one can be created.
    meta = await get_session_metadata(session_id)
    await update_session_metadata(
        session_id, meta.model_copy(update={"e2b_sandbox_id": None})
    )
    return None


async def get_or_create_sandbox(
    session_id: str,
    api_key: str,
    template: str = "base",
    pause_timeout: int = _E2B_PAUSE_TIMEOUT,
    e2b_sandbox_id: str | None = None,
) -> AsyncSandbox:
    """Return the existing E2B sandbox for *session_id* or create a new one.

    The sandbox_id is persisted in ``ChatSession.metadata`` (DB) so the same
    sandbox is reused across turns and service restarts.  Concurrent calls
    for the same session are serialised via a Redis ``SET NX`` creation lock.

    *e2b_sandbox_id* may be supplied by the caller when the session metadata
    is already loaded (e.g. from the session object), avoiding a redundant DB
    round-trip.  When omitted the function fetches it from the DB.

    *pause_timeout* controls how long the e2b sandbox may run continuously
    before the ``on_timeout: pause`` lifecycle rule fires (default: 4 h).
    """
    redis = await get_redis_async()
    lock_key = f"{_CREATION_LOCK_PREFIX}{session_id}"

    # 1. Try reconnecting to an existing sandbox.
    # Use the caller-supplied sandbox_id when available to avoid a DB round-trip.
    if e2b_sandbox_id is None:
        meta = await get_session_metadata(session_id)
        e2b_sandbox_id = meta.e2b_sandbox_id
    if e2b_sandbox_id:
        sandbox = await _try_reconnect(e2b_sandbox_id, session_id, api_key)
        if sandbox:
            logger.info(
                "[E2B] Reconnected to %.12s for session %.12s",
                e2b_sandbox_id,
                session_id,
            )
            return sandbox

    # 2. Claim creation lock. If another request holds it, wait for the result.
    claimed = await redis.set(lock_key, "1", nx=True, ex=_CREATION_LOCK_TTL)
    if not claimed:
        for _ in range(_MAX_WAIT_ATTEMPTS):
            await asyncio.sleep(0.5)
            meta = await get_session_metadata(session_id)
            if meta.e2b_sandbox_id:
                sandbox = await _try_reconnect(meta.e2b_sandbox_id, session_id, api_key)
                if sandbox:
                    return sandbox
                break  # Stale sandbox cleared — fall through to create

        # Try to claim creation lock again after waiting.
        claimed = await redis.set(lock_key, "1", nx=True, ex=_CREATION_LOCK_TTL)
        if not claimed:
            # Another process may have created a sandbox — try to use it.
            meta = await get_session_metadata(session_id)
            if meta.e2b_sandbox_id:
                sandbox = await _try_reconnect(meta.e2b_sandbox_id, session_id, api_key)
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
            meta = await get_session_metadata(session_id)
            await update_session_metadata(
                session_id,
                meta.model_copy(update={"e2b_sandbox_id": sandbox.sandbox_id}),
            )
        except Exception:
            # Metadata save failed — kill the sandbox to avoid leaking it.
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
    clear_metadata: bool = False,
    e2b_sandbox_id: str | None = None,
) -> bool:
    """Connect to the sandbox for *session_id* and run *fn* on it.

    Shared by ``pause_sandbox`` and ``kill_sandbox``.  Returns ``True`` on
    success, ``False`` when no sandbox is found or the action fails.
    If *clear_metadata* is ``True``, the sandbox_id is removed from metadata
    only after the action succeeds so a failed kill can be retried.

    *e2b_sandbox_id* may be supplied by the caller when the metadata is
    already available, avoiding a redundant DB round-trip.
    """
    if e2b_sandbox_id is None:
        meta = await get_session_metadata(session_id)
        e2b_sandbox_id = meta.e2b_sandbox_id
    else:
        meta = None  # Will be loaded lazily only if clear_metadata is True
    if not e2b_sandbox_id:
        return False

    sandbox_id = e2b_sandbox_id

    async def _connect_and_act():
        sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
        await fn(sandbox)

    try:
        await asyncio.wait_for(_connect_and_act(), timeout=10)
        if clear_metadata:
            if meta is None:
                meta = await get_session_metadata(session_id)
            await update_session_metadata(
                session_id, meta.model_copy(update={"e2b_sandbox_id": None})
            )
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

    Returns ``True`` if the sandbox was found and paused, ``False`` otherwise.
    Safe to call even when no sandbox exists for the session.
    """
    return await _act_on_sandbox(session_id, api_key, "pause", lambda sb: sb.pause())


async def kill_sandbox(
    session_id: str,
    api_key: str,
    e2b_sandbox_id: str | None = None,
    clear_metadata: bool = True,
) -> bool:
    """Kill the E2B sandbox for *session_id* and clear its metadata entry.

    Returns ``True`` if a sandbox was found and killed, ``False`` otherwise.
    Safe to call even when no sandbox exists for the session.

    *e2b_sandbox_id* may be supplied by the caller when the metadata is
    already available (e.g. fetched before the session record was deleted),
    avoiding a redundant — and potentially failing — DB round-trip.

    *clear_metadata*: set to ``False`` when the session record has already been
    deleted (e.g. called from the delete-session route) to avoid a spurious
    DB update attempt on a non-existent row.
    """
    return await _act_on_sandbox(
        session_id,
        api_key,
        "kill",
        lambda sb: sb.kill(),
        clear_metadata=clear_metadata,
        e2b_sandbox_id=e2b_sandbox_id,
    )
