"""E2B sandbox lifecycle for CoPilot: persistent cloud execution + workspace sync.

Each session gets a long-lived E2B cloud sandbox.  ``bash_exec`` runs commands
directly on the sandbox via E2B's commands API.  The SDK file tools
(Read/Write/Edit/Glob/Grep) operate on a **local** workspace directory
(``sdk_cwd``), which is kept in sync with the sandbox's ``/home/user``
directory via E2B's HTTP-based filesystem API.

Turn lifecycle
--------------
1. **Turn start** – connect to the existing sandbox (sandbox_id in Redis) or
   create a new one.  Download any files in ``/home/user`` that are not in
   ``sdk_cwd`` (changed between turns).
2. **Execution** – ``bash_exec`` routes commands to ``sandbox.commands.run()``;
   SDK file tools operate on the local snapshot transparently.
3. **Turn end** – upload any files modified or created locally to the sandbox.
   The E2B sandbox stays alive; files persist across turns.
4. **Session expiry** – E2B sandbox is killed by its own timeout (session_ttl).

Sync design
-----------
The sync is *incremental*: only non-hidden files (those not starting with ``.``)
are tracked.  dot-files like ``.bashrc`` are left alone.  The sync is
best-effort — errors are logged but do not abort the session.

Infrastructure requirements
---------------------------
No special capabilities required beyond the standard container setup:
- ``E2B_API_KEY`` / ``CHAT_E2B_API_KEY`` environment variable or explicit config.
- Outbound HTTPS (port 443) from the container — already open for API calls.
"""

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from backend.data.redis_client import get_redis_async

if TYPE_CHECKING:
    from e2b import AsyncSandbox

logger = logging.getLogger(__name__)

# Redis key prefix for sandbox_id persistence
_SANDBOX_REDIS_PREFIX = "copilot:e2b:sandbox:"

# Directory inside the E2B sandbox that we sync with sdk_cwd
_E2B_WORKDIR = "/home/user"

# Maximum file size to sync (skip huge files to avoid long round-trips)
_MAX_SYNC_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

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
                        pass  # fall through to create

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


# ---------------------------------------------------------------------------
# Workspace sync: E2B ↔ local sdk_cwd
# ---------------------------------------------------------------------------


async def sync_from_sandbox(sandbox: "AsyncSandbox", local_dir: str) -> None:
    """Download files from the sandbox's /home/user into *local_dir*.

    Only non-hidden files are synced (files/dirs whose names start with ``.``
    are skipped).  Existing local files are overwritten so the local snapshot
    reflects the latest sandbox state.

    Errors for individual files are logged and skipped — a partial sync is
    better than no sync.
    """
    try:
        entries = await sandbox.files.list(_E2B_WORKDIR, depth=100)
    except Exception as exc:
        logger.error("[E2B] sync_from_sandbox: list failed: %s", exc)
        return

    local_dir_abs = os.path.normpath(os.path.abspath(local_dir))

    for entry in entries:
        rel = os.path.relpath(entry.path, _E2B_WORKDIR)
        # Skip hidden files / hidden ancestor directories
        if any(part.startswith(".") for part in rel.split(os.sep)):
            continue

        local_path = os.path.normpath(os.path.join(local_dir, rel))

        # Guard against path traversal: entry.path must stay within local_dir
        if not (
            local_path == local_dir_abs or local_path.startswith(local_dir_abs + os.sep)
        ):
            logger.warning(
                "[E2B] sync_from_sandbox: skipping path outside workspace: %s", rel
            )
            continue

        if getattr(entry.type, "value", str(entry.type)) == "dir":
            os.makedirs(local_path, exist_ok=True)
            continue

        # Skip large files
        if entry.size and entry.size > _MAX_SYNC_FILE_BYTES:
            logger.debug(
                "[E2B] sync_from_sandbox: skipping large file %s (%d bytes)",
                rel,
                entry.size,
            )
            continue

        try:
            content = await sandbox.files.read(entry.path, format="bytes")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(
                    content
                    if isinstance(content, (bytes, bytearray))
                    else content.encode()
                )
        except Exception as exc:
            logger.warning(
                "[E2B] sync_from_sandbox: failed to download %s: %s", rel, exc
            )

    logger.info("[E2B] sync_from_sandbox: done for %s", local_dir)


async def sync_to_sandbox(sandbox: "AsyncSandbox", local_dir: str) -> None:
    """Upload files from *local_dir* to the sandbox's /home/user.

    Only non-hidden files are synced.  Files in the sandbox that are not
    present locally are left alone (no deletion — other turns may have
    created them).

    Errors for individual files are logged and skipped.
    """
    if not os.path.isdir(local_dir):
        return

    for dirpath, dirnames, filenames in os.walk(local_dir):
        # Skip hidden directories in-place
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for fname in filenames:
            if fname.startswith("."):
                continue

            local_path = os.path.join(dirpath, fname)
            rel = os.path.relpath(local_path, local_dir)
            remote_path = f"{_E2B_WORKDIR}/{rel}"

            try:
                size = os.path.getsize(local_path)
            except OSError:
                continue

            if size > _MAX_SYNC_FILE_BYTES:
                logger.debug(
                    "[E2B] sync_to_sandbox: skipping large file %s (%d bytes)",
                    rel,
                    size,
                )
                continue

            try:
                with open(local_path, "rb") as f:
                    data = f.read()
                await sandbox.files.write(remote_path, data)
            except Exception as exc:
                logger.warning(
                    "[E2B] sync_to_sandbox: failed to upload %s: %s", rel, exc
                )

    logger.info("[E2B] sync_to_sandbox: done from %s", local_dir)
