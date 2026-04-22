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
Sandboxes are created with a configurable ``on_timeout`` lifecycle action
(default: ``"pause"``) and ``auto_resume`` (default: ``True``).  The explicit
per-turn ``pause_sandbox()`` call is the primary mechanism; the lifecycle
timeout is a safety net (default: 5 min).  ``auto_resume`` ensures that paused
sandboxes wake transparently on SDK activity, making the aggressive safety-net
timeout safe.  Paused sandboxes are free.

The sandbox_id is stored in Redis.  The same key doubles as a creation lock:
a ``"creating"`` sentinel value is written with a short TTL while a new sandbox
is being provisioned, preventing duplicate creation under concurrent requests.

Sandbox lifetime
----------------
E2B assigns each sandbox an absolute ``end_at`` timestamp at create time:
``end_at = now + timeout``.  Pausing does NOT extend ``end_at``; only
``connect()`` extends it (by ``timeout`` seconds from the moment of reconnect).
Active sessions therefore stay alive as long as turns arrive within the timeout
window.  Orphaned sandboxes (e.g. leaked by a failed create retry) are paused
(not killed) at ``end_at`` under the default ``on_timeout="pause"`` lifecycle;
they persist until explicitly killed or until E2B's platform-level cleanup
applies (30-day limit during beta).
"""

import asyncio
import contextlib
import logging
import math
from typing import Any, Awaitable, Callable, Literal

from e2b import AsyncSandbox, SandboxLifecycle

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_SANDBOX_KEY_PREFIX = "copilot:e2b:sandbox:"
_CREATING_SENTINEL = "creating"

# Per-attempt timeout for AsyncSandbox.create().  E2B normally provisions a
# sandbox in 5-15 s; 30 s gives generous headroom while ensuring a slow/hung
# E2B API call fails fast rather than blocking an executor goroutine for hours.
_SANDBOX_CREATE_TIMEOUT_SECONDS = 30

# Number of creation attempts before giving up.  Three attempts with 1 s / 2 s
# backoff means the worst-case wait is ~93 s (30+1+30+2+30) — far better than
# the indefinite hang that caused the original incident.
_SANDBOX_CREATE_MAX_RETRIES = 3

# Short TTL for the "creating" sentinel — if the process dies mid-creation the
# lock auto-expires so other callers are not blocked forever.
# Must be ≥ worst-case retry time: _SANDBOX_CREATE_MAX_RETRIES ×
# _SANDBOX_CREATE_TIMEOUT_SECONDS + inter-retry backoff ≈ 93 s → 120 s.
_CREATION_LOCK_TTL = 120  # seconds

# Wait interval for followers polling the "creating" sentinel.
_WAIT_INTERVAL_SECONDS = 0.5

# Derive follower budget from the lock TTL so it automatically tracks future
# TTL changes.  Add a 20% safety margin to handle slight clock drift / late
# sentinel expiry.  Result: ceil(120 / 0.5 * 1.2) = 288 iterations ≈ 144 s.
_MAX_WAIT_ATTEMPTS = math.ceil(_CREATION_LOCK_TTL / _WAIT_INTERVAL_SECONDS * 1.2)

# Timeout for E2B API calls (pause/kill) — short because these are control-plane
# operations; if the sandbox is unreachable, fail fast and retry on the next turn.
_E2B_API_TIMEOUT_SECONDS = 10

# Redis TTL for the sandbox key.  Must be ≥ the E2B project "paused sandbox
# lifetime" setting (recommended: set both to 48 h).
_SANDBOX_ID_TTL = 48 * 3600  # 48 hours

# --- Sandbox bootstrap --------------------------------------------------------
# The E2B "base" template does not ship the GitHub CLI, so we install it on
# first create.  The copilot prompt instructs the model to run ``gh auth
# status`` before prompting the user to connect GitHub; without this bootstrap
# that check flakily fails with "command not found".  The script is idempotent:
# it exits immediately if ``gh`` is already on PATH.  Installation is to
# ``/usr/local/bin`` via sudo (E2B base grants passwordless sudo to the default
# user), so every subsequent ``bash_exec`` sees ``gh`` on the hardcoded PATH.
#
# Version pinned so sandboxes across a deploy are consistent.  Bump as needed.
_GH_CLI_VERSION = "2.78.0"
_SANDBOX_BOOTSTRAP_SCRIPT = f"""
set -e
if command -v gh >/dev/null 2>&1; then exit 0; fi
case "$(uname -m)" in
  x86_64) arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "unsupported arch: $(uname -m)" >&2; exit 2 ;;
esac
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "https://github.com/cli/cli/releases/download/v{_GH_CLI_VERSION}/gh_{_GH_CLI_VERSION}_linux_${{arch}}.tar.gz" -o "$tmp/gh.tgz"
tar -xzf "$tmp/gh.tgz" -C "$tmp"
sudo -n mv "$tmp"/gh_*/bin/gh /usr/local/bin/gh
""".strip()

# Bootstrap should finish in <20 s on a good link; 60 s leaves headroom for
# slow CDN responses without blocking sandbox creation for users.
_BOOTSTRAP_TIMEOUT_SECONDS = 60


def _sandbox_key(session_id: str) -> str:
    return f"{_SANDBOX_KEY_PREFIX}{session_id}"


async def _get_stored_sandbox_id(session_id: str) -> str | None:
    redis = await get_redis_async()
    raw = await redis.get(_sandbox_key(session_id))
    value = raw.decode() if isinstance(raw, bytes) else raw
    return None if value == _CREATING_SENTINEL else value


async def _set_stored_sandbox_id(session_id: str, sandbox_id: str) -> None:
    redis = await get_redis_async()
    await redis.set(_sandbox_key(session_id), sandbox_id, ex=_SANDBOX_ID_TTL)


async def _clear_stored_sandbox_id(session_id: str) -> None:
    redis = await get_redis_async()
    await redis.delete(_sandbox_key(session_id))


async def _bootstrap_sandbox(sandbox: AsyncSandbox, session_id: str) -> None:
    """Install tools the copilot prompt expects on a fresh sandbox.

    Currently installs the GitHub CLI (``gh``) if missing — the E2B ``base``
    template does not include it, and the prompt instructs the model to run
    ``gh auth status`` to check GitHub connection state.

    Best-effort: on failure, log a warning and proceed.  The copilot's
    existing ``connect_integration`` fallback still works for GitHub auth; a
    missing ``gh`` just means the status check reports "not installed" and
    the model prompts the user to connect.
    """
    try:
        # AsyncSandbox.commands.run() already executes through a shell, so
        # the script can be passed straight through — no bash -c wrapper.
        result = await sandbox.commands.run(
            _SANDBOX_BOOTSTRAP_SCRIPT,
            timeout=_BOOTSTRAP_TIMEOUT_SECONDS,
        )
        if result.exit_code != 0:
            logger.warning(
                "[E2B] sandbox bootstrap exit %s for %.12s: %s",
                result.exit_code,
                session_id,
                (result.stderr or "").strip()[:200],
            )
    except Exception as exc:
        logger.warning(
            "[E2B] sandbox bootstrap failed for %.12s: %s",
            session_id,
            exc,
        )


async def _try_reconnect(
    sandbox_id: str, session_id: str, api_key: str
) -> "AsyncSandbox | None":
    """Try to reconnect to an existing sandbox. Returns None on failure."""
    try:
        sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
        if await sandbox.is_running():
            # Refresh TTL so an active session cannot lose its sandbox_id at expiry.
            await _set_stored_sandbox_id(session_id, sandbox_id)
            return sandbox
    except Exception as exc:
        logger.warning("[E2B] Reconnect to %.12s failed: %s", sandbox_id, exc)

    # Stale — clear the sandbox_id from Redis so a new one can be created.
    await _clear_stored_sandbox_id(session_id)
    return None


async def get_or_create_sandbox(
    session_id: str,
    api_key: str,
    timeout: int,
    template: str = "base",
    on_timeout: Literal["kill", "pause"] = "pause",
) -> AsyncSandbox:
    """Return the existing E2B sandbox for *session_id* or create a new one.

    The sandbox key in Redis serves a dual purpose: it stores the sandbox_id
    and acts as a creation lock via a ``"creating"`` sentinel value.  This
    removes the need for a separate lock key.

    *timeout* controls how long the e2b sandbox may run continuously before
    the ``on_timeout`` lifecycle rule fires (default: 5 min).
    *on_timeout* controls what happens on timeout: ``"pause"`` (default, free)
    or ``"kill"``.  When ``"pause"``, ``auto_resume`` is enabled so paused
    sandboxes wake transparently on SDK activity.
    """
    redis = await get_redis_async()
    key = _sandbox_key(session_id)

    for _ in range(_MAX_WAIT_ATTEMPTS):
        raw = await redis.get(key)
        value = raw.decode() if isinstance(raw, bytes) else raw

        if value and value != _CREATING_SENTINEL:
            # Existing sandbox ID — try to reconnect (auto-resumes if paused).
            sandbox = await _try_reconnect(value, session_id, api_key)
            if sandbox:
                logger.info(
                    "[E2B] Reconnected to %.12s for session %.12s",
                    value,
                    session_id,
                )
                return sandbox
            # _try_reconnect cleared the key — loop to create a new sandbox.
            continue

        if value == _CREATING_SENTINEL:
            # Another coroutine is creating — wait for it to finish.
            await asyncio.sleep(_WAIT_INTERVAL_SECONDS)
            continue

        # No sandbox and no active creation — atomically claim the creation slot.
        claimed = await redis.set(
            key, _CREATING_SENTINEL, nx=True, ex=_CREATION_LOCK_TTL
        )
        if not claimed:
            # Race lost — another coroutine just claimed it.
            await asyncio.sleep(0.1)
            continue

        # We hold the slot — create the sandbox with per-attempt timeout and
        # retry.  The sentinel remains held throughout so concurrent callers
        # for the same session wait rather than racing to create duplicates.
        sandbox: AsyncSandbox | None = None
        try:
            lifecycle = SandboxLifecycle(
                on_timeout=on_timeout,
                auto_resume=on_timeout == "pause",
            )
            # Note: asyncio.wait_for() only cancels the client-side wait;
            # E2B may complete provisioning server-side after a timeout.
            # Since AsyncSandbox.create() returns no sandbox_id before
            # completion, recovery via connect() is not possible and each
            # timed-out attempt may leak a sandbox.  Under the default
            # on_timeout="pause" lifecycle, leaked orphans are paused (not
            # killed) at end_at and persist until explicitly cleaned up.
            # At most _SANDBOX_CREATE_MAX_RETRIES − 1 = 2 sandboxes can
            # leak per incident.
            last_exc: Exception | None = None
            for attempt in range(1, _SANDBOX_CREATE_MAX_RETRIES + 1):
                try:
                    sandbox = await asyncio.wait_for(
                        AsyncSandbox.create(
                            template=template,
                            api_key=api_key,
                            timeout=timeout,
                            lifecycle=lifecycle,
                        ),
                        timeout=_SANDBOX_CREATE_TIMEOUT_SECONDS,
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "[E2B] Sandbox creation attempt %d/%d failed for session %.12s: %s",
                        attempt,
                        _SANDBOX_CREATE_MAX_RETRIES,
                        session_id,
                        exc,
                    )
                    if attempt < _SANDBOX_CREATE_MAX_RETRIES:
                        await asyncio.sleep(2 ** (attempt - 1))  # 1 s, 2 s

            if last_exc is not None:
                raise last_exc

            assert sandbox is not None  # guaranteed: last_exc is None iff break was hit
            # Install gh CLI etc. on first create so the prompt's `gh auth
            # status` check works reliably.  Filesystem persists across
            # pause/resume, so this runs at most once per session.
            await _bootstrap_sandbox(sandbox, session_id)
            try:
                await _set_stored_sandbox_id(session_id, sandbox.sandbox_id)
            except Exception:
                # Redis save failed — kill the sandbox to avoid leaking it.
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        sandbox.kill(), timeout=_E2B_API_TIMEOUT_SECONDS
                    )
                raise
        except asyncio.CancelledError:
            # Task cancelled during creation — release the slot so followers
            # are not blocked for the full TTL (120 s).  CancelledError inherits
            # from BaseException, not Exception, so it is not caught above.
            # Kill the sandbox if it was already created to avoid leaking it
            # (can happen when cancellation fires during _set_stored_sandbox_id).
            # Suppress BaseException (including a second CancelledError) so a
            # re-entrant cancellation during cleanup cannot skip the redis.delete.
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await redis.delete(key)
            if sandbox is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(
                        sandbox.kill(), timeout=_E2B_API_TIMEOUT_SECONDS
                    )
            raise
        except Exception:
            # Release the creation slot so other callers can proceed.
            await redis.delete(key)
            raise

        logger.info(
            "[E2B] Created sandbox %.12s for session %.12s",
            sandbox.sandbox_id,
            session_id,
        )
        return sandbox

    raise RuntimeError(f"Could not acquire E2B sandbox for session {session_id[:12]}")


async def _act_on_sandbox(
    session_id: str,
    api_key: str,
    action: str,
    fn: Callable[[AsyncSandbox], Awaitable[Any]],
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

    async def _run() -> None:
        await fn(await AsyncSandbox.connect(sandbox_id, api_key=api_key))

    try:
        await asyncio.wait_for(_run(), timeout=_E2B_API_TIMEOUT_SECONDS)
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

    Prefer ``pause_sandbox_direct()`` when the sandbox object is already in
    scope — it skips the Redis lookup and reconnect round-trip.

    Returns ``True`` if the sandbox was found and paused, ``False`` otherwise.
    Safe to call even when no sandbox exists for the session.
    """
    return await _act_on_sandbox(session_id, api_key, "pause", lambda sb: sb.pause())


async def pause_sandbox_direct(sandbox: "AsyncSandbox", session_id: str) -> bool:
    """Pause an already-connected sandbox without a reconnect round-trip.

    Use this in callers that already hold the live sandbox object (e.g. turn
    teardown in ``service.py``).  Saves the Redis lookup and
    ``AsyncSandbox.connect()`` call that ``pause_sandbox()`` would make.

    Returns ``True`` on success, ``False`` on failure or timeout.
    """
    try:
        await asyncio.wait_for(sandbox.pause(), timeout=_E2B_API_TIMEOUT_SECONDS)
        logger.info(
            "[E2B] Paused sandbox %.12s for session %.12s",
            sandbox.sandbox_id,
            session_id,
        )
        return True
    except Exception as exc:
        logger.warning(
            "[E2B] Failed to pause sandbox %.12s for session %.12s: %s",
            sandbox.sandbox_id,
            session_id,
            exc,
        )
        return False


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
