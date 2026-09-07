"""E2B sandbox lifecycle for CoPilot: persistent cloud execution.

Each session gets a long-lived E2B cloud sandbox.  ``bash_exec`` runs commands
directly on the sandbox via ``sandbox.commands.run()``.  SDK file tools
(read_file/write_file/edit_file/glob/grep) route to the sandbox's
``/home/user`` directory via E2B's HTTP-based filesystem API — all tools
share a single coherent filesystem with no local sync required.

Ownership
---------
A sandbox belongs to a :class:`SandboxOwner`:

* ``session`` — scratch for one chat; killed when the chat is deleted.
* ``expert`` — the hired expert's own computer.  Every session that runs as
  that expert (chats, ``delegate_to_expert`` sub-sessions, scheduled
  kickoffs) reconnects to the same box, so tools it installs, logins it keeps
  in its browser and files it writes are still there next time.  Deleting a
  chat never touches it; archiving the expert does
  (``kill_expert_sandboxes``).  The box mounts the expert's own durable volume
  at ``~/workspace`` and the owning user's volume at ``~/shared`` (see
  ``workspace_volume_mounts``).

Lifecycle
---------
1. **Turn start** – connect to the existing sandbox (sandbox_id in Redis) or
   create a new one via ``get_or_create_sandbox()``.
   ``connect()`` in e2b v2 auto-resumes paused sandboxes.
2. **Execution** – ``bash_exec`` and MCP file tools operate directly on the
   sandbox's ``/home/user`` filesystem.
3. **Turn end** – the sandbox is paused via ``pause_sandbox()`` (fire-and-forget)
   so idle time between turns costs nothing.  Paused sandboxes have no compute
   cost.  An expert's box is only paused once its *last* concurrent turn
   ends — two sessions of the same expert may be mid-command on it at once,
   and pausing under one of them would sever its command stream.
4. **Session delete** – ``kill_sandbox()`` fully terminates a session sandbox.

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
(not killed) at ``end_at`` under the default ``on_timeout="pause"`` lifecycle.

Paused sandboxes have no time-to-live on E2B's side — no expiry, no storage
billing, no concurrency cost — which is what makes an expert's box durable.
Redis only *caches* its id: every sandbox is stamped with ``autogpt_owner`` /
``autogpt_kind`` metadata, and ``find_owned_sandbox_id`` recovers an expert's
box through the E2B API if the cache is ever lost.
"""

import asyncio
import contextlib
import logging
import math
from typing import Any, Awaitable, Callable, Literal, Mapping

from e2b import (
    AsyncSandbox,
    AsyncVolume,
    SandboxInfo,
    SandboxLifecycle,
    SandboxQuery,
    SandboxState,
)
from pydantic import BaseModel, ConfigDict

from backend.blocks.desktop._api import resolve_volume
from backend.data.redis_client import get_redis_async
from backend.util.e2b_template import ensure_template
from backend.util.sandbox_metadata import MountState, SandboxMetadata

logger = logging.getLogger(__name__)

_SANDBOX_KEY_PREFIX = "copilot:e2b:sandbox:"
_DESKTOP_KEY_PREFIX = "copilot:e2b:desktop:"
_EXPERT_KEY_PREFIX = "copilot:e2b:expert:"
_CREATING_SENTINEL = "creating"

# E2B sandbox metadata that lets an owner find its box without Redis.
METADATA_OWNER = "autogpt_owner"
METADATA_KIND = "autogpt_kind"
# "attached" when the workspace volumes were mounted, "none" when creation had
# to fall back to a volume-less box — visible in the E2B dashboard and API.
METADATA_MOUNTS = "autogpt_mounts"

SandboxKind = Literal["shell", "desktop"]

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

# Timeout for E2B API calls (pause/kill/list) — short because these are
# control-plane operations; if the sandbox is unreachable, fail fast and retry
# on the next turn.
_E2B_API_TIMEOUT_SECONDS = 10

# Redis TTL for a session sandbox key.  Must be ≥ the E2B project "paused
# sandbox lifetime" setting (recommended: set both to 48 h).
_SANDBOX_ID_TTL = 48 * 3600  # 48 hours

# An expert's box is meant to outlive any one chat.  The key is refreshed on
# every use and E2B metadata recovers it after expiry, so this is only a cache
# TTL — not a lifetime.
_EXPERT_ID_TTL = 30 * 24 * 3600

# Leak guard for the per-expert active-turn counter: a turn that dies without
# releasing its slot stops blocking the pause after this long.  Past it the
# lifecycle timeout has long since paused the box anyway.
_ACTIVE_TURN_TTL = 3600


class SandboxOwner(BaseModel):
    """Who a CoPilot sandbox belongs to — and therefore how long it lives.

    ``session`` sandboxes are scratch for one chat.  ``expert`` sandboxes are
    a hired expert's own persistent computer, shared by every session that
    runs as that expert — one box per expert, not one per account, so two
    experts never see each other's logins or files.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["session", "expert"]
    id: str

    @classmethod
    def for_session(
        cls, session_id: str, expert_id: str | None = None
    ) -> "SandboxOwner":
        """Expert sessions run on the expert's box; everything else per-session."""
        if expert_id:
            return cls(kind="expert", id=expert_id)
        return cls(kind="session", id=session_id)

    @property
    def is_expert(self) -> bool:
        return self.kind == "expert"

    def key(self, sandbox_kind: SandboxKind = "shell") -> str:
        """Redis key caching this owner's sandbox id (doubles as creation lock)."""
        if self.is_expert:
            return f"{_EXPERT_KEY_PREFIX}{self.id}:{sandbox_kind}"
        prefix = _SANDBOX_KEY_PREFIX if sandbox_kind == "shell" else _DESKTOP_KEY_PREFIX
        return f"{prefix}{self.id}"

    @property
    def ttl(self) -> int:
        return _EXPERT_ID_TTL if self.is_expert else _SANDBOX_ID_TTL

    def metadata(self, sandbox_kind: SandboxKind = "shell") -> dict[str, str]:
        """The identity keys a lookup filters on; a subset of ``creation_metadata``."""
        return {METADATA_OWNER: f"{self.kind}:{self.id}", METADATA_KIND: sandbox_kind}

    def creation_metadata(
        self,
        sandbox_kind: SandboxKind = "shell",
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        template: str | None = None,
        mounts: MountState | None = None,
    ) -> dict[str, str]:
        """Identity plus provenance, stamped on the box when it is created."""
        return SandboxMetadata.for_copilot(
            f"{self.kind}:{self.id}",
            sandbox_kind,
            user_id=user_id,
            session_id=session_id,
            expert_id=self.id if self.is_expert else None,
            template=template,
            mounts=mounts,
        ).as_e2b()

    def __str__(self) -> str:
        return f"{self.kind} {self.id[:12]}"


def _as_owner(owner: "SandboxOwner | str") -> SandboxOwner:
    """Accept a bare session id where an owner is expected."""
    if isinstance(owner, SandboxOwner):
        return owner
    return SandboxOwner(kind="session", id=owner)


def _sandbox_key(session_id: str) -> str:
    return SandboxOwner(kind="session", id=session_id).key()


async def _get_stored_sandbox_id(
    owner: SandboxOwner, sandbox_kind: SandboxKind = "shell"
) -> str | None:
    redis = await get_redis_async()
    raw = await redis.get(owner.key(sandbox_kind))
    value = raw.decode() if isinstance(raw, bytes) else raw
    return None if value == _CREATING_SENTINEL else value


async def _set_stored_sandbox_id(
    owner: SandboxOwner, sandbox_id: str, sandbox_kind: SandboxKind = "shell"
) -> None:
    redis = await get_redis_async()
    await redis.set(owner.key(sandbox_kind), sandbox_id, ex=owner.ttl)


async def _clear_stored_sandbox_id(
    owner: SandboxOwner, sandbox_kind: SandboxKind = "shell"
) -> None:
    redis = await get_redis_async()
    await redis.delete(owner.key(sandbox_kind))


async def list_owned_sandboxes(
    owner: SandboxOwner, sandbox_kind: SandboxKind, api_key: str
) -> list[SandboxInfo]:
    """The owner's boxes of one kind as E2B lists them, newest first.

    A running box sorts before a paused one.  Listing never connects, so a
    paused box stays paused — connecting is what auto-resume reacts to.
    Returns ``[]`` on any API failure so callers degrade to "nothing found".
    """
    try:
        paginator = AsyncSandbox.list(
            query=SandboxQuery(
                metadata=owner.metadata(sandbox_kind),
                state=[SandboxState.RUNNING, SandboxState.PAUSED],
            ),
            limit=10,
            api_key=api_key,
        )
        infos = await asyncio.wait_for(
            paginator.next_items(), timeout=_E2B_API_TIMEOUT_SECONDS
        )
    except Exception as exc:
        logger.warning(
            "[E2B] Metadata lookup for %s %s failed: %s", owner, sandbox_kind, exc
        )
        return []
    infos = sorted(infos, key=lambda info: info.started_at, reverse=True)
    running = [info for info in infos if info.state == SandboxState.RUNNING]
    paused = [info for info in infos if info.state != SandboxState.RUNNING]
    return running + paused


async def find_owned_sandbox_id(
    owner: SandboxOwner, sandbox_kind: SandboxKind, api_key: str
) -> str | None:
    """Recover an expert's box through the E2B API when Redis has forgotten it.

    Session sandboxes are Redis-only: losing that key just means a fresh
    scratch sandbox, which is not worth a control-plane round-trip.  For an
    expert the box *is* the state, so we query E2B by the owner metadata every
    sandbox is stamped with, preferring a running box over a paused one and
    the newest of several (a lost creation race can leave duplicates).
    """
    if not owner.is_expert:
        return None
    infos = await list_owned_sandboxes(owner, sandbox_kind, api_key)
    if not infos:
        return None
    chosen = infos[0]
    if len(infos) > 1:
        logger.warning(
            "[E2B] %s has %d %s sandboxes; using %.12s",
            owner,
            len(infos),
            sandbox_kind,
            chosen.sandbox_id,
        )
    return chosen.sandbox_id


async def _try_reconnect(
    sandbox_id: str, owner: "SandboxOwner | str", api_key: str
) -> "AsyncSandbox | None":
    """Try to reconnect to an existing sandbox. Returns None on failure."""
    owner = _as_owner(owner)
    try:
        sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
        if await sandbox.is_running():
            # Refresh TTL so an active owner cannot lose its sandbox_id at expiry.
            await _set_stored_sandbox_id(owner, sandbox_id)
            return sandbox
    except Exception as exc:
        logger.warning("[E2B] Reconnect to %.12s failed: %s", sandbox_id, exc)

    # Stale — clear the sandbox_id from Redis so a new one can be created.
    await _clear_stored_sandbox_id(owner)
    return None


async def _resolve_volume_mounts(
    volume_mounts: Mapping[str, str] | None, api_key: str
) -> dict[str, "AsyncVolume | str"] | None:
    """Build the ``volume_mounts`` mapping (path -> volume) for named volumes.

    Creates each volume if it does not exist yet, otherwise mounts it by name
    (``resolve_volume`` bounds each call).  Volumes resolve concurrently so the
    creation lock is held for one round-trip, not one per mount.  Returns
    ``None`` when no volumes are requested; never raises.
    """
    if not volume_mounts:
        return None
    paths = list(volume_mounts)
    volumes = await asyncio.gather(
        *(resolve_volume(volume_mounts[path], api_key) for path in paths)
    )
    return dict(zip(paths, volumes))


# ---------------------------------------------------------------------------
# Expert boxes: concurrent-turn accounting
# ---------------------------------------------------------------------------


def _active_turns_key(owner: SandboxOwner) -> str:
    return f"{owner.key('shell')}:active"


async def _acquire_turn(owner: SandboxOwner) -> None:
    """Count this turn on an expert's box so another turn's end can't pause it."""
    if not owner.is_expert:
        return
    try:
        redis = await get_redis_async()
        key = _active_turns_key(owner)
        await redis.incr(key)
        await redis.expire(key, _ACTIVE_TURN_TTL)
    except Exception as exc:
        logger.warning("[E2B] Could not record active turn for %s: %s", owner, exc)


async def _release_turn(owner: SandboxOwner) -> bool:
    """Return ``True`` when the box may be paused: no other turn is still on it."""
    if not owner.is_expert:
        return True
    try:
        redis = await get_redis_async()
        key = _active_turns_key(owner)
        remaining = await redis.decr(key)
        if remaining <= 0:
            await redis.delete(key)
            return True
        logger.info(
            "[E2B] %s still has %d active turn(s); leaving its box running",
            owner,
            remaining,
        )
        return False
    except Exception as exc:
        # Fail closed: without the counter we cannot know whether another turn
        # is on the box, and pausing under one severs its command stream. The
        # lifecycle timeout still pauses the box once it goes idle.
        logger.warning(
            "[E2B] Could not release active turn for %s (%s); leaving its box running",
            owner,
            exc,
        )
        return False


async def get_or_create_sandbox(
    session_id: str,
    api_key: str,
    timeout: int,
    template: str = "base",
    on_timeout: Literal["kill", "pause"] = "pause",
    volume_mounts: Mapping[str, str] | None = None,
    *,
    expert_id: str | None = None,
    user_id: str | None = None,
) -> AsyncSandbox:
    """Return the existing E2B sandbox for this turn's owner or create a new one.

    The owner is the session, or — when *expert_id* is set — the expert, whose
    box every one of its sessions shares.  The owner's key in Redis serves a
    dual purpose: it stores the sandbox_id and acts as a creation lock via a
    ``"creating"`` sentinel value.  This removes the need for a separate lock
    key.

    *timeout* controls how long the e2b sandbox may run continuously before
    the ``on_timeout`` lifecycle rule fires (default: 5 min).
    *on_timeout* controls what happens on timeout: ``"pause"`` (default, free)
    or ``"kill"``.  When ``"pause"``, ``auto_resume`` is enabled so paused
    sandboxes wake transparently on SDK activity.
    *volume_mounts* maps mount paths to durable volume names (see
    ``workspace_volume_mounts``) so the agent shell shares a persistent
    ``~/workspace`` with the on-demand desktop and, for an expert, the owning
    user's ``~/shared``.  A mount failure degrades to a volume-less sandbox
    rather than failing the session.
    """
    owner = SandboxOwner.for_session(session_id, expert_id)
    redis = await get_redis_async()
    key = owner.key("shell")
    # Boxes E2B still lists but we could not reconnect to (mid-teardown, an
    # unresumable snapshot). Without this an expert owner would re-find the
    # same id on every iteration and never fall through to creating a fresh one.
    failed_ids: set[str] = set()

    for _ in range(_MAX_WAIT_ATTEMPTS):
        raw = await redis.get(key)
        value = raw.decode() if isinstance(raw, bytes) else raw

        if not value and owner.is_expert:
            # Redis is only a cache for an expert's box; E2B is the record.
            value = await find_owned_sandbox_id(owner, "shell", api_key)
            if value in failed_ids:
                value = None
            elif value:
                await _set_stored_sandbox_id(owner, value)

        if value and value != _CREATING_SENTINEL:
            # Existing sandbox ID — try to reconnect (auto-resumes if paused).
            sandbox = await _try_reconnect(value, owner, api_key)
            if sandbox:
                logger.info("[E2B] Reconnected to %.12s for %s", value, owner)
                await _acquire_turn(owner)
                return sandbox
            # _try_reconnect cleared the key — loop to create a new sandbox.
            failed_ids.add(value)
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
        # for the same owner wait rather than racing to create duplicates.
        sandbox: AsyncSandbox | None = None
        try:
            # Our own image is built on the team the first time it is needed.
            await ensure_template(template, api_key)
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
            mounts = await _resolve_volume_mounts(volume_mounts, api_key)
            last_exc: Exception | None = None
            for attempt in range(1, _SANDBOX_CREATE_MAX_RETRIES + 1):
                try:
                    sandbox = await asyncio.wait_for(
                        AsyncSandbox.create(
                            template=template,
                            api_key=api_key,
                            timeout=timeout,
                            lifecycle=lifecycle,
                            volume_mounts=mounts,
                            metadata=owner.creation_metadata(
                                "shell",
                                user_id=user_id,
                                session_id=session_id,
                                template=template,
                                mounts="attached" if mounts else "none",
                            ),
                        ),
                        timeout=_SANDBOX_CREATE_TIMEOUT_SECONDS,
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "[E2B] Sandbox creation attempt %d/%d failed for %s: %s",
                        attempt,
                        _SANDBOX_CREATE_MAX_RETRIES,
                        owner,
                        exc,
                    )
                    if (
                        mounts is not None
                        and attempt == _SANDBOX_CREATE_MAX_RETRIES - 1
                    ):
                        # A volume problem must not cost the user their shell,
                        # but a transient failure must not cost an expert its
                        # durable home either: keep the mounts through the
                        # retries and only make the last attempt volume-less.
                        logger.warning(
                            "[E2B] Final attempt for %s will run without workspace volumes",
                            owner,
                        )
                        mounts = None
                    if attempt < _SANDBOX_CREATE_MAX_RETRIES:
                        await asyncio.sleep(2 ** (attempt - 1))  # 1 s, 2 s

            if last_exc is not None:
                raise last_exc

            assert sandbox is not None  # guaranteed: last_exc is None iff break was hit
            if mounts:
                with contextlib.suppress(Exception):
                    await sandbox.commands.run(
                        "mkdir -p " + " ".join(f"'{path}'" for path in mounts)
                    )
            try:
                await _set_stored_sandbox_id(owner, sandbox.sandbox_id)
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

        logger.info("[E2B] Created sandbox %.12s for %s", sandbox.sandbox_id, owner)
        await _acquire_turn(owner)
        return sandbox

    raise RuntimeError(f"Could not acquire E2B sandbox for {owner}")


async def _act_on_sandbox(
    owner: SandboxOwner,
    api_key: str,
    action: str,
    fn: Callable[[AsyncSandbox], Awaitable[Any]],
    *,
    sandbox_kind: SandboxKind = "shell",
    sandbox_id: str | None = None,
    clear_stored_id: bool = False,
) -> bool:
    """Connect to the owner's sandbox and run *fn* on it.

    Shared by ``pause_sandbox``, ``kill_sandbox`` and
    ``kill_expert_sandboxes``.  Returns ``True`` on success, ``False`` when no
    sandbox is found or the action fails.  If *clear_stored_id* is ``True``,
    the sandbox_id is removed from Redis only after the action succeeds so a
    failed kill can be retried.
    """
    if sandbox_id is None:
        sandbox_id = await _get_stored_sandbox_id(owner, sandbox_kind)
    if not sandbox_id:
        return False

    async def _run() -> None:
        await fn(await AsyncSandbox.connect(sandbox_id, api_key=api_key))

    try:
        await asyncio.wait_for(_run(), timeout=_E2B_API_TIMEOUT_SECONDS)
        if clear_stored_id:
            await _clear_stored_sandbox_id(owner, sandbox_kind)
        logger.info(
            "[E2B] %s %s sandbox %.12s for %s",
            action.capitalize(),
            sandbox_kind,
            sandbox_id,
            owner,
        )
        return True
    except Exception as exc:
        logger.warning(
            "[E2B] Failed to %s %s sandbox %.12s for %s: %s",
            action,
            sandbox_kind,
            sandbox_id,
            owner,
            exc,
        )
        return False


async def pause_sandbox(
    session_id: str, api_key: str, *, expert_id: str | None = None
) -> bool:
    """Pause the E2B sandbox for this turn's owner to stop billing between turns.

    Paused sandboxes cost nothing and are resumed automatically by
    ``get_or_create_sandbox()`` on the next turn (via ``AsyncSandbox.connect()``).
    The sandbox_id is kept in Redis so reconnection works seamlessly.  An
    expert's box is left running while another of its turns is still active.

    Prefer ``pause_sandbox_direct()`` when the sandbox object is already in
    scope — it skips the Redis lookup and reconnect round-trip.

    Returns ``True`` if the sandbox was found and paused, ``False`` otherwise.
    Safe to call even when no sandbox exists for the session.
    """
    owner = SandboxOwner.for_session(session_id, expert_id)
    if not await _release_turn(owner):
        return False
    return await _act_on_sandbox(owner, api_key, "pause", lambda sb: sb.pause())


async def pause_sandbox_direct(
    sandbox: "AsyncSandbox", session_id: str, *, expert_id: str | None = None
) -> bool:
    """Pause an already-connected sandbox without a reconnect round-trip.

    Use this in callers that already hold the live sandbox object (e.g. turn
    teardown in ``service.py``).  Saves the Redis lookup and
    ``AsyncSandbox.connect()`` call that ``pause_sandbox()`` would make.  An
    expert's box is left running while another of its turns is still active.

    Returns ``True`` on success, ``False`` on failure, timeout, or when the
    box is deliberately left running.
    """
    owner = SandboxOwner.for_session(session_id, expert_id)
    if not await _release_turn(owner):
        return False
    try:
        await asyncio.wait_for(sandbox.pause(), timeout=_E2B_API_TIMEOUT_SECONDS)
        logger.info("[E2B] Paused sandbox %.12s for %s", sandbox.sandbox_id, owner)
        return True
    except Exception as exc:
        logger.warning(
            "[E2B] Failed to pause sandbox %.12s for %s: %s",
            sandbox.sandbox_id,
            owner,
            exc,
        )
        return False


async def kill_sandbox(
    session_id: str,
    api_key: str,
) -> bool:
    """Kill the session's E2B sandboxes (shell and on-demand desktop).

    Only ever touches *session* sandboxes: an expert session has none under
    its own id, so deleting an expert chat leaves the expert's computer alone
    (see ``kill_expert_sandboxes`` for the archive path).  The desktop is
    included because nothing else ever stops it: it is deliberately not paused
    at turn end, so an orphaned one would bill until its timeout and then sit
    paused forever.

    Returns ``True`` if at least one sandbox was found and killed, ``False``
    otherwise.  Safe to call even when no sandbox exists for the session.
    """
    owner = SandboxOwner(kind="session", id=session_id)
    killed = False
    for sandbox_kind in ("shell", "desktop"):
        if await _act_on_sandbox(
            owner,
            api_key,
            "kill",
            lambda sb: sb.kill(),
            sandbox_kind=sandbox_kind,
            clear_stored_id=True,
        ):
            killed = True
    return killed


async def kill_expert_sandboxes(expert_id: str, api_key: str) -> int:
    """Kill an expert's shell and desktop boxes — its computer goes on archive.

    The expert's volume is deliberately kept: files outlive the machine, and
    destroying user data is not something a best-effort cleanup should do.
    Falls back to the E2B metadata lookup for each box so a stale Redis cache
    cannot leave a paused machine behind.  Returns the number killed.
    """
    owner = SandboxOwner(kind="expert", id=expert_id)
    killed = 0
    for sandbox_kind in ("shell", "desktop"):
        sandbox_id = await _get_stored_sandbox_id(owner, sandbox_kind)
        if not sandbox_id:
            sandbox_id = await find_owned_sandbox_id(owner, sandbox_kind, api_key)
        if not sandbox_id:
            continue
        if await _act_on_sandbox(
            owner,
            api_key,
            "kill",
            lambda sb: sb.kill(),
            sandbox_kind=sandbox_kind,
            sandbox_id=sandbox_id,
            clear_stored_id=True,
        ):
            killed += 1
    with contextlib.suppress(Exception):
        redis = await get_redis_async()
        await redis.delete(_active_turns_key(owner))
    return killed
