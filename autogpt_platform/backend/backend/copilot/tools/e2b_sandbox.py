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
  kickoffs) reconnects to the same box, so tools it installs, files it writes
  and anything left signed in in its browser are still there next time.  The
  model has a root shell in that same VM, so nothing on the box, browser
  sessions included, is private from it.  Deleting a
  chat never touches it; archiving the expert does
  (``kill_expert_sandbox``).  The box mounts the expert's own durable volume
  at ``~/workspace`` and the owning user's volume at ``~/shared`` (see
  ``workspace_volume_mounts``).

There is exactly one sandbox per owner.  It runs our desktop image
(``backend.util.e2b_template``), and ``backend.copilot.computer`` turns its
screen on *in place* when a desktop is wanted: no second box.

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
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from e2b import (
    AsyncSandbox,
    AsyncVolume,
    SandboxInfo,
    SandboxLifecycle,
    SandboxQuery,
    SandboxState,
)
from e2b.exceptions import NotFoundException
from pydantic import BaseModel, ConfigDict
from redis_lua_py import Key, redis, script

from backend.blocks.desktop._api import DesktopSession, resolve_volume
from backend.data.redis_client import get_redis_async
from backend.util.e2b_network import (
    EgressOwner,
    connect_sandbox,
    create_sandbox,
    forget_sandbox,
)
from backend.util.e2b_template import ensure_template, forget_template
from backend.util.sandbox_metadata import MountState, SandboxMetadata, owned_by_user

logger = logging.getLogger(__name__)

_SANDBOX_KEY_PREFIX = "copilot:e2b:sandbox:"
_EXPERT_KEY_PREFIX = "copilot:e2b:expert:"
_CREATING_SENTINEL = "creating"

# E2B sandbox metadata that lets an owner find its box without Redis.
METADATA_OWNER = "autogpt_owner"
METADATA_KIND = "autogpt_kind"
# "attached" when the workspace volumes were mounted, "none" when creation had
# to fall back to a volume-less box — visible in the E2B dashboard and API.
METADATA_MOUNTS = "autogpt_mounts"

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

# Bound on stopping the screen's stream before a pause: a box that does not
# answer must not hold the pause up for long.
_STOP_STREAM_TIMEOUT_SECONDS = 5

# Held in place of a stream password once the stream has been stopped in the
# box: nothing is serving, so a reconnect has nothing to stop.  Empty on
# purpose, so every reader that wants a password sees none.
_STREAM_STOPPED = ""

# Redis TTL for a session sandbox key.  Must be ≥ the E2B project "paused
# sandbox lifetime" setting (recommended: set both to 48 h).
_SANDBOX_ID_TTL = 48 * 3600  # 48 hours

# An expert's box is meant to outlive any one chat.  The key is refreshed on
# every use and E2B metadata recovers it after expiry, so this is only a cache
# TTL — not a lifetime.
_EXPERT_ID_TTL = 30 * 24 * 3600

# Leak guard for the per-expert active-turn counter: a turn that dies without
# releasing its slot stops blocking the pause after this long.  A turn may
# legitimately run for the 6 h executor limit (``consumer-timeout`` in
# ``executor/utils.py``), so the guard sits just past that; every acquire
# refreshes it.
_ACTIVE_TURN_TTL = 6 * 3600 + 15 * 60


# Count one turn and (re)arm the key's expiry in one script: an INCR that
# lands without its EXPIRE would leave a count nothing ever releases, and the
# box would never pause at turn end.
@script
def _incr_active_turns(key: Key, ttl_seconds: int) -> int:
    n = redis.incr(key)
    redis.expire(key, ttl_seconds)
    return n


# Release one turn and report how many are left; the last one out deletes
# the key.  One script, so a turn that starts between the DECR and the DEL
# can never have its count wiped.
@script
def _decr_active_turns(key: Key) -> int:
    n = redis.decr(key)
    if n <= 0:
        redis.delete(key)
        return 0
    return n


# One more reconnect attempt before a transient error surfaces: a box that
# fails once on a blip must not be replaced, whoever owns it.
_RECONNECT_RETRY_DELAY_SECONDS = 1.0

# Boxes from before one box per owner: a separate desktop, stamped
# ``autogpt_kind=desktop`` and cached under its own key, that this code no
# longer opens.  Left alone it would sit paused on E2B, browser profile and
# all, for as long as E2B keeps paused boxes.  The kill paths sweep them so a
# deleted chat or an archived expert takes its old desktop with it; once no
# box from before the switch can exist any more this can go.
_LEGACY_DESKTOP_KIND = "desktop"
_LEGACY_DESKTOP_KEY_PREFIX = "copilot:e2b:desktop:"


class SandboxOwner(BaseModel):
    """Who a CoPilot sandbox belongs to — and therefore how long it lives.

    ``session`` sandboxes are scratch for one chat.  ``expert`` sandboxes are
    a hired expert's own persistent computer, shared by every session that
    runs as that expert — one box per expert, not one per account, so two
    experts never see each other's files or browser sessions.
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

    def key(self) -> str:
        """Redis key caching this owner's sandbox id (doubles as creation lock)."""
        if self.is_expert:
            return f"{_EXPERT_KEY_PREFIX}{self.id}:shell"
        return f"{_SANDBOX_KEY_PREFIX}{self.id}"

    def display_key(self) -> str:
        """Redis key remembering which box ``open_desktop`` turned the screen on in."""
        return f"{self.key()}:display"

    def stream_key(self) -> str:
        """Redis key holding the screen's current stream password.

        The password never rests on the box (see ``DesktopSession.start_stream``);
        this is what lets a re-open hand back the URL the user already holds.
        When we pause the box the stream is stopped and the password dropped
        (``_revoke_stream``), so a URL that may have leaked stops working at
        the turn-end pause.  A pause E2B makes on its own timeout is only
        caught up with at our next connect (``_settle_stream``).
        """
        return f"{self.key()}:stream"

    def egress_owner(self, user_id: str | None) -> EgressOwner:
        """Who the egress proxy sees this box as (``backend.util.e2b_network``)."""
        return EgressOwner(kind=self.kind, id=self.id, user_id=user_id)

    def display_lock_key(self) -> str:
        """Redis key held by whoever is turning the screen on right now."""
        return f"{self.display_key()}:lock"

    def legacy_desktop_key(self) -> str:
        """Where the pre-one-box desktop's id was cached; swept on kill."""
        if self.is_expert:
            return f"{_EXPERT_KEY_PREFIX}{self.id}:desktop"
        return f"{_LEGACY_DESKTOP_KEY_PREFIX}{self.id}"

    @property
    def ttl(self) -> int:
        return _EXPERT_ID_TTL if self.is_expert else _SANDBOX_ID_TTL

    def metadata(self) -> dict[str, str]:
        """The identity keys a lookup filters on; a subset of ``creation_metadata``."""
        return {METADATA_OWNER: f"{self.kind}:{self.id}", METADATA_KIND: "shell"}

    def creation_metadata(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        template: str | None = None,
        mounts: MountState | None = None,
    ) -> dict[str, str]:
        """Identity plus provenance, stamped on the box when it is created."""
        return SandboxMetadata.for_copilot(
            f"{self.kind}:{self.id}",
            "shell",
            user_id=user_id,
            session_id=session_id,
            expert_id=self.id if self.is_expert else None,
            template=template,
            mounts=mounts,
        ).as_e2b()

    def __str__(self) -> str:
        return f"{self.kind} {self.id[:12]}"


class SandboxNotOwnedError(Exception):
    """A sandbox id resolved to a box that is not the owner's."""


async def connect_owned(
    sandbox_id: str,
    owner: SandboxOwner,
    api_key: str,
    *,
    timeout: int | None = None,
    user_id: str | None = None,
    pin_egress: bool = True,
) -> AsyncSandbox:
    """Connect to *sandbox_id* only if E2B says it belongs to *owner*.

    Under the platform's E2B key any sandbox id connects, so the id alone
    (from Redis, a message, or a caller) must never be trusted: the box's
    stamped ``autogpt_owner`` / ``autogpt_kind`` metadata is the record of
    who it belongs to.  Raises :class:`SandboxNotOwnedError` otherwise.

    The stamp is read *before* connecting: a connect is not passive, it
    resumes a paused box and re-arms its running-time limit, so a foreign
    id must be refused without ever waking someone else's box.  *timeout*
    is that limit for the owner's box (a resumed box would otherwise get the
    SDK's default).  A connect that will run work re-pins the box's egress
    (``backend.util.e2b_network``) for *user_id*; one that only pauses or
    kills passes ``pin_egress=False``.
    """
    info = await _owned_info(sandbox_id, owner, api_key)
    return await _connect_pinned(
        sandbox_id,
        info,
        owner,
        api_key,
        timeout=timeout,
        user_id=user_id,
        pin_egress=pin_egress,
    )


async def _owned_info(
    sandbox_id: str, owner: SandboxOwner, api_key: str
) -> SandboxInfo:
    """What E2B says about *sandbox_id*, refused unless it is *owner*'s box."""
    info = await AsyncSandbox.get_info(sandbox_id, api_key=api_key)
    expected = owner.metadata()
    stamped = info.metadata or {}
    if any(stamped.get(key) != value for key, value in expected.items()):
        raise SandboxNotOwnedError(f"Sandbox {sandbox_id[:12]} is not {owner}'s box")
    return info


async def _connect_pinned(
    sandbox_id: str,
    info: SandboxInfo,
    owner: SandboxOwner,
    api_key: str,
    *,
    timeout: int | None,
    user_id: str | None,
    pin_egress: bool,
) -> AsyncSandbox:
    """Connect to *sandbox_id*, which *info* already showed to be *owner*'s."""
    stamped = info.metadata or {}
    # Whose credentials the proxy may swap in is the box's own record too,
    # not the caller's word: processes of the user it was created for may
    # still be running in it.  An expert has one owner today, but nothing at
    # this layer says so, and a box re-pinned for whoever reconnects would
    # let those processes act as them.  A mismatch pins the box with no user:
    # it keeps its egress and gets nothing swapped in.
    swap_user_id = user_id if owned_by_user(stamped, user_id) else None
    if pin_egress and user_id and swap_user_id is None:
        logger.warning(
            "[E2B] Sandbox %.12s was not created for the user reconnecting to "
            "it; pinning it without credentials",
            sandbox_id,
        )
    return await connect_sandbox(
        AsyncSandbox,
        sandbox_id,
        owner.egress_owner(swap_user_id),
        apply_network=pin_egress,
        api_key=api_key,
        timeout=timeout,
    )


def _as_owner(owner: "SandboxOwner | str") -> SandboxOwner:
    """Accept a bare session id where an owner is expected."""
    if isinstance(owner, SandboxOwner):
        return owner
    return SandboxOwner(kind="session", id=owner)


def _sandbox_key(session_id: str) -> str:
    return SandboxOwner(kind="session", id=session_id).key()


async def _get_stored_sandbox_id(owner: SandboxOwner) -> str | None:
    redis = await get_redis_async()
    raw = await redis.get(owner.key())
    value = raw.decode() if isinstance(raw, bytes) else raw
    return None if value == _CREATING_SENTINEL else value


async def _set_stored_sandbox_id(owner: SandboxOwner, sandbox_id: str) -> None:
    redis = await get_redis_async()
    await redis.set(owner.key(), sandbox_id, ex=owner.ttl)


async def _clear_stored_sandbox_id(owner: SandboxOwner) -> None:
    redis = await get_redis_async()
    await redis.delete(owner.key())


class SandboxLookupError(Exception):
    """E2B could not tell us which boxes an owner has.

    Distinct from "none": for an expert the box *is* the state, so a failed
    lookup must not be read as "no box" and answered with a fresh one.
    """


async def list_owned_sandboxes(owner: SandboxOwner, api_key: str) -> list[SandboxInfo]:
    """The owner's boxes as E2B lists them, newest first.

    A running box sorts before a paused one.  Listing never connects, so a
    paused box stays paused — connecting is what auto-resume reacts to.
    Raises :class:`SandboxLookupError` when the API call fails or times out.
    """
    try:
        paginator = AsyncSandbox.list(
            query=SandboxQuery(
                metadata=owner.metadata(),
                state=[SandboxState.RUNNING, SandboxState.PAUSED],
            ),
            limit=10,
            api_key=api_key,
        )
        infos = await asyncio.wait_for(
            paginator.next_items(), timeout=_E2B_API_TIMEOUT_SECONDS
        )
    except Exception as exc:
        raise SandboxLookupError(
            f"E2B lookup of {owner}'s boxes failed: {exc}"
        ) from exc
    infos = sorted(infos, key=lambda info: info.started_at, reverse=True)
    running = [info for info in infos if info.state == SandboxState.RUNNING]
    paused = [info for info in infos if info.state != SandboxState.RUNNING]
    return running + paused


async def find_owned_sandbox_id(owner: SandboxOwner, api_key: str) -> str | None:
    """Recover an expert's box through the E2B API when Redis has forgotten it.

    Session sandboxes are Redis-only: losing that key just means a fresh
    scratch sandbox, which is not worth a control-plane round-trip.  For an
    expert the box *is* the state, so we query E2B by the owner metadata every
    sandbox is stamped with, preferring a running box over a paused one and
    the newest of several (a lost creation race can leave duplicates).
    A failed lookup raises :class:`SandboxLookupError` rather than answering
    ``None``: "unknown" must never turn into a second box.
    """
    if not owner.is_expert:
        return None
    infos = await list_owned_sandboxes(owner, api_key)
    if not infos:
        return None
    chosen = infos[0]
    if len(infos) > 1:
        logger.warning(
            "[E2B] %s has %d sandboxes; using %.12s",
            owner,
            len(infos),
            chosen.sandbox_id,
        )
    return chosen.sandbox_id


async def _try_reconnect(
    sandbox_id: str,
    owner: "SandboxOwner | str",
    api_key: str,
    *,
    timeout: int | None = None,
    user_id: str | None = None,
) -> "AsyncSandbox | None":
    """Reconnect to the owner's box, or ``None`` if it is gone.

    Gone means E2B no longer has it, it is stamped for someone else, or it
    came back not running: the cached id is dropped so a replacement can be
    created.  Anything else (a 5xx, a network blip) is raised, not swallowed.
    The box may be perfectly fine, and replacing it on a guess would fork
    everything on it that is not in a volume: the screen, running processes,
    installed tools.  *timeout* re-arms the box's running-time limit.
    """
    owner = _as_owner(owner)
    try:
        # Same order as ``connect_owned``: the stamp is read before the connect
        # wakes anything.  The state read with it says whether this connect is
        # what resumes the box.
        info = await _owned_info(sandbox_id, owner, api_key)
        sandbox = await _connect_pinned(
            sandbox_id,
            info,
            owner,
            api_key,
            timeout=timeout,
            user_id=user_id,
            pin_egress=True,
        )
    except SandboxNotOwnedError as exc:
        logger.warning("[E2B] Refusing reconnect: %s", exc)
    except NotFoundException as exc:
        logger.warning("[E2B] Box %.12s is gone: %s", sandbox_id, exc)
    else:
        if await sandbox.is_running():
            # Refresh TTL so an active owner cannot lose its sandbox_id at expiry.
            await _set_stored_sandbox_id(owner, sandbox_id)
            await _settle_stream(
                owner,
                sandbox,
                resumed=info.state == SandboxState.PAUSED,
                timeout=timeout,
            )
            return sandbox
        logger.warning("[E2B] Box %.12s came back not running", sandbox_id)

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
    return f"{owner.key()}:active"


async def _acquire_turn(owner: SandboxOwner) -> None:
    """Count this turn on an expert's box so another turn's end can't pause it.

    Fails closed: if the count cannot be recorded the turn must not run on
    the box, because its eventual release would decrement a count it never
    added and could pause the box under a concurrent turn.
    """
    if not owner.is_expert:
        return
    redis = await get_redis_async()
    key = _active_turns_key(owner)
    await _incr_active_turns(redis, key=key, ttl_seconds=_ACTIVE_TURN_TTL)


async def count_expert_turn(session_id: str, expert_id: str | None) -> None:
    """Count a turn on the expert's box (no-op for a plain session).

    For callers that opened the box with ``count_turn=False`` because work
    that could still fail sits between opening it and the ``try`` whose
    ``finally`` releases the turn.  Count only once that ``try`` is reached.
    """
    await _acquire_turn(SandboxOwner.for_session(session_id, expert_id))


async def _release_turn(owner: SandboxOwner) -> bool:
    """Return ``True`` when the box may be paused: no other turn is still on it."""
    if not owner.is_expert:
        return True
    try:
        redis = await get_redis_async()
        key = _active_turns_key(owner)
        remaining = await _decr_active_turns(redis, key=key)
        if remaining <= 0:
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


async def get_or_create_owner_sandbox(
    owner: SandboxOwner,
    api_key: str,
    timeout: int,
    template: str = "base",
    on_timeout: Literal["kill", "pause"] = "pause",
    volume_mounts: Mapping[str, str] | None = None,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    count_turn: bool = True,
) -> AsyncSandbox:
    """Return the owner's E2B sandbox, creating it if needed.

    The owner's key in Redis serves a dual purpose: it stores the sandbox_id
    and acts as a creation lock via a ``"creating"`` sentinel value.  This
    removes the need for a separate lock key.

    *timeout* controls how long the e2b sandbox may run continuously before
    the ``on_timeout`` lifecycle rule fires (default: 5 min).
    *on_timeout* controls what happens on timeout: ``"pause"`` (default, free)
    or ``"kill"``.  When ``"pause"``, ``auto_resume`` is enabled so paused
    sandboxes wake transparently on SDK activity.
    *volume_mounts* maps mount paths to durable volume names (see
    ``workspace_volume_mounts``) so the box's ``~/workspace`` and, for an
    expert, the owning user's ``~/shared`` persist across boxes.  A mount
    failure degrades to a volume-less sandbox rather than failing the session.
    *count_turn* records this caller as an active turn on an expert's box so
    another turn's end cannot pause it; pass ``False`` when opening the box
    from outside a turn (turning its screen on from the UI), or when the
    release is not yet guaranteed to run and ``count_expert_turn`` follows.
    *user_id* / *session_id* are provenance only, stamped on a newly created box.

    Raises :class:`SandboxLookupError` when E2B cannot say whether an expert
    already has a box: a fresh box would fork the expert's durable state.
    """
    redis = await get_redis_async()
    key = owner.key()
    # Boxes E2B still lists but that are gone by the time we connect (a
    # teardown in progress).  Without this an expert owner would re-find the
    # same id on every iteration and never fall through to creating a fresh one.
    failed_ids: set[str] = set()
    # Boxes that already had their one retry after a transient error.
    retried_ids: set[str] = set()

    for _ in range(_MAX_WAIT_ATTEMPTS):
        raw = await redis.get(key)
        value = raw.decode() if isinstance(raw, bytes) else raw

        if not value and owner.is_expert:
            # Redis is only a cache for an expert's box; E2B is the record.
            value = await find_owned_sandbox_id(owner, api_key)
            if value in failed_ids:
                value = None
            elif value:
                await _set_stored_sandbox_id(owner, value)

        if value and value != _CREATING_SENTINEL:
            # Existing sandbox ID — try to reconnect (auto-resumes if paused).
            try:
                sandbox = await _try_reconnect(
                    value, owner, api_key, timeout=timeout, user_id=user_id
                )
            except Exception as exc:
                if value in retried_ids:
                    raise
                # One more chance before the error surfaces: a single blip
                # must not replace a box, and asking the caller to try
                # again is better than forking what is on it.
                retried_ids.add(value)
                logger.warning(
                    "[E2B] Reconnect to %.12s failed (%s); retrying once", value, exc
                )
                await asyncio.sleep(_RECONNECT_RETRY_DELAY_SECONDS)
                continue
            if sandbox:
                logger.info("[E2B] Reconnected to %.12s for %s", value, owner)
                if count_turn:
                    await _acquire_turn(owner)
                return sandbox
            # The box is gone and _try_reconnect cleared the key — loop to
            # create a new sandbox.
            failed_ids.add(value)
            continue

        if value == _CREATING_SENTINEL:
            # Another coroutine is creating — wait for it to finish.
            await asyncio.sleep(_WAIT_INTERVAL_SECONDS)
            continue

        # No sandbox and no active creation.  Our own image is built on the
        # team the first time it is needed; that happens before the creation
        # slot is claimed because a build can outlive the slot's TTL.  Cached
        # per process after the first check, so the repeat is free.
        await ensure_template(template, api_key)

        # Atomically claim the creation slot.
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
                        create_sandbox(
                            AsyncSandbox,
                            owner.egress_owner(user_id),
                            template=template,
                            api_key=api_key,
                            timeout=timeout,
                            lifecycle=lifecycle,
                            volume_mounts=mounts,
                            metadata=owner.creation_metadata(
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
                # The template may have gone away since this process last
                # confirmed it; make the next attempt look again.
                forget_template(template, api_key)
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
                await forget_sandbox(sandbox.sandbox_id)
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
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await forget_sandbox(sandbox.sandbox_id)
            raise
        except Exception:
            # Release the creation slot so other callers can proceed.
            await redis.delete(key)
            raise

        logger.info("[E2B] Created sandbox %.12s for %s", sandbox.sandbox_id, owner)
        if count_turn:
            await _acquire_turn(owner)
        return sandbox

    raise RuntimeError(f"Could not acquire E2B sandbox for {owner}")


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
    count_turn: bool = True,
) -> AsyncSandbox:
    """The sandbox for this turn (the session's, or its expert's), counting the turn.

    Thin wrapper over ``get_or_create_owner_sandbox`` for the chat engines,
    which know a session and maybe an expert rather than an owner.
    """
    return await get_or_create_owner_sandbox(
        SandboxOwner.for_session(session_id, expert_id),
        api_key,
        timeout=timeout,
        template=template,
        on_timeout=on_timeout,
        volume_mounts=volume_mounts,
        user_id=user_id,
        session_id=session_id,
        count_turn=count_turn,
    )


async def _act_on_sandbox(
    owner: SandboxOwner,
    api_key: str,
    action: str,
    fn: Callable[[AsyncSandbox], Awaitable[Any]],
    *,
    sandbox_id: str | None = None,
    clear_stored_id: bool = False,
    timeout: float = _E2B_API_TIMEOUT_SECONDS,
) -> bool:
    """Connect to the owner's sandbox and run *fn* on it.

    Shared by ``pause_sandbox``, ``kill_sandbox`` and
    ``kill_expert_sandbox``.  Returns ``True`` on success, ``False`` when no
    sandbox is found or the action fails.  If *clear_stored_id* is ``True``,
    the sandbox_id is removed from Redis only after the action succeeds so a
    failed kill can be retried.
    """
    if sandbox_id is None:
        sandbox_id = await _get_stored_sandbox_id(owner)
    if not sandbox_id:
        return False

    async def _run() -> None:
        # Nothing egresses before a pause or kill: no re-pin.
        await fn(await connect_owned(sandbox_id, owner, api_key, pin_egress=False))

    try:
        await asyncio.wait_for(_run(), timeout=timeout)
        # Paused or killed, the box will not present its proxy credential
        # again: a resume mints a fresh one.
        await forget_sandbox(sandbox_id)
        if clear_stored_id:
            await _clear_stored_sandbox_id(owner)
        logger.info(
            "[E2B] %s sandbox %.12s for %s", action.capitalize(), sandbox_id, owner
        )
        return True
    except SandboxNotOwnedError as exc:
        # A cached id that is not the owner's box any more is never going to
        # be: forget it, or every later pause and kill fails the same way.
        logger.warning("[E2B] Refusing to %s: %s", action, exc)
        await _clear_stored_sandbox_id(owner)
        return False
    except Exception as exc:
        logger.warning(
            "[E2B] Failed to %s sandbox %.12s for %s: %s",
            action,
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
    revoked = False

    async def _pause(sandbox: AsyncSandbox) -> None:
        nonlocal revoked
        revoked = await _revoke_stream(owner, sandbox)
        await sandbox.pause()

    # The stop gets its own share of the budget, so a box slow to answer it
    # still leaves the connect and the pause the time they always had.
    paused = await _act_on_sandbox(
        owner,
        api_key,
        "pause",
        _pause,
        timeout=_E2B_API_TIMEOUT_SECONDS + _STOP_STREAM_TIMEOUT_SECONDS,
    )
    if paused and not revoked:
        await _forget_stream(owner)
    return paused


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
    revoked = await _revoke_stream(owner, sandbox)
    try:
        await asyncio.wait_for(sandbox.pause(), timeout=_E2B_API_TIMEOUT_SECONDS)
        logger.info("[E2B] Paused sandbox %.12s for %s", sandbox.sandbox_id, owner)
        await forget_sandbox(sandbox.sandbox_id)
        if not revoked:
            await _forget_stream(owner)
        return True
    except Exception as exc:
        logger.warning(
            "[E2B] Failed to pause sandbox %.12s for %s: %s",
            sandbox.sandbox_id,
            owner,
            exc,
        )
        return False


async def kill_sandbox(session_id: str, api_key: str) -> bool:
    """Kill a session's box: the chat is gone, so is its scratch computer.

    Only ever touches a *session* box: an expert session has none under its
    own id, so deleting an expert chat leaves the expert's computer alone
    (see ``kill_expert_sandbox`` for the archive path).

    Returns ``True`` if a sandbox was found and killed, ``False`` otherwise.
    Safe to call even when no sandbox exists for the session.
    """
    owner = SandboxOwner(kind="session", id=session_id)
    sandbox_id = await _get_stored_sandbox_id(owner)
    if not sandbox_id:
        return await _kill_legacy_desktops(owner, api_key) > 0
    killed = await _act_on_sandbox(
        owner,
        api_key,
        "kill",
        lambda sb: sb.kill(),
        sandbox_id=sandbox_id,
        clear_stored_id=True,
    )
    if killed:
        await _forget_owner_state(owner)
    await _kill_legacy_desktops(owner, api_key)
    # The current box is the verdict: a swept old desktop must not report a
    # failed kill of it as done.
    return killed


async def kill_expert_sandbox(expert_id: str, api_key: str) -> bool:
    """Kill an expert's box: its computer goes on archive.

    The expert's volume is deliberately kept: files outlive the machine, and
    destroying user data is not something a best-effort cleanup should do.
    Falls back to the E2B metadata lookup so a stale Redis cache cannot leave
    a paused machine behind.  Returns ``True`` if a box was killed.
    """
    owner = SandboxOwner(kind="expert", id=expert_id)
    swept = await _kill_legacy_desktops(owner, api_key)
    sandbox_id = await _get_stored_sandbox_id(owner)
    if not sandbox_id:
        try:
            sandbox_id = await find_owned_sandbox_id(owner, api_key)
        except SandboxLookupError as exc:
            # Unknown whether a box exists: not done, whatever was swept.
            logger.warning("[E2B] Archive of %s: %s", owner, exc)
            return False
    if not sandbox_id:
        return swept > 0
    killed = await _act_on_sandbox(
        owner,
        api_key,
        "kill",
        lambda sb: sb.kill(),
        sandbox_id=sandbox_id,
        clear_stored_id=True,
    )
    if killed:
        await _forget_owner_state(owner)
    # The current box is the verdict: a swept old desktop must not report a
    # failed kill of it as done.
    return killed


async def _kill_legacy_desktops(owner: SandboxOwner, api_key: str) -> int:
    """Kill the owner's pre-one-box desktop boxes, if any are still around.

    Found by stamp, not by cache, so a desktop whose key expired is swept
    too.  Killed by id without connecting: a paused desktop must not be
    resumed (and billed) just to be destroyed.  Returns how many were killed.
    """
    try:
        paginator = AsyncSandbox.list(
            query=SandboxQuery(
                metadata={**owner.metadata(), METADATA_KIND: _LEGACY_DESKTOP_KIND},
                state=[SandboxState.RUNNING, SandboxState.PAUSED],
            ),
            limit=10,
            api_key=api_key,
        )
        infos = await asyncio.wait_for(
            paginator.next_items(), timeout=_E2B_API_TIMEOUT_SECONDS
        )
    except Exception as exc:
        logger.warning("[E2B] Could not list %s's old desktop boxes: %s", owner, exc)
        return 0
    killed = 0
    for info in infos:
        try:
            await asyncio.wait_for(
                AsyncSandbox.kill(info.sandbox_id, api_key=api_key),
                timeout=_E2B_API_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning(
                "[E2B] Failed to kill old desktop %.12s for %s: %s",
                info.sandbox_id,
                owner,
                exc,
            )
            continue
        killed += 1
        logger.info("[E2B] Killed old desktop %.12s for %s", info.sandbox_id, owner)
    with contextlib.suppress(Exception):
        redis = await get_redis_async()
        await redis.delete(owner.legacy_desktop_key())
    return killed


async def _forget_owner_state(owner: SandboxOwner) -> None:
    """Drop the screen flag, stream password and turn counter once their box
    is really gone."""
    with contextlib.suppress(Exception):
        redis = await get_redis_async()
        await redis.delete(
            owner.display_key(), owner.stream_key(), _active_turns_key(owner)
        )


async def _revoke_stream(owner: SandboxOwner, sandbox: AsyncSandbox) -> bool:
    """Stop the screen's stream in *sandbox* so its password stops working.

    Forgetting the password is not enough on its own: a pause keeps the box's
    processes, so the stream would come back with the box and still answer to
    it.  Only the stream goes; the display stays up and the next open serves
    it again under a fresh password.  A box whose screen was never turned on
    costs one Redis read and no command.  Never raises, because a box that
    cannot be told to stop must still pause.  Returns whether the stream is
    known to be stopped.
    """
    try:
        redis = await get_redis_async()
        if not await _screen_started_in(owner, sandbox.sandbox_id):
            return False
        await asyncio.wait_for(
            DesktopSession(sandbox).stop_stream(),
            timeout=_STOP_STREAM_TIMEOUT_SECONDS,
        )
        await redis.set(owner.stream_key(), _STREAM_STOPPED, ex=owner.ttl)
    except Exception as exc:
        logger.warning(
            "[E2B] Could not stop the screen stream in %.12s for %s: %s",
            sandbox.sandbox_id,
            owner,
            exc,
        )
        return False
    logger.info("[E2B] Stopped the screen stream in %.12s", sandbox.sandbox_id)
    return True


async def _settle_stream(
    owner: SandboxOwner,
    sandbox: AsyncSandbox,
    *,
    resumed: bool,
    timeout: int | None,
) -> None:
    """On reconnect, stop a stream that outlived its running stretch.

    *resumed* is exact: E2B reported the box paused just before this connect
    woke it.  Unless the stream is already known to be stopped, it came back
    with the box (E2B paused it on its own timeout, or our stop before the
    pause failed) and is stopped now, whatever Redis still remembers.

    A box that was already running is left alone while its password is
    remembered, and the password's expiry is pushed out to the running-time
    limit this connect just re-armed, so a stream someone is watching on a
    box that never pauses is not mistaken for a leftover.  Only a running box
    with no stream key at all is stopped: the password outlived the box's
    limit, so the box did pause and something other than us resumed it.  An
    open in progress is left alone there; it restarts the stream itself.

    What this cannot reach: a box E2B paused on its timeout is resumed by any
    request to the stream URL, with no call through here, and serves under
    the old password until our next connect.  Closing that takes a stop the
    box runs itself, or a sweep, neither of which exists yet.
    """
    try:
        redis = await get_redis_async()
        if not await _screen_started_in(owner, sandbox.sandbox_id):
            return
        raw = await redis.get(owner.stream_key())
        password = raw.decode() if isinstance(raw, bytes) else raw
        if password == _STREAM_STOPPED:
            return
        if not resumed:
            if password is not None:
                if timeout:
                    await redis.expire(owner.stream_key(), timeout)
                return
            if await redis.get(owner.display_lock_key()) is not None:
                return
    except Exception as exc:
        logger.warning("[E2B] Could not read %s's screen state: %s", owner, exc)
        return
    await _revoke_stream(owner, sandbox)


async def _screen_started_in(owner: SandboxOwner, sandbox_id: str) -> bool:
    redis = await get_redis_async()
    raw = await redis.get(owner.display_key())
    value = raw.decode() if isinstance(raw, bytes) else raw
    return value == sandbox_id


async def _forget_stream(owner: SandboxOwner) -> None:
    """Drop the stream password: the screen's next open issues a fresh one.

    Called when we pause the box without its stream known to be stopped; the
    next connect finds the box paused and stops the stream then
    (``_settle_stream``).
    """
    with contextlib.suppress(Exception):
        redis = await get_redis_async()
        await redis.delete(owner.stream_key())
