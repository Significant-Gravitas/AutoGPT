"""An owner's computer on E2B: what exists, and how to get its desktop up.

Both the expert page's Computer tab and the copilot side panel's Computer
view read from here, so they agree on what "the box" is: the owner's shell
sandbox and its on-demand desktop, found through the same E2B metadata the
lifecycle module stamps on every sandbox (``copilot/tools/e2b_sandbox``).

``describe_computer`` only lists — it never connects, so a paused box stays
paused (connecting is what E2B's auto-resume reacts to).  ``open_desktop`` is
the one write: it creates or resumes the owner's desktop and hands back the
live stream, exactly as the ``start_desktop`` tool does from inside a turn.
"""

import asyncio
import contextlib
import logging
import uuid
from datetime import datetime
from typing import Any, Literal, Mapping, Optional

from e2b import SandboxState
from pydantic import BaseModel

from backend.blocks.desktop._api import DesktopSession, DesktopStream
from backend.blocks.desktop._common import (
    SHARED_PATH,
    WORKSPACE_PATH,
    workspace_volume_mounts,
)
from backend.copilot.sdk.env import config as chat_config
from backend.copilot.tools.e2b_sandbox import (
    METADATA_MOUNTS,
    SandboxKind,
    SandboxOwner,
    find_owned_sandbox_id,
    list_owned_sandboxes,
)
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_DESKTOP_RESOLUTION = (1280, 720)
_KILL_TIMEOUT_SECONDS = 10

# Opening a desktop can be volume resolution, two create attempts, the
# display coming up and the home setup: about four minutes at the very worst.
# The open is cut off before the lock can lapse, so a second opener can
# never slip in under a still-running first one, and it waits about as long.
_DESKTOP_LOCK_TTL_SECONDS = 300
_DESKTOP_OPEN_DEADLINE_SECONDS = _DESKTOP_LOCK_TTL_SECONDS - 15
_DESKTOP_LOCK_WAIT_SECONDS = 300
_DESKTOP_LOCK_POLL_SECONDS = 0.5
_UNLOCK_SCRIPT = (
    'if redis.call("get", KEYS[1]) == ARGV[1] then '
    'return redis.call("del", KEYS[1]) else return 0 end'
)


class SandboxSummary(BaseModel):
    """One of the owner's sandboxes as E2B reports it, without waking it."""

    kind: SandboxKind
    sandbox_id: str
    state: Literal["running", "paused"]
    started_at: datetime
    cpu_count: int
    memory_mb: int
    template_id: str
    mounts_attached: bool


class ComputerInfo(BaseModel):
    """The owner's computer: its boxes and the volumes they mount."""

    owner_kind: Literal["session", "expert"]
    owner_id: str
    e2b_active: bool
    shell: Optional[SandboxSummary] = None
    desktop: Optional[SandboxSummary] = None
    # path -> volume name; empty when the owner has no volumes (no user).
    mounts: dict[str, str] = {}
    workspace_path: str = WORKSPACE_PATH
    shared_path: str = SHARED_PATH


def computer_owner(session_id: str, expert_id: Optional[str]) -> SandboxOwner:
    return SandboxOwner.for_session(session_id, expert_id)


async def describe_computer(
    owner: SandboxOwner, mounts: Mapping[str, str]
) -> ComputerInfo:
    """List the owner's shell and desktop boxes without resuming either."""
    api_key = chat_config.active_e2b_api_key
    info = ComputerInfo(
        owner_kind=owner.kind,
        owner_id=owner.id,
        e2b_active=api_key is not None,
        mounts=dict(mounts),
    )
    if not api_key:
        return info
    kinds: tuple[SandboxKind, ...] = ("shell", "desktop")
    # Two independent E2B round-trips on an endpoint polled every 15 s.
    listings = await asyncio.gather(
        *(list_owned_sandboxes(owner, kind, api_key) for kind in kinds),
        return_exceptions=True,
    )
    for kind, boxes in zip(kinds, listings):
        if isinstance(boxes, BaseException):
            # A listing is informational; show nothing rather than fail the page.
            logger.warning("[E2B] describe_computer: %s", boxes)
            continue
        if not boxes:
            continue
        box = boxes[0]
        summary = SandboxSummary(
            kind=kind,
            sandbox_id=box.sandbox_id,
            state="running" if box.state == SandboxState.RUNNING else "paused",
            started_at=box.started_at,
            cpu_count=box.cpu_count,
            memory_mb=box.memory_mb,
            template_id=box.template_id,
            mounts_attached=(box.metadata or {}).get(METADATA_MOUNTS) == "attached",
        )
        if kind == "shell":
            info.shell = summary
        else:
            info.desktop = summary
    return info


async def open_desktop(
    owner: SandboxOwner,
    mounts: Mapping[str, str],
    api_key: str,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> tuple[DesktopStream, bool, bool]:
    """Return ``(stream, created, shared)`` — resuming the owner's desktop if it exists.

    Shared by the ``start_desktop`` tool and the HTTP endpoints, so a desktop
    opened from the expert page is the same box the expert's next turn finds.
    *user_id* / *session_id* are provenance only, stamped on a newly created box.

    One opener at a time per owner: the panel's Start and the model's
    ``start_desktop`` (or two tabs) can both miss the cache, and without the
    lock each would create a box, one of which nothing would ever find again.
    A second opener waits for the first and then reattaches to its box.
    """
    redis = await get_redis_async()
    key = owner.key("desktop")
    lock_key = f"{key}:lock"
    token = uuid.uuid4().hex
    waited = 0.0
    while not await redis.set(lock_key, token, nx=True, ex=_DESKTOP_LOCK_TTL_SECONDS):
        if waited >= _DESKTOP_LOCK_WAIT_SECONDS:
            raise RuntimeError(f"Another request is still opening {owner}'s desktop")
        await asyncio.sleep(_DESKTOP_LOCK_POLL_SECONDS)
        waited += _DESKTOP_LOCK_POLL_SECONDS
    try:
        return await asyncio.wait_for(
            _open_desktop_locked(
                owner,
                mounts,
                api_key,
                redis,
                key,
                user_id=user_id,
                session_id=session_id,
            ),
            timeout=_DESKTOP_OPEN_DEADLINE_SECONDS,
        )
    finally:
        await redis.eval(_UNLOCK_SCRIPT, 1, lock_key, token)


async def _open_desktop_locked(
    owner: SandboxOwner,
    mounts: Mapping[str, str],
    api_key: str,
    redis: Any,
    key: str,
    *,
    user_id: Optional[str],
    session_id: Optional[str],
) -> tuple[DesktopStream, bool, bool]:
    raw = await redis.get(key)
    sandbox_id = raw.decode() if isinstance(raw, bytes) else raw
    if not sandbox_id:
        # An expert's desktop outlives the Redis cache; E2B metadata is the record.
        sandbox_id = await find_owned_sandbox_id(owner, "desktop", api_key)
    if sandbox_id:
        desktop = await _reconnect_desktop(sandbox_id, api_key, redis, key)
        if desktop is not None:
            # From here on a failure is a real error on a live box, not a
            # reason to abandon it and create another.
            await desktop.ensure_display(*_DESKTOP_RESOLUTION)
            await redis.set(key, sandbox_id, ex=owner.ttl)
            stream = await desktop.start_stream()
            return stream, False, await desktop.is_workspace_mounted()

    desktop, persistence = await DesktopSession.create(
        api_key=api_key,
        timeout_seconds=chat_config.e2b_desktop_timeout,
        width=_DESKTOP_RESOLUTION[0],
        height=_DESKTOP_RESOLUTION[1],
        volume_mounts=dict(mounts) or None,
        template=chat_config.e2b_desktop_template,
        metadata=owner.creation_metadata(
            "desktop",
            user_id=user_id,
            session_id=session_id,
            template=chat_config.e2b_desktop_template,
            mounts="attached" if mounts else "none",
        ),
    )
    try:
        await redis.set(key, desktop.sandbox_id, ex=owner.ttl)
    except Exception:
        if not owner.is_expert:
            # Nothing else can find a session desktop: no metadata recovery,
            # no archive path.  Kill it rather than bill it until timeout.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(desktop.kill(), timeout=_KILL_TIMEOUT_SECONDS)
        raise
    stream = await desktop.start_stream()
    return stream, True, persistence.volume_mounted


async def _reconnect_desktop(
    sandbox_id: str, api_key: str, redis: Any, key: str
) -> Optional[DesktopSession]:
    """Reattach to a cached or recovered desktop, or ``None`` if it is gone."""
    try:
        return await DesktopSession.connect(
            sandbox_id, api_key, timeout_seconds=chat_config.e2b_desktop_timeout
        )
    except Exception as exc:
        logger.warning("[E2B] Desktop %.12s reconnect failed: %s", sandbox_id, exc)
        await redis.delete(key)
        return None


def mounts_for(user_id: Optional[str], expert_id: Optional[str]) -> dict[str, str]:
    """The desktop mounts exactly what the owner's shell mounts.

    Same rule as ``workspace_volume_mounts``: an expert always gets its own
    home, the user's shared volume only when there is a user.
    """
    return workspace_volume_mounts(user_id, expert_id)
