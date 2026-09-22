"""An owner's computer on E2B: what exists, and how to turn its screen on.

One box per owner: the same sandbox ``bash_exec`` runs in, on our desktop
image (``backend.util.e2b_template``).  Nothing graphical runs until someone
asks for the screen.  ``open_desktop`` starts X, XFCE and the VNC stream *in
that box* and hands back the live stream, whether the ask comes from the
``start_desktop`` tool inside a turn or from the Computer tab and side
panel.  The box pauses at turn end like any other and comes back with the
screen exactly as it was, but under a fresh stream password: the password
is kept here, off the box, and when the turn-end pause comes the stream is
stopped and the password forgotten, so a stream URL that may have leaked
stops working there.  "Screen on" therefore means the display was started
in this box; the next open is what serves it again.

One pause is out of our hands: a box left idle is paused by E2B's own
timeout with its stream still up, and any request to the stream URL resumes
it.  That stream is stopped at our next connect to the box
(``e2b_sandbox._settle_stream``), not before.

``describe_computer`` only lists: it never connects, so a paused box stays
paused (connecting is what E2B's auto-resume reacts to).  Whether the screen
is on is remembered in Redis next to the box id, because asking the box
would wake it.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Literal, Mapping, Optional

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
    SandboxLookupError,
    SandboxOwner,
    get_or_create_owner_sandbox,
    list_owned_sandboxes,
)
from backend.data.redis_client import get_redis_async
from backend.util.desktop_preview import create_preview_link

logger = logging.getLogger(__name__)

_DESKTOP_RESOLUTION = (1280, 720)

# Turning the screen on can be a box creation (volume resolution, up to three
# create attempts), the display coming up and the home setup: about four
# minutes at the very worst.  The open is cut off before the lock can lapse,
# so a second opener can never slip in under a still-running first one, and
# it waits about as long.
_DESKTOP_LOCK_TTL_SECONDS = 300
_DESKTOP_OPEN_DEADLINE_SECONDS = _DESKTOP_LOCK_TTL_SECONDS - 15
_DESKTOP_LOCK_WAIT_SECONDS = 300
_DESKTOP_LOCK_POLL_SECONDS = 0.5
_UNLOCK_SCRIPT = (
    'if redis.call("get", KEYS[1]) == ARGV[1] then '
    'return redis.call("del", KEYS[1]) else return 0 end'
)


class SandboxSummary(BaseModel):
    """The owner's box as E2B reports it, without waking it."""

    sandbox_id: str
    state: Literal["running", "paused"]
    started_at: datetime
    cpu_count: int
    memory_mb: int
    template_id: str
    mounts_attached: bool


class ComputerInfo(BaseModel):
    """The owner's computer: its one box, whether its screen is on, its volumes."""

    owner_kind: Literal["session", "expert"]
    owner_id: str
    e2b_active: bool
    box: Optional[SandboxSummary] = None
    screen_on: bool = False
    # path -> volume name; empty when the owner has no volumes (no user).
    mounts: dict[str, str] = {}
    workspace_path: str = WORKSPACE_PATH
    shared_path: str = SHARED_PATH


def computer_owner(session_id: str, expert_id: Optional[str]) -> SandboxOwner:
    return SandboxOwner.for_session(session_id, expert_id)


async def describe_computer(
    owner: SandboxOwner, mounts: Mapping[str, str]
) -> ComputerInfo:
    """List the owner's box and screen state without resuming anything."""
    api_key = chat_config.active_e2b_api_key
    info = ComputerInfo(
        owner_kind=owner.kind,
        owner_id=owner.id,
        e2b_active=api_key is not None,
        mounts=dict(mounts),
    )
    if not api_key:
        return info
    try:
        boxes = await list_owned_sandboxes(owner, api_key)
    except SandboxLookupError as exc:
        # A listing is informational; show nothing rather than fail the page.
        logger.warning("[E2B] describe_computer: %s", exc)
        return info
    if not boxes:
        return info
    box = boxes[0]
    info.box = SandboxSummary(
        sandbox_id=box.sandbox_id,
        state="running" if box.state == SandboxState.RUNNING else "paused",
        started_at=box.started_at,
        cpu_count=box.cpu_count,
        memory_mb=box.memory_mb,
        template_id=box.template_id,
        mounts_attached=(box.metadata or {}).get(METADATA_MOUNTS) == "attached",
    )
    info.screen_on = await screen_is_on(owner, box.sandbox_id)
    return info


async def open_desktop(
    owner: SandboxOwner,
    mounts: Mapping[str, str],
    api_key: str,
    *,
    user_id: Optional[str],
    session_id: Optional[str] = None,
) -> tuple[DesktopStream, bool, bool]:
    """Turn the screen on in the owner's box; return ``(stream, first_time, shared)``.

    Finds or creates the owner's box the same way a turn does, then starts
    the display and stream inside it.  Opening from outside a turn does not
    count as one, so the turn-end pause still fires when the agent finishes;
    an idle box opened from the UI is paused by the lifecycle timeout instead.
    *user_id* is who the stream link is issued to (see ``_owner_bound``) and,
    with *session_id*, provenance stamped on a newly created box.

    One opener at a time per owner: the panel's button and the model's
    ``start_desktop`` (or two tabs) would otherwise race to start a second
    display and VNC stack in the same box.  A second opener waits for the
    first and then reuses its stream.
    """
    if not user_id:
        raise ValueError("A desktop needs an authenticated user to issue its link to")
    redis = await get_redis_async()
    lock_key = owner.display_lock_key()
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
                owner, mounts, api_key, user_id=user_id, session_id=session_id
            ),
            timeout=_DESKTOP_OPEN_DEADLINE_SECONDS,
        )
    finally:
        await redis.eval(_UNLOCK_SCRIPT, 1, lock_key, token)


async def _open_desktop_locked(
    owner: SandboxOwner,
    mounts: Mapping[str, str],
    api_key: str,
    *,
    user_id: str,
    session_id: Optional[str],
) -> tuple[DesktopStream, bool, bool]:
    sandbox = await get_or_create_owner_sandbox(
        owner,
        api_key,
        timeout=chat_config.e2b_sandbox_timeout,
        template=chat_config.e2b_sandbox_template,
        on_timeout=chat_config.e2b_sandbox_on_timeout,
        volume_mounts=mounts,
        user_id=user_id,
        session_id=session_id,
        count_turn=False,
    )
    was_on = await screen_is_on(owner, sandbox.sandbox_id)
    desktop = DesktopSession(sandbox)
    await desktop.ensure_display(*_DESKTOP_RESOLUTION)
    shared = await desktop.is_workspace_mounted()
    if shared and not was_on:
        # Browser downloads and saved files land in the durable home.
        await desktop.ensure_persistent_home()
    # The password issued last time, if the box has run without a pause
    # since; otherwise the stack restarts under a new one.
    stream, password = await desktop.start_stream(
        await _stream_password(owner) if was_on else None
    )
    await _remember_screen(owner, sandbox.sandbox_id, password)
    return _owner_bound(stream, user_id), not was_on, shared


async def screen_is_on(owner: SandboxOwner, sandbox_id: str) -> bool:
    """Whether ``open_desktop`` has run on *this* box (a replaced box starts off)."""
    redis = await get_redis_async()
    raw = await redis.get(owner.display_key())
    value = raw.decode() if isinstance(raw, bytes) else raw
    return value == sandbox_id


async def _stream_password(owner: SandboxOwner) -> Optional[str]:
    redis = await get_redis_async()
    raw = await redis.get(owner.stream_key())
    value = raw.decode() if isinstance(raw, bytes) else raw
    return value or None


async def _remember_screen(owner: SandboxOwner, sandbox_id: str, password: str) -> None:
    """Record the screen as on in this box, and the password its stream uses.

    The password is remembered only as long as the box could have kept
    running: its expiry is the box's running-time limit, pushed out again
    whenever a connect re-arms that limit (``e2b_sandbox._settle_stream``),
    and the turn-end pause stops the stream and drops it outright
    (``e2b_sandbox._revoke_stream``).
    """
    redis = await get_redis_async()
    await redis.set(owner.display_key(), sandbox_id, ex=owner.ttl)
    await redis.set(owner.stream_key(), password, ex=chat_config.e2b_sandbox_timeout)


def _owner_bound(stream: DesktopStream, user_id: str) -> DesktopStream:
    """The stream with its URL replaced by an owner-bound link.

    The real URL carries the desktop's password.  It stays on the backend;
    what the tool result, the chat message and the API hand out is a link
    to ``/api/desktop-preview`` that only *user_id* can redeem.
    """
    return stream.model_copy(
        update={"url": create_preview_link(user_id, stream.url), "requires_auth": True}
    )


def mounts_for(user_id: Optional[str], expert_id: Optional[str]) -> dict[str, str]:
    """The screen is on the same box as the shell, so it mounts what the shell mounts.

    Same rule as ``workspace_volume_mounts``: an expert always gets its own
    home, the user's shared volume only when there is a user.
    """
    return workspace_volume_mounts(user_id, expert_id)
