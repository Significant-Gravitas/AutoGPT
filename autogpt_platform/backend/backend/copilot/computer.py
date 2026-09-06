"""An owner's computer on E2B: what exists, and how to turn its screen on.

One box per owner: the same sandbox ``bash_exec`` runs in, on our desktop
image (``backend.util.e2b_template``).  Nothing graphical runs until someone
asks for the screen.  ``open_desktop`` starts X, XFCE and the VNC stream *in
that box* and hands back the live stream, whether the ask comes from the
``start_desktop`` tool inside a turn or from the Computer tab and side
panel.  The box pauses at turn end like any other and comes back with the
screen exactly as it was.

``describe_computer`` only lists: it never connects, so a paused box stays
paused (connecting is what E2B's auto-resume reacts to).  Whether the screen
is on is remembered in Redis next to the box id, because asking the box
would wake it.
"""

import logging
from datetime import datetime
from typing import Literal, Mapping, Optional

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
    SandboxOwner,
    get_or_create_owner_sandbox,
    list_owned_sandboxes,
)
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_DESKTOP_RESOLUTION = (1280, 720)


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
    boxes = await list_owned_sandboxes(owner, api_key)
    if not boxes:
        return info
    box = boxes[0]
    info.box = SandboxSummary(
        sandbox_id=box.sandbox_id,
        state="running" if box.state.value == "running" else "paused",
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
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> tuple[DesktopStream, bool, bool]:
    """Turn the screen on in the owner's box and return ``(stream, first_time, shared)``.

    Finds or creates the owner's box the same way a turn does, then starts
    the display and stream inside it.  Opening from outside a turn does not
    count as one, so the turn-end pause still fires when the agent finishes;
    an idle box opened from the UI is paused by the lifecycle timeout instead.
    *user_id* / *session_id* are provenance only, stamped on a newly created box.
    """
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
    stream = await desktop.start_stream()
    await _remember_screen(owner, sandbox.sandbox_id)
    return stream, not was_on, shared


async def screen_is_on(owner: SandboxOwner, sandbox_id: str) -> bool:
    """Whether ``open_desktop`` has run on *this* box (a replaced box starts off)."""
    redis = await get_redis_async()
    raw = await redis.get(owner.display_key())
    value = raw.decode() if isinstance(raw, bytes) else raw
    return value == sandbox_id


async def _remember_screen(owner: SandboxOwner, sandbox_id: str) -> None:
    redis = await get_redis_async()
    await redis.set(owner.display_key(), sandbox_id, ex=owner.ttl)


def mounts_for(user_id: Optional[str], expert_id: Optional[str]) -> dict[str, str]:
    return workspace_volume_mounts(user_id, expert_id) if user_id else {}
