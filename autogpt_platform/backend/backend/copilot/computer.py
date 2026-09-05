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
    SandboxKind,
    SandboxOwner,
    find_owned_sandbox_id,
    list_owned_sandboxes,
)
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_DESKTOP_RESOLUTION = (1280, 720)


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
    for kind in ("shell", "desktop"):
        boxes = await list_owned_sandboxes(owner, kind, api_key)
        if not boxes:
            continue
        box = boxes[0]
        summary = SandboxSummary(
            kind=kind,
            sandbox_id=box.sandbox_id,
            state="running" if box.state.value == "running" else "paused",
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
    owner: SandboxOwner, mounts: Mapping[str, str], api_key: str
) -> tuple[DesktopStream, bool, bool]:
    """Return ``(stream, created, shared)`` — resuming the owner's desktop if it exists.

    Shared by the ``start_desktop`` tool and the HTTP endpoints, so a desktop
    opened from the expert page is the same box the expert's next turn finds.
    """
    redis = await get_redis_async()
    key = owner.key("desktop")
    raw = await redis.get(key)
    sandbox_id = raw.decode() if isinstance(raw, bytes) else raw
    if not sandbox_id:
        # An expert's desktop outlives the Redis cache; E2B metadata is the record.
        sandbox_id = await find_owned_sandbox_id(owner, "desktop", api_key)
    if sandbox_id:
        try:
            desktop = await DesktopSession.connect(sandbox_id, api_key)
            await desktop.ensure_display(*_DESKTOP_RESOLUTION)
            await redis.set(key, sandbox_id, ex=owner.ttl)
            stream = await desktop.start_stream()
            return stream, False, await desktop.is_workspace_mounted()
        except Exception as exc:
            logger.warning("[E2B] Desktop %.12s reconnect failed: %s", sandbox_id, exc)
            await redis.delete(key)

    desktop, persistence = await DesktopSession.create(
        api_key=api_key,
        timeout_seconds=chat_config.e2b_desktop_timeout,
        width=_DESKTOP_RESOLUTION[0],
        height=_DESKTOP_RESOLUTION[1],
        volume_mounts=dict(mounts) or None,
        template=chat_config.e2b_desktop_template,
        metadata=owner.metadata("desktop"),
    )
    await redis.set(key, desktop.sandbox_id, ex=owner.ttl)
    stream = await desktop.start_stream()
    return stream, True, persistence.volume_mounted


def mounts_for(user_id: Optional[str], expert_id: Optional[str]) -> dict[str, str]:
    return workspace_volume_mounts(user_id, expert_id) if user_id else {}
