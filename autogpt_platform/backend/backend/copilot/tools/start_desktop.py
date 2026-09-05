"""On-demand interactive desktop for CoPilot sessions (E2B).

The session's bash sandbox stays on the cheap headless ``base`` template —
E2B bills actual vCPU+RAM per running second, so putting every session on the
GUI-capable ``desktop`` template would tax the majority of sessions that never
need a screen. Instead this tool manages a SEPARATE desktop-template sandbox
per owner: created on first use, kept running after the turn ends so the
user can watch and control the live stream, auto-paused when idle (free), and
resumed in about a second on the next call.

The owner is the session — or, in an expert session, the expert itself, so
the desktop is the expert's own persistent computer: the same box (browser
profile, logins, installed apps and all) comes back for every chat,
delegation and scheduled run that happens as that expert.

Both the desktop and the owner's bash sandbox mount the SAME durable volumes
(see ``workspace_volume_mounts``): ``/home/user/workspace`` is the owner's
persistent home and, for an expert, ``/home/user/shared`` is the owning
user's workspace — so files the agent writes appear on the desktop and vice
versa, and both survive across sessions. The desktop additionally redirects
Downloads/Desktop/Documents into the home volume, so a person's browser
downloads and saved files persist too.
"""

import logging
from typing import Any

from backend.blocks.desktop._api import SHARED_PATH, WORKSPACE_PATH, DesktopSession
from backend.blocks.desktop._common import workspace_volume_mounts
from backend.copilot.model import ChatSession
from backend.copilot.sdk.env import config as chat_config
from backend.copilot.tools.e2b_sandbox import SandboxOwner, find_owned_sandbox_id
from backend.data.redis_client import get_redis_async

from .base import BaseTool
from .models import DesktopStreamToolResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_DESKTOP_RESOLUTION = (1280, 720)


class StartDesktopTool(BaseTool):
    """Start (or resume) the owner's on-demand desktop and return its stream."""

    @property
    def name(self) -> str:
        return "start_desktop"

    @property
    def description(self) -> str:
        return (
            "Start (or resume) an interactive cloud desktop and return a live "
            "stream the user can watch and control. The desktop shares your "
            "persistent ~/workspace with bash_exec — files you write there "
            "appear on the desktop live and vice versa, and persist across "
            "sessions. In an expert session the desktop is the expert's own "
            "persistent computer (~/workspace is its home, ~/shared is the "
            "user's shared workspace, and browser logins persist). It keeps "
            "running after your turn and suspends itself when idle. Use when a "
            "task needs a browser or GUI app the user should see."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not session_id:
            return ErrorResponse(
                message="No active session to attach a desktop to.",
                error="session_unavailable",
            )
        api_key = chat_config.active_e2b_api_key
        if not api_key:
            return ErrorResponse(
                message="E2B is not configured, so a desktop cannot be started.",
                error="e2b_unconfigured",
                session_id=session_id,
            )

        expert_id = session.expert_id if session else None
        owner = SandboxOwner.for_session(session_id, expert_id)
        mounts = workspace_volume_mounts(user_id, expert_id) if user_id else {}
        try:
            desktop, created, shared = await _get_or_create_desktop(
                owner, api_key, mounts
            )
            stream = await desktop.start_stream()
        except Exception as exc:
            logger.error("[E2B] start_desktop failed: %s", exc, exc_info=True)
            return ErrorResponse(
                message=f"Failed to start the desktop: {exc}",
                error="desktop_start_failed",
                session_id=session_id,
            )

        return DesktopStreamToolResponse(
            message=_build_message(created, shared, expert=owner.is_expert),
            desktop_stream=stream.model_dump(),
            session_id=session_id,
        )


def _build_message(created: bool, shared: bool, *, expert: bool = False) -> str:
    state = "started" if created else "resumed"
    if shared and expert:
        files = (
            f"This is your own persistent computer. {WORKSPACE_PATH} is your "
            "durable home — customise it freely; installed tools and browser "
            f"logins survive between sessions — and {SHARED_PATH} is the user's "
            "shared workspace, so put anything the user should see there."
        )
    elif shared:
        files = (
            f"It shares your persistent {WORKSPACE_PATH} with bash_exec live, "
            "so put anything the user should see (or that should persist) there."
        )
    else:
        files = (
            "No per-user volume is attached (no user context), so this desktop "
            "is ephemeral and not shared with bash_exec."
        )
    return (
        f"Desktop {state}. The user can watch and control it via the live "
        f"stream. It stays up after this turn and suspends itself when idle; "
        f"calling start_desktop again resumes it with state intact. {files}"
    )


async def _get_or_create_desktop(
    owner: SandboxOwner, api_key: str, mounts: dict[str, str]
) -> tuple[DesktopSession, bool, bool]:
    """Return (desktop, created, shared) — resuming the owner's desktop if one exists."""
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
            return desktop, False, await desktop.is_workspace_mounted()
        except Exception as exc:
            logger.warning("[E2B] Desktop %.12s reconnect failed: %s", sandbox_id, exc)
            await redis.delete(key)

    desktop, persistence = await DesktopSession.create(
        api_key=api_key,
        timeout_seconds=chat_config.e2b_desktop_timeout,
        width=_DESKTOP_RESOLUTION[0],
        height=_DESKTOP_RESOLUTION[1],
        volume_mounts=mounts or None,
        template=chat_config.e2b_desktop_template,
        metadata=owner.metadata("desktop"),
    )
    await redis.set(key, desktop.sandbox_id, ex=owner.ttl)
    return desktop, True, persistence.volume_mounted
