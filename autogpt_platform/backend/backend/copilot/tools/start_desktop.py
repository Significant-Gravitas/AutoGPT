"""On-demand interactive desktop for CoPilot sessions (E2B).

The session's bash sandbox stays on the cheap headless ``base`` template —
E2B bills actual vCPU+RAM per running second, so putting every session on the
GUI-capable ``desktop`` template would tax the majority of sessions that never
need a screen. Instead this tool manages a SEPARATE desktop-template sandbox
per session: created on first use, kept running after the turn ends so the
user can watch and control the live stream, auto-paused when idle (free), and
resumed in about a second on the next call.

Both the desktop and the session's bash sandbox mount the SAME per-user volume
at ``/home/user/workspace`` (see ``get_or_create_sandbox``), so the desktop and
the agent's shell share one durable, live workspace — files the agent writes to
``~/workspace`` appear on the desktop and vice versa, and both survive across
sessions. The desktop additionally redirects Downloads/Desktop/Documents into
the volume, so a person's browser downloads and saved files persist too.
"""

import logging
from typing import Any

from backend.blocks.desktop._api import WORKSPACE_PATH, DesktopSession
from backend.blocks.desktop._common import user_volume_name
from backend.copilot.model import ChatSession
from backend.copilot.sdk.env import config as chat_config
from backend.data.redis_client import get_redis_async

from .base import BaseTool
from .models import DesktopStreamToolResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_DESKTOP_KEY_PREFIX = "copilot:e2b:desktop:"
_DESKTOP_ID_TTL = 48 * 3600
_DESKTOP_RESOLUTION = (1280, 720)


class StartDesktopTool(BaseTool):
    """Start (or resume) the session's on-demand desktop and return its stream."""

    @property
    def name(self) -> str:
        return "start_desktop"

    @property
    def description(self) -> str:
        return (
            "Start (or resume) an interactive cloud desktop for this session "
            "and return a live stream the user can watch and control. The "
            "desktop shares your persistent ~/workspace with bash_exec — files "
            "you write there appear on the desktop live and vice versa, and "
            "persist across sessions. It keeps running after your turn and "
            "suspends itself when idle. Use when a task needs a browser or GUI "
            "app the user should see."
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

        volume_name = user_volume_name(user_id) if user_id else None
        try:
            desktop, created, shared = await _get_or_create_desktop(
                session_id, api_key, volume_name
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
            message=_build_message(created, shared),
            desktop_stream=stream.model_dump(),
            session_id=session_id,
        )


def _build_message(created: bool, shared: bool) -> str:
    state = "started" if created else "resumed"
    if shared:
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
    session_id: str, api_key: str, volume_name: str | None
) -> tuple[DesktopSession, bool, bool]:
    """Return (desktop, created, shared) — resuming the session's desktop if one exists."""
    redis = await get_redis_async()
    key = f"{_DESKTOP_KEY_PREFIX}{session_id}"
    raw = await redis.get(key)
    sandbox_id = raw.decode() if isinstance(raw, bytes) else raw
    if sandbox_id:
        try:
            desktop = await DesktopSession.connect(sandbox_id, api_key)
            await desktop.ensure_display(*_DESKTOP_RESOLUTION)
            await redis.set(key, sandbox_id, ex=_DESKTOP_ID_TTL)
            return desktop, False, await desktop.is_workspace_mounted()
        except Exception as exc:
            logger.warning("[E2B] Desktop %.12s reconnect failed: %s", sandbox_id, exc)
            await redis.delete(key)

    desktop, persistence = await DesktopSession.create(
        api_key=api_key,
        timeout_seconds=chat_config.e2b_desktop_timeout,
        width=_DESKTOP_RESOLUTION[0],
        height=_DESKTOP_RESOLUTION[1],
        volume_name=volume_name,
        template=chat_config.e2b_desktop_template,
    )
    await redis.set(key, desktop.sandbox_id, ex=_DESKTOP_ID_TTL)
    return desktop, True, persistence.volume_mounted
