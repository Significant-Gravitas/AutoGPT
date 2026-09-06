"""Turn the screen on in the session's computer (E2B) and stream it.

There is one box per owner — the session, or in an expert session the
expert itself — and it is the same sandbox ``bash_exec`` runs in.  It runs
our desktop image with nothing graphical started, so a screen costs nothing
until this tool starts X, XFCE and the VNC stream *in that box*
(``backend.copilot.computer.open_desktop``).  The expert page's Computer tab
and the copilot side panel open the very same screen.

Everything is shared because it is one machine: files, processes, installed
tools, and the durable volumes (``/home/user/workspace`` is the owner's
persistent home; for an expert ``/home/user/shared`` is the owning user's
workspace).  The box pauses at turn end like always and the screen comes
back exactly as it was on the next resume.
"""

import logging
from typing import Any

from backend.blocks.desktop._api import SHARED_PATH, WORKSPACE_PATH
from backend.copilot.computer import computer_owner, mounts_for, open_desktop
from backend.copilot.model import ChatSession
from backend.copilot.sdk.env import config as chat_config

from .base import BaseTool
from .models import DesktopStreamToolResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class StartDesktopTool(BaseTool):
    """Turn on the screen of the session's computer and return its live stream."""

    @property
    def name(self) -> str:
        return "start_desktop"

    @property
    def description(self) -> str:
        return (
            "Turn on the screen of your computer and return a live stream the "
            "user can watch and control. It is the same machine bash_exec runs "
            "in, so files, ~/workspace and running processes are shared (in an "
            "expert session it is the expert's own persistent computer; "
            "~/shared is the user's workspace). The machine suspends between "
            "turns and the screen comes back as you left it. Use when a task "
            "needs a browser or GUI app."
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
                message="No active session to turn a screen on for.",
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
        owner = computer_owner(session_id, expert_id)
        try:
            stream, first_time, shared = await open_desktop(
                owner,
                mounts_for(user_id, expert_id),
                api_key,
                user_id=user_id,
                session_id=session_id,
            )
        except Exception as exc:
            logger.error("[E2B] start_desktop failed: %s", exc, exc_info=True)
            return ErrorResponse(
                message=f"Failed to start the desktop: {exc}",
                error="desktop_start_failed",
                session_id=session_id,
            )

        return DesktopStreamToolResponse(
            message=_build_message(first_time, shared, expert=owner.is_expert),
            desktop_stream=stream.model_dump(),
            session_id=session_id,
        )


def _build_message(first_time: bool, shared: bool, *, expert: bool = False) -> str:
    state = "Screen is on" if first_time else "Screen was already on; stream refreshed"
    if shared and expert:
        files = (
            f"This is your own persistent computer. {WORKSPACE_PATH} is your "
            "durable home — customise it freely; installed tools and browser "
            f"logins survive between sessions — and {SHARED_PATH} is the user's "
            "shared workspace, so put anything the user should see there."
        )
    elif shared:
        files = (
            f"Your persistent {WORKSPACE_PATH} is on it, so put anything the "
            "user should see (or that should persist) there."
        )
    else:
        files = (
            "No per-user volume is attached (no user context), so nothing on "
            "this machine persists beyond the sandbox."
        )
    return (
        f"{state}. This is the same machine bash_exec runs in, so files, "
        "processes and installed tools are shared. The user can watch and "
        "control it via the live stream. The machine suspends between turns "
        f"and the screen comes back as you left it. {files}"
    )
