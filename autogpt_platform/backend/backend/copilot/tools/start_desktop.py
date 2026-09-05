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
delegation and scheduled run that happens as that expert.  The expert page's
Computer tab and the copilot side panel open the very same box through
``backend.copilot.computer``.

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

from backend.blocks.desktop._api import SHARED_PATH, WORKSPACE_PATH
from backend.copilot.computer import computer_owner, mounts_for, open_desktop
from backend.copilot.model import ChatSession
from backend.copilot.sdk.env import config as chat_config

from .base import BaseTool
from .models import DesktopStreamToolResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class StartDesktopTool(BaseTool):
    """Start (or resume) the owner's on-demand desktop and return its stream."""

    @property
    def name(self) -> str:
        return "start_desktop"

    @property
    def description(self) -> str:
        return (
            "Start or resume an interactive cloud desktop and return a live "
            "stream the user can watch and control. It shares your persistent "
            "~/workspace with bash_exec (in an expert session it is the expert's "
            "own persistent computer; ~/shared is the user's workspace). Stays "
            "up after your turn, suspends when idle. Use when a task needs a "
            "browser or GUI app."
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
        owner = computer_owner(session_id, expert_id)
        try:
            stream, created, shared = await open_desktop(
                owner, mounts_for(user_id, expert_id), api_key
            )
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
