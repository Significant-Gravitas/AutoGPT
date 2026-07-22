"""On-demand interactive desktop for CoPilot sessions (E2B).

The session's bash sandbox stays on the cheap headless ``base`` template —
E2B bills actual vCPU+RAM per running second, so putting every session on the
GUI-capable ``desktop`` template would tax the majority of sessions that
never need a screen. Instead this tool manages a SEPARATE desktop-template
sandbox per session: created on first use, kept running after the turn ends
so the user can watch and control the live stream, auto-paused when its idle
timeout lapses (paused sandboxes cost nothing and are kept indefinitely),
and resumed in about a second on the next call.

E2B has no cross-sandbox mounts (volumes are in private beta), so the
session workspace is tar-copied INTO the desktop once at creation — a
snapshot, not a live share. The response says so explicitly so the model
does not assume bash_exec and the desktop see the same files.
"""

import logging
from typing import Any, Optional

from e2b import AsyncSandbox

from backend.blocks.desktop._api import DesktopSession
from backend.copilot.context import E2B_WORKDIR, get_current_sandbox
from backend.copilot.model import ChatSession
from backend.copilot.sdk.env import config as chat_config
from backend.data.redis_client import get_redis_async

from .base import BaseTool
from .models import DesktopStreamToolResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_DESKTOP_KEY_PREFIX = "copilot:e2b:desktop:"
_DESKTOP_ID_TTL = 48 * 3600
_SYNC_LIMIT_BYTES = 50 * 1024 * 1024
_SYNC_TAR = "/tmp/copilot_workspace_sync.tgz"
_SESSION_COPY_DIR = f"{E2B_WORKDIR}/session"
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
            "desktop is a separate machine from bash_exec: your session files "
            "are copied to ~/session on it when it is first created, but "
            "changes are not synced back. It keeps running after your turn "
            "ends and suspends itself when idle. Use when a task needs a "
            "browser or GUI app the user should see."
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

        try:
            desktop, created = await _get_or_create_desktop(session_id, api_key)
            synced = False
            if created:
                synced = await _sync_workspace(get_current_sandbox(), desktop)
            stream = await desktop.start_stream()
        except Exception as exc:
            logger.error("[E2B] start_desktop failed: %s", exc, exc_info=True)
            return ErrorResponse(
                message=f"Failed to start the desktop: {exc}",
                error="desktop_start_failed",
                session_id=session_id,
            )

        return DesktopStreamToolResponse(
            message=_build_message(created, synced),
            desktop_stream=stream.model_dump(),
            session_id=session_id,
        )


def _build_message(created: bool, synced: bool) -> str:
    state = "started" if created else "resumed"
    if created and synced:
        files = f"Session files were copied to {_SESSION_COPY_DIR} (one-time snapshot, not live-synced with bash_exec)."
    elif created:
        files = "Session files were NOT copied over (no session sandbox or workspace too large); the desktop starts empty."
    else:
        files = "It still has the files from when it was last used; nothing new was synced from the session."
    return (
        f"Desktop {state}. The user can watch and control it via the live "
        f"stream. It stays up after this turn and suspends itself when idle; "
        f"calling start_desktop again resumes it with state intact. {files}"
    )


async def _get_or_create_desktop(
    session_id: str, api_key: str
) -> tuple[DesktopSession, bool]:
    """Return (desktop, created) — resuming the session's desktop if one exists."""
    redis = await get_redis_async()
    key = f"{_DESKTOP_KEY_PREFIX}{session_id}"
    raw = await redis.get(key)
    sandbox_id = raw.decode() if isinstance(raw, bytes) else raw
    if sandbox_id:
        try:
            desktop = await DesktopSession.connect(sandbox_id, api_key)
            await desktop.ensure_display(*_DESKTOP_RESOLUTION)
            await redis.set(key, sandbox_id, ex=_DESKTOP_ID_TTL)
            return desktop, False
        except Exception as exc:
            logger.warning("[E2B] Desktop %.12s reconnect failed: %s", sandbox_id, exc)
            await redis.delete(key)

    desktop, _ = await DesktopSession.create(
        api_key=api_key,
        timeout_seconds=chat_config.e2b_desktop_timeout,
        width=_DESKTOP_RESOLUTION[0],
        height=_DESKTOP_RESOLUTION[1],
        volume_name=None,
        template=chat_config.e2b_desktop_template,
    )
    await redis.set(key, desktop.sandbox_id, ex=_DESKTOP_ID_TTL)
    return desktop, True


async def _sync_workspace(
    session_sandbox: Optional[AsyncSandbox], desktop: DesktopSession
) -> bool:
    """Best-effort one-shot copy of the session workspace into the desktop."""
    if session_sandbox is None:
        return False
    try:
        result = await session_sandbox.commands.run(
            f"tar czf {_SYNC_TAR} -C {E2B_WORKDIR} "
            "--exclude=node_modules --exclude=.git --exclude=__pycache__ "
            f"--exclude={_SYNC_TAR.split('/')[-1]} . && stat -c%s {_SYNC_TAR}"
        )
        size = int(result.stdout.strip().splitlines()[-1])
        if size > _SYNC_LIMIT_BYTES:
            logger.warning("[E2B] Workspace sync skipped: %d bytes > limit", size)
            return False
        data = await session_sandbox.files.read(_SYNC_TAR, format="bytes")
        await desktop.sandbox.files.write(_SYNC_TAR, bytes(data))
        await desktop.run_command(
            f"mkdir -p {_SESSION_COPY_DIR} && "
            f"tar xzf {_SYNC_TAR} -C {_SESSION_COPY_DIR} && rm -f {_SYNC_TAR}"
        )
        return True
    except Exception as exc:
        logger.warning("[E2B] Workspace sync into desktop failed: %s", exc)
        return False
    finally:
        try:
            await session_sandbox.commands.run(f"rm -f {_SYNC_TAR}")
        except Exception:
            pass
