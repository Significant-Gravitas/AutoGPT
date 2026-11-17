"""Drive the browser on the session's desktop through a fixed set of actions.

The protocol and how it runs are in ``backend.copilot.desktop_browser``; this
is the tool around it: the checks before a call, the screenshot a ``look``
saves to the user's workspace, and the response the model reads.
"""

import base64
import logging
import secrets
from datetime import UTC, datetime
from typing import Any

from backend.copilot import desktop_browser as driver
from backend.copilot.context import get_current_sandbox, is_local_pc_executor
from backend.copilot.model import ChatSession
from backend.copilot.sdk.env import config as chat_config
from backend.util.sandbox_login import LoginChainChanged

from .base import BaseTool
from .models import DesktopBrowserResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_ACTION_PROPERTIES: dict[str, Any] = {
    "action": {"type": "string", "enum": sorted(driver.ACTIONS)},
    "url": {"type": "string", "description": "goto"},
    "ref": {"type": "string", "description": "@eN from the snapshot"},
    "kind": {"type": "string", "enum": ["left", "double"], "description": "click"},
    "text": {"type": "string", "description": "fill replaces, type appends; wait"},
    "text_gone": {"type": "string", "description": "wait"},
    "key": {"type": "string", "description": "press, e.g. Enter"},
    "value": {"type": "string", "description": "select"},
    "full_page": {"type": "boolean", "description": "look"},
    "time_ms": {"type": "integer", "description": "wait: only a delay the page states"},
    "timeout_ms": {"type": "integer", "description": "wait for text"},
}


class DesktopBrowserTool(BaseTool):
    """Act in the desktop's browser, the one the user watches, by refs from
    its accessibility snapshot."""

    # ``look`` writes a screenshot to the workspace, which another call in the
    # same message may list or read.
    writes_files = True

    @property
    def name(self) -> str:
        return "desktop_browser"

    @property
    def description(self) -> str:
        return (
            "Drive the Chrome on your desktop, which the user sees (start_desktop "
            "first): an ordered list of actions, stopped at the first failure. "
            "Each call returns url, title and a snapshot whose @eN refs work only "
            "with the ref_scope it returns; no refs after a navigation in the "
            "same call. Receipt unknown means it may have run: check before "
            "repeating. look saves a screenshot to the workspace."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": "Actions, run in order",
                    "maxItems": driver.MAX_ACTIONS,
                    "items": {
                        "type": "object",
                        "properties": _ACTION_PROPERTIES,
                        "required": ["action"],
                    },
                },
                "ref_scope": {
                    "type": "string",
                    "description": "From the last call; needed to use its refs",
                },
            },
            "required": ["actions"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        # The desktop is an E2B box; without E2B there is nothing to drive.
        return bool(chat_config.active_e2b_api_key)

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        actions: Any = None,
        ref_scope: Any = None,
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        sandbox = get_current_sandbox()
        if sandbox is None or is_local_pc_executor(sandbox) or not session_id:
            return ErrorResponse(
                message="There is no cloud computer in this chat to drive a browser on.",
                error="no_desktop",
                session_id=session_id,
            )
        # Imported here: ``computer`` imports the sandbox module, whose
        # package imports this tool.
        from backend.copilot.computer import computer_owner, screen_is_on

        owner = computer_owner(session_id, session.expert_id)
        if not await screen_is_on(owner, sandbox.sandbox_id):
            return ErrorResponse(
                message="The desktop is off. Call start_desktop first, then retry.",
                error="desktop_off",
                session_id=session_id,
            )
        try:
            async with driver.call_lock(sandbox.sandbox_id, driver.LOCK_HOLD_SECONDS):
                # Checked under the lock: the snapshot it is checked against
                # is still the newest when the call runs.
                parsed = driver.parse_call(
                    actions, ref_scope, await driver.live_scope(sandbox.sandbox_id)
                )
                return await self._run(user_id, session, sandbox, parsed)
        except driver.CallRefused as refused:
            return ErrorResponse(
                message=f"Nothing ran: {refused}",
                error="call_refused",
                session_id=session_id,
            )
        except driver.BrowserBusy as busy:
            return ErrorResponse(
                message=f"Nothing ran: {busy}. Retry when it has finished.",
                error="browser_busy",
                session_id=session_id,
            )
        except (driver.BrowserUnavailable, LoginChainChanged) as error:
            return ErrorResponse(
                message=f"The desktop browser is not available: {error}",
                error="browser_unavailable",
                session_id=session_id,
            )

    async def _run(
        self,
        user_id: str | None,
        session: ChatSession,
        sandbox: Any,
        actions: list[driver.Action],
    ) -> ToolResponseBase:
        session_name = await driver.ensure_browser(sandbox)
        look_path = f"/tmp/otto-look-{secrets.token_hex(6)}.png"
        groups = [driver.commands_for(a, look_path) for a in actions]
        receipts = await self._act(sandbox, session_name, actions, groups)
        observation = await self._observe(sandbox, session_name)
        if observation is None:
            # A snapshot-only call ran no command of its own: it did what it
            # was asked only if the snapshot came back.
            for receipt in receipts:
                if receipt.action == "snapshot" and receipt.status == "completed":
                    receipt.status, receipt.reason = "unknown", "no_snapshot"
        # Every older ref_scope is void now, whether or not this one came back.
        token = await driver.issue_scope(
            sandbox.sandbox_id, observation.refs if observation else []
        )
        file_id = await self._save_look(
            user_id, session, sandbox, actions, receipts, look_path
        )
        ran = sum(r.status == "completed" for r in receipts)
        stopped = next((r for r in receipts if r.status != "completed"), None)
        message = f"Ran {ran} of {len(receipts)} actions."
        if stopped:
            message += (
                f" Stopped at {stopped.action}: {stopped.status} ({stopped.reason})."
            )
        if observation is None:
            message += " The page could not be read after the call; take a snapshot."
        receipt_dicts = [r.as_dict() for r in receipts]
        # What the page said: its snapshot, title and URL, and what actions
        # read from it or failed with. The content screen judges these.
        from_page = [
            value
            for r in receipts
            for key, value in r.result.items()
            if key in ("text", "url", "title", "error")
        ]
        if observation:
            from_page += [observation.snapshot, observation.title, observation.url]
        return DesktopBrowserResponse(
            message=message,
            receipts=receipt_dicts,
            url=observation.url if observation else "",
            title=observation.title if observation else "",
            ref_scope=token if observation else None,
            snapshot=observation.snapshot if observation else "",
            screenshot_file_id=file_id,
            session_id=session.session_id,
        ).from_outside(*from_page)

    async def _act(
        self,
        sandbox: Any,
        session_name: str,
        actions: list[driver.Action],
        groups: list[list[list[str]]],
    ) -> list[driver.Receipt]:
        """Receipts for the call's batch. A batch that failed without telling
        what ran (a transport error, the time limit) leaves every action
        ``unknown``: any of them may have run."""
        try:
            entries = await driver.run_batch(
                sandbox,
                session_name,
                [c for group in groups for c in group],
                bail=True,
                timeout=driver.call_seconds(actions),
            )
        except LoginChainChanged:
            raise
        except Exception:
            logger.warning(
                "[desktop_browser] the batch returned nothing", exc_info=True
            )
            return [
                driver.Receipt(action=a.name, status="unknown", reason="no_result")
                for a in actions
            ]
        return driver.receipts_for(actions, groups, entries)

    async def _observe(
        self, sandbox: Any, session_name: str
    ) -> driver.Observation | None:
        try:
            return driver.observation_from(
                await driver.run_batch(
                    sandbox, session_name, driver.OBSERVE, bail=False, timeout=30
                )
            )
        except LoginChainChanged:
            raise
        except Exception:
            logger.warning("[desktop_browser] could not read the page", exc_info=True)
            return None

    async def _save_look(
        self,
        user_id: str | None,
        session: ChatSession,
        sandbox: Any,
        actions: list[driver.Action],
        receipts: list[driver.Receipt],
        look_path: str,
    ) -> str | None:
        """The workspace file id of the call's screenshot, if it took one."""
        look = next(
            (
                r
                for a, r in zip(actions, receipts)
                if a.name == "look" and r.status == "completed"
            ),
            None,
        )
        if look is None:
            return None
        try:
            data = await driver.read_screenshot(sandbox, look_path)
        except Exception:
            logger.warning(
                "[desktop_browser] could not read the screenshot", exc_info=True
            )
            look.result["error"] = "the screenshot could not be read"
            return None
        if data is None:
            look.result["error"] = "screenshot over the size limit"
            return None
        # Imported here: workspace_files imports from .models.
        from .workspace_files import WorkspaceWriteResponse, WriteWorkspaceFileTool

        filename = (
            f"desktop-{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}.png"
        )
        try:
            written = await WriteWorkspaceFileTool()._execute(
                user_id=user_id,
                session=session,
                filename=filename,
                content_base64=base64.b64encode(data).decode(),
            )
        except Exception:
            logger.warning(
                "[desktop_browser] could not save %s", filename, exc_info=True
            )
            written = None
        if not isinstance(written, WorkspaceWriteResponse):
            logger.warning("[desktop_browser] could not save %s", filename)
            look.result["error"] = "screenshot could not be saved to the workspace"
            return None
        look.result["file_id"] = written.file_id
        look.result["filename"] = filename
        return written.file_id
