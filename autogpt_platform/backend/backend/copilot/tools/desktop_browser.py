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
from backend.copilot.gate.policy import Effect
from backend.copilot.gate.subject import Subject
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
    "answer": {
        "type": "string",
        "enum": ["accept", "dismiss"],
        "description": "dialog the page put up",
    },
}


class DesktopBrowserTool(BaseTool):
    """Act in the desktop's browser, the one the user watches, by refs from
    its accessibility snapshot."""

    # ``look`` writes a screenshot to the workspace, which another call in the
    # same message may list or read.
    writes_files = True
    has_gate_subject = True

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
            "repeating; not_started page_changed means take a new snapshot. "
            "look saves a screenshot to the workspace. A confirm or prompt the "
            "page puts up blocks it until a dialog action answers it."
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

    async def gate_subject(
        self, user_id: str, session: ChatSession, args: dict[str, Any]
    ) -> Subject | None:
        """A call that only reads the page, or goes to one, runs like the
        other browser reads; anything that types, clicks or answers asks."""
        actions = args.get("actions")
        if (
            isinstance(actions, list)
            and actions
            and all(
                isinstance(a, dict) and a.get("action") in driver.READ_ACTIONS
                for a in actions
            )
        ):
            return Subject(
                key="desktop_browser:read",
                name="the desktop browser",
                effect=Effect.READ,
            )
        return None

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
                live = await driver.live_scope(sandbox.sandbox_id)
                parsed = driver.parse_call(actions, ref_scope, live)
                return await self._run(
                    user_id, session, sandbox, parsed, (live or {}).get("doc")
                )
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
        doc: str | None,
    ) -> ToolResponseBase:
        look_path = f"/tmp/otto-look-{secrets.token_hex(6)}.png"
        # Each ref action first checks the page is the one its snapshot read.
        groups = [
            driver.guarded(a, driver.commands_for(a, look_path), doc) for a in actions
        ]
        # Refused here, before the browser is touched.
        driver.batch_payload([c for group in groups for c in group])
        session_name = await driver.ensure_browser(sandbox)
        if actions[0].name != "dialog":
            dialog = await driver.open_dialog(sandbox, session_name)
            if dialog is not None:
                return self._blocked(session, actions, dialog)
        receipts, acted = await self._act(sandbox, session_name, actions, groups)
        stamp = driver.new_doc_stamp()
        observation, observed = await self._observe(
            sandbox, session_name, actions, stamp
        )
        if observation is None:
            # A snapshot-only call ran no command of its own: it did what it
            # was asked only if the snapshot came back.
            for receipt in receipts:
                if receipt.action == "snapshot" and receipt.status == "completed":
                    receipt.status, receipt.reason = "unknown", "no_snapshot"
        # Every older ref_scope is void now, whether or not this one came back.
        try:
            token = await driver.issue_scope(
                sandbox.sandbox_id,
                observation.refs if observation else [],
                # Unstamped, the snapshot's refs stand on ref_scope alone: a
                # check against a stamp the page never got would refuse them all.
                stamp if observation and observation.stamped else None,
            )
        except Exception:
            # The receipts still go back; with no scope, the next ref action
            # is refused until a snapshot issues one.
            logger.warning("[desktop_browser] could not issue a scope", exc_info=True)
            token = None
        file_id = await self._save_look(
            user_id, session, sandbox, actions, receipts, look_path
        )
        notes = driver.page_notes(acted, observed)
        ran = sum(r.status == "completed" for r in receipts)
        stopped = next((r for r in receipts if r.status != "completed"), None)
        message = f"Ran {ran} of {len(receipts)} actions."
        if stopped:
            message += (
                f" Stopped at {stopped.action}: {stopped.status} ({stopped.reason})."
            )
        if observation is None:
            message += (
                " The page could not be read after the call; see page_notes, then "
                "take a snapshot."
            )
        if notes:
            message += " The page reported something in page_notes."
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
            ref_scope=token if observation else None,
            url=observation.url if observation else "",
            title=observation.title if observation else "",
            receipts=receipt_dicts,
            page_notes=notes,
            snapshot=observation.snapshot if observation else "",
            screenshot_file_id=file_id,
            session_id=session.session_id,
        ).from_outside(*from_page, *notes)

    def _blocked(
        self, session: ChatSession, actions: list[driver.Action], dialog: str
    ) -> DesktopBrowserResponse:
        """A call that cannot run while the page waits on a dialog: nothing in
        it started, and the old scope still stands."""
        return DesktopBrowserResponse(
            message=(
                "Nothing ran: a dialog is open on the page (see page_notes). "
                "Answer it with a dialog action, or ask the user to answer it on "
                "the desktop."
            ),
            receipts=[
                driver.Receipt(
                    action=a.name, status="not_started", reason="dialog_open"
                ).as_dict()
                for a in actions
            ],
            page_notes=[dialog],
            session_id=session.session_id,
        ).from_outside(dialog)

    async def _act(
        self,
        sandbox: Any,
        session_name: str,
        actions: list[driver.Action],
        groups: list[list[list[str]]],
    ) -> tuple[list[driver.Receipt], list[dict]]:
        """Receipts for the call's batch, and its entries. A batch that failed
        without telling what ran (a transport error, the time limit) leaves
        every action ``unknown``: any of them may have run."""
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
            ], []
        return driver.receipts_for(actions, groups, entries), entries

    async def _observe(
        self,
        sandbox: Any,
        session_name: str,
        actions: list[driver.Action],
        stamp: str,
    ) -> tuple[driver.Observation | None, list[dict]]:
        """The page after the call, and the closing read's entries (a
        dialog's warning, or why it could not be read)."""
        dialog = await driver.open_dialog(sandbox, session_name)
        if dialog is not None:
            # Script in the page would wait on the dialog: no snapshot now.
            return None, [{"success": False, "error": dialog}]
        commands = driver.observe_commands(actions, stamp)
        settled = commands[0][:2] == ["wait", "--load"]
        try:
            entries = await driver.run_batch(
                sandbox, session_name, commands, bail=False, timeout=45
            )
        except LoginChainChanged as refused:
            # The batch has run by now: its receipts still go back, and the
            # caller voids every older ref_scope, as for any unread page.
            logger.warning("[desktop_browser] the page was not read: %s", refused)
            return None, [{"success": False, "error": str(refused)}]
        except Exception:
            logger.warning("[desktop_browser] could not read the page", exc_info=True)
            return None, []
        return (
            driver.observation_from(entries, settled),
            entries[(1 if settled else 0) :],
        )

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
            if any(a.name == "look" for a in actions):
                await driver.remove_file(sandbox, look_path)
            return None
        try:
            return await self._store_look(user_id, session, sandbox, look, look_path)
        finally:
            await driver.remove_file(sandbox, look_path)

    async def _store_look(
        self,
        user_id: str | None,
        session: ChatSession,
        sandbox: Any,
        look: driver.Receipt,
        look_path: str,
    ) -> str | None:
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
