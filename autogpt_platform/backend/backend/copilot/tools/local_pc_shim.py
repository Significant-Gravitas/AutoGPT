"""
Platform-side binding for the autogpt-local-executor shim.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from .local_pc_errors import translate_shim_error
from .local_pc_metrics import record_rpc_retry
from .recording_models import RecordingSummary, TrajectoryStep, WorkflowRecording

logger = logging.getLogger(__name__)


# In-flight-on-disconnect semantics — see PROTOCOL.md
# "In-flight semantics on disconnect". Ops in this set are repeat-safe at the
# protocol level: re-issuing the same wire op produces the same observable
# state on the user's machine, so the platform adapter MAY auto-retry once
# after reconnect when a timeout/disconnect races a response.
_IDEMPOTENT_OPS: frozenset[str] = frozenset(
    {
        "FILE_READ",
        "FILE_STAT",
        "FILE_LIST",
        "CURSOR_POSITION_REQUEST",
        "DISPLAY_INFO_REQUEST",
        "WINDOW_LIST_REQUEST",
        "APP_LIST_REQUEST",
        "CLIPBOARD_READ",
        "PERMISSIONS_CHECK_REQUEST",
        "SCREENSHOT_REQUEST",
    }
)


class OpUnconfirmedError(RuntimeError):
    """Raised when a non-idempotent wire op was sent but never acknowledged.

    The wire op left the platform but the platform-side `_rpc` either timed
    out waiting for a response or the WS dropped before the shim flushed
    one. The side effect may or may not have happened on the user's
    machine, so the platform MUST NOT auto-retry — the LLM owns the
    recovery decision (typically: probe state with an idempotent op like
    `FILE_STAT`, then re-issue if needed).

    Attributes:
        code: synthetic shim error code surfaced to the translator
            (``"WRITE_UNCONFIRMED"`` for FILE_WRITE, ``"OP_UNCONFIRMED"``
            for everything else).
        op: the wire op name (``"FILE_WRITE"``, ``"EXECUTE_COMMAND"``, ...).
        wire_id: the original wire-correlation `id` so callers can correlate
            audit-log entries on both sides.
    """

    def __init__(
        self,
        op: str,
        wire_id: str,
        *,
        code: str = "OP_UNCONFIRMED",
        message: str | None = None,
    ) -> None:
        super().__init__(message or f"[LocalPC] {op} unconfirmed (wire id={wire_id})")
        self.code = code
        self.op = op
        self.wire_id = wire_id


class WriteUnconfirmedError(OpUnconfirmedError):
    """Specialization of OpUnconfirmedError for FILE_WRITE.

    The bytes may or may not have hit disk on the shim host. Caller should
    `FILE_STAT` the target path to check actual state.
    """

    def __init__(self, wire_id: str, message: str | None = None) -> None:
        super().__init__(
            op="FILE_WRITE",
            wire_id=wire_id,
            code="WRITE_UNCONFIRMED",
            message=message,
        )


# Backpressure — see PROTOCOL.md §Concurrency. When the platform asks
# `_rpc` to send a wire op but the shim's most recent
# `pending_capacity` signal is 0, the call blocks on
# ``_capacity_available`` for at most this many seconds before raising
# ``ShimOverloadedError``. Matches the wire-level ``SHIM_OVERLOADED``
# semantic so the translator can render the same recovery hint either way.
_CAPACITY_WAIT_TIMEOUT_SECONDS: float = 30.0


class ShimOverloadedError(RuntimeError):
    """Raised proactively when the shim's pending capacity stays at 0 too long.

    Wire-level ``SHIM_OVERLOADED`` arrives as a normal `ERROR` envelope
    after the shim refuses an over-cap request. This client-side variant
    short-circuits before sending — once the platform learns the shim is
    full (from a prior response or STATUS frame), there's no value in
    spending a round-trip just to receive `SHIM_OVERLOADED` back. The
    error code matches the wire code so existing translator + retry logic
    handles both surfaces uniformly.
    """

    code = "SHIM_OVERLOADED"


def _friendly(payload: dict, shim: "LocalPCShim | None", fallback: str) -> str:
    """Translate a wire ERROR payload into an actionable English message."""
    if not isinstance(payload, dict):
        return fallback
    return translate_shim_error(
        payload.get("code", "INTERNAL_ERROR"),
        payload.get("message", "") or fallback,
        payload.get("details"),
        shim,
    )


def _translate_unconfirmed(
    exc: OpUnconfirmedError,
    shim: "LocalPCShim | None",
    *,
    fallback: str,
    extra_details: dict | None = None,
) -> str:
    """Build the LLM-friendly message for an OpUnconfirmedError.

    Shared by every non-idempotent proxy method so the catch+wrap shape
    in CommandsProxy / FilesProxy.delete+move / ComputerProxy mirrors
    FilesProxy.write — the LLM sees the same actionable English ("the
    op was sent but the connection dropped before the shim acknowledged
    — verify state with an idempotent probe and re-issue if needed")
    regardless of which proxy raised.
    """
    details: dict = {"op": exc.op}
    if extra_details:
        details.update(extra_details)
    return _friendly(
        {"code": exc.code, "message": str(exc), "details": details},
        shim,
        fallback,
    )


_shim_manager: "ShimConnectionManager | None" = None


def get_shim_manager() -> "ShimConnectionManager":
    global _shim_manager
    if _shim_manager is None:
        _shim_manager = ShimConnectionManager()
    return _shim_manager


@dataclass
class ShimHello:
    """HELLO payload captured by the route on connect, surfaced to LocalPCShim."""

    machine_id: str = ""
    platform: str = ""
    arch: str = ""
    shim_version: str = ""
    allowed_root: str = ""
    capabilities: list[str] = field(default_factory=list)
    screen_resolution: tuple[int, int] | None = None
    local_llm_models: list[str] = field(default_factory=list)
    hardware_devices: list[dict] = field(default_factory=list)
    computer_use_features: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict) -> "ShimHello":
        sr = payload.get("screen_resolution")
        return cls(
            machine_id=payload.get("machine_id", ""),
            platform=payload.get("platform", ""),
            arch=payload.get("arch", ""),
            shim_version=payload.get("shim_version", ""),
            allowed_root=payload.get("allowed_root", ""),
            capabilities=list(payload.get("capabilities") or []),
            screen_resolution=(
                tuple(sr) if isinstance(sr, (list, tuple)) and len(sr) == 2 else None
            ),
            local_llm_models=list(payload.get("local_llm_models") or []),
            hardware_devices=list(payload.get("hardware_devices") or []),
            computer_use_features=list(payload.get("computer_use_features") or []),
        )


class ShimConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._hellos: dict[str, ShimHello] = {}
        self._waiters: dict[str, list[asyncio.Future[WebSocket]]] = {}
        # (user_id, client_id) -> set[session_id]. Lets revoke_user_shims
        # find every active shim belonging to a user+app without scanning
        # the full connection dict.
        self._by_owner: dict[tuple[str | None, str | None], set[str]] = {}
        # Reverse index for fast unregister cleanup.
        self._owner_of: dict[str, tuple[str | None, str | None]] = {}

    def register(
        self,
        session_id: str,
        ws: WebSocket,
        hello: ShimHello | None = None,
        *,
        user_id: str | None = None,
        client_id: str | None = None,
    ) -> None:
        self._connections[session_id] = ws
        if hello is not None:
            self._hellos[session_id] = hello
        owner = (user_id, client_id)
        self._owner_of[session_id] = owner
        self._by_owner.setdefault(owner, set()).add(session_id)
        for fut in self._waiters.pop(session_id, []):
            if not fut.done():
                fut.set_result(ws)
        logger.info("[LocalPC] Shim registered for session %s", session_id[:12])

    def unregister(self, session_id: str) -> None:
        self._connections.pop(session_id, None)
        self._hellos.pop(session_id, None)
        owner = self._owner_of.pop(session_id, None)
        if owner is not None:
            sessions = self._by_owner.get(owner)
            if sessions is not None:
                sessions.discard(session_id)
                if not sessions:
                    self._by_owner.pop(owner, None)
        logger.info("[LocalPC] Shim unregistered for session %s", session_id[:12])

    async def revoke_user_shims(
        self,
        user_id: str,
        client_id: str | None,
        *,
        reason: str = "user_revoked",
    ) -> int:
        """Push SESSION_REVOKED to every shim owned by (user_id, client_id).

        Returns the count of shims actually notified. Called by
        ``/auth/revoke`` after a successful token revocation so the user's
        connected shims tear down their WS without waiting for the next
        op to 401. The SESSION_REVOKED frame shape mirrors the shim's
        ``protocol.SessionRevokedPayload`` — see PROTOCOL.md Session
        ownership section. The shim daemon audits the event, closes the
        WS, does NOT auto-reconnect.

        ``client_id=None`` means "all of this user's shims, any app."
        Used for platform-level revocations (e.g. account-wide kill
        switch) rather than per-OAuth-token revocations.
        """
        if client_id is not None:
            target_owners: list[tuple[str | None, str | None]] = [(user_id, client_id)]
        else:
            target_owners = [k for k in self._by_owner if k[0] == user_id]

        notified = 0
        for owner in target_owners:
            # Snapshot — sending SESSION_REVOKED eventually closes the WS,
            # which calls unregister, which mutates _by_owner.
            for session_id in list(self._by_owner.get(owner, ())):
                ws = self._connections.get(session_id)
                if ws is None:
                    continue
                envelope = {
                    "type": "SESSION_REVOKED",
                    "id": str(uuid.uuid4()),
                    "ts": time.time(),
                    "payload": {"reason": reason},
                }
                try:
                    await ws.send_text(json.dumps(envelope))
                    notified += 1
                except Exception:
                    logger.debug(
                        "[LocalPC] SESSION_REVOKED send failed for session %s "
                        "(ws likely already closed)",
                        session_id[:12],
                    )
                # Close ourselves; 4428 disables shim auto-reconnect per spec.
                try:
                    await ws.close(code=4428, reason="Token revoked")
                except Exception:
                    pass
        if notified:
            logger.info(
                "[LocalPC] Revoked %d shim session(s) for user %s app %s",
                notified,
                user_id[:12] if user_id else "?",
                client_id or "*",
            )
        return notified

    async def wait_for(self, session_id: str, timeout: float = 30.0) -> WebSocket:
        if session_id in self._connections:
            return self._connections[session_id]
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[WebSocket] = loop.create_future()
        self._waiters.setdefault(session_id, []).append(fut)
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"[LocalPC] Shim for session {session_id[:12]} did not connect within {timeout}s"
            )

    def get(self, session_id: str) -> WebSocket | None:
        return self._connections.get(session_id)

    def get_hello(self, session_id: str) -> ShimHello | None:
        return self._hellos.get(session_id)


class _FilesProxy:
    def __init__(self, shim: "LocalPCShim") -> None:
        self._shim = shim

    async def read(self, path: str, *, format: str = "text") -> str | bytes:
        wire_encoding = "base64" if format == "bytes" else "utf-8"
        resp = await self._shim._rpc(
            "FILE_READ", {"path": path, "encoding": wire_encoding}
        )
        if resp.get("type") == "ERROR":
            raise OSError(
                _friendly(resp.get("payload", {}), self._shim, "FILE_READ failed")
            )
        content = resp["payload"]["content"]
        if format == "bytes":
            return base64.b64decode(content)
        return content

    async def write(self, path: str, content: str | bytes) -> None:
        if isinstance(content, (bytes, bytearray, memoryview)):
            wire_content = base64.b64encode(bytes(content)).decode("ascii")
            wire_encoding = "base64"
        else:
            wire_content = content
            wire_encoding = "utf-8"
        try:
            resp = await self._shim._rpc(
                "FILE_WRITE",
                {
                    "path": path,
                    "content": wire_content,
                    "encoding": wire_encoding,
                    "create_parents": True,
                },
            )
        except WriteUnconfirmedError as exc:
            # Synthesize an ERROR envelope so the LLM sees the translator's
            # WRITE_UNCONFIRMED hint, not the raw Python exception text.
            raise OSError(
                _friendly(
                    {
                        "code": "WRITE_UNCONFIRMED",
                        "message": str(exc),
                        "details": {"path": path, "op": "FILE_WRITE"},
                    },
                    self._shim,
                    "FILE_WRITE unconfirmed",
                )
            ) from exc
        if resp.get("type") == "ERROR":
            raise OSError(
                _friendly(resp.get("payload", {}), self._shim, "FILE_WRITE failed")
            )

    async def stat(self, path: str, *, follow_symlinks: bool = True) -> dict:
        """Cross-OS portable replacement for shell `stat` / `readlink -f` / `test -e`."""
        resp = await self._shim._rpc(
            "FILE_STAT", {"path": path, "follow_symlinks": follow_symlinks}
        )
        if resp.get("type") == "ERROR":
            raise OSError(
                _friendly(resp.get("payload", {}), self._shim, "FILE_STAT failed")
            )
        return resp["payload"]

    async def list(
        self,
        path: str,
        *,
        glob: str | None = None,
        recursive: bool = False,
        include_hidden: bool = False,
        max_entries: int = 1000,
    ) -> dict:
        """Cross-OS portable replacement for shell `ls` / `find`."""
        resp = await self._shim._rpc(
            "FILE_LIST",
            {
                "path": path,
                "glob": glob,
                "recursive": recursive,
                "include_hidden": include_hidden,
                "max_entries": max_entries,
            },
        )
        if resp.get("type") == "ERROR":
            raise OSError(
                _friendly(resp.get("payload", {}), self._shim, "FILE_LIST failed")
            )
        return resp["payload"]

    async def delete(
        self, path: str, *, recursive: bool = False, missing_ok: bool = False
    ) -> None:
        """Cross-OS portable replacement for shell `rm` / `del`."""
        try:
            resp = await self._shim._rpc(
                "FILE_DELETE",
                {"path": path, "recursive": recursive, "missing_ok": missing_ok},
            )
        except OpUnconfirmedError as exc:
            raise OSError(
                _translate_unconfirmed(
                    exc,
                    self._shim,
                    fallback="FILE_DELETE unconfirmed",
                    extra_details={"path": path},
                )
            ) from exc
        if resp.get("type") == "ERROR":
            raise OSError(
                _friendly(resp.get("payload", {}), self._shim, "FILE_DELETE failed")
            )

    async def move(self, src: str, dst: str, *, overwrite: bool = False) -> None:
        """Cross-OS portable replacement for shell `mv` / `move`."""
        try:
            resp = await self._shim._rpc(
                "FILE_MOVE",
                {"src": src, "dst": dst, "overwrite": overwrite},
            )
        except OpUnconfirmedError as exc:
            raise OSError(
                _translate_unconfirmed(
                    exc,
                    self._shim,
                    fallback="FILE_MOVE unconfirmed",
                    extra_details={"src": src, "dst": dst},
                )
            ) from exc
        if resp.get("type") == "ERROR":
            raise OSError(
                _friendly(resp.get("payload", {}), self._shim, "FILE_MOVE failed")
            )


class ShimComputerUseError(RuntimeError):
    """Raised when a computer-use wire op returns a structured ERROR.

    ``code`` mirrors the wire ``payload.code`` so MCP-tool handlers can
    branch on the structured surface defined in COMPUTER_USE.md
    (PERMISSION_PENDING, FEATURE_NOT_SUPPORTED, WINDOW_STALE,
    INPUT_OUT_OF_BOUNDS, CLIPBOARD_CONCEALED, ...) without parsing the
    human ``message`` string. ``message`` is the already-LLM-friendly text
    produced by :mod:`local_pc_errors`.
    """

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _raise_computer_use(resp: dict, shim: "LocalPCShim | None", fallback: str) -> None:
    """Translate a wire ERROR response into a typed ShimComputerUseError."""
    if resp.get("type") != "ERROR":
        return
    payload = resp.get("payload") or {}
    raise ShimComputerUseError(
        code=str(payload.get("code") or "INTERNAL_ERROR"),
        message=_friendly(payload, shim, fallback),
        details=(
            payload.get("details") if isinstance(payload.get("details"), dict) else {}
        ),
    )


class _ComputerProxy:
    """Wire-op wrapper for the shim's computer-use surface.

    Mirrors the spec in
    ``experimental/local-pc-executor/docs/COMPUTER_USE.md``. Each method
    sends one wire op and returns the parsed payload (or raises
    :class:`ShimComputerUseError` on a structured ERROR response). The
    MCP-tool layer in ``tool_adapter.py`` translates the typed error to a
    text block Claude can read.
    """

    def __init__(self, shim: "LocalPCShim") -> None:
        self._shim = shim

    # --- Screenshot --------------------------------------------------------

    async def screenshot(
        self,
        *,
        monitor: int = 0,
        region: list[int] | tuple[int, int, int, int] | None = None,
        window_id: str | None = None,
        format: str = "png",
        include_cursor: bool = False,
        quality: int = 75,
    ) -> dict:
        if region is not None and window_id is not None:
            raise ValueError(
                "LocalPCShim.computer.screenshot: region and window_id are mutually exclusive"
            )
        payload: dict[str, Any] = {
            "monitor": monitor,
            "quality": quality,
            "format": format,
            "include_cursor": include_cursor,
        }
        if region is not None:
            payload["region"] = list(region)
        if window_id is not None:
            payload["window_id"] = window_id
        resp = await self._shim._rpc("SCREENSHOT_REQUEST", payload)
        _raise_computer_use(resp, self._shim, "SCREENSHOT_REQUEST failed")
        return resp.get("payload") or {}

    # --- INPUT_ACTION verbs ------------------------------------------------

    async def _input(self, action: str, **fields: Any) -> dict:
        payload: dict[str, Any] = {"action": action}
        for k, v in fields.items():
            if v is not None:
                payload[k] = v
        try:
            resp = await self._shim._rpc("INPUT_ACTION", payload)
        except OpUnconfirmedError as exc:
            # INPUT_ACTION is non-idempotent (clicks at the same coord
            # are NOT the same op — the OS might have a different element
            # under the cursor on retry). Translate to the structured
            # OP_UNCONFIRMED shape so Claude knows to take a screenshot
            # and re-evaluate rather than blindly re-clicking.
            raise ShimComputerUseError(
                code=exc.code,
                message=_translate_unconfirmed(
                    exc,
                    self._shim,
                    fallback=f"INPUT_ACTION {action} unconfirmed",
                    extra_details={"action": action},
                ),
                details={"action": action, "op": exc.op},
            ) from exc
        _raise_computer_use(resp, self._shim, f"INPUT_ACTION {action} failed")
        return resp.get("payload") or {}

    async def click(
        self,
        coordinate: list[int] | tuple[int, int],
        *,
        button: str = "left",
        modifiers: list[str] | None = None,
    ) -> None:
        action = {
            "left": "left_click",
            "right": "right_click",
            "middle": "middle_click",
        }.get(button, "left_click")
        await self._input(
            action,
            coordinate=list(coordinate),
            button=button,
            modifiers=modifiers,
        )

    async def double_click(
        self,
        coordinate: list[int] | tuple[int, int],
        *,
        modifiers: list[str] | None = None,
    ) -> None:
        await self._input(
            "double_click", coordinate=list(coordinate), modifiers=modifiers
        )

    async def triple_click(
        self,
        coordinate: list[int] | tuple[int, int],
        *,
        modifiers: list[str] | None = None,
    ) -> None:
        await self._input(
            "triple_click", coordinate=list(coordinate), modifiers=modifiers
        )

    async def middle_click(self, coordinate: list[int] | tuple[int, int]) -> None:
        await self._input("middle_click", coordinate=list(coordinate))

    async def mouse_move(self, coordinate: list[int] | tuple[int, int]) -> None:
        await self._input("mouse_move", coordinate=list(coordinate))

    async def mouse_down(
        self,
        coordinate: list[int] | tuple[int, int],
        *,
        button: str = "left",
    ) -> None:
        await self._input("mouse_down", coordinate=list(coordinate), button=button)

    async def mouse_up(
        self,
        coordinate: list[int] | tuple[int, int],
        *,
        button: str = "left",
    ) -> None:
        await self._input("mouse_up", coordinate=list(coordinate), button=button)

    async def drag(
        self,
        path: list[list[int]] | list[tuple[int, int]],
        *,
        button: str = "left",
        duration_ms: int | None = None,
    ) -> None:
        await self._input(
            "drag",
            path=[list(pt) for pt in path],
            button=button,
            duration_ms=duration_ms,
        )

    async def scroll(
        self,
        coordinate: list[int] | tuple[int, int],
        *,
        direction: str = "down",
        scroll_amount: int = 1,
        modifiers: list[str] | None = None,
    ) -> None:
        await self._input(
            "scroll",
            coordinate=list(coordinate),
            scroll_direction=direction,
            scroll_amount=scroll_amount,
            modifiers=modifiers,
        )

    async def type(
        self,
        text: str,
        *,
        paste: bool = False,
        preserve_clipboard: bool = False,
    ) -> None:
        await self._input(
            "type",
            text=text,
            paste=paste,
            preserve_clipboard=preserve_clipboard,
        )

    async def key(self, key: str) -> None:
        await self._input("key", key=key)

    async def hold_key(self, key: str, duration_ms: int) -> None:
        await self._input("hold_key", key=key, duration_ms=duration_ms)

    async def wait(self, duration_ms: int) -> None:
        await self._input("wait", duration_ms=duration_ms)

    # --- Cursor / display --------------------------------------------------

    async def cursor_position(self) -> dict:
        resp = await self._shim._rpc("CURSOR_POSITION_REQUEST", {})
        _raise_computer_use(resp, self._shim, "CURSOR_POSITION_REQUEST failed")
        return resp.get("payload") or {}

    async def display_info(self) -> dict:
        resp = await self._shim._rpc("DISPLAY_INFO_REQUEST", {})
        _raise_computer_use(resp, self._shim, "DISPLAY_INFO_REQUEST failed")
        return resp.get("payload") or {}

    # --- Windows -----------------------------------------------------------

    async def list_windows(
        self,
        *,
        app_bundle_id: str | None = None,
        include_minimized: bool = False,
        include_offscreen: bool = False,
    ) -> list[dict]:
        resp = await self._shim._rpc(
            "WINDOW_LIST_REQUEST",
            {
                "app_bundle_id": app_bundle_id,
                "include_minimized": include_minimized,
                "include_offscreen": include_offscreen,
            },
        )
        _raise_computer_use(resp, self._shim, "WINDOW_LIST_REQUEST failed")
        return list((resp.get("payload") or {}).get("windows") or [])

    async def focus_window(self, window_id: str, *, raise_: bool = True) -> None:
        try:
            resp = await self._shim._rpc(
                "WINDOW_FOCUS", {"window_id": window_id, "raise": raise_}
            )
        except OpUnconfirmedError as exc:
            raise ShimComputerUseError(
                code=exc.code,
                message=_translate_unconfirmed(
                    exc,
                    self._shim,
                    fallback="WINDOW_FOCUS unconfirmed",
                    extra_details={"window_id": window_id},
                ),
                details={"window_id": window_id, "op": exc.op},
            ) from exc
        _raise_computer_use(resp, self._shim, "WINDOW_FOCUS failed")

    # --- Apps --------------------------------------------------------------

    async def list_apps(self, *, include_background: bool = False) -> list[dict]:
        resp = await self._shim._rpc(
            "APP_LIST_REQUEST", {"include_background": include_background}
        )
        _raise_computer_use(resp, self._shim, "APP_LIST_REQUEST failed")
        return list((resp.get("payload") or {}).get("apps") or [])

    async def launch_app(
        self,
        *,
        bundle_id: str | None = None,
        executable_path: str | None = None,
        args: list[str] | None = None,
        activate: bool = True,
    ) -> dict:
        if not bundle_id and not executable_path:
            raise ValueError(
                "LocalPCShim.computer.launch_app: bundle_id or executable_path is required"
            )
        try:
            resp = await self._shim._rpc(
                "APP_LAUNCH",
                {
                    "bundle_id": bundle_id,
                    "executable_path": executable_path,
                    "args": list(args or []),
                    "activate": activate,
                },
            )
        except OpUnconfirmedError as exc:
            raise ShimComputerUseError(
                code=exc.code,
                message=_translate_unconfirmed(
                    exc,
                    self._shim,
                    fallback="APP_LAUNCH unconfirmed",
                    extra_details={
                        "bundle_id": bundle_id,
                        "executable_path": executable_path,
                    },
                ),
                details={"op": exc.op},
            ) from exc
        _raise_computer_use(resp, self._shim, "APP_LAUNCH failed")
        return resp.get("payload") or {}

    # --- Clipboard ---------------------------------------------------------

    async def clipboard_read(self, *, format: str = "text") -> str | None:
        resp = await self._shim._rpc("CLIPBOARD_READ", {"format": format})
        _raise_computer_use(resp, self._shim, "CLIPBOARD_READ failed")
        return (resp.get("payload") or {}).get("content")

    async def clipboard_write(self, content: str, *, format: str = "text") -> None:
        try:
            resp = await self._shim._rpc(
                "CLIPBOARD_WRITE", {"format": format, "content": content}
            )
        except OpUnconfirmedError as exc:
            raise ShimComputerUseError(
                code=exc.code,
                message=_translate_unconfirmed(
                    exc,
                    self._shim,
                    fallback="CLIPBOARD_WRITE unconfirmed",
                    extra_details={"content_length": len(content)},
                ),
                details={"op": exc.op},
            ) from exc
        _raise_computer_use(resp, self._shim, "CLIPBOARD_WRITE failed")

    # --- Permissions -------------------------------------------------------

    async def permissions_check(self, permissions: list[str] | None = None) -> dict:
        resp = await self._shim._rpc(
            "PERMISSIONS_CHECK_REQUEST",
            {
                "permissions": permissions
                or ["screen_recording", "accessibility", "input_monitoring"]
            },
        )
        _raise_computer_use(resp, self._shim, "PERMISSIONS_CHECK_REQUEST failed")
        return (resp.get("payload") or {}).get("permissions") or {}


class _CommandsProxy:
    def __init__(self, shim: "LocalPCShim") -> None:
        self._shim = shim

    async def run(
        self,
        command: str = "",
        *,
        argv: list[str] | None = None,
        shell: str = "auto",
        cwd: str | None = None,
        timeout: int | None = None,
        envs: dict[str, str] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {}
        if argv is not None:
            payload["argv"] = argv
        elif command:
            payload["command"] = command
            payload["shell"] = shell
        else:
            raise ValueError(
                "LocalPCShim.commands.run: either command or argv must be set"
            )
        if cwd:
            payload["cwd"] = cwd
        if timeout:
            payload["timeout_seconds"] = timeout
        if envs:
            payload["env"] = envs
        try:
            resp = await self._shim._rpc("EXECUTE_COMMAND", payload)
        except OpUnconfirmedError as exc:
            raise RuntimeError(
                _translate_unconfirmed(
                    exc,
                    self._shim,
                    fallback="EXECUTE_COMMAND unconfirmed",
                    extra_details={"command": (command or " ".join(argv or []))[:200]},
                )
            ) from exc
        if resp.get("type") == "ERROR":
            raise RuntimeError(
                _friendly(resp.get("payload", {}), self._shim, "EXECUTE_COMMAND failed")
            )
        return _CommandResult(resp["payload"])


class _CommandResult:
    def __init__(self, payload: dict) -> None:
        self.stdout = payload.get("stdout", "")
        self.stderr = payload.get("stderr", "")
        self.exit_code = payload.get("exit_code", -1)
        self.timed_out = payload.get("timed_out", False)


# ── Local LLM routing ────────────────────────────────────────────────────────
#
# When ``LocalLLMRouter`` greenlights local routing, ``_LocalLLMProxy`` sends
# a LOCAL_LLM_COMPLETION over the WS and consumes the shim's streaming
# LOCAL_LLM_COMPLETION_CHUNK frames + terminal LOCAL_LLM_COMPLETION_RESPONSE.
# See experimental/local-pc-executor/docs/LOCAL_LLM.md for the wire spec.


class LocalLLMError(RuntimeError):
    """Raised when a local LLM completion fails on the shim.

    ``code`` mirrors the wire error (``MODEL_NOT_AVAILABLE`` /
    ``LOCAL_LLM_BUSY`` / ``LOCAL_LLM_FAILED``) so the platform's error
    translator can branch on it.
    """

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class _LocalLLMProxy:
    """Stream completions from the shim's local LLM backend.

    Two surfaces:
      * ``complete(model, messages, **opts)`` — async iterator that yields
        text deltas in order, then raises StopAsyncIteration when the
        shim emits the terminal RESPONSE. Errors surface as
        :class:`LocalLLMError`.
      * ``complete_blocking(...)`` — non-streaming convenience that
        returns the assembled content as a single string.

    Both use the shim's per-request streaming queue (see
    :meth:`LocalPCShim._register_stream` / :meth:`_dispatch_stream_frame`).
    """

    def __init__(self, shim: "LocalPCShim") -> None:
        self._shim = shim

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        """Stream deltas. Yields ``str`` chunks; raises LocalLLMError on
        shim-side failure. The async generator drives the WS round-trip;
        callers MUST consume it to completion (or close it) so the
        per-request queue gets cleaned up.

        See LOCAL_LLM.md for the wire payload shape.
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        msg_id = str(uuid.uuid4())
        queue = self._shim._register_stream(msg_id)
        envelope = {
            "type": "LOCAL_LLM_COMPLETION",
            "id": msg_id,
            "ts": time.time(),
            "payload": payload,
        }
        try:
            await self._shim._ws.send_text(json.dumps(envelope))
        except Exception as exc:
            self._shim._cleanup_stream(msg_id)
            raise LocalLLMError(
                code="LOCAL_LLM_FAILED",
                message=f"[LocalPC] Failed to send LOCAL_LLM_COMPLETION: {exc}",
            ) from exc

        try:
            while True:
                frame = await queue.get()
                msg_type = frame.get("type")
                payload_in = frame.get("payload") or {}
                if msg_type == "LOCAL_LLM_COMPLETION_CHUNK":
                    delta = payload_in.get("delta") or ""
                    finish_reason = payload_in.get("finish_reason")
                    if delta:
                        yield delta
                    if finish_reason is not None:
                        # Terminal chunk marker — the RESPONSE will follow.
                        continue
                elif msg_type == "LOCAL_LLM_COMPLETION_RESPONSE":
                    # End of stream; we're done.
                    return
                elif msg_type == "ERROR":
                    code = payload_in.get("code", "LOCAL_LLM_FAILED")
                    message = payload_in.get("message", "Local LLM completion failed")
                    details = payload_in.get("details") or {}
                    raise LocalLLMError(code=code, message=message, details=details)
                else:
                    # Unknown frame type for this id — log + skip.
                    logger.debug(
                        "[LocalLLM] Unknown frame on stream %s: type=%s",
                        msg_id,
                        msg_type,
                    )
        finally:
            self._shim._cleanup_stream(msg_id)

    async def complete_blocking(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Run a streaming completion and return the assembled string.

        Convenience for callers that want the whole response in one go
        (tests, the platform-side adapter when wrapped in a non-streaming
        path). Errors propagate as :class:`LocalLLMError`.
        """
        chunks: list[str] = []
        async for delta in self.complete(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            chunks.append(delta)
        return "".join(chunks)


# ── Workflow recording ───────────────────────────────────────────────────────
#
# `_RecordingProxy` wraps the §6 wire ops: START_RECORDING / STOP_RECORDING /
# RECORDING_FETCH. Demonstration mode buffers on the shim and the platform
# pulls via `fetch()` after STOP + user approval. Co-pilot mode additionally
# streams RECORDING_STEP frames — unsolicited, non-acked, modeled like STATUS
# (§6) — which the recv loop fans out per recording_id into a queue the live
# co-pilot loop drains via `stream_steps()`.
#
# See experimental/local-pc-executor/docs/WORKFLOW_RECORDING.md.


class ShimRecordingError(RuntimeError):
    """Raised when a recording wire op returns a structured ERROR.

    ``code`` mirrors the wire ``payload.code`` (RECORDING_NOT_FOUND,
    RECORDING_CHANNEL_UNAVAILABLE, RECORDING_ALREADY_ACTIVE,
    CONSENT_REQUIRED, INTERPRETATION_UNAVAILABLE) so the MCP-tool layer can
    branch on the structured surface without parsing the human ``message``.
    ``message`` is already LLM-friendly (produced by :mod:`local_pc_errors`).
    """

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _raise_recording(resp: dict, shim: "LocalPCShim | None", fallback: str) -> None:
    """Translate a wire ERROR response into a typed ShimRecordingError."""
    if resp.get("type") != "ERROR":
        return
    payload = resp.get("payload") or {}
    raise ShimRecordingError(
        code=str(payload.get("code") or "INTERNAL_ERROR"),
        message=_friendly(payload, shim, fallback),
        details=(
            payload.get("details") if isinstance(payload.get("details"), dict) else {}
        ),
    )


class _RecordingProxy:
    """Wire-op wrapper for the shim's workflow-recording surface.

    Mirrors ``experimental/local-pc-executor/docs/WORKFLOW_RECORDING.md``
    §6. ``start`` / ``stop`` / ``fetch`` are request/response (count against
    in-flight). RECORDING_STEP frames (co-pilot mode) arrive out-of-band and
    are exposed via :meth:`stream_steps`.
    """

    def __init__(self, shim: "LocalPCShim") -> None:
        self._shim = shim

    async def start(
        self,
        *,
        mode: str,
        interpretation_route: str,
        channels: list[str],
        consent_token: str,
    ) -> str:
        """START_RECORDING → return the new recording_id.

        ``consent_token`` is REQUIRED — the platform cannot self-assert it
        (§9); START without a valid shim-issued token gets CONSENT_REQUIRED.
        """
        resp = await self._shim._rpc(
            "START_RECORDING",
            {
                "mode": mode,
                "interpretation_route": interpretation_route,
                "channels": list(channels),
                "consent_token": consent_token,
            },
        )
        _raise_recording(resp, self._shim, "START_RECORDING failed")
        recording_id = str((resp.get("payload") or {}).get("recording_id") or "")
        if recording_id:
            # Pre-create the step buffer so a fast first RECORDING_STEP frame
            # (co-pilot mode) doesn't race the START response and get dropped.
            self._shim._ensure_recording_buffer(recording_id)
        return recording_id

    async def stop(self, recording_id: str) -> "RecordingSummary":
        """STOP_RECORDING → return the RECORDING_SUMMARY."""
        resp = await self._shim._rpc("STOP_RECORDING", {"recording_id": recording_id})
        _raise_recording(resp, self._shim, "STOP_RECORDING failed")
        return RecordingSummary.from_payload(resp.get("payload") or {})

    async def fetch(self, recording_id: str) -> "WorkflowRecording":
        """RECORDING_FETCH → return the full post-redaction WorkflowRecording.

        For demonstration mode this is the only path the data leaves the
        machine — the shim buffers until STOP + user approval, then the
        platform pulls (§6).
        """
        resp = await self._shim._rpc("RECORDING_FETCH", {"recording_id": recording_id})
        _raise_recording(resp, self._shim, "RECORDING_FETCH failed")
        return WorkflowRecording.from_payload(resp.get("payload") or {})

    def stream_steps(self, recording_id: str):
        """Async iterator over live RECORDING_STEP frames (co-pilot mode).

        Yields :class:`TrajectoryStep` as the shim emits them. The iterator
        runs until the caller breaks out of it (e.g. after STOP); the
        underlying buffer is dropped via :meth:`LocalPCShim.close_recording`.
        Demonstration mode never streams — this iterator simply blocks
        until the buffer is closed.
        """
        return self._shim._iter_recording_steps(recording_id)


class _RpcAttemptFailed(Exception):
    """Internal: one `_send_and_wait` attempt failed (timeout or WS error).

    Carries the wire `id` so the outer `_rpc` can convert a non-retryable
    attempt into an :class:`OpUnconfirmedError` / :class:`WriteUnconfirmedError`
    with the original correlation id intact.
    """

    def __init__(self, wire_id: str, msg_type: str, *, timed_out: bool) -> None:
        super().__init__(f"{msg_type} attempt failed (wire id={wire_id})")
        self.wire_id = wire_id
        self.msg_type = msg_type
        self.timed_out = timed_out


class LocalPCShim:
    """
    Drop-in replacement for E2B AsyncSandbox that routes execution to the
    user's local machine via the autogpt-local-executor shim.

    Duck-type contract: .commands.run(), .files.read(), .files.write(),
                        .pause(), .kill(), .sandbox_id

    Extended attributes (LocalPC-only; safe to read via isinstance check):
        .allowed_root, .machine_id, .platform, .arch, .capabilities,
        .shim_version, .screen_resolution, .local_llm_models, .hardware_devices,
        .computer_use_features

    Computer-use surface:
        .computer.screenshot(...), .computer.click(...), .computer.type(...),
        and friends — see ``_ComputerProxy`` and
        ``experimental/local-pc-executor/docs/COMPUTER_USE.md``.

    Local LLM surface:
        .local_llm.complete(...) (async iterator of deltas) and
        .local_llm.complete_blocking(...) (string). Routed only when
        ``LocalLLMRouter.should_route`` returns a model — see
        ``local_llm_router.py`` and
        ``experimental/local-pc-executor/docs/LOCAL_LLM.md``.

    Workflow-recording surface:
        .recording.start(...), .recording.stop(...), .recording.fetch(...),
        and .recording.stream_steps(...) — see ``_RecordingProxy`` and
        ``experimental/local-pc-executor/docs/WORKFLOW_RECORDING.md``.
        Only usable when the shim advertised the ``recording`` capability.
    """

    def __init__(
        self,
        session_id: str,
        ws: WebSocket,
        hello: ShimHello | None = None,
        *,
        manager: "ShimConnectionManager | None" = None,
    ) -> None:
        self.sandbox_id = session_id
        self._ws = ws
        self._manager = manager
        hello = hello or ShimHello()
        self.machine_id = hello.machine_id
        self.platform = hello.platform
        self.arch = hello.arch
        self.shim_version = hello.shim_version
        self.allowed_root = hello.allowed_root
        self.capabilities = hello.capabilities
        self.screen_resolution = hello.screen_resolution
        self.local_llm_models = hello.local_llm_models
        self.hardware_devices = hello.hardware_devices
        self.computer_use_features = hello.computer_use_features
        self._pending: dict[str, asyncio.Future[dict]] = {}
        # Streaming requests (LOCAL_LLM_COMPLETION) accumulate multiple
        # frames per request id. The queue collects every CHUNK + the
        # terminal RESPONSE (or ERROR); the consumer in _LocalLLMProxy
        # drains it and unregisters when the stream closes.
        self._streaming: dict[str, asyncio.Queue[dict]] = {}
        # Workflow recording — co-pilot mode streams RECORDING_STEP frames
        # (unsolicited, non-acked, like STATUS — §6) keyed by recording_id.
        # Each recording gets a queue the live co-pilot loop drains via
        # `recording.stream_steps()`. A sentinel `None` put on the queue
        # signals the iterator to stop (set by `close_recording`).
        self._recording_steps: dict[str, asyncio.Queue[TrajectoryStep | None]] = {}
        # Backpressure — see PROTOCOL.md §Concurrency + STATUS frame
        # support. `pending_capacity` is the shim's self-reported headroom:
        # 0 = at the concurrency cap, refuse-new-work; >0 = slots free; None
        # = unknown (pre-STATUS or shim doesn't advertise capacity yet).
        # Updated by both per-response envelopes and periodic STATUS frames.
        self._pending_capacity: int | None = None
        self._capacity_available = asyncio.Event()
        self._capacity_available.set()  # default: assume open until told otherwise
        self.files = _FilesProxy(self)
        self.commands = _CommandsProxy(self)
        self.computer = _ComputerProxy(self)
        self.local_llm = _LocalLLMProxy(self)
        self.recording = _RecordingProxy(self)
        self._recv_task = asyncio.create_task(self._recv_loop())

    @property
    def pending_capacity(self) -> int | None:
        """Last-known shim-side request-slot headroom.

        ``None`` until the shim advertises capacity (either embedded in a
        response envelope's ``pending_capacity`` field or via a periodic
        STATUS frame). ``0`` means the shim is at its ``max_concurrent`` cap
        — new `_rpc` calls will block on the capacity event for up to 30s
        before raising :class:`ShimOverloadedError`.
        """
        return self._pending_capacity

    @classmethod
    async def for_session(
        cls,
        session_id: str,
        *,
        manager: ShimConnectionManager,
        connect_timeout: float = 30.0,
    ) -> "LocalPCShim":
        ws = await manager.wait_for(session_id, timeout=connect_timeout)
        hello = manager.get_hello(session_id)
        return cls(session_id, ws, hello, manager=manager)

    def _update_pending_capacity(self, value: Any, *, source: str) -> None:
        """Defensive: accept whatever the shim sent and only honor sane ints.

        STATUS / response shapes drift more than wire docs admit. A
        non-int, missing, or negative value leaves the prior reading
        unchanged rather than crashing the recv loop. Logged at DEBUG so
        operators can spot a shim that's regressed its self-report.
        """
        if value is None:
            return
        try:
            capacity = int(value)
        except (TypeError, ValueError):
            logger.debug(
                "[LocalPC] Ignoring non-int pending_capacity=%r from %s",
                value,
                source,
            )
            return
        if capacity < 0:
            logger.debug(
                "[LocalPC] Ignoring negative pending_capacity=%d from %s",
                capacity,
                source,
            )
            return
        self._pending_capacity = capacity
        if capacity > 0:
            self._capacity_available.set()
        else:
            self._capacity_available.clear()

    async def _await_capacity(
        self, msg_type: str, *, timeout: float = _CAPACITY_WAIT_TIMEOUT_SECONDS
    ) -> None:
        """Block until the shim reports headroom, or raise SHIM_OVERLOADED.

        If ``pending_capacity`` is 0, wait up to ``timeout`` seconds for a
        subsequent response or STATUS frame to clear the gate. Past that,
        raise :class:`ShimOverloadedError` proactively — sending the op
        anyway would just get bounced with the same code by the shim.
        """
        # Defensive against test fixtures that construct shims via
        # ``__new__`` and skip ``__init__``.
        capacity = getattr(self, "_pending_capacity", None)
        event = getattr(self, "_capacity_available", None)
        if capacity != 0:
            return
        if event is None or event.is_set():
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise ShimOverloadedError(
                f"[LocalPC] Shim has reported pending_capacity=0 for {timeout}s; "
                f"refusing to send {msg_type}"
            ) from exc

    async def _send_and_wait(
        self,
        msg_type: str,
        payload: dict,
        *,
        timeout: float,
    ) -> dict:
        """One attempt at sending a wire op and awaiting its response.

        Raises ``TimeoutError`` if the response doesn't arrive within
        ``timeout`` or the underlying WS send fails before the response.
        """
        msg_id = str(uuid.uuid4())
        msg = {"type": msg_type, "id": msg_id, "ts": time.time(), "payload": payload}
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        self._pending[msg_id] = fut
        try:
            await self._ws.send_text(json.dumps(msg))
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(msg_id, None)
            # Tag the timeout with the wire id so the caller can convert it
            # into a typed OpUnconfirmedError (non-idempotent ops) or trigger
            # a single auto-retry (idempotent ops).
            raise _RpcAttemptFailed(msg_id, msg_type, timed_out=True) from exc
        except Exception as exc:
            # WS disconnect / send error / recv loop closure — the response
            # cannot arrive on this connection. Surface as unconfirmed so the
            # caller decides retry vs. raise based on idempotency.
            self._pending.pop(msg_id, None)
            raise _RpcAttemptFailed(msg_id, msg_type, timed_out=False) from exc

    async def _rpc(
        self, msg_type: str, payload: dict, *, timeout: float = 30.0
    ) -> dict:
        """Send a wire op and await its response.

        Disconnect/timeout semantics follow the per-op idempotency table in
        ``experimental/local-pc-executor/docs/PROTOCOL.md``:

        - Idempotent ops (FILE_READ, FILE_STAT, ...): on
          timeout/disconnect, schedule one automatic retry once the WS is
          back up. The retry is invisible to the caller — they get the
          eventual result or one final error.
        - Non-idempotent ops (FILE_WRITE, EXECUTE_COMMAND, ...): raise
          :class:`WriteUnconfirmedError` / :class:`OpUnconfirmedError`
          immediately so the LLM can probe state instead of double-applying
          a side effect.

        Backpressure: if the shim's most recent ``pending_capacity`` signal
        is 0, this blocks for up to 30s waiting for headroom before sending,
        then raises :class:`ShimOverloadedError`.
        """
        await self._await_capacity(msg_type)
        try:
            return await self._send_and_wait(msg_type, payload, timeout=timeout)
        except _RpcAttemptFailed as first:
            if msg_type in _IDEMPOTENT_OPS:
                # One retry, after the WS is back. wait_for_reconnect is a
                # best-effort no-op if the manager isn't wired in (tests).
                try:
                    await self._await_reconnect_for_retry()
                except Exception:
                    pass
                try:
                    result = await self._send_and_wait(
                        msg_type, payload, timeout=timeout
                    )
                except _RpcAttemptFailed as second:
                    record_rpc_retry(msg_type, recovered=False)
                    if second.timed_out:
                        raise TimeoutError(
                            f"[LocalPC] RPC {msg_type} timed out after {timeout}s "
                            "(retry also failed)"
                        ) from second
                    raise OpUnconfirmedError(
                        op=msg_type,
                        wire_id=second.wire_id,
                        message=(
                            f"[LocalPC] RPC {msg_type} disconnected mid-call "
                            "(retry also failed)"
                        ),
                    ) from second
                record_rpc_retry(msg_type, recovered=True)
                return result
            # Non-idempotent: bubble up as a typed unconfirmed error so the
            # caller's translator can surface actionable English.
            if msg_type == "FILE_WRITE":
                raise WriteUnconfirmedError(
                    wire_id=first.wire_id,
                    message=(
                        f"[LocalPC] FILE_WRITE (id={first.wire_id}) was sent but "
                        "the shim did not ACK before the connection dropped"
                    ),
                ) from first
            raise OpUnconfirmedError(
                op=msg_type,
                wire_id=first.wire_id,
                message=(
                    f"[LocalPC] {msg_type} (id={first.wire_id}) was sent but no "
                    "response arrived before the connection dropped"
                ),
            ) from first

    async def _await_reconnect_for_retry(self, *, timeout: float = 30.0) -> None:
        """Wait for the shim to be reachable again before an idempotent retry.

        Best-effort: if a connection manager isn't attached (unit tests
        stub `_rpc` directly), this returns immediately. The retry then
        re-uses ``self._ws`` and will fail fast if the WS is still dead.
        """
        manager = getattr(self, "_manager", None)
        if manager is None:
            return
        try:
            ws = await manager.wait_for(self.sandbox_id, timeout=timeout)
        except Exception:
            return
        # If the manager produced a fresh WS, swap it in so the retry rides
        # the new connection.
        if ws is not self._ws:
            self._ws = ws

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws.iter_text():
                try:
                    msg = json.loads(raw)
                    self._handle_envelope_capacity(msg)
                    msg_type = msg.get("type")
                    if msg_type == "STATUS":
                        self._handle_status_frame(msg)
                        continue
                    if msg_type == "RECORDING_STEP":
                        # Unsolicited, non-acked, out-of-band (co-pilot
                        # mode only — §6). Buffered per recording_id; never
                        # routed through _pending and never auto-retried.
                        self._handle_recording_step(msg)
                        continue
                    msg_id = msg.get("id")
                    # Streaming dispatch: LOCAL_LLM_COMPLETION_CHUNK and the
                    # terminal LOCAL_LLM_COMPLETION_RESPONSE share a wire id
                    # with the original LOCAL_LLM_COMPLETION request. ERROR
                    # for a streaming op also flows through the same queue.
                    if msg_id and msg_id in self._streaming:
                        if msg_type in (
                            "LOCAL_LLM_COMPLETION_CHUNK",
                            "LOCAL_LLM_COMPLETION_RESPONSE",
                            "ERROR",
                        ):
                            await self._streaming[msg_id].put(msg)
                            continue
                    if msg_id and msg_id in self._pending:
                        fut = self._pending.pop(msg_id)
                        if not fut.done():
                            fut.set_result(msg)
                except Exception:
                    logger.exception("[LocalPC] Error processing shim message")
        except Exception:
            logger.debug("[LocalPC] Shim recv loop ended for %s", self.sandbox_id[:12])

    def _register_stream(self, msg_id: str) -> asyncio.Queue[dict]:
        """Register a streaming request and return its inbound-frame queue.

        Used by ``_LocalLLMProxy.complete()``. The queue receives every
        CHUNK / RESPONSE / ERROR frame the recv loop sees for ``msg_id``.
        Callers MUST call :meth:`_cleanup_stream` when done so the dict
        doesn't grow without bound.
        """
        queue: asyncio.Queue[dict] = asyncio.Queue()
        self._streaming[msg_id] = queue
        return queue

    def _cleanup_stream(self, msg_id: str) -> None:
        """Drop the streaming queue. Safe to call multiple times."""
        self._streaming.pop(msg_id, None)

    def _handle_envelope_capacity(self, msg: Any) -> None:
        """Mine ``pending_capacity`` out of any shim → platform envelope.

        Per the shim partner's backpressure work, every response (and
        STATUS frame) carries the shim's current headroom. We accept it
        from either the top-level envelope or the payload — different shim
        versions have placed it in different spots, and the platform side
        shouldn't crash on either. Missing field is fine — leave capacity
        unchanged.
        """
        if not isinstance(msg, dict):
            return
        if "pending_capacity" in msg:
            self._update_pending_capacity(
                msg.get("pending_capacity"), source="envelope"
            )
        payload = msg.get("payload")
        if isinstance(payload, dict) and "pending_capacity" in payload:
            self._update_pending_capacity(
                payload.get("pending_capacity"), source="payload"
            )

    def _handle_status_frame(self, msg: Any) -> None:
        """Periodic STATUS frame: log the snapshot + refresh capacity.

        Frame shape (per shim partner spec):
            {type: "STATUS",
             payload: {in_flight, max_concurrent, queue_depth,
                       audit_log_bytes, uptime_seconds, pending_capacity?}}

        If ``pending_capacity`` is present, it wins. Otherwise we derive
        ``max_concurrent - in_flight`` as a fallback for shim versions that
        omit it. Logged at DEBUG so the snapshot is available for diagnosis
        without spamming production logs.
        """
        if not isinstance(msg, dict):
            return
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            return
        if "pending_capacity" in payload:
            self._update_pending_capacity(
                payload.get("pending_capacity"), source="STATUS"
            )
        else:
            max_concurrent = payload.get("max_concurrent")
            in_flight = payload.get("in_flight")
            try:
                if max_concurrent is not None and in_flight is not None:
                    derived = max(0, int(max_concurrent) - int(in_flight))
                    self._update_pending_capacity(derived, source="STATUS-derived")
            except (TypeError, ValueError):
                pass
        logger.debug(
            "[LocalPC] STATUS frame for %s: in_flight=%r max_concurrent=%r "
            "queue_depth=%r audit_log_bytes=%r uptime_seconds=%r capacity=%r",
            self.sandbox_id[:12],
            payload.get("in_flight"),
            payload.get("max_concurrent"),
            payload.get("queue_depth"),
            payload.get("audit_log_bytes"),
            payload.get("uptime_seconds"),
            self._pending_capacity,
        )

    # --- Workflow recording (RECORDING_STEP buffering) ---------------------

    def _ensure_recording_buffer(
        self, recording_id: str
    ) -> "asyncio.Queue[TrajectoryStep | None]":
        """Return (creating if needed) the per-recording step queue.

        Pre-created on START_RECORDING so a fast first RECORDING_STEP frame
        can't race the START response and be dropped.
        """
        queue = self._recording_steps.get(recording_id)
        if queue is None:
            queue = asyncio.Queue()
            self._recording_steps[recording_id] = queue
        return queue

    def _handle_recording_step(self, msg: Any) -> None:
        """Fan a RECORDING_STEP frame into its per-recording queue.

        Frame shape (§6):
            {type: "RECORDING_STEP",
             payload: {recording_id, step: {TrajectoryStep}}}

        A frame for an unknown recording_id is buffered anyway (the START
        response may still be in flight) so nothing is lost; if it's truly
        orphaned it's harmless — the queue is dropped on close_recording.
        """
        if not isinstance(msg, dict):
            return
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            return
        recording_id = str(payload.get("recording_id") or "")
        if not recording_id:
            logger.debug("[LocalPC] RECORDING_STEP without recording_id; dropping")
            return
        step_payload = payload.get("step")
        if not isinstance(step_payload, dict):
            logger.debug(
                "[LocalPC] RECORDING_STEP for %s missing step body; dropping",
                recording_id,
            )
            return
        step = TrajectoryStep.from_payload(step_payload)
        queue = self._ensure_recording_buffer(recording_id)
        queue.put_nowait(step)

    async def _iter_recording_steps(self, recording_id: str):
        """Async iterator yielding TrajectoryStep frames as they arrive.

        Backs ``recording.stream_steps()``. Runs until a sentinel ``None``
        is enqueued by :meth:`close_recording`, then stops and drops the
        buffer.
        """
        queue = self._ensure_recording_buffer(recording_id)
        try:
            while True:
                step = await queue.get()
                if step is None:
                    return
                yield step
        finally:
            self._recording_steps.pop(recording_id, None)

    def close_recording(self, recording_id: str) -> None:
        """Signal the step iterator for ``recording_id`` to stop.

        Enqueues the stop sentinel so a live ``stream_steps()`` consumer
        finishes cleanly after STOP_RECORDING. Safe to call multiple times
        and for recordings that never streamed.
        """
        queue = self._recording_steps.get(recording_id)
        if queue is not None:
            queue.put_nowait(None)

    async def pause(self) -> None:
        pass  # no billing on local machine

    async def kill(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass
        if not self._recv_task.done():
            self._recv_task.cancel()
