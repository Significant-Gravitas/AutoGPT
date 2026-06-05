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

logger = logging.getLogger(__name__)


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

    def register(
        self, session_id: str, ws: WebSocket, hello: ShimHello | None = None
    ) -> None:
        self._connections[session_id] = ws
        if hello is not None:
            self._hellos[session_id] = hello
        for fut in self._waiters.pop(session_id, []):
            if not fut.done():
                fut.set_result(ws)
        logger.info("[LocalPC] Shim registered for session %s", session_id[:12])

    def unregister(self, session_id: str) -> None:
        self._connections.pop(session_id, None)
        self._hellos.pop(session_id, None)
        logger.info("[LocalPC] Shim unregistered for session %s", session_id[:12])

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
        resp = await self._shim._rpc(
            "FILE_WRITE",
            {
                "path": path,
                "content": wire_content,
                "encoding": wire_encoding,
                "create_parents": True,
            },
        )
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
        resp = await self._shim._rpc(
            "FILE_DELETE",
            {"path": path, "recursive": recursive, "missing_ok": missing_ok},
        )
        if resp.get("type") == "ERROR":
            raise OSError(
                _friendly(resp.get("payload", {}), self._shim, "FILE_DELETE failed")
            )

    async def move(self, src: str, dst: str, *, overwrite: bool = False) -> None:
        """Cross-OS portable replacement for shell `mv` / `move`."""
        resp = await self._shim._rpc(
            "FILE_MOVE",
            {"src": src, "dst": dst, "overwrite": overwrite},
        )
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
        resp = await self._shim._rpc("INPUT_ACTION", payload)
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
        resp = await self._shim._rpc(
            "WINDOW_FOCUS", {"window_id": window_id, "raise": raise_}
        )
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
        resp = await self._shim._rpc(
            "APP_LAUNCH",
            {
                "bundle_id": bundle_id,
                "executable_path": executable_path,
                "args": list(args or []),
                "activate": activate,
            },
        )
        _raise_computer_use(resp, self._shim, "APP_LAUNCH failed")
        return resp.get("payload") or {}

    # --- Clipboard ---------------------------------------------------------

    async def clipboard_read(self, *, format: str = "text") -> str | None:
        resp = await self._shim._rpc("CLIPBOARD_READ", {"format": format})
        _raise_computer_use(resp, self._shim, "CLIPBOARD_READ failed")
        return (resp.get("payload") or {}).get("content")

    async def clipboard_write(self, content: str, *, format: str = "text") -> None:
        resp = await self._shim._rpc(
            "CLIPBOARD_WRITE", {"format": format, "content": content}
        )
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
        resp = await self._shim._rpc("EXECUTE_COMMAND", payload)
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
    """

    def __init__(
        self,
        session_id: str,
        ws: WebSocket,
        hello: ShimHello | None = None,
    ) -> None:
        self.sandbox_id = session_id
        self._ws = ws
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
        self.files = _FilesProxy(self)
        self.commands = _CommandsProxy(self)
        self.computer = _ComputerProxy(self)
        self._recv_task = asyncio.create_task(self._recv_loop())

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
        return cls(session_id, ws, hello)

    async def _rpc(
        self, msg_type: str, payload: dict, *, timeout: float = 30.0
    ) -> dict:
        msg_id = str(uuid.uuid4())
        msg = {"type": msg_type, "id": msg_id, "ts": time.time(), "payload": payload}
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        self._pending[msg_id] = fut
        try:
            await self._ws.send_text(json.dumps(msg))
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise TimeoutError(f"[LocalPC] RPC {msg_type} timed out after {timeout}s")

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws.iter_text():
                try:
                    msg = json.loads(raw)
                    msg_id = msg.get("id")
                    if msg_id and msg_id in self._pending:
                        fut = self._pending.pop(msg_id)
                        if not fut.done():
                            fut.set_result(msg)
                except Exception:
                    logger.exception("[LocalPC] Error processing shim message")
        except Exception:
            logger.debug("[LocalPC] Shim recv loop ended for %s", self.sandbox_id[:12])

    async def pause(self) -> None:
        pass  # no billing on local machine

    async def kill(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass
        if not self._recv_task.done():
            self._recv_task.cancel()
