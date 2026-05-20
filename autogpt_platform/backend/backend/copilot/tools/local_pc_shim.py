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

logger = logging.getLogger(__name__)

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
            screen_resolution=tuple(sr) if isinstance(sr, (list, tuple)) and len(sr) == 2 else None,
            local_llm_models=list(payload.get("local_llm_models") or []),
            hardware_devices=list(payload.get("hardware_devices") or []),
        )


class ShimConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._hellos: dict[str, ShimHello] = {}
        self._waiters: dict[str, list[asyncio.Future[WebSocket]]] = {}

    def register(self, session_id: str, ws: WebSocket, hello: ShimHello | None = None) -> None:
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
            raise OSError(resp["payload"].get("message", "FILE_READ failed"))
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
            raise OSError(resp["payload"].get("message", "FILE_WRITE failed"))

    async def stat(self, path: str, *, follow_symlinks: bool = True) -> dict:
        """Cross-OS portable replacement for shell `stat` / `readlink -f` / `test -e`."""
        resp = await self._shim._rpc(
            "FILE_STAT", {"path": path, "follow_symlinks": follow_symlinks}
        )
        if resp.get("type") == "ERROR":
            raise OSError(resp["payload"].get("message", "FILE_STAT failed"))
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
            raise OSError(resp["payload"].get("message", "FILE_LIST failed"))
        return resp["payload"]

    async def delete(self, path: str, *, recursive: bool = False, missing_ok: bool = False) -> None:
        """Cross-OS portable replacement for shell `rm` / `del`."""
        resp = await self._shim._rpc(
            "FILE_DELETE",
            {"path": path, "recursive": recursive, "missing_ok": missing_ok},
        )
        if resp.get("type") == "ERROR":
            raise OSError(resp["payload"].get("message", "FILE_DELETE failed"))

    async def move(self, src: str, dst: str, *, overwrite: bool = False) -> None:
        """Cross-OS portable replacement for shell `mv` / `move`."""
        resp = await self._shim._rpc(
            "FILE_MOVE",
            {"src": src, "dst": dst, "overwrite": overwrite},
        )
        if resp.get("type") == "ERROR":
            raise OSError(resp["payload"].get("message", "FILE_MOVE failed"))


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
            raise ValueError("LocalPCShim.commands.run: either command or argv must be set")
        if cwd:
            payload["cwd"] = cwd
        if timeout:
            payload["timeout_seconds"] = timeout
        if envs:
            payload["env"] = envs
        resp = await self._shim._rpc("EXECUTE_COMMAND", payload)
        if resp.get("type") == "ERROR":
            raise RuntimeError(resp["payload"].get("message", "EXECUTE_COMMAND failed"))
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
        .shim_version, .screen_resolution, .local_llm_models, .hardware_devices
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
        self._pending: dict[str, asyncio.Future[dict]] = {}
        self.files = _FilesProxy(self)
        self.commands = _CommandsProxy(self)
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

    async def _rpc(self, msg_type: str, payload: dict, *, timeout: float = 30.0) -> dict:
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
