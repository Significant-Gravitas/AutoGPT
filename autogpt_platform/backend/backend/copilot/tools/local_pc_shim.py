"""
Platform-side binding for the autogpt-local-executor shim.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

_shim_manager: "ShimConnectionManager | None" = None


def get_shim_manager() -> "ShimConnectionManager":
    global _shim_manager
    if _shim_manager is None:
        _shim_manager = ShimConnectionManager()
    return _shim_manager


class ShimConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._waiters: dict[str, list[asyncio.Future[WebSocket]]] = {}

    def register(self, session_id: str, ws: WebSocket) -> None:
        self._connections[session_id] = ws
        for fut in self._waiters.pop(session_id, []):
            if not fut.done():
                fut.set_result(ws)
        logger.info("[LocalPC] Shim registered for session %s", session_id[:12])

    def unregister(self, session_id: str) -> None:
        self._connections.pop(session_id, None)
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


class _FilesProxy:
    def __init__(self, shim: "LocalPCShim") -> None:
        self._shim = shim

    async def read(self, path: str, *, encoding: str = "utf-8") -> str:
        resp = await self._shim._rpc("FILE_READ", {"path": path, "encoding": encoding})
        if resp.get("type") == "ERROR":
            raise OSError(resp["payload"].get("message", "FILE_READ failed"))
        return resp["payload"]["content"]

    async def write(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        resp = await self._shim._rpc(
            "FILE_WRITE",
            {"path": path, "content": content, "encoding": encoding, "create_parents": True},
        )
        if resp.get("type") == "ERROR":
            raise OSError(resp["payload"].get("message", "FILE_WRITE failed"))


class _CommandsProxy:
    def __init__(self, shim: "LocalPCShim") -> None:
        self._shim = shim

    async def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
        envs: dict[str, str] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {"command": command}
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
    """

    def __init__(self, session_id: str, ws: WebSocket) -> None:
        self.sandbox_id = session_id
        self._ws = ws
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
        return cls(session_id, ws)

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
