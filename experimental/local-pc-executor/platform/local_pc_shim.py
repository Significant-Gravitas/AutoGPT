"""
LocalPCShim — platform-side proxy for a connected local PC shim.

⚠️  EXPERIMENTAL / NOT IMPLEMENTED ⚠️

This class is duck-type compatible with E2B's AsyncSandbox, so it can be returned
from _setup_executor() (formerly _setup_e2b()) and used transparently by all
downstream code in sdk/service.py and e2b_file_tools.py.

The real work happens over WebSocket on the user's machine.
This class is the platform-side handle that converts AsyncSandbox method calls
into WebSocket messages and awaits the responses.

Insertion point in sdk/service.py (~line 3815):

    async def _setup_executor():
        # NEW: local PC shim branch
        if config.use_local_pc_executor:
            shim = await LocalPCShim.for_session(session_id, user_id=user_id)
            if shim:
                return shim
        # existing E2B branch follows...

Duck-type interface required by downstream code:
    sandbox.sandbox_id          → str
    sandbox.commands.run(cmd, cwd, timeout) → CommandResult
    sandbox.files.read(path)    → bytes
    sandbox.files.write(path, content) → None
    await sandbox.pause()       → None
    await sandbox.kill()        → None
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Mirrors E2B's CommandResult interface."""
    stdout: str
    stderr: str
    exit_code: int


class LocalPCShim:
    """
    Platform-side handle for a connected local PC shim.

    One instance per active session. Created by LocalPCShim.for_session()
    which looks up the WebSocket connection in ShimConnectionManager.

    All I/O methods send a protocol message and await the response,
    with a configurable timeout.
    """

    def __init__(
        self,
        session_id: str,
        machine_id: str,
        capabilities: list[str],
        allowed_root: str,
        ws_send: Any,  # callable: async (dict) -> dict
    ) -> None:
        self.session_id = session_id
        self.machine_id = machine_id
        self.capabilities = capabilities
        self.allowed_root = allowed_root
        self._send = ws_send  # injected by ShimConnectionManager

        # Duck-type compatibility with E2B AsyncSandbox
        self.sandbox_id = f"localpc:{machine_id}:{session_id}"
        self.commands = _CommandsProxy(self)
        self.files = _FilesProxy(self)

    @classmethod
    async def for_session(
        cls,
        session_id: str,
        user_id: str,
    ) -> Optional["LocalPCShim"]:
        """
        Look up the active shim connection for this session.
        Returns None if no shim is connected.

        TODO: implement ShimConnectionManager registry.
        The manager maintains a dict of {session_id: LocalPCShim} populated
        when a shim connects to /ws/local-executor/{session_id}.
        """
        # TODO: return ShimConnectionManager.get(session_id)
        logger.warning(
            "[LocalPC] for_session() not yet implemented — ShimConnectionManager missing"
        )
        return None

    async def pause(self) -> None:
        """
        Called at end of each turn (mirrors E2B sandbox.pause()).
        For local shim: send a PAUSE message so the shim can flush buffers,
        release file locks, etc. No-op if shim doesn't support it.
        """
        # TODO: await self._send({"type": "PAUSE", "id": str(uuid.uuid4()), ...})
        pass

    async def kill(self) -> None:
        """
        Called on session end. Sends DISCONNECT to the shim.
        The shim will close its WebSocket and stop processing.
        """
        # TODO: await self._send({"type": "DISCONNECT", ...})
        pass

    async def _request(self, msg_type: str, payload: dict, timeout: float = 30.0) -> dict:
        """
        Send a message to the shim and await the response.
        Correlates by message id. Raises TimeoutError on no response.

        TODO: implement once ShimConnectionManager exists.
        """
        msg_id = str(uuid.uuid4())
        msg = {
            "type": msg_type,
            "id": msg_id,
            "ts": time.time(),
            "payload": payload,
        }
        response = await asyncio.wait_for(self._send(msg), timeout=timeout)
        if response.get("type") == "ERROR":
            raise RuntimeError(
                f"[LocalPC] shim error: {response['payload']['code']}: "
                f"{response['payload']['message']}"
            )
        return response


class _CommandsProxy:
    """Mirrors E2B's sandbox.commands interface."""

    def __init__(self, shim: LocalPCShim) -> None:
        self._shim = shim

    async def run(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = 30,
        env: dict | None = None,
    ) -> CommandResult:
        """
        Execute a shell command on the user's machine via the shim.

        This is called by e2b_file_tools.py and sdk/service.py wherever
        E2B's sandbox.commands.run() is currently called.
        """
        response = await self._shim._request(
            "EXECUTE_COMMAND",
            {
                "command": command,
                "cwd": cwd or self._shim.allowed_root,
                "timeout_seconds": timeout,
                "env": env or {},
            },
            timeout=timeout + 5.0,
        )
        p = response["payload"]
        return CommandResult(
            stdout=p["stdout"],
            stderr=p["stderr"],
            exit_code=p["exit_code"],
        )


class _FilesProxy:
    """Mirrors E2B's sandbox.files interface."""

    def __init__(self, shim: LocalPCShim) -> None:
        self._shim = shim

    async def read(self, path: str, encoding: str = "utf-8") -> bytes | str:
        """Read a file from the user's machine."""
        response = await self._shim._request(
            "FILE_READ",
            {"path": path, "encoding": encoding},
        )
        return response["payload"]["content"]

    async def write(self, path: str, content: bytes | str) -> None:
        """Write a file to the user's machine."""
        if isinstance(content, bytes):
            import base64
            encoded = base64.b64encode(content).decode("ascii")
            encoding = "base64"
        else:
            encoded = content
            encoding = "utf-8"

        await self._shim._request(
            "FILE_WRITE",
            {
                "path": path,
                "content": encoded,
                "encoding": encoding,
                "create_parents": True,
            },
        )
