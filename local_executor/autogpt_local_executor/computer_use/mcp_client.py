"""A minimal MCP client over a child process's stdio.

cua-driver speaks MCP (JSON-RPC 2.0, one message per line) on stdin and
stdout. This client starts the process, runs the initialize handshake, and
sends requests with a deadline each. It understands only what the shim
needs: tools/list, tools/call, cancellation, and the server's ping.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# The protocol revision the shim asks for. A server may answer with another
# it supports; every revision since 2024-11-05 keeps tools/list and
# tools/call the same for what the shim uses.
MCP_PROTOCOL_VERSION = "2025-06-18"

# One stdout line holds one whole message, a window screenshot included.
STDOUT_LINE_LIMIT_BYTES = 64 * 1024 * 1024
STDERR_TAIL_LINES = 200
# How long a stopping driver gets after its stdin closes, and after SIGTERM.
CLOSE_GRACE_SECONDS = 5.0
KILL_GRACE_SECONDS = 2.0

# JSON-RPC error codes.
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# tools/list pages followed before the server counts as broken. cua-driver
# 0.38 lists every tool on one page.
MAX_TOOL_LIST_PAGES = 50


class McpError(Exception):
    """The server answered a request with a JSON-RPC error."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"{message} (JSON-RPC {code})")
        self.code = code
        self.message = message
        self.data = data


class McpClosedError(Exception):
    """The server process exited or closed its output."""


@dataclass
class InitializeResult:
    protocol_version: str
    server_name: str
    server_version: str
    instructions: str
    capabilities: dict[str, Any] = field(default_factory=dict)


class StdioMcpClient:
    """One MCP server process. Not reusable: start it once, close it once."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        client_name: str = "autogpt-local-executor",
        client_version: str = "0",
    ) -> None:
        self._argv = list(argv)
        self._env = dict(env) if env is not None else None
        self._client_info = {"name": client_name, "version": client_version}
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr_reader: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=STDERR_TAIL_LINES)
        self._closed_error: McpClosedError | None = None
        self.initialize_result: InitializeResult | None = None

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self, *, timeout: float) -> InitializeResult:
        """Start the process and complete the initialize handshake."""
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            # The shim may run without a console; don't flash one up.
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            # A terminal's Ctrl-C reaches the shim, which stops the driver
            # itself, in order.
            kwargs["start_new_session"] = True
        self._process = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            limit=STDOUT_LINE_LIMIT_BYTES,
            **kwargs,
        )
        self._reader = asyncio.create_task(self._read_stdout(), name="mcp-stdout")
        self._stderr_reader = asyncio.create_task(self._read_stderr(), name="mcp-stderr")
        try:
            result = await self.request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": self._client_info,
                },
                timeout=timeout,
            )
            # Checked before the process is handed over: an error past this
            # block would leave it running, unheld.
            server = result.get("serverInfo") or {}
            capabilities = result.get("capabilities") or {}
            if not isinstance(server, dict) or not isinstance(capabilities, dict):
                raise McpError(
                    INTERNAL_ERROR,
                    "initialize answered with a serverInfo or capabilities that isn't an object",
                )
            await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except BaseException:
            await self.close()
            raise
        self.initialize_result = InitializeResult(
            protocol_version=str(result.get("protocolVersion") or ""),
            server_name=str(server.get("name") or ""),
            server_version=str(server.get("version") or ""),
            instructions=str(result.get("instructions") or ""),
            capabilities=dict(capabilities),
        )
        return self.initialize_result

    @property
    def alive(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
            and self._closed_error is None
        )

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def stderr_tail(self, lines: int = 20) -> str:
        return "\n".join(list(self._stderr_tail)[-lines:])

    async def close(self, *, now: bool = False) -> None:
        """Stop the server: close its stdin (an MCP server's cue to exit),
        then terminate it, then kill it, each after a grace period. `now`
        terminates it at once, for a server busy with a request it won't
        stop: it reads its stdin only between requests."""
        process = self._process
        if process is not None and process.returncode is None:
            exited = False
            if not now:
                if process.stdin is not None and not process.stdin.is_closing():
                    process.stdin.close()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(process.wait(), CLOSE_GRACE_SECONDS)
                    exited = True
            if not exited:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), KILL_GRACE_SECONDS)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(process.wait(), KILL_GRACE_SECONDS)
        self._fail_pending(McpClosedError("the driver was stopped"))
        for task in (self._reader, self._stderr_reader):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    # ── Requests ─────────────────────────────────────────────────────

    async def request(
        self, method: str, params: dict[str, Any] | None, *, timeout: float
    ) -> dict[str, Any]:
        """Send one request and wait for its result, both within `timeout`
        seconds: a driver that stops reading its input holds the write. A
        request that times out or is cancelled is cancelled at the server
        too, as MCP has it. cua-driver 0.38 runs one request at a time and
        drops that notice, so CuaDriverBridge stops the driver as well."""
        if self._closed_error is not None:
            raise self._closed_error
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            async with asyncio.timeout(timeout):
                await self._send(message)
                return await future
        except (TimeoutError, asyncio.CancelledError) as exc:
            reason = "timed out" if isinstance(exc, TimeoutError) else "cancelled"
            self._notify_cancelled(request_id, reason)
            raise
        finally:
            self._pending.pop(request_id, None)

    async def list_tools(self, *, timeout: float) -> list[dict[str, Any]]:
        """Every page of tools/list, within `timeout` seconds in all. A server
        that repeats a cursor, or pages past MAX_TOOL_LIST_PAGES, is broken:
        following it would hold the caller for good."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(MAX_TOOL_LIST_PAGES):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"tools/list took more than {timeout:g} seconds")
            # Each page gets what is left, rather than a timeout around the
            # loop, whose cancel Python 3.11's wait_for can drop.
            result = await self.request(
                "tools/list", {"cursor": cursor} if cursor else {}, timeout=remaining
            )
            tools.extend(t for t in result.get("tools") or [] if isinstance(t, dict))
            next_cursor = result.get("nextCursor")
            if next_cursor is None or next_cursor == "":
                return tools
            if not isinstance(next_cursor, str) or next_cursor in seen:
                raise McpError(INTERNAL_ERROR, "tools/list repeated a cursor or sent a bad one")
            seen.add(next_cursor)
            cursor = next_cursor
        raise McpError(INTERNAL_ERROR, f"tools/list ran past {MAX_TOOL_LIST_PAGES} pages")

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, timeout: float
    ) -> dict[str, Any]:
        return await self.request(
            "tools/call", {"name": name, "arguments": arguments}, timeout=timeout
        )

    # ── Plumbing ─────────────────────────────────────────────────────

    async def _send(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise McpClosedError("the driver is not running")
        line = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        async with self._write_lock:
            try:
                process.stdin.write(line)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise self._closed(f"the driver closed its input: {exc}") from exc

    def _notify_cancelled(self, request_id: int, reason: str) -> None:
        """Queue the cancel notice without waiting for the driver to read it:
        one that has stopped reading would hold the caller, and the bridge
        then stops it anyway. A line is queued whole, after any other."""
        process = self._process
        if not self.alive or process is None or process.stdin is None:
            return
        notice = {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": request_id, "reason": reason},
        }
        with contextlib.suppress(Exception):
            process.stdin.write((json.dumps(notice, separators=(",", ":")) + "\n").encode())

    async def _read_stdout(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            while True:
                try:
                    line = await process.stdout.readline()
                except ValueError:
                    # A line past the limit: the stream can't be resynced.
                    self._fail_pending(
                        self._closed("the driver sent a message over the size limit")
                    )
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    return
                if not line:
                    break
                await self._dispatch(line)
        finally:
            returncode = process.returncode
            if returncode is None:
                with contextlib.suppress(TimeoutError):
                    returncode = await asyncio.wait_for(process.wait(), KILL_GRACE_SECONDS)
            tail = self.stderr_tail(5)
            detail = f" (exit code {returncode})" if returncode is not None else ""
            self._fail_pending(
                self._closed(f"the driver exited{detail}" + (f": {tail}" if tail else ""))
            )

    async def _dispatch(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except ValueError:
            logger.debug("cua-driver wrote a non-JSON line: %r", line[:200])
            return
        if not isinstance(message, dict):
            return
        method = message.get("method")
        if method is None:
            request_id = message.get("id")
            future = self._pending.get(request_id) if isinstance(request_id, int) else None
            if future is None or future.done():
                return
            error = message.get("error")
            if isinstance(error, dict):
                future.set_exception(
                    McpError(
                        int(error.get("code") or 0),
                        str(error.get("message") or "error"),
                        error.get("data"),
                    )
                )
            else:
                result = message.get("result")
                future.set_result(result if isinstance(result, dict) else {})
            return
        if "id" in message:
            await self._answer_server_request(message["id"], str(method))
        elif method != "notifications/message":
            logger.debug("cua-driver notification: %s", method)

    async def _answer_server_request(self, request_id: Any, method: str) -> None:
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        if method == "ping":
            reply["result"] = {}
        elif method == "roots/list":
            reply["result"] = {"roots": []}
        else:
            reply["error"] = {"code": METHOD_NOT_FOUND, "message": f"{method} is not supported"}
        with contextlib.suppress(McpClosedError):
            await self._send(reply)

    async def _read_stderr(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        while True:
            try:
                line = await process.stderr.readline()
            except ValueError:
                continue
            if not line:
                return
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                self._stderr_tail.append(text)
                logger.debug("cua-driver: %s", text)

    def _closed(self, reason: str) -> McpClosedError:
        if self._closed_error is None:
            self._closed_error = McpClosedError(reason)
        return self._closed_error

    def _fail_pending(self, error: Exception) -> None:
        if self._closed_error is None and isinstance(error, McpClosedError):
            self._closed_error = error
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)


__all__ = [
    "INVALID_PARAMS",
    "MCP_PROTOCOL_VERSION",
    "InitializeResult",
    "McpClosedError",
    "McpError",
    "StdioMcpClient",
]
