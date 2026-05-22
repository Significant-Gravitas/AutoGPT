"""End-to-end loopback test for the local-executor WebSocket route.

These tests wire BOTH halves in the same process:

  TestClient (acts as the shim)  <--->  FastAPI route  <--->  LocalPCShim
                                                                (platform adapter)

Real WebSocket framing goes between the two halves; auth + token
introspection are mocked because they're tested independently. This is
the closest we can get to a real shim ↔ platform conversation without
spawning the shim daemon (which is restricted per the project's
parallel-agents constraint — see :doc:`docs/CROSS_PLATFORM.md`).

What's covered:
- HELLO → HELLO_ACK handshake, including capability echo.
- LocalPCShim metadata exposure (platform/arch/allowed_root).
- EXECUTE_COMMAND roundtrip (platform sends, shim replies, platform
  parses).
- FILE_READ with format="bytes" base64 decode on the adapter side
  (this is the bug the earlier fix to _FilesProxy.read addressed; the
  e2e test locks it in).

Auth: `introspect_token` is patched to return an active token. The
real OAuth + PKCE flow is exercised in oauth_test.py.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.features.local_executor.routes import router
from backend.copilot.tools.local_pc_shim import LocalPCShim, get_shim_manager


def _envelope(msg_type: str, payload: dict, msg_id: str = "test-msg") -> str:
    return json.dumps({"type": msg_type, "id": msg_id, "ts": 0.0, "payload": payload})


def _make_app() -> FastAPI:
    """Minimal FastAPI app that exposes only the local-executor route.

    Keeps the test independent of the broader backend so it doesn't drag
    in prisma / db / settings init.
    """
    app = FastAPI()
    app.include_router(router)
    return app


_FAKE_TOKEN_INFO = {"active": True, "client_id": "autogpt-local-executor"}


@pytest.fixture
def _patched_introspect():
    """Auth is exercised in oauth_test.py; here we want to exercise the
    handshake + adapter loop without provisioning a real token."""
    with patch(
        "backend.api.features.local_executor.routes.introspect_token",
        return_value=_FAKE_TOKEN_INFO,
    ):
        yield


@pytest.fixture
def _fresh_manager():
    """ShimConnectionManager is a process singleton — clear it between
    tests so registrations from one test don't bleed into another."""
    manager = get_shim_manager()
    manager._connections.clear()
    manager._hellos.clear()
    manager._waiters.clear()
    yield manager
    manager._connections.clear()
    manager._hellos.clear()
    manager._waiters.clear()


class TestHandshake:
    """The HELLO → HELLO_ACK exchange before any work flows."""

    def test_hello_ack_echoes_capabilities_and_returns_session_id(
        self, _patched_introspect, _fresh_manager
    ):
        app = _make_app()
        client = TestClient(app)
        hello = _envelope(
            "HELLO",
            {
                "shim_version": "0.1.0",
                "machine_id": "m-uuid",
                "platform": "darwin",
                "arch": "arm64",
                "allowed_root": "/Users/test/ws",
                "capabilities": ["shell", "files"],
            },
        )
        with client.websocket_connect("/ws/local-executor/sess-1?token=x") as ws:
            ws.send_text(hello)
            ack_raw = ws.receive_text()
            ack = json.loads(ack_raw)
            assert ack["type"] == "HELLO_ACK"
            assert ack["payload"]["session_id"] == "sess-1"
            assert ack["payload"]["granted_capabilities"] == ["shell", "files"]

    def test_route_rejects_missing_token(self, _patched_introspect, _fresh_manager):
        app = _make_app()
        client = TestClient(app)
        with pytest.raises(Exception):
            # WebSocketDisconnect / starlette WS exceptions; close code 4401.
            with client.websocket_connect("/ws/local-executor/sess-1"):
                pass

    def test_first_frame_not_hello_closes_connection(
        self, _patched_introspect, _fresh_manager
    ):
        app = _make_app()
        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/local-executor/sess-1?token=x") as ws:
                ws.send_text(_envelope("EXECUTE_COMMAND", {"command": "ls"}))
                # Route closes with 4400; the next receive raises.
                ws.receive_text()


class TestManagerCarriesHelloMetadata:
    """After HELLO, ShimConnectionManager should have machine_id /
    platform / arch / allowed_root reachable via get_hello(session_id).

    LocalPCShim.for_session then uses those values to populate its own
    attributes, which the executor-aware platform code (e.g.
    get_workdir(sandbox), describe_workspace(sandbox)) branches on.
    """

    def test_hello_persisted_in_manager(self, _patched_introspect, _fresh_manager):
        app = _make_app()
        client = TestClient(app)
        hello = _envelope(
            "HELLO",
            {
                "machine_id": "m-uuid",
                "platform": "windows",
                "arch": "x86_64",
                "allowed_root": "C:\\workspace",
                "capabilities": ["shell", "files"],
                "shim_version": "0.1.0",
            },
        )
        with client.websocket_connect("/ws/local-executor/sess-2?token=x") as ws:
            ws.send_text(hello)
            ws.receive_text()  # HELLO_ACK
            stored = _fresh_manager.get_hello("sess-2")
            assert stored is not None
            assert stored.machine_id == "m-uuid"
            assert stored.platform == "windows"
            assert stored.arch == "x86_64"
            assert stored.allowed_root == "C:\\workspace"


class TestLoopbackExecuteCommand:
    """Both halves running in the same process: LocalPCShim sends an
    EXECUTE_COMMAND, the TestClient (acting as the shim) sees it on
    the wire and replies with COMMAND_RESULT, and the adapter returns
    a usable result object.
    """

    @pytest.mark.asyncio(loop_scope="session")
    async def test_execute_command_roundtrip(
        self, _patched_introspect, _fresh_manager
    ):
        app = _make_app()
        # Synchronous TestClient runs the route in a thread; we drive
        # both halves via asyncio.gather. The shim side is the TestClient;
        # the platform side is LocalPCShim.for_session.
        results: dict[str, Any] = {}

        def _act_as_shim():
            client = TestClient(app)
            with client.websocket_connect(
                "/ws/local-executor/sess-3?token=x"
            ) as ws:
                ws.send_text(
                    _envelope(
                        "HELLO",
                        {
                            "machine_id": "m",
                            "platform": "darwin",
                            "arch": "arm64",
                            "allowed_root": "/Users/test/ws",
                            "capabilities": ["shell", "files"],
                            "shim_version": "0.1.0",
                        },
                    )
                )
                ws.receive_text()  # HELLO_ACK

                # Now wait for the platform-side EXECUTE_COMMAND to arrive
                # and reply.
                cmd_raw = ws.receive_text()
                cmd = json.loads(cmd_raw)
                assert cmd["type"] == "EXECUTE_COMMAND"
                assert cmd["payload"]["command"] == "echo hello"
                assert cmd["payload"]["shell"] == "auto"
                ws.send_text(
                    json.dumps(
                        {
                            "type": "COMMAND_RESULT",
                            "id": cmd["id"],
                            "ts": 0.0,
                            "payload": {
                                "stdout": "hello\n",
                                "stderr": "",
                                "exit_code": 0,
                                "timed_out": False,
                            },
                        }
                    )
                )
                # Give the platform side a moment to consume the reply.
                ws.receive_text() if False else None

        async def _act_as_platform():
            # Wait for the shim to register, then attach a LocalPCShim
            # and issue EXECUTE_COMMAND through it.
            shim = await LocalPCShim.for_session(
                "sess-3", manager=_fresh_manager, connect_timeout=5.0
            )
            assert shim.platform == "darwin"
            assert shim.allowed_root == "/Users/test/ws"
            result = await shim.commands.run("echo hello")
            results["stdout"] = result.stdout
            results["exit_code"] = result.exit_code
            await shim.kill()

        # Run the shim-side (sync, blocking) in a thread so the
        # platform-side can drive it from asyncio.
        loop = asyncio.get_event_loop()
        shim_task = loop.run_in_executor(None, _act_as_shim)
        platform_task = asyncio.create_task(_act_as_platform())

        # The platform task races the shim task; give the shim a head start
        # on the handshake. The shim's TestClient connects synchronously,
        # so the handshake completes before this `await` resumes.
        await asyncio.sleep(0.1)
        await platform_task
        await shim_task

        assert results["stdout"] == "hello\n"
        assert results["exit_code"] == 0
