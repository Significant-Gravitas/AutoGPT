"""
Endpoints for the autogpt-local-executor shim.

- ``/ws/local-executor/{session_id}`` (WebSocket): the shim dials in here.
  Auth: bearer access token validated via ``introspect_token``. On success
  the WebSocket is registered in ``ShimConnectionManager`` so
  ``LocalPCShim.for_session()`` can find it.

- ``/api/copilot/sessions/{session_id}/executor`` (GET): the frontend
  ``LocalPCBadge`` polls this to render live shim metadata (platform, arch,
  workspace, capabilities) when a shim is connected for the session. Auth:
  Supabase user session via ``auth.get_user_id``.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from autogpt_libs import auth
from fastapi import APIRouter, Security, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.copilot.tools.local_pc_shim import ShimHello, get_shim_manager
from backend.data.auth.oauth import introspect_token

logger = logging.getLogger(__name__)

router = APIRouter()


class ExecutorStatus(BaseModel):
    """Per-session executor metadata for the frontend.

    ``kind`` is the only field guaranteed present. Everything else is
    populated only when ``kind == "shim"`` and a shim is currently
    connected for the session on this worker. (Multi-worker accuracy
    needs a Redis-backed registry — follow-up; per-worker is enough to
    drive the UI when there's one worker, which is most dev / smoke
    setups today.)
    """

    kind: Literal["shim", "none"]
    platform: str | None = None
    arch: str | None = None
    allowed_root: str | None = None
    machine_id: str | None = None
    shim_version: str | None = None
    capabilities: list[str] | None = None
    computer_use_features: list[str] | None = None


@router.get(
    "/api/copilot/sessions/{session_id}/executor",
    response_model=ExecutorStatus,
    tags=["copilot", "local-executor"],
)
async def get_session_executor(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> ExecutorStatus:
    """Return executor metadata for ``session_id`` if a shim is connected.

    Returns ``{kind: "none"}`` when the session isn't routed to a shim on
    this worker. The frontend treats that as "no shim" and falls back to
    the static "Local PC mode" pill if the LD flag is on but no shim has
    handshaken yet.
    """
    hello = get_shim_manager().get_hello(session_id)
    if hello is None:
        return ExecutorStatus(kind="none")
    return ExecutorStatus(
        kind="shim",
        platform=hello.platform or None,
        arch=hello.arch or None,
        allowed_root=hello.allowed_root or None,
        machine_id=hello.machine_id or None,
        shim_version=hello.shim_version or None,
        capabilities=hello.capabilities or None,
        computer_use_features=hello.computer_use_features or None,
    )


@router.websocket("/ws/local-executor/{session_id}")
async def local_executor_ws(session_id: str, websocket: WebSocket) -> None:
    token = websocket.query_params.get("token", "")
    if not token:
        await websocket.close(code=4401, reason="Missing token")
        return

    try:
        token_info = await introspect_token(token, token_type_hint="access_token")
        if not token_info or not token_info.get("active"):
            await websocket.close(code=4401, reason="Invalid or expired token")
            return
    except Exception:
        logger.exception(
            "[LocalPC] Token introspection failed for session %s", session_id[:12]
        )
        await websocket.close(code=4500, reason="Auth error")
        return

    await websocket.accept()

    # Handshake: expect HELLO, send HELLO_ACK
    try:
        import json
        import time
        import uuid

        raw = await websocket.receive_text()
        hello_msg = json.loads(raw)
        if hello_msg.get("type") != "HELLO":
            await websocket.close(code=4400, reason="Expected HELLO")
            return

        hello_payload = hello_msg.get("payload", {}) or {}
        hello = ShimHello.from_payload(hello_payload)
        ack = {
            "type": "HELLO_ACK",
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "payload": {
                "session_id": session_id,
                "granted_capabilities": hello.capabilities,
                "server_version": "0.0.1",
            },
        }
        await websocket.send_text(json.dumps(ack))
    except Exception:
        logger.exception("[LocalPC] Handshake failed for session %s", session_id[:12])
        await websocket.close(code=4500, reason="Handshake error")
        return

    manager = get_shim_manager()
    manager.register(session_id, websocket, hello)
    logger.info(
        "[LocalPC] Shim connected for session %s (platform=%s arch=%s machine=%s)",
        session_id[:12],
        hello.platform or "?",
        hello.arch or "?",
        hello.machine_id[:12] if hello.machine_id else "?",
    )

    try:
        # Keep connection alive; LocalPCShim._recv_loop drives the traffic
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        manager.unregister(session_id)
        logger.info("[LocalPC] Shim disconnected for session %s", session_id[:12])
