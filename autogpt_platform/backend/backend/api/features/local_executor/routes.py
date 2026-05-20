"""
WebSocket endpoint for the autogpt-local-executor shim.

The shim dials in with:
    ws://<host>/ws/local-executor/<session_id>?token=<access_token>

Auth: the token is validated via introspect_token() before the connection
is accepted. On success the WebSocket is registered in ShimConnectionManager
so LocalPCShim.for_session() can find it.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.copilot.tools.local_pc_shim import ShimHello, get_shim_manager
from backend.data.auth.oauth import introspect_token

logger = logging.getLogger(__name__)

router = APIRouter()


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
        logger.exception("[LocalPC] Token introspection failed for session %s", session_id[:12])
        await websocket.close(code=4500, reason="Auth error")
        return

    await websocket.accept()

    # Handshake: expect HELLO, send HELLO_ACK
    try:
        import json, time, uuid
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
