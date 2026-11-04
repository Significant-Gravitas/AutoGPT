"""WebSocket authentication and lifecycle for the Local PC executor."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from asyncio import sleep

from fastapi import APIRouter, WebSocket
from pydantic import BaseModel, ValidationError

from backend.api.features.local_executor.gating import is_local_executor_enabled
from backend.api.features.local_executor.models import SessionID
from backend.copilot.config import ChatConfig
from backend.copilot.model import (
    LocalExecutionTargetMetadata,
    get_chat_session_metadata,
)
from backend.copilot.tools.local_pc_metrics import (
    record_handshake_failure,
    record_shim_connected,
    record_shim_disconnected,
)
from backend.copilot.tools.local_pc_relay import get_local_pc_relay
from backend.copilot.tools.local_pc_relay_protocol import machine_scope_id
from backend.copilot.tools.local_pc_shim import (
    COMMAND_TIMEOUT_SECONDS,
    MAX_FILE_SIZE_BYTES,
    ShimHello,
    get_shim_manager,
)
from backend.data.auth.oauth import introspect_token

logger = logging.getLogger(__name__)

router = APIRouter()

# 2.0: computer use relays the shim's cua-driver tools
# (autogpt-local-executor docs/PROTOCOL.md, Versioning).
LOCAL_EXECUTOR_PROTOCOL_VERSION = "2.0"
MAX_HELLO_BYTES = 64 * 1024
HELLO_TIMEOUT_SECONDS = 10.0
_TOKEN_RECHECK_SECONDS = 60.0
# The OAuth scope a Local PC connection needs, and the only one it needs.
LOCAL_EXECUTOR_SCOPE = "CONNECT_LOCAL_EXECUTOR"
MAX_CONCURRENT_REQUESTS = 4
SERVER_SUPPORTED_CAPABILITIES = frozenset(
    {
        "shell",
        "files",
        "computer_use",
        "local_llm",
        "recording",
        "directory_browse",
    }
)


class ShimIdentity(BaseModel):
    user_id: str
    client_id: str
    expires_at: int
    execution_target: LocalExecutionTargetMetadata | None = None


@router.websocket("/ws/local-executor")
async def local_executor_machine_ws(websocket: WebSocket) -> None:
    """Keep one authenticated machine-control channel online across chats."""
    identity = await _authenticate_machine_connection(websocket)
    if identity is None:
        return

    await websocket.accept()
    connection_id = str(uuid.uuid4())
    hello = await _receive_hello(
        None,
        websocket,
        allow_null_root=True,
        connection_id=connection_id,
    )
    if hello is None:
        return

    scope_id = machine_scope_id(identity.user_id, hello.machine_id)
    relay_hello = hello.model_dump(mode="json")
    relay_hello["allowed_root"] = None
    relay_hello["connection_kind"] = "machine"
    token_recheck = asyncio.Event()
    relay_task = asyncio.create_task(
        get_local_pc_relay().serve_websocket(
            scope_id,
            websocket,
            hello=relay_hello,
            user_id=identity.user_id,
            client_id=identity.client_id,
            connection_id=connection_id,
            on_token_recheck=token_recheck.set,
        )
    )
    expiry_task = asyncio.create_task(
        _close_when_token_ends(websocket, identity.expires_at, token_recheck)
    )
    try:
        await _serve_until_either_ends(relay_task, expiry_task)
    finally:
        await _stop_tasks(relay_task, expiry_task)


@router.websocket("/ws/local-executor/{session_id}")
async def local_executor_ws(
    session_id: SessionID,
    websocket: WebSocket,
) -> None:
    """Bind one authenticated, owner-matched shim to a Copilot session."""
    identity = await _authenticate_shim_session(session_id, websocket)
    if identity is None:
        return
    user_id = identity.user_id
    client_id = identity.client_id

    await websocket.accept()
    hello = await _receive_hello(
        session_id, websocket, execution_target=identity.execution_target
    )
    if hello is None:
        return

    manager = get_shim_manager()
    manager.remember_hello(session_id, hello)
    record_shim_connected(
        platform=hello.platform,
        arch=hello.arch,
        shim_version=hello.shim_version,
    )
    logger.info(
        "[LocalPC] Shim connected for session %s (platform=%s arch=%s machine=%s)",
        session_id[:12],
        hello.platform or "?",
        hello.arch or "?",
        hello.machine_id[:12] if hello.machine_id else "?",
    )

    token_recheck = asyncio.Event()
    relay_task = asyncio.create_task(
        manager.serve_websocket(
            session_id,
            websocket,
            hello,
            user_id=user_id,
            client_id=client_id,
            on_token_recheck=token_recheck.set,
        )
    )
    expiry_task = asyncio.create_task(
        _close_when_token_ends(websocket, identity.expires_at, token_recheck)
    )
    try:
        await _serve_until_either_ends(relay_task, expiry_task)
    finally:
        try:
            await _stop_tasks(relay_task, expiry_task)
        finally:
            # Even when the handler is cancelled while it stops them, so the
            # active-connection gauge stays balanced.
            record_shim_disconnected(platform=hello.platform, arch=hello.arch)
            logger.info("[LocalPC] Shim disconnected for session %s", session_id[:12])


async def _serve_until_either_ends(
    relay_task: asyncio.Task[None], expiry_task: asyncio.Task[None]
) -> None:
    """Return once the relay or the token watch ends, raising what it raised."""
    done, _ = await asyncio.wait(
        {relay_task, expiry_task}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in done:
        if not task.cancelled():
            task.result()


async def _stop_tasks(*tasks: asyncio.Task[None]) -> None:
    """Cancel *tasks* and wait for them to finish.

    asyncio.wait, unlike gather, never surfaces a child's CancelledError: if
    this handler is cancelled meanwhile, it raises its own. A server that
    cancels the handler (Starlette's TestClient does, as soon as it closes)
    gets back its own cancellation, which anyio recognises; a child's,
    raised through gather, could escape its cancel scope instead.
    """
    for task in tasks:
        task.cancel()
    unfinished = [task for task in tasks if not task.done()]
    if unfinished:
        await asyncio.wait(unfinished)
    for task in tasks:
        if not task.cancelled():
            # Retrieved, so asyncio does not log it as never retrieved.
            task.exception()


async def _authenticate_shim_session(
    session_id: str, websocket: WebSocket
) -> ShimIdentity | None:
    return await _authenticate_connection(websocket, session_id=session_id)


async def _authenticate_machine_connection(
    websocket: WebSocket,
) -> ShimIdentity | None:
    return await _authenticate_connection(websocket, session_id=None)


async def _authenticate_connection(
    websocket: WebSocket,
    *,
    session_id: str | None,
) -> ShimIdentity | None:
    token = _bearer_token(websocket)
    if token is None:
        record_handshake_failure("missing_token")
        await _deny(websocket, 401, "Missing bearer token", close_code=4401)
        return None

    try:
        token_info = await introspect_token(token, token_type_hint="access_token")
        allowed_client_id = ChatConfig().local_pc_executor_oauth_client_id
        now = int(time.time())
        if (
            not token_info.active
            or token_info.token_type != "access_token"
            or not token_info.user_id
            or token_info.client_id != allowed_client_id
            or token_info.exp is None
            or token_info.exp <= now
        ):
            record_handshake_failure("invalid_token")
            await _deny(websocket, 401, "Invalid or expired token", close_code=4401)
            return None
        if LOCAL_EXECUTOR_SCOPE not in (token_info.scopes or []):
            # A valid token without the scope: refreshing it keeps its scopes,
            # so 403, which the companion does not retry.
            record_handshake_failure("missing_scope")
            await _deny(
                websocket,
                403,
                "This computer's sign-in does not include the Local PC permission. "
                "Run `autogpt-shim auth` again.",
                close_code=4403,
            )
            return None
        user_id = token_info.user_id
        client_id = token_info.client_id
        assert client_id is not None
        execution_target = None
        if session_id is not None:
            session = await get_chat_session_metadata(session_id, user_id)
            if session is None or session.metadata.execution_target.kind != "local":
                record_handshake_failure("session_access_denied")
                await _deny(
                    websocket,
                    403,
                    "Session is not bound to this Local PC executor",
                    close_code=4403,
                )
                return None
            execution_target = session.metadata.execution_target
        if not await is_local_executor_enabled(user_id):
            record_handshake_failure("feature_disabled")
            await _deny(
                websocket,
                403,
                "Local PC executor is not enabled",
                close_code=4403,
            )
            return None
    except Exception:
        record_handshake_failure("auth_error")
        logger.exception(
            "[LocalPC] Authentication failed for %s",
            session_id[:12] if session_id else "machine control",
        )
        await _deny(websocket, 503, "Authentication unavailable", close_code=4500)
        return None

    return ShimIdentity(
        user_id=user_id,
        client_id=client_id,
        expires_at=token_info.exp,
        execution_target=execution_target,
    )


async def _deny(
    websocket: WebSocket,
    status_code: int,
    detail: str,
    *,
    close_code: int,
) -> None:
    if "websocket.http.response" not in websocket.scope.get("extensions", {}):
        await websocket.close(code=close_code, reason=detail)
        return

    body = json.dumps({"detail": detail}, separators=(",", ":")).encode()
    await websocket.send(
        {
            "type": "websocket.http.response.start",
            "status": status_code,
            "headers": [],
        }
    )
    await websocket.send(
        {
            "type": "websocket.http.response.body",
            "body": body,
            "more_body": False,
        }
    )


async def _close_when_token_ends(
    websocket: WebSocket, expires_at: int, recheck: asyncio.Event | None = None
) -> None:
    """Close the connection when its access token expires or stops being active.

    Revoking one of the owner's tokens sets ``recheck`` through Redis, so the
    check runs at once, and only a connection whose own token was revoked
    closes. The periodic check bounds how long a connection outlives a
    revocation that push missed."""
    while (remaining := expires_at - time.time()) > 0:
        await _wait_for_recheck(recheck, min(remaining, _TOKEN_RECHECK_SECONDS))
        if expires_at - time.time() > 0 and not await _token_still_active(websocket):
            # 4428 tells the companion not to reconnect.
            await websocket.close(code=4428, reason="Token revoked")
            return
    await websocket.close(code=4401, reason="Access token expired")


async def _wait_for_recheck(recheck: asyncio.Event | None, timeout: float) -> None:
    """Wait ``timeout`` seconds, or less if a revocation asks for a check."""
    if recheck is None:
        await sleep(timeout)
        return
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(recheck.wait(), timeout)
    recheck.clear()


async def _token_still_active(websocket: WebSocket) -> bool:
    token = _bearer_token(websocket)
    if token is None:
        return False
    try:
        info = await introspect_token(token, token_type_hint="access_token")
    except Exception:
        # A failed lookup says nothing about the token; its expiry still bounds
        # the connection, and the next check runs on schedule.
        logger.warning(
            "[LocalPC] Could not recheck a Local PC access token", exc_info=True
        )
        return True
    return info.active


def _bearer_token(websocket: WebSocket) -> str | None:
    authorization = websocket.headers.get("authorization", "")
    scheme, separator, credentials = authorization.partition(" ")
    if separator and scheme.lower() == "bearer" and credentials.strip():
        return credentials.strip()
    return None


async def _receive_hello(
    session_id: str | None,
    websocket: WebSocket,
    *,
    allow_null_root: bool = False,
    connection_id: str | None = None,
    execution_target: LocalExecutionTargetMetadata | None = None,
) -> ShimHello | None:
    try:
        raw = await asyncio.wait_for(
            websocket.receive_text(), timeout=HELLO_TIMEOUT_SECONDS
        )
        if len(raw.encode()) > MAX_HELLO_BYTES:
            record_handshake_failure("hello_too_large")
            await websocket.close(code=4400, reason="HELLO payload too large")
            return None
        message = json.loads(raw)
        if not isinstance(message, dict) or message.get("type") != "HELLO":
            record_handshake_failure("expected_hello")
            await websocket.close(code=4400, reason="Expected HELLO")
            return None
        message_id = message.get("id")
        if not isinstance(message_id, str) or not message_id:
            record_handshake_failure("invalid_hello_id")
            await websocket.close(code=4400, reason="Invalid HELLO correlation ID")
            return None
        payload = message.get("payload")
        if not isinstance(payload, dict):
            record_handshake_failure("invalid_hello")
            await websocket.close(code=4400, reason="Invalid HELLO payload")
            return None
        try:
            validation_payload = payload
            if allow_null_root:
                if payload.get("allowed_root") is not None:
                    raise ValueError("Machine-control HELLO allowed_root must be null")
                validation_payload = {**payload, "allowed_root": "."}
            hello = ShimHello.from_payload(validation_payload)
        except (ValidationError, ValueError):
            record_handshake_failure("invalid_hello")
            await websocket.close(code=4400, reason="Invalid HELLO payload")
            return None
        shim_major, shim_minor = _protocol_parts(hello.protocol_version)
        server_major, server_minor = _protocol_parts(LOCAL_EXECUTOR_PROTOCOL_VERSION)
        if shim_major != server_major:
            record_handshake_failure("protocol_version_mismatch")
            await websocket.close(code=4426, reason="Protocol major version mismatch")
            return None
        if execution_target is not None and (
            hello.machine_id != execution_target.machine_id
            or hello.allowed_root != execution_target.allowed_root
        ):
            record_handshake_failure("session_binding_mismatch")
            await websocket.close(
                code=4403, reason="Local PC executor does not match the session binding"
            )
            return None
        negotiated_protocol_version = f"{server_major}.{min(shim_minor, server_minor)}"
        granted_capabilities = list(
            dict.fromkeys(
                capability
                for capability in hello.capabilities
                if capability in SERVER_SUPPORTED_CAPABILITIES
            )
        )
        # Every 2.x shim speaks the machine-control protocol.
        if allow_null_root and "directory_browse" not in granted_capabilities:
            record_handshake_failure("machine_control_unsupported")
            await websocket.close(
                code=4426,
                reason="Machine control requires directory_browse",
            )
            return None
        hello = hello.model_copy(update={"capabilities": granted_capabilities})
        await websocket.send_text(
            json.dumps(
                {
                    "type": "HELLO_ACK",
                    "id": message_id,
                    "ts": time.time(),
                    "payload": {
                        "session_id": session_id,
                        "granted_capabilities": granted_capabilities,
                        "protocol_version": negotiated_protocol_version,
                        "max_file_size_bytes": MAX_FILE_SIZE_BYTES,
                        "command_timeout_seconds": COMMAND_TIMEOUT_SECONDS,
                        "max_concurrent": MAX_CONCURRENT_REQUESTS,
                        **(
                            {"connection_id": connection_id}
                            if connection_id is not None
                            else {}
                        ),
                    },
                }
            )
        )
        return hello
    except TimeoutError:
        record_handshake_failure("hello_timeout")
        await websocket.close(code=4408, reason="HELLO timeout")
        return None
    except Exception:
        record_handshake_failure("handshake_error")
        logger.exception(
            "[LocalPC] Handshake failed for %s",
            session_id[:12] if session_id else "machine control",
        )
        await websocket.close(code=4500, reason="Handshake error")
        return None


def _protocol_parts(version: str) -> tuple[int, int]:
    major, minor = version.split(".", 1)
    return int(major), int(minor)
