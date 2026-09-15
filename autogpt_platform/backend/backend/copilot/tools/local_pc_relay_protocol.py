"""Types, keys, and envelope validation for the Local PC Redis relay."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator
from typing import Any, Protocol

from pydantic import BaseModel, Field

MAX_ENVELOPE_BYTES = 16 * 1024 * 1024
PLATFORM_REQUEST_TYPES = frozenset(
    {
        "EXECUTE_COMMAND",
        "FILE_READ",
        "FILE_WRITE",
        "FILE_STAT",
        "FILE_LIST",
        "FILE_DELETE",
        "FILE_MOVE",
        "SCREENSHOT_REQUEST",
        "INPUT_ACTION",
        "CURSOR_POSITION_REQUEST",
        "DISPLAY_INFO_REQUEST",
        "WINDOW_LIST_REQUEST",
        "WINDOW_FOCUS",
        "APP_LIST_REQUEST",
        "APP_LAUNCH",
        "CLIPBOARD_READ",
        "CLIPBOARD_WRITE",
        "PERMISSIONS_CHECK_REQUEST",
        "LOCAL_LLM_COMPLETION",
        "REQUEST_RECORDING_CONSENT",
        "START_RECORDING",
        "STOP_RECORDING",
        "RECORDING_FETCH",
        "APPLY_RECORDING_REVIEW",
        "DIRECTORY_LIST_REQUEST",
        "ATTACH_SESSION",
        "ACTIVATE_SESSION",
        "RESTORE_SESSION",
        "DETACH_SESSION",
        "PING",
    }
)
SHIM_RESPONSE_TYPES = frozenset(
    {
        "COMMAND_RESULT",
        "FILE_CONTENTS",
        "FILE_STAT_RESPONSE",
        "FILE_LIST_RESPONSE",
        "SCREENSHOT_RESPONSE",
        "CURSOR_POSITION_RESPONSE",
        "DISPLAY_INFO_RESPONSE",
        "WINDOW_LIST_RESPONSE",
        "APP_LIST_RESPONSE",
        "CLIPBOARD_READ_RESPONSE",
        "PERMISSIONS_CHECK_RESPONSE",
        "LOCAL_LLM_COMPLETION_CHUNK",
        "LOCAL_LLM_COMPLETION_RESPONSE",
        "RECORDING_CONSENT_RESULT",
        "RECORDING_STARTED",
        "RECORDING_SUMMARY",
        "RECORDING_DATA",
        "RECORDING_REVIEW_APPLIED",
        "RECORDING_STEP",
        "DIRECTORY_LIST_RESPONSE",
        "SESSION_ATTACHED",
        "SESSION_ACTIVATED",
        "SESSION_RESTORED",
        "STATUS",
        "ACK",
        "ERROR",
        "PONG",
    }
)
EXPECTED_RESPONSE_TYPES: dict[str, frozenset[str]] = {
    "EXECUTE_COMMAND": frozenset({"COMMAND_RESULT"}),
    "FILE_READ": frozenset({"FILE_CONTENTS"}),
    "FILE_WRITE": frozenset({"ACK"}),
    "FILE_STAT": frozenset({"FILE_STAT_RESPONSE"}),
    "FILE_LIST": frozenset({"FILE_LIST_RESPONSE"}),
    "FILE_DELETE": frozenset({"ACK"}),
    "FILE_MOVE": frozenset({"ACK"}),
    "SCREENSHOT_REQUEST": frozenset({"SCREENSHOT_RESPONSE"}),
    "INPUT_ACTION": frozenset({"ACK"}),
    "CURSOR_POSITION_REQUEST": frozenset({"CURSOR_POSITION_RESPONSE"}),
    "DISPLAY_INFO_REQUEST": frozenset({"DISPLAY_INFO_RESPONSE"}),
    "WINDOW_LIST_REQUEST": frozenset({"WINDOW_LIST_RESPONSE"}),
    "WINDOW_FOCUS": frozenset({"ACK"}),
    "APP_LIST_REQUEST": frozenset({"APP_LIST_RESPONSE"}),
    "APP_LAUNCH": frozenset({"ACK"}),
    "CLIPBOARD_READ": frozenset({"CLIPBOARD_READ_RESPONSE"}),
    "CLIPBOARD_WRITE": frozenset({"ACK"}),
    "PERMISSIONS_CHECK_REQUEST": frozenset({"PERMISSIONS_CHECK_RESPONSE"}),
    "LOCAL_LLM_COMPLETION": frozenset(
        {"LOCAL_LLM_COMPLETION_CHUNK", "LOCAL_LLM_COMPLETION_RESPONSE"}
    ),
    "REQUEST_RECORDING_CONSENT": frozenset({"RECORDING_CONSENT_RESULT"}),
    "START_RECORDING": frozenset({"RECORDING_STARTED"}),
    "STOP_RECORDING": frozenset({"RECORDING_SUMMARY"}),
    "RECORDING_FETCH": frozenset({"RECORDING_DATA"}),
    "APPLY_RECORDING_REVIEW": frozenset({"RECORDING_REVIEW_APPLIED"}),
    "DIRECTORY_LIST_REQUEST": frozenset({"DIRECTORY_LIST_RESPONSE"}),
    "ATTACH_SESSION": frozenset({"SESSION_ATTACHED"}),
    "ACTIVATE_SESSION": frozenset({"SESSION_ACTIVATED"}),
    "RESTORE_SESSION": frozenset({"SESSION_RESTORED"}),
    "DETACH_SESSION": frozenset({"ACK"}),
    "PING": frozenset({"PONG"}),
}
UNSOLICITED_RESPONSE_TYPES = frozenset({"STATUS", "RECORDING_STEP"})
NONTERMINAL_RESPONSE_TYPES = frozenset({"LOCAL_LLM_COMPLETION_CHUNK"})
_REPLY_ID_PATTERN = r"^[0-9a-f]{32}$"


class RelayPresence(BaseModel):
    session_id: str
    connection_id: str
    user_id: str
    client_id: str
    hello: dict[str, Any] = Field(default_factory=dict)
    expires_at: float


class RelayRequestTarget(BaseModel):
    request_type: str
    reply_id: str = Field(pattern=_REPLY_ID_PATTERN)


class TextTransport(Protocol):
    async def send_text(self, data: str) -> None: ...

    def iter_text(self) -> AsyncIterator[str]: ...

    async def close(self) -> None: ...


class RelayWebSocket(TextTransport, Protocol):
    async def close(self, code: int = 1000, reason: str = "") -> None: ...


class RelayBackend(Protocol):
    async def wait_for_presence(
        self, session_id: str, *, timeout: float
    ) -> RelayPresence: ...

    async def get_presence(self, session_id: str) -> RelayPresence | None: ...

    async def open_transport(self, presence: RelayPresence) -> TextTransport: ...

    async def serve_websocket(
        self,
        session_id: str,
        websocket: RelayWebSocket,
        *,
        hello: dict[str, Any],
        user_id: str,
        client_id: str,
        connection_id: str | None = None,
    ) -> None: ...

    async def revoke_owner(
        self, user_id: str, client_id: str | None, *, reason: str
    ) -> int: ...


class RelayConnectionReplaced(ConnectionError):
    pass


class RelayBacklogExceeded(ConnectionError):
    pass


def presence_key(session_id: str) -> str:
    return f"local-executor:{{{_session_tag(session_id)}}}:presence"


def owner_presence_index_key(
    user_id: str,
    client_id: str | None,
    connection_kind: str | None = None,
) -> str:
    scope = json.dumps([user_id, client_id, connection_kind], separators=(",", ":"))
    digest = hashlib.sha256(scope.encode()).hexdigest()
    return f"local-executor:owner:{{{digest}}}:presences"


def machine_scope_id(user_id: str, machine_id: str) -> str:
    """Return the opaque relay namespace for one user's installed machine."""
    digest = hashlib.sha256(f"{user_id}\0{machine_id}".encode()).hexdigest()
    return f"machine-{digest}"


def stream_key(session_id: str, connection_id: str, direction: str) -> str:
    return f"local-executor:{{{_session_tag(session_id)}}}:{connection_id}:{direction}"


def response_stream_key(session_id: str, connection_id: str, reply_id: str) -> str:
    if re.fullmatch(_REPLY_ID_PATTERN, reply_id) is None:
        raise ValueError("Local executor request has an invalid reply id")
    return stream_key(session_id, connection_id, f"responses:{reply_id}")


def as_text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def decode_stream_entries(result: Any) -> list[tuple[str, dict[str, str]]]:
    return [
        (
            as_text(message_id),
            {as_text(key): as_text(value) for key, value in fields.items()},
        )
        for _stream, messages in result or []
        for message_id, fields in messages
    ]


def validate_envelope(raw: str, allowed_types: frozenset[str]) -> dict[str, Any]:
    if len(raw.encode()) > MAX_ENVELOPE_BYTES:
        raise ValueError("Local executor envelope exceeds the relay size limit")
    message = json.loads(raw)
    if not isinstance(message, dict):
        raise ValueError("Local executor envelope must be an object")
    if message.get("type") not in allowed_types:
        raise ValueError("Local executor envelope has an invalid message type")
    if not isinstance(message.get("id"), str) or not message["id"]:
        raise ValueError("Local executor envelope must have a non-empty string id")
    if not isinstance(message.get("payload"), dict):
        raise ValueError("Local executor envelope payload must be an object")
    return message


def register_pending_request(
    message: dict[str, Any],
    reply_id: str,
    pending_requests: dict[str, RelayRequestTarget],
) -> RelayRequestTarget:
    message_id = message["id"]
    if message_id in pending_requests:
        raise ValueError("Local executor request reused an outstanding correlation id")
    target = RelayRequestTarget(request_type=message["type"], reply_id=reply_id)
    pending_requests[message_id] = target
    return target


def validate_response_correlation(
    message: dict[str, Any], pending_requests: dict[str, RelayRequestTarget]
) -> RelayRequestTarget | None:
    message_type = message["type"]
    if message_type in UNSOLICITED_RESPONSE_TYPES:
        return None

    message_id = message["id"]
    target = pending_requests.get(message_id)
    if target is None:
        raise ValueError("Local executor response has no outstanding request")
    expected_types = EXPECTED_RESPONSE_TYPES[target.request_type]
    if message_type != "ERROR" and message_type not in expected_types:
        raise ValueError(
            f"Local executor returned {message_type} for {target.request_type}"
        )
    if message_type not in NONTERMINAL_RESPONSE_TYPES:
        pending_requests.pop(message_id, None)
    return target


def _session_tag(session_id: str) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()[:32]
