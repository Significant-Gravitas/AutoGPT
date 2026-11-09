"""
Platform-side binding for the autogpt-local-executor shim.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import unicodedata
import uuid
import weakref
from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.copilot.local_executor import LocalPCExecutorMarker

from .local_pc_errors import translate_shim_error
from .local_pc_llm import LocalLLMError
from .local_pc_llm import LocalLLMProxy as _LocalLLMProxy
from .local_pc_metrics import record_rpc, record_rpc_error, record_rpc_retry
from .local_pc_relay import get_local_pc_relay
from .local_pc_relay_protocol import (
    EXPECTED_RESPONSE_TYPES,
    RelayBackend,
    RelayWebSocket,
    TextTransport,
)
from .recording_models import (
    RecordingConsentResult,
    RecordingReviewApplied,
    RecordingSummary,
    TrajectoryStep,
    WorkflowRecording,
)

logger = logging.getLogger(__name__)

_LLM_STREAM_MAX_FRAMES = 64
_LLM_STREAM_MAX_FRAME_BYTES = 64 * 1024
_LLM_STREAM_MAX_CONCURRENT = 4

# The longest the companion runs one command. HELLO_ACK sends it, and the
# companion keeps the smaller of this and its own setting.
COMMAND_TIMEOUT_SECONDS = 30
# The most the companion reads or writes in one file message. HELLO_ACK sends
# it, and the companion keeps the smaller of this and its own setting.
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
# Time the companion needs to stop a command that ran out of time and send back
# what it printed.
_COMMAND_RESULT_GRACE_SECONDS = 10.0


# In-flight-on-disconnect semantics — see PROTOCOL.md
# "In-flight semantics on disconnect". Ops in this set are repeat-safe at the
# protocol level: re-issuing the same wire op produces the same observable
# state on the user's machine, so the platform adapter MAY auto-retry once
# after reconnect when a timeout/disconnect races a response.
_IDEMPOTENT_OPS: frozenset[str] = frozenset(
    {
        "FILE_READ",
        "FILE_STAT",
        "FILE_LIST",
        "COMPUTER_TOOLS_REQUEST",
        "RECORDING_FETCH",
    }
)


class OpUnconfirmedError(RuntimeError):
    """Raised when a non-idempotent wire op was sent but never acknowledged.

    The wire op left the platform but the platform-side `_rpc` either timed
    out waiting for a response or the WS dropped before the shim flushed
    one. The side effect may or may not have happened on the user's
    machine, so the platform MUST NOT auto-retry — the LLM owns the
    recovery decision (typically: probe state with an idempotent op like
    `FILE_STAT`, then re-issue if needed).

    Attributes:
        code: synthetic shim error code surfaced to the translator
            (``"WRITE_UNCONFIRMED"`` for FILE_WRITE, ``"OP_UNCONFIRMED"``
            for everything else).
        op: the wire op name (``"FILE_WRITE"``, ``"EXECUTE_COMMAND"``, ...).
        wire_id: the original wire-correlation `id` so callers can correlate
            audit-log entries on both sides.
    """

    def __init__(
        self,
        op: str,
        wire_id: str,
        *,
        code: str = "OP_UNCONFIRMED",
        message: str | None = None,
    ) -> None:
        super().__init__(message or f"[LocalPC] {op} unconfirmed (wire id={wire_id})")
        self.code = code
        self.op = op
        self.wire_id = wire_id


class WriteUnconfirmedError(OpUnconfirmedError):
    """Specialization of OpUnconfirmedError for FILE_WRITE.

    The bytes may or may not have hit disk on the shim host. Caller should
    `FILE_STAT` the target path to check actual state.
    """

    def __init__(self, wire_id: str, message: str | None = None) -> None:
        super().__init__(
            op="FILE_WRITE",
            wire_id=wire_id,
            code="WRITE_UNCONFIRMED",
            message=message,
        )


# Backpressure — see PROTOCOL.md §Concurrency. When the platform asks
# `_rpc` to send a wire op but the shim's most recent
# `pending_capacity` signal is 0, the call blocks on
# ``_capacity_available`` for at most this many seconds before raising
# ``ShimOverloadedError``. Matches the wire-level ``SHIM_OVERLOADED``
# semantic so the translator can render the same recovery hint either way.
_CAPACITY_WAIT_TIMEOUT_SECONDS: float = 30.0
_DUPLICATE_SESSION_CLOSE_CODE = 4427
_REVOCATION_OPERATION_TIMEOUT_SECONDS = 2.0
_MAX_RECORDING_BUFFERS = 8
_MAX_RECORDING_STEP_BUFFER = 256


class ShimOverloadedError(RuntimeError):
    """Raised proactively when the shim's pending capacity stays at 0 too long.

    Wire-level ``SHIM_OVERLOADED`` arrives as a normal `ERROR` envelope
    after the shim refuses an over-cap request. This client-side variant
    short-circuits before sending — once the platform learns the shim is
    full (from a prior response or STATUS frame), there's no value in
    spending a round-trip just to receive `SHIM_OVERLOADED` back. The
    error code matches the wire code so existing translator + retry logic
    handles both surfaces uniformly.
    """

    code = "SHIM_OVERLOADED"


class ShimProtocolError(RuntimeError):
    """Raised when a correlated shim response violates the wire contract."""


class ShimFileTooLargeError(OSError):
    """The companion refused a FILE_READ over its file size limit."""


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


def _translate_unconfirmed(
    exc: OpUnconfirmedError,
    shim: "LocalPCShim | None",
    *,
    fallback: str,
    extra_details: dict | None = None,
) -> str:
    """Build the LLM-friendly message for an OpUnconfirmedError.

    Shared by every non-idempotent proxy method so the catch+wrap shape
    in CommandsProxy / FilesProxy.delete+move / ComputerProxy mirrors
    FilesProxy.write — the LLM sees the same actionable English ("the
    op was sent but the connection dropped before the shim acknowledged
    — verify state with an idempotent probe and re-issue if needed")
    regardless of which proxy raised.
    """
    details: dict = {"op": exc.op}
    if extra_details:
        details.update(extra_details)
    return _friendly(
        {"code": exc.code, "message": str(exc), "details": details},
        shim,
        fallback,
    )


_shim_manager: "ShimConnectionManager | None" = None


def get_shim_manager() -> "ShimConnectionManager":
    global _shim_manager
    if _shim_manager is None:
        _shim_manager = ShimConnectionManager()
    return _shim_manager


def _hello_string_list(payload: dict, field_name: str) -> list[str]:
    value = payload.get(field_name, [])
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"HELLO.{field_name} must be a list of strings")
    return value


def _required_hello_string(payload: dict, field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"HELLO.{field_name} must be a non-empty string")
    return value


def _optional_hello_string(payload: dict, field_name: str, *, default: str = "") -> str:
    value = payload.get(field_name, default)
    if not isinstance(value, str):
        raise ValueError(f"HELLO.{field_name} must be a string")
    return value


def _validate_safe_hello_text(value: str, field_name: str) -> str:
    if "<" in value or ">" in value:
        raise ValueError(f"{field_name} cannot contain angle brackets")
    if any(
        unicodedata.category(character).startswith("C")
        or character in {"\u2028", "\u2029"}
        for character in value
    ):
        raise ValueError(f"{field_name} cannot contain control characters")
    return value


# cua-driver tool names, as the shim relays them (PROTOCOL.md, 2.0).
_COMPUTER_TOOL_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_MAX_COMPUTER_USE_TOOLS = 128
# How long the platform waits past a computer tool call's own deadline, so
# the shim's COMMAND_TIMEOUT arrives before the request is abandoned.
_COMPUTER_TOOL_CALL_MARGIN_SECONDS = 15.0
_DEFAULT_COMPUTER_TOOL_TIMEOUT_SECONDS = 120


class ComputerUseDriver(BaseModel):
    """The GUI driver behind a shim's computer use (HELLO.computer_use_driver)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str = Field(min_length=1, max_length=64)
    version: str = Field(min_length=1, max_length=64)
    runtime: Literal["direct", "service"]

    @field_validator("name", "version")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_safe_hello_text(value, "computer_use_driver")

    @property
    def label(self) -> str:
        return f"{self.name} {self.version}"


def _canonical_computer_use_tools(values: list[str]) -> list[str]:
    if len(values) > _MAX_COMPUTER_USE_TOOLS:
        raise ValueError("computer_use_tools contains too many entries")
    if any(_COMPUTER_TOOL_NAME.fullmatch(value) is None for value in values):
        raise ValueError("computer_use_tools contains an invalid tool name")
    return sorted(set(values))


def _hello_driver(payload: dict) -> ComputerUseDriver | None:
    raw = payload.get("computer_use_driver")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("computer_use_driver must be an object")
    return ComputerUseDriver.model_validate(raw)


class ShimHello(BaseModel):
    """HELLO payload captured by the route on connect, surfaced to LocalPCShim."""

    machine_id: str = Field(default="", max_length=128)
    display_name: str = Field(default="", max_length=128)
    platform: str = ""
    arch: str = ""
    shim_version: str = Field(default="", max_length=64)
    allowed_root: str = Field(default="", max_length=4096)
    capabilities: list[str] = Field(default_factory=list)
    screen_resolution: tuple[int, int] | None = None
    local_llm_models: list[str] = Field(default_factory=list)
    hardware_devices: list[dict] = Field(default_factory=list)
    computer_use_driver: ComputerUseDriver | None = None
    computer_use_tools: list[str] = Field(default_factory=list)
    recording_channels: list[str] = Field(default_factory=list)
    recording_routes: list[str] = Field(default_factory=list)
    protocol_version: str = Field(default="1.0", max_length=16)

    @field_validator("machine_id")
    @classmethod
    def validate_machine_id(cls, value: str) -> str:
        _validate_safe_hello_text(value, "machine_id")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value) is None:
            raise ValueError("machine_id contains unsupported characters")
        return value

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        return _validate_safe_hello_text(value, "display_name")

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, value: str) -> str:
        if value not in {"darwin", "linux", "windows", "wsl2"}:
            raise ValueError("platform is unsupported")
        return value

    @field_validator("arch")
    @classmethod
    def validate_arch(cls, value: str) -> str:
        if value not in {"x86_64", "arm64"}:
            raise ValueError("arch is unsupported")
        return value

    @field_validator("shim_version")
    @classmethod
    def validate_shim_version(cls, value: str) -> str:
        if not value:
            return value
        _validate_safe_hello_text(value, "shim_version")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", value) is None:
            raise ValueError("shim_version contains unsupported characters")
        return value

    @field_validator("allowed_root")
    @classmethod
    def validate_allowed_root(cls, value: str) -> str:
        _validate_safe_hello_text(value, "allowed_root")
        if not value.strip():
            raise ValueError("allowed_root must be non-empty")
        return value

    @field_validator("protocol_version")
    @classmethod
    def validate_protocol_version(cls, value: str) -> str:
        if re.fullmatch(r"\d+\.\d+", value) is None:
            raise ValueError("protocol_version must use major.minor format")
        return value

    @field_validator("recording_channels")
    @classmethod
    def validate_recording_channels(cls, values: list[str]) -> list[str]:
        allowed = {"floor", "browser", "desktop_ax"}
        if any(value not in allowed for value in values):
            raise ValueError("recording_channels contains an unsupported channel")
        return list(dict.fromkeys(values))

    @field_validator("computer_use_tools")
    @classmethod
    def validate_computer_use_tools(cls, values: list[str]) -> list[str]:
        return _canonical_computer_use_tools(values)

    @field_validator("recording_routes")
    @classmethod
    def validate_recording_routes(cls, values: list[str]) -> list[str]:
        allowed = {"extract_then_cloud", "local_vlm", "screenshots_to_cloud"}
        if any(value not in allowed for value in values):
            raise ValueError("recording_routes contains an unsupported route")
        return list(dict.fromkeys(values))

    @classmethod
    def from_payload(cls, payload: dict) -> "ShimHello":
        sr = payload.get("screen_resolution")
        screen_resolution: tuple[int, int] | None = None
        if isinstance(sr, (list, tuple)) and len(sr) == 2:
            try:
                screen_resolution = (int(sr[0]), int(sr[1]))
            except (TypeError, ValueError):
                screen_resolution = None
        return cls(
            machine_id=_required_hello_string(payload, "machine_id"),
            display_name=_optional_hello_string(payload, "display_name"),
            platform=_required_hello_string(payload, "platform"),
            arch=_required_hello_string(payload, "arch"),
            shim_version=_optional_hello_string(payload, "shim_version"),
            allowed_root=_required_hello_string(payload, "allowed_root"),
            capabilities=_hello_string_list(payload, "capabilities"),
            screen_resolution=screen_resolution,
            local_llm_models=_hello_string_list(payload, "local_llm_models"),
            hardware_devices=[
                device
                for device in payload.get("hardware_devices") or []
                if isinstance(device, dict)
            ],
            computer_use_driver=_hello_driver(payload),
            computer_use_tools=_hello_string_list(payload, "computer_use_tools"),
            recording_channels=_hello_string_list(payload, "recording_channels"),
            recording_routes=_hello_string_list(payload, "recording_routes"),
            protocol_version=_optional_hello_string(
                payload, "protocol_version", default="1.0"
            ),
        )


class ShimConnectionGuard(BaseModel):
    model_config = ConfigDict(frozen=True)

    generation: int
    machine_id: str
    computer_use_driver: str
    computer_use_tools: tuple[str, ...]


class ShimConnectionManager:
    def __init__(self, *, relay: RelayBackend | None = None) -> None:
        self._connections: dict[str, RelayWebSocket] = {}
        self._relay_transports: dict[str, TextTransport] = {}
        self._relay_connection_ids: dict[str, str] = {}
        self._hellos: dict[str, ShimHello] = {}
        self._shims: weakref.WeakValueDictionary[str, LocalPCShim] = (
            weakref.WeakValueDictionary()
        )
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._waiters: dict[str, list[asyncio.Future[RelayWebSocket]]] = {}
        # (user_id, client_id) -> set[session_id]. Lets revoke_user_shims
        # find every active shim belonging to a user+app without scanning
        # the full connection dict.
        self._by_owner: dict[tuple[str | None, str | None], set[str]] = {}
        # Reverse index for fast unregister cleanup.
        self._owner_of: dict[str, tuple[str | None, str | None]] = {}
        self._relay = relay or get_local_pc_relay()

    def register(
        self,
        session_id: str,
        ws: RelayWebSocket,
        hello: ShimHello | None = None,
        *,
        user_id: str | None = None,
        client_id: str | None = None,
    ) -> None:
        previous_websocket = self._connections.get(session_id)
        self._connections[session_id] = ws
        if hello is not None:
            self._hellos[session_id] = hello
        shim = self._shims.get(session_id)
        previous_reader: asyncio.Task[None] | None = None
        if shim is not None:
            if previous_websocket is not None and previous_websocket is not ws:
                shim._fail_in_flight(
                    ConnectionError(
                        f"[LocalPC] Another shim replaced session {session_id[:12]}"
                    )
                )
            previous_reader = shim._replace_connection(ws, hello or ShimHello())
        owner = (user_id, client_id)
        previous_owner = self._owner_of.get(session_id)
        if previous_owner is not None and previous_owner != owner:
            previous_sessions = self._by_owner.get(previous_owner)
            if previous_sessions is not None:
                previous_sessions.discard(session_id)
                if not previous_sessions:
                    self._by_owner.pop(previous_owner, None)
        self._owner_of[session_id] = owner
        self._by_owner.setdefault(owner, set()).add(session_id)
        for fut in self._waiters.pop(session_id, []):
            if not fut.done():
                fut.set_result(ws)
        if previous_websocket is not None and previous_websocket is not ws:
            task = asyncio.create_task(
                self._close_superseded_connection(
                    session_id, previous_websocket, previous_reader
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        logger.info("[LocalPC] Shim registered for session %s", session_id[:12])

    async def _close_superseded_connection(
        self,
        session_id: str,
        websocket: RelayWebSocket,
        reader: asyncio.Task[None] | None,
    ) -> None:
        envelope = {
            "type": "SESSION_REVOKED",
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "payload": {"reason": "another_shim_connected"},
        }
        try:
            await websocket.send_text(json.dumps(envelope))
        except Exception:
            logger.debug(
                "[LocalPC] Failed to notify superseded shim for session %s",
                session_id[:12],
            )
        try:
            await websocket.close(
                code=_DUPLICATE_SESSION_CLOSE_CODE,
                reason="Another shim connected for this session",
            )
        except Exception:
            logger.debug(
                "[LocalPC] Failed to close superseded shim for session %s",
                session_id[:12],
            )
        if reader is not None and not reader.done():
            reader.cancel()

    def unregister(
        self, session_id: str, *, websocket: TextTransport | None = None
    ) -> None:
        current_direct = self._connections.get(session_id)
        current_relay = self._relay_transports.get(session_id)
        if websocket is not None:
            if current_direct is websocket:
                self._connections.pop(session_id, None)
            elif current_relay is websocket:
                self._relay_transports.pop(session_id, None)
                self._relay_connection_ids.pop(session_id, None)
            else:
                return
        else:
            self._connections.pop(session_id, None)
            self._relay_transports.pop(session_id, None)
            self._relay_connection_ids.pop(session_id, None)
        self._hellos.pop(session_id, None)
        owner = self._owner_of.pop(session_id, None)
        if owner is not None:
            sessions = self._by_owner.get(owner)
            if sessions is not None:
                sessions.discard(session_id)
                if not sessions:
                    self._by_owner.pop(owner, None)
        logger.info("[LocalPC] Shim unregistered for session %s", session_id[:12])

    def get_or_create_shim(self, session_id: str) -> "LocalPCShim":
        """Return the one receive-loop owner for a registered session."""
        transport: TextTransport | None = self._connections.get(session_id)
        if transport is None:
            transport = self._relay_transports.get(session_id)
        if transport is None:
            raise ConnectionError(
                f"[LocalPC] Shim for session {session_id[:12]} is not connected"
            )
        shim = self._shims.get(session_id)
        if shim is None:
            shim = LocalPCShim(
                session_id,
                transport,
                self._hellos.get(session_id),
                manager=self,
            )
            self._shims[session_id] = shim
        elif shim._ws is not transport:
            shim._replace_connection(
                transport, self._hellos.get(session_id) or ShimHello()
            )
        return shim

    async def get_or_create_shim_for_session(
        self, session_id: str, *, timeout: float = 1.0
    ) -> "LocalPCShim":
        await self.wait_for(session_id, timeout=timeout)
        return self.get_or_create_shim(session_id)

    def remove_shim(self, session_id: str, shim: "LocalPCShim") -> None:
        """Forget a cached adapter if it is still the current instance."""
        if self._shims.get(session_id) is shim:
            self._shims.pop(session_id, None)

    async def close_existing_session(self, session_id: str) -> None:
        """Close cached resources without waiting for or creating a connection."""
        transport: TextTransport | None = self._connections.get(session_id)
        if transport is None:
            transport = self._relay_transports.get(session_id)
        shim = self._shims.get(session_id)
        try:
            if shim is not None:
                await shim.kill()
            elif transport is not None:
                await transport.close()
        finally:
            if transport is not None:
                self.unregister(session_id, websocket=transport)

    async def push_token_revocation(self, user_id: str, client_id: str) -> int:
        """After one of the user's tokens for ``client_id`` is revoked, close
        only the connections that used it.

        Each computer signs in on its own, so revoking one computer's tokens
        must leave the user's other computers connected. Relay connections
        recheck their own token at once and close with 4428 only if it is no
        longer active. Connections served directly in this process have no
        token watcher, so they are revoked outright, as before.
        """
        delivery_failures: list[Exception] = []
        notified = await self._revoke_direct_connections(
            user_id, client_id, reason="user_revoked", failures=delivery_failures
        )
        try:
            async with asyncio.timeout(_REVOCATION_OPERATION_TIMEOUT_SECONDS):
                relay_notified = await self._relay.request_token_recheck(
                    user_id, client_id
                )
        except Exception as exc:
            delivery_failures.append(exc)
            relay_notified = 0
        if delivery_failures:
            raise RuntimeError(
                "Failed to deliver a Local PC token recheck through "
                f"{len(delivery_failures)} transport operation(s)"
            ) from delivery_failures[0]
        return max(notified, relay_notified)

    async def revoke_user_shims(
        self,
        user_id: str,
        client_id: str | None,
        *,
        reason: str = "user_revoked",
    ) -> int:
        """Push SESSION_REVOKED to every shim owned by (user_id, client_id).

        Returns the count of shims actually notified. The SESSION_REVOKED
        frame shape mirrors the shim's ``protocol.SessionRevokedPayload`` —
        see PROTOCOL.md Session ownership section. The shim daemon audits the
        event, closes the WS, does NOT auto-reconnect.

        ``client_id=None`` means "all of this user's shims, any app."
        Used for platform-level revocations (e.g. account-wide kill
        switch). A revoked OAuth token uses :meth:`push_token_revocation`,
        which leaves the user's other computers connected.
        """
        delivery_failures: list[Exception] = []
        notified = await self._revoke_direct_connections(
            user_id, client_id, reason=reason, failures=delivery_failures
        )
        try:
            async with asyncio.timeout(_REVOCATION_OPERATION_TIMEOUT_SECONDS):
                relay_notified = await self._relay.revoke_owner(
                    user_id, client_id, reason=reason
                )
        except Exception as exc:
            delivery_failures.append(exc)
            relay_notified = 0
        if delivery_failures:
            raise RuntimeError(
                "Failed to deliver Local PC session revocation through "
                f"{len(delivery_failures)} transport operation(s)"
            ) from delivery_failures[0]
        return max(notified, relay_notified)

    async def _revoke_direct_connections(
        self,
        user_id: str,
        client_id: str | None,
        *,
        reason: str,
        failures: list[Exception],
    ) -> int:
        """Send SESSION_REVOKED to, and close, the owner's in-process
        connections. Delivery errors are appended to ``failures``."""
        if client_id is not None:
            target_owners: list[tuple[str | None, str | None]] = [(user_id, client_id)]
        else:
            target_owners = [k for k in self._by_owner if k[0] == user_id]

        notified = 0
        for owner in target_owners:
            # Snapshot — sending SESSION_REVOKED eventually closes the WS,
            # which calls unregister, which mutates _by_owner.
            for session_id in list(self._by_owner.get(owner, ())):
                ws = self._connections.get(session_id)
                if ws is None:
                    continue
                envelope = {
                    "type": "SESSION_REVOKED",
                    "id": str(uuid.uuid4()),
                    "ts": time.time(),
                    "payload": {"reason": reason},
                }
                try:
                    async with asyncio.timeout(_REVOCATION_OPERATION_TIMEOUT_SECONDS):
                        await ws.send_text(json.dumps(envelope))
                    notified += 1
                except Exception as exc:
                    failures.append(exc)
                    logger.debug(
                        "[LocalPC] SESSION_REVOKED send failed for session %s "
                        "(ws likely already closed)",
                        session_id[:12],
                        exc_info=True,
                    )
                # Close ourselves; 4428 disables shim auto-reconnect per spec.
                try:
                    async with asyncio.timeout(_REVOCATION_OPERATION_TIMEOUT_SECONDS):
                        await ws.close(code=4428, reason="Token revoked")
                except Exception as exc:
                    failures.append(exc)
                    logger.debug(
                        "[LocalPC] SESSION_REVOKED close failed for session %s",
                        session_id[:12],
                        exc_info=True,
                    )
                else:
                    self.unregister(session_id, websocket=ws)
        if notified:
            logger.info(
                "[LocalPC] Revoked %d shim session(s) for user %s app %s",
                notified,
                user_id[:12] if user_id else "?",
                client_id or "*",
            )
        return notified

    async def wait_for(self, session_id: str, timeout: float = 30.0) -> TextTransport:
        direct = self._connections.get(session_id)
        if direct is not None:
            return direct
        loop = asyncio.get_running_loop()
        timeout = max(0.0, timeout)
        deadline = loop.time() + timeout
        try:
            async with asyncio.timeout(timeout):
                cached = await self._reuse_cached_transport(session_id)
                if cached is not None:
                    return cached
                return await self._wait_for_new_transport(session_id, deadline)
        except TimeoutError as exc:
            direct = self._connections.get(session_id)
            if direct is not None:
                return direct
            raise TimeoutError(
                f"[LocalPC] Shim for session {session_id[:12]} did not connect "
                f"within {timeout}s"
            ) from exc

    async def _reuse_cached_transport(self, session_id: str) -> TextTransport | None:
        relay_transport = self._relay_transports.get(session_id)
        if relay_transport is None:
            return None
        presence = await self._relay.get_presence(session_id)
        direct = self._connections.get(session_id)
        if direct is not None:
            self._drop_relay_transport(session_id, fail_in_flight=False)
            await relay_transport.close()
            return direct
        if (
            presence is not None
            and self._relay_connection_ids.get(session_id) == presence.connection_id
        ):
            return relay_transport
        self._drop_relay_transport(session_id, fail_in_flight=True)
        await relay_transport.close()
        return None

    def _drop_relay_transport(self, session_id: str, *, fail_in_flight: bool) -> None:
        self._relay_transports.pop(session_id, None)
        self._relay_connection_ids.pop(session_id, None)
        if fail_in_flight:
            self._hellos.pop(session_id, None)
            shim = self._shims.get(session_id)
            if shim is not None:
                shim._fail_in_flight(
                    ConnectionError(
                        f"[LocalPC] Shim connection closed for session {session_id[:12]}"
                    )
                )

    async def _wait_for_new_transport(
        self, session_id: str, deadline: float
    ) -> TextTransport:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[RelayWebSocket] = loop.create_future()
        self._waiters.setdefault(session_id, []).append(fut)
        direct = self._connections.get(session_id)
        if direct is not None:
            fut.set_result(direct)
        relay_task = asyncio.create_task(
            self._relay.wait_for_presence(
                session_id, timeout=max(0.0, deadline - loop.time())
            )
        )
        try:
            done, _pending = await asyncio.wait(
                {fut, relay_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if fut in done:
                return fut.result()
            try:
                presence = relay_task.result()
            except TimeoutError:
                return await fut
            transport = await self._relay.open_transport(presence)
            if fut.done():
                await transport.close()
                return fut.result()
            self._relay_transports[session_id] = transport
            self._relay_connection_ids[session_id] = presence.connection_id
            self._hellos[session_id] = ShimHello.from_payload(presence.hello)
            shim = self._shims.get(session_id)
            if shim is not None:
                shim._replace_connection(transport, self._hellos[session_id])
            return transport
        finally:
            if not relay_task.done():
                relay_task.cancel()
            await asyncio.gather(relay_task, return_exceptions=True)
            if not fut.done():
                fut.cancel()
            waiters = self._waiters.get(session_id)
            if waiters is not None:
                if fut in waiters:
                    waiters.remove(fut)
                if not waiters:
                    self._waiters.pop(session_id, None)

    def get(self, session_id: str) -> RelayWebSocket | None:
        return self._connections.get(session_id)

    def get_hello(self, session_id: str) -> ShimHello | None:
        return self._hellos.get(session_id)

    def remember_hello(self, session_id: str, hello: ShimHello) -> None:
        self._hellos[session_id] = hello

    async def get_hello_async(self, session_id: str) -> ShimHello | None:
        presence = await self._relay.get_presence(session_id)
        if presence is None:
            if session_id in self._connections:
                return self._hellos.get(session_id)
            self._relay_transports.pop(session_id, None)
            self._relay_connection_ids.pop(session_id, None)
            self._hellos.pop(session_id, None)
            return None
        hello = ShimHello.from_payload(presence.hello)
        self._hellos[session_id] = hello
        return hello

    async def serve_websocket(
        self,
        session_id: str,
        websocket: RelayWebSocket,
        hello: ShimHello,
        *,
        user_id: str,
        client_id: str,
        on_token_recheck: Callable[[], None] | None = None,
    ) -> None:
        await self._relay.serve_websocket(
            session_id,
            websocket,
            hello=hello.model_dump(mode="json"),
            user_id=user_id,
            client_id=client_id,
            on_token_recheck=on_token_recheck,
        )


class _FilesProxy:
    def __init__(self, shim: "LocalPCShim") -> None:
        self._shim = shim

    async def read(self, path: str, *, format: str = "text") -> str | bytes:
        wire_encoding = "base64" if format == "bytes" else "utf-8"
        content = await self._file_read({"path": path, "encoding": wire_encoding})
        if format == "bytes":
            return base64.b64decode(content)
        return content

    async def read_capped(self, path: str, max_bytes: int) -> bytes:
        """At most ``max_bytes + 1`` bytes of *path*, as ``read_sandbox_file``
        reads an E2B file: a longer result means the file is over the limit.

        The companion refuses any read over its own limit, which is at most
        ``MAX_FILE_SIZE_BYTES``. A cap under that is asked for as a range, so
        the companion stops reading there. Above it, or when this companion's
        own limit is under the range, a whole read is refused past that lower
        limit, so it still ends under the cap.
        """
        whole = {"path": path, "encoding": "base64"}
        length = max_bytes + 1
        if length <= MAX_FILE_SIZE_BYTES:
            try:
                return base64.b64decode(
                    await self._file_read({**whole, "offset": 0, "length": length})
                )
            except ShimFileTooLargeError:
                pass
        return base64.b64decode(await self._file_read(whole))

    async def _file_read(self, request: dict[str, Any]) -> str:
        resp = await self._shim._rpc("FILE_READ", request)
        if resp.get("type") == "ERROR":
            error = resp.get("payload", {})
            message = _friendly(error, self._shim, "FILE_READ failed")
            if isinstance(error, dict) and error.get("code") == "FILE_TOO_LARGE":
                raise ShimFileTooLargeError(message)
            raise OSError(message)
        return resp["payload"]["content"]

    async def write(self, path: str, content: str | bytes) -> None:
        if isinstance(content, (bytes, bytearray, memoryview)):
            wire_content = base64.b64encode(bytes(content)).decode("ascii")
            wire_encoding = "base64"
        else:
            wire_content = content
            wire_encoding = "utf-8"
        try:
            resp = await self._shim._rpc(
                "FILE_WRITE",
                {
                    "path": path,
                    "content": wire_content,
                    "encoding": wire_encoding,
                    "create_parents": True,
                },
            )
        except WriteUnconfirmedError as exc:
            # Synthesize an ERROR envelope so the LLM sees the translator's
            # WRITE_UNCONFIRMED hint, not the raw Python exception text.
            raise OSError(
                _friendly(
                    {
                        "code": "WRITE_UNCONFIRMED",
                        "message": str(exc),
                        "details": {"path": path, "op": "FILE_WRITE"},
                    },
                    self._shim,
                    "FILE_WRITE unconfirmed",
                )
            ) from exc
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
        try:
            resp = await self._shim._rpc(
                "FILE_DELETE",
                {"path": path, "recursive": recursive, "missing_ok": missing_ok},
            )
        except OpUnconfirmedError as exc:
            raise OSError(
                _translate_unconfirmed(
                    exc,
                    self._shim,
                    fallback="FILE_DELETE unconfirmed",
                    extra_details={"path": path},
                )
            ) from exc
        if resp.get("type") == "ERROR":
            raise OSError(
                _friendly(resp.get("payload", {}), self._shim, "FILE_DELETE failed")
            )

    async def move(self, src: str, dst: str, *, overwrite: bool = False) -> None:
        """Cross-OS portable replacement for shell `mv` / `move`."""
        try:
            resp = await self._shim._rpc(
                "FILE_MOVE",
                {"src": src, "dst": dst, "overwrite": overwrite},
            )
        except OpUnconfirmedError as exc:
            raise OSError(
                _translate_unconfirmed(
                    exc,
                    self._shim,
                    fallback="FILE_MOVE unconfirmed",
                    extra_details={"src": src, "dst": dst},
                )
            ) from exc
        if resp.get("type") == "ERROR":
            raise OSError(
                _friendly(resp.get("payload", {}), self._shim, "FILE_MOVE failed")
            )


class ShimComputerUseError(RuntimeError):
    """Raised when a computer-use wire op returns a structured ERROR.

    ``code`` mirrors the wire ``payload.code`` (PERMISSION_PENDING,
    FEATURE_NOT_SUPPORTED, CLIPBOARD_CONCEALED, DEPENDENCY_MISSING,
    COMMAND_TIMEOUT, ...) so the MCP-tool handlers can branch without
    parsing ``message``, the LLM-friendly text from :mod:`local_pc_errors`.
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


# The transport's own errors, which the computer-use handlers and the held-call
# replay don't catch: the shim too busy to take the call (nothing ran), a read
# that timed out or lost its connection on the retry too, and a reply that
# broke the wire contract. Each is raised as a ShimComputerUseError instead.
_COMPUTER_USE_TRANSPORT_ERRORS = (
    TimeoutError,
    OpUnconfirmedError,
    ShimOverloadedError,
    ShimProtocolError,
)


def _computer_use_transport_error(
    exc: BaseException, shim: "LocalPCShim | None", details: dict[str, Any]
) -> ShimComputerUseError:
    if isinstance(exc, (ShimOverloadedError, OpUnconfirmedError)):
        code = exc.code
    elif isinstance(exc, TimeoutError):
        code = "COMMAND_TIMEOUT"
    else:
        code = "INTERNAL_ERROR"
    return ShimComputerUseError(
        code=code,
        message=_friendly(
            {"code": code, "message": str(exc), "details": details},
            shim,
            f"{details.get('tool') or details.get('op')} failed",
        ),
        details=details,
    )


class _ComputerProxy:
    """The shim's cua-driver tools: COMPUTER_TOOLS_REQUEST lists them with
    the driver's guidance, COMPUTER_TOOL_CALL runs one. See
    ``autogpt-local-executor/docs/CUA_DRIVER.md``."""

    def __init__(self, shim: "LocalPCShim") -> None:
        self._shim = shim
        # The list for one connection; a reconnect may bring another driver.
        self._catalog: tuple[int, dict] | None = None

    async def tools(self, *, _guard: ShimConnectionGuard | None = None) -> dict:
        """``{driver, instructions, tools: [{name, description, input_schema,
        annotations, approval_arguments}]}`` as the shim sent it, cached per
        connection."""
        generation = self._shim._connection_generation
        if self._catalog is not None and self._catalog[0] == generation:
            return self._catalog[1]
        try:
            resp = await self._shim._rpc(
                "COMPUTER_TOOLS_REQUEST", {}, connection_guard=_guard
            )
        except _COMPUTER_USE_TRANSPORT_ERRORS as exc:
            raise _computer_use_transport_error(
                exc, self._shim, {"op": "COMPUTER_TOOLS_REQUEST"}
            ) from exc
        _raise_computer_use(resp, self._shim, "COMPUTER_TOOLS_REQUEST failed")
        payload = resp.get("payload") or {}
        self._catalog = (generation, payload)
        return payload

    async def call(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
        approved_arguments: Sequence[str] = (),
        _guard: ShimConnectionGuard | None = None,
    ) -> dict:
        """``{tool, is_error, content, structured_content}``. Never retried:
        most tools act, and the model decides what to do after a failure.
        ``approved_arguments``: the tool's approval arguments (disk paths) the
        user approved for this call; the shim refuses one that isn't named."""
        payload: dict[str, Any] = {"tool": tool, "arguments": arguments}
        if timeout_seconds is not None:
            payload["timeout_seconds"] = timeout_seconds
        if approved_arguments:
            payload["approved_arguments"] = sorted(approved_arguments)
        deadline = timeout_seconds or _DEFAULT_COMPUTER_TOOL_TIMEOUT_SECONDS
        try:
            resp = await self._shim._rpc(
                "COMPUTER_TOOL_CALL",
                payload,
                timeout=deadline + _COMPUTER_TOOL_CALL_MARGIN_SECONDS,
                connection_guard=_guard,
            )
        except OpUnconfirmedError as exc:
            raise ShimComputerUseError(
                code=exc.code,
                message=_translate_unconfirmed(
                    exc,
                    self._shim,
                    fallback=f"{tool} unconfirmed",
                    extra_details={"tool": tool},
                ),
                details={"tool": tool, "op": exc.op},
            ) from exc
        except _COMPUTER_USE_TRANSPORT_ERRORS as exc:
            raise _computer_use_transport_error(
                exc, self._shim, {"tool": tool, "op": "COMPUTER_TOOL_CALL"}
            ) from exc
        _raise_computer_use(resp, self._shim, f"{tool} failed")
        return resp.get("payload") or {}


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
        # Wait out the companion's own deadline: a command that ran out of time
        # still has to be stopped and reported, and an RPC that gave up first
        # would report a plain timeout as an unconfirmed side effect.
        deadline = min(timeout or COMMAND_TIMEOUT_SECONDS, COMMAND_TIMEOUT_SECONDS)
        try:
            resp = await self._shim._rpc(
                "EXECUTE_COMMAND",
                payload,
                timeout=deadline + _COMMAND_RESULT_GRACE_SECONDS,
            )
        except OpUnconfirmedError as exc:
            raise RuntimeError(
                _translate_unconfirmed(
                    exc,
                    self._shim,
                    fallback="EXECUTE_COMMAND unconfirmed",
                    extra_details={"command": (command or " ".join(argv or []))[:200]},
                )
            ) from exc
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
        self.output_truncated = bool(payload.get("output_truncated", False))


# ── Workflow recording ───────────────────────────────────────────────────────
#
# `_RecordingProxy` wraps the §6 wire ops: START_RECORDING / STOP_RECORDING /
# RECORDING_FETCH. Demonstration mode buffers on the shim and the platform
# pulls via `fetch()` after STOP + user approval. Co-pilot mode additionally
# streams RECORDING_STEP frames — unsolicited, non-acked, modeled like STATUS
# (§6) — which the recv loop fans out per recording_id into a queue the live
# co-pilot loop drains via `stream_steps()`.
#
# See autogpt-local-executor/docs/WORKFLOW_RECORDING.md.


class ShimRecordingError(RuntimeError):
    """Raised when a recording wire op returns a structured ERROR.

    ``code`` mirrors the wire ``payload.code`` (RECORDING_NOT_FOUND,
    RECORDING_CHANNEL_UNAVAILABLE, RECORDING_ALREADY_ACTIVE,
    CONSENT_REQUIRED, INTERPRETATION_UNAVAILABLE) so the MCP-tool layer can
    branch on the structured surface without parsing the human ``message``.
    ``message`` is already LLM-friendly (produced by :mod:`local_pc_errors`).
    """

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _raise_recording(resp: dict, shim: "LocalPCShim | None", fallback: str) -> None:
    """Translate a wire ERROR response into a typed ShimRecordingError."""
    if resp.get("type") != "ERROR":
        return
    payload = resp.get("payload") or {}
    raise ShimRecordingError(
        code=str(payload.get("code") or "INTERNAL_ERROR"),
        message=_friendly(payload, shim, fallback),
        details=(
            payload.get("details") if isinstance(payload.get("details"), dict) else {}
        ),
    )


class _RecordingProxy:
    """Wire-op wrapper for the shim's workflow-recording surface.

    Mirrors ``autogpt-local-executor/docs/WORKFLOW_RECORDING.md``
    §6. ``start`` / ``stop`` / ``fetch`` are request/response (count against
    in-flight). RECORDING_STEP frames (co-pilot mode) arrive out-of-band and
    are exposed via :meth:`stream_steps`.
    """

    def __init__(self, shim: "LocalPCShim") -> None:
        self._shim = shim
        self._effective_routes: dict[str, str] = {}

    def _validate_advertised(
        self, *, interpretation_route: str, channels: list[str]
    ) -> None:
        unavailable_channels = sorted(
            set(channels) - set(self._shim.recording_channels)
        )
        if unavailable_channels:
            raise ShimRecordingError(
                "RECORDING_CHANNEL_UNAVAILABLE",
                "The requested recording channel was not advertised by the local executor.",
                {"unavailable_channels": unavailable_channels},
            )
        if interpretation_route not in self._shim.recording_routes:
            raise ShimRecordingError(
                "INTERPRETATION_UNAVAILABLE",
                "The requested interpretation route was not advertised by the local executor.",
                {"interpretation_route": interpretation_route},
            )

    def effective_route_for(self, recording_id: str, fallback: str) -> str:
        return self._effective_routes.get(recording_id, fallback)

    async def request_consent(
        self,
        *,
        mode: str = "copilot",
        interpretation_route: str = "extract_then_cloud",
        channels: list[str] | None = None,
    ) -> RecordingConsentResult:
        """Ask the shim to show its native recording-consent prompt."""
        capture_channels = list(channels or ["floor"])
        self._validate_advertised(
            interpretation_route=interpretation_route, channels=capture_channels
        )
        resp = await self._shim._rpc(
            "REQUEST_RECORDING_CONSENT",
            {
                "mode": mode,
                "interpretation_route": interpretation_route,
                "channels": capture_channels,
            },
        )
        _raise_recording(resp, self._shim, "Recording consent request failed")
        if resp.get("type") != "RECORDING_CONSENT_RESULT":
            raise ShimRecordingError(
                "PROTOCOL_ERROR",
                "The local executor returned an invalid recording consent response.",
            )
        result = RecordingConsentResult.from_payload(resp.get("payload") or {})
        if result.mode != mode:
            raise ShimRecordingError(
                "CONSENT_SCOPE_MISMATCH",
                "The local executor consent was issued for different recording settings.",
                {
                    "requested_mode": mode,
                    "approved_mode": result.mode,
                },
            )
        if result.interpretation_route not in self._shim.recording_routes:
            raise ShimRecordingError(
                "CONSENT_SCOPE_MISMATCH",
                "The local executor returned a consent route it did not advertise.",
                {
                    "requested_interpretation_route": interpretation_route,
                    "approved_interpretation_route": result.interpretation_route,
                },
            )
        if result.approved and (
            result.expires_at is None or result.expires_at <= time.time()
        ):
            raise ShimRecordingError(
                "CONSENT_REQUIRED",
                "The local executor returned expired recording consent.",
            )
        return result

    async def start_with_consent(
        self,
        *,
        mode: str = "copilot",
        interpretation_route: str = "extract_then_cloud",
        channels: list[str] | None = None,
    ) -> str:
        """Obtain native consent and start without exposing its token."""
        capture_channels = list(channels or ["floor"])
        consent = await self.request_consent(
            mode=mode,
            interpretation_route=interpretation_route,
            channels=capture_channels,
        )
        if not consent.approved:
            raise ShimRecordingError(
                "CONSENT_DENIED", "The user declined workflow recording."
            )
        if not consent.consent_token:
            raise ShimRecordingError(
                "CONSENT_REQUIRED",
                "The local executor approved recording without issuing a consent token.",
            )
        recording_id = await self.start(
            mode=mode,
            interpretation_route=consent.interpretation_route,
            channels=capture_channels,
            consent_token=consent.consent_token,
        )
        self._effective_routes[recording_id] = consent.interpretation_route
        return recording_id

    async def start(
        self,
        *,
        mode: str,
        interpretation_route: str,
        channels: list[str],
        consent_token: str,
    ) -> str:
        """START_RECORDING → return the new recording_id.

        ``consent_token`` is REQUIRED — the platform cannot self-assert it
        (§9); START without a valid shim-issued token gets CONSENT_REQUIRED.
        """
        self._validate_advertised(
            interpretation_route=interpretation_route, channels=list(channels)
        )
        resp = await self._shim._rpc(
            "START_RECORDING",
            {
                "mode": mode,
                "interpretation_route": interpretation_route,
                "channels": list(channels),
                "consent_token": consent_token,
            },
        )
        _raise_recording(resp, self._shim, "START_RECORDING failed")
        recording_id = str((resp.get("payload") or {}).get("recording_id") or "")
        if not recording_id:
            raise ShimRecordingError(
                "PROTOCOL_ERROR",
                "The local executor returned an empty recording ID.",
            )
        self._shim._ensure_recording_buffer(recording_id, started=True)
        return recording_id

    async def stop(self, recording_id: str) -> "RecordingSummary":
        """STOP_RECORDING → return the RECORDING_SUMMARY."""
        resp = await self._shim._rpc("STOP_RECORDING", {"recording_id": recording_id})
        _raise_recording(resp, self._shim, "STOP_RECORDING failed")
        summary = RecordingSummary.from_payload(resp.get("payload") or {})
        if summary.recording_id != recording_id:
            raise ShimRecordingError(
                "PROTOCOL_ERROR",
                "The local executor returned a summary for a different recording.",
            )
        return summary

    async def apply_review(
        self,
        recording_id: str,
        *,
        removed_step_seqs: list[int],
        redacted_step_seqs: list[int],
    ) -> RecordingReviewApplied:
        """Persist authoritative user review edits in shim-owned storage."""
        resp = await self._shim._rpc(
            "APPLY_RECORDING_REVIEW",
            {
                "recording_id": recording_id,
                "removed_step_seqs": list(removed_step_seqs),
                "redacted_step_seqs": list(redacted_step_seqs),
            },
        )
        _raise_recording(resp, self._shim, "Applying recording review failed")
        if resp.get("type") != "RECORDING_REVIEW_APPLIED":
            raise ShimRecordingError(
                "PROTOCOL_ERROR",
                "The local executor returned an invalid recording review response.",
            )
        applied = RecordingReviewApplied.from_payload(resp.get("payload") or {})
        if applied.recording_id != recording_id:
            raise ShimRecordingError(
                "PROTOCOL_ERROR",
                "The local executor applied review to a different recording.",
            )
        return applied

    async def fetch(self, recording_id: str) -> "WorkflowRecording":
        """RECORDING_FETCH → return the full post-redaction WorkflowRecording.

        For demonstration mode this is the only path the data leaves the
        machine — the shim buffers until STOP + user approval, then the
        platform pulls (§6).
        """
        resp = await self._shim._rpc("RECORDING_FETCH", {"recording_id": recording_id})
        _raise_recording(resp, self._shim, "RECORDING_FETCH failed")
        payload = resp.get("payload") or {}
        recording = payload.get("recording") if isinstance(payload, dict) else None
        if not isinstance(recording, dict):
            raise ShimRecordingError(
                "PROTOCOL_ERROR",
                "The local executor returned an invalid recording payload.",
            )
        parsed = WorkflowRecording.from_payload(recording)
        if parsed.recording_id != recording_id:
            raise ShimRecordingError(
                "PROTOCOL_ERROR",
                "The local executor returned data for a different recording.",
            )
        return parsed

    def stream_steps(self, recording_id: str):
        """Async iterator over live RECORDING_STEP frames (co-pilot mode).

        Yields :class:`TrajectoryStep` as the shim emits them. The iterator
        runs until the caller breaks out of it (e.g. after STOP); the
        underlying buffer is dropped via :meth:`LocalPCShim.close_recording`.
        Demonstration mode never streams — this iterator simply blocks
        until the buffer is closed.
        """
        return self._shim._iter_recording_steps(recording_id)


class _RpcAttemptFailed(Exception):
    """Internal: one `_send_and_wait` attempt failed (timeout or WS error).

    Carries the wire `id` so the outer `_rpc` can convert a non-retryable
    attempt into an :class:`OpUnconfirmedError` / :class:`WriteUnconfirmedError`
    with the original correlation id intact.
    """

    def __init__(self, wire_id: str, msg_type: str, *, timed_out: bool) -> None:
        super().__init__(f"{msg_type} attempt failed (wire id={wire_id})")
        self.wire_id = wire_id
        self.msg_type = msg_type
        self.timed_out = timed_out


class LocalPCShim(LocalPCExecutorMarker):
    """
    Drop-in replacement for E2B AsyncSandbox that routes execution to the
    user's local machine via the autogpt-local-executor shim.

    Executor contract: .commands.run(), .files.read(), .files.write(),
                       .pause(), .kill(), .sandbox_id

    Extended attributes (LocalPC-only; safe to read via isinstance check):
        .allowed_root, .machine_id, .platform, .arch, .capabilities,
        .shim_version, .screen_resolution, .local_llm_models, .hardware_devices,
        .computer_use_driver, .computer_use_tools

    Computer-use surface:
        .computer.tools() and .computer.call(tool, arguments): the shim's
        cua-driver tools — see ``_ComputerProxy`` and
        ``autogpt-local-executor/docs/CUA_DRIVER.md``.

    Dormant local-model surface (not connected to the chat service):
        .local_llm.complete(...) (async iterator of deltas) and
        .local_llm.complete_blocking(...) (string). The companion keeps
        this capability disabled pending full integration — see
        ``local_llm_router.py`` and
        ``autogpt-local-executor/docs/LOCAL_LLM.md``.

    Workflow-recording surface:
        .recording.start(...), .recording.stop(...), .recording.fetch(...),
        and .recording.stream_steps(...) — see ``_RecordingProxy`` and
        ``autogpt-local-executor/docs/WORKFLOW_RECORDING.md``.
        Only usable when the shim advertised the ``recording`` capability.
    """

    def __init__(
        self,
        session_id: str,
        ws: TextTransport,
        hello: ShimHello | None = None,
        *,
        manager: "ShimConnectionManager | None" = None,
    ) -> None:
        self.sandbox_id = session_id
        self._ws = ws
        self._manager = manager
        self._connection_generation = 1
        hello = hello or ShimHello()
        self._apply_hello(hello)
        self._pending: dict[str, tuple[frozenset[str], asyncio.Future[dict]]] = {}
        # Streaming requests (LOCAL_LLM_COMPLETION) accumulate multiple
        # frames per request id. The queue collects every CHUNK + the
        # terminal RESPONSE (or ERROR); the consumer in _LocalLLMProxy
        # drains it and unregisters when the stream closes.
        self._streaming: dict[str, asyncio.Queue[dict]] = {}
        # Workflow recording — co-pilot mode streams RECORDING_STEP frames
        # (unsolicited, non-acked, like STATUS — §6) keyed by recording_id.
        # Each recording gets a queue the live co-pilot loop drains via
        # `recording.stream_steps()`. A sentinel `None` put on the queue
        # signals the iterator to stop (set by `close_recording`).
        self._recording_steps: dict[str, asyncio.Queue[TrajectoryStep | None]] = {}
        self._started_recording_ids: set[str] = set()
        # Backpressure — see PROTOCOL.md §Concurrency + STATUS frame
        # support. `pending_capacity` is the shim's self-reported headroom:
        # 0 = at the concurrency cap, refuse-new-work; >0 = slots free; None
        # = unknown (pre-STATUS or shim doesn't advertise capacity yet).
        # Updated by both per-response envelopes and periodic STATUS frames.
        self._pending_capacity: int | None = None
        self._capacity_available = asyncio.Event()
        self._capacity_available.set()  # default: assume open until told otherwise
        self.files = _FilesProxy(self)
        self.commands = _CommandsProxy(self)
        self.computer = _ComputerProxy(self)
        self.local_llm = _LocalLLMProxy(self)
        self.recording = _RecordingProxy(self)
        self._recv_task = asyncio.create_task(self._recv_loop(ws))

    @property
    def pending_capacity(self) -> int | None:
        """Last-known shim-side request-slot headroom.

        ``None`` until the shim advertises capacity (either embedded in a
        response envelope's ``pending_capacity`` field or via a periodic
        STATUS frame). ``0`` means the shim is at its ``max_concurrent`` cap
        — new `_rpc` calls will block on the capacity event for up to 30s
        before raising :class:`ShimOverloadedError`.
        """
        return self._pending_capacity

    @classmethod
    async def for_session(
        cls,
        session_id: str,
        *,
        manager: ShimConnectionManager,
        connect_timeout: float = 30.0,
    ) -> "LocalPCShim":
        await manager.wait_for(session_id, timeout=connect_timeout)
        return manager.get_or_create_shim(session_id)

    def _apply_hello(self, hello: ShimHello) -> None:
        self.machine_id = hello.machine_id
        self.platform = hello.platform
        self.arch = hello.arch
        self.shim_version = hello.shim_version
        self.allowed_root = hello.allowed_root
        self.capabilities = list(hello.capabilities)
        self.capability_set = frozenset(hello.capabilities)
        self.screen_resolution = hello.screen_resolution
        self.local_llm_models = list(hello.local_llm_models)
        self.hardware_devices = list(hello.hardware_devices)
        self.computer_use_driver = hello.computer_use_driver
        self.computer_use_tools = list(hello.computer_use_tools)
        self.recording_channels = list(hello.recording_channels)
        self.recording_routes = list(hello.recording_routes)
        self.protocol_version = hello.protocol_version

    def capture_connection_guard(self) -> ShimConnectionGuard:
        return ShimConnectionGuard(
            generation=getattr(self, "_connection_generation", 0),
            machine_id=self.machine_id,
            computer_use_driver=(
                self.computer_use_driver.label if self.computer_use_driver else ""
            ),
            computer_use_tools=tuple(self.computer_use_tools),
        )

    def connection_guard_matches(self, guard: ShimConnectionGuard) -> bool:
        return guard == self.capture_connection_guard()

    def _replace_connection(
        self, websocket: TextTransport, hello: ShimHello
    ) -> asyncio.Task[None] | None:
        """Attach a reconnect to this adapter without creating a second reader."""
        self._connection_generation += 1
        self._apply_hello(hello)
        if websocket is self._ws and not self._recv_task.done():
            return None
        previous_reader = self._recv_task
        self._ws = websocket
        # The old connection's headroom says nothing about the new one; its
        # first response or STATUS frame reports the new figure.
        self._pending_capacity = None
        self._capacity_available.set()
        self._recv_task = asyncio.create_task(self._recv_loop(websocket))
        return previous_reader

    async def wait_closed(self) -> None:
        """Wait for the adapter-owned receive loop to finish."""
        reader = self._recv_task
        try:
            await reader
        except asyncio.CancelledError:
            if reader is not self._recv_task:
                return
            raise

    def _update_pending_capacity(self, value: Any, *, source: str) -> None:
        """Defensive: accept whatever the shim sent and only honor sane ints.

        STATUS / response shapes drift more than wire docs admit. A
        non-int, missing, or negative value leaves the prior reading
        unchanged rather than crashing the recv loop. Logged at DEBUG so
        operators can spot a shim that's regressed its self-report.
        """
        if value is None:
            return
        try:
            capacity = int(value)
        except (TypeError, ValueError):
            logger.debug(
                "[LocalPC] Ignoring non-int pending_capacity=%r from %s",
                value,
                source,
            )
            return
        if capacity < 0:
            logger.debug(
                "[LocalPC] Ignoring negative pending_capacity=%d from %s",
                capacity,
                source,
            )
            return
        self._pending_capacity = capacity
        if capacity > 0:
            self._capacity_available.set()
        else:
            self._capacity_available.clear()

    async def _await_capacity(
        self, msg_type: str, *, timeout: float = _CAPACITY_WAIT_TIMEOUT_SECONDS
    ) -> None:
        """Block until the shim reports headroom, or raise SHIM_OVERLOADED.

        If ``pending_capacity`` is 0, wait up to ``timeout`` seconds for a
        subsequent response or STATUS frame to clear the gate. Past that,
        raise :class:`ShimOverloadedError` proactively — sending the op
        anyway would just get bounced with the same code by the shim.
        """
        # Defensive against test fixtures that construct shims via
        # ``__new__`` and skip ``__init__``.
        capacity = getattr(self, "_pending_capacity", None)
        event = getattr(self, "_capacity_available", None)
        if capacity != 0:
            return
        if event is None or event.is_set():
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise ShimOverloadedError(
                f"[LocalPC] Shim has reported pending_capacity=0 for {timeout}s; "
                f"refusing to send {msg_type}"
            ) from exc

    async def _send_and_wait(
        self,
        msg_type: str,
        payload: dict,
        *,
        timeout: float,
        connection_guard: ShimConnectionGuard | None = None,
    ) -> dict:
        """One attempt at sending a wire op and awaiting its response.

        Raises ``TimeoutError`` if the response doesn't arrive within
        ``timeout`` or the underlying WS send fails before the response.
        """
        if connection_guard is not None and not self.connection_guard_matches(
            connection_guard
        ):
            raise ShimComputerUseError(
                "COMPUTER_USE_CONNECTION_CHANGED",
                "The Local PC executor reconnected while computer access was being "
                "checked. Retry after the current machine and capabilities are reviewed.",
            )
        websocket = self._ws
        msg_id = str(uuid.uuid4())
        msg = {"type": msg_type, "id": msg_id, "ts": time.time(), "payload": payload}
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        expected_types = EXPECTED_RESPONSE_TYPES.get(msg_type)
        if expected_types is None:
            raise ShimProtocolError(f"No response contract is defined for {msg_type}")
        self._pending[msg_id] = (expected_types, fut)
        started = time.monotonic()
        # Stays None on cancellation, which says nothing about the machine.
        outcome: str | None = None
        try:
            await websocket.send_text(json.dumps(msg))
            response = await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
            outcome = _record_response_outcome(msg_type, response)
            return response
        except ShimProtocolError:
            outcome = "error"
            raise
        except asyncio.TimeoutError as exc:
            outcome = "timeout"
            # Tag the timeout with the wire id so the caller can convert it
            # into a typed OpUnconfirmedError (non-idempotent ops) or trigger
            # a single auto-retry (idempotent ops).
            raise _RpcAttemptFailed(msg_id, msg_type, timed_out=True) from exc
        except Exception as exc:
            outcome = "error"
            # WS disconnect / send error / recv loop closure — the response
            # cannot arrive on this connection. Surface as unconfirmed so the
            # caller decides retry vs. raise based on idempotency.
            raise _RpcAttemptFailed(msg_id, msg_type, timed_out=False) from exc
        finally:
            if outcome is not None:
                record_rpc(
                    msg_type,
                    outcome=outcome,
                    duration_seconds=time.monotonic() - started,
                )
            pending = self._pending.pop(msg_id, None)
            if pending is not None and not fut.done():
                fut.cancel()

    async def _rpc(
        self,
        msg_type: str,
        payload: dict,
        *,
        timeout: float = 30.0,
        connection_guard: ShimConnectionGuard | None = None,
    ) -> dict:
        """Send a wire op and await its response.

        Disconnect/timeout semantics follow the per-op idempotency table in
        ``autogpt-local-executor/docs/PROTOCOL.md``:

        - Idempotent ops (FILE_READ, FILE_STAT, ...): on
          timeout/disconnect, schedule one automatic retry once the WS is
          back up. The retry is invisible to the caller — they get the
          eventual result or one final error.
        - Non-idempotent ops (FILE_WRITE, EXECUTE_COMMAND, ...): raise
          :class:`WriteUnconfirmedError` / :class:`OpUnconfirmedError`
          immediately so the LLM can probe state instead of double-applying
          a side effect.

        Backpressure: if the shim's most recent ``pending_capacity`` signal
        is 0, this blocks for up to 30s waiting for headroom before sending,
        then raises :class:`ShimOverloadedError`.
        """
        await self._await_capacity(msg_type)
        try:
            if connection_guard is None:
                return await self._send_and_wait(msg_type, payload, timeout=timeout)
            return await self._send_and_wait(
                msg_type, payload, timeout=timeout, connection_guard=connection_guard
            )
        except _RpcAttemptFailed as first:
            if msg_type in _IDEMPOTENT_OPS:
                # One retry, after the WS is back. wait_for_reconnect is a
                # best-effort no-op if the manager isn't wired in (tests).
                try:
                    await self._await_reconnect_for_retry()
                except Exception:
                    pass
                if connection_guard is not None and not self.connection_guard_matches(
                    connection_guard
                ):
                    raise ShimComputerUseError(
                        "COMPUTER_USE_CONNECTION_CHANGED",
                        "The Local PC executor reconnected before this computer-use "
                        "operation could be retried. Review the current machine and "
                        "capabilities before trying again.",
                    )
                try:
                    if connection_guard is None:
                        result = await self._send_and_wait(
                            msg_type, payload, timeout=timeout
                        )
                    else:
                        result = await self._send_and_wait(
                            msg_type,
                            payload,
                            timeout=timeout,
                            connection_guard=connection_guard,
                        )
                except _RpcAttemptFailed as second:
                    record_rpc_retry(msg_type, recovered=False)
                    if second.timed_out:
                        raise TimeoutError(
                            f"[LocalPC] RPC {msg_type} timed out after {timeout}s "
                            "(retry also failed)"
                        ) from second
                    raise OpUnconfirmedError(
                        op=msg_type,
                        wire_id=second.wire_id,
                        message=(
                            f"[LocalPC] RPC {msg_type} disconnected mid-call "
                            "(retry also failed)"
                        ),
                    ) from second
                record_rpc_retry(msg_type, recovered=True)
                return result
            # Non-idempotent: bubble up as a typed unconfirmed error so the
            # caller's translator can surface actionable English.
            if msg_type == "FILE_WRITE":
                raise WriteUnconfirmedError(
                    wire_id=first.wire_id,
                    message=(
                        f"[LocalPC] FILE_WRITE (id={first.wire_id}) was sent but "
                        "the shim did not ACK before the connection dropped"
                    ),
                ) from first
            raise OpUnconfirmedError(
                op=msg_type,
                wire_id=first.wire_id,
                message=(
                    f"[LocalPC] {msg_type} (id={first.wire_id}) was sent but no "
                    "response arrived before the connection dropped"
                ),
            ) from first

    async def _await_reconnect_for_retry(self, *, timeout: float = 30.0) -> None:
        """Wait for the shim to be reachable again before an idempotent retry.

        Best-effort: if a connection manager isn't attached (unit tests
        stub `_rpc` directly), this returns immediately. The retry then
        re-uses ``self._ws`` and will fail fast if the WS is still dead.
        """
        manager = getattr(self, "_manager", None)
        if manager is None:
            return
        try:
            ws = await manager.wait_for(self.sandbox_id, timeout=timeout)
        except Exception:
            return
        # If the manager produced a fresh WS, swap it in so the retry rides
        # the new connection.
        if ws is not self._ws:
            self._replace_connection(
                ws, manager.get_hello(self.sandbox_id) or ShimHello()
            )

    async def _recv_loop(self, websocket: TextTransport) -> None:
        disconnect_error = ConnectionError(
            f"[LocalPC] Shim connection closed for session {self.sandbox_id[:12]}"
        )
        try:
            async for raw in websocket.iter_text():
                self._process_raw_message(raw)
        except asyncio.CancelledError:
            disconnect_error = ConnectionError(
                f"[LocalPC] Shim connection cancelled for session {self.sandbox_id[:12]}"
            )
            raise
        except Exception:
            disconnect_error = ConnectionError(
                f"[LocalPC] Shim connection lost for session {self.sandbox_id[:12]}"
            )
            logger.debug(
                "[LocalPC] Shim recv loop ended for %s",
                self.sandbox_id[:12],
                exc_info=True,
            )
        finally:
            if websocket is self._ws:
                self._fail_in_flight(disconnect_error)
                if self._manager is not None:
                    self._manager.unregister(self.sandbox_id, websocket=websocket)

    def _process_raw_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            if not isinstance(msg, dict):
                return
            self._handle_envelope_capacity(msg)
            msg_type = msg.get("type")
            if msg_type == "STATUS":
                self._handle_status_frame(msg)
                return
            if msg_type == "RECORDING_STEP":
                self._handle_recording_step(msg)
                return
            msg_id = msg.get("id")
            if isinstance(msg_id, str) and msg_id in self._streaming:
                if msg_type in (
                    "LOCAL_LLM_COMPLETION_CHUNK",
                    "LOCAL_LLM_COMPLETION_RESPONSE",
                    "ERROR",
                ):
                    queue = self._streaming[msg_id]
                    if queue.full() or len(raw.encode()) > _LLM_STREAM_MAX_FRAME_BYTES:
                        self._terminate_stream(
                            msg_id,
                            "LOCAL_LLM_STREAM_LIMIT",
                            "Local LLM response exceeded its buffer limit",
                        )
                    else:
                        queue.put_nowait(msg)
                    return
            if isinstance(msg_id, str) and msg_id in self._pending:
                expected_types, future = self._pending.pop(msg_id)
                if msg_type != "ERROR" and msg_type not in expected_types:
                    if not future.done():
                        future.set_exception(
                            ShimProtocolError(
                                f"Unexpected {msg_type!r} response for correlation id "
                                f"{msg_id}; expected {sorted(expected_types)}"
                            )
                        )
                    return
                if not future.done():
                    future.set_result(msg)
        except Exception:
            logger.exception("[LocalPC] Error processing shim message")

    def _fail_in_flight(self, error: ConnectionError) -> None:
        """Fail every waiter immediately when its connection cannot respond."""
        pending = [future for _expected, future in self._pending.values()]
        self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(error)

        for msg_id in list(self._streaming):
            self._terminate_stream(msg_id, "CONNECTION_LOST", str(error))
        for recording_id in list(self._recording_steps):
            self.close_recording(recording_id)

    def _register_stream(self, msg_id: str) -> asyncio.Queue[dict]:
        """Register a streaming request and return its inbound-frame queue.

        Used by ``_LocalLLMProxy.complete()``. The queue receives every
        CHUNK / RESPONSE / ERROR frame the recv loop sees for ``msg_id``.
        Callers MUST call :meth:`_cleanup_stream` when done so the dict
        doesn't grow without bound.
        """
        if len(self._streaming) >= _LLM_STREAM_MAX_CONCURRENT:
            raise LocalLLMError(
                "LOCAL_LLM_BUSY", "Too many Local LLM streams are active"
            )
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=_LLM_STREAM_MAX_FRAMES)
        self._streaming[msg_id] = queue
        return queue

    def _cleanup_stream(self, msg_id: str) -> None:
        """Drop the streaming queue. Safe to call multiple times."""
        self._streaming.pop(msg_id, None)

    def _terminate_stream(self, msg_id: str, code: str, message: str) -> None:
        queue = self._streaming.pop(msg_id, None)
        if queue is None:
            return
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(
            {"type": "ERROR", "payload": {"code": code, "message": message}}
        )

    def _handle_envelope_capacity(self, msg: Any) -> None:
        """Mine ``pending_capacity`` out of any shim → platform envelope.

        Per the shim partner's backpressure work, every response (and
        STATUS frame) carries the shim's current headroom. We accept it
        from either the top-level envelope or the payload — different shim
        versions have placed it in different spots, and the platform side
        shouldn't crash on either. Missing field is fine — leave capacity
        unchanged.
        """
        if not isinstance(msg, dict):
            return
        if "pending_capacity" in msg:
            self._update_pending_capacity(
                msg.get("pending_capacity"), source="envelope"
            )
        payload = msg.get("payload")
        if isinstance(payload, dict) and "pending_capacity" in payload:
            self._update_pending_capacity(
                payload.get("pending_capacity"), source="payload"
            )

    def _handle_status_frame(self, msg: Any) -> None:
        """Periodic STATUS frame: log the snapshot + refresh capacity.

        Frame shape (per shim partner spec):
            {type: "STATUS",
             payload: {in_flight, max_concurrent, queue_depth,
                       audit_log_bytes, uptime_seconds, pending_capacity?}}

        If ``pending_capacity`` is present, it wins. Otherwise we derive
        ``max_concurrent - in_flight`` as a fallback for shim versions that
        omit it. Logged at DEBUG so the snapshot is available for diagnosis
        without spamming production logs.
        """
        if not isinstance(msg, dict):
            return
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            return
        if "pending_capacity" in payload:
            self._update_pending_capacity(
                payload.get("pending_capacity"), source="STATUS"
            )
        else:
            max_concurrent = payload.get("max_concurrent")
            in_flight = payload.get("in_flight")
            try:
                if max_concurrent is not None and in_flight is not None:
                    derived = max(0, int(max_concurrent) - int(in_flight))
                    self._update_pending_capacity(derived, source="STATUS-derived")
            except (TypeError, ValueError):
                pass
        logger.debug(
            "[LocalPC] STATUS frame for %s: in_flight=%r max_concurrent=%r "
            "queue_depth=%r audit_log_bytes=%r uptime_seconds=%r capacity=%r",
            self.sandbox_id[:12],
            payload.get("in_flight"),
            payload.get("max_concurrent"),
            payload.get("queue_depth"),
            payload.get("audit_log_bytes"),
            payload.get("uptime_seconds"),
            self._pending_capacity,
        )

    # --- Workflow recording (RECORDING_STEP buffering) ---------------------

    def _ensure_recording_buffer(
        self, recording_id: str, *, started: bool = False
    ) -> "asyncio.Queue[TrajectoryStep | None] | None":
        """Return (creating if needed) the per-recording step queue.

        A confirmed START can always make room. Unsolicited frames may evict
        the oldest orphan buffer, but never a confirmed recording buffer.
        """
        started_recording_ids = self._get_started_recording_ids()
        queue = self._recording_steps.get(recording_id)
        if queue is not None:
            if started:
                started_recording_ids.add(recording_id)
                self._recording_steps.pop(recording_id)
                self._recording_steps[recording_id] = queue
            return queue

        if len(self._recording_steps) >= _MAX_RECORDING_BUFFERS:
            orphan_id = next(
                (
                    buffered_id
                    for buffered_id in self._recording_steps
                    if buffered_id not in started_recording_ids
                ),
                None,
            )
            if orphan_id is not None:
                self._drop_recording_buffer(orphan_id)
            elif started:
                self._drop_recording_buffer(next(iter(self._recording_steps)))
            else:
                logger.debug(
                    "[LocalPC] Recording buffer limit reached; dropping orphan %s",
                    recording_id,
                )
                return None

        # Reserve one slot beyond the step cap so close_recording can always
        # enqueue its sentinel without blocking or raising QueueFull.
        queue = asyncio.Queue(maxsize=_MAX_RECORDING_STEP_BUFFER + 1)
        self._recording_steps[recording_id] = queue
        if started:
            started_recording_ids.add(recording_id)
        return queue

    def _get_started_recording_ids(self) -> set[str]:
        started_recording_ids = getattr(self, "_started_recording_ids", None)
        if started_recording_ids is None:
            started_recording_ids = set()
            self._started_recording_ids = started_recording_ids
        return started_recording_ids

    @staticmethod
    def _signal_recording_buffer_closed(
        queue: "asyncio.Queue[TrajectoryStep | None]",
    ) -> None:
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(None)

    def _drop_recording_buffer(self, recording_id: str) -> None:
        queue = self._recording_steps.pop(recording_id, None)
        self._get_started_recording_ids().discard(recording_id)
        if queue is not None:
            self._signal_recording_buffer_closed(queue)

    def _handle_recording_step(self, msg: Any) -> None:
        """Fan a RECORDING_STEP frame into its per-recording queue.

        Frame shape (§6):
            {type: "RECORDING_STEP",
             payload: {recording_id, step: {TrajectoryStep}}}

        A frame for an unknown recording_id is buffered because the START
        response may still be in flight. Orphan buffers are globally bounded.
        """
        if not isinstance(msg, dict):
            return
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            return
        recording_id = str(payload.get("recording_id") or "")
        if not recording_id:
            logger.debug("[LocalPC] RECORDING_STEP without recording_id; dropping")
            return
        step_payload = payload.get("step")
        if not isinstance(step_payload, dict):
            logger.debug(
                "[LocalPC] RECORDING_STEP for %s missing step body; dropping",
                recording_id,
            )
            return
        step = TrajectoryStep.from_payload(step_payload)
        queue = self._ensure_recording_buffer(recording_id)
        if queue is None:
            return
        if queue.qsize() >= _MAX_RECORDING_STEP_BUFFER:
            logger.debug(
                "[LocalPC] Recording step buffer full for %s; dropping seq=%s",
                recording_id,
                step.seq,
            )
            return
        queue.put_nowait(step)

    async def _iter_recording_steps(self, recording_id: str):
        """Async iterator yielding TrajectoryStep frames as they arrive.

        Backs ``recording.stream_steps()``. Runs until a sentinel ``None``
        is enqueued by :meth:`close_recording`, then stops and drops the
        buffer.
        """
        queue = self._ensure_recording_buffer(recording_id, started=True)
        assert queue is not None
        try:
            while True:
                step = await queue.get()
                if step is None:
                    return
                yield step
        finally:
            if self._recording_steps.get(recording_id) is queue:
                self._recording_steps.pop(recording_id, None)
                self._get_started_recording_ids().discard(recording_id)

    def close_recording(self, recording_id: str) -> None:
        """Signal the step iterator for ``recording_id`` to stop.

        Enqueues the stop sentinel so a live ``stream_steps()`` consumer
        finishes cleanly after STOP_RECORDING. Safe to call multiple times
        and for recordings that never streamed.
        """
        self._drop_recording_buffer(recording_id)

    async def pause(self) -> None:
        pass  # no billing on local machine

    async def kill(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass
        if not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._manager is not None:
            self._manager.remove_shim(self.sandbox_id, self)


def _record_response_outcome(msg_type: str, response: dict) -> str:
    """Count a wire ERROR under its code; return the attempt's outcome."""
    if response.get("type") != "ERROR":
        return "ok"
    payload = response.get("payload")
    code = payload.get("code") if isinstance(payload, dict) else None
    record_rpc_error(msg_type, str(code or "INTERNAL_ERROR"))
    return "error"
