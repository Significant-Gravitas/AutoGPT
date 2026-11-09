"""
Protocol messages — pydantic models, envelope, and helpers.

Every wire message has the envelope:

    {"type": ..., "id": ..., "ts": ..., "payload": ...}

See docs/PROTOCOL.md for the authoritative spec. The models below are the
single source of truth on the shim side; if a model and the doc disagree,
the doc wins — fix the model.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

# ── Protocol version ─────────────────────────────────────────────────────────

# Wire protocol version this shim speaks. Format: "<major>.<minor>" string.
# Negotiation rules (see docs/PROTOCOL.md → Versioning):
#   * Major MUST match between shim and platform. Mismatch → connection
#     closed with WS code 4426 and reason PROTOCOL_VERSION_MISMATCH; shim
#     MUST NOT auto-reconnect.
#   * Minor floor wins: effective negotiated minor is min(shim, platform).
#     Both sides MUST tolerate forward-compatible additions within a major.
#   * Every envelope SHOULD carry `version` matching the negotiated value,
#     but receivers MUST be lenient — HELLO-time negotiation is the truth.
VERSION: str = "2.0"

# The platform relay and shim both cap decoded WebSocket messages at 16 MiB.
# JSON can expand a text byte to six bytes (for example NUL -> ``\u0000``),
# so text-producing handlers retain a smaller payload with framing headroom.
MAX_WEBSOCKET_MESSAGE_BYTES: int = 16 * 1024 * 1024
MAX_WIRE_TEXT_CONTENT_BYTES: int = (MAX_WEBSOCKET_MESSAGE_BYTES - 64 * 1024) // 6
MAX_WIRE_BASE64_CONTENT_BYTES: int = (MAX_WEBSOCKET_MESSAGE_BYTES - 64 * 1024) * 3 // 4
SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$"


def _split_version(v: str) -> tuple[int, int]:
    """Parse a "major.minor" string. Raises ValueError on malformed input.

    Pre-release/build suffixes are not part of the wire format — we only
    need major.minor for compat negotiation.
    """
    parts = v.split(".")
    if len(parts) != 2:
        raise ValueError(f"protocol version must be 'major.minor', got {v!r}")
    # int() would also take signs, spaces, underscores and non-ASCII digits.
    if not all(part.isascii() and part.isdigit() for part in parts):
        raise ValueError(f"protocol version components must be ASCII digits, got {v!r}")
    try:
        return int(parts[0]), int(parts[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"protocol version components must be integers, got {v!r}") from exc


def negotiate_version(shim_max: str, platform_max: str) -> str:
    """Compute the effective negotiated protocol version.

    Returns a "major.minor" string. Raises ProtocolVersionMismatch when the
    majors don't match — caller is responsible for tearing down the WS with
    code 4426 and surfacing the structured close reason.
    """
    shim_major, shim_minor = _split_version(shim_max)
    plat_major, plat_minor = _split_version(platform_max)
    if shim_major != plat_major:
        raise ProtocolVersionMismatch(
            shim_max=shim_max,
            platform_max=platform_max,
            hint=(
                "Major version mismatch — update the side running the older "
                "major. Shims and platforms within a major are "
                "forward-compatible on minor."
            ),
        )
    return f"{shim_major}.{min(shim_minor, plat_minor)}"


def same_major(version: str, negotiated: str) -> bool:
    """Whether a frame's `version` shares the negotiated major. A version
    that doesn't parse never does."""
    try:
        return _split_version(version)[0] == _split_version(negotiated)[0]
    except ValueError:
        return False


class ProtocolVersionMismatch(Exception):
    """Raised when shim_max and platform_max have different majors.

    Carries the structured fields that go into the WS close reason and
    the SESSION_REVOKED-style audit record.
    """

    def __init__(self, *, shim_max: str, platform_max: str, hint: str) -> None:
        self.shim_max = shim_max
        self.platform_max = platform_max
        self.hint = hint
        super().__init__(
            f"protocol version mismatch: shim_max={shim_max} platform_max={platform_max}"
        )

    def to_close_reason(self) -> dict[str, str]:
        """The fields the 4426 close reason carries. The hint is left out: a
        close reason holds at most 123 bytes (RFC 6455 §5.5), and the hint
        alone is about that long, so it goes to the log instead."""
        return {
            "error": "PROTOCOL_VERSION_MISMATCH",
            "shim_max": self.shim_max,
            "platform_max": self.platform_max,
        }


# ── Enums ────────────────────────────────────────────────────────────────────


class MessageType(StrEnum):
    HELLO = "HELLO"
    HELLO_ACK = "HELLO_ACK"
    DIRECTORY_LIST_REQUEST = "DIRECTORY_LIST_REQUEST"
    DIRECTORY_LIST_RESPONSE = "DIRECTORY_LIST_RESPONSE"
    ATTACH_SESSION = "ATTACH_SESSION"
    SESSION_ATTACHED = "SESSION_ATTACHED"
    ACTIVATE_SESSION = "ACTIVATE_SESSION"
    SESSION_ACTIVATED = "SESSION_ACTIVATED"
    RESTORE_SESSION = "RESTORE_SESSION"
    SESSION_RESTORED = "SESSION_RESTORED"
    DETACH_SESSION = "DETACH_SESSION"
    EXECUTE_COMMAND = "EXECUTE_COMMAND"
    COMMAND_RESULT = "COMMAND_RESULT"
    FILE_READ = "FILE_READ"
    FILE_CONTENTS = "FILE_CONTENTS"
    FILE_WRITE = "FILE_WRITE"
    FILE_STAT = "FILE_STAT"
    FILE_STAT_RESPONSE = "FILE_STAT_RESPONSE"
    FILE_LIST = "FILE_LIST"
    FILE_LIST_RESPONSE = "FILE_LIST_RESPONSE"
    FILE_DELETE = "FILE_DELETE"
    FILE_MOVE = "FILE_MOVE"
    # Computer use relays cua-driver's own tools (see docs/CUA_DRIVER.md).
    COMPUTER_TOOLS_REQUEST = "COMPUTER_TOOLS_REQUEST"
    COMPUTER_TOOLS_RESPONSE = "COMPUTER_TOOLS_RESPONSE"
    COMPUTER_TOOL_CALL = "COMPUTER_TOOL_CALL"
    COMPUTER_TOOL_RESULT = "COMPUTER_TOOL_RESULT"
    ACK = "ACK"
    ERROR = "ERROR"
    PING = "PING"
    PONG = "PONG"
    # Session ownership / lifecycle (see PROTOCOL.md → Session ownership).
    SESSION_REVOKED = "SESSION_REVOKED"
    # Proactive backpressure / health (see PROTOCOL.md → Backpressure).
    STATUS = "STATUS"
    # Local LLM routing (see PROTOCOL.md → Local LLM + docs/LOCAL_LLM.md).
    LOCAL_LLM_COMPLETION = "LOCAL_LLM_COMPLETION"
    LOCAL_LLM_COMPLETION_CHUNK = "LOCAL_LLM_COMPLETION_CHUNK"
    LOCAL_LLM_COMPLETION_RESPONSE = "LOCAL_LLM_COMPLETION_RESPONSE"
    # Workflow recording (see docs/WORKFLOW_RECORDING.md §6).
    REQUEST_RECORDING_CONSENT = "REQUEST_RECORDING_CONSENT"
    RECORDING_CONSENT_RESULT = "RECORDING_CONSENT_RESULT"
    START_RECORDING = "START_RECORDING"
    RECORDING_STARTED = "RECORDING_STARTED"
    STOP_RECORDING = "STOP_RECORDING"
    RECORDING_SUMMARY = "RECORDING_SUMMARY"
    RECORDING_FETCH = "RECORDING_FETCH"
    RECORDING_DATA = "RECORDING_DATA"
    APPLY_RECORDING_REVIEW = "APPLY_RECORDING_REVIEW"
    RECORDING_REVIEW_APPLIED = "RECORDING_REVIEW_APPLIED"
    # Unsolicited co-pilot-mode step stream — modeled like STATUS: not acked,
    # not counted against in-flight / max_concurrent, never auto-retried.
    RECORDING_STEP = "RECORDING_STEP"


class ErrorCode(StrEnum):
    PATH_OUTSIDE_ALLOWED_ROOT = "PATH_OUTSIDE_ALLOWED_ROOT"
    PATH_RESERVED_NAME = "PATH_RESERVED_NAME"
    PATH_INVALID_CHARS = "PATH_INVALID_CHARS"
    PATH_NOT_FOUND = "PATH_NOT_FOUND"
    PATH_NOT_EMPTY = "PATH_NOT_EMPTY"
    PATH_EXISTS = "PATH_EXISTS"
    COMMAND_TIMEOUT = "COMMAND_TIMEOUT"
    SHELL_NOT_AVAILABLE = "SHELL_NOT_AVAILABLE"
    UNSUPPORTED_ARCH = "UNSUPPORTED_ARCH"
    CAPABILITY_NOT_GRANTED = "CAPABILITY_NOT_GRANTED"
    AUTH_FAILED = "AUTH_FAILED"
    SHIM_OVERLOADED = "SHIM_OVERLOADED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    DIRECTORY_REFERENCE_INVALID = "DIRECTORY_REFERENCE_INVALID"
    DIRECTORY_UNAVAILABLE = "DIRECTORY_UNAVAILABLE"
    ROOT_GRANT_INVALID = "ROOT_GRANT_INVALID"
    SESSION_NOT_ATTACHED = "SESSION_NOT_ATTACHED"
    SESSION_ALREADY_ACTIVE = "SESSION_ALREADY_ACTIVE"
    SESSION_REVISION_MISMATCH = "SESSION_REVISION_MISMATCH"
    # Computer-use additions (see docs/COMPUTER_USE.md).
    PERMISSION_PENDING = "PERMISSION_PENDING"
    FEATURE_NOT_SUPPORTED = "FEATURE_NOT_SUPPORTED"
    CLIPBOARD_CONCEALED = "CLIPBOARD_CONCEALED"
    # A computer tool call used a disk argument the person didn't approve.
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    # Local LLM additions (see docs/LOCAL_LLM.md).
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    LOCAL_LLM_BUSY = "LOCAL_LLM_BUSY"
    LOCAL_LLM_FAILED = "LOCAL_LLM_FAILED"
    # Workflow recording additions (see docs/WORKFLOW_RECORDING.md §6).
    RECORDING_NOT_FOUND = "RECORDING_NOT_FOUND"
    RECORDING_CHANNEL_UNAVAILABLE = "RECORDING_CHANNEL_UNAVAILABLE"
    RECORDING_ALREADY_ACTIVE = "RECORDING_ALREADY_ACTIVE"
    RECORDING_REVIEW_INVALID = "RECORDING_REVIEW_INVALID"
    # START_RECORDING arrived without a valid shim-issued consent token.
    CONSENT_REQUIRED = "CONSENT_REQUIRED"
    # The requested interpretation_route needs a local model the machine
    # lacks, and the user declined the cloud fallback (§9.1, §10).
    INTERPRETATION_UNAVAILABLE = "INTERPRETATION_UNAVAILABLE"


class Platform(StrEnum):
    DARWIN = "darwin"
    LINUX = "linux"
    WINDOWS = "windows"
    WSL2 = "wsl2"


class Arch(StrEnum):
    X86_64 = "x86_64"
    ARM64 = "arm64"


class Shell(StrEnum):
    AUTO = "auto"
    BASH = "bash"
    SH = "sh"
    ZSH = "zsh"
    PWSH = "pwsh"
    POWERSHELL = "powershell"
    CMD = "cmd"


class Encoding(StrEnum):
    UTF8 = "utf-8"
    BASE64 = "base64"


class FileFormat(StrEnum):
    TEXT = "text"
    BYTES = "bytes"


# ── Workflow-recording type aliases (see docs/WORKFLOW_RECORDING.md) ──────────
# Defined here (ahead of the payload models) because HELLO advertises recording
# channels + routes. The recording payload/step models live further down.

# Interaction modes (§2). Demonstration buffers + fetch; co-pilot streams.
RecordingMode = Literal["demonstration", "copilot"]

# On-device interpretation routes (§3). The default keeps pixels local;
# screenshots_to_cloud is the only route that crosses a new privacy line and
# therefore the only one that needs the §9.1 consent prompt.
InterpretationRoute = Literal[
    "extract_then_cloud",  # default — text/structure only leaves the machine
    "local_vlm",  # pixel-local upgrade; structured browser review still occurs
    "screenshots_to_cloud",  # fallback; gated on the shim-enforced consent
]

# Capture channels (§4). The floor is universal; browser/desktop_ax enrich it.
RecordingChannel = Literal["floor", "browser", "desktop_ax"]


# ── Payload models ───────────────────────────────────────────────────────────


class _Payload(BaseModel):
    """Common config — payloads forbid extras so unknown fields fail loudly."""

    model_config = ConfigDict(extra="forbid")


class ComputerDriverInfo(BaseModel):
    """Which GUI driver serves computer use on this machine. `runtime` is
    `direct`: the driver process the shim started owns the automation
    runtime. `service`, a driver relaying to a shared service such as
    CuaDriver.app, is reserved."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    version: str = Field(min_length=1, max_length=64)
    runtime: Literal["direct", "service"]


class HelloPayload(_Payload):
    shim_version: str
    machine_id: str
    display_name: str = Field(default="", max_length=128)
    platform: Platform
    arch: Arch
    screen_resolution: tuple[int, int] | None = None
    capabilities: list[str]
    allowed_root: str | None
    local_llm_models: list[str] = Field(default_factory=list)
    hardware_devices: list[dict[str, Any]] = Field(default_factory=list)
    # The driver behind computer use and the names of the tools this machine
    # offers, when the capability is advertised. Their schemas come from
    # COMPUTER_TOOLS_REQUEST (see docs/CUA_DRIVER.md).
    computer_use_driver: ComputerDriverInfo | None = None
    computer_use_tools: list[str] = Field(default_factory=list)
    # Workflow-recording advertisement (see docs/WORKFLOW_RECORDING.md §6).
    # `recording_channels` are the capture channels this shim can offer (floor
    # is universal; browser/desktop_ax light up as their enrichers land).
    # `recording_routes` are the interpretation routes available on this
    # machine right now — so the platform can gate route selection. Empty when
    # the recording capability isn't advertised.
    recording_channels: list[RecordingChannel] = Field(default_factory=list)
    recording_routes: list[InterpretationRoute] = Field(default_factory=list)
    # Highest wire-protocol version this shim supports. "major.minor".
    # Receiving side negotiates the effective version (see VERSION docs).
    protocol_version: str = VERSION


class HelloAckPayload(_Payload):
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)
    connection_id: str | None = None
    granted_capabilities: list[str]
    max_file_size_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    command_timeout_seconds: int = Field(default=30, gt=0)
    max_concurrent: int = Field(default=4, gt=0)
    # Highest wire-protocol version this platform supports. "major.minor".
    # Effective negotiated version = same major, min(shim_minor, plat_minor).
    protocol_version: str = VERSION


class DirectoryListRequestPayload(_Payload):
    browse_id: str | None = Field(default=None, min_length=1, max_length=256)
    directory_ref: str | None = Field(default=None, min_length=1, max_length=256)
    cursor: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_reference_pair(self) -> DirectoryListRequestPayload:
        if (self.browse_id is None) != (self.directory_ref is None):
            raise ValueError("browse_id and directory_ref must both be null or both be set")
        if self.cursor is not None and self.browse_id is None:
            raise ValueError("cursor requires browse_id and directory_ref")
        return self


class DirectoryReferencePayload(_Payload):
    directory_ref: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=1024)
    path: str = Field(min_length=1, max_length=32767)


class DirectoryListResponsePayload(_Payload):
    browse_id: str = Field(min_length=1, max_length=256)
    current: DirectoryReferencePayload | None = None
    parent_ref: str | None = Field(default=None, min_length=1, max_length=256)
    entries: list[DirectoryReferencePayload] = Field(max_length=200)
    next_cursor: str | None = Field(default=None, min_length=1, max_length=256)
    truncated: bool = False
    expires_at: float


class AttachSessionPayload(_Payload):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    browse_id: str = Field(min_length=1, max_length=256)
    directory_ref: str = Field(min_length=1, max_length=256)
    expected_connection_id: str | None = Field(default=None, min_length=1, max_length=256)


class SessionAttachedPayload(_Payload):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    allowed_root: str = Field(min_length=1, max_length=32767)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: int = Field(ge=1)
    root_grant: str = Field(min_length=1, max_length=131072)


class ActivateSessionPayload(_Payload):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    revision: int = Field(ge=1)


class SessionActivatedPayload(_Payload):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    allowed_root: str = Field(min_length=1, max_length=32767)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: int = Field(ge=1)


class RestoreSessionPayload(_Payload):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    root_grant: str = Field(min_length=1, max_length=131072)


class SessionRestoredPayload(_Payload):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    allowed_root: str = Field(min_length=1, max_length=32767)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: int = Field(ge=1)
    root_grant: str = Field(min_length=1, max_length=131072)


class DetachSessionPayload(_Payload):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)


class ExecuteCommandPayload(_Payload):
    command: str | None = None
    argv: list[str] | None = None
    shell: Shell = Shell.AUTO
    cwd: str | None = None
    timeout_seconds: int | None = Field(default=None, gt=0)
    env: dict[str, str] = Field(default_factory=dict)


class CommandResultPayload(_Payload):
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_seconds: float
    output_truncated: bool = False


class FileReadPayload(_Payload):
    path: str
    encoding: Encoding = Encoding.UTF8
    format: FileFormat = FileFormat.TEXT
    offset: int = Field(default=0, ge=0)
    length: int | None = Field(default=None, gt=0)


class FileContentsPayload(_Payload):
    content: str
    encoding: Encoding
    size_bytes: int
    truncated: bool = False


class FileWritePayload(_Payload):
    path: str
    content: str
    encoding: Encoding = Encoding.UTF8
    create_parents: bool = False


class AckPayload(_Payload):
    ok: bool = True


class FileStatPayload(_Payload):
    path: str
    follow_symlinks: bool = True


class FileStatResponsePayload(_Payload):
    exists: bool
    is_file: bool | None = None
    is_dir: bool | None = None
    is_symlink: bool | None = None
    size_bytes: int | None = None
    mtime: float | None = None
    ctime: float | None = None
    mode: str | None = None
    owner_uid: int | None = None
    owner_gid: int | None = None
    mime_type: str | None = None
    path: str | None = None  # resolved path, when follow_symlinks=True


class FileListPayload(_Payload):
    path: str
    glob: str | None = None
    recursive: bool = False
    include_hidden: bool = False
    max_entries: int = 1000


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    is_file: bool
    is_dir: bool
    is_symlink: bool
    size_bytes: int | None = None
    mtime: float | None = None


class FileListResponsePayload(_Payload):
    entries: list[FileEntry]
    truncated: bool = False


class FileDeletePayload(_Payload):
    path: str
    recursive: bool = False
    missing_ok: bool = False


class FileMovePayload(_Payload):
    src: str
    dst: str
    overwrite: bool = False


COMPUTER_TOOL_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
# Longest a platform may ask one computer tool call to run.
MAX_COMPUTER_TOOL_TIMEOUT_SECONDS = 600


class ComputerToolsRequestPayload(_Payload):
    pass


class ComputerToolSpec(BaseModel):
    """One driver tool the platform may call, as the driver describes it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=COMPUTER_TOOL_NAME_PATTERN)
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] | None = None
    # Arguments that reach the computer's disk, each with what it does there.
    # A call using one runs only when it names the argument in
    # `approved_arguments`, which the platform sets once the person approved.
    approval_arguments: dict[str, str] = Field(default_factory=dict)


class ComputerToolsResponsePayload(_Payload):
    driver: ComputerDriverInfo
    # The driver's own operating guidance, with the shim's notes appended.
    instructions: str
    tools: list[ComputerToolSpec]


class ComputerToolCallPayload(_Payload):
    tool: str = Field(pattern=COMPUTER_TOOL_NAME_PATTERN)
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int | None = Field(default=None, gt=0, le=MAX_COMPUTER_TOOL_TIMEOUT_SECONDS)
    # The tool's approval arguments the person approved for this call.
    approved_arguments: list[str] = Field(default_factory=list, max_length=8)


class ComputerTextContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ComputerImageContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    data: str  # base64
    mime_type: str


ComputerContent = Annotated[ComputerTextContent | ComputerImageContent, Field(discriminator="type")]


class ComputerToolResultPayload(_Payload):
    """What the driver returned. `is_error` is the driver's own verdict: a
    refused or failed action the model should read, not a shim failure,
    which comes back as ERROR instead."""

    tool: str = Field(pattern=COMPUTER_TOOL_NAME_PATTERN)
    is_error: bool = False
    content: list[ComputerContent] = Field(default_factory=list)
    structured_content: dict[str, Any] | None = None


class ErrorPayload(_Payload):
    code: ErrorCode | str
    message: str
    fatal: bool = False
    details: dict[str, Any] | None = None


class PingPongPayload(_Payload):
    pass


# ── Session ownership ────────────────────────────────────────────────────────

# Reason values for SESSION_REVOKED. Extending this enum is a forward-compatible
# minor-version change — receivers tolerate unknown reasons and treat them as
# the catch-all "revoked, do not auto-reconnect" case.
SESSION_REVOKED_REASONS = (
    "another_shim_connected",
    "user_revoked",
    "platform_shutdown",
)


class SessionRevokedPayload(_Payload):
    """Platform → shim notification that this session is no longer valid.

    On receipt the shim MUST:
      1. Audit a SESSION_REVOKED record with the carried reason.
      2. Send no further frames on this connection.
      3. Gracefully close its half of the WebSocket.
      4. NOT auto-reconnect to the same session_id.

    `new_shim_machine_id` is set when reason is `another_shim_connected` and
    the platform knows the takeover machine; it's purely informational.
    """

    reason: str
    new_shim_machine_id: str | None = None


# ── Backpressure / health (#38) ──────────────────────────────────────────────


class StatusPayload(_Payload):
    """Periodic, unsolicited shim health snapshot.

    Emitted every 30s during a healthy session and on the full → not-full
    capacity edge (so the platform's throttle releases promptly). All
    counts reflect the shim's view at emission time; the platform should
    treat them as advisory — `pending_capacity` on response envelopes
    remains the per-response authoritative number.
    """

    in_flight: int
    max_concurrent: int
    queue_depth: int = 0
    audit_log_bytes: int = 0
    uptime_seconds: float = 0.0


# ── Local LLM routing (see docs/LOCAL_LLM.md) ────────────────────────────────


class LocalLLMMessage(BaseModel):
    """One element of `LOCAL_LLM_COMPLETION.messages` — chat-completions style."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=262144)


# Finish reasons follow the chat-completions vocabulary. v1.x: these three;
# future minor versions MAY add more (e.g. "tool_call") — receivers tolerate.
LocalLLMFinishReason = Literal["stop", "length", "content_filter"]


class LocalLLMCompletionPayload(_Payload):
    """Request a completion from the local LLM backend.

    `model` MUST be in `HELLO.local_llm_models`. `stream=True` (default)
    emits LOCAL_LLM_COMPLETION_CHUNK frames followed by a single
    LOCAL_LLM_COMPLETION_RESPONSE; `stream=False` emits only the
    response.
    """

    model: str = Field(min_length=1, max_length=256)
    messages: list[LocalLLMMessage] = Field(min_length=1, max_length=256)
    max_tokens: int = Field(default=1024, ge=1, le=8192, strict=True)
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    stream: bool = True

    @model_validator(mode="after")
    def bound_prompt_bytes(self) -> LocalLLMCompletionPayload:
        if sum(len(message.content.encode("utf-8")) for message in self.messages) > 524288:
            raise ValueError("Local model prompt exceeds its size limit")
        return self


class LocalLLMCompletionChunkPayload(_Payload):
    """One streaming chunk. Final chunk in a stream has `finish_reason` set."""

    delta: str
    finish_reason: LocalLLMFinishReason | None = None


class LocalLLMTokensUsage(BaseModel):
    """Best-effort token accounting. Any sub-field MAY be null if the
    backend doesn't report it."""

    model_config = ConfigDict(extra="forbid")

    prompt: int | None = None
    completion: int | None = None
    total: int | None = None


class LocalLLMCompletionResponsePayload(_Payload):
    """Terminal frame for a LOCAL_LLM_COMPLETION request.

    Always emitted exactly once per request, after any/all CHUNK frames
    in the streaming case or as the sole response in the non-streaming
    case.
    """

    content: str
    finish_reason: LocalLLMFinishReason
    tokens: LocalLLMTokensUsage = Field(default_factory=LocalLLMTokensUsage)
    duration_seconds: float


# ── Workflow recording (see docs/WORKFLOW_RECORDING.md) ──────────────────────

# Semantic verbs (§1.1) — intentful actions, not a raw input tape. `wait` and
# `assert` are the replay-control verbs the recorder synthesizes or the user
# adds (§9).
SemanticAction = Literal[
    "navigate",
    "fill",
    "select",
    "click",
    "submit",
    "upload",
    "launch_app",
    "focus_window",
    "copy",
    "paste",
    "run",
    "file_op",
    "wait",
    "assert",
]

# Enrichment kind (§1). dom = browser DOM channel resolved selectors; ax =
# desktop accessibility tree resolved a path; none = pure visual step, replay
# degrades to vision grounding (§7) — not a failure, just lower fidelity.
EnrichmentKind = Literal["dom", "ax", "none"]

# Value type for the demonstrated input (§1). `secret` marks a field whose raw
# value was dropped by best-effort hygiene redaction (§9).
ValueType = Literal["text", "email", "number", "date", "secret", "file", "enum"]


class Selector(BaseModel):
    """One browser-DOM selector candidate. Replay tries these most-stable-first
    (§7: id > name > label > role+text > xpath)."""

    model_config = ConfigDict(extra="forbid")

    strategy: str  # e.g. "id" | "name" | "label" | "role+text" | "xpath"
    value: str


class StepEnrichment(BaseModel):
    """Additive structure on a step, present only when a channel resolved it.

    `kind` discriminates the source. For `dom`, `selectors` is populated by the
    browser channel; for `ax`, `ax_path` by the desktop a11y channel; for
    `none`, neither resolved and replay falls back to visual grounding (§7).
    """

    model_config = ConfigDict(extra="forbid")

    kind: EnrichmentKind
    selectors: list[Selector] = Field(default_factory=list)  # browser DOM only
    ax_path: str | None = None  # desktop a11y only — element path in the AX tree
    role: str | None = None
    label: str | None = None
    url: str | None = None


class StepValue(BaseModel):
    """The demonstrated input value for a step (§1).

    `raw` is handled per interpretation route (§3). When best-effort hygiene
    (§9) flags a secret, `raw` is set to null and `type` to `secret`.
    `is_parameter` is None until inference is CONFIRMED during generalization
    (§8) — never an auto-saved guess.
    """

    model_config = ConfigDict(extra="forbid")

    raw: str | None = None
    type: ValueType = "text"
    is_parameter: bool | None = None


class TrajectoryStep(BaseModel):
    """One ordered step: what the user did + what the screen looked like (§1).

    The floor fields (`action`, `screenshot_ref`, `cursor`, `active_app`,
    `active_window`) are ALWAYS present and form the primary key. The
    `enrichment` block is additive. `screenshot_ref` is a stub id — NEVER
    inline image bytes.
    """

    model_config = ConfigDict(extra="forbid")

    seq: int
    ts: float
    actor: Literal["human", "agent"] = "human"

    # --- the universal floor (always present) ---
    action: SemanticAction
    screenshot_ref: str  # pre-action frame, stub id — never inline bytes
    cursor: tuple[int, int]  # where the action landed, display-global px
    active_app: str
    active_window: str

    # --- enrichment: present only when the channel resolved it ---
    enrichment: StepEnrichment = Field(default_factory=lambda: StepEnrichment(kind="none"))

    value: StepValue = Field(default_factory=StepValue)

    narration: str | None = None  # co-pilot mode live narration
    outcome: Literal["ok", "error", "unknown"] = "ok"
    redacted: bool = False


class WorkflowRecording(BaseModel):
    """A full recording: ordered trajectory + metadata (§1).

    Carried in RECORDING_DATA after redaction. `redaction_applied` reflects
    that best-effort hygiene ran (§9) — it is NOT a guarantee that all secrets
    were caught; the real privacy control is `interpretation_route` (§3).
    """

    model_config = ConfigDict(extra="forbid")

    recording_id: str  # "rec_<uuid>" — shim-minted, never reused
    version: str = "1.0"
    created_at: float
    machine_id: str
    interpretation_route: InterpretationRoute
    steps: list[TrajectoryStep] = Field(default_factory=list)
    redaction_applied: bool = True


class EnrichmentCoverage(BaseModel):
    """Per-kind step counts for the recording summary (§6)."""

    model_config = ConfigDict(extra="forbid")

    dom: int = 0
    ax: int = 0
    none: int = 0


class RequestRecordingConsentPayload(_Payload):
    """platform → shim. Asks the shim to render its local consent prompt."""

    mode: RecordingMode
    interpretation_route: InterpretationRoute = "extract_then_cloud"
    channels: list[RecordingChannel] = Field(
        default_factory=lambda: ["floor"]  # type: ignore[arg-type]
    )


class RecordingConsentResultPayload(_Payload):
    """shim → platform. The token exists only after local approval."""

    approved: bool
    mode: RecordingMode
    interpretation_route: InterpretationRoute
    consent_token: str | None = None
    expires_at: float | None = None


class StartRecordingPayload(_Payload):
    """platform → shim. `consent_token` is REQUIRED and must be a valid
    shim-issued token — the platform cannot self-assert consent (§9). A START
    without a valid token gets CONSENT_REQUIRED."""

    mode: RecordingMode
    interpretation_route: InterpretationRoute = "extract_then_cloud"
    channels: list[RecordingChannel] = Field(
        default_factory=lambda: ["floor"]  # type: ignore[arg-type]
    )
    consent_token: str


class RecordingStartedPayload(_Payload):
    recording_id: str


class StopRecordingPayload(_Payload):
    recording_id: str


class RecordingSummaryPayload(_Payload):
    recording_id: str
    step_count: int
    enrichment_coverage: EnrichmentCoverage = Field(default_factory=EnrichmentCoverage)
    duration_seconds: float


class RecordingFetchPayload(_Payload):
    recording_id: str


class RecordingDataPayload(_Payload):
    """The full WorkflowRecording, post-redaction (§6)."""

    recording: WorkflowRecording


class ApplyRecordingReviewPayload(_Payload):
    """User-approved removals and redactions applied to the shim-side copy."""

    recording_id: str
    removed_step_seqs: list[int] = Field(default_factory=list)
    redacted_step_seqs: list[int] = Field(default_factory=list)


class RecordingReviewAppliedPayload(_Payload):
    recording_id: str
    step_count: int


class RecordingStepPayload(_Payload):
    """One streamed step (co-pilot mode only).

    Unsolicited, non-acked, out-of-band — modeled like STATUS. Carries the
    recording_id so the platform can correlate, plus the step itself.
    """

    recording_id: str
    step: TrajectoryStep


# ── Envelopes ────────────────────────────────────────────────────────────────


class _Envelope(BaseModel):
    """Per-type envelope. Subclasses fix `type` and `payload`.

    The `version` field carries the wire-protocol version this sender speaks
    (or, post-HELLO_ACK, the negotiated version). Receivers MUST be lenient:
    if it's missing or differs in minor from the negotiation, treat the
    HELLO-time negotiation as truth. A differing major on a non-HELLO frame
    is a hard error (drop the frame, log loudly).

    `pending_capacity` is the shim's free request slots AFTER this frame
    (max_concurrent - in_flight). Shim-→-platform response frames populate
    it as a proactive backpressure signal so the platform can pause
    issuance before SHIM_OVERLOADED kicks in. Always-null on platform-→-shim
    requests and on shim-internal frames like HELLO. See
    PROTOCOL.md → Backpressure.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    ts: float
    version: str = VERSION
    pending_capacity: int | None = None


class HelloMessage(_Envelope):
    type: Literal[MessageType.HELLO] = MessageType.HELLO
    payload: HelloPayload


class HelloAckMessage(_Envelope):
    type: Literal[MessageType.HELLO_ACK] = MessageType.HELLO_ACK
    payload: HelloAckPayload


class DirectoryListRequestMessage(_Envelope):
    type: Literal[MessageType.DIRECTORY_LIST_REQUEST] = MessageType.DIRECTORY_LIST_REQUEST
    payload: DirectoryListRequestPayload


class DirectoryListResponseMessage(_Envelope):
    type: Literal[MessageType.DIRECTORY_LIST_RESPONSE] = MessageType.DIRECTORY_LIST_RESPONSE
    payload: DirectoryListResponsePayload


class AttachSessionMessage(_Envelope):
    type: Literal[MessageType.ATTACH_SESSION] = MessageType.ATTACH_SESSION
    payload: AttachSessionPayload


class SessionAttachedMessage(_Envelope):
    type: Literal[MessageType.SESSION_ATTACHED] = MessageType.SESSION_ATTACHED
    payload: SessionAttachedPayload


class ActivateSessionMessage(_Envelope):
    type: Literal[MessageType.ACTIVATE_SESSION] = MessageType.ACTIVATE_SESSION
    payload: ActivateSessionPayload


class SessionActivatedMessage(_Envelope):
    type: Literal[MessageType.SESSION_ACTIVATED] = MessageType.SESSION_ACTIVATED
    payload: SessionActivatedPayload


class RestoreSessionMessage(_Envelope):
    type: Literal[MessageType.RESTORE_SESSION] = MessageType.RESTORE_SESSION
    payload: RestoreSessionPayload


class SessionRestoredMessage(_Envelope):
    type: Literal[MessageType.SESSION_RESTORED] = MessageType.SESSION_RESTORED
    payload: SessionRestoredPayload


class DetachSessionMessage(_Envelope):
    type: Literal[MessageType.DETACH_SESSION] = MessageType.DETACH_SESSION
    payload: DetachSessionPayload


class ExecuteCommandMessage(_Envelope):
    type: Literal[MessageType.EXECUTE_COMMAND] = MessageType.EXECUTE_COMMAND
    payload: ExecuteCommandPayload


class CommandResultMessage(_Envelope):
    type: Literal[MessageType.COMMAND_RESULT] = MessageType.COMMAND_RESULT
    payload: CommandResultPayload


class FileReadMessage(_Envelope):
    type: Literal[MessageType.FILE_READ] = MessageType.FILE_READ
    payload: FileReadPayload


class FileContentsMessage(_Envelope):
    type: Literal[MessageType.FILE_CONTENTS] = MessageType.FILE_CONTENTS
    payload: FileContentsPayload


class FileWriteMessage(_Envelope):
    type: Literal[MessageType.FILE_WRITE] = MessageType.FILE_WRITE
    payload: FileWritePayload


class AckMessage(_Envelope):
    type: Literal[MessageType.ACK] = MessageType.ACK
    payload: AckPayload


class FileStatMessage(_Envelope):
    type: Literal[MessageType.FILE_STAT] = MessageType.FILE_STAT
    payload: FileStatPayload


class FileStatResponseMessage(_Envelope):
    type: Literal[MessageType.FILE_STAT_RESPONSE] = MessageType.FILE_STAT_RESPONSE
    payload: FileStatResponsePayload


class FileListMessage(_Envelope):
    type: Literal[MessageType.FILE_LIST] = MessageType.FILE_LIST
    payload: FileListPayload


class FileListResponseMessage(_Envelope):
    type: Literal[MessageType.FILE_LIST_RESPONSE] = MessageType.FILE_LIST_RESPONSE
    payload: FileListResponsePayload


class FileDeleteMessage(_Envelope):
    type: Literal[MessageType.FILE_DELETE] = MessageType.FILE_DELETE
    payload: FileDeletePayload


class FileMoveMessage(_Envelope):
    type: Literal[MessageType.FILE_MOVE] = MessageType.FILE_MOVE
    payload: FileMovePayload


class ComputerToolsRequestMessage(_Envelope):
    type: Literal[MessageType.COMPUTER_TOOLS_REQUEST] = MessageType.COMPUTER_TOOLS_REQUEST
    payload: ComputerToolsRequestPayload = Field(default_factory=ComputerToolsRequestPayload)


class ComputerToolsResponseMessage(_Envelope):
    type: Literal[MessageType.COMPUTER_TOOLS_RESPONSE] = MessageType.COMPUTER_TOOLS_RESPONSE
    payload: ComputerToolsResponsePayload


class ComputerToolCallMessage(_Envelope):
    type: Literal[MessageType.COMPUTER_TOOL_CALL] = MessageType.COMPUTER_TOOL_CALL
    payload: ComputerToolCallPayload


class ComputerToolResultMessage(_Envelope):
    type: Literal[MessageType.COMPUTER_TOOL_RESULT] = MessageType.COMPUTER_TOOL_RESULT
    payload: ComputerToolResultPayload


class ErrorMessage(_Envelope):
    type: Literal[MessageType.ERROR] = MessageType.ERROR
    payload: ErrorPayload


class PingMessage(_Envelope):
    type: Literal[MessageType.PING] = MessageType.PING
    payload: PingPongPayload = Field(default_factory=PingPongPayload)


class PongMessage(_Envelope):
    type: Literal[MessageType.PONG] = MessageType.PONG
    payload: PingPongPayload = Field(default_factory=PingPongPayload)


class SessionRevokedMessage(_Envelope):
    type: Literal[MessageType.SESSION_REVOKED] = MessageType.SESSION_REVOKED
    payload: SessionRevokedPayload


class StatusMessage(_Envelope):
    type: Literal[MessageType.STATUS] = MessageType.STATUS
    payload: StatusPayload


class LocalLLMCompletionMessage(_Envelope):
    type: Literal[MessageType.LOCAL_LLM_COMPLETION] = MessageType.LOCAL_LLM_COMPLETION
    payload: LocalLLMCompletionPayload


class LocalLLMCompletionChunkMessage(_Envelope):
    type: Literal[MessageType.LOCAL_LLM_COMPLETION_CHUNK] = MessageType.LOCAL_LLM_COMPLETION_CHUNK
    payload: LocalLLMCompletionChunkPayload


class LocalLLMCompletionResponseMessage(_Envelope):
    type: Literal[MessageType.LOCAL_LLM_COMPLETION_RESPONSE] = (
        MessageType.LOCAL_LLM_COMPLETION_RESPONSE
    )
    payload: LocalLLMCompletionResponsePayload


class RequestRecordingConsentMessage(_Envelope):
    type: Literal[MessageType.REQUEST_RECORDING_CONSENT] = MessageType.REQUEST_RECORDING_CONSENT
    payload: RequestRecordingConsentPayload


class RecordingConsentResultMessage(_Envelope):
    type: Literal[MessageType.RECORDING_CONSENT_RESULT] = MessageType.RECORDING_CONSENT_RESULT
    payload: RecordingConsentResultPayload


class StartRecordingMessage(_Envelope):
    type: Literal[MessageType.START_RECORDING] = MessageType.START_RECORDING
    payload: StartRecordingPayload


class RecordingStartedMessage(_Envelope):
    type: Literal[MessageType.RECORDING_STARTED] = MessageType.RECORDING_STARTED
    payload: RecordingStartedPayload


class StopRecordingMessage(_Envelope):
    type: Literal[MessageType.STOP_RECORDING] = MessageType.STOP_RECORDING
    payload: StopRecordingPayload


class RecordingSummaryMessage(_Envelope):
    type: Literal[MessageType.RECORDING_SUMMARY] = MessageType.RECORDING_SUMMARY
    payload: RecordingSummaryPayload


class RecordingFetchMessage(_Envelope):
    type: Literal[MessageType.RECORDING_FETCH] = MessageType.RECORDING_FETCH
    payload: RecordingFetchPayload


class RecordingDataMessage(_Envelope):
    type: Literal[MessageType.RECORDING_DATA] = MessageType.RECORDING_DATA
    payload: RecordingDataPayload


class ApplyRecordingReviewMessage(_Envelope):
    type: Literal[MessageType.APPLY_RECORDING_REVIEW] = MessageType.APPLY_RECORDING_REVIEW
    payload: ApplyRecordingReviewPayload


class RecordingReviewAppliedMessage(_Envelope):
    type: Literal[MessageType.RECORDING_REVIEW_APPLIED] = MessageType.RECORDING_REVIEW_APPLIED
    payload: RecordingReviewAppliedPayload


class RecordingStepMessage(_Envelope):
    """Unsolicited co-pilot-mode step frame. Like STATUS, the daemon sends it
    outside the request/response accounting and never expects an ACK."""

    type: Literal[MessageType.RECORDING_STEP] = MessageType.RECORDING_STEP
    payload: RecordingStepPayload


# Discriminated union — used when parsing inbound frames.
Message = Annotated[
    HelloMessage
    | HelloAckMessage
    | DirectoryListRequestMessage
    | DirectoryListResponseMessage
    | AttachSessionMessage
    | SessionAttachedMessage
    | ActivateSessionMessage
    | SessionActivatedMessage
    | RestoreSessionMessage
    | SessionRestoredMessage
    | DetachSessionMessage
    | ExecuteCommandMessage
    | CommandResultMessage
    | FileReadMessage
    | FileContentsMessage
    | FileWriteMessage
    | AckMessage
    | FileStatMessage
    | FileStatResponseMessage
    | FileListMessage
    | FileListResponseMessage
    | FileDeleteMessage
    | FileMoveMessage
    | ComputerToolsRequestMessage
    | ComputerToolsResponseMessage
    | ComputerToolCallMessage
    | ComputerToolResultMessage
    | ErrorMessage
    | PingMessage
    | PongMessage
    | SessionRevokedMessage
    | StatusMessage
    | LocalLLMCompletionMessage
    | LocalLLMCompletionChunkMessage
    | LocalLLMCompletionResponseMessage
    | RequestRecordingConsentMessage
    | RecordingConsentResultMessage
    | StartRecordingMessage
    | RecordingStartedMessage
    | StopRecordingMessage
    | RecordingSummaryMessage
    | RecordingFetchMessage
    | RecordingDataMessage
    | ApplyRecordingReviewMessage
    | RecordingReviewAppliedMessage
    | RecordingStepMessage,
    Field(discriminator="type"),
]

_MESSAGE_ADAPTER: TypeAdapter[Message] = TypeAdapter(Message)


# ── Public helpers ───────────────────────────────────────────────────────────


def new_id() -> str:
    return str(uuid.uuid4())


def now_ts() -> float:
    return time.time()


def parse_message(raw: str | bytes) -> Any:
    """
    Parse a raw WebSocket frame into a typed message.

    Returns one of the *Message classes from this module. Raises
    ValidationError on malformed/unknown frames.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        data = json.loads(
            raw,
            parse_float=_finite_float,
            parse_int=_double_range_int,
            parse_constant=_no_constant,
        )
    except RecursionError:
        raise ValueError("frame nested too deeply to parse") from None
    if _nesting_depth(data) > MAX_FRAME_NESTING:
        raise ValueError(f"frame nested more than {MAX_FRAME_NESTING} levels deep")
    return _MESSAGE_ADAPTER.validate_python(data)


# JSON has no NaN or Infinity, but Python's json reads them, and reads a
# number past a float's range (1e999) as infinity. A frame carrying one is
# refused: the audit record of a computer tool call holding one failed (the
# canonical form has no spelling for them) while the call went on. So is one
# with an integer past a double's range, which Python reads exactly.
def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"frame holds a number that isn't finite: {text}")
    return value


def _double_range_int(text: str) -> int:
    # A whole number takes no float form, so its own parse needs the same
    # bound: a 400-digit integer reached the driver otherwise.
    value = int(text)
    try:
        float(value)
    except OverflowError:
        raise ValueError(f"frame holds a number too large for a double: {text[:32]}...") from None
    return value


def _no_constant(name: str) -> Any:
    raise ValueError(f"frame holds a number that isn't finite: {name}")


# How deeply an inbound frame may nest. No message comes close: a run_steps
# call is about six levels deep. The audit writer, its canonicalizer and the
# argument redaction all recurse, and a computer tool call nested some
# hundreds of levels deep went to the driver while its audit record failed.
MAX_FRAME_NESTING = 64


def _nesting_depth(value: Any) -> int:
    """How many objects and arrays deep `value` goes, counted without
    recursing, and no further than one level past MAX_FRAME_NESTING."""
    deepest = 0
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, dict):
            children: Any = item.values()
        elif isinstance(item, list):
            children = item
        else:
            continue
        deepest = max(deepest, depth)
        if deepest > MAX_FRAME_NESTING:
            break
        # A scalar adds no depth, and a long flat array of them would only
        # grow the stack.
        stack.extend((child, depth + 1) for child in children if isinstance(child, dict | list))
    return deepest


def dump_message(msg: BaseModel) -> str:
    """Serialize a message to a JSON string for sending on the wire."""
    return msg.model_dump_json()


def make_error(
    msg_id: str,
    code: ErrorCode | str,
    message: str,
    fatal: bool = False,
    details: dict[str, Any] | None = None,
) -> ErrorMessage:
    return ErrorMessage(
        id=msg_id,
        ts=now_ts(),
        payload=ErrorPayload(code=code, message=message, fatal=fatal, details=details),
    )


def make_ack(msg_id: str, ok: bool = True) -> AckMessage:
    return AckMessage(id=msg_id, ts=now_ts(), payload=AckPayload(ok=ok))


def make_pong(msg_id: str) -> PongMessage:
    return PongMessage(id=msg_id, ts=now_ts(), payload=PingPongPayload())


__all__ = [
    "VERSION",
    "ProtocolVersionMismatch",
    "negotiate_version",
    "Arch",
    "AckMessage",
    "AckPayload",
    "ActivateSessionMessage",
    "ActivateSessionPayload",
    "ApplyRecordingReviewMessage",
    "ApplyRecordingReviewPayload",
    "AttachSessionMessage",
    "AttachSessionPayload",
    "CommandResultMessage",
    "CommandResultPayload",
    "COMPUTER_TOOL_NAME_PATTERN",
    "MAX_COMPUTER_TOOL_TIMEOUT_SECONDS",
    "ComputerContent",
    "ComputerDriverInfo",
    "ComputerImageContent",
    "ComputerTextContent",
    "ComputerToolCallMessage",
    "ComputerToolCallPayload",
    "ComputerToolResultMessage",
    "ComputerToolResultPayload",
    "ComputerToolSpec",
    "ComputerToolsRequestMessage",
    "ComputerToolsRequestPayload",
    "ComputerToolsResponseMessage",
    "ComputerToolsResponsePayload",
    "DetachSessionMessage",
    "DetachSessionPayload",
    "DirectoryListRequestMessage",
    "DirectoryListRequestPayload",
    "DirectoryListResponseMessage",
    "DirectoryListResponsePayload",
    "DirectoryReferencePayload",
    "Encoding",
    "EnrichmentCoverage",
    "EnrichmentKind",
    "ErrorCode",
    "ErrorMessage",
    "ErrorPayload",
    "ExecuteCommandMessage",
    "ExecuteCommandPayload",
    "FileContentsMessage",
    "FileContentsPayload",
    "FileDeleteMessage",
    "FileDeletePayload",
    "FileEntry",
    "FileFormat",
    "FileListMessage",
    "FileListPayload",
    "FileListResponseMessage",
    "FileListResponsePayload",
    "FileMoveMessage",
    "FileMovePayload",
    "FileReadMessage",
    "FileReadPayload",
    "FileStatMessage",
    "FileStatPayload",
    "FileStatResponseMessage",
    "FileStatResponsePayload",
    "FileWriteMessage",
    "FileWritePayload",
    "HelloAckMessage",
    "HelloAckPayload",
    "HelloMessage",
    "HelloPayload",
    "InterpretationRoute",
    "LocalLLMCompletionChunkMessage",
    "LocalLLMCompletionChunkPayload",
    "LocalLLMCompletionMessage",
    "LocalLLMCompletionPayload",
    "LocalLLMCompletionResponseMessage",
    "LocalLLMCompletionResponsePayload",
    "LocalLLMFinishReason",
    "LocalLLMMessage",
    "LocalLLMTokensUsage",
    "Message",
    "MessageType",
    "PingMessage",
    "PingPongPayload",
    "Platform",
    "PongMessage",
    "RecordingChannel",
    "RecordingConsentResultMessage",
    "RecordingConsentResultPayload",
    "RecordingDataMessage",
    "RecordingDataPayload",
    "RecordingFetchMessage",
    "RecordingFetchPayload",
    "RecordingMode",
    "RequestRecordingConsentMessage",
    "RequestRecordingConsentPayload",
    "RestoreSessionMessage",
    "RestoreSessionPayload",
    "RecordingStartedMessage",
    "RecordingStartedPayload",
    "RecordingReviewAppliedMessage",
    "RecordingReviewAppliedPayload",
    "RecordingStepMessage",
    "RecordingStepPayload",
    "RecordingSummaryMessage",
    "RecordingSummaryPayload",
    "SESSION_REVOKED_REASONS",
    "Selector",
    "SemanticAction",
    "SessionRevokedMessage",
    "SessionRevokedPayload",
    "SessionActivatedMessage",
    "SessionActivatedPayload",
    "SessionAttachedMessage",
    "SessionAttachedPayload",
    "SessionRestoredMessage",
    "SessionRestoredPayload",
    "Shell",
    "StartRecordingMessage",
    "StartRecordingPayload",
    "StatusMessage",
    "StatusPayload",
    "StepEnrichment",
    "StepValue",
    "StopRecordingMessage",
    "StopRecordingPayload",
    "TrajectoryStep",
    "ValidationError",
    "ValueType",
    "WorkflowRecording",
    "dump_message",
    "make_ack",
    "make_error",
    "make_pong",
    "new_id",
    "now_ts",
    "parse_message",
]
