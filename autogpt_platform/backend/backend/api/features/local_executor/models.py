"""API models for Local PC executor HTTP and WebSocket routes."""

from typing import Annotated, Literal

from fastapi import Path
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.api.features.local_executor.consent import ComputerUseConsent
from backend.copilot.tools.recording_models import RecordingSummary, WorkflowRecording

SessionID = Annotated[
    str,
    Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9-]+$"),
]
RecordingID = Annotated[
    str,
    Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
]
MachineID = Annotated[
    str,
    Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]


class ExecutorMachine(BaseModel):
    machine_id: str
    connection_id: str
    display_name: str
    platform: str
    arch: str
    shim_version: str
    capabilities: list[str]


class ExecutorsResponse(BaseModel):
    executors: list[ExecutorMachine]


class DirectoryListRequest(BaseModel):
    expected_connection_id: str = Field(min_length=1, max_length=128)
    browse_id: str | None = Field(default=None, min_length=1, max_length=256)
    directory_ref: str | None = Field(default=None, min_length=1, max_length=256)
    cursor: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_reference_pair(self) -> "DirectoryListRequest":
        if (self.browse_id is None) != (self.directory_ref is None):
            raise ValueError(
                "browse_id and directory_ref must both be set or both be null"
            )
        if self.cursor is not None and self.browse_id is None:
            raise ValueError("cursor requires browse_id and directory_ref")
        return self


class DirectoryEntry(BaseModel):
    directory_ref: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=1024)
    path: str = Field(min_length=1, max_length=32_767)


class DirectoryListResponse(BaseModel):
    connection_id: str
    browse_id: str
    current: DirectoryEntry | None = None
    parent_ref: str | None = None
    entries: list[DirectoryEntry] = Field(max_length=200)
    next_cursor: str | None = Field(default=None, min_length=1, max_length=256)
    truncated: bool = False
    expires_at: float


class SessionBindingPayload(BaseModel):
    session_id: str
    allowed_root: str = Field(min_length=1, max_length=32_767)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: int = Field(ge=1)
    root_grant: str = Field(min_length=1, max_length=131_072)


class ExecutorStatus(BaseModel):
    kind: Literal["shim", "none"]
    computer_use_consent: ComputerUseConsent
    platform: str | None = None
    arch: str | None = None
    allowed_root: str | None = None
    machine_id: str | None = None
    shim_version: str | None = None
    capabilities: list[str] | None = None
    computer_use_features: list[str] | None = None
    computer_use_features_coarse: list[str] | None = None
    recording_channels: list[str] | None = None
    recording_routes: list[str] | None = None


class ComputerUseConsentRequest(BaseModel):
    approved: bool
    expected_machine_id: str | None = Field(default=None, min_length=1, max_length=128)
    expected_features_coarse: (
        list[Annotated[str, Field(min_length=1, max_length=128)]] | None
    ) = Field(default=None, max_length=128)
    expected_features: (
        list[Annotated[str, Field(min_length=1, max_length=128)]] | None
    ) = Field(default=None, max_length=128)

    @field_validator("expected_features_coarse", "expected_features")
    @classmethod
    def normalize_expected_features(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        return sorted(set(values))

    @model_validator(mode="after")
    def require_approval_scope(self) -> "ComputerUseConsentRequest":
        if self.approved and (
            self.expected_machine_id is None
            or self.expected_features_coarse is None
            or self.expected_features is None
        ):
            raise ValueError("Approval requires the disclosed Local PC executor scope")
        return self


class ComputerUseConsentResponse(BaseModel):
    computer_use_consent: ComputerUseConsent


class RecordingStartRequest(BaseModel):
    mode: Literal["copilot", "demonstration"] = "copilot"
    interpretation_route: Literal[
        "extract_then_cloud", "local_vlm", "screenshots_to_cloud"
    ] = "extract_then_cloud"
    channels: list[Literal["floor", "browser", "desktop_ax"]] = Field(
        default_factory=lambda: ["floor"], min_length=1
    )


class RecordingStartResponse(BaseModel):
    recording_id: str


class RecordingStopRequest(BaseModel):
    recording_id: str = Field(min_length=1, max_length=128)


class RecordingStopResponse(BaseModel):
    summary: RecordingSummary
    recording: WorkflowRecording


class RecordingReviewRequest(BaseModel):
    removed_step_seqs: list[Annotated[int, Field(ge=0)]] = Field(default_factory=list)
    redacted_step_seqs: list[Annotated[int, Field(ge=0)]] = Field(default_factory=list)
