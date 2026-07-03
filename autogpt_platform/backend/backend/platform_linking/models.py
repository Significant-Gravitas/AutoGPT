"""Pydantic models for platform_linking requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Platform(str, Enum):
    """Mirror of the Prisma PlatformType enum."""

    DISCORD = "DISCORD"
    TELEGRAM = "TELEGRAM"
    SLACK = "SLACK"
    TEAMS = "TEAMS"
    WHATSAPP = "WHATSAPP"
    GITHUB = "GITHUB"
    LINEAR = "LINEAR"


class LinkType(str, Enum):
    SERVER = "SERVER"
    USER = "USER"


# ── Request Models ─────────────────────────────────────────────────────


class CreateLinkTokenRequest(BaseModel):
    platform: Platform = Field(description="Platform name")
    platform_server_id: str = Field(
        description="Server/guild/group ID on the platform",
        min_length=1,
        max_length=255,
    )
    platform_user_id: str = Field(
        description="Platform user ID of the person claiming ownership",
        min_length=1,
        max_length=255,
    )
    platform_username: str | None = Field(
        default=None,
        description="Display name of the person claiming ownership",
        max_length=255,
    )
    server_name: str | None = Field(
        default=None,
        description="Display name of the server/group",
        max_length=255,
    )
    channel_id: str | None = Field(
        default=None,
        description="Channel ID so the bot can send a confirmation message",
        max_length=255,
    )


class CreateUserLinkTokenRequest(BaseModel):
    platform: Platform
    platform_user_id: str = Field(
        description="Platform user ID of the person linking their DMs",
        min_length=1,
        max_length=255,
    )
    platform_username: str | None = Field(
        default=None,
        description="Their display name (best-effort for audit)",
        max_length=255,
    )


class ResolveServerRequest(BaseModel):
    platform: Platform
    platform_server_id: str = Field(
        description="Server/guild/group ID to look up",
        min_length=1,
        max_length=255,
    )


class ResolveUserRequest(BaseModel):
    platform: Platform
    platform_user_id: str = Field(
        description="Platform user ID to look up",
        min_length=1,
        max_length=255,
    )


class BotChatRequest(BaseModel):
    """Bot message request. If ``platform_server_id`` is set, the turn is
    billed to that server's owner; otherwise billed to ``platform_user_id``
    (DM context)."""

    platform: Platform
    platform_server_id: str | None = Field(
        default=None,
        description="Server/guild/group ID — null for DM context",
        min_length=1,
        max_length=255,
    )
    platform_user_id: str = Field(
        description="Platform user ID of the person who sent the message",
        min_length=1,
        max_length=255,
    )
    message: str = Field(
        description="The user's message", min_length=1, max_length=32000
    )
    session_id: str | None = Field(
        default=None,
        description="Existing CoPilot session ID. If omitted, a new session is created.",
    )
    file_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Workspace file IDs (already uploaded) to attach to the turn.",
    )


class WorkspaceUploadRequest(BaseModel):
    """Upload a user-attached file into the conversation owner's workspace.

    Resolved to the same owner a chat turn bills to (server owner, or the
    DM-linked user). The file runs through the same AV scan + storage as the
    web upload endpoint."""

    platform: Platform
    platform_server_id: str | None = Field(default=None, min_length=1, max_length=255)
    platform_user_id: str = Field(min_length=1, max_length=255)
    filename: str = Field(min_length=1, max_length=512)
    mime_type: str = Field(min_length=1, max_length=255)
    content: bytes
    # Write into this session's folder (/sessions/<id>/) so AutoPilot reads the
    # file during the turn, same as a web upload. Resolved by the caller before
    # uploading (see ``ensure_chat_session``).
    session_id: str | None = Field(default=None, min_length=1, max_length=255)


# ── Response Models ────────────────────────────────────────────────────


class WorkspaceUploadResult(BaseModel):
    """Outcome of one file upload. ``file_id`` is set on success; otherwise
    ``error`` carries a stable code the bot maps to user-facing text."""

    filename: str
    file_id: str | None = None
    error: str | None = None


class LinkTokenResponse(BaseModel):
    token: str
    expires_at: datetime
    link_url: str


class LinkTokenStatusResponse(BaseModel):
    status: Literal["pending", "linked", "expired"]


class LinkTokenInfoResponse(BaseModel):
    platform: str
    link_type: LinkType
    server_name: str | None = None


class ResolveResponse(BaseModel):
    linked: bool


class PlatformLinkInfo(BaseModel):
    id: str
    platform: str
    platform_server_id: str
    owner_platform_user_id: str
    server_name: str | None
    linked_at: datetime


class PlatformUserLinkInfo(BaseModel):
    id: str
    platform: str
    platform_user_id: str
    platform_username: str | None
    linked_at: datetime


class BotPlatformInfo(BaseModel):
    """A bot platform enabled on this deployment plus the caller's links to it.

    Platforms whose adapter isn't configured (missing token/credentials) are
    omitted from the response entirely — the Bots settings page hides them.
    """

    platform: str
    display_name: str
    icon: str
    add_bot_url: str | None = None
    dm_link: PlatformUserLinkInfo | None = None
    server_links: list[PlatformLinkInfo] = Field(default_factory=list)


class ConfirmLinkResponse(BaseModel):
    success: bool
    link_type: LinkType = LinkType.SERVER
    platform: str
    platform_server_id: str
    server_name: str | None


class ConfirmUserLinkResponse(BaseModel):
    success: bool
    link_type: LinkType = LinkType.USER
    platform: str
    platform_user_id: str


class DeleteLinkResponse(BaseModel):
    success: bool


class TurnDenial(BaseModel):
    """Why a copilot turn was refused before it ran, with the message (and
    optional CTA button) the bot should show the user.

    Lets ``start_chat_turn`` enforce the same subscription paywall + usage
    rate limits the web UI applies, since the bot bypasses both by enqueuing
    directly into the executor.
    """

    reason: Literal["paywalled", "rate_limited", "unavailable"]
    message: str
    button_label: str | None = None
    button_url: str | None = None


class EnsureSessionResult(BaseModel):
    """Result of resolving a session ahead of attachment uploads.

    When ``denial`` is set the turn gate refused the user before anything was
    uploaded — the bot renders the denial and skips both the upload and the
    turn, so a capped/paywalled user's files are never scanned or stored.
    """

    session_id: str | None = None
    denial: TurnDenial | None = None


class ChatTurnHandle(BaseModel):
    """Subscribe keys for a pending copilot turn.

    When ``denial`` is set the turn was refused (paywall / rate limit) and
    nothing was enqueued — the caller surfaces the denial instead of
    subscribing to a stream.
    """

    session_id: str
    turn_id: str
    user_id: str
    subscribe_from: str = "0-0"
    denial: TurnDenial | None = None


class ChatSessionSummary(BaseModel):
    """Minimal chat-session fields safe to send to a bot client."""

    session_id: str
    title: str | None = None
    updated_at: datetime


class ListUserChatsResponse(BaseModel):
    sessions: list[ChatSessionSummary]
    total: int


class WorkspaceArtifact(BaseModel):
    """A user-owned workspace file resolved to bytes for bot attachment.

    Returned by ``fetch_workspace_artifact`` when the file exists, belongs to
    the session's owning user, and fits under the requested ``max_bytes``.
    The bot uses ``content`` to attach the file inline; over-the-limit or
    missing files come back as ``None`` and trigger the link-to-chat fallback.
    """

    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    content: bytes


class BotEventInput(BaseModel):
    """A single bot usage event. Never carries message content."""

    platform: Platform
    event_type: str
    server_id: str | None = None
    channel_type: str | None = None
    command_name: str | None = None
    error_kind: str | None = None
    char_count: int | None = None
    duration_ms: int | None = None


class BotGuildInput(BaseModel):
    """Presence of the bot in a single server."""

    platform: Platform
    server_id: str
    name: str | None = None
