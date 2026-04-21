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


# ── Response Models ────────────────────────────────────────────────────


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


class ChatTurnHandle(BaseModel):
    """Subscribe keys for a pending copilot turn."""

    session_id: str
    turn_id: str
    user_id: str
    subscribe_from: str = "0-0"
