"""Pydantic models for the platform bot linking API."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Platform(str, Enum):
    """Supported platform types (mirrors Prisma PlatformType)."""

    DISCORD = "DISCORD"
    TELEGRAM = "TELEGRAM"
    SLACK = "SLACK"
    TEAMS = "TEAMS"
    WHATSAPP = "WHATSAPP"
    GITHUB = "GITHUB"
    LINEAR = "LINEAR"


# ── Request Models ─────────────────────────────────────────────────────


class CreateLinkTokenRequest(BaseModel):
    """
    Request from the bot service to create a linking token for a server.

    Called when no PlatformLink exists for the given server. The bot sends
    the resulting link URL to the user who triggered the interaction — they
    become the server owner when they complete the link.
    """

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


class ResolveRequest(BaseModel):
    """Check whether a platform server is linked to an AutoGPT owner account."""

    platform: Platform
    platform_server_id: str = Field(
        description="Server/guild/group ID to look up",
        min_length=1,
        max_length=255,
    )


class BotChatRequest(BaseModel):
    """
    Request from the bot to send a message on behalf of a server user.

    The backend resolves the AutoGPT owner from platform_server_id internally —
    the bot never handles AutoGPT user IDs directly.
    """

    platform: Platform
    platform_server_id: str = Field(
        description="Server/guild/group ID (used to resolve the owner)",
        min_length=1,
        max_length=255,
    )
    platform_user_id: str = Field(
        description="Platform user ID of the person who sent the message (for per-user session keying)",
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


class ResolveResponse(BaseModel):
    linked: bool


class PlatformLinkInfo(BaseModel):
    id: str
    platform: str
    platform_server_id: str
    owner_platform_user_id: str
    server_name: str | None
    linked_at: datetime


class ConfirmLinkResponse(BaseModel):
    success: bool
    platform: str
    platform_server_id: str
    server_name: str | None


class DeleteLinkResponse(BaseModel):
    success: bool


class BotChatSessionResponse(BaseModel):
    """Returned when creating a new session via the bot proxy."""

    session_id: str
