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
    """Request from the bot service to create a linking token."""

    platform: Platform = Field(description="Platform name")
    platform_user_id: str = Field(
        description="The user's ID on the platform",
        min_length=1,
        max_length=255,
    )
    platform_username: str | None = Field(
        default=None,
        description="Display name (best effort)",
        max_length=255,
    )
    channel_id: str | None = Field(
        default=None,
        description="Channel ID for sending confirmation back",
        max_length=255,
    )


class ResolveRequest(BaseModel):
    """Resolve a platform identity to an AutoGPT user."""

    platform: Platform
    platform_user_id: str = Field(min_length=1, max_length=255)


class BotChatRequest(BaseModel):
    """Request from the bot to chat as a linked user."""

    user_id: str = Field(description="The linked AutoGPT user ID")
    message: str = Field(
        description="The user's message", min_length=1, max_length=32000
    )
    session_id: str | None = Field(
        default=None,
        description="Existing chat session ID. If omitted, a new session is created.",
    )


# ── Response Models ────────────────────────────────────────────────────


class LinkTokenResponse(BaseModel):
    token: str
    expires_at: datetime
    link_url: str


class LinkTokenStatusResponse(BaseModel):
    status: Literal["pending", "linked", "expired"]
    user_id: str | None = None


class ResolveResponse(BaseModel):
    linked: bool
    user_id: str | None = None


class PlatformLinkInfo(BaseModel):
    id: str
    platform: str
    platform_user_id: str
    platform_username: str | None
    linked_at: datetime


class ConfirmLinkResponse(BaseModel):
    success: bool
    platform: str
    platform_user_id: str
    platform_username: str | None


class DeleteLinkResponse(BaseModel):
    success: bool


class BotChatSessionResponse(BaseModel):
    """Returned when creating a new session via the bot proxy."""

    session_id: str
