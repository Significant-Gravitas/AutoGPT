"""Shared models and helpers for the Google Chat blocks."""

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from backend.util.exceptions import BlockExecutionError, BlockInputError
from backend.util.settings import Settings

from ._auth import GoogleCredentials

CHAT_SPACES_READONLY_SCOPE = "https://www.googleapis.com/auth/chat.spaces.readonly"
CHAT_SPACES_CREATE_SCOPE = "https://www.googleapis.com/auth/chat.spaces.create"
CHAT_MEMBERSHIPS_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/chat.memberships.readonly"
)
CHAT_MESSAGES_READONLY_SCOPE = "https://www.googleapis.com/auth/chat.messages.readonly"
CHAT_MESSAGES_CREATE_SCOPE = "https://www.googleapis.com/auth/chat.messages.create"
CHAT_READ_STATE_READONLY_SCOPE = (
    "https://www.googleapis.com/auth/chat.users.readstate.readonly"
)

_SPACE_IN_TEXT = re.compile(
    r"(?:spaces/|/room/|/dm/|chat/space/|chat/dm/)([A-Za-z0-9_-]+)"
)
_THREAD_NAME = re.compile(r"spaces/[A-Za-z0-9_-]+/threads/[^/\s?#]+")


class ChatSpaceType(str, Enum):
    ANY = "any"
    SPACE = "space"
    GROUP_CHAT = "group_chat"
    DIRECT_MESSAGE = "direct_message"


API_SPACE_TYPES: dict[ChatSpaceType, str] = {
    ChatSpaceType.SPACE: "SPACE",
    ChatSpaceType.GROUP_CHAT: "GROUP_CHAT",
    ChatSpaceType.DIRECT_MESSAGE: "DIRECT_MESSAGE",
}
_SPACE_TYPE_NAMES: dict[str, str] = {
    api: kind.value for kind, api in API_SPACE_TYPES.items()
}


class ChatSpace(BaseModel):
    """A Google Chat conversation: a named space, a group chat or a direct message."""

    id: str = Field(
        description="The conversation's ID (spaces/...), which the other Google Chat blocks take"
    )
    display_name: str = Field(
        default="",
        description="The space's name (empty for direct messages and most group chats)",
    )
    space_type: str = Field(description="space, group_chat or direct_message")
    url: Optional[str] = Field(
        default=None, description="Link that opens the conversation in Google Chat"
    )
    member_count: Optional[int] = Field(
        default=None,
        description="People who have joined, not counting members of joined Google Groups",
    )
    description: Optional[str] = Field(
        default=None, description="The space's description"
    )
    last_active_time: Optional[str] = Field(
        default=None, description="When the last message was sent (RFC 3339)"
    )


class ChatUser(BaseModel):
    """A person or Chat app, as Google Chat reports them."""

    id: str = Field(description="The user's ID (users/...)")
    display_name: Optional[str] = Field(
        default=None,
        description="Their name. Google leaves it out for people you don't share a space with.",
    )
    email: Optional[str] = Field(
        default=None, description="Their email address, when Google shares it"
    )
    type: str = Field(description="human or app")


class ChatAttachment(BaseModel):
    """A file attached to a Google Chat message."""

    id: str = Field(description="The attachment's ID (spaces/.../attachments/...)")
    file_name: Optional[str] = Field(default=None, description="The file's name")
    mime_type: Optional[str] = Field(default=None, description="The file's MIME type")
    drive_file_id: Optional[str] = Field(
        default=None,
        description="The Google Drive file ID, when the attachment is a Drive file",
    )


class ChatMessage(BaseModel):
    """A Google Chat message."""

    id: str = Field(description="The message's ID (spaces/.../messages/...)")
    space_id: str = Field(description="ID of the conversation it is in (spaces/...)")
    thread_id: Optional[str] = Field(
        default=None,
        description="ID of its thread (spaces/.../threads/...), for replying in the same thread",
    )
    is_thread_reply: bool = Field(
        default=False, description="Whether it is a reply inside a thread"
    )
    text: str = Field(default="", description="The message text")
    sender: Optional[ChatUser] = Field(default=None, description="Who sent it")
    create_time: Optional[str] = Field(
        default=None, description="When it was sent (RFC 3339)"
    )
    last_update_time: Optional[str] = Field(
        default=None, description="When it was last edited, if ever (RFC 3339)"
    )
    attachments: list[ChatAttachment] = Field(
        default_factory=list, description="Files attached to the message"
    )


def build_chat_service(credentials: GoogleCredentials):
    settings = Settings()
    creds = Credentials(
        token=(
            credentials.access_token.get_secret_value()
            if credentials.access_token
            else None
        ),
        refresh_token=(
            credentials.refresh_token.get_secret_value()
            if credentials.refresh_token
            else None
        ),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.secrets.google_client_id,
        client_secret=settings.secrets.google_client_secret,
        scopes=credentials.scopes,
    )
    return build("chat", "v1", credentials=creds, cache_discovery=False)


def to_chat_space(item: dict[str, Any]) -> ChatSpace:
    """Map a Chat API space resource to a ChatSpace."""
    api_type: str = item.get("spaceType") or ""
    counts = item.get("membershipCount")
    return ChatSpace(
        id=item["name"],
        display_name=item.get("displayName", ""),
        space_type=_SPACE_TYPE_NAMES.get(api_type, api_type.lower()),
        url=item.get("spaceUri"),
        # The API leaves zero counts out of membershipCount.
        member_count=counts.get("joinedDirectHumanUserCount", 0) if counts else None,
        description=(item.get("spaceDetails") or {}).get("description"),
        last_active_time=item.get("lastActiveTime"),
    )


def to_chat_message(item: dict[str, Any]) -> ChatMessage:
    """Map a Chat API message resource to a ChatMessage."""
    name = item["name"]
    sender = item.get("sender")
    return ChatMessage(
        id=name,
        space_id="/".join(name.split("/")[:2]),
        thread_id=(item.get("thread") or {}).get("name"),
        is_thread_reply=bool(item.get("threadReply")),
        text=item.get("text", ""),
        sender=_to_chat_user(sender) if sender else None,
        create_time=item.get("createTime"),
        last_update_time=item.get("lastUpdateTime"),
        attachments=[_to_attachment(a) for a in item.get("attachment", [])],
    )


def _to_chat_user(item: dict[str, Any]) -> ChatUser:
    return ChatUser(
        id=item.get("name", ""),
        display_name=item.get("displayName"),
        email=item.get("email"),
        type="app" if item.get("type") == "BOT" else "human",
    )


def _to_attachment(item: dict[str, Any]) -> ChatAttachment:
    return ChatAttachment(
        id=item.get("name", ""),
        file_name=item.get("contentName"),
        mime_type=item.get("contentType"),
        drive_file_id=(item.get("driveDataRef") or {}).get("driveFileId"),
    )


def parse_space_name(value: str) -> str:
    """Accept a space ID, a spaces/... name or a Google Chat link; return spaces/..."""
    value = value.strip()
    if not value:
        return ""
    match = _SPACE_IN_TEXT.search(value)
    return f"spaces/{match.group(1) if match else value}"


def require_space(value: str, block_name: str, block_id: str) -> str:
    space = parse_space_name(value)
    if not space:
        raise BlockInputError(
            message="Give a Google Chat space ID (spaces/...) or a link to the conversation.",
            block_name=block_name,
            block_id=block_id,
        )
    return space


def resolve_thread(value: str, space: str, block_name: str, block_id: str) -> str:
    """Turn a thread ID or a spaces/.../threads/... name into a thread in ``space``."""
    value = value.strip()
    match = _THREAD_NAME.search(value)
    thread = match.group(0) if match else f"{space}/threads/{value}"
    if not thread.startswith(f"{space}/threads/"):
        raise BlockInputError(
            message="That thread is in a different conversation from the space you gave.",
            block_name=block_name,
            block_id=block_id,
        )
    return thread


def to_user_name(value: str) -> str:
    """Accept an email address, a user ID or users/...; return users/..."""
    value = value.strip()
    return value if value.startswith("users/") else f"users/{value}"


def rfc3339(value: datetime) -> str:
    """Format a time for a Chat filter, reading a time without a zone as UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def quote_filter_value(value: str) -> str:
    """Quote a string for a Chat API filter."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def chat_error(
    exc: HttpError, block_name: str, block_id: str, not_found: str | None = None
) -> BlockExecutionError:
    """Turn a Chat API error into a message the user can act on."""
    reason = str(exc.reason)
    lowered = reason.lower()
    if "chat app not found" in lowered:
        message = (
            "Google Chat isn't set up for this AutoGPT instance: the Google Cloud "
            "project behind its Google sign-in needs a configured Chat app. Ask "
            f"your AutoGPT administrator to set one up. (Google said: {reason})"
        )
    elif exc.status_code == 403 and (
        "has not been used" in lowered or "is disabled" in lowered
    ):
        message = (
            "The Google Chat API isn't enabled for this AutoGPT instance. Ask your "
            f"AutoGPT administrator to enable it. (Google said: {reason})"
        )
    elif exc.status_code == 403 and "insufficient" in lowered:
        message = (
            "The connected Google account hasn't granted the Google Chat access "
            "this block needs. Reconnect Google and approve Chat access."
        )
    elif exc.status_code == 404:
        message = not_found or (
            "Google Chat couldn't find that conversation, or the connected Google "
            "account isn't a member of it."
        )
    elif exc.status_code == 400:
        message = f"Google Chat rejected the request: {reason}"
    else:
        message = f"Google Chat API error {exc.status_code}: {reason}"
    return BlockExecutionError(
        message=message, block_name=block_name, block_id=block_id
    )
