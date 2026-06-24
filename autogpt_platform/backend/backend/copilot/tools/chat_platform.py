"""Tools for the copilot to post to a linked chat platform on the user's behalf.

``post_to_chat_platform`` lets AutoPilot send a standalone message or open a new
thread in a channel of a chat platform (Discord today; Slack/Telegram as their
adapters land) the user has linked. The headline use case is *scheduled*
output: "every Monday post an update in #standup" — AutoPilot schedules a
follow-up turn (via ``schedule_followup``) whose message instructs it to post,
and at fire time it calls this tool.

A single ``platform`` enum keeps the tool surface flat as platforms are added,
instead of one ``post_to_<platform>`` tool per platform. Delivery and
authorization live in the bot bridge (``CoPilotChatBridge``): the tool only
forwards over RPC, and a post is only allowed into channels of a server the
calling user has linked — enforced bridge-side.

``list_chat_platform_channels`` backs channel-name disambiguation and the
"pick a channel" flow.
"""

import logging
from functools import lru_cache
from typing import Any

from backend.copilot.model import ChatSession
from backend.platform_linking.models import Platform
from backend.util.clients import get_copilot_chat_bridge_client
from backend.util.settings import Settings

from .base import BaseTool
from .models import (
    ChatPlatformChannelListResponse,
    ChatPlatformChannelSummary,
    ChatPlatformPostedResponse,
    ErrorResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

# Chat platforms with a wired bridge adapter. Add a value here (and its bot
# token check in ``_any_chat_platform_configured``) when a new adapter ships —
# the tool surface stays the same.
SUPPORTED_PLATFORMS: tuple[str, ...] = ("discord",)

# Maps the bridge's stable DeliveryResult error codes to user-facing text the
# model can relay or act on. Anything unmapped falls back to the raw code.
_ERROR_MESSAGES: dict[str, str] = {
    "no_linked_servers": (
        "No server is linked to this AutoGPT account on that platform yet. "
        "Link one via the bot's setup command before posting."
    ),
    "channel_not_found": (
        "No channel matching that reference was found in your linked server(s). "
        "Call list_chat_platform_channels to see valid options."
    ),
    "not_authorized": (
        "That channel belongs to a server that isn't linked to your account, "
        "so the bot won't post there."
    ),
    "ambiguous_channel": (
        "More than one channel matches that name across your linked servers. "
        "Use the numeric channel ID, or call list_chat_platform_channels."
    ),
    "send_failed": (
        "The platform rejected the message — the bot likely lacks permission "
        "to post in that channel."
    ),
    "thread_failed": (
        "Couldn't create the thread — the bot likely lacks permission to "
        "create public threads in that channel."
    ),
}


@lru_cache(maxsize=1)
def _any_chat_platform_configured() -> bool:
    """Whether any chat-platform bot is configured, gating tool availability.

    Cached because ``get_available_tools`` reads ``is_available`` for every tool
    on every request, and constructing ``Settings()`` re-parses the env each
    time. The bot token is fixed at deploy time, so a one-time read is safe.

    Assumes the copilot/executor process shares the bot token env with the
    bridge pod (true for ``poetry run app`` and the standard deployment).
    """
    return bool(Settings().secrets.autopilot_bot_discord_token)


def _error_message(code: str | None) -> str:
    if not code:
        return "The post could not be completed."
    return _ERROR_MESSAGES.get(code, f"The post failed ({code}).")


def _platform_param() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": list(SUPPORTED_PLATFORMS),
        "description": "Chat platform to post to; defaults to 'discord'.",
    }


def _resolve_platform(value: str | None) -> tuple[Platform | None, str]:
    """Map a platform string to the Platform enum; default to the sole one."""
    name = (value or SUPPORTED_PLATFORMS[0]).lower()
    if name not in SUPPORTED_PLATFORMS:
        return None, name
    return Platform(name.upper()), name


class PostToChatPlatformTool(BaseTool):
    """Post a message or open a thread in a linked chat-platform channel."""

    @property
    def name(self) -> str:
        return "post_to_chat_platform"

    @property
    def description(self) -> str:
        return (
            "Post to a linked chat platform (e.g. Discord) for the user. "
            "mode='message' sends a message; mode='thread' opens a new thread "
            "(needs thread_name). 'channel' is a name (#standup) or numeric ID. "
            "Pair with schedule_followup for recurring posts. If the channel "
            "can't be resolved, call list_chat_platform_channels first."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return _any_chat_platform_configured()

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "platform": _platform_param(),
                "channel": {
                    "type": "string",
                    "description": "Channel name (#standup) or numeric ID.",
                },
                "content": {
                    "type": "string",
                    "description": "Message body to post.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["message", "thread"],
                    "description": "'message' or 'thread'. Defaults to 'message'.",
                },
                "thread_name": {
                    "type": "string",
                    "description": "Thread title; required when mode='thread'.",
                },
            },
            "required": ["channel", "content"],
        }

    @staticmethod
    def _validate_params(session_id: str | None, **kwargs) -> ErrorResponse | None:
        """Return an ``ErrorResponse`` for the first invalid param, else None."""
        _platform, platform_name = _resolve_platform(kwargs.get("platform"))
        if _platform is None:
            return ErrorResponse(
                message=f"Unsupported platform '{platform_name}'.",
                error="unsupported_platform",
                session_id=session_id,
            )
        channel: str | None = kwargs.get("channel")
        content: str | None = kwargs.get("content")
        mode: str = kwargs.get("mode") or "message"
        if not channel or not channel.strip():
            return ErrorResponse(
                message="`channel` is required.",
                error="missing_channel",
                session_id=session_id,
            )
        if not content or not content.strip():
            return ErrorResponse(
                message="`content` is required.",
                error="missing_content",
                session_id=session_id,
            )
        if mode not in ("message", "thread"):
            return ErrorResponse(
                message="`mode` must be 'message' or 'thread'.",
                error="invalid_mode",
                session_id=session_id,
            )
        if mode == "thread":
            thread_name: str | None = kwargs.get("thread_name")
            if not thread_name or not thread_name.strip():
                return ErrorResponse(
                    message="`thread_name` is required when mode='thread'.",
                    error="missing_thread_name",
                    session_id=session_id,
                )
        return None

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )
        invalid = self._validate_params(session_id, **kwargs)
        if invalid is not None:
            return invalid

        platform, platform_name = _resolve_platform(kwargs.get("platform"))
        if platform is None:  # already validated; narrows the type
            return ErrorResponse(
                message=f"Unsupported platform '{platform_name}'.",
                error="unsupported_platform",
                session_id=session_id,
            )
        channel: str = kwargs["channel"]
        content: str = kwargs["content"]
        mode: str = kwargs.get("mode") or "message"

        client = get_copilot_chat_bridge_client()
        if mode == "thread":
            result = await client.create_thread_in_channel(
                platform=platform,
                user_id=user_id,
                channel=channel,
                thread_name=kwargs["thread_name"],
                content=content,
            )
        else:
            result = await client.send_message_to_channel(
                platform=platform,
                user_id=user_id,
                channel=channel,
                content=content,
            )

        if not result.ok:
            return ErrorResponse(
                message=_error_message(result.error),
                error=result.error or "chat_platform_post_failed",
                session_id=session_id,
            )

        where = "thread" if result.kind == "thread" else "message"
        link_note = f" ({result.url})" if result.url else ""
        return ChatPlatformPostedResponse(
            message=f"Posted {where} to {platform_name}{link_note}.",
            platform=platform_name,
            kind=result.kind,
            channel_id=result.channel_id or channel,
            ref_id=result.ref_id,
            url=result.url,
            session_id=session_id,
        )


class ListChatPlatformChannelsTool(BaseTool):
    """List channels the bot can post to across the user's linked servers."""

    @property
    def name(self) -> str:
        return "list_chat_platform_channels"

    @property
    def description(self) -> str:
        return (
            "List channels the bot can post to on a linked chat platform "
            "(e.g. Discord). Use to resolve a channel name to an ID or pick "
            "one before post_to_chat_platform."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return _any_chat_platform_configured()

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"platform": _platform_param()},
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )
        platform, platform_name = _resolve_platform(kwargs.get("platform"))
        if platform is None:
            return ErrorResponse(
                message=f"Unsupported platform '{platform_name}'.",
                error="unsupported_platform",
                session_id=session_id,
            )

        channels = await get_copilot_chat_bridge_client().list_channels(
            platform=platform,
            user_id=user_id,
        )
        summaries = [
            ChatPlatformChannelSummary(
                id=c.id,
                name=c.name,
                server_id=c.server_id,
                server_name=c.server_name,
            )
            for c in channels
        ]
        if summaries:
            message = (
                f"Found {len(summaries)} channel(s) you can post to on {platform_name}."
            )
        else:
            message = (
                f"No postable {platform_name} channels found. Link a server via "
                "the bot's setup command, or check the bot's channel permissions."
            )
        return ChatPlatformChannelListResponse(
            message=message,
            platform=platform_name,
            channels=summaries,
            count=len(summaries),
            session_id=session_id,
        )
