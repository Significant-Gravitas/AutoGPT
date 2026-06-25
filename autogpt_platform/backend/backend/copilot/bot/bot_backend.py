"""Bot-side facade over `PlatformLinkingManagerClient` + `stream_registry`.

The `BotBackend` class is the bot's single entry point into the AutoGPT
backend — it wraps the linking RPC client and the copilot stream registry
behind plain string-typed methods. Adapters import this directly so the
discord/telegram/slack code never touches Pyro / Redis Streams plumbing.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

from pydantic import BaseModel

from backend.copilot import stream_registry
from backend.copilot.model import get_chat_session
from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamTextDelta,
    StreamToolOutputAvailable,
)
from backend.platform_linking.models import (
    BotChatRequest,
    BotEventInput,
    BotGuildInput,
    CreateLinkTokenRequest,
    CreateUserLinkTokenRequest,
    Platform,
    WorkspaceArtifact,
)
from backend.util.clients import get_platform_linking_manager_client
from backend.util.exceptions import (
    DuplicateChatMessageError,
    LinkAlreadyExistsError,
    NotFoundError,
)

# How long to wait for a single chunk from the copilot stream before giving
# up. Covers the case where the backend crashes mid-stream and never sends
# ``StreamFinish`` — without this, the bot would hang forever on ``queue.get()``.
STREAM_CHUNK_TIMEOUT_SECONDS = 120

logger = logging.getLogger(__name__)


class BotStreamError(Exception):
    """A copilot stream couldn't produce a successful reply.

    Carries a bounded ``error_kind`` so the handler can attribute analytics
    accurately instead of guessing from the inline text.
    """

    def __init__(self, error_kind: str, message: str):
        super().__init__(message)
        self.error_kind = error_kind


__all__ = [
    "BotBackend",
    "BotStreamError",
    "ChatSummary",
    "DuplicateChatMessageError",
    "LinkAlreadyExistsError",
    "LinkTokenResult",
    "NotFoundError",
    "ResolveResult",
]


@dataclass
class ResolveResult:
    linked: bool


@dataclass
class LinkTokenResult:
    token: str
    link_url: str
    expires_at: str


class ChatSummary(BaseModel):
    session_id: str
    title: str | None
    updated_at: datetime


SetupRequiredCallback = Callable[
    [str, dict[str, Any], str | None],
    Awaitable[None],
]

# Fired when a setup_requirements payload arrives corrupted and is dropped, so
# the adapter can surface a user-facing notice instead of leaving the user
# staring at a sign-in prompt that never renders. Args: (session_id, tool_name).
SetupDroppedCallback = Callable[
    [str, str | None],
    Awaitable[None],
]


class BotBackend:
    """Bot-side linking + chat operations, routed over cluster-internal RPC."""

    def __init__(self):
        self._client = get_platform_linking_manager_client()
        self._analytics_tasks: set[asyncio.Task] = set()

    async def close(self) -> None:
        # The client's lifecycle is owned by the thread-cached factory; nothing
        # to close here. Kept for API compatibility with older bot code.
        pass

    # ── Analytics (fire-and-forget) ──────────────────────────────────────
    # Usage telemetry must never block or break a user's reply, so every
    # write is scheduled as a background task that swallows its own errors.
    # No message content is ever sent — only counts, enums and metrics.

    def _fire_and_forget(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._analytics_tasks.add(task)
        task.add_done_callback(self._on_analytics_done)

    def _on_analytics_done(self, task: asyncio.Task) -> None:
        self._analytics_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("Bot analytics write failed: %s", exc)

    # NOTE: each track method defers Platform() and Pydantic construction into
    # the background coroutine — that way a bad platform string or validation
    # error fails the analytics task (which is swallowed) instead of leaking
    # back into the caller's reply path.

    def track_event(
        self,
        *,
        platform: str,
        event_type: str,
        server_id: str | None = None,
        channel_type: str | None = None,
        command_name: str | None = None,
        error_kind: str | None = None,
        char_count: int | None = None,
        duration_ms: int | None = None,
    ) -> None:
        async def _send() -> None:
            await self._client.record_bot_event(
                event=BotEventInput(
                    platform=Platform(platform.upper()),
                    event_type=event_type,
                    server_id=server_id,
                    channel_type=channel_type,
                    command_name=command_name,
                    error_kind=error_kind,
                    char_count=char_count,
                    duration_ms=duration_ms,
                )
            )

        self._fire_and_forget(_send())

    def track_guild_joined(
        self, platform: str, server_id: str, name: str | None = None
    ) -> None:
        async def _send() -> None:
            await self._client.record_guild_joined(
                guild=BotGuildInput(
                    platform=Platform(platform.upper()),
                    server_id=server_id,
                    name=name,
                )
            )

        self._fire_and_forget(_send())

    def track_guild_left(self, platform: str, server_id: str) -> None:
        async def _send() -> None:
            await self._client.mark_guild_left(
                platform=Platform(platform.upper()),
                server_id=server_id,
            )

        self._fire_and_forget(_send())

    def sync_guilds(self, platform: str, guilds: list[tuple[str, str | None]]) -> None:
        async def _send() -> None:
            platform_enum = Platform(platform.upper())
            await self._client.sync_guild_presence(
                platform=platform_enum,
                guilds=[
                    BotGuildInput(
                        platform=platform_enum,
                        server_id=server_id,
                        name=name,
                    )
                    for server_id, name in guilds
                ],
            )

        self._fire_and_forget(_send())

    async def resolve_server(
        self, platform: str, platform_server_id: str
    ) -> ResolveResult:
        resp = await self._client.resolve_server_link(
            platform=Platform(platform.upper()),
            platform_server_id=platform_server_id,
        )
        return ResolveResult(linked=resp.linked)

    async def resolve_user(self, platform: str, platform_user_id: str) -> ResolveResult:
        resp = await self._client.resolve_user_link(
            platform=Platform(platform.upper()),
            platform_user_id=platform_user_id,
        )
        return ResolveResult(linked=resp.linked)

    async def list_linked_server_ids(self, platform: str, user_id: str) -> list[str]:
        """Return the platform server (guild) IDs ``user_id`` has linked.

        Backs the proactive-output authorization check: a scheduled/autopilot
        post is only allowed into a channel whose server appears in this list.
        """
        return await self._client.list_user_server_ids(
            platform=Platform(platform.upper()),
            user_id=user_id,
        )

    async def refresh_server_name(
        self, platform: str, platform_server_id: str, server_name: str
    ) -> None:
        """Push the bot's authoritative display name for a server into the DB.

        Called by adapters whenever they learn or re-learn a server's name —
        e.g. Discord's on_ready and on_guild_join. The Bots settings page
        renders whatever is in the DB, so this keeps stale or missing names
        in sync with what the bot is actually connected to. Best-effort:
        failures are logged on the manager side, never raised.
        """
        if not server_name:
            return
        await self._client.refresh_server_link_name(
            platform=Platform(platform.upper()),
            platform_server_id=platform_server_id,
            server_name=server_name,
        )

    async def create_link_token(
        self,
        platform: str,
        platform_server_id: str,
        platform_user_id: str,
        platform_username: str,
        server_name: str,
        channel_id: str = "",
    ) -> LinkTokenResult:
        resp = await self._client.create_server_link_token(
            request=CreateLinkTokenRequest(
                platform=Platform(platform.upper()),
                platform_server_id=platform_server_id,
                platform_user_id=platform_user_id,
                platform_username=platform_username or None,
                server_name=server_name or None,
                channel_id=channel_id or None,
            )
        )
        return LinkTokenResult(
            token=resp.token,
            link_url=resp.link_url,
            expires_at=resp.expires_at.isoformat(),
        )

    async def create_user_link_token(
        self,
        platform: str,
        platform_user_id: str,
        platform_username: str,
    ) -> LinkTokenResult:
        resp = await self._client.create_user_link_token(
            request=CreateUserLinkTokenRequest(
                platform=Platform(platform.upper()),
                platform_user_id=platform_user_id,
                platform_username=platform_username or None,
            )
        )
        return LinkTokenResult(
            token=resp.token,
            link_url=resp.link_url,
            expires_at=resp.expires_at.isoformat(),
        )

    async def get_session_title(self, session_id: str) -> str | None:
        session = await get_chat_session(session_id)
        return session.title if session else None

    async def list_user_chats(
        self, platform: str, platform_user_id: str, limit: int = 25
    ) -> list[ChatSummary]:
        resp = await self._client.list_user_chats(
            platform=Platform(platform.upper()),
            platform_user_id=platform_user_id,
            limit=limit,
        )
        return [
            ChatSummary(
                session_id=s.session_id,
                title=s.title,
                updated_at=s.updated_at,
            )
            for s in resp.sessions
        ]

    async def fetch_workspace_artifact(
        self, session_id: str, file_id: str, max_bytes: int
    ) -> WorkspaceArtifact | None:
        """Resolve a ``workspace://`` URI from the chat stream to bytes,
        scoped to the session's owning user. Returns ``None`` when the file
        is too large, missing, or doesn't belong to the session — in which
        case the caller drops a link-to-chat fallback button."""
        return await self._client.fetch_workspace_artifact(
            session_id=session_id, file_id=file_id, max_bytes=max_bytes
        )

    async def stream_chat(
        self,
        platform: str,
        platform_user_id: str,
        message: str,
        session_id: Optional[str] = None,
        platform_server_id: Optional[str] = None,
        on_session_id: Optional[Callable[[str], Awaitable[None]]] = None,
        on_setup_required: SetupRequiredCallback | None = None,
        on_setup_dropped: SetupDroppedCallback | None = None,
    ) -> AsyncGenerator[str, None]:
        """Start a copilot turn and yield text deltas from the stream.

        Raises :class:`DuplicateChatMessageError` if the same message is
        already in flight for this session.
        """
        handle = await self._client.start_chat_turn(
            request=BotChatRequest(
                platform=Platform(platform.upper()),
                platform_user_id=platform_user_id,
                message=message,
                session_id=session_id,
                platform_server_id=platform_server_id,
            )
        )
        if on_session_id:
            await on_session_id(handle.session_id)

        queue = await stream_registry.subscribe_to_session(
            session_id=handle.session_id,
            user_id=handle.user_id,
            last_message_id=handle.subscribe_from,
        )
        if queue is None:
            raise BotStreamError(
                "subscribe_failed",
                "failed to subscribe to response stream",
            )

        setup_notified = False
        setup_drop_notified = False

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        queue.get(), timeout=STREAM_CHUNK_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Stream idle timeout after %ss for session %s",
                        STREAM_CHUNK_TIMEOUT_SECONDS,
                        handle.session_id,
                    )
                    raise BotStreamError(
                        "stream_timeout",
                        "response timed out",
                    )
                if isinstance(chunk, StreamTextDelta):
                    if chunk.delta:
                        yield chunk.delta
                elif isinstance(chunk, StreamToolOutputAvailable):
                    setup_output = _extract_setup_requirements(chunk.output)
                    if setup_output and on_setup_required and not setup_notified:
                        setup_notified = True
                        await on_setup_required(
                            handle.session_id,
                            setup_output,
                            chunk.toolName,
                        )
                    elif (
                        setup_output is None
                        and not setup_notified
                        and not setup_drop_notified
                        and on_setup_dropped
                        and _is_corrupted_setup_requirements(chunk.output)
                    ):
                        # The link was dropped (corrupted payload). Tell the
                        # user once so they aren't left waiting on a sign-in
                        # prompt that will never arrive.
                        setup_drop_notified = True
                        await on_setup_dropped(handle.session_id, chunk.toolName)
                elif isinstance(chunk, StreamFinish):
                    return
                elif isinstance(chunk, StreamError):
                    logger.error("Stream error from backend: %s", chunk.errorText)
                    raise BotStreamError(
                        "backend_stream_error",
                        chunk.errorText,
                    )
                # Other StreamX types (StreamStart, StreamTextStart, tool events,
                # etc.) are emitted by the executor for the frontend UI and
                # aren't useful for the plain-text bot transcript.
        finally:
            await stream_registry.unsubscribe_from_session(
                session_id=handle.session_id,
                subscriber_queue=queue,
            )


def _is_corrupted_setup_requirements(output: str | dict[str, Any]) -> bool:
    """True when *output* names setup_requirements but doesn't parse as JSON.

    This is the truncation/corruption signature: the tool intended to surface a
    sign-in card, but the payload arrived mangled, so the user gets nothing
    unless the caller surfaces a notice.
    """
    if not isinstance(output, str):
        return False
    if '"setup_requirements"' not in output:
        return False
    try:
        json.loads(output)
    except json.JSONDecodeError:
        return True
    return False


def _extract_setup_requirements(output: str | dict[str, Any]) -> dict[str, Any] | None:
    """Return setup-requirements payloads from structured tool output."""
    if isinstance(output, str):
        try:
            parsed: Any = json.loads(output)
        except json.JSONDecodeError:
            if '"setup_requirements"' in output:
                # A payload that mentions setup_requirements but doesn't parse
                # is almost certainly a truncated/corrupted tool output — the
                # user never gets their sign-in link if we drop it silently.
                logger.warning(
                    "Dropping unparseable setup_requirements tool output "
                    "(%d chars) — sign-in link will not be sent",
                    len(output),
                )
            return None
    else:
        parsed = output

    if not isinstance(parsed, dict):
        return None
    if parsed.get("type") != "setup_requirements":
        return None
    return parsed
