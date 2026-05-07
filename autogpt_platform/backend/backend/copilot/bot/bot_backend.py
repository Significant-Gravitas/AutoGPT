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
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

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
    CreateLinkTokenRequest,
    CreateUserLinkTokenRequest,
    Platform,
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

__all__ = [
    "BotBackend",
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


SetupRequiredCallback = Callable[
    [str, dict[str, Any], str | None],
    Awaitable[None],
]


class BotBackend:
    """Bot-side linking + chat operations, routed over cluster-internal RPC."""

    def __init__(self):
        self._client = get_platform_linking_manager_client()

    async def close(self) -> None:
        # The client's lifecycle is owned by the thread-cached factory; nothing
        # to close here. Kept for API compatibility with older bot code.
        pass

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

    async def stream_chat(
        self,
        platform: str,
        platform_user_id: str,
        message: str,
        session_id: Optional[str] = None,
        platform_server_id: Optional[str] = None,
        on_session_id: Optional[Callable[[str], Awaitable[None]]] = None,
        on_setup_required: SetupRequiredCallback | None = None,
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
            yield "\n[Error: failed to subscribe to response stream]"
            return

        setup_notified = False

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
                    yield "\n[Error: response timed out]"
                    return
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
                elif isinstance(chunk, StreamFinish):
                    return
                elif isinstance(chunk, StreamError):
                    logger.error("Stream error from backend: %s", chunk.errorText)
                    yield f"\n[Error: {chunk.errorText}]"
                    return
                # Other StreamX types (StreamStart, StreamTextStart, tool events,
                # etc.) are emitted by the executor for the frontend UI and
                # aren't useful for the plain-text bot transcript.
        finally:
            await stream_registry.unsubscribe_from_session(
                session_id=handle.session_id,
                subscriber_queue=queue,
            )


def _extract_setup_requirements(output: str | dict[str, Any]) -> dict[str, Any] | None:
    """Return setup-requirements payloads from structured tool output."""
    if isinstance(output, str):
        try:
            parsed: Any = json.loads(output)
        except json.JSONDecodeError:
            return None
    else:
        parsed = output

    if not isinstance(parsed, dict):
        return None
    if parsed.get("type") != "setup_requirements":
        return None
    return parsed
