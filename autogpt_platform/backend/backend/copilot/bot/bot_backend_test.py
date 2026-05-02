"""Tests for the bot's thin facade over PlatformLinkingManagerClient."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.response_model import StreamError, StreamFinish, StreamTextDelta
from backend.platform_linking.models import (
    ChatTurnHandle,
    LinkTokenResponse,
    ResolveResponse,
)
from backend.util.exceptions import (
    DuplicateChatMessageError,
    LinkAlreadyExistsError,
    NotFoundError,
)

from .bot_backend import BotBackend


@pytest.fixture
def api() -> BotBackend:
    with patch("backend.copilot.bot.bot_backend.get_platform_linking_manager_client"):
        instance = BotBackend()
    # Swap in a MagicMock whose RPC methods are AsyncMocks — simpler than
    # patching each call site.
    instance._client = MagicMock()
    return instance


class TestResolve:
    @pytest.mark.asyncio
    async def test_resolve_server(self, api: BotBackend):
        api._client.resolve_server_link = AsyncMock(
            return_value=ResolveResponse(linked=True)
        )
        result = await api.resolve_server("discord", "g1")
        assert result.linked is True
        api._client.resolve_server_link.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resolve_user(self, api: BotBackend):
        api._client.resolve_user_link = AsyncMock(
            return_value=ResolveResponse(linked=False)
        )
        result = await api.resolve_user("discord", "u1")
        assert result.linked is False


class TestCreateLinkTokens:
    @pytest.mark.asyncio
    async def test_create_server_link_token(self, api: BotBackend):
        api._client.create_server_link_token = AsyncMock(
            return_value=LinkTokenResponse(
                token="abc",
                expires_at=datetime.now(timezone.utc),
                link_url="https://example.com/link/abc",
            )
        )
        result = await api.create_link_token(
            platform="discord",
            platform_server_id="g1",
            platform_user_id="u1",
            platform_username="Bently",
            server_name="Test",
        )
        assert result.token == "abc"
        assert result.link_url.endswith("/link/abc")

    @pytest.mark.asyncio
    async def test_create_server_link_token_propagates_already_exists(
        self, api: BotBackend
    ):
        api._client.create_server_link_token = AsyncMock(
            side_effect=LinkAlreadyExistsError("already linked")
        )
        with pytest.raises(LinkAlreadyExistsError):
            await api.create_link_token(
                platform="discord",
                platform_server_id="g1",
                platform_user_id="u1",
                platform_username="",
                server_name="",
            )

    @pytest.mark.asyncio
    async def test_create_user_link_token(self, api: BotBackend):
        api._client.create_user_link_token = AsyncMock(
            return_value=LinkTokenResponse(
                token="xyz",
                expires_at=datetime.now(timezone.utc),
                link_url="https://example.com/link/xyz",
            )
        )
        result = await api.create_user_link_token(
            platform="discord", platform_user_id="u1", platform_username="Bently"
        )
        assert result.token == "xyz"


class TestStreamChat:
    @pytest.mark.asyncio
    async def test_yields_text_deltas_and_terminates_on_finish(self, api: BotBackend):
        handle = ChatTurnHandle(session_id="sess", turn_id="turn", user_id="u1")
        api._client.start_chat_turn = AsyncMock(return_value=handle)

        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(StreamTextDelta(id="1", delta="Hello "))
        await queue.put(StreamTextDelta(id="2", delta="world"))
        await queue.put(StreamFinish())

        captured_session_ids: list[str] = []

        async def capture(sid: str) -> None:
            captured_session_ids.append(sid)

        with (
            patch(
                "backend.copilot.bot.bot_backend.stream_registry.subscribe_to_session",
                new=AsyncMock(return_value=queue),
            ),
            patch(
                "backend.copilot.bot.bot_backend.stream_registry.unsubscribe_from_session",
                new=AsyncMock(),
            ),
        ):
            chunks: list[str] = []
            async for chunk in api.stream_chat(
                platform="discord",
                platform_user_id="u1",
                message="hi",
                on_session_id=capture,
            ):
                chunks.append(chunk)

        assert "".join(chunks) == "Hello world"
        assert captured_session_ids == ["sess"]

    @pytest.mark.asyncio
    async def test_surfaces_stream_error(self, api: BotBackend):
        handle = ChatTurnHandle(session_id="sess", turn_id="turn", user_id="u1")
        api._client.start_chat_turn = AsyncMock(return_value=handle)

        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(StreamError(errorText="executor crashed"))

        with (
            patch(
                "backend.copilot.bot.bot_backend.stream_registry.subscribe_to_session",
                new=AsyncMock(return_value=queue),
            ),
            patch(
                "backend.copilot.bot.bot_backend.stream_registry.unsubscribe_from_session",
                new=AsyncMock(),
            ),
        ):
            chunks: list[str] = []
            async for chunk in api.stream_chat(
                platform="discord", platform_user_id="u1", message="hi"
            ):
                chunks.append(chunk)

        assert any("executor crashed" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_duplicate_message_propagates(self, api: BotBackend):
        api._client.start_chat_turn = AsyncMock(
            side_effect=DuplicateChatMessageError("in flight")
        )

        with pytest.raises(DuplicateChatMessageError):
            async for _ in api.stream_chat(
                platform="discord", platform_user_id="u1", message="hi"
            ):
                pass

    @pytest.mark.asyncio
    async def test_session_not_found_propagates(self, api: BotBackend):
        api._client.start_chat_turn = AsyncMock(
            side_effect=NotFoundError("session gone")
        )

        with pytest.raises(NotFoundError):
            async for _ in api.stream_chat(
                platform="discord",
                platform_user_id="u1",
                message="hi",
                session_id="missing",
            ):
                pass

    @pytest.mark.asyncio
    async def test_subscribe_returns_none_yields_error(self, api: BotBackend):
        handle = ChatTurnHandle(session_id="sess", turn_id="turn", user_id="u1")
        api._client.start_chat_turn = AsyncMock(return_value=handle)

        with (
            patch(
                "backend.copilot.bot.bot_backend.stream_registry.subscribe_to_session",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "backend.copilot.bot.bot_backend.stream_registry.unsubscribe_from_session",
                new=AsyncMock(),
            ),
        ):
            chunks: list[str] = []
            async for chunk in api.stream_chat(
                platform="discord", platform_user_id="u1", message="hi"
            ):
                chunks.append(chunk)

        assert any("failed to subscribe" in c.lower() for c in chunks)
