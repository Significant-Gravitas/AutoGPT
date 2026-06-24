"""Tests for the bot's thin facade over PlatformLinkingManagerClient."""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamTextDelta,
    StreamToolOutputAvailable,
)
from backend.platform_linking.models import (
    ChatTurnHandle,
    LinkTokenResponse,
    Platform,
    ResolveResponse,
)
from backend.util.exceptions import (
    DuplicateChatMessageError,
    LinkAlreadyExistsError,
    NotFoundError,
)

from .bot_backend import (
    BotBackend,
    BotStreamError,
    _extract_setup_requirements,
    _is_corrupted_setup_requirements,
)


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


class TestListLinkedServerIds:
    @pytest.mark.asyncio
    async def test_forwards_uppercased_platform_and_returns_ids(self, api: BotBackend):
        api._client.list_user_server_ids = AsyncMock(return_value=["g1", "g2"])
        result = await api.list_linked_server_ids("discord", "user-1")
        assert result == ["g1", "g2"]
        api._client.list_user_server_ids.assert_awaited_once_with(
            platform=Platform.DISCORD, user_id="user-1"
        )

    @pytest.mark.asyncio
    async def test_empty_when_no_links(self, api: BotBackend):
        api._client.list_user_server_ids = AsyncMock(return_value=[])
        assert await api.list_linked_server_ids("discord", "user-1") == []


class TestRefreshServerName:
    @pytest.mark.asyncio
    async def test_forwards_uppercased_platform(self, api: BotBackend):
        api._client.refresh_server_link_name = AsyncMock()
        await api.refresh_server_name("discord", "g1", "AutoGPT HQ")
        api._client.refresh_server_link_name.assert_awaited_once_with(
            platform=Platform.DISCORD,
            platform_server_id="g1",
            server_name="AutoGPT HQ",
        )

    @pytest.mark.asyncio
    async def test_skips_call_when_server_name_is_empty(self, api: BotBackend):
        api._client.refresh_server_link_name = AsyncMock()
        await api.refresh_server_name("discord", "g1", "")
        api._client.refresh_server_link_name.assert_not_awaited()


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
            pytest.raises(BotStreamError) as excinfo,
        ):
            async for _ in api.stream_chat(
                platform="discord", platform_user_id="u1", message="hi"
            ):
                pass

        assert excinfo.value.error_kind == "backend_stream_error"
        assert "executor crashed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_notifies_setup_requirements_tool_output(self, api: BotBackend):
        handle = ChatTurnHandle(session_id="sess", turn_id="turn", user_id="u1")
        api._client.start_chat_turn = AsyncMock(return_value=handle)

        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(
            StreamToolOutputAvailable(
                toolCallId="tool-1",
                toolName="connect_integration",
                output='{"type":"setup_requirements","message":"Connect GitHub"}',
            )
        )
        await queue.put(StreamTextDelta(id="1", delta="After setup"))
        await queue.put(StreamFinish())

        setup_calls: list[tuple[str, dict, str | None]] = []

        async def on_setup(session_id: str, output: dict, tool_name: str | None):
            setup_calls.append((session_id, output, tool_name))

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
                on_setup_required=on_setup,
            ):
                chunks.append(chunk)

        assert chunks == ["After setup"]
        assert setup_calls == [
            (
                "sess",
                {"type": "setup_requirements", "message": "Connect GitHub"},
                "connect_integration",
            )
        ]

    @pytest.mark.asyncio
    async def test_notifies_setup_dropped_on_corrupted_output(self, api: BotBackend):
        handle = ChatTurnHandle(session_id="sess", turn_id="turn", user_id="u1")
        api._client.start_chat_turn = AsyncMock(return_value=handle)

        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(
            StreamToolOutputAvailable(
                toolCallId="tool-1",
                toolName="connect_integration",
                output='{"type":"setup_requirements","message":"Connect Goo',
            )
        )
        await queue.put(StreamFinish())

        setup_calls: list[tuple[str, dict, str | None]] = []
        dropped_calls: list[tuple[str, str | None]] = []

        async def on_setup(session_id: str, output: dict, tool_name: str | None):
            setup_calls.append((session_id, output, tool_name))

        async def on_dropped(session_id: str, tool_name: str | None):
            dropped_calls.append((session_id, tool_name))

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
            async for _ in api.stream_chat(
                platform="discord",
                platform_user_id="u1",
                message="hi",
                on_setup_required=on_setup,
                on_setup_dropped=on_dropped,
            ):
                pass

        # The corrupted payload yields no card, so on_setup_required must not
        # fire — but the user is told the link was dropped.
        assert setup_calls == []
        assert dropped_calls == [("sess", "connect_integration")]

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
    async def test_subscribe_returns_none_raises(self, api: BotBackend):
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
            pytest.raises(BotStreamError) as excinfo,
        ):
            async for _ in api.stream_chat(
                platform="discord", platform_user_id="u1", message="hi"
            ):
                pass

        assert excinfo.value.error_kind == "subscribe_failed"


class TestExtractSetupRequirements:
    def test_extracts_from_json_string(self):
        payload = '{"type":"setup_requirements","message":"Connect GitHub"}'
        assert _extract_setup_requirements(payload) == {
            "type": "setup_requirements",
            "message": "Connect GitHub",
        }

    def test_truncated_setup_requirements_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ):
        """A corrupted setup_requirements payload must not be dropped silently —
        the user never receives their sign-in link, so we need telemetry."""
        truncated = '{"type":"setup_requirements","message":"Connect Goo'
        with caplog.at_level(logging.WARNING):
            assert _extract_setup_requirements(truncated) is None
        assert any("setup_requirements" in record.message for record in caplog.records)

    def test_non_setup_unparseable_output_stays_silent(
        self, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING):
            assert _extract_setup_requirements("plain text tool result") is None
        assert not caplog.records


class TestIsCorruptedSetupRequirements:
    def test_truncated_setup_requirements_is_corrupted(self):
        assert _is_corrupted_setup_requirements(
            '{"type":"setup_requirements","message":"Connect Goo'
        )

    def test_valid_setup_requirements_is_not_corrupted(self):
        assert not _is_corrupted_setup_requirements(
            '{"type":"setup_requirements","message":"Connect GitHub"}'
        )

    def test_unparseable_non_setup_output_is_not_corrupted(self):
        assert not _is_corrupted_setup_requirements('{"type":"other","x')

    def test_plain_text_and_dict_outputs_are_not_corrupted(self):
        assert not _is_corrupted_setup_requirements("plain text tool result")
        assert not _is_corrupted_setup_requirements({"type": "setup_requirements"})
