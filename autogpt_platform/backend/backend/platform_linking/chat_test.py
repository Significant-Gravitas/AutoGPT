"""Tests for chat-turn orchestration — esp. the duplicate-message guard."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.exceptions import DuplicateChatMessageError, NotFoundError

from .chat import list_user_chats, start_chat_turn
from .models import BotChatRequest, Platform


def _request(**overrides) -> BotChatRequest:
    defaults = dict(
        platform=Platform.DISCORD,
        platform_user_id="pu1",
        message="hello",
    )
    defaults.update(overrides)
    return BotChatRequest(**defaults)


class TestStartChatTurn:
    @pytest.mark.asyncio
    async def test_no_user_link_raises_not_found(self):
        db_mock = MagicMock()
        db_mock.find_user_link_owner = AsyncMock(return_value=None)
        with patch(
            "backend.platform_linking.chat.platform_linking_db",
            return_value=db_mock,
        ):
            with pytest.raises(NotFoundError):
                await start_chat_turn(_request())

    @pytest.mark.asyncio
    async def test_duplicate_message_raises_and_skips_stream_create(self):
        # append_and_save_message returns None → duplicate.
        # Verify we raise and do NOT create a stream session.
        db_mock = MagicMock()
        db_mock.find_user_link_owner = AsyncMock(return_value="owner-1")
        session = MagicMock(session_id="sess-existing")

        with (
            patch(
                "backend.platform_linking.chat.platform_linking_db",
                return_value=db_mock,
            ),
            patch(
                "backend.platform_linking.chat.create_chat_session",
                new=AsyncMock(return_value=session),
            ),
            patch(
                "backend.platform_linking.chat.append_and_save_message",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "backend.platform_linking.chat.stream_registry"
            ) as mock_stream_registry,
            patch(
                "backend.platform_linking.chat.enqueue_copilot_turn",
                new=AsyncMock(),
            ) as mock_enqueue,
        ):
            mock_stream_registry.create_session = AsyncMock()

            with pytest.raises(DuplicateChatMessageError):
                await start_chat_turn(_request())

        mock_stream_registry.create_session.assert_not_awaited()
        mock_enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_happy_path_creates_stream_and_enqueues(self):
        db_mock = MagicMock()
        db_mock.find_user_link_owner = AsyncMock(return_value="owner-1")
        session = MagicMock(session_id="sess-new")
        mock_create_chat_session = AsyncMock(return_value=session)

        with (
            patch(
                "backend.platform_linking.chat.platform_linking_db",
                return_value=db_mock,
            ),
            patch(
                "backend.platform_linking.chat.create_chat_session",
                new=mock_create_chat_session,
            ),
            patch(
                "backend.platform_linking.chat.append_and_save_message",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch(
                "backend.platform_linking.chat.stream_registry"
            ) as mock_stream_registry,
            patch(
                "backend.platform_linking.chat.enqueue_copilot_turn",
                new=AsyncMock(),
            ) as mock_enqueue,
        ):
            mock_stream_registry.create_session = AsyncMock()
            handle = await start_chat_turn(_request())

        assert handle.session_id == "sess-new"
        assert handle.user_id == "owner-1"
        assert handle.turn_id
        assert handle.subscribe_from == "0-0"
        create_call = mock_create_chat_session.await_args
        assert create_call is not None
        assert create_call.kwargs["source_platform"] == "discord"
        mock_stream_registry.create_session.assert_awaited_once()
        mock_enqueue.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stale_session_id_falls_back_to_fresh_session(self):
        # The bot caches session IDs across turns; a cached session can be
        # deleted from the web app in between. start_chat_turn must recover
        # by creating a fresh session instead of raising.
        db_mock = MagicMock()
        db_mock.find_user_link_owner = AsyncMock(return_value="owner-1")
        session = MagicMock(session_id="sess-fresh")

        with (
            patch(
                "backend.platform_linking.chat.platform_linking_db",
                return_value=db_mock,
            ),
            patch(
                "backend.platform_linking.chat.get_chat_session",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "backend.platform_linking.chat.create_chat_session",
                new=AsyncMock(return_value=session),
            ) as mock_create,
            patch(
                "backend.platform_linking.chat.append_and_save_message",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch(
                "backend.platform_linking.chat.stream_registry"
            ) as mock_stream_registry,
            patch(
                "backend.platform_linking.chat.enqueue_copilot_turn",
                new=AsyncMock(),
            ),
        ):
            mock_stream_registry.create_session = AsyncMock()
            handle = await start_chat_turn(_request(session_id="deleted-session"))

        assert handle.session_id == "sess-fresh"
        mock_create.assert_awaited_once()


class TestListUserChats:
    @pytest.mark.asyncio
    async def test_unlinked_dm_raises_not_found(self):
        db_mock = MagicMock()
        db_mock.find_user_link_owner = AsyncMock(return_value=None)
        with patch(
            "backend.platform_linking.chat.platform_linking_db",
            return_value=db_mock,
        ):
            with pytest.raises(NotFoundError):
                await list_user_chats(Platform.DISCORD, "pu1")

    @pytest.mark.asyncio
    async def test_lists_only_the_resolved_owners_sessions(self):
        # Ownership is resolved from the caller's own DM link, and the
        # listing is scoped to exactly that owner — never a server.
        db_mock = MagicMock()
        db_mock.find_user_link_owner = AsyncMock(return_value="owner-1")
        session = MagicMock(
            session_id="sess-1",
            title="My chat",
            updated_at=datetime(2026, 5, 1),
        )

        with (
            patch(
                "backend.platform_linking.chat.platform_linking_db",
                return_value=db_mock,
            ),
            patch(
                "backend.platform_linking.chat.get_user_sessions",
                new=AsyncMock(return_value=([session], 1)),
            ) as mock_get_sessions,
        ):
            result = await list_user_chats(Platform.DISCORD, "pu1")

        db_mock.find_user_link_owner.assert_awaited_once_with("DISCORD", "pu1")
        get_sessions_call = mock_get_sessions.await_args
        assert get_sessions_call is not None
        assert get_sessions_call.args[0] == "owner-1"
        assert result.total == 1
        assert [s.session_id for s in result.sessions] == ["sess-1"]
        assert result.sessions[0].title == "My chat"

    @pytest.mark.asyncio
    async def test_clamps_pagination_inputs(self):
        """Negative or oversized pagination args must be clamped before they
        hit get_user_sessions — guards the DB and lines up with /resume's
        Discord-imposed 25-option select-menu limit."""
        db_mock = MagicMock()
        db_mock.find_user_link_owner = AsyncMock(return_value="owner-1")

        with (
            patch(
                "backend.platform_linking.chat.platform_linking_db",
                return_value=db_mock,
            ),
            patch(
                "backend.platform_linking.chat.get_user_sessions",
                new=AsyncMock(return_value=([], 0)),
            ) as mock_get_sessions,
        ):
            await list_user_chats(Platform.DISCORD, "pu1", limit=10_000, offset=-50)

        mock_get_sessions.assert_awaited_once_with("owner-1", limit=25, offset=0)


class TestUploadChatFile:
    @staticmethod
    def _req(**overrides):
        from .models import WorkspaceUploadRequest

        defaults = dict(
            platform=Platform.DISCORD,
            platform_user_id="pu1",
            filename="a.png",
            mime_type="image/png",
            content=b"data",
        )
        defaults.update(overrides)
        return WorkspaceUploadRequest(**defaults)

    @staticmethod
    def _patches(write_file_mock):
        db = MagicMock()
        db.find_user_link_owner = AsyncMock(return_value="owner-1")
        ws_db = MagicMock()
        ws_db.get_or_create_workspace = AsyncMock(return_value=MagicMock(id="ws-1"))
        manager = MagicMock()
        manager.write_file = write_file_mock
        return (
            patch("backend.platform_linking.chat.platform_linking_db", return_value=db),
            patch("backend.platform_linking.chat.workspace_db", return_value=ws_db),
            patch(
                "backend.platform_linking.chat.WorkspaceManager", return_value=manager
            ),
        )

    @pytest.mark.asyncio
    async def test_success_returns_file_id(self):
        from .chat import upload_chat_file

        write = AsyncMock(return_value=MagicMock(id="file-1"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3:
            result = await upload_chat_file(self._req())
        assert result.file_id == "file-1"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_virus_detected_maps_to_error(self):
        from backend.api.features.store.exceptions import VirusDetectedError

        from .chat import upload_chat_file

        write = AsyncMock(side_effect=VirusDetectedError("EICAR-Test"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3:
            result = await upload_chat_file(self._req())
        assert result.file_id is None
        assert result.error == "virus_detected"

    @pytest.mark.asyncio
    async def test_size_or_quota_maps_to_rejected(self):
        from .chat import upload_chat_file

        write = AsyncMock(side_effect=ValueError("File too large"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3:
            result = await upload_chat_file(self._req())
        assert result.error == "rejected"

    @pytest.mark.asyncio
    async def test_unlinked_user_raises_not_found(self):
        from .chat import upload_chat_file

        db = MagicMock()
        db.find_user_link_owner = AsyncMock(return_value=None)
        with patch(
            "backend.platform_linking.chat.platform_linking_db", return_value=db
        ):
            with pytest.raises(NotFoundError):
                await upload_chat_file(self._req())
