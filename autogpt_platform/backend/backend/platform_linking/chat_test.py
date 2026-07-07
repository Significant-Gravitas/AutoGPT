"""Tests for chat-turn orchestration — esp. the duplicate-message guard."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot.rate_limit import RateLimitExceeded, RateLimitUnavailable
from backend.util.exceptions import DuplicateChatMessageError, NotFoundError

from .chat import (
    ensure_chat_session,
    evaluate_turn_gate,
    list_user_chats,
    start_chat_turn,
    upload_workspace_file,
)
from .models import BotChatRequest, Platform, TurnDenial, WorkspaceUploadRequest


def _request(**overrides) -> BotChatRequest:
    defaults = dict(
        platform=Platform.DISCORD,
        platform_user_id="pu1",
        message="hello",
    )
    defaults.update(overrides)
    return BotChatRequest(**defaults)


class TestStartChatTurn:
    @pytest.fixture(autouse=True)
    def _allow_turn(self):
        # The paywall/rate-limit gate is covered in TestEvaluateTurnGate; here
        # default it to "allow" so these tests don't reach DB/Redis for it.
        with patch(
            "backend.platform_linking.chat.evaluate_turn_gate",
            new=AsyncMock(return_value=None),
        ):
            yield

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

    @pytest.mark.asyncio
    async def test_attachment_turn_with_missing_session_raises(self):
        # An attachment turn's files are pinned to the supplied session; if it's
        # gone, fail rather than silently switching to a new (fileless) session.
        db_mock = MagicMock()
        db_mock.find_user_link_owner = AsyncMock(return_value="owner-1")
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
                "backend.platform_linking.chat.create_chat_session", new=AsyncMock()
            ) as mock_create,
        ):
            with pytest.raises(NotFoundError):
                await start_chat_turn(_request(session_id="gone", file_ids=["f1"]))

        mock_create.assert_not_awaited()


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


class TestUploadWorkspaceFile:
    @staticmethod
    def _req(**overrides):
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
        write = AsyncMock(return_value=MagicMock(id="file-1"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3:
            result = await upload_workspace_file(self._req())
        assert result.file_id == "file-1"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_virus_detected_maps_to_error(self):
        write = AsyncMock(side_effect=VirusDetectedError("EICAR-Test"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3:
            result = await upload_workspace_file(self._req())
        assert result.file_id is None
        assert result.error == "virus_detected"

    @pytest.mark.asyncio
    async def test_scan_unavailable_maps_to_error(self):
        write = AsyncMock(side_effect=VirusScanError("clamd down"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3:
            result = await upload_workspace_file(self._req())
        assert result.error == "scan_unavailable"

    @pytest.mark.asyncio
    async def test_size_or_quota_maps_to_rejected(self):
        write = AsyncMock(side_effect=ValueError("File too large"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3:
            result = await upload_workspace_file(self._req())
        assert result.error == "rejected"

    @pytest.mark.asyncio
    async def test_unexpected_error_maps_to_upload_failed(self):
        write = AsyncMock(side_effect=RuntimeError("storage exploded"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3:
            result = await upload_workspace_file(self._req())
        assert result.error == "upload_failed"

    @pytest.mark.asyncio
    async def test_writes_into_session_scoped_manager(self):
        write = AsyncMock(return_value=MagicMock(id="file-1"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3 as mock_wm:
            await upload_workspace_file(self._req(session_id="sess-1"))
        # Session-scoped manager (like the web upload) plus a flat filename —
        # write_file defaults the path to /sessions/<id>/<name> where AutoPilot
        # reads it. No explicit uploads/<uuid> path.
        mock_wm.assert_called_once_with("owner-1", "ws-1", "sess-1")
        kwargs = write.await_args.kwargs
        assert kwargs["filename"] == "a.png"
        assert "path" not in kwargs
        # Overwrite so a re-uploaded same-named file replaces rather than
        # erroring (a conflict would be mislabelled a size/quota rejection).
        assert kwargs["overwrite"] is True

    @pytest.mark.asyncio
    async def test_filename_path_components_are_stripped(self):
        write = AsyncMock(return_value=MagicMock(id="file-1"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3:
            await upload_workspace_file(self._req(filename="../../etc/passwd"))
        # Only the sanitized basename reaches the storage backend.
        kwargs = write.await_args.kwargs
        assert kwargs["filename"] == "passwd"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", [".", "..", "dir/..", "/"])
    async def test_dot_filenames_fall_back_to_safe_name(self, name: str):
        # "."/".." survive basename stripping and would be a special path
        # segment, so they must become "file".
        write = AsyncMock(return_value=MagicMock(id="file-1"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3:
            await upload_workspace_file(self._req(filename=name))
        assert write.await_args.kwargs["filename"] == "file"

    @pytest.mark.asyncio
    async def test_unlinked_user_raises_not_found(self):
        db = MagicMock()
        db.find_user_link_owner = AsyncMock(return_value=None)
        with patch(
            "backend.platform_linking.chat.platform_linking_db", return_value=db
        ):
            with pytest.raises(NotFoundError):
                await upload_workspace_file(self._req())

    @pytest.mark.asyncio
    async def test_not_found_in_try_propagates_not_rejected(self):
        # NotFoundError subclasses ValueError; if one is raised inside the
        # write path it must propagate, not be swallowed as "rejected".
        write = AsyncMock(side_effect=NotFoundError("workspace gone"))
        p1, p2, p3 = self._patches(write)
        with p1, p2, p3:
            with pytest.raises(NotFoundError):
                await upload_workspace_file(self._req())


class TestEnsureChatSession:
    @pytest.fixture(autouse=True)
    def _allow_turn(self):
        # Gate behaviour is covered in TestEvaluateTurnGate and the dedicated
        # denial test below; default to "allow" so the happy-path tests don't
        # reach DB/Redis for it.
        with patch(
            "backend.platform_linking.chat.evaluate_turn_gate",
            new=AsyncMock(return_value=None),
        ):
            yield

    @pytest.mark.asyncio
    async def test_reuses_valid_cached_session(self):
        db = MagicMock()
        db.find_user_link_owner = AsyncMock(return_value="owner-1")
        with (
            patch("backend.platform_linking.chat.platform_linking_db", return_value=db),
            patch(
                "backend.platform_linking.chat.get_chat_session",
                new=AsyncMock(return_value=MagicMock(session_id="cached")),
            ),
            patch(
                "backend.platform_linking.chat.create_chat_session", new=AsyncMock()
            ) as mock_create,
        ):
            result = await ensure_chat_session(Platform.DISCORD, "pu1", None, "cached")

        assert result.session_id == "cached"
        assert result.denial is None
        mock_create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_creates_session_when_none_cached(self):
        db = MagicMock()
        db.find_user_link_owner = AsyncMock(return_value="owner-1")
        with (
            patch("backend.platform_linking.chat.platform_linking_db", return_value=db),
            patch(
                "backend.platform_linking.chat.create_chat_session",
                new=AsyncMock(return_value=MagicMock(session_id="fresh")),
            ) as mock_create,
        ):
            result = await ensure_chat_session(Platform.DISCORD, "pu1", None, None)

        assert result.session_id == "fresh"
        assert mock_create.await_args.kwargs["source_platform"] == "discord"

    @pytest.mark.asyncio
    async def test_denied_user_gets_denial_and_no_session(self):
        # A capped/paywalled user is refused BEFORE any session is created —
        # the caller then skips the file upload entirely (nothing scanned or
        # stored for a denied user).
        db = MagicMock()
        db.find_user_link_owner = AsyncMock(return_value="owner-1")
        denial = TurnDenial(reason="rate_limited", message="limit reached")
        with (
            patch("backend.platform_linking.chat.platform_linking_db", return_value=db),
            patch(
                "backend.platform_linking.chat.evaluate_turn_gate",
                new=AsyncMock(return_value=denial),
            ),
            patch(
                "backend.platform_linking.chat.create_chat_session", new=AsyncMock()
            ) as mock_create,
        ):
            result = await ensure_chat_session(Platform.DISCORD, "pu1", None, None)

        assert result.session_id is None
        assert result.denial == denial
        mock_create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unlinked_dm_raises_not_found(self):
        db = MagicMock()
        db.find_user_link_owner = AsyncMock(return_value=None)
        with patch(
            "backend.platform_linking.chat.platform_linking_db", return_value=db
        ):
            with pytest.raises(NotFoundError):
                await ensure_chat_session(Platform.DISCORD, "pu1", None, None)


class TestEvaluateTurnGate:
    _PATH = "backend.platform_linking.chat"

    @pytest.mark.asyncio
    async def test_paywalled_returns_denial_with_button(self):
        with (
            patch(f"{self._PATH}.is_user_paywalled", AsyncMock(return_value=True)),
            patch(
                f"{self._PATH}._billing_url",
                return_value="https://app/settings/billing",
            ),
        ):
            denial = await evaluate_turn_gate("user-1")

        assert denial is not None
        assert denial.reason == "paywalled"
        assert denial.button_url == "https://app/settings/billing"
        assert denial.button_label == "Subscribe"

    @pytest.mark.asyncio
    async def test_rate_limited_returns_window_and_reset_message(self):
        from datetime import timedelta, timezone

        resets_at = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        with (
            patch(f"{self._PATH}.is_user_paywalled", AsyncMock(return_value=False)),
            patch(
                f"{self._PATH}.get_global_rate_limits",
                AsyncMock(return_value=(100, 500, "BASIC")),
            ),
            patch(
                f"{self._PATH}.check_rate_limit",
                AsyncMock(side_effect=RateLimitExceeded("daily", resets_at)),
            ),
        ):
            denial = await evaluate_turn_gate("user-1")

        assert denial is not None
        assert denial.reason == "rate_limited"
        assert "daily usage limit" in denial.message
        assert "Resets in" in denial.message

    @pytest.mark.asyncio
    async def test_rate_unavailable_returns_transient_message_no_button(self):
        with (
            patch(f"{self._PATH}.is_user_paywalled", AsyncMock(return_value=False)),
            patch(
                f"{self._PATH}.get_global_rate_limits",
                AsyncMock(return_value=(100, 500, "BASIC")),
            ),
            patch(
                f"{self._PATH}.check_rate_limit",
                AsyncMock(side_effect=RateLimitUnavailable()),
            ),
        ):
            denial = await evaluate_turn_gate("user-1")

        assert denial is not None
        assert denial.reason == "unavailable"
        assert denial.button_url is None

    @pytest.mark.asyncio
    async def test_paywall_lookup_failure_fails_closed_as_unavailable(self):
        # A transient tier-lookup error must NOT let the turn through unmetered;
        # fail closed with a retry message (mirrors the web route's 503).
        with patch(
            f"{self._PATH}.is_user_paywalled",
            AsyncMock(side_effect=RuntimeError("supabase down")),
        ):
            denial = await evaluate_turn_gate("user-1")

        assert denial is not None
        assert denial.reason == "unavailable"
        assert denial.button_url is None

    @pytest.mark.asyncio
    async def test_within_limits_returns_none(self):
        with (
            patch(f"{self._PATH}.is_user_paywalled", AsyncMock(return_value=False)),
            patch(
                f"{self._PATH}.get_global_rate_limits",
                AsyncMock(return_value=(100, 500, "BASIC")),
            ),
            patch(f"{self._PATH}.check_rate_limit", AsyncMock(return_value=None)),
        ):
            denial = await evaluate_turn_gate("user-1")

        assert denial is None
