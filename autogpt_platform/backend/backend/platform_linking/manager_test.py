"""Tests for PlatformLinkingManager RPC wiring and confirm-token races."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.exceptions import LinkTokenExpiredError

from .db import confirm_server_link, confirm_user_link
from .manager import PlatformLinkingManager, PlatformLinkingManagerClient
from .models import (
    BotChatRequest,
    CreateLinkTokenRequest,
    CreateUserLinkTokenRequest,
    LinkType,
    Platform,
    ResolveResponse,
)


@asynccontextmanager
async def _fake_transaction():
    yield MagicMock()


class TestManagerWiring:
    def test_get_port(self):
        assert PlatformLinkingManager.get_port() == 8009

    def test_client_exposes_expected_rpc_surface(self):
        service_type = PlatformLinkingManagerClient.get_service_type()
        assert service_type is PlatformLinkingManager

        expected = {
            "resolve_server_link",
            "resolve_user_link",
            "create_server_link_token",
            "create_user_link_token",
            "get_link_token_status",
            "start_chat_turn",
        }
        for name in expected:
            assert hasattr(
                PlatformLinkingManagerClient, name
            ), f"Client missing RPC stub: {name}"

        for name in (
            "confirm_server_link",
            "confirm_user_link",
            "list_server_links",
            "list_user_links",
            "delete_server_link",
            "delete_user_link",
        ):
            assert not hasattr(
                PlatformLinkingManagerClient, name
            ), f"User-facing method leaked to bot client: {name}"

    @pytest.mark.asyncio
    async def test_resolve_server_link_delegates_to_accessor(self):
        manager = PlatformLinkingManager()
        db_mock = MagicMock()
        db_mock.resolve_server_link = AsyncMock(
            return_value=ResolveResponse(linked=True)
        )
        with patch(
            "backend.platform_linking.manager.platform_linking_db",
            return_value=db_mock,
        ):
            result = await manager.resolve_server_link(Platform.DISCORD, "g1")
        db_mock.resolve_server_link.assert_awaited_once_with("DISCORD", "g1")
        assert result.linked is True

    @pytest.mark.asyncio
    async def test_resolve_user_link_delegates_to_accessor(self):
        manager = PlatformLinkingManager()
        db_mock = MagicMock()
        db_mock.resolve_user_link = AsyncMock(
            return_value=ResolveResponse(linked=False)
        )
        with patch(
            "backend.platform_linking.manager.platform_linking_db",
            return_value=db_mock,
        ):
            result = await manager.resolve_user_link(Platform.DISCORD, "pu1")
        db_mock.resolve_user_link.assert_awaited_once_with("DISCORD", "pu1")
        assert result.linked is False

    @pytest.mark.asyncio
    async def test_create_server_link_token_delegates(self):
        manager = PlatformLinkingManager()
        req = CreateLinkTokenRequest(
            platform=Platform.DISCORD,
            platform_server_id="g1",
            platform_user_id="u1",
        )
        fake_response = MagicMock()
        db_mock = MagicMock()
        db_mock.create_server_link_token = AsyncMock(return_value=fake_response)
        with patch(
            "backend.platform_linking.manager.platform_linking_db",
            return_value=db_mock,
        ):
            result = await manager.create_server_link_token(req)
        db_mock.create_server_link_token.assert_awaited_once_with(req)
        assert result is fake_response

    @pytest.mark.asyncio
    async def test_create_user_link_token_delegates(self):
        manager = PlatformLinkingManager()
        req = CreateUserLinkTokenRequest(
            platform=Platform.DISCORD, platform_user_id="pu1"
        )
        fake_response = MagicMock()
        db_mock = MagicMock()
        db_mock.create_user_link_token = AsyncMock(return_value=fake_response)
        with patch(
            "backend.platform_linking.manager.platform_linking_db",
            return_value=db_mock,
        ):
            result = await manager.create_user_link_token(req)
        db_mock.create_user_link_token.assert_awaited_once_with(req)
        assert result is fake_response

    @pytest.mark.asyncio
    async def test_get_link_token_status_delegates(self):
        manager = PlatformLinkingManager()
        fake_response = MagicMock()
        db_mock = MagicMock()
        db_mock.get_link_token_status = AsyncMock(return_value=fake_response)
        with patch(
            "backend.platform_linking.manager.platform_linking_db",
            return_value=db_mock,
        ):
            result = await manager.get_link_token_status("tok")
        db_mock.get_link_token_status.assert_awaited_once_with("tok")
        assert result is fake_response

    @pytest.mark.asyncio
    async def test_start_chat_turn_delegates(self):
        manager = PlatformLinkingManager()
        req = BotChatRequest(
            platform=Platform.DISCORD,
            platform_user_id="pu1",
            message="hi",
        )
        fake_response = MagicMock()
        with patch(
            "backend.platform_linking.manager.start_chat_turn",
            new=AsyncMock(return_value=fake_response),
        ) as stub:
            result = await manager.start_chat_turn(req)
        stub.assert_awaited_once_with(req)
        assert result is fake_response


class TestAdversarialConfirmRace:
    """Concurrent confirm of one token: exactly one winner via ``update_many``
    guarded on ``usedAt = None``."""

    @pytest.mark.asyncio
    async def test_second_confirm_loses(self):
        # update_many returns 0 → caller lost the race
        fake_token = MagicMock(
            linkType=LinkType.SERVER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) + timedelta(minutes=10),
            platform="DISCORD",
            platformServerId="g1",
        )

        with (
            patch("backend.platform_linking.db.PlatformLinkToken") as mock_token,
            patch("backend.platform_linking.db.PlatformLink") as mock_link,
            patch("backend.platform_linking.db.transaction", new=_fake_transaction),
        ):
            mock_token.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            mock_link.prisma.return_value.find_first = AsyncMock(return_value=None)
            mock_token.prisma.return_value.update_many = AsyncMock(return_value=0)

            with pytest.raises(LinkTokenExpiredError):
                await confirm_server_link("abc", "user-late")

    @pytest.mark.asyncio
    async def test_second_confirm_wins_when_update_many_returns_one(self):
        fake_token = MagicMock(
            linkType=LinkType.SERVER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) + timedelta(minutes=10),
            platform="DISCORD",
            platformServerId="g1",
            platformUserId="pu1",
            serverName="S1",
        )

        with (
            patch("backend.platform_linking.db.PlatformLinkToken") as mock_token,
            patch("backend.platform_linking.db.PlatformLink") as mock_link,
            patch("backend.platform_linking.db.transaction", new=_fake_transaction),
        ):
            mock_token.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            mock_link.prisma.return_value.find_first = AsyncMock(return_value=None)
            mock_token.prisma.return_value.update_many = AsyncMock(return_value=1)
            mock_link.prisma.return_value.create = AsyncMock(return_value=MagicMock())

            result = await confirm_server_link("abc", "user-winner")

        assert result.success is True
        assert result.platform_server_id == "g1"

    @pytest.mark.asyncio
    async def test_gather_confirm_same_user_one_winner(self):
        fake_token = MagicMock(
            linkType=LinkType.SERVER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) + timedelta(minutes=10),
            platform="DISCORD",
            platformServerId="g1",
            platformUserId="pu1",
            serverName="S1",
        )
        update_results = [1, 0]

        async def flaky_update_many(*args, **kwargs):
            return update_results.pop(0)

        with (
            patch("backend.platform_linking.db.PlatformLinkToken") as mock_token,
            patch("backend.platform_linking.db.PlatformLink") as mock_link,
            patch("backend.platform_linking.db.transaction", new=_fake_transaction),
        ):
            mock_token.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            mock_link.prisma.return_value.find_first = AsyncMock(return_value=None)
            mock_token.prisma.return_value.update_many = flaky_update_many
            mock_link.prisma.return_value.create = AsyncMock(return_value=MagicMock())

            results = await asyncio.gather(
                confirm_server_link("abc", "u1"),
                confirm_server_link("abc", "u1"),
                return_exceptions=True,
            )

        successes = [r for r in results if not isinstance(r, Exception)]
        losses = [r for r in results if isinstance(r, LinkTokenExpiredError)]
        assert len(successes) == 1
        assert len(losses) == 1

    @pytest.mark.asyncio
    async def test_gather_confirm_different_users_one_winner_no_hijack(self):
        # Different users racing the same token: still exactly one winner,
        # and the other gets a clean LinkTokenExpiredError (no partial state).
        fake_token = MagicMock(
            linkType=LinkType.SERVER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) + timedelta(minutes=10),
            platform="DISCORD",
            platformServerId="g1",
            platformUserId="pu1",
            serverName="S1",
        )
        update_results = [1, 0]

        async def flaky_update_many(*args, **kwargs):
            return update_results.pop(0)

        created_link_user_ids: list[str] = []

        async def record_create(*, data):
            created_link_user_ids.append(data["userId"])
            return MagicMock()

        with (
            patch("backend.platform_linking.db.PlatformLinkToken") as mock_token,
            patch("backend.platform_linking.db.PlatformLink") as mock_link,
            patch("backend.platform_linking.db.transaction", new=_fake_transaction),
        ):
            mock_token.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            mock_link.prisma.return_value.find_first = AsyncMock(return_value=None)
            mock_token.prisma.return_value.update_many = flaky_update_many
            mock_link.prisma.return_value.create = record_create

            results = await asyncio.gather(
                confirm_server_link("abc", "user-a"),
                confirm_server_link("abc", "user-b"),
                return_exceptions=True,
            )

        successes = [r for r in results if not isinstance(r, Exception)]
        losses = [r for r in results if isinstance(r, LinkTokenExpiredError)]
        assert len(successes) == 1
        assert len(losses) == 1
        assert len(created_link_user_ids) == 1
        assert created_link_user_ids[0] in ("user-a", "user-b")

    @pytest.mark.asyncio
    async def test_gather_confirm_user_link_one_winner(self):
        fake_token = MagicMock(
            linkType=LinkType.USER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) + timedelta(minutes=10),
            platform="DISCORD",
            platformUserId="pu1",
            platformUsername="pu_name",
        )
        update_results = [1, 0]

        async def flaky_update_many(*args, **kwargs):
            return update_results.pop(0)

        with (
            patch("backend.platform_linking.db.PlatformLinkToken") as mock_token,
            patch("backend.platform_linking.db.PlatformUserLink") as mock_user_link,
            patch("backend.platform_linking.db.transaction", new=_fake_transaction),
        ):
            mock_token.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            mock_user_link.prisma.return_value.find_unique = AsyncMock(
                return_value=None
            )
            mock_token.prisma.return_value.update_many = flaky_update_many
            mock_user_link.prisma.return_value.create = AsyncMock(
                return_value=MagicMock()
            )

            results = await asyncio.gather(
                confirm_user_link("abc", "user-a"),
                confirm_user_link("abc", "user-b"),
                return_exceptions=True,
            )

        successes = [r for r in results if not isinstance(r, Exception)]
        losses = [r for r in results if isinstance(r, LinkTokenExpiredError)]
        assert len(successes) == 1
        assert len(losses) == 1
