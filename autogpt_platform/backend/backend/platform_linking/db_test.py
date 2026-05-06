"""Unit tests for platform_linking DB operations."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.exceptions import (
    LinkAlreadyExistsError,
    LinkFlowMismatchError,
    LinkTokenExpiredError,
    NotAuthorizedError,
    NotFoundError,
)

from .db import (
    LINK_TOKEN_RETENTION_HOURS,
    cleanup_expired_platform_link_tokens,
    confirm_server_link,
    confirm_user_link,
    create_server_link_token,
    create_user_link_token,
    delete_server_link,
    delete_user_link,
    get_link_token_info,
    get_link_token_status,
    resolve_server_link,
    resolve_user_link,
)
from .models import (
    CreateLinkTokenRequest,
    CreateUserLinkTokenRequest,
    LinkType,
    Platform,
)


@asynccontextmanager
async def _fake_transaction():
    # Avoids Prisma's tx binding asyncio primitives to the wrong loop in tests.
    yield MagicMock()


# ── Resolve ──────────────────────────────────────────────────────────


class TestResolve:
    @pytest.mark.asyncio
    async def test_server_linked(self):
        with patch("backend.platform_linking.db.PlatformLink") as mock_link:
            mock_link.prisma.return_value.find_first = AsyncMock(
                return_value=MagicMock(userId="u-123")
            )
            result = await resolve_server_link("DISCORD", "g1")
        assert result.linked is True

    @pytest.mark.asyncio
    async def test_server_unlinked(self):
        with patch("backend.platform_linking.db.PlatformLink") as mock_link:
            mock_link.prisma.return_value.find_first = AsyncMock(return_value=None)
            result = await resolve_server_link("DISCORD", "g1")
        assert result.linked is False

    @pytest.mark.asyncio
    async def test_user_linked(self):
        with patch("backend.platform_linking.db.PlatformUserLink") as mock_user_link:
            mock_user_link.prisma.return_value.find_unique = AsyncMock(
                return_value=MagicMock(userId="u-xyz")
            )
            result = await resolve_user_link("DISCORD", "pu1")
        assert result.linked is True

    @pytest.mark.asyncio
    async def test_user_unlinked(self):
        with patch("backend.platform_linking.db.PlatformUserLink") as mock_user_link:
            mock_user_link.prisma.return_value.find_unique = AsyncMock(
                return_value=None
            )
            result = await resolve_user_link("DISCORD", "pu1")
        assert result.linked is False


# ── Token creation ───────────────────────────────────────────────────


class TestCreateServerLinkToken:
    @pytest.mark.asyncio
    async def test_creates_token_for_unlinked_server(self):
        with (
            patch("backend.platform_linking.db.PlatformLink") as mock_link,
            patch(
                "backend.platform_linking.db.transaction",
                new=_fake_transaction,
            ),
            patch("backend.platform_linking.db.PlatformLinkToken") as mock_token_model,
        ):
            mock_link.prisma.return_value.find_first = AsyncMock(return_value=None)
            mock_token_model.prisma.return_value.update_many = AsyncMock(return_value=0)
            mock_token_model.prisma.return_value.create = AsyncMock(
                return_value=MagicMock()
            )

            result = await create_server_link_token(
                CreateLinkTokenRequest(
                    platform=Platform.DISCORD,
                    platform_server_id="g1",
                    platform_user_id="u1",
                    server_name="Test",
                ),
            )

        assert result.token
        assert result.token in result.link_url
        assert "?platform=DISCORD" in result.link_url

    @pytest.mark.asyncio
    async def test_rejects_when_already_linked(self):
        with patch("backend.platform_linking.db.PlatformLink") as mock_link:
            mock_link.prisma.return_value.find_first = AsyncMock(
                return_value=MagicMock(userId="u-owner")
            )
            with pytest.raises(LinkAlreadyExistsError):
                await create_server_link_token(
                    CreateLinkTokenRequest(
                        platform=Platform.DISCORD,
                        platform_server_id="g1",
                        platform_user_id="u1",
                    ),
                )


class TestCreateUserLinkToken:
    @pytest.mark.asyncio
    async def test_creates_token_for_unlinked_user(self):
        with (
            patch("backend.platform_linking.db.PlatformUserLink") as mock_user_link,
            patch(
                "backend.platform_linking.db.transaction",
                new=_fake_transaction,
            ),
            patch("backend.platform_linking.db.PlatformLinkToken") as mock_token_model,
        ):
            mock_user_link.prisma.return_value.find_unique = AsyncMock(
                return_value=None
            )
            mock_token_model.prisma.return_value.update_many = AsyncMock(return_value=0)
            mock_token_model.prisma.return_value.create = AsyncMock(
                return_value=MagicMock()
            )

            result = await create_user_link_token(
                CreateUserLinkTokenRequest(
                    platform=Platform.DISCORD,
                    platform_user_id="pu1",
                    platform_username="Bently",
                ),
            )

        assert result.token
        assert result.token in result.link_url

    @pytest.mark.asyncio
    async def test_rejects_when_already_linked(self):
        with patch("backend.platform_linking.db.PlatformUserLink") as mock_user_link:
            mock_user_link.prisma.return_value.find_unique = AsyncMock(
                return_value=MagicMock(userId="u-owner")
            )
            with pytest.raises(LinkAlreadyExistsError):
                await create_user_link_token(
                    CreateUserLinkTokenRequest(
                        platform=Platform.DISCORD,
                        platform_user_id="pu1",
                    ),
                )


# ── Token status / info ───────────────────────────────────────────────


class TestGetLinkTokenStatus:
    @pytest.mark.asyncio
    async def test_not_found(self):
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=None)
            with pytest.raises(NotFoundError):
                await get_link_token_status("abc")

    @pytest.mark.asyncio
    async def test_pending(self):
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        fake_token = MagicMock(usedAt=None, expiresAt=future)
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            result = await get_link_token_status("abc")
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_expired_by_time(self):
        past = datetime.now(timezone.utc) - timedelta(minutes=10)
        fake_token = MagicMock(usedAt=None, expiresAt=past)
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            result = await get_link_token_status("abc")
        assert result.status == "expired"

    @pytest.mark.asyncio
    async def test_used_with_user_link_reports_linked(self):
        fake_token = MagicMock(
            usedAt=datetime.now(timezone.utc),
            linkType=LinkType.USER.value,
            platform="DISCORD",
            platformUserId="pu1",
        )
        with (
            patch("backend.platform_linking.db.PlatformLinkToken") as mock_token,
            patch("backend.platform_linking.db.PlatformUserLink") as mock_user_link,
        ):
            mock_token.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            mock_user_link.prisma.return_value.find_unique = AsyncMock(
                return_value=MagicMock(userId="u-owner")
            )
            result = await get_link_token_status("abc")
        assert result.status == "linked"

    @pytest.mark.asyncio
    async def test_used_without_link_reports_expired(self):
        # Superseded token: usedAt set, but no backing link row.
        fake_token = MagicMock(
            usedAt=datetime.now(timezone.utc),
            linkType=LinkType.SERVER.value,
            platform="DISCORD",
            platformServerId="g1",
        )
        with (
            patch("backend.platform_linking.db.PlatformLinkToken") as mock_token,
            patch("backend.platform_linking.db.PlatformLink") as mock_link,
        ):
            mock_token.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            mock_link.prisma.return_value.find_first = AsyncMock(return_value=None)
            result = await get_link_token_status("abc")
        assert result.status == "expired"


class TestGetLinkTokenInfo:
    @pytest.mark.asyncio
    async def test_not_found(self):
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=None)
            with pytest.raises(NotFoundError):
                await get_link_token_info("abc")

    @pytest.mark.asyncio
    async def test_used_returns_not_found(self):
        fake_token = MagicMock(usedAt=datetime.now(timezone.utc))
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(NotFoundError):
                await get_link_token_info("abc")

    @pytest.mark.asyncio
    async def test_expired_raises_expired(self):
        past = datetime.now(timezone.utc) - timedelta(minutes=5)
        fake_token = MagicMock(usedAt=None, expiresAt=past)
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(LinkTokenExpiredError):
                await get_link_token_info("abc")

    @pytest.mark.asyncio
    async def test_success_returns_display_info(self):
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        fake_token = MagicMock(
            usedAt=None,
            expiresAt=future,
            platform="DISCORD",
            linkType=LinkType.SERVER.value,
            serverName="My Server",
        )
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            result = await get_link_token_info("abc")
        assert result.platform == "DISCORD"
        assert result.link_type == LinkType.SERVER
        assert result.server_name == "My Server"


# ── Confirmation ─────────────────────────────────────────────────────


class TestConfirmServerLink:
    @pytest.mark.asyncio
    async def test_not_found(self):
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=None)
            with pytest.raises(NotFoundError):
                await confirm_server_link("abc", "u1")

    @pytest.mark.asyncio
    async def test_wrong_link_type_rejected(self):
        fake_token = MagicMock(linkType=LinkType.USER.value)
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(LinkFlowMismatchError):
                await confirm_server_link("abc", "u1")

    @pytest.mark.asyncio
    async def test_already_used(self):
        fake_token = MagicMock(
            linkType=LinkType.SERVER.value, usedAt=datetime.now(timezone.utc)
        )
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(LinkTokenExpiredError):
                await confirm_server_link("abc", "u1")

    @pytest.mark.asyncio
    async def test_expired_by_time(self):
        fake_token = MagicMock(
            linkType=LinkType.SERVER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(LinkTokenExpiredError):
                await confirm_server_link("abc", "u1")

    @pytest.mark.asyncio
    async def test_already_linked_to_same_user(self):
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
        ):
            mock_token.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            mock_link.prisma.return_value.find_first = AsyncMock(
                return_value=MagicMock(userId="u1")
            )
            with pytest.raises(LinkAlreadyExistsError) as exc_info:
                await confirm_server_link("abc", "u1")
        assert "your account" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_already_linked_to_other_user(self):
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
        ):
            mock_token.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            mock_link.prisma.return_value.find_first = AsyncMock(
                return_value=MagicMock(userId="other-user")
            )
            with pytest.raises(LinkAlreadyExistsError) as exc_info:
                await confirm_server_link("abc", "u1")
        assert "another" in str(exc_info.value)


class TestConfirmUserLink:
    @pytest.mark.asyncio
    async def test_not_found(self):
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=None)
            with pytest.raises(NotFoundError):
                await confirm_user_link("abc", "u1")

    @pytest.mark.asyncio
    async def test_wrong_link_type_rejected(self):
        fake_token = MagicMock(linkType=LinkType.SERVER.value)
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(LinkFlowMismatchError):
                await confirm_user_link("abc", "u1")

    @pytest.mark.asyncio
    async def test_expired_by_time(self):
        fake_token = MagicMock(
            linkType=LinkType.USER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            with pytest.raises(LinkTokenExpiredError):
                await confirm_user_link("abc", "u1")

    @pytest.mark.asyncio
    async def test_already_linked_to_other_user(self):
        fake_token = MagicMock(
            linkType=LinkType.USER.value,
            usedAt=None,
            expiresAt=datetime.now(timezone.utc) + timedelta(minutes=10),
            platform="DISCORD",
            platformUserId="pu1",
        )
        with (
            patch("backend.platform_linking.db.PlatformLinkToken") as mock_token,
            patch("backend.platform_linking.db.PlatformUserLink") as mock_user_link,
        ):
            mock_token.prisma.return_value.find_unique = AsyncMock(
                return_value=fake_token
            )
            mock_user_link.prisma.return_value.find_unique = AsyncMock(
                return_value=MagicMock(userId="other-user")
            )
            with pytest.raises(LinkAlreadyExistsError):
                await confirm_user_link("abc", "u1")


# ── Delete (authz checks) ────────────────────────────────────────────


class TestDeleteLinks:
    @pytest.mark.asyncio
    async def test_delete_server_link_not_found(self):
        with patch("backend.platform_linking.db.PlatformLink") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=None)
            with pytest.raises(NotFoundError):
                await delete_server_link("x", "u1")

    @pytest.mark.asyncio
    async def test_delete_server_link_not_owned(self):
        link = MagicMock(userId="owner-A", platform="DISCORD", platformServerId="g1")
        with patch("backend.platform_linking.db.PlatformLink") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=link)
            with pytest.raises(NotAuthorizedError):
                await delete_server_link("x", "u-other")

    @pytest.mark.asyncio
    async def test_delete_user_link_not_found(self):
        with patch("backend.platform_linking.db.PlatformUserLink") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=None)
            with pytest.raises(NotFoundError):
                await delete_user_link("x", "u1")

    @pytest.mark.asyncio
    async def test_delete_user_link_not_owned(self):
        link = MagicMock(userId="owner-A", platform="DISCORD")
        with patch("backend.platform_linking.db.PlatformUserLink") as mock_model:
            mock_model.prisma.return_value.find_unique = AsyncMock(return_value=link)
            with pytest.raises(NotAuthorizedError):
                await delete_user_link("x", "u-other")


# ── Cleanup ──────────────────────────────────────────────────────────


class TestCleanupExpired:
    @pytest.mark.asyncio
    async def test_deletes_with_retention_window_cutoff(self):
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.delete_many = AsyncMock(return_value=7)
            count = await cleanup_expired_platform_link_tokens()

        assert count == 7
        mock_model.prisma.return_value.delete_many.assert_awaited_once()
        where = mock_model.prisma.return_value.delete_many.await_args.kwargs["where"]
        assert "expiresAt" in where and "lt" in where["expiresAt"]
        cutoff = where["expiresAt"]["lt"]
        delta = datetime.now(timezone.utc) - cutoff
        assert (
            timedelta(hours=LINK_TOKEN_RETENTION_HOURS - 1)
            < delta
            < timedelta(hours=LINK_TOKEN_RETENTION_HOURS + 1)
        )

    @pytest.mark.asyncio
    async def test_returns_zero_when_nothing_to_delete(self):
        with patch("backend.platform_linking.db.PlatformLinkToken") as mock_model:
            mock_model.prisma.return_value.delete_many = AsyncMock(return_value=0)
            count = await cleanup_expired_platform_link_tokens()
        assert count == 0
