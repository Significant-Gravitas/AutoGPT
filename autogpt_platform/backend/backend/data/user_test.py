"""Unit tests for helpers in backend.data.user."""

import re
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data import user as user_module
from backend.data.user import get_or_create_user, update_user_timezone
from backend.util.exceptions import DatabaseError


class TestGetOrCreateUserProfile:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_new_user_gets_default_marketplace_profile(self, server):
        """A first-time user must get a Profile row, like the old auth.users
        trigger provided — without one, store submissions are rejected."""
        from backend.data.db import prisma

        user_id = str(uuid.uuid4())
        email = f"profile-test-{user_id[:8]}@example.com"

        await get_or_create_user({"sub": user_id, "email": email})

        profile = await prisma.profile.find_first(where={"userId": user_id})
        assert profile is not None
        assert profile.name == email.split("@")[0]
        assert re.fullmatch(r"[a-z]+-[a-z]+-\d{5}", profile.username)

    @pytest.mark.asyncio
    async def test_username_collision_retries_instead_of_skipping_profile(self):
        """A username collision at insert time must not be mistaken for a
        concurrent same-user insert — the user would end up with no Profile."""
        from prisma.errors import UniqueViolationError

        create_mock = AsyncMock(
            side_effect=[UniqueViolationError(MagicMock()), MagicMock()]
        )
        # No profile exists for the user at any point; only the generated
        # username collides on first insert.
        find_first_mock = AsyncMock(return_value=None)

        with patch.object(user_module, "prisma") as mock_prisma:
            mock_prisma.profile.find_first = find_first_mock
            mock_prisma.profile.create = create_mock
            await user_module._ensure_default_profile("user-1", "x@example.com")

        assert create_mock.await_count == 2
        first_username = create_mock.await_args_list[0].kwargs["data"]["username"]
        second_username = create_mock.await_args_list[1].kwargs["data"]["username"]
        assert first_username != second_username

    @pytest.mark.asyncio
    async def test_concurrent_profile_insert_is_tolerated(self):
        """If another request created the user's profile mid-insert, the
        violation is benign and must not raise."""
        from prisma.errors import UniqueViolationError

        create_mock = AsyncMock(side_effect=UniqueViolationError(MagicMock()))
        # First check (pre-insert): no profile. Username check: free.
        # Post-violation check: profile now exists.
        find_first_mock = AsyncMock(side_effect=[None, None, MagicMock()])

        with patch.object(user_module, "prisma") as mock_prisma:
            mock_prisma.profile.find_first = find_first_mock
            mock_prisma.profile.create = create_mock
            await user_module._ensure_default_profile("user-1", "x@example.com")

        assert create_mock.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_existing_user_without_profile_is_backfilled(self, server):
        """Users created before the profile fix get one on their next request."""
        from backend.data.db import prisma

        user_id = str(uuid.uuid4())
        email = f"profile-backfill-{user_id[:8]}@example.com"

        await prisma.user.create(data={"id": user_id, "email": email})
        get_or_create_user.cache_clear()

        await get_or_create_user({"sub": user_id, "email": email})

        profile = await prisma.profile.find_first(where={"userId": user_id})
        assert profile is not None


class TestUpdateUserTimezone:
    @pytest.mark.asyncio
    async def test_invalidates_all_three_user_caches(self):
        prisma_user = MagicMock(id="user-1", email="user@example.com")
        sentinel_user = MagicMock()

        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.User, "from_db", return_value=sentinel_user),
            patch.object(user_module.get_user_by_id, "cache_delete") as by_id_del,
            patch.object(user_module.get_user_by_email, "cache_delete") as by_email_del,
            patch.object(user_module.get_or_create_user, "cache_clear") as goc_clear,
        ):
            mock_prisma_user.prisma.return_value.update = AsyncMock(
                return_value=prisma_user
            )
            result = await update_user_timezone("user-1", "Europe/London")

        assert result is sentinel_user
        by_id_del.assert_called_once_with("user-1")
        by_email_del.assert_called_once_with("user@example.com")
        goc_clear.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_skips_email_cache_invalidation_when_email_missing(self):
        prisma_user = MagicMock(id="user-1", email=None)
        sentinel_user = MagicMock()

        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.User, "from_db", return_value=sentinel_user),
            patch.object(user_module.get_user_by_id, "cache_delete") as by_id_del,
            patch.object(user_module.get_user_by_email, "cache_delete") as by_email_del,
            patch.object(user_module.get_or_create_user, "cache_clear") as goc_clear,
        ):
            mock_prisma_user.prisma.return_value.update = AsyncMock(
                return_value=prisma_user
            )
            await update_user_timezone("user-1", "Europe/London")

        by_id_del.assert_called_once_with("user-1")
        by_email_del.assert_not_called()
        goc_clear.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_wraps_prisma_errors_in_database_error(self):
        with patch.object(user_module, "PrismaUser") as mock_prisma_user:
            mock_prisma_user.prisma.return_value.update = AsyncMock(
                side_effect=RuntimeError("connection lost")
            )
            with pytest.raises(DatabaseError) as exc:
                await update_user_timezone("user-1", "Europe/London")

        assert "user-1" in str(exc.value)
        assert "connection lost" in str(exc.value)
