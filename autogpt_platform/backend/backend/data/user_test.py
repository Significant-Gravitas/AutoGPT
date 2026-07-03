"""Unit tests for helpers in backend.data.user."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.errors
import pytest

from backend.data import user as user_module
from backend.data.user import update_user_timezone
from backend.util.exceptions import DatabaseError


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

    @pytest.mark.asyncio
    async def test_eagerly_re_registers_dream_schedules_with_force_refresh(self):
        """APScheduler cron triggers bind to the timezone at registration
        time. A profile-page timezone change MUST eagerly re-register
        the dream-system crons so they fire at the right local time
        without waiting for the 7-day Redis dedup-key TTL to expire."""
        from backend.copilot.dream import scheduling as dream_scheduling

        prisma_user = MagicMock(id="user-tz", email="user@example.com")
        captured: list[tuple[str, bool]] = []

        async def fake_ensure(user_id: str, *, force_refresh: bool = False):
            captured.append((user_id, force_refresh))
            return {}

        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.User, "from_db", return_value=MagicMock()),
            patch.object(user_module.get_user_by_id, "cache_delete"),
            patch.object(user_module.get_user_by_email, "cache_delete"),
            patch.object(user_module.get_or_create_user, "cache_clear"),
            patch.object(
                dream_scheduling, "ensure_dream_system_scheduled", new=fake_ensure
            ),
        ):
            mock_prisma_user.prisma.return_value.update = AsyncMock(
                return_value=prisma_user
            )
            await update_user_timezone("user-tz", "Europe/Paris")
            # Yield once so the asyncio.create_task body runs before we
            # assert it was called.
            await asyncio.sleep(0)

        assert captured == [("user-tz", True)]

    @pytest.mark.asyncio
    async def test_re_register_task_is_retained_and_its_failure_logged(self):
        """The event loop holds only weak refs to tasks — an unretained
        fire-and-forget re-register can be GC'd mid-flight and its
        exception never observed. The spawn must keep a strong ref in
        ``_background_tasks`` until done and log failures via the
        done-callback instead of dropping them."""
        from backend.copilot.dream import scheduling as dream_scheduling

        prisma_user = MagicMock(id="user-tz", email="user@example.com")

        async def failing_ensure(user_id: str, *, force_refresh: bool = False):
            raise RuntimeError("scheduler unreachable")

        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.User, "from_db", return_value=MagicMock()),
            patch.object(user_module.get_user_by_id, "cache_delete"),
            patch.object(user_module.get_user_by_email, "cache_delete"),
            patch.object(user_module.get_or_create_user, "cache_clear"),
            patch.object(
                dream_scheduling, "ensure_dream_system_scheduled", new=failing_ensure
            ),
            patch.object(user_module.logger, "warning") as warn_mock,
        ):
            mock_prisma_user.prisma.return_value.update = AsyncMock(
                return_value=prisma_user
            )
            await update_user_timezone("user-tz", "Europe/Paris")

            spawned = [
                t
                for t in user_module._background_tasks
                if t.get_name() == "tz-reregister-user-tz"
            ]
            assert spawned, "task must be strongly referenced until it completes"

            await asyncio.gather(*spawned, return_exceptions=True)
            # One more tick so the done-callback (scheduled via
            # call_soon) runs.
            await asyncio.sleep(0)

        assert not user_module._background_tasks & set(spawned)
        warn_mock.assert_called_once()
        assert isinstance(warn_mock.call_args.kwargs["exc_info"], RuntimeError)


class TestGetOrCreateUserProfile:
    """get_or_create_user must guarantee a marketplace Profile exists, since
    the auth.users trigger that used to do this is unreliable."""

    @pytest.mark.asyncio
    async def test_creates_profile_when_missing(self):
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-new", email="alice@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(user_module.User, "from_db", return_value=MagicMock()),
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)
            # No existing profile, and the generated username is free.
            mock_prisma.profile.find_unique = AsyncMock(return_value=None)
            mock_prisma.profile.create = AsyncMock()

            await user_module.get_or_create_user(
                {"sub": "user-new", "email": "alice@example.com"}
            )

        mock_prisma.profile.create.assert_awaited_once()
        created = mock_prisma.profile.create.await_args.kwargs["data"]
        assert created["userId"] == "user-new"
        # name defaults to the email local-part
        assert created["name"] == "alice"
        assert created["username"]

    @pytest.mark.asyncio
    async def test_does_not_create_profile_when_one_exists(self):
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-has", email="bob@example.com", name="Bob")

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(user_module.User, "from_db", return_value=MagicMock()),
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)
            mock_prisma.profile.find_unique = AsyncMock(return_value=MagicMock())
            mock_prisma.profile.create = AsyncMock()

            await user_module.get_or_create_user(
                {"sub": "user-has", "email": "bob@example.com"}
            )

        mock_prisma.profile.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_profile_creation_failure_does_not_block_user(self):
        """Profile creation is best-effort: a failure is logged but the user
        is still resolved so login/auth isn't broken."""
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-err", email="carol@example.com", name=None)
        sentinel_user = MagicMock()

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(user_module.User, "from_db", return_value=sentinel_user),
            patch.object(user_module.logger, "warning") as warn_mock,
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)
            mock_prisma.profile.find_unique = AsyncMock(return_value=None)
            mock_prisma.profile.create = AsyncMock(
                side_effect=RuntimeError("db hiccup")
            )

            result = await user_module.get_or_create_user(
                {"sub": "user-err", "email": "carol@example.com"}
            )

        assert result is sentinel_user
        warn_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_profile_create_on_username_collision(self):
        """A UniqueViolationError from a username clash (not a userId race)
        must retry with a fresh handle so the user still gets a Profile."""
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-clash", email="dave@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(user_module.User, "from_db", return_value=MagicMock()),
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)
            # userId never resolves to a Profile (so the clash is on username),
            # and generated usernames pre-check as free.
            mock_prisma.profile.find_unique = AsyncMock(return_value=None)
            # First create() collides on username, the retry succeeds.
            mock_prisma.profile.create = AsyncMock(
                side_effect=[prisma.errors.UniqueViolationError({}), None]
            )

            await user_module.get_or_create_user(
                {"sub": "user-clash", "email": "dave@example.com"}
            )

        assert mock_prisma.profile.create.await_count == 2
