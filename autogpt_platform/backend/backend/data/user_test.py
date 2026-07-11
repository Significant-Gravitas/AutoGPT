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


class TestTableBackedCredentials:
    """get/set_user_credentials — the IntegrationCredential-backed seam the
    credential store runs on after the blob→table migration."""

    @pytest.mark.asyncio
    async def test_get_decrypts_active_user_rows(self, mocker):
        from unittest.mock import AsyncMock, MagicMock

        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials
        from backend.data.user import get_user_credentials
        from backend.util.encryption import JSONCryptor

        cred = APIKeyCredentials(
            id="cred-1", provider="github", api_key=SecretStr("sk-1"), title="GH"
        )
        row = MagicMock()
        row.id = "cred-1"
        row.encryptedPayload = JSONCryptor().encrypt(cred.model_dump())

        mock_prisma = MagicMock()
        mock_prisma.integrationcredential.find_many = AsyncMock(return_value=[row])
        mocker.patch("backend.data.user.prisma", mock_prisma)

        result = await get_user_credentials("u1")

        assert len(result) == 1
        assert result[0].id == "cred-1"
        assert result[0].api_key.get_secret_value() == "sk-1"
        where = mock_prisma.integrationcredential.find_many.call_args.kwargs["where"]
        assert where == {"ownerType": "USER", "ownerId": "u1", "status": "active"}

    @pytest.mark.asyncio
    async def test_set_updates_existing_creates_new_revokes_missing(self, mocker):
        from unittest.mock import AsyncMock, MagicMock

        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials
        from backend.data.user import set_user_credentials

        kept = APIKeyCredentials(
            id="cred-kept", provider="github", api_key=SecretStr("sk-2"), title="GH"
        )
        new = APIKeyCredentials(
            id="cred-new", provider="notion", api_key=SecretStr("sk-3"), title="N"
        )

        row_kept = MagicMock()
        row_kept.id = "cred-kept"
        row_kept.status = "active"
        row_gone = MagicMock()
        row_gone.id = "cred-gone"
        row_gone.status = "active"

        mock_prisma = MagicMock()
        mock_prisma.integrationcredential.find_many = AsyncMock(
            return_value=[row_kept, row_gone]
        )
        mock_prisma.integrationcredential.update = AsyncMock()
        mock_prisma.integrationcredential.create = AsyncMock()
        mock_prisma.organization.find_first = AsyncMock(
            return_value=MagicMock(id="org-personal")
        )
        mocker.patch("backend.data.user.prisma", mock_prisma)

        await set_user_credentials("u1", [kept, new])

        create_data = mock_prisma.integrationcredential.create.call_args.kwargs["data"]
        assert create_data["id"] == "cred-new"
        assert create_data["organizationId"] == "org-personal"

        update_calls = {
            c.kwargs["where"]["id"]: c.kwargs["data"]
            for c in mock_prisma.integrationcredential.update.call_args_list
        }
        # kept: payload refresh; gone: revoked
        assert "encryptedPayload" in update_calls["cred-kept"]
        assert update_calls["cred-gone"] == {"status": "revoked"}

    @pytest.mark.asyncio
    async def test_set_raises_without_personal_org_for_new_cred(self, mocker):
        from unittest.mock import AsyncMock, MagicMock

        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials
        from backend.data.user import set_user_credentials
        from backend.util.exceptions import DatabaseError

        new = APIKeyCredentials(
            id="cred-new", provider="notion", api_key=SecretStr("sk-3"), title="N"
        )
        mock_prisma = MagicMock()
        mock_prisma.integrationcredential.find_many = AsyncMock(return_value=[])
        mock_prisma.organization.find_first = AsyncMock(return_value=None)
        mocker.patch("backend.data.user.prisma", mock_prisma)

        with pytest.raises(DatabaseError):
            await set_user_credentials("u1", [new])


class TestAccessibleCredentials:
    """get_accessible_credentials — the USER→TEAM→ORG resolution that the
    credential LISTING and execution FETCH paths run on. USER-cred behavior
    stays identical to get_user_credentials; team/org creds are surfaced
    only for live (ACTIVE) members and never written back."""

    @staticmethod
    def _row(cred, owner_type: str):
        from backend.util.encryption import JSONCryptor

        row = MagicMock()
        row.id = cred.id
        row.ownerType = owner_type
        row.encryptedPayload = JSONCryptor().encrypt(cred.model_dump())
        return row

    @pytest.mark.asyncio
    async def test_resolves_user_team_and_org_creds_for_member(self, mocker):
        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials
        from backend.data.user import get_accessible_credentials

        user_cred = APIKeyCredentials(
            id="u", provider="github", api_key=SecretStr("s"), title="U"
        )
        team_cred = APIKeyCredentials(
            id="t", provider="notion", api_key=SecretStr("s"), title="T"
        )
        org_cred = APIKeyCredentials(
            id="o", provider="slant3d", api_key=SecretStr("s"), title="O"
        )

        mock_prisma = MagicMock()
        mock_prisma.teammember.find_many = AsyncMock(
            return_value=[MagicMock(teamId="team-a")]
        )
        mock_prisma.orgmember.find_many = AsyncMock(
            return_value=[MagicMock(orgId="org-1")]
        )
        mock_prisma.integrationcredential.find_many = AsyncMock(
            return_value=[
                self._row(user_cred, "USER"),
                self._row(team_cred, "TEAM"),
                self._row(org_cred, "ORG"),
            ]
        )
        mocker.patch("backend.data.user.prisma", mock_prisma)

        result = await get_accessible_credentials("u1")

        assert {c.id for c in result} == {"u", "t", "o"}
        where = mock_prisma.integrationcredential.find_many.call_args.kwargs["where"]
        assert where["status"] == "active"
        assert {"ownerType": "USER", "ownerId": "u1"} in where["OR"]
        assert {"ownerType": "TEAM", "ownerId": {"in": ["team-a"]}} in where["OR"]
        assert {"ownerType": "ORG", "ownerId": {"in": ["org-1"]}} in where["OR"]

    @pytest.mark.asyncio
    async def test_non_member_cannot_query_other_teams(self, mocker):
        """A user in no teams gets no TEAM clause at all — there is no way
        for the query to return another team's credentials."""
        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials
        from backend.data.user import get_accessible_credentials

        user_cred = APIKeyCredentials(
            id="u", provider="github", api_key=SecretStr("s"), title="U"
        )
        mock_prisma = MagicMock()
        mock_prisma.teammember.find_many = AsyncMock(return_value=[])
        mock_prisma.orgmember.find_many = AsyncMock(
            return_value=[MagicMock(orgId="org-personal")]
        )
        mock_prisma.integrationcredential.find_many = AsyncMock(
            return_value=[self._row(user_cred, "USER")]
        )
        mocker.patch("backend.data.user.prisma", mock_prisma)

        result = await get_accessible_credentials("u1")

        assert [c.id for c in result] == ["u"]
        where = mock_prisma.integrationcredential.find_many.call_args.kwargs["where"]
        owner_types = {clause.get("ownerType") for clause in where["OR"]}
        assert "TEAM" not in owner_types

    @pytest.mark.asyncio
    async def test_left_team_user_loses_team_creds(self, mocker):
        """Someone who has left a team no longer has an ACTIVE membership, so
        the membership query (filtered to ACTIVE) returns nothing and the
        team's credential is never resolved — the check that makes a run
        scheduled pre-departure stop seeing team creds at execution time."""
        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials
        from backend.data.user import get_accessible_credentials

        user_cred = APIKeyCredentials(
            id="u", provider="github", api_key=SecretStr("s"), title="U"
        )
        mock_prisma = MagicMock()
        mock_prisma.teammember.find_many = AsyncMock(return_value=[])
        mock_prisma.orgmember.find_many = AsyncMock(return_value=[])
        mock_prisma.integrationcredential.find_many = AsyncMock(
            return_value=[self._row(user_cred, "USER")]
        )
        mocker.patch("backend.data.user.prisma", mock_prisma)

        result = await get_accessible_credentials("u1")

        assert [c.id for c in result] == ["u"]
        assert (
            mock_prisma.teammember.find_many.call_args.kwargs["where"]["status"]
            == "ACTIVE"
        )
        assert (
            mock_prisma.orgmember.find_many.call_args.kwargs["where"]["status"]
            == "ACTIVE"
        )

    @pytest.mark.asyncio
    async def test_id_collision_prefers_user_over_team(self, mocker):
        """Defensive precedence: on a (practically impossible) id clash the
        USER-owned credential wins over TEAM/ORG, regardless of DB row order."""
        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials
        from backend.data.user import get_accessible_credentials

        user_cred = APIKeyCredentials(
            id="dup", provider="github", api_key=SecretStr("user"), title="U"
        )
        team_cred = APIKeyCredentials(
            id="dup", provider="notion", api_key=SecretStr("team"), title="T"
        )
        mock_prisma = MagicMock()
        mock_prisma.teammember.find_many = AsyncMock(
            return_value=[MagicMock(teamId="team-a")]
        )
        mock_prisma.orgmember.find_many = AsyncMock(return_value=[])
        # TEAM row returned first to prove precedence, not input order, decides.
        mock_prisma.integrationcredential.find_many = AsyncMock(
            return_value=[self._row(team_cred, "TEAM"), self._row(user_cred, "USER")]
        )
        mocker.patch("backend.data.user.prisma", mock_prisma)

        result = await get_accessible_credentials("u1")

        assert len(result) == 1
        assert result[0].id == "dup"
        assert result[0].provider == "github"


class TestGetOrCreateUserProfile:
    """get_or_create_user must guarantee a marketplace Profile exists, since
    the auth.users trigger that used to do this is unreliable."""

    @pytest.fixture(autouse=True)
    def stub_ensure_personal_org(self):
        """Stub the personal-org bootstrap for the Profile-focused tests.

        The real bootstrap hits the DB; these tests only exercise the Profile
        branch. Tests that assert on the bootstrap use the yielded mock.
        """
        with patch.object(
            user_module, "ensure_personal_org", new_callable=AsyncMock
        ) as m:
            yield m

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


class TestGetOrCreateUserPersonalOrg:
    """get_or_create_user must bootstrap a personal org so new sign-ups don't
    hit "No organization context available" on every org-scoped endpoint.

    Unlike the marketplace Profile, this is NOT best-effort: without an org the
    account is unusable, so a bootstrap failure must fail the request loudly.
    """

    @pytest.mark.asyncio
    async def test_bootstraps_personal_org_for_user(self):
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-org", email="erin@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(user_module.User, "from_db", return_value=MagicMock()),
            patch.object(user_module, "_ensure_user_profile", new_callable=AsyncMock),
            patch.object(
                user_module, "ensure_personal_org", new_callable=AsyncMock
            ) as ensure_org,
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)

            await user_module.get_or_create_user(
                {"sub": "user-org", "email": "erin@example.com"}
            )

        ensure_org.assert_awaited_once_with("user-org")

    @pytest.mark.asyncio
    async def test_org_bootstrap_failure_fails_loudly(self):
        """A failed org bootstrap must raise (DatabaseError) — never return a
        bricked account. Contrast with the best-effort Profile branch."""
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-brick", email="frank@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(user_module.User, "from_db", return_value=MagicMock()),
            patch.object(user_module, "_ensure_user_profile", new_callable=AsyncMock),
            patch.object(
                user_module,
                "ensure_personal_org",
                new_callable=AsyncMock,
                side_effect=RuntimeError("org bootstrap exploded"),
            ),
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)

            with pytest.raises(DatabaseError) as exc:
                await user_module.get_or_create_user(
                    {"sub": "user-brick", "email": "frank@example.com"}
                )

        assert "org bootstrap exploded" in str(exc.value)
