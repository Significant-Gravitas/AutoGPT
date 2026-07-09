"""Tests for the personal org bootstrap migration.

Tests the migration logic including slug resolution, idempotency,
and correct data mapping. Uses mocks for Prisma DB calls since the
test infrastructure does not provide a live database connection.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.org_migration import (
    _release_migration_lock,
    _renew_migration_lock,
    _resolve_unique_slug,
    _sanitize_slug,
    assign_resources_to_teams,
    create_orgs_for_existing_users,
    migrate_credit_transactions,
    migrate_org_balances,
    migrate_store_listings,
    run_migration,
)


@pytest.fixture(autouse=True)
def mock_prisma(mocker):
    """Replace the prisma client in org_migration with a full mock."""
    mock = MagicMock()
    # Default: all find_unique calls return None (no collisions)
    mock.organization.find_unique = AsyncMock(return_value=None)
    mock.organizationalias.find_unique = AsyncMock(return_value=None)
    mock.organization.create = AsyncMock(return_value=MagicMock(id="org-1"))
    mock.orgmember.create = AsyncMock()
    mock.team.create = AsyncMock(return_value=MagicMock(id="ws-1"))
    mock.teammember.create = AsyncMock()
    mock.organizationprofile.create = AsyncMock()
    mock.organizationseatassignment.create = AsyncMock()
    mock.query_raw = AsyncMock(return_value=[])
    mock.execute_raw = AsyncMock(return_value=0)
    mocker.patch("backend.data.org_migration.prisma", mock)
    return mock


# ---------------------------------------------------------------------------
# _sanitize_slug
# ---------------------------------------------------------------------------


class TestSanitizeSlug:
    def test_lowercase_and_hyphens(self):
        assert _sanitize_slug("Hello World") == "hello-world"

    def test_strips_special_chars(self):
        assert _sanitize_slug("user@name!#$%") == "user-name"

    def test_collapses_multiple_hyphens(self):
        assert _sanitize_slug("a---b") == "a-b"

    def test_strips_leading_trailing_hyphens(self):
        assert _sanitize_slug("-hello-") == "hello"

    def test_empty_string_returns_user(self):
        assert _sanitize_slug("") == "user"

    def test_only_special_chars_returns_user(self):
        assert _sanitize_slug("@#$%") == "user"

    def test_numeric_slug(self):
        assert _sanitize_slug("12345") == "12345"

    def test_preserves_hyphens(self):
        assert _sanitize_slug("my-cool-agent") == "my-cool-agent"

    def test_unicode_stripped(self):
        assert _sanitize_slug("caf\u00e9-latt\u00e9") == "caf-latt"

    def test_whitespace_only(self):
        assert _sanitize_slug("   ") == "user"


# ---------------------------------------------------------------------------
# _resolve_unique_slug
# ---------------------------------------------------------------------------


class TestResolveUniqueSlug:
    @pytest.mark.asyncio
    async def test_slug_available_returns_as_is(self, mock_prisma):
        result = await _resolve_unique_slug("my-org")
        assert result == "my-org"

    @pytest.mark.asyncio
    async def test_slug_taken_by_org_gets_suffix(self, mock_prisma):
        async def org_find(where):
            slug = where.get("slug", "")
            if slug == "taken":
                return MagicMock(id="existing-org")
            return None

        mock_prisma.organization.find_unique = AsyncMock(side_effect=org_find)

        result = await _resolve_unique_slug("taken")
        assert result == "taken-1"

    @pytest.mark.asyncio
    async def test_slug_taken_by_alias_gets_suffix(self, mock_prisma):
        async def alias_find(where):
            slug = where.get("aliasSlug", "")
            if slug == "aliased":
                return MagicMock()
            return None

        mock_prisma.organizationalias.find_unique = AsyncMock(side_effect=alias_find)

        result = await _resolve_unique_slug("aliased")
        assert result == "aliased-1"

    @pytest.mark.asyncio
    async def test_multiple_collisions_increments(self, mock_prisma):
        async def org_find(where):
            slug = where.get("slug", "")
            if slug in ("x", "x-1", "x-2"):
                return MagicMock(id="existing")
            return None

        mock_prisma.organization.find_unique = AsyncMock(side_effect=org_find)

        result = await _resolve_unique_slug("x")
        assert result == "x-3"


# ---------------------------------------------------------------------------
# create_orgs_for_existing_users
# ---------------------------------------------------------------------------


class TestCreateOrgsForExistingUsers:
    @pytest.mark.asyncio
    async def test_no_users_without_org_is_noop(self, mock_prisma):
        result = await create_orgs_for_existing_users()
        assert result == 0

    @pytest.mark.asyncio
    async def test_user_with_profile_gets_profile_username_slug(self, mock_prisma):
        mock_prisma.query_raw = AsyncMock(
            return_value=[
                {
                    "id": "user-1",
                    "email": "alice@example.com",
                    "name": "Alice",
                    "stripeCustomerId": "cus_123",
                    "topUpConfig": None,
                    "profile_username": "alice",
                    "profile_name": "Alice Smith",
                    "profile_description": "A developer",
                    "profile_avatar_url": "https://example.com/avatar.png",
                    "profile_links": ["https://github.com/alice"],
                },
            ]
        )

        result = await create_orgs_for_existing_users()
        assert result == 1

        # Verify org was created with profile-derived slug
        mock_prisma.organization.create.assert_called_once()
        create_data = mock_prisma.organization.create.call_args[1]["data"]
        assert create_data["slug"] == "alice"
        assert create_data["name"] == "Alice Smith"
        assert create_data["isPersonal"] is True
        assert create_data["stripeCustomerId"] == "cus_123"
        assert create_data["bootstrapUserId"] == "user-1"

        # Verify workspace created
        mock_prisma.team.create.assert_called_once()
        ws_data = mock_prisma.team.create.call_args[1]["data"]
        assert ws_data["name"] == "Default"
        assert ws_data["isDefault"] is True
        assert ws_data["joinPolicy"] == "OPEN"

    @pytest.mark.asyncio
    async def test_json_profile_fields_are_wrapped_for_prisma(self, mock_prisma):
        """Regression (live-repro'd on real seeded data): Profile.links and
        User.topUpConfig are Json columns; query_raw returns them as parsed
        Python objects, but prisma create() rejects a raw list/dict — it
        must be re-wrapped as prisma.Json. A pure AsyncMock accepts the raw
        value, so this asserts the WRAPPING, not just that create was called.
        Without the wrap, the whole startup org migration crashes for any
        user who has a populated store profile."""
        import prisma as prisma_pkg

        mock_prisma.query_raw = AsyncMock(
            return_value=[
                {
                    "id": "user-1",
                    "email": "alice@example.com",
                    "name": "Alice",
                    "stripeCustomerId": "cus_123",
                    "topUpConfig": {"threshold": 100, "amount": 500},
                    "profile_username": "alice",
                    "profile_name": "Alice Smith",
                    "profile_description": "A developer",
                    "profile_avatar_url": "https://example.com/avatar.png",
                    "profile_links": ["https://github.com/alice"],
                },
            ]
        )

        result = await create_orgs_for_existing_users()
        assert result == 1

        org_data = mock_prisma.organization.create.call_args[1]["data"]
        assert isinstance(org_data["topUpConfig"], prisma_pkg.Json)
        assert not isinstance(org_data["topUpConfig"], (dict, list))

        profile_data = mock_prisma.organizationprofile.create.call_args[1]["data"]
        assert isinstance(profile_data["socialLinks"], prisma_pkg.Json)
        assert not isinstance(profile_data["socialLinks"], (dict, list))

    @pytest.mark.asyncio
    async def test_user_without_profile_uses_email_slug(self, mock_prisma):
        mock_prisma.query_raw = AsyncMock(
            return_value=[
                {
                    "id": "user-2",
                    "email": "bob@company.org",
                    "name": None,
                    "stripeCustomerId": None,
                    "topUpConfig": None,
                    "profile_username": None,
                    "profile_name": None,
                    "profile_description": None,
                    "profile_avatar_url": None,
                    "profile_links": None,
                },
            ]
        )

        result = await create_orgs_for_existing_users()
        assert result == 1

        create_data = mock_prisma.organization.create.call_args[1]["data"]
        assert create_data["slug"] == "bob"
        assert create_data["name"] == "bob"

    @pytest.mark.asyncio
    async def test_user_with_name_no_profile_uses_name_slug(self, mock_prisma):
        mock_prisma.query_raw = AsyncMock(
            return_value=[
                {
                    "id": "user-3",
                    "email": "charlie@example.com",
                    "name": "Charlie Brown",
                    "stripeCustomerId": None,
                    "topUpConfig": None,
                    "profile_username": None,
                    "profile_name": None,
                    "profile_description": None,
                    "profile_avatar_url": None,
                    "profile_links": None,
                },
            ]
        )

        result = await create_orgs_for_existing_users()
        assert result == 1

        create_data = mock_prisma.organization.create.call_args[1]["data"]
        assert create_data["slug"] == "charlie-brown"
        assert create_data["name"] == "Charlie Brown"

    @pytest.mark.asyncio
    async def test_user_with_empty_email_uses_id_slug(self, mock_prisma):
        mock_prisma.query_raw = AsyncMock(
            return_value=[
                {
                    "id": "abcdef12-3456-7890-abcd-ef1234567890",
                    "email": "",
                    "name": None,
                    "stripeCustomerId": None,
                    "topUpConfig": None,
                    "profile_username": None,
                    "profile_name": None,
                    "profile_description": None,
                    "profile_avatar_url": None,
                    "profile_links": None,
                },
            ]
        )

        result = await create_orgs_for_existing_users()
        assert result == 1

        create_data = mock_prisma.organization.create.call_args[1]["data"]
        assert create_data["slug"] == "user-abcdef12"

    @pytest.mark.asyncio
    async def test_stripe_customer_id_included_when_present(self, mock_prisma):
        mock_prisma.query_raw = AsyncMock(
            return_value=[
                {
                    "id": "user-stripe",
                    "email": "stripe@test.com",
                    "name": "Stripe User",
                    "stripeCustomerId": "cus_abc123",
                    # query_raw hands back Json columns as parsed Python
                    # objects (see test_json_profile_fields_are_wrapped...).
                    "topUpConfig": {"amount": 1000},
                    "profile_username": "stripeuser",
                    "profile_name": "Stripe User",
                    "profile_description": None,
                    "profile_avatar_url": None,
                    "profile_links": None,
                },
            ]
        )

        result = await create_orgs_for_existing_users()
        assert result == 1

        import prisma as prisma_pkg

        create_data = mock_prisma.organization.create.call_args[1]["data"]
        assert create_data["stripeCustomerId"] == "cus_abc123"
        # topUpConfig re-wrapped for prisma create (not the raw dict).
        assert isinstance(create_data["topUpConfig"], prisma_pkg.Json)

    @pytest.mark.asyncio
    async def test_stripe_fields_omitted_when_none(self, mock_prisma):
        mock_prisma.query_raw = AsyncMock(
            return_value=[
                {
                    "id": "user-no-stripe",
                    "email": "nostripe@test.com",
                    "name": None,
                    "stripeCustomerId": None,
                    "topUpConfig": None,
                    "profile_username": "nostripe",
                    "profile_name": None,
                    "profile_description": None,
                    "profile_avatar_url": None,
                    "profile_links": None,
                },
            ]
        )

        result = await create_orgs_for_existing_users()
        assert result == 1

        create_data = mock_prisma.organization.create.call_args[1]["data"]
        assert "stripeCustomerId" not in create_data
        assert "topUpConfig" not in create_data

    @pytest.mark.asyncio
    async def test_org_profile_omits_none_optional_fields(self, mock_prisma):
        """Profile creation should not pass None for optional JSON fields."""
        mock_prisma.query_raw = AsyncMock(
            return_value=[
                {
                    "id": "user-minimal",
                    "email": "minimal@test.com",
                    "name": "Min",
                    "stripeCustomerId": None,
                    "topUpConfig": None,
                    "profile_username": "minimal",
                    "profile_name": "Min",
                    "profile_description": None,
                    "profile_avatar_url": None,
                    "profile_links": None,
                },
            ]
        )

        await create_orgs_for_existing_users()

        profile_data = mock_prisma.organizationprofile.create.call_args[1]["data"]
        assert "avatarUrl" not in profile_data
        assert "bio" not in profile_data
        assert "socialLinks" not in profile_data
        assert profile_data["username"] == "minimal"
        assert profile_data["displayName"] == "Min"

    @pytest.mark.asyncio
    async def test_creates_all_required_records(self, mock_prisma):
        """Verify the full set of records created per user."""
        mock_prisma.query_raw = AsyncMock(
            return_value=[
                {
                    "id": "user-full",
                    "email": "full@test.com",
                    "name": "Full User",
                    "stripeCustomerId": None,
                    "topUpConfig": None,
                    "profile_username": "fulluser",
                    "profile_name": "Full User",
                    "profile_description": "Bio here",
                    "profile_avatar_url": "https://example.com/avatar.png",
                    "profile_links": ["https://github.com/fulluser"],
                },
            ]
        )

        await create_orgs_for_existing_users()

        # Verify all 6 records created
        mock_prisma.organization.create.assert_called_once()
        mock_prisma.orgmember.create.assert_called_once()
        mock_prisma.team.create.assert_called_once()
        mock_prisma.teammember.create.assert_called_once()
        mock_prisma.organizationprofile.create.assert_called_once()
        mock_prisma.organizationseatassignment.create.assert_called_once()

        # Verify OrgMember is owner+admin
        member_data = mock_prisma.orgmember.create.call_args[1]["data"]
        assert member_data["isOwner"] is True
        assert member_data["isAdmin"] is True

        # Verify workspace is default+open
        ws_data = mock_prisma.team.create.call_args[1]["data"]
        assert ws_data["isDefault"] is True
        assert ws_data["joinPolicy"] == "OPEN"

        # Verify seat is FREE+ACTIVE
        seat_data = mock_prisma.organizationseatassignment.create.call_args[1]["data"]
        assert seat_data["seatType"] == "FREE"
        assert seat_data["status"] == "ACTIVE"


# ---------------------------------------------------------------------------
# create_personal_org (single-user bootstrap)
# ---------------------------------------------------------------------------


class _FakeTx:
    """Records the create() calls a transaction body makes, per model."""

    def __init__(self):
        self.organization = MagicMock()
        self.organization.create = AsyncMock(return_value=MagicMock(id="org-new"))
        self.orgmember = MagicMock(create=AsyncMock())
        self.team = MagicMock()
        self.team.create = AsyncMock(return_value=MagicMock(id="ws-new"))
        self.teammember = MagicMock(create=AsyncMock())
        self.organizationprofile = MagicMock(create=AsyncMock())
        self.organizationseatassignment = MagicMock(create=AsyncMock())
        self.orgbalance = MagicMock(create=AsyncMock())


@pytest.fixture
def fake_tx(mocker):
    """Patch org_migration.transaction to yield a records-capturing fake tx."""
    tx = _FakeTx()

    class _CM:
        async def __aenter__(self):
            return tx

        async def __aexit__(self, *exc):
            return False

    mocker.patch("backend.data.org_migration.transaction", lambda *a, **k: _CM())
    return tx


class TestCreatePersonalOrg:
    @pytest.mark.asyncio
    async def test_creates_full_record_shape_in_one_transaction(
        self, mock_prisma, fake_tx
    ):
        from backend.data.org_migration import create_personal_org

        org = await create_personal_org("user-9", "alice", "Alice")

        assert org is fake_tx.organization.create.return_value

        org_data = fake_tx.organization.create.call_args[1]["data"]
        assert org_data["isPersonal"] is True
        assert org_data["bootstrapUserId"] == "user-9"
        assert org_data["slug"] == "alice"

        member_data = fake_tx.orgmember.create.call_args[1]["data"]
        assert member_data["isOwner"] is True
        assert member_data["isAdmin"] is True
        assert member_data["status"] == "ACTIVE"
        assert member_data["userId"] == "user-9"

        ws_data = fake_tx.team.create.call_args[1]["data"]
        assert ws_data["isDefault"] is True
        assert ws_data["joinPolicy"] == "OPEN"

        # A default team member, a profile, a FREE seat, and a zero balance
        # must all be created in the same transaction.
        fake_tx.teammember.create.assert_awaited_once()
        fake_tx.organizationprofile.create.assert_awaited_once()
        seat_data = fake_tx.organizationseatassignment.create.call_args[1]["data"]
        assert seat_data["seatType"] == "FREE"
        balance_data = fake_tx.orgbalance.create.call_args[1]["data"]
        assert balance_data["balance"] == 0

    @pytest.mark.asyncio
    async def test_resolves_slug_collision_before_creating(self, mock_prisma, fake_tx):
        from backend.data.org_migration import create_personal_org

        async def org_find(where):
            return MagicMock() if where.get("slug") == "taken" else None

        mock_prisma.organization.find_unique = AsyncMock(side_effect=org_find)

        await create_personal_org("user-9", "taken", "Taken")

        assert fake_tx.organization.create.call_args[1]["data"]["slug"] == "taken-1"


# ---------------------------------------------------------------------------
# _derive_personal_org_identity
# ---------------------------------------------------------------------------


def _mock_named(name=None, **attrs):
    """MagicMock with a real ``name`` attribute (constructor name= is special)."""
    m = MagicMock(**attrs)
    m.name = name
    return m


class TestDerivePersonalOrgIdentity:
    @pytest.mark.asyncio
    async def test_prefers_profile_username_then_profile_name(self, mock_prisma):
        from backend.data.org_migration import _derive_personal_org_identity

        mock_prisma.user.find_unique = AsyncMock(
            return_value=_mock_named(name="User Name", email="a@b.com")
        )
        mock_prisma.profile.find_unique = AsyncMock(
            return_value=_mock_named(name="Cool Person", username="CoolHandle")
        )

        slug_base, display_name = await _derive_personal_org_identity("user-1")

        assert slug_base == "coolhandle"
        assert display_name == "Cool Person"

    @pytest.mark.asyncio
    async def test_falls_back_to_email_local_part_without_profile(self, mock_prisma):
        from backend.data.org_migration import _derive_personal_org_identity

        mock_prisma.user.find_unique = AsyncMock(
            return_value=_mock_named(email="bob@company.org")
        )
        mock_prisma.profile.find_unique = AsyncMock(return_value=None)

        slug_base, display_name = await _derive_personal_org_identity("user-2")

        assert slug_base == "bob"
        assert display_name == "bob"


# ---------------------------------------------------------------------------
# ensure_personal_org (idempotent, race-safe sign-up bootstrap)
# ---------------------------------------------------------------------------


class TestEnsurePersonalOrg:
    @pytest.mark.asyncio
    async def test_creates_org_when_user_has_none(self, mock_prisma, mocker):
        from backend.data import org_migration

        mock_prisma.orgmember.find_first = AsyncMock(return_value=None)
        mock_prisma.user.find_unique = AsyncMock(
            return_value=_mock_named(email="new@user.com")
        )
        mock_prisma.profile.find_unique = AsyncMock(return_value=None)
        create = mocker.patch.object(
            org_migration, "create_personal_org", new_callable=AsyncMock
        )

        await org_migration.ensure_personal_org("user-new")

        create.assert_awaited_once()
        assert create.await_args.args[0] == "user-new"
        assert create.await_args.args[1] == "new"  # email local-part slug

    @pytest.mark.asyncio
    async def test_is_noop_when_user_already_owns_personal_org(
        self, mock_prisma, mocker
    ):
        from backend.data import org_migration

        mock_prisma.orgmember.find_first = AsyncMock(return_value=MagicMock())
        create = mocker.patch.object(
            org_migration, "create_personal_org", new_callable=AsyncMock
        )

        await org_migration.ensure_personal_org("user-has-org")

        create.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_race_does_not_create_second_org(
        self, mock_prisma, mocker
    ):
        """Two first-requests must not create two personal orgs: the loser hits
        a slug UniqueViolation, re-checks, finds the winner's org, and returns."""
        from prisma.errors import UniqueViolationError

        from backend.data import org_migration

        # First find_first: no org (pre-create). Second (after violation): the
        # concurrent winner's org now exists.
        mock_prisma.orgmember.find_first = AsyncMock(side_effect=[None, MagicMock()])
        mock_prisma.user.find_unique = AsyncMock(
            return_value=_mock_named(email="racer@user.com")
        )
        mock_prisma.profile.find_unique = AsyncMock(return_value=None)
        create = mocker.patch.object(
            org_migration,
            "create_personal_org",
            new_callable=AsyncMock,
            side_effect=UniqueViolationError({}),
        )

        await org_migration.ensure_personal_org("user-racer")

        # Only one create attempt — the loser did not retry into a 2nd org.
        create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_on_different_user_slug_collision(self, mock_prisma, mocker):
        """A slug clash with a *different* user (re-check still finds no org for
        us) must retry until the create succeeds with a fresh slug."""
        from prisma.errors import UniqueViolationError

        from backend.data import org_migration

        mock_prisma.orgmember.find_first = AsyncMock(return_value=None)
        mock_prisma.user.find_unique = AsyncMock(
            return_value=_mock_named(email="dup@user.com")
        )
        mock_prisma.profile.find_unique = AsyncMock(return_value=None)
        create = mocker.patch.object(
            org_migration,
            "create_personal_org",
            new_callable=AsyncMock,
            side_effect=[UniqueViolationError({}), None],
        )

        await org_migration.ensure_personal_org("user-dup")

        assert create.await_count == 2

    @pytest.mark.asyncio
    async def test_raises_when_bootstrap_never_succeeds(self, mock_prisma, mocker):
        """Persistent failure must raise (fail loud) — never a silent no-op."""
        from prisma.errors import UniqueViolationError

        from backend.data import org_migration

        mock_prisma.orgmember.find_first = AsyncMock(return_value=None)
        mock_prisma.user.find_unique = AsyncMock(
            return_value=_mock_named(email="stuck@user.com")
        )
        mock_prisma.profile.find_unique = AsyncMock(return_value=None)
        mocker.patch.object(
            org_migration,
            "create_personal_org",
            new_callable=AsyncMock,
            side_effect=UniqueViolationError({}),
        )

        with pytest.raises(RuntimeError, match="no usable organization context"):
            await org_migration.ensure_personal_org("user-stuck")


# ---------------------------------------------------------------------------
# migrate_org_balances
# ---------------------------------------------------------------------------


class TestMigrateOrgBalances:
    @pytest.mark.asyncio
    async def test_returns_count(self, mock_prisma):
        mock_prisma.execute_raw = AsyncMock(return_value=5)
        result = await migrate_org_balances()
        assert result == 5


# ---------------------------------------------------------------------------
# migrate_credit_transactions
# ---------------------------------------------------------------------------


class TestMigrateCreditTransactions:
    @pytest.mark.asyncio
    async def test_returns_count(self, mock_prisma):
        mock_prisma.execute_raw = AsyncMock(return_value=42)
        result = await migrate_credit_transactions()
        assert result == 42


# ---------------------------------------------------------------------------
# assign_resources_to_teams
# ---------------------------------------------------------------------------


class TestAssignResources:
    @pytest.mark.asyncio
    async def test_updates_all_tables(self, mock_prisma, mocker):
        mocker.patch(
            "backend.data.org_migration._assign_team_tenancy",
            new_callable=AsyncMock,
            return_value=10,
        )
        mocker.patch(
            "backend.data.org_migration._assign_team_tenancy_batched",
            new_callable=AsyncMock,
            return_value=10,
        )
        mock_prisma.execute_raw = AsyncMock(return_value=10)

        result = await assign_resources_to_teams()

        # 9 tables with workspace + 3 tables org-only = 12 entries
        assert len(result) == 12
        assert result["AgentGraph"] == 10
        assert result["ChatSession"] == 10
        assert result["UserNotificationBatch"] == 10
        assert result["BuilderSearchHistory"] == 10
        assert result["PendingHumanReview"] == 10
        assert result["StoreListingVersion"] == 10

    @pytest.mark.asyncio
    async def test_zero_updates_still_returns(self, mock_prisma, mocker):
        mocker.patch(
            "backend.data.org_migration._assign_team_tenancy",
            new_callable=AsyncMock,
            return_value=0,
        )
        mocker.patch(
            "backend.data.org_migration._assign_team_tenancy_batched",
            new_callable=AsyncMock,
            return_value=0,
        )
        mock_prisma.execute_raw = AsyncMock(return_value=0)
        result = await assign_resources_to_teams()
        assert all(v == 0 for v in result.values())

    @pytest.mark.asyncio
    async def test_big_tables_use_batched_updates(self, mock_prisma, mocker):
        """AgentGraphExecution and ChatSession must go through the batched
        path — a single full-table UPDATE exceeds the platform statement
        timeout on production-sized data (this took down the dev deploy)."""
        single = mocker.patch(
            "backend.data.org_migration._assign_team_tenancy",
            new_callable=AsyncMock,
            return_value=0,
        )
        batched = mocker.patch(
            "backend.data.org_migration._assign_team_tenancy_batched",
            new_callable=AsyncMock,
            return_value=0,
        )
        mock_prisma.execute_raw = AsyncMock(return_value=0)

        await assign_resources_to_teams()

        batched_tables = {call.args[0] for call in batched.await_args_list}
        assert batched_tables == {"AgentGraphExecution", "ChatSession"}
        single_sql = " ".join(call.args[0] for call in single.await_args_list)
        assert '"AgentGraphExecution"' not in single_sql
        assert 'UPDATE "ChatSession"' not in single_sql

    @pytest.mark.asyncio
    async def test_renews_after_each_assignment_statement(self, mock_prisma):
        mock_prisma.execute_raw = AsyncMock(return_value=0)
        renew_lock = AsyncMock()

        result = await assign_resources_to_teams(renew_lock=renew_lock)

        assert len(result) == 12
        assert mock_prisma.execute_raw.await_count == 12
        assert renew_lock.await_count == 12


class TestBatchedTenancy:
    @pytest.mark.asyncio
    async def test_loops_until_no_rows_remain(self, mock_prisma):
        from backend.data.org_migration import _assign_team_tenancy_batched

        mock_prisma.execute_raw = AsyncMock(side_effect=[3, 2, 0])

        total = await _assign_team_tenancy_batched(
            "SomeTable", 'UPDATE "SomeTable" SET x = 1'
        )

        assert total == 5
        assert mock_prisma.execute_raw.await_count == 3

    @pytest.mark.asyncio
    async def test_renews_lock_after_each_batch_statement(self, mock_prisma):
        from backend.data.org_migration import _assign_team_tenancy_batched

        mock_prisma.execute_raw = AsyncMock(side_effect=[3, 2, 0])
        renew_lock = AsyncMock()

        total = await _assign_team_tenancy_batched(
            "SomeTable", 'UPDATE "SomeTable" SET x = 1', renew_lock=renew_lock
        )

        assert total == 5
        assert renew_lock.await_count == 3

    @pytest.mark.asyncio
    async def test_empty_table_returns_zero_after_one_probe(self, mock_prisma):
        from backend.data.org_migration import _assign_team_tenancy_batched

        mock_prisma.execute_raw = AsyncMock(return_value=0)

        total = await _assign_team_tenancy_batched(
            "SomeTable", 'UPDATE "SomeTable" SET x = 1'
        )

        assert total == 0
        assert mock_prisma.execute_raw.await_count == 1


class TestMigrationLock:
    @pytest.mark.asyncio
    async def test_renew_extends_owned_lock(self):
        redis = AsyncMock()
        redis.execute_command = AsyncMock(return_value=1)

        await _renew_migration_lock(redis, "lock-key", "token", 300)

        redis.execute_command.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_renew_fails_loud_when_lock_is_lost(self):
        redis = AsyncMock()
        redis.execute_command = AsyncMock(return_value=0)

        with pytest.raises(RuntimeError, match="lost the distributed bootstrap lock"):
            await _renew_migration_lock(redis, "lock-key", "token", 300)

    @pytest.mark.asyncio
    async def test_release_uses_token_safe_script(self):
        redis = AsyncMock()
        redis.execute_command = AsyncMock(return_value=0)

        released = await _release_migration_lock(redis, "lock-key", "token")

        assert released is False
        redis.delete.assert_not_called()


# ---------------------------------------------------------------------------
# migrate_store_listings
# ---------------------------------------------------------------------------


class TestMigrateStoreListings:
    @pytest.mark.asyncio
    async def test_returns_count(self, mock_prisma):
        mock_prisma.execute_raw = AsyncMock(return_value=3)
        result = await migrate_store_listings()
        assert result == 3


# ---------------------------------------------------------------------------
# run_migration (orchestrator)
# ---------------------------------------------------------------------------


class TestRunMigration:
    @pytest.mark.asyncio
    async def test_calls_all_steps_in_order(self, mocker):
        calls: list[str] = []
        redis = AsyncMock()
        redis.set = AsyncMock(return_value=True)
        redis.execute_command = AsyncMock(return_value=1)

        mocker.patch(
            "backend.data.redis_client.get_redis_async",
            new_callable=AsyncMock,
            return_value=redis,
        )
        mocker.patch("backend.data.org_migration.uuid4", return_value="lock-token")

        mocker.patch(
            "backend.data.org_migration.create_orgs_for_existing_users",
            new_callable=lambda: lambda: _track(calls, "create_orgs", 1),
        )
        mocker.patch(
            "backend.data.org_migration.migrate_org_balances",
            new_callable=lambda: lambda: _track(calls, "balances", 0),
        )
        mocker.patch(
            "backend.data.org_migration.migrate_credit_transactions",
            new_callable=lambda: lambda: _track(calls, "credits", 0),
        )

        async def assign_resources(renew_lock=None):
            if renew_lock is not None:
                await renew_lock()
            return await _track(calls, "assign_resources", {"AgentGraph": 5})

        mocker.patch(
            "backend.data.org_migration.assign_resources_to_teams",
            new=assign_resources,
        )
        mocker.patch(
            "backend.data.org_migration.migrate_store_listings",
            new_callable=lambda: lambda: _track(calls, "store_listings", 0),
        )
        mocker.patch(
            "backend.data.org_migration.create_store_listing_aliases",
            new_callable=lambda: lambda: _track(calls, "aliases", 0),
        )
        mocker.patch(
            "backend.data.org_migration.migrate_credentials_to_table",
            new_callable=lambda: lambda: _track(calls, "credentials", 0),
        )

        await run_migration()

        redis.set.assert_awaited_once_with(
            "org-migration-bootstrap-lock", "lock-token", nx=True, ex=300
        )
        assert redis.execute_command.await_count == 9
        assert calls == [
            "create_orgs",
            "balances",
            "credits",
            "assign_resources",
            "store_listings",
            "aliases",
            "credentials",
        ]


async def _track(calls: list[str], name: str, result):
    calls.append(name)
    return result


class TestCredentialMigration:
    """Blob → IntegrationCredential table copy (big-bang, blob preserved)."""

    @pytest.mark.asyncio
    async def test_migrates_blob_credentials_to_rows(self, mock_prisma):
        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials, UserIntegrations
        from backend.data.org_migration import migrate_credentials_to_table
        from backend.util.encryption import JSONCryptor

        cred = APIKeyCredentials(
            id="cred-1",
            provider="github",
            api_key=SecretStr("sk-live-1"),
            title="GitHub",
        )
        blob = JSONCryptor().encrypt(
            UserIntegrations(credentials=[cred]).model_dump(exclude_none=True)
        )
        user = MagicMock()
        user.id = "u1"
        user.integrations = blob
        mock_prisma.user.find_many = AsyncMock(return_value=[user])
        mock_prisma.organization.find_first = AsyncMock(
            return_value=MagicMock(id="org-personal")
        )
        mock_prisma.integrationcredential.find_many = AsyncMock(return_value=[])
        mock_prisma.integrationcredential.create = AsyncMock()

        created = await migrate_credentials_to_table()

        assert created == 1
        data = mock_prisma.integrationcredential.create.call_args.kwargs["data"]
        # Row id MUST reuse the credential UUID so graph credentials_id
        # references keep resolving after the store switches to the table.
        assert data["id"] == "cred-1"
        assert data["organizationId"] == "org-personal"
        assert data["ownerType"] == "USER"
        assert data["ownerId"] == "u1"
        assert data["provider"] == "github"
        payload = JSONCryptor().decrypt(data["encryptedPayload"])
        assert payload["api_key"] == "sk-live-1"
        assert payload["id"] == "cred-1"

    @pytest.mark.asyncio
    async def test_skips_already_migrated_credentials(self, mock_prisma):
        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials, UserIntegrations
        from backend.data.org_migration import migrate_credentials_to_table
        from backend.util.encryption import JSONCryptor

        cred = APIKeyCredentials(
            id="cred-1",
            provider="github",
            api_key=SecretStr("sk-live-1"),
            title="GitHub",
        )
        blob = JSONCryptor().encrypt(
            UserIntegrations(credentials=[cred]).model_dump(exclude_none=True)
        )
        user = MagicMock()
        user.id = "u1"
        user.integrations = blob
        mock_prisma.user.find_many = AsyncMock(return_value=[user])
        mock_prisma.organization.find_first = AsyncMock(
            return_value=MagicMock(id="org-personal")
        )
        existing = MagicMock()
        existing.id = "cred-1"
        mock_prisma.integrationcredential.find_many = AsyncMock(return_value=[existing])
        mock_prisma.integrationcredential.create = AsyncMock()

        created = await migrate_credentials_to_table()

        assert created == 0
        mock_prisma.integrationcredential.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_defers_user_without_personal_org(self, mock_prisma):
        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials, UserIntegrations
        from backend.data.org_migration import migrate_credentials_to_table
        from backend.util.encryption import JSONCryptor

        cred = APIKeyCredentials(
            id="cred-1",
            provider="github",
            api_key=SecretStr("sk-live-1"),
            title="GitHub",
        )
        blob = JSONCryptor().encrypt(
            UserIntegrations(credentials=[cred]).model_dump(exclude_none=True)
        )
        user = MagicMock()
        user.id = "u1"
        user.integrations = blob
        mock_prisma.user.find_many = AsyncMock(return_value=[user])
        mock_prisma.organization.find_first = AsyncMock(return_value=None)
        mock_prisma.integrationcredential.create = AsyncMock()

        created = await migrate_credentials_to_table()

        assert created == 0
        mock_prisma.integrationcredential.create.assert_not_called()
