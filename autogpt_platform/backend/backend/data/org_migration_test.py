"""Tests for the personal org bootstrap migration.

Tests the migration logic including slug resolution, idempotency,
and correct data mapping. Uses mocks for Prisma DB calls since the
test infrastructure does not provide a live database connection.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.org_migration import (
    _resolve_unique_slug,
    _sanitize_slug,
    assign_resources_to_workspaces,
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
    mock.orgworkspace.create = AsyncMock(return_value=MagicMock(id="ws-1"))
    mock.orgworkspacemember.create = AsyncMock()
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
        mock_prisma.orgworkspace.create.assert_called_once()
        ws_data = mock_prisma.orgworkspace.create.call_args[1]["data"]
        assert ws_data["name"] == "Default"
        assert ws_data["isDefault"] is True
        assert ws_data["joinPolicy"] == "OPEN"

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
# assign_resources_to_workspaces
# ---------------------------------------------------------------------------


class TestAssignResources:
    @pytest.mark.asyncio
    async def test_updates_all_tables(self, mock_prisma, mocker):
        mocker.patch(
            "backend.data.org_migration._assign_workspace_tenancy",
            new_callable=AsyncMock,
            return_value=10,
        )
        mock_prisma.execute_raw = AsyncMock(return_value=10)

        result = await assign_resources_to_workspaces()

        # 8 tables with workspace + 3 tables org-only = 11 entries
        assert len(result) == 11
        assert result["AgentGraph"] == 10
        assert result["ChatSession"] == 10
        assert result["BuilderSearchHistory"] == 10
        assert result["PendingHumanReview"] == 10
        assert result["StoreListingVersion"] == 10

    @pytest.mark.asyncio
    async def test_zero_updates_still_returns(self, mock_prisma, mocker):
        mocker.patch(
            "backend.data.org_migration._assign_workspace_tenancy",
            new_callable=AsyncMock,
            return_value=0,
        )
        mock_prisma.execute_raw = AsyncMock(return_value=0)
        result = await assign_resources_to_workspaces()
        assert all(v == 0 for v in result.values())


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
        mocker.patch(
            "backend.data.org_migration.assign_resources_to_workspaces",
            new_callable=lambda: lambda: _track(
                calls, "assign_resources", {"AgentGraph": 5}
            ),
        )
        mocker.patch(
            "backend.data.org_migration.migrate_store_listings",
            new_callable=lambda: lambda: _track(calls, "store_listings", 0),
        )
        mocker.patch(
            "backend.data.org_migration.create_store_listing_aliases",
            new_callable=lambda: lambda: _track(calls, "aliases", 0),
        )

        await run_migration()

        assert calls == [
            "create_orgs",
            "balances",
            "credits",
            "assign_resources",
            "store_listings",
            "aliases",
        ]


async def _track(calls: list[str], name: str, result):
    calls.append(name)
    return result
