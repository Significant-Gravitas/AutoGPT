"""Tests for grant access resolution (share-with-team, teams-only v1)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import GrantCapability, GrantCredentialMode, GrantPrincipalType

from backend.data.grants import (
    GrantPrincipalNotSupportedError,
    grant_covers_version,
    resolve_execution_credentials_owner,
    resolve_graph_grant,
)


def _grant(
    *,
    principal_type=GrantPrincipalType.TEAM,
    principal_id="team-1",
    capability=GrantCapability.EXECUTE,
    version=3,
    follow_latest=False,
    grant_id="grant-1",
):
    row = MagicMock()
    row.id = grant_id
    row.principalType = principal_type
    row.principalId = principal_id
    row.capability = capability
    row.agentGraphVersion = version
    row.followLatest = follow_latest
    return row


@pytest.fixture
def mock_prisma(mocker):
    mock = MagicMock()
    mock.agentgraphgrant.find_many = AsyncMock(return_value=[])
    mock.teammember.find_first = AsyncMock(return_value=None)
    mocker.patch("backend.data.grants.prisma", mock)
    return mock


class TestResolveGraphGrant:
    @pytest.mark.asyncio
    async def test_no_grants_returns_none(self, mock_prisma):
        assert (
            await resolve_graph_grant("u1", "g1", capability=GrantCapability.EXECUTE)
            is None
        )
        mock_prisma.teammember.find_first.assert_not_called()

    @pytest.mark.asyncio
    async def test_active_team_member_gets_grant(self, mock_prisma):
        grant = _grant()
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[grant])
        membership = MagicMock()
        membership.teamId = "team-1"
        mock_prisma.teammember.find_first = AsyncMock(return_value=membership)

        result = await resolve_graph_grant(
            "u1", "g1", capability=GrantCapability.EXECUTE
        )

        assert result is grant
        where = mock_prisma.teammember.find_first.call_args.kwargs["where"]
        assert where["status"] == "ACTIVE"
        assert where["teamId"] == {"in": ["team-1"]}
        # Archived workspaces must not keep granting access.
        assert where["Team"] == {"is": {"archivedAt": None}}

    @pytest.mark.asyncio
    async def test_non_member_gets_nothing(self, mock_prisma):
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[_grant()])
        mock_prisma.teammember.find_first = AsyncMock(return_value=None)

        assert (
            await resolve_graph_grant("u1", "g1", capability=GrantCapability.EXECUTE)
            is None
        )

    @pytest.mark.asyncio
    async def test_view_check_satisfied_by_view_or_execute_grant(self, mock_prisma):
        view_grant = _grant(capability=GrantCapability.VIEW)
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[view_grant])
        membership = MagicMock()
        membership.teamId = "team-1"
        mock_prisma.teammember.find_first = AsyncMock(return_value=membership)

        assert (
            await resolve_graph_grant("u1", "g1", capability=GrantCapability.VIEW)
            is view_grant
        )

    @pytest.mark.asyncio
    async def test_execute_check_rejects_view_only_grant(self, mock_prisma):
        mock_prisma.agentgraphgrant.find_many = AsyncMock(
            return_value=[_grant(capability=GrantCapability.VIEW)]
        )

        assert (
            await resolve_graph_grant("u1", "g1", capability=GrantCapability.EXECUTE)
            is None
        )
        mock_prisma.teammember.find_first.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_team_principal_raises_loudly(self, mock_prisma):
        """A USER-principal row can only exist by bypassing the grants API;
        enforcement must fail loudly, never silently skip it."""
        mock_prisma.agentgraphgrant.find_many = AsyncMock(
            return_value=[
                _grant(),
                _grant(
                    principal_type=GrantPrincipalType.USER,
                    principal_id="u2",
                    grant_id="grant-2",
                ),
            ]
        )

        with pytest.raises(GrantPrincipalNotSupportedError, match="grant-2"):
            await resolve_graph_grant("u1", "g1", capability=GrantCapability.EXECUTE)

    @pytest.mark.asyncio
    async def test_membership_in_second_of_two_granted_teams(self, mock_prisma):
        first = _grant(principal_id="team-1", grant_id="grant-1")
        second = _grant(principal_id="team-2", grant_id="grant-2")
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[first, second])
        membership = MagicMock()
        membership.teamId = "team-2"
        mock_prisma.teammember.find_first = AsyncMock(return_value=membership)

        result = await resolve_graph_grant(
            "u1", "g1", capability=GrantCapability.EXECUTE
        )

        assert result is second


class TestGrantCoversVersion:
    def test_pinned_grant_covers_only_pinned_version(self):
        grant = _grant(version=3, follow_latest=False)
        assert grant_covers_version(grant, 3)
        assert not grant_covers_version(grant, 4)

    def test_follow_latest_covers_any_version(self):
        grant = _grant(version=3, follow_latest=True)
        assert grant_covers_version(grant, 7)


def _cred_grant(
    *,
    credential_mode=GrantCredentialMode.OWNER,
    org_id="org-1",
    version=3,
    follow_latest=False,
    grant_id="grant-1",
):
    row = MagicMock()
    row.id = grant_id
    row.credentialMode = credential_mode
    row.organizationId = org_id
    row.agentGraphVersion = version
    row.followLatest = follow_latest
    return row


def _cred_graph(*, user_id="owner-1", org_id="org-1", version=3, is_active=True):
    g = MagicMock()
    g.userId = user_id
    g.organizationId = org_id
    g.version = version
    g.isActive = is_active
    return g


class TestResolveExecutionCredentialsOwner:
    """The decision of whether a run executes on the graph OWNER's credentials."""

    @pytest.fixture
    def mock_prisma(self, mocker):
        mock = MagicMock()
        mock.agentgraph.find_unique = AsyncMock(return_value=_cred_graph())
        mock.agentgraph.find_first = AsyncMock(return_value=_cred_graph())
        mocker.patch("backend.data.grants.prisma", mock)
        return mock

    @pytest.fixture
    def patch_grant(self, mocker):
        def _set(grant):
            mocker.patch(
                "backend.data.grants.resolve_graph_grant",
                AsyncMock(return_value=grant),
            )

        return _set

    @pytest.mark.asyncio
    async def test_owner_mode_grant_returns_owner_and_grant_id(
        self, mock_prisma, patch_grant
    ):
        patch_grant(_cred_grant(credential_mode=GrantCredentialMode.OWNER))

        result = await resolve_execution_credentials_owner("consumer-1", "g1", 3)

        assert result == ("owner-1", "grant-1")

    @pytest.mark.asyncio
    async def test_consumer_mode_grant_returns_none(self, mock_prisma, patch_grant):
        patch_grant(_cred_grant(credential_mode=GrantCredentialMode.CONSUMER))

        assert await resolve_execution_credentials_owner("consumer-1", "g1", 3) is None

    @pytest.mark.asyncio
    async def test_graph_owner_running_own_graph_is_inert(self, mock_prisma, mocker):
        # Owner runs own graph: never OWNER mode, and no grant lookup needed.
        spy = mocker.patch(
            "backend.data.grants.resolve_graph_grant", AsyncMock(return_value=None)
        )

        assert await resolve_execution_credentials_owner("owner-1", "g1", 3) is None
        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_grant_returns_none(self, mock_prisma, patch_grant):
        # e.g. a marketplace/library run with no team grant at all.
        patch_grant(None)

        assert await resolve_execution_credentials_owner("consumer-1", "g1", 3) is None

    @pytest.mark.asyncio
    async def test_org_mismatch_returns_none(self, mock_prisma, patch_grant):
        patch_grant(_cred_grant(org_id="other-org"))

        assert await resolve_execution_credentials_owner("consumer-1", "g1", 3) is None

    @pytest.mark.asyncio
    async def test_pinned_version_not_covered_returns_none(
        self, mock_prisma, patch_grant
    ):
        # Grant pins v2; the run is v3.
        patch_grant(_cred_grant(version=2, follow_latest=False))

        assert await resolve_execution_credentials_owner("consumer-1", "g1", 3) is None

    @pytest.mark.asyncio
    async def test_missing_graph_returns_none(self, mock_prisma, patch_grant):
        mock_prisma.agentgraph.find_unique = AsyncMock(return_value=None)
        patch_grant(_cred_grant())

        assert await resolve_execution_credentials_owner("consumer-1", "g1", 3) is None

    @pytest.mark.asyncio
    async def test_version_none_resolves_active_version(self, mock_prisma, patch_grant):
        mock_prisma.agentgraph.find_first = AsyncMock(
            return_value=_cred_graph(version=5)
        )
        # follow_latest grant so any active version is covered.
        patch_grant(_cred_grant(follow_latest=True))

        result = await resolve_execution_credentials_owner("consumer-1", "g1", None)

        assert result == ("owner-1", "grant-1")
        # Active-version lookup, not a pinned find_unique.
        mock_prisma.agentgraph.find_unique.assert_not_called()
        where = mock_prisma.agentgraph.find_first.call_args.kwargs["where"]
        assert where["isActive"] is True

    @pytest.mark.asyncio
    async def test_follow_latest_grant_on_non_active_version_returns_none(
        self, mock_prisma, patch_grant
    ):
        # followLatest covers only the active version; a non-active version
        # reached via another access path stays CONSUMER.
        mock_prisma.agentgraph.find_unique = AsyncMock(
            return_value=_cred_graph(version=2, is_active=False)
        )
        patch_grant(_cred_grant(follow_latest=True))

        assert await resolve_execution_credentials_owner("consumer-1", "g1", 2) is None
