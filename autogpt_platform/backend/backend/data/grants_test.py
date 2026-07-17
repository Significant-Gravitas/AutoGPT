"""Tests for grant access resolution (share-with-team, teams-only v1)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import GrantCapability, GrantPrincipalType

from backend.data.grants import (
    GrantPrincipalNotSupportedError,
    grant_covers_version,
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
