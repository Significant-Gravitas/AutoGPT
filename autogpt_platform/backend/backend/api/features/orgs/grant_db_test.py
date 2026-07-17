"""Tests for grant CRUD validation (share-with-team, teams-only v1)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import GrantPrincipalType

from backend.api.features.orgs import grant_db
from backend.util.exceptions import NotFoundError


def _graph(*, graph_id="g1", version=3, user_id="owner-1"):
    graph = MagicMock()
    graph.id = graph_id
    graph.version = version
    graph.userId = user_id
    return graph


def _grant_row():
    row = MagicMock()
    row.id = "grant-1"
    row.agentGraphId = "g1"
    row.agentGraphVersion = 3
    row.followLatest = False
    row.principalType = GrantPrincipalType.TEAM
    row.principalId = "team-1"
    row.capability = "EXECUTE"
    row.credentialMode = "CONSUMER"
    row.organizationId = "org-1"
    row.createdByUserId = "owner-1"
    row.createdAt = MagicMock()
    return row


@pytest.fixture
def mock_prisma(mocker):
    mock = MagicMock()
    mock.team.find_first = AsyncMock(return_value=MagicMock(id="team-1"))
    mock.agentgraph.find_first = AsyncMock(return_value=_graph())
    mock.agentgraphgrant.upsert = AsyncMock(return_value=_grant_row())
    mock.agentgraphgrant.find_first = AsyncMock(return_value=_grant_row())
    mock.agentgraphgrant.find_many = AsyncMock(return_value=[])
    mock.agentgraphgrant.delete = AsyncMock()
    mock.teammember.find_many = AsyncMock(return_value=[])
    mocker.patch("backend.api.features.orgs.grant_db.prisma", mock)
    return mock


def _upsert_kwargs(**overrides):
    kwargs = dict(
        principal_type="TEAM",
        principal_id="team-1",
        graph_version=None,
        capability="EXECUTE",
        credential_mode="CONSUMER",
        follow_latest=False,
        created_by_user_id="owner-1",
        sharer_is_org_admin=False,
    )
    kwargs.update(overrides)
    return kwargs


class TestUpsertGrant:
    @pytest.mark.asyncio
    async def test_user_principal_rejected(self, mock_prisma):
        with pytest.raises(ValueError, match="Only TEAM principals"):
            await grant_db.upsert_grant(
                "org-1", "g1", **_upsert_kwargs(principal_type="USER")
            )
        mock_prisma.agentgraphgrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_capability_rejected(self, mock_prisma):
        with pytest.raises(ValueError, match="capability"):
            await grant_db.upsert_grant(
                "org-1", "g1", **_upsert_kwargs(capability="ADMIN")
            )

    @pytest.mark.asyncio
    async def test_team_outside_org_not_found(self, mock_prisma):
        mock_prisma.team.find_first = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError, match="Team"):
            await grant_db.upsert_grant("org-1", "g1", **_upsert_kwargs())

    @pytest.mark.asyncio
    async def test_graph_outside_org_not_found(self, mock_prisma):
        mock_prisma.agentgraph.find_first = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError, match="Graph"):
            await grant_db.upsert_grant("org-1", "g1", **_upsert_kwargs())

    @pytest.mark.asyncio
    async def test_non_owner_non_admin_cannot_share(self, mock_prisma):
        with pytest.raises(ValueError, match="owner or an org admin"):
            await grant_db.upsert_grant(
                "org-1", "g1", **_upsert_kwargs(created_by_user_id="someone-else")
            )

    @pytest.mark.asyncio
    async def test_org_admin_can_share_others_graph(self, mock_prisma):
        result = await grant_db.upsert_grant(
            "org-1",
            "g1",
            **_upsert_kwargs(
                created_by_user_id="someone-else", sharer_is_org_admin=True
            ),
        )

        assert result.id == "grant-1"

    @pytest.mark.asyncio
    async def test_default_pin_is_active_version(self, mock_prisma):
        await grant_db.upsert_grant("org-1", "g1", **_upsert_kwargs())

        graph_where = mock_prisma.agentgraph.find_first.call_args.kwargs["where"]
        assert graph_where["isActive"] is True
        upsert_data = mock_prisma.agentgraphgrant.upsert.call_args.kwargs["data"]
        assert upsert_data["create"]["agentGraphVersion"] == 3
        assert upsert_data["update"]["agentGraphVersion"] == 3

    @pytest.mark.asyncio
    async def test_explicit_version_pin(self, mock_prisma):
        mock_prisma.agentgraph.find_first = AsyncMock(return_value=_graph(version=2))

        await grant_db.upsert_grant("org-1", "g1", **_upsert_kwargs(graph_version=2))

        graph_where = mock_prisma.agentgraph.find_first.call_args.kwargs["where"]
        assert graph_where["version"] == 2
        assert "isActive" not in graph_where


class TestRevokeGrant:
    @pytest.mark.asyncio
    async def test_missing_grant_not_found(self, mock_prisma):
        mock_prisma.agentgraphgrant.find_first = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await grant_db.revoke_grant(
                "org-1",
                "g1",
                "grant-1",
                revoked_by_user_id="owner-1",
                revoker_is_org_admin=False,
            )
        mock_prisma.agentgraphgrant.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_owner_non_admin_cannot_revoke(self, mock_prisma):
        with pytest.raises(ValueError, match="owner or an org admin"):
            await grant_db.revoke_grant(
                "org-1",
                "g1",
                "grant-1",
                revoked_by_user_id="someone-else",
                revoker_is_org_admin=False,
            )

    @pytest.mark.asyncio
    async def test_owner_revokes(self, mock_prisma):
        await grant_db.revoke_grant(
            "org-1",
            "g1",
            "grant-1",
            revoked_by_user_id="owner-1",
            revoker_is_org_admin=False,
        )

        mock_prisma.agentgraphgrant.delete.assert_called_once_with(
            where={"id": "grant-1"}
        )


class TestListReceivedGrants:
    @pytest.mark.asyncio
    async def test_no_team_memberships_short_circuits(self, mock_prisma):
        result = await grant_db.list_received_grants("org-1", "u1")

        assert result == []
        mock_prisma.agentgraphgrant.find_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_queries_only_active_memberships_in_org(self, mock_prisma):
        membership = MagicMock()
        membership.teamId = "team-1"
        mock_prisma.teammember.find_many = AsyncMock(return_value=[membership])

        await grant_db.list_received_grants("org-1", "u1")

        member_where = mock_prisma.teammember.find_many.call_args.kwargs["where"]
        assert member_where["status"] == "ACTIVE"
        assert member_where["Team"] == {"is": {"orgId": "org-1"}}
        grant_where = mock_prisma.agentgraphgrant.find_many.call_args.kwargs["where"]
        assert grant_where["principalId"] == {"in": ["team-1"]}
        assert grant_where["principalType"] == GrantPrincipalType.TEAM
