"""Tests for grant access resolution (share-with-team, teams-only v1)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import GrantCapability, GrantCredentialMode, GrantPrincipalType

from backend.data.grants import (
    AmbiguousGrantCredentialModeError,
    GrantPrincipalNotSupportedError,
    OwnerGrantConsentError,
    grant_covers_version,
    resolve_execution_credentials_owner,
    resolve_graph_grant,
    validate_execution_credentials_owner,
)


def _grant(
    *,
    principal_type=GrantPrincipalType.TEAM,
    principal_id="team-1",
    capability=GrantCapability.EXECUTE,
    version=3,
    follow_latest=False,
    grant_id="grant-1",
    org_id="org-1",
):
    row = MagicMock()
    row.id = grant_id
    row.principalType = principal_type
    row.principalId = principal_id
    row.capability = capability
    row.agentGraphVersion = version
    row.followLatest = follow_latest
    row.organizationId = org_id
    return row


def _membership(team_id="team-1", org_id="org-1"):
    membership = MagicMock()
    membership.teamId = team_id
    membership.Team = MagicMock(orgId=org_id)
    membership.isAdmin = False
    membership.isBillingManager = False
    return membership


def _org_membership(org_id="org-1", *, billing_only=False):
    return MagicMock(
        orgId=org_id,
        isOwner=False,
        isAdmin=False,
        isBillingManager=billing_only,
    )


@pytest.fixture
def mock_prisma(mocker):
    mock = MagicMock()
    mock.agentgraphgrant.find_many = AsyncMock(return_value=[])
    mock.teammember.find_many = AsyncMock(return_value=[])
    mock.orgmember.find_many = AsyncMock(return_value=[_org_membership()])
    mock.orgmember.find_first = AsyncMock(return_value=_org_membership())
    mocker.patch("backend.data.grants.prisma", mock)
    return mock


class TestResolveGraphGrant:
    @pytest.mark.asyncio
    async def test_no_grants_returns_none(self, mock_prisma):
        assert (
            await resolve_graph_grant("u1", "g1", capability=GrantCapability.EXECUTE)
            is None
        )
        mock_prisma.teammember.find_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_active_team_member_gets_grant(self, mock_prisma):
        grant = _grant()
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[grant])
        membership = _membership()
        mock_prisma.teammember.find_many = AsyncMock(return_value=[membership])

        result = await resolve_graph_grant(
            "u1", "g1", capability=GrantCapability.EXECUTE
        )

        assert result is grant
        where = mock_prisma.teammember.find_many.call_args.kwargs["where"]
        assert where["status"] == "ACTIVE"
        assert where["teamId"] == {"in": ["team-1"]}
        # Archived workspaces must not keep granting access.
        assert where["Team"] == {"is": {"archivedAt": None}}
        assert mock_prisma.teammember.find_many.call_args.kwargs["include"] == {
            "Team": True
        }
        org_where = mock_prisma.orgmember.find_many.call_args.kwargs["where"]
        assert org_where["status"] == "ACTIVE"
        assert org_where["Org"] == {"is": {"deletedAt": None}}

    @pytest.mark.asyncio
    async def test_non_member_gets_nothing(self, mock_prisma):
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[_grant()])
        mock_prisma.teammember.find_many = AsyncMock(return_value=[])

        assert (
            await resolve_graph_grant("u1", "g1", capability=GrantCapability.EXECUTE)
            is None
        )

    @pytest.mark.asyncio
    async def test_view_check_satisfied_by_view_or_execute_grant(self, mock_prisma):
        view_grant = _grant(capability=GrantCapability.VIEW)
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[view_grant])
        membership = _membership()
        mock_prisma.teammember.find_many = AsyncMock(return_value=[membership])

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
        mock_prisma.teammember.find_many.assert_not_called()

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
        membership = _membership("team-2")
        mock_prisma.teammember.find_many = AsyncMock(return_value=[membership])

        result = await resolve_graph_grant(
            "u1", "g1", capability=GrantCapability.EXECUTE
        )

        assert result is second

    @pytest.mark.asyncio
    async def test_multiple_memberships_choose_deterministically(self, mock_prisma):
        pinned_old = _grant(principal_id="team-1", grant_id="z-grant", version=2)
        follow_latest = _grant(
            principal_id="team-2",
            grant_id="a-grant",
            version=1,
            follow_latest=True,
        )
        mock_prisma.agentgraphgrant.find_many = AsyncMock(
            return_value=[pinned_old, follow_latest]
        )
        first_membership = _membership("team-1")
        second_membership = _membership("team-2")
        mock_prisma.teammember.find_many = AsyncMock(
            return_value=[first_membership, second_membership]
        )

        result = await resolve_graph_grant(
            "u1", "g1", capability=GrantCapability.EXECUTE
        )

        assert result is follow_latest

    @pytest.mark.asyncio
    async def test_team_membership_from_different_org_does_not_cover_grant(
        self, mock_prisma
    ):
        grant = _grant(org_id="graph-org")
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[grant])
        mock_prisma.teammember.find_many = AsyncMock(
            return_value=[_membership(org_id="different-org")]
        )

        assert (
            await resolve_graph_grant("u1", "g1", capability=GrantCapability.EXECUTE)
            is None
        )

    @pytest.mark.asyncio
    async def test_deleted_or_inactive_org_membership_makes_grant_inert(
        self, mock_prisma
    ):
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[_grant()])
        mock_prisma.teammember.find_many = AsyncMock(return_value=[_membership()])
        mock_prisma.orgmember.find_many = AsyncMock(return_value=[])

        assert (
            await resolve_graph_grant("u1", "g1", capability=GrantCapability.EXECUTE)
            is None
        )

    @pytest.mark.asyncio
    async def test_org_billing_only_membership_makes_grant_inert(self, mock_prisma):
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[_grant()])
        mock_prisma.teammember.find_many = AsyncMock(return_value=[_membership()])
        mock_prisma.orgmember.find_many = AsyncMock(
            return_value=[_org_membership(billing_only=True)]
        )

        assert (
            await resolve_graph_grant("u1", "g1", capability=GrantCapability.EXECUTE)
            is None
        )


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
    row.createdByUserId = "owner-1"
    row.principalType = GrantPrincipalType.TEAM
    row.principalId = "team-1"
    row.capability = GrantCapability.EXECUTE
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
        mock.orgmember.find_first = AsyncMock(return_value=_org_membership())
        mocker.patch("backend.data.grants.prisma", mock)
        return mock

    @pytest.fixture
    def patch_grant(self, mocker):
        def _set(grant):
            mocker.patch(
                "backend.data.grants.resolve_graph_grants",
                AsyncMock(return_value=[grant] if grant else []),
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
    async def test_owner_transfer_requires_fresh_owner_consent(
        self, mock_prisma, mocker
    ):
        grant = _cred_grant()
        grant.createdByUserId = "previous-owner"
        mocker.patch(
            "backend.data.grants.resolve_graph_grants",
            AsyncMock(return_value=[grant]),
        )

        with pytest.raises(OwnerGrantConsentError, match="current owner"):
            await resolve_execution_credentials_owner("consumer-1", "g1", 3)

    @pytest.mark.asyncio
    async def test_owner_org_removal_rejects_owner_mode_resolution(
        self, mock_prisma, patch_grant
    ):
        patch_grant(_cred_grant())
        mock_prisma.orgmember.find_first = AsyncMock(return_value=None)

        with pytest.raises(OwnerGrantConsentError, match="no longer an active member"):
            await resolve_execution_credentials_owner("consumer-1", "g1", 3)

    @pytest.mark.asyncio
    async def test_owner_billing_downgrade_rejects_owner_mode_resolution(
        self, mock_prisma, patch_grant
    ):
        patch_grant(_cred_grant())
        mock_prisma.orgmember.find_first = AsyncMock(
            return_value=_org_membership(billing_only=True)
        )

        with pytest.raises(OwnerGrantConsentError, match="no longer an active member"):
            await resolve_execution_credentials_owner("consumer-1", "g1", 3)

    @pytest.mark.asyncio
    async def test_conflicting_covering_modes_fail_closed(self, mock_prisma, mocker):
        owner = _cred_grant(
            credential_mode=GrantCredentialMode.OWNER,
            grant_id="grant-owner",
        )
        consumer = _cred_grant(
            credential_mode=GrantCredentialMode.CONSUMER,
            grant_id="grant-consumer",
        )
        mocker.patch(
            "backend.data.grants.resolve_graph_grants",
            AsyncMock(return_value=[owner, consumer]),
        )

        with pytest.raises(AmbiguousGrantCredentialModeError, match="conflicting"):
            await resolve_execution_credentials_owner("consumer-1", "g1", 3)

    @pytest.mark.asyncio
    async def test_different_version_consumer_grant_is_not_ambiguous(
        self, mock_prisma, mocker
    ):
        owner = _cred_grant(
            credential_mode=GrantCredentialMode.OWNER,
            grant_id="grant-owner",
            version=3,
        )
        consumer = _cred_grant(
            credential_mode=GrantCredentialMode.CONSUMER,
            grant_id="grant-consumer",
            version=2,
        )
        mocker.patch(
            "backend.data.grants.resolve_graph_grants",
            AsyncMock(return_value=[owner, consumer]),
        )

        assert await resolve_execution_credentials_owner("consumer-1", "g1", 3) == (
            "owner-1",
            "grant-owner",
        )

    @pytest.mark.asyncio
    async def test_graph_owner_running_own_graph_is_inert(self, mock_prisma, mocker):
        # Owner runs own graph: never OWNER mode, and no grant lookup needed.
        spy = mocker.patch(
            "backend.data.grants.resolve_graph_grants", AsyncMock(return_value=[])
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


class TestValidateExecutionCredentialsOwner:
    @pytest.fixture
    def mock_prisma(self, mocker):
        mock = MagicMock()
        mock.agentgraph.find_unique = AsyncMock(return_value=_cred_graph())
        mock.agentgraphgrant.find_many = AsyncMock(return_value=[])
        mock.teammember.find_many = AsyncMock(return_value=[])
        mock.orgmember.find_many = AsyncMock(return_value=[_org_membership()])
        mock.orgmember.find_first = AsyncMock(return_value=_org_membership())
        mock.team.find_first = AsyncMock(return_value=MagicMock())
        mocker.patch("backend.data.grants.prisma", mock)
        return mock

    def _authorize(self, mock_prisma):
        grant = _cred_grant()
        membership = _membership()
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[grant])
        mock_prisma.teammember.find_many = AsyncMock(return_value=[membership])
        return grant

    @pytest.mark.asyncio
    async def test_exact_selected_grant_is_valid(self, mock_prisma):
        self._authorize(mock_prisma)

        assert await validate_execution_credentials_owner(
            "consumer-1", "g1", 3, "owner-1", "grant-1"
        )

    @pytest.mark.asyncio
    async def test_removed_membership_invalidates_queued_grant(self, mock_prisma):
        self._authorize(mock_prisma)
        mock_prisma.teammember.find_many = AsyncMock(return_value=[])

        assert not await validate_execution_credentials_owner(
            "consumer-1", "g1", 3, "owner-1", "grant-1"
        )
        mock_prisma.team.find_first.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_owner_change_invalidates_queued_grant(self, mock_prisma):
        self._authorize(mock_prisma)
        mock_prisma.agentgraph.find_unique = AsyncMock(
            return_value=_cred_graph(user_id="new-owner")
        )

        assert not await validate_execution_credentials_owner(
            "consumer-1", "g1", 3, "owner-1", "grant-1"
        )

    @pytest.mark.asyncio
    async def test_owner_org_removal_invalidates_queued_grant(self, mock_prisma):
        self._authorize(mock_prisma)
        mock_prisma.orgmember.find_first = AsyncMock(return_value=None)

        assert not await validate_execution_credentials_owner(
            "consumer-1", "g1", 3, "owner-1", "grant-1"
        )

    @pytest.mark.asyncio
    async def test_owner_billing_downgrade_invalidates_queued_grant(self, mock_prisma):
        self._authorize(mock_prisma)
        mock_prisma.orgmember.find_first = AsyncMock(
            return_value=_org_membership(billing_only=True)
        )

        assert not await validate_execution_credentials_owner(
            "consumer-1", "g1", 3, "owner-1", "grant-1"
        )

    @pytest.mark.asyncio
    async def test_archived_or_wrong_org_team_invalidates_queued_grant(
        self, mock_prisma
    ):
        self._authorize(mock_prisma)
        mock_prisma.team.find_first = AsyncMock(return_value=None)

        assert not await validate_execution_credentials_owner(
            "consumer-1", "g1", 3, "owner-1", "grant-1"
        )
        where = mock_prisma.team.find_first.call_args.kwargs["where"]
        assert where == {"id": "team-1", "orgId": "org-1", "archivedAt": None}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mutation",
        [
            "consumer_mode",
            "view_capability",
            "different_pin",
            "inactive_follow_latest",
            "stale_consent",
        ],
    )
    async def test_selected_grant_mutations_invalidate_authorization(
        self, mock_prisma, mutation
    ):
        grant = self._authorize(mock_prisma)
        if mutation == "consumer_mode":
            grant.credentialMode = GrantCredentialMode.CONSUMER
        elif mutation == "view_capability":
            grant.capability = GrantCapability.VIEW
        elif mutation == "different_pin":
            grant.agentGraphVersion = 2
        elif mutation == "inactive_follow_latest":
            grant.followLatest = True
            mock_prisma.agentgraph.find_unique = AsyncMock(
                return_value=_cred_graph(is_active=False)
            )
        elif mutation == "stale_consent":
            grant.createdByUserId = "former-owner"

        assert not await validate_execution_credentials_owner(
            "consumer-1", "g1", 3, "owner-1", "grant-1"
        )

    @pytest.mark.asyncio
    async def test_deleted_selected_grant_invalidates_even_if_another_owner_path_exists(
        self, mock_prisma
    ):
        other = _cred_grant(grant_id="grant-2")
        other.principalId = "team-2"
        mock_prisma.agentgraphgrant.find_many = AsyncMock(return_value=[other])
        mock_prisma.teammember.find_many = AsyncMock(
            return_value=[_membership("team-2")]
        )

        assert not await validate_execution_credentials_owner(
            "consumer-1", "g1", 3, "owner-1", "grant-1"
        )

    @pytest.mark.asyncio
    async def test_newly_ambiguous_modes_raise_configuration_error(self, mock_prisma):
        owner = _cred_grant(grant_id="grant-1")
        consumer = _cred_grant(
            grant_id="grant-2", credential_mode=GrantCredentialMode.CONSUMER
        )
        consumer.principalId = "team-2"
        mock_prisma.agentgraphgrant.find_many = AsyncMock(
            return_value=[owner, consumer]
        )
        mock_prisma.teammember.find_many = AsyncMock(
            return_value=[_membership("team-1"), _membership("team-2")]
        )

        with pytest.raises(AmbiguousGrantCredentialModeError):
            await validate_execution_credentials_owner(
                "consumer-1", "g1", 3, "owner-1", "grant-1"
            )


def test_grant_configuration_errors_are_expected_value_errors():
    assert issubclass(AmbiguousGrantCredentialModeError, ValueError)
    assert issubclass(OwnerGrantConsentError, ValueError)
