"""Access resolution for team grants on AgentGraph."""

from prisma.enums import GrantCapability, GrantCredentialMode, GrantPrincipalType
from prisma.models import AgentGraphGrant

from backend.data.db import prisma


class GrantPrincipalNotSupportedError(Exception):
    """A grant row carries a principal type enforcement does not support yet."""


class AmbiguousGrantCredentialModeError(ValueError):
    """Covering grants disagree about whose credentials an execution uses."""


class OwnerGrantConsentError(ValueError):
    """An OWNER grant lacks consent from the graph's current owner."""


async def _is_active_org_member(user_id: str, organization_id: str | None) -> bool:
    if organization_id is None:
        return False
    return (
        await prisma.orgmember.find_first(
            where={
                "userId": user_id,
                "orgId": organization_id,
                "status": "ACTIVE",
                "Org": {"is": {"deletedAt": None}},
            }
        )
        is not None
    )


async def resolve_graph_grant(
    user_id: str,
    graph_id: str,
    *,
    capability: GrantCapability,
    graph_version: int | None = None,
) -> AgentGraphGrant | None:
    grants = await resolve_graph_grants(
        user_id,
        graph_id,
        capability=capability,
    )
    if graph_version is not None:
        grants = [
            grant for grant in grants if grant_covers_version(grant, graph_version)
        ]
    return grants[0] if grants else None


async def resolve_graph_grants(
    user_id: str, graph_id: str, *, capability: GrantCapability
) -> list[AgentGraphGrant]:
    grant_rows = await prisma.agentgraphgrant.find_many(
        where={"agentGraphId": graph_id}
    )
    if not grant_rows:
        return []

    for row in grant_rows:
        if row.principalType != GrantPrincipalType.TEAM:
            raise GrantPrincipalNotSupportedError(
                f"Grant {row.id} on graph {graph_id} has unsupported principal "
                f"type {row.principalType}; only TEAM principals are enforced"
            )

    eligible = [
        row
        for row in grant_rows
        if capability == GrantCapability.VIEW
        or row.capability == GrantCapability.EXECUTE
    ]
    if not eligible:
        return []

    memberships = await prisma.teammember.find_many(
        where={
            "userId": user_id,
            "status": "ACTIVE",
            "teamId": {"in": [row.principalId for row in eligible]},
            "Team": {"is": {"archivedAt": None}},
        },
        include={"Team": True},
    )
    org_memberships = await prisma.orgmember.find_many(
        where={
            "userId": user_id,
            "status": "ACTIVE",
            "orgId": {"in": sorted({row.organizationId for row in eligible})},
            "Org": {"is": {"deletedAt": None}},
        }
    )
    active_org_ids = {membership.orgId for membership in org_memberships}
    active_team_orgs = {
        membership.teamId: membership.Team.orgId
        for membership in memberships
        if membership.Team is not None
    }
    matched = [
        row
        for row in eligible
        if row.organizationId in active_org_ids
        and active_team_orgs.get(row.principalId) == row.organizationId
    ]
    return sorted(
        matched,
        key=lambda row: (
            not row.followLatest,
            -row.agentGraphVersion,
            row.id,
        ),
    )


def grant_covers_version(grant: AgentGraphGrant, version: int) -> bool:
    return grant.followLatest or grant.agentGraphVersion == version


async def resolve_execution_credentials_owner(
    user_id: str, graph_id: str, graph_version: int | None = None
) -> tuple[str, str] | None:
    if graph_version is not None:
        graph = await prisma.agentgraph.find_unique(
            where={"graphVersionId": {"id": graph_id, "version": graph_version}}
        )
    else:
        graph = await prisma.agentgraph.find_first(
            where={"id": graph_id, "isActive": True}, order={"version": "desc"}
        )
    if graph is None or graph.userId == user_id:
        return None
    resolved_version: int = graph.version

    grants = await resolve_graph_grants(
        user_id, graph_id, capability=GrantCapability.EXECUTE
    )
    covering_grants = [
        grant
        for grant in grants
        if graph.organizationId == grant.organizationId
        and grant_covers_version(grant, resolved_version)
        and (not grant.followLatest or graph.isActive)
    ]
    if not covering_grants:
        return None

    credential_modes = {grant.credentialMode for grant in covering_grants}
    if len(credential_modes) > 1:
        grant_ids = ", ".join(grant.id for grant in covering_grants)
        raise AmbiguousGrantCredentialModeError(
            f"Graph #{graph_id} has covering execution grants with conflicting "
            f"credential modes ({grant_ids}); make the grants agree before running"
        )
    if credential_modes == {GrantCredentialMode.CONSUMER}:
        return None

    owner_grants = [
        grant
        for grant in covering_grants
        if grant.credentialMode == GrantCredentialMode.OWNER
    ]
    if not owner_grants:
        return None

    unconsented = [
        grant for grant in owner_grants if grant.createdByUserId != graph.userId
    ]
    if unconsented:
        grant_ids = ", ".join(grant.id for grant in unconsented)
        raise OwnerGrantConsentError(
            f"OWNER grants {grant_ids} were not consented to by graph "
            f"#{graph_id}'s current owner"
        )
    if not await _is_active_org_member(graph.userId, graph.organizationId):
        raise OwnerGrantConsentError(
            f"Graph #{graph_id}'s current owner is no longer an active member "
            "of the grant organization"
        )

    return graph.userId, owner_grants[0].id


async def validate_execution_credentials_owner(
    user_id: str,
    graph_id: str,
    graph_version: int,
    owner_user_id: str,
    grant_id: str,
) -> bool:
    graph = await prisma.agentgraph.find_unique(
        where={"graphVersionId": {"id": graph_id, "version": graph_version}}
    )
    if graph is None or graph.userId != owner_user_id or graph.userId == user_id:
        return False
    if not await _is_active_org_member(owner_user_id, graph.organizationId):
        return False

    grants = await resolve_graph_grants(
        user_id, graph_id, capability=GrantCapability.EXECUTE
    )
    covering_grants = [
        grant
        for grant in grants
        if graph.organizationId == grant.organizationId
        and grant_covers_version(grant, graph_version)
        and (not grant.followLatest or graph.isActive)
    ]
    credential_modes = {grant.credentialMode for grant in covering_grants}
    if len(credential_modes) > 1:
        raise AmbiguousGrantCredentialModeError(
            f"Graph #{graph_id} now has covering grants with conflicting "
            "credential modes"
        )
    if credential_modes != {GrantCredentialMode.OWNER}:
        return False

    selected = next((grant for grant in covering_grants if grant.id == grant_id), None)
    if not (
        selected
        and selected.createdByUserId == owner_user_id
        and selected.principalType == GrantPrincipalType.TEAM
        and selected.capability == GrantCapability.EXECUTE
    ):
        return False
    return (
        await prisma.team.find_first(
            where={
                "id": selected.principalId,
                "orgId": selected.organizationId,
                "archivedAt": None,
            }
        )
        is not None
    )
