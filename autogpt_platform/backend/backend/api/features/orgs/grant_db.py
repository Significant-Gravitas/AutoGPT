"""Database operations for agent-graph grants (share-with-team)."""

import logging

from prisma.enums import GrantCapability, GrantCredentialMode, GrantPrincipalType

from backend.data.db import prisma
from backend.util.exceptions import NotAuthorizedError, NotFoundError

from .grant_model import GrantResponse, ReceivedGrantResponse

logger = logging.getLogger(__name__)


async def upsert_grant(
    org_id: str,
    graph_id: str,
    *,
    principal_type: str,
    principal_id: str,
    graph_version: int | None,
    capability: str,
    credential_mode: str,
    follow_latest: bool,
    created_by_user_id: str,
    sharer_is_org_admin: bool,
) -> GrantResponse:
    """Create or update the grant for (graph, principal), pinning a version.

    Re-sharing updates the existing row's pin/capability/mode rather than
    stacking a second grant. Raises ``ValueError`` for unsupported principals
    or capability/mode values (route maps to 400) and ``NotFoundError`` when
    the graph/team isn't in this org.
    """
    if principal_type != GrantPrincipalType.TEAM:
        raise ValueError(
            "Only TEAM principals are supported; USER/PERSONA grants are not "
            "enabled yet"
        )
    if capability not in (GrantCapability.VIEW, GrantCapability.EXECUTE):
        raise ValueError(f"Unknown capability '{capability}'")
    if credential_mode not in (
        GrantCredentialMode.CONSUMER,
        GrantCredentialMode.OWNER,
    ):
        raise ValueError(f"Unknown credential mode '{credential_mode}'")

    team = await prisma.team.find_first(
        where={"id": principal_id, "orgId": org_id, "archivedAt": None}
    )
    if team is None:
        raise NotFoundError(f"Team #{principal_id} not found in this organization")

    graph_where: dict = {"id": graph_id, "organizationId": org_id}
    if graph_version is not None:
        graph_where["version"] = graph_version
    else:
        graph_where["isActive"] = True
    graph = await prisma.agentgraph.find_first(
        where=graph_where, order={"version": "desc"}
    )
    if graph is None:
        raise NotFoundError(
            f"Graph #{graph_id}"
            + (f" v{graph_version}" if graph_version is not None else "")
            + " not found in this organization"
        )
    if graph.userId != created_by_user_id and not sharer_is_org_admin:
        raise NotAuthorizedError("Only the graph's owner or an org admin can share it")

    # OWNER credential-mode exposes the graph owner's stored credentials to
    # everyone on the team at execution time. Only the owner may consent to
    # that — an org admin sharing someone else's graph must not be able to
    # hand out a third party's secrets. (400 via ValueError.)
    if (
        credential_mode == GrantCredentialMode.OWNER
        and graph.userId != created_by_user_id
    ):
        raise ValueError(
            "OWNER credential-mode grants may only be created by the graph's "
            "owner, not by an org admin sharing another user's graph"
        )

    grant = await prisma.agentgraphgrant.upsert(
        where={
            "agentGraphId_principalType_principalId": {
                "agentGraphId": graph_id,
                "principalType": GrantPrincipalType.TEAM,
                "principalId": principal_id,
            }
        },
        data={
            "create": {
                "agentGraphId": graph_id,
                "agentGraphVersion": graph.version,
                "followLatest": follow_latest,
                "principalType": GrantPrincipalType.TEAM,
                "principalId": principal_id,
                "capability": GrantCapability(capability),
                "credentialMode": GrantCredentialMode(credential_mode),
                "organizationId": org_id,
                "createdByUserId": created_by_user_id,
            },
            "update": {
                "agentGraphVersion": graph.version,
                "followLatest": follow_latest,
                "capability": GrantCapability(capability),
                "credentialMode": GrantCredentialMode(credential_mode),
            },
        },
    )
    logger.info(
        f"Grant upserted: graph {graph_id} v{graph.version} -> team "
        f"{principal_id} ({capability}, follow_latest={follow_latest})"
    )
    return GrantResponse.from_db(grant)


async def list_grants_for_graph(org_id: str, graph_id: str) -> list[GrantResponse]:
    """List grants on a graph within this org."""
    grants = await prisma.agentgraphgrant.find_many(
        where={"agentGraphId": graph_id, "organizationId": org_id},
        order={"createdAt": "asc"},
    )
    return [GrantResponse.from_db(g) for g in grants]


async def list_received_grants(
    org_id: str, user_id: str
) -> list[ReceivedGrantResponse]:
    """List grants shared with any team the user is an ACTIVE member of."""
    memberships = await prisma.teammember.find_many(
        where={
            "userId": user_id,
            "status": "ACTIVE",
            "Team": {"is": {"orgId": org_id, "archivedAt": None}},
        }
    )
    team_ids = [m.teamId for m in memberships]
    if not team_ids:
        return []

    grants = await prisma.agentgraphgrant.find_many(
        where={
            "organizationId": org_id,
            "principalType": GrantPrincipalType.TEAM,
            "principalId": {"in": team_ids},
        },
        include={"AgentGraph": True},
        order={"createdAt": "desc"},
    )
    return [ReceivedGrantResponse.from_db(g) for g in grants]


async def revoke_grant(
    org_id: str,
    graph_id: str,
    grant_id: str,
    *,
    revoked_by_user_id: str,
    revoker_is_org_admin: bool,
) -> None:
    """Delete a grant. Raises ``NotFoundError`` if it isn't on this org+graph."""
    grant = await prisma.agentgraphgrant.find_first(
        where={"id": grant_id, "agentGraphId": graph_id, "organizationId": org_id}
    )
    if grant is None:
        raise NotFoundError(f"Grant #{grant_id} not found")

    graph = await prisma.agentgraph.find_first(
        where={"id": graph_id}, order={"version": "desc"}
    )
    if (
        graph is not None
        and graph.userId != revoked_by_user_id
        and not revoker_is_org_admin
    ):
        raise NotAuthorizedError(
            "Only the graph's owner or an org admin can revoke a grant"
        )

    await prisma.agentgraphgrant.delete(where={"id": grant_id})
    logger.info(f"Grant {grant_id} on graph {graph_id} revoked")
