"""Access resolution for AgentGraph grants (share-with-team).

v1 enforces TEAM principals only. The schema is deliberately polymorphic
(``GrantPrincipalType``) so USER/PERSONA principals can ship later without a
table migration — but until they do, encountering a non-TEAM row is a hard
error, never a silent skip: a row like that can only exist by bypassing the
grants API, and quietly ignoring it would hide both the bypass and the moment
someone flips on a principal type the enforcement below doesn't actually check.
"""

import logging

from prisma.enums import GrantCapability, GrantCredentialMode, GrantPrincipalType
from prisma.models import AgentGraphGrant

from backend.data.db import prisma

logger = logging.getLogger(__name__)


class GrantPrincipalNotSupportedError(Exception):
    """A grant row carries a principal type enforcement does not support yet."""


async def resolve_graph_grant(
    user_id: str, graph_id: str, *, capability: GrantCapability
) -> AgentGraphGrant | None:
    """Return a grant giving *user_id* the *capability* on *graph_id*, if any.

    EXECUTE implies VIEW: a VIEW check is satisfied by either capability, an
    EXECUTE check only by an EXECUTE grant.

    Raises ``GrantPrincipalNotSupportedError`` if any grant row on the graph
    has a non-TEAM principal.
    """
    grant_rows = await prisma.agentgraphgrant.find_many(
        where={"agentGraphId": graph_id}
    )
    if not grant_rows:
        return None

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
        return None

    membership = await prisma.teammember.find_first(
        where={
            "userId": user_id,
            "status": "ACTIVE",
            "teamId": {"in": [row.principalId for row in eligible]},
            # Archived workspaces stop granting access, matching create-time
            # rules (upsert_grant rejects archived teams) and list_teams.
            "Team": {"is": {"archivedAt": None}},
        }
    )
    if membership is None:
        return None

    for row in eligible:
        if row.principalId == membership.teamId:
            return row
    return None


def grant_covers_version(grant: AgentGraphGrant, version: int) -> bool:
    """Whether *grant* allows access to this exact graph *version*.

    Pinned grants cover only the pinned version. ``followLatest`` grants cover
    any version — the caller must additionally ensure the version it resolved
    is the graph's active one (see ``get_graph``'s grant fallback and the
    execution check, which pass through the active-version constraint).
    """
    return grant.followLatest or grant.agentGraphVersion == version


async def resolve_execution_credentials_owner(
    user_id: str, graph_id: str, graph_version: int | None = None
) -> tuple[str, str] | None:
    """Resolve whether *user_id*'s run of *graph_id* must use the graph OWNER's
    credentials, and if so on whose behalf.

    ``graph_version`` defaults to the graph's active version when omitted (the
    same version ``get_graph`` resolves for a version-less run).

    Returns ``(owner_user_id, grant_id)`` when the consumer reaches this graph
    purely via an **OWNER**-mode EXECUTE team grant, else ``None``.

    Enforcement details (each returns ``None``):
    - Graph missing, or the caller **owns** the graph (OWNER mode is inert for
      an owner running their own graph — they already use their own store).
    - No EXECUTE grant, or the grant's ``credentialMode`` is CONSUMER.
    - The grant is for a different org than the graph, or does not cover the
      resolved version (pinned/``followLatest`` semantics via ``grant_covers_version``).
    - A ``followLatest`` grant on a non-active version: it only covers the
      active one (matching ``validate_graph_execution_permissions``), so a
      non-active version reached via another access path stays CONSUMER.

    This reuses :func:`resolve_graph_grant`, so the ACTIVE team-membership check
    that authorizes *access* is the same one that gates credential mode — there
    is no second authorization path. Because ``resolve_graph_grant`` re-checks
    live membership every call, a consumer removed from the team stops getting
    OWNER-credential resolution immediately (their run then fails closed on the
    missing credential rather than silently using their own).
    """
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
    graph_version = graph.version

    grant = await resolve_graph_grant(
        user_id, graph_id, capability=GrantCapability.EXECUTE
    )
    if grant is None or grant.credentialMode != GrantCredentialMode.OWNER:
        return None
    if graph.organizationId != grant.organizationId:
        return None
    if not grant_covers_version(grant, graph_version):
        return None
    if grant.followLatest and not graph.isActive:
        return None

    return graph.userId, grant.id
