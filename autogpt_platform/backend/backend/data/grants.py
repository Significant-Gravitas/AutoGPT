"""Access resolution for AgentGraph grants (share-with-team).

v1 enforces TEAM principals only. The schema is deliberately polymorphic
(``GrantPrincipalType``) so USER/PERSONA principals can ship later without a
table migration — but until they do, encountering a non-TEAM row is a hard
error, never a silent skip: a row like that can only exist by bypassing the
grants API, and quietly ignoring it would hide both the bypass and the moment
someone flips on a principal type the enforcement below doesn't actually check.
"""

import logging

from prisma.enums import GrantCapability, GrantPrincipalType
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
