"""Per-turn re-verification of a chat session's persisted org/team.

A :class:`ChatSession` row is the authoritative tenancy for every turn in
it, but membership is only validated when the session is *created*. A user
who is later removed or suspended from that org would otherwise keep acting
under the stale org through every existing session — spending credits and
running agents under a tenancy they no longer belong to.

This module re-verifies ACTIVE membership at every turn-dispatch choke
point (the HTTP ``/stream`` handler and the queue-promotion hook), mirroring
the checks :func:`autogpt_libs.auth.get_request_context` runs for the
*header* org:

- No ACTIVE ``OrgMember`` for the session's org → access revoked. The caller
  surfaces this as an honest failure (HTTP 403 on the request path); we do
  NOT silently strip to personal context, which would run agents under the
  wrong tenancy without the user understanding.
- Team removal is routine, not revocation: a stale ``team_id`` (no ACTIVE
  ``TeamMember``) on a still-valid org is stripped to org-home, matching
  ``get_request_context``'s existing team fallback.
"""

import logging

from prisma.enums import OrgMemberStatus

from backend.data.db import prisma

logger = logging.getLogger(__name__)


class SessionOrgMembershipRevoked(Exception):
    """The user is no longer an ACTIVE member of the session's org.

    Carries the offending ``organization_id`` so callers can log/branch
    without re-deriving it.
    """

    def __init__(self, organization_id: str) -> None:
        self.organization_id = organization_id
        super().__init__(
            f"User is no longer an active member of organization {organization_id}"
        )


async def verify_session_org_membership(
    *,
    user_id: str,
    organization_id: str,
    team_id: str | None,
) -> str | None:
    """Re-verify ACTIVE membership for a session's persisted tenancy.

    Runs up to two indexed lookups on the membership unique constraints
    (``OrgMember@@unique(orgId, userId)`` and, only when *team_id* is set,
    ``TeamMember@@unique(teamId, userId)``).

    Returns the ``team_id`` the turn should run under: the input value when
    the team membership is still ACTIVE, or ``None`` (org-home) when the team
    is stale. Raises :class:`SessionOrgMembershipRevoked` when the user has
    no ACTIVE ``OrgMember`` row for *organization_id* — the org membership is
    a hard gate, the team is not.
    """
    org_member = await prisma.orgmember.find_unique(
        where={"orgId_userId": {"orgId": organization_id, "userId": user_id}},
    )
    if org_member is None or org_member.status != OrgMemberStatus.ACTIVE:
        raise SessionOrgMembershipRevoked(organization_id)

    if team_id is None:
        return None

    team_member = await prisma.teammember.find_unique(
        where={"teamId_userId": {"teamId": team_id, "userId": user_id}},
    )
    if team_member is None or team_member.status != OrgMemberStatus.ACTIVE:
        logger.info(
            "Stripping stale team %s to org-home for user %s in org %s",
            team_id,
            user_id,
            organization_id,
        )
        return None

    return team_id
