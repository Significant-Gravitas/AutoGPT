"""Per-turn re-verification of a chat session's persisted org/team.

**This module docstring is the single source of truth for the policy — call
sites reference it rather than restating it.**

A :class:`ChatSession` row is the authoritative tenancy for every turn in
it, but membership is only validated when the session is *created*. A user
who is later removed or suspended from that org would otherwise keep acting
under the stale org through every existing session — spending credits and
running agents under a tenancy they no longer belong to.

:func:`resolve_session_tenancy` is applied at complementary authorization
boundaries: HTTP ``/stream`` and pending-message handlers reject stale access
early, queue promotion checks queued work before claiming it, and the CoPilot
processor re-reads authoritative database metadata immediately before either
execution engine starts. Scheduled followups also check before creating a
fresh session. Together these boundaries cover non-HTTP dispatch and close
request-to-execution races while enforcing the same checks
:func:`autogpt_libs.auth.get_request_context` runs for the *header* org:

- Org is a hard gate. No ACTIVE ``OrgMember``, or a soft-deleted
  ``Organization``, means access was revoked; the caller surfaces this as an
  honest failure (HTTP 403 on the request path). We do NOT silently strip to
  personal context, which would run agents under the wrong tenancy without
  the user understanding.
- Team is a soft strip. Team removal is routine, not revocation: a stale
  ``team_id`` (no ACTIVE ``TeamMember``, or a team that no longer belongs to
  the session's org) on a still-valid org is stripped to org-home, matching
  ``get_request_context``'s existing team fallback.

The lookups go through :func:`backend.data.db_accessors.orgs_db` rather than
the global Prisma client: :func:`backend.copilot.turn_queue.dispatch_next_for_user`
runs inside the CoPilot executor, which has no Prisma connection and must
reach the database over the DatabaseManager RPC.
"""

import logging

from backend.data.db_accessors import orgs_db

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


async def resolve_session_tenancy(
    *,
    user_id: str,
    organization_id: str,
    team_id: str | None,
) -> str | None:
    """Re-verify a session's persisted tenancy and resolve the team to use.

    Returns the ``team_id`` the turn should run under: the input value when
    the team membership is still ACTIVE and the team still belongs to
    *organization_id*, or ``None`` (org-home) when the team is stale.

    Raises :class:`SessionOrgMembershipRevoked` when the org tenancy no
    longer holds — see the module docstring for why org is a hard gate and
    team is not.
    """
    membership = await orgs_db().get_session_tenancy_membership(
        user_id=user_id,
        organization_id=organization_id,
        team_id=team_id,
    )
    if not membership.org_active:
        raise SessionOrgMembershipRevoked(organization_id)

    if team_id is None:
        return None

    if not membership.team_active:
        logger.info(
            "Stripping stale team %s to org-home for user %s in org %s",
            team_id,
            user_id,
            organization_id,
        )
        return None

    return team_id
