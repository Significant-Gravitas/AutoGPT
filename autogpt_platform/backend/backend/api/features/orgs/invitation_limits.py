"""How often an organization may send invitations.

Creating and resending share one budget per organization, and resending one
invitation has a smaller budget of its own, so nobody can mail an address over
and over. Like every ``RateLimiter``, these fail open when Redis is down.
"""

from backend.api.utils.rate_limit import RateLimiter

org_invitation_limiter = RateLimiter(
    "org-invitations", max_requests=60, window_seconds=3600
)
invitation_resend_limiter = RateLimiter(
    "org-invitation-resends", max_requests=3, window_seconds=3600
)


async def check_invitation_rate(org_id: str, invitation_id: str | None = None) -> None:
    """Raise 429 when *org_id* has sent too many invitations, or this one has
    been resent too often."""
    await org_invitation_limiter.check(org_id)
    if invitation_id is not None:
        await invitation_resend_limiter.check(invitation_id)
