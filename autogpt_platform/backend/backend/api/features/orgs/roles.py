"""Who may hand out an organization's roles.

Only the owner makes someone an admin or a billing manager, or takes that
away, whether on a member or on an invitation. Admins still manage plain
members: they invite them and remove them.
"""

from autogpt_libs.auth.models import RequestContext

from backend.util.exceptions import NotAuthorizedError

OWNER_ONLY_ROLES = (
    "Only the organization's owner can make someone an admin or billing "
    "manager, or take that role away."
)


def require_owner_for_role_change(
    actor: RequestContext | None,
    *,
    is_admin: bool,
    is_billing_manager: bool,
    wants_admin: bool | None,
    wants_billing_manager: bool | None,
) -> None:
    """Refuse a role change by anyone but the owner. ``actor`` is the live
    context taken under the membership lock; None is a system change."""
    changes_admin = wants_admin is not None and wants_admin != is_admin
    changes_billing = (
        wants_billing_manager is not None
        and wants_billing_manager != is_billing_manager
    )
    if actor is None or actor.is_org_owner or not (changes_admin or changes_billing):
        return
    raise NotAuthorizedError(OWNER_ONLY_ROLES)
