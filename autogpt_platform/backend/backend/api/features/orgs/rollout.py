"""The one gate on organizations: whether an account may use them."""

from fastapi import HTTPException

from backend.data.org_access import has_org_access
from backend.util.feature_flag import Flag, _env_flag_override, is_feature_enabled


async def require_org_collaboration(user_id: str) -> None:
    if not await org_collaboration_enabled(user_id):
        raise HTTPException(
            status_code=403,
            detail="Organization collaboration is not enabled for this account",
        )


async def org_collaboration_enabled(user_id: str) -> bool:
    """Whether *user_id* may use organizations.

    An explicit ``FORCE_FLAG_SHOW_ORG_SETTINGS`` decides for every account on
    an installation (the flag read applies it). Otherwise an admin's grant
    turns organizations on (the account's own, or one a team organization it
    belongs to or is invited to passes on), and the rollout flag decides the
    rest.
    """
    if await org_access_granted(user_id):
        return True
    return await is_feature_enabled(Flag.SHOW_ORG_SETTINGS, user_id, default=False)


async def org_access_granted(user_id: str) -> bool:
    """An admin's grant applies unless an env override decides for everyone."""
    if _env_flag_override(Flag.SHOW_ORG_SETTINGS) is not None:
        return False
    return await has_org_access(user_id)
