from fastapi import HTTPException

from backend.util.feature_flag import Flag, is_feature_enabled


async def require_org_collaboration(user_id: str) -> None:
    if not await is_feature_enabled(Flag.SHOW_ORG_SETTINGS, user_id, default=False):
        raise HTTPException(
            status_code=403,
            detail="Organization collaboration is not enabled for this account",
        )
