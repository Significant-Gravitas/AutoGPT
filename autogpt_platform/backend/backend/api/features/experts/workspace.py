from autogpt_libs.auth.models import RequestContext
from fastapi import HTTPException

from backend.api.features.orgs.db import get_user_default_team


async def is_personal_expert_workspace(ctx: RequestContext) -> bool:
    organization_id, team_id = await get_user_default_team(ctx.user_id)
    return bool(
        organization_id
        and ctx.org_id == organization_id
        and (ctx.team_id is None or ctx.team_id == team_id)
    )


async def require_personal_expert_workspace(ctx: RequestContext) -> None:
    if not await is_personal_expert_workspace(ctx):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "experts_personal_workspace_only",
                "message": "Experts currently belong to your personal workspace.",
            },
        )
