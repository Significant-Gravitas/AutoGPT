import autogpt_libs.auth as autogpt_auth_lib
from autogpt_libs.auth.models import RequestContext
from fastapi import APIRouter, Security

from backend.api.live_auth import requires_live_resource_permission

from .models import HomeDashboardResponse
from .service import build_home_dashboard

router = APIRouter(
    prefix="/home",
    tags=["home", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get("", operation_id="get_home_dashboard")
async def get_home_dashboard(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    ctx: RequestContext = requires_live_resource_permission(
        autogpt_auth_lib.OrgAction.VIEW_RESOURCES,
        autogpt_auth_lib.TeamAction.VIEW_AGENTS,
    ),
) -> HomeDashboardResponse:
    return await build_home_dashboard(
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=ctx.team_id,
    )
