import autogpt_libs.auth as autogpt_auth_lib
from autogpt_libs.auth import get_request_context
from autogpt_libs.auth.models import RequestContext
from fastapi import APIRouter, Security

from backend.data.tenancy import get_user_team_ids

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
    ctx: RequestContext = Security(get_request_context),
) -> HomeDashboardResponse:
    team_ids = await get_user_team_ids(user_id, ctx.org_id) if ctx.org_id else []
    return await build_home_dashboard(
        user_id=user_id,
        organization_id=ctx.org_id,
        team_ids=team_ids,
    )
