import autogpt_libs.auth as autogpt_auth_lib
import fastapi
from fastapi import APIRouter, Security
from pydantic import BaseModel, Field

from backend.api.features.experts import experts_db
from backend.api.features.experts.models import Expert, ExpertWorkflowRef, HireResult

router = APIRouter(
    prefix="/experts",
    tags=["experts", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


class HireRequest(BaseModel):
    template_id: str
    name: str | None = Field(default=None, max_length=100)


class InstallWorkflowRequest(BaseModel):
    store_listing_version_id: str


@router.get("/templates", operation_id="list_expert_templates")
async def list_expert_templates() -> list[Expert]:
    return await experts_db.list_templates()


@router.post(
    "",
    operation_id="hire_expert",
    responses={404: {"description": "Expert template not found"}},
)
async def hire_expert(
    request: HireRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> HireResult:
    try:
        return await experts_db.hire_expert(user_id, request.template_id, request.name)
    except experts_db.ExpertTemplateNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))


@router.get("", operation_id="list_experts")
async def list_experts(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> list[Expert]:
    return await experts_db.list_experts(user_id)


@router.get(
    "/{expert_id}",
    operation_id="get_expert",
    responses={404: {"description": "Expert not found"}},
)
async def get_expert(
    expert_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Expert:
    expert = await experts_db.get_expert(user_id, expert_id)
    if expert is None:
        raise fastapi.HTTPException(status_code=404, detail="Expert not found")
    return expert


@router.post(
    "/{expert_id}/workflows",
    operation_id="install_expert_workflow",
    responses={404: {"description": "Expert not found"}},
)
async def install_expert_workflow(
    expert_id: str,
    request: InstallWorkflowRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ExpertWorkflowRef:
    try:
        return await experts_db.install_workflow(
            user_id, expert_id, request.store_listing_version_id
        )
    except experts_db.ExpertNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))


@router.delete(
    "/{expert_id}",
    operation_id="archive_expert",
    status_code=204,
    responses={404: {"description": "Expert not found"}},
)
async def archive_expert(
    expert_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> fastapi.Response:
    try:
        await experts_db.archive_expert(user_id, expert_id)
    except experts_db.ExpertNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))
    return fastapi.Response(status_code=204)
