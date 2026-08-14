import autogpt_libs.auth as autogpt_auth_lib
import fastapi
from fastapi import APIRouter, Security
from pydantic import BaseModel, Field

from backend.api.features.experts import experts_db, scheduling
from backend.api.features.experts.models import (
    Expert,
    ExpertDetachPreview,
    ExpertRun,
    ExpertSoulUpdate,
    ExpertWorkflowRef,
    HireResult,
)

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


@router.get(
    "/{expert_id}/runs",
    operation_id="list_expert_runs",
    responses={404: {"description": "Expert not found"}},
)
async def list_expert_runs(
    expert_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> list[ExpertRun]:
    """Recent executions attributed to this expert, with a classified output
    type for the Work surface's typed viewer."""
    try:
        return await experts_db.list_expert_runs(user_id, expert_id)
    except experts_db.ExpertNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))


@router.patch(
    "/{expert_id}/soul",
    operation_id="update_expert_soul",
    responses={404: {"description": "Expert not found"}},
)
async def update_expert_soul(
    expert_id: str,
    request: ExpertSoulUpdate,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Expert:
    try:
        return await experts_db.update_soul(user_id, expert_id, request)
    except experts_db.ExpertNotFoundError as e:
        raise fastapi.HTTPException(status_code=404, detail=str(e))


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


@router.get(
    "/{expert_id}/detach-preview",
    operation_id="get_expert_detach_preview",
    responses={404: {"description": "Expert not found"}},
)
async def get_expert_detach_preview(
    expert_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ExpertDetachPreview:
    """What archiving this expert would pause — schedules and triggers —
    so the client can show a clear confirmation prompt."""
    expert = await experts_db.get_expert(user_id, expert_id, include_workflows=False)
    if expert is None:
        raise fastapi.HTTPException(status_code=404, detail="Expert not found")
    return await scheduling.get_detach_preview(user_id, expert_id)


@router.post(
    "/{expert_id}/schedules/resume",
    operation_id="resume_expert_schedules",
    responses={404: {"description": "Expert not found"}},
)
async def resume_expert_schedules(
    expert_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Expert:
    """One-click reversal of a budget/archive pause."""
    resumed = await scheduling.resume_expert_schedules(user_id, expert_id)
    if not resumed:
        raise fastapi.HTTPException(status_code=404, detail="Expert not found")
    expert = await experts_db.get_expert(user_id, expert_id)
    if expert is None:
        raise fastapi.HTTPException(status_code=404, detail="Expert not found")
    return expert


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
