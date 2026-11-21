import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Path, Query, Response, Security

from backend.util.exceptions import NotFoundError
from backend.util.feature_flag import Flag, is_feature_enabled

from . import db as pages_db
from .page import ResolvedWorkflowPage
from .service import read_page, remove_section, restore_previous

router = APIRouter(
    prefix="/workflow-pages",
    tags=["workflow-pages"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get("/{library_agent_id}", operation_id="get_workflow_page")
async def get_workflow_page(
    library_agent_id: str = Path(max_length=64),
    run_id: str | None = Query(
        None, max_length=64, description="Feed the result from this run."
    ),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ResolvedWorkflowPage:
    await _require_flag(user_id)
    return await read_page(user_id, library_agent_id, run_id=run_id)


@router.delete(
    "/{library_agent_id}/sections/{index}",
    operation_id="remove_workflow_page_section",
)
async def remove_workflow_page_section(
    library_agent_id: str = Path(max_length=64),
    index: int = Path(ge=0, le=20),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ResolvedWorkflowPage:
    await _require_flag(user_id)
    return await remove_section(user_id, library_agent_id, index)


@router.post("/{library_agent_id}/restore", operation_id="restore_workflow_page")
async def restore_workflow_page(
    library_agent_id: str = Path(max_length=64),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ResolvedWorkflowPage:
    await _require_flag(user_id)
    return await restore_previous(user_id, library_agent_id)


@router.delete(
    "/{library_agent_id}", operation_id="delete_workflow_page", status_code=204
)
async def delete_workflow_page(
    library_agent_id: str = Path(max_length=64),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Response:
    await _require_flag(user_id)
    if not await pages_db.delete_workflow_page(user_id, library_agent_id):
        raise NotFoundError(f"No page for library agent #{library_agent_id}")
    return Response(status_code=204)


async def _require_flag(user_id: str) -> None:
    if not await is_feature_enabled(Flag.WORKFLOW_PAGES, user_id, default=False):
        raise NotFoundError("Not found")
