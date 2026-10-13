"""Edits to saved OpenUI view files (``.openui``)."""

from typing import Annotated

from autogpt_libs.auth.dependencies import get_user_id, requires_user
from fastapi import APIRouter, Security

from backend.copilot.ui_view_files import (
    UIFileStateResponse,
    UIFileStateWrite,
    read_file_state,
    write_file_state,
)

router = APIRouter(dependencies=[Security(requires_user)])
UserID = Annotated[str, Security(get_user_id)]


@router.get(
    "/files/{file_id}/view-state",
    summary="Get a saved view and its edits",
    operation_id="getWorkspaceFileViewState",
    responses={404: {"description": "Saved view not found"}},
)
async def get_file_view_state(file_id: str, user_id: UserID) -> UIFileStateResponse:
    return await read_file_state(user_id, file_id)


@router.put(
    "/files/{file_id}/view-state",
    summary="Save edits to a saved view",
    operation_id="putWorkspaceFileViewState",
    responses={
        404: {"description": "Saved view not found"},
        409: {"description": "The view or its edits changed; reload first"},
    },
)
async def put_file_view_state(
    file_id: str, user_id: UserID, request: UIFileStateWrite
) -> UIFileStateResponse:
    return await write_file_state(user_id, file_id, request)
