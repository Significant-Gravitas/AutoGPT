from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, Path, Query, Security

from backend.copilot.ui_drafts import (
    UIDraftResponse,
    UIDraftWrite,
    read_draft,
    write_draft,
)

router = APIRouter(tags=["chat"], dependencies=[Security(auth.requires_user)])
ViewID = Annotated[str, Path(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,79}$")]
UserID = Annotated[str, Security(auth.get_user_id)]


@router.get("/sessions/{session_id}/ui-drafts/{view_id}")
async def get_ui_draft(
    session_id: str,
    view_id: ViewID,
    user_id: UserID,
    tool_call_id: Annotated[str, Query(min_length=1, max_length=256)],
) -> UIDraftResponse:
    return await read_draft(user_id, session_id, view_id, tool_call_id)


@router.put("/sessions/{session_id}/ui-drafts/{view_id}")
async def save_ui_draft(
    session_id: str, view_id: ViewID, user_id: UserID, request: UIDraftWrite
) -> UIDraftResponse:
    return await write_draft(user_id, session_id, view_id, request)
