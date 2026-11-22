from typing import Annotated

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Path, Query, Response, Security
from pydantic import BaseModel, Field

from backend.util.exceptions import NotFoundError
from backend.util.feature_flag import Flag, is_feature_enabled

from . import db, service
from .resolved import ResolvedPage
from .service import PageListItem

router = APIRouter(
    prefix="/pages",
    tags=["pages"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)

PageID = Annotated[str, Path(max_length=64)]
RunIDs = Annotated[
    list[str], Query(max_length=6, description="Runs started from the page, to follow.")
]


class PageListResponse(BaseModel):
    pages: list[PageListItem]


class PageEdit(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=120)
    order: list[int] | None = Field(
        None, max_length=12, description="Section indices, new order."
    )
    pin_to: str | None = Field(
        None, max_length=64, description="A library workflow id."
    )
    unpin: bool = False


@router.get("", operation_id="list_pages")
async def list_pages(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> PageListResponse:
    await _require_flag(user_id)
    return PageListResponse(pages=await service.list_pages(user_id))


@router.get("/pinned/{library_agent_id}", operation_id="get_pinned_page")
async def get_pinned_page(
    library_agent_id: PageID,
    run_id: RunIDs = [],
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ResolvedPage:
    await _require_flag(user_id)
    return await service.read_pinned_page(user_id, library_agent_id, run_ids=run_id)


@router.get("/{page_id}", operation_id="get_page")
async def get_page(
    page_id: PageID,
    run_id: RunIDs = [],
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ResolvedPage:
    await _require_flag(user_id)
    return await service.read_page(user_id, page_id, run_ids=run_id)


@router.patch("/{page_id}", operation_id="edit_page")
async def edit_page(
    page_id: PageID,
    edit: PageEdit,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ResolvedPage:
    await _require_flag(user_id)
    if edit.pin_to or edit.unpin:
        await db.set_page_pin(user_id, page_id, None if edit.unpin else edit.pin_to)
    if edit.title is not None or edit.order is not None:
        return await service.edit_page(
            user_id, page_id, title=edit.title, order=edit.order
        )
    return await service.read_page(user_id, page_id)


@router.delete("/{page_id}/sections/{index}", operation_id="remove_page_section")
async def remove_page_section(
    page_id: PageID,
    index: int = Path(ge=0, le=20),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ResolvedPage:
    await _require_flag(user_id)
    return await service.remove_section(user_id, page_id, index)


@router.post("/{page_id}/restore", operation_id="restore_page")
async def restore_page(
    page_id: PageID,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ResolvedPage:
    await _require_flag(user_id)
    record = await db.restore_previous(user_id, page_id)
    return await service.resolve_record(user_id, record)


@router.delete("/{page_id}", operation_id="delete_page", status_code=204)
async def delete_page(
    page_id: PageID,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Response:
    await _require_flag(user_id)
    if not await db.delete_page(user_id, page_id):
        raise NotFoundError(f"Page #{page_id} not found")
    return Response(status_code=204)


async def _require_flag(user_id: str) -> None:
    if not await is_feature_enabled(Flag.PAGES, user_id, default=False):
        raise NotFoundError("Not found")
