"""Admin endpoints for the IPs and email domains refused at sign-up and sign-in."""

from autogpt_libs.auth import requires_admin_user
from autogpt_libs.auth.models import User as AuthUser
from fastapi import APIRouter, Query, Security
from prisma.enums import AuthAccessBlockKind
from pydantic import AwareDatetime, BaseModel, Field

from backend.data.access_blocks import (
    DEFAULT_PAGE_SIZE,
    MAX_REASON_LENGTH,
    AccessBlock,
    AccessBlockPage,
    create_access_block,
    delete_access_block,
    list_active_access_blocks,
)

router = APIRouter(
    prefix="/admin/access-blocks",
    dependencies=[Security(requires_admin_user)],
)


class CreateAccessBlockRequest(BaseModel):
    kind: AuthAccessBlockKind
    value: str = Field(min_length=1, max_length=255)
    reason: str = Field(min_length=1, max_length=MAX_REASON_LENGTH)
    # A naive time can't be compared with now; the client sends UTC.
    expires_at: AwareDatetime | None = None


@router.get("", summary="List access blocks")
async def get_access_blocks(
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> AccessBlockPage:
    """The IP and email domain blocks in force, newest first. Admin-only."""
    return await list_active_access_blocks(limit=limit, offset=offset)


@router.post("", summary="Create access block", status_code=201)
async def post_access_block(
    request: CreateAccessBlockRequest,
    admin: AuthUser = Security(requires_admin_user),
) -> AccessBlock:
    """Refuse new sign-ups (and, for an IP, sign-ins) from an IP or email
    domain. 409 while a block on the same value is in force. Admin-only."""
    return await create_access_block(
        kind=request.kind,
        value=request.value,
        reason=request.reason,
        expires_at=request.expires_at,
        created_by_id=admin.user_id,
    )


@router.delete("/{block_id}", summary="Delete access block", status_code=204)
async def remove_access_block(block_id: str) -> None:
    """Lift a block. Admin-only."""
    await delete_access_block(block_id)
