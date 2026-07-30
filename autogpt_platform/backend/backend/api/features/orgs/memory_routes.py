"""Org shared-memory governance routes (nested under /api/orgs/{org_id}/memory).

Org-admin facing review queue for *tentative* ("held") shared memories. Gated
with ``requires_org_permission(OrgAction.MANAGE_MEMBERS)`` — reviewing and
ratifying member-submitted org/team memory is a governance action over org
content, and MANAGE_MEMBERS is the closest management action (resolves to
{owner, admin}, matching "org admins only"). Personal tiers are never exposed.
"""

from typing import Annotated

from autogpt_libs.auth import requires_org_permission
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction
from fastapi import APIRouter, HTTPException, Query, Security

from . import memory_db
from .memory_model import HeldMemoryListResponse, MemoryActionResult

router = APIRouter()


def _verify_org_path(ctx: RequestContext, org_id: str) -> None:
    """MANAGE_MEMBERS only proves the caller administers their *active* org
    (ctx.org_id); without this an admin of org A could review org B's queue."""
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")


@router.get(
    "/held",
    summary="List held (tentative) shared memories",
    tags=["orgs", "memory"],
)
async def list_held_memories(
    org_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_MEMBERS)),
    ],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> HeldMemoryListResponse:
    _verify_org_path(ctx, org_id)
    return await memory_db.list_held_memories(org_id, limit)


@router.post(
    "/held/{memory_id}/approve",
    summary="Approve a held memory (ratify tentative → active)",
    tags=["orgs", "memory"],
)
async def approve_held_memory(
    org_id: str,
    memory_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_MEMBERS)),
    ],
) -> MemoryActionResult:
    _verify_org_path(ctx, org_id)
    return await memory_db.approve_held_memory(org_id, memory_id, ctx.user_id)


@router.post(
    "/held/{memory_id}/reject",
    summary="Reject a held memory (soft-retract)",
    tags=["orgs", "memory"],
)
async def reject_held_memory(
    org_id: str,
    memory_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_MEMBERS)),
    ],
) -> MemoryActionResult:
    _verify_org_path(ctx, org_id)
    return await memory_db.reject_held_memory(org_id, memory_id, ctx.user_id)
