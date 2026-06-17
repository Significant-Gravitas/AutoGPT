"""Resource transfer API routes."""

from typing import Annotated

from autogpt_libs.auth import requires_org_permission
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction
from fastapi import APIRouter, Security

from . import db as transfer_db
from .model import CreateTransferRequest, TransferResponse

router = APIRouter()


@router.post(
    "",
    summary="Create transfer request",
    tags=["transfers"],
)
async def create_transfer(
    request: CreateTransferRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.TRANSFER_RESOURCES)),
    ],
) -> TransferResponse:
    return await transfer_db.create_transfer(
        source_org_id=ctx.org_id,
        target_org_id=request.target_organization_id,
        resource_type=request.resource_type,
        resource_id=request.resource_id,
        user_id=ctx.user_id,
        reason=request.reason,
    )


@router.get(
    "",
    summary="List transfers for active organization",
    tags=["transfers"],
)
async def list_transfers(
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.TRANSFER_RESOURCES)),
    ],
) -> list[TransferResponse]:
    return await transfer_db.list_transfers(ctx.org_id)


@router.post(
    "/{transfer_id}/approve",
    summary="Approve transfer request",
    tags=["transfers"],
)
async def approve_transfer(
    transfer_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.TRANSFER_RESOURCES)),
    ],
) -> TransferResponse:
    return await transfer_db.approve_transfer(
        transfer_id=transfer_id,
        user_id=ctx.user_id,
        org_id=ctx.org_id,
    )


@router.post(
    "/{transfer_id}/reject",
    summary="Reject transfer request",
    tags=["transfers"],
)
async def reject_transfer(
    transfer_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.TRANSFER_RESOURCES)),
    ],
) -> TransferResponse:
    return await transfer_db.reject_transfer(
        transfer_id=transfer_id,
        user_id=ctx.user_id,
        org_id=ctx.org_id,
    )


@router.post(
    "/{transfer_id}/execute",
    summary="Execute approved transfer",
    tags=["transfers"],
)
async def execute_transfer(
    transfer_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.TRANSFER_RESOURCES)),
    ],
) -> TransferResponse:
    return await transfer_db.execute_transfer(
        transfer_id=transfer_id,
        user_id=ctx.user_id,
    )
