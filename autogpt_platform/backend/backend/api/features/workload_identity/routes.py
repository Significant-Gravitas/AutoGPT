"""Manage the workload identities that may exchange OIDC tokens for API access."""

from typing import Annotated, Literal

from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from fastapi import APIRouter, Security
from prisma.enums import WorkloadIdentityStatus
from pydantic import BaseModel

from backend.data.auth import workload_identity as workload_identity_db
from backend.data.auth.workload_identity import (
    WorkloadIdentityInfo,
    WorkloadIdentitySpec,
)

router = APIRouter(dependencies=[Security(requires_user)])


class SetWorkloadIdentityStatusRequest(BaseModel):
    status: Literal[WorkloadIdentityStatus.ACTIVE, WorkloadIdentityStatus.DISABLED]


@router.post("", summary="Trust a workload identity")
async def create_workload_identity(
    spec: WorkloadIdentitySpec,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> WorkloadIdentityInfo:
    """Let a workload act as you on the External API by exchanging the OIDC
    tokens its platform issues, with no API key to store or rotate."""
    return await workload_identity_db.create_workload_identity(
        user_id, spec, organization_id=ctx.org_id or None
    )


@router.get("", summary="List workload identities")
async def list_workload_identities(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[WorkloadIdentityInfo]:
    return await workload_identity_db.list_workload_identities(
        user_id, organization_id=ctx.org_id or None
    )


@router.get("/{identity_id}", summary="Get a workload identity")
async def get_workload_identity(
    identity_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> WorkloadIdentityInfo:
    return await workload_identity_db.get_user_workload_identity(
        identity_id, user_id, organization_id=ctx.org_id or None
    )


@router.patch("/{identity_id}", summary="Enable or disable a workload identity")
async def set_workload_identity_status(
    identity_id: str,
    request: SetWorkloadIdentityStatusRequest,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> WorkloadIdentityInfo:
    """A disabled identity can't exchange tokens, and the tokens it already
    holds stop working until it is enabled again."""
    return await workload_identity_db.set_workload_identity_status(
        identity_id, user_id, request.status, organization_id=ctx.org_id or None
    )


@router.delete("/{identity_id}", summary="Revoke a workload identity")
async def revoke_workload_identity(
    identity_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> WorkloadIdentityInfo:
    """Revoke the identity and every token minted for it, permanently."""
    return await workload_identity_db.revoke_workload_identity(
        identity_id, user_id, organization_id=ctx.org_id or None
    )
