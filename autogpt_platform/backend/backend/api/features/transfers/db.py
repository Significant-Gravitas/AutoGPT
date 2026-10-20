"""Database operations for resource transfer management."""

import logging
from datetime import UTC, datetime
from typing import NoReturn

from prisma import Prisma
from prisma.models import TransferRequest
from prisma.types import TransferRequestWhereInput

from backend.data.db import prisma, transaction
from backend.util.exceptions import NotFoundError
from backend.util.json import SafeJson

from .model import TransferResponse

logger = logging.getLogger(__name__)

_VALID_RESOURCE_TYPES = {"AgentGraph", "StoreListing"}
_OPEN_STATUSES = ["PENDING", "SOURCE_APPROVED", "TARGET_APPROVED"]
_TERMINAL_STATUSES = ["COMPLETED", "REJECTED", "CANCELLED"]


class _SourceNoLongerOwnsResource(Exception):
    """Raised inside the execute transaction to roll it back when the source
    org no longer owns the resource (e.g. a competing transfer ran first)."""


async def create_transfer(
    source_org_id: str,
    target_org_id: str,
    resource_type: str,
    resource_id: str,
    user_id: str,
    reason: str | None = None,
) -> TransferResponse:
    """Create a new transfer request from source org to target org.

    Validates:
    - resource_type is one of the allowed types
    - source and target orgs are different
    - target org exists
    - the resource exists and belongs to the source org
    """
    if resource_type not in _VALID_RESOURCE_TYPES:
        raise ValueError(
            f"Invalid resource_type '{resource_type}'. "
            f"Must be one of: {', '.join(sorted(_VALID_RESOURCE_TYPES))}"
        )

    if source_org_id == target_org_id:
        raise ValueError("Source and target organizations must be different")

    target_org = await prisma.organization.find_unique(where={"id": target_org_id})
    if target_org is None or target_org.deletedAt is not None:
        raise NotFoundError(f"Target organization {target_org_id} not found")

    await _validate_resource_ownership(resource_type, resource_id, source_org_id)

    tr = await prisma.transferrequest.create(
        data={
            "resourceType": resource_type,
            "resourceId": resource_id,
            "sourceOrganizationId": source_org_id,
            "targetOrganizationId": target_org_id,
            "initiatedByUserId": user_id,
            "status": "PENDING",
            "reason": reason,
        }
    )
    return TransferResponse.from_db(tr)


async def list_transfers(org_id: str) -> list[TransferResponse]:
    """List all transfer requests where org is source OR target."""
    transfers = await prisma.transferrequest.find_many(
        where={
            "OR": [
                {"sourceOrganizationId": org_id},
                {"targetOrganizationId": org_id},
            ]
        },
        order={"createdAt": "desc"},
    )
    return [TransferResponse.from_db(t) for t in transfers]


async def approve_transfer(
    transfer_id: str,
    user_id: str,
    org_id: str,
) -> TransferResponse:
    """Approve a transfer from the source or target side.

    - If user's active org is the source org, sets sourceApprovedByUserId.
    - If user's active org is the target org, sets targetApprovedByUserId.
    - Advances the status accordingly.
    """
    tr = await prisma.transferrequest.find_unique(where={"id": transfer_id})
    if tr is None:
        raise NotFoundError(f"Transfer request {transfer_id} not found")

    if tr.status in _TERMINAL_STATUSES:
        raise ValueError(f"Cannot approve a transfer with status '{tr.status}'")

    update_data: dict = {}
    where: TransferRequestWhereInput = {
        "id": transfer_id,
        "status": {"in": _OPEN_STATUSES},
    }

    if org_id == tr.sourceOrganizationId:
        if tr.sourceApprovedByUserId is not None:
            raise ValueError("Source organization has already approved this transfer")
        update_data["sourceApprovedByUserId"] = user_id
        where["sourceApprovedByUserId"] = None
        if tr.targetApprovedByUserId is not None:
            # Both sides approved — ready for execution (NOT completed yet)
            update_data["status"] = "TARGET_APPROVED"
        else:
            update_data["status"] = "SOURCE_APPROVED"

    elif org_id == tr.targetOrganizationId:
        if tr.targetApprovedByUserId is not None:
            raise ValueError("Target organization has already approved this transfer")
        update_data["targetApprovedByUserId"] = user_id
        where["targetApprovedByUserId"] = None
        if tr.sourceApprovedByUserId is not None:
            # Both sides approved — ready for execution (NOT completed yet)
            update_data["status"] = "SOURCE_APPROVED"
        else:
            update_data["status"] = "TARGET_APPROVED"

    else:
        raise ValueError("Your active organization is not a party to this transfer")

    approved = await prisma.transferrequest.update_many(where=where, data=update_data)
    if approved == 0:
        raise ValueError(
            "Transfer is no longer open or your organization already approved"
        )
    updated = await prisma.transferrequest.find_unique_or_raise(
        where={"id": transfer_id}
    )
    return TransferResponse.from_db(updated)


async def reject_transfer(
    transfer_id: str,
    user_id: str,
    org_id: str,
) -> TransferResponse:
    """Reject a pending transfer request. Caller must be in source or target org."""
    tr = await prisma.transferrequest.find_unique(where={"id": transfer_id})
    if tr is None:
        raise NotFoundError(f"Transfer request {transfer_id} not found")

    if tr.status in _TERMINAL_STATUSES:
        raise ValueError(f"Cannot reject a transfer with status '{tr.status}'")

    if org_id not in (tr.sourceOrganizationId, tr.targetOrganizationId):
        raise ValueError("Your active organization is not a party to this transfer")

    # Conditional, so a reject racing an execute can't flip a COMPLETED
    # transfer (whose resource has already moved) to REJECTED.
    rejected = await prisma.transferrequest.update_many(
        where={"id": transfer_id, "status": {"not_in": _TERMINAL_STATUSES}},
        data={"status": "REJECTED"},
    )
    if rejected == 0:
        raise ValueError("Transfer is no longer open")
    updated = await prisma.transferrequest.find_unique_or_raise(
        where={"id": transfer_id}
    )
    return TransferResponse.from_db(updated)


async def execute_transfer(
    transfer_id: str,
    user_id: str,
    org_id: str,
) -> TransferResponse:
    """Atomically execute a transfer approved by both organizations.

    The caller's active org must be a party, since TRANSFER_RESOURCES alone
    is granted to every personal-org owner.
    """
    tr = await prisma.transferrequest.find_unique(where={"id": transfer_id})
    if tr is None:
        raise NotFoundError(f"Transfer request {transfer_id} not found")

    if org_id not in (tr.sourceOrganizationId, tr.targetOrganizationId):
        raise ValueError("Your organization is not a party to this transfer request")

    if tr.sourceApprovedByUserId is None or tr.targetApprovedByUserId is None:
        raise ValueError(
            "Transfer requires approval from both source and target organizations"
        )

    if tr.status == "COMPLETED":
        raise ValueError("Transfer has already been executed")

    if tr.status in _TERMINAL_STATUSES:
        raise ValueError(f"Cannot execute a transfer with status '{tr.status}'")

    try:
        async with transaction() as tx:
            await _execute_in_transaction(tx, tr, user_id)
    except _SourceNoLongerOwnsResource:
        await _reject_stale_transfer(transfer_id)

    updated = await prisma.transferrequest.find_unique_or_raise(
        where={"id": transfer_id}
    )
    return TransferResponse.from_db(updated)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _execute_in_transaction(
    tx: Prisma, transfer: TransferRequest, user_id: str
) -> None:
    """Move, complete, audit and close competing requests in the caller's transaction.

    Lock the resource before any TransferRequest row to serialize competing executes.
    """
    try:
        await _validate_resource_ownership(
            transfer.resourceType,
            transfer.resourceId,
            transfer.sourceOrganizationId,
            db=tx,
        )
    except ValueError as e:
        raise _SourceNoLongerOwnsResource() from e

    moved = await _move_resource(
        resource_type=transfer.resourceType,
        resource_id=transfer.resourceId,
        source_org_id=transfer.sourceOrganizationId,
        target_org_id=transfer.targetOrganizationId,
        db=tx,
    )
    if moved == 0:
        raise _SourceNoLongerOwnsResource()

    await _claim_completion(tx, transfer.id)
    await _create_audit_logs(transfer=transfer, actor_user_id=user_id, db=tx)
    await _reject_competing_transfers(tx, transfer)


async def _claim_completion(tx: Prisma, transfer_id: str) -> None:
    """Claim an open, fully approved request after its resource has been locked."""
    claimed = await tx.transferrequest.update_many(
        where={
            "id": transfer_id,
            "status": {"not_in": _TERMINAL_STATUSES},
            "sourceApprovedByUserId": {"not": None},
            "targetApprovedByUserId": {"not": None},
        },
        data={"status": "COMPLETED", "completedAt": datetime.now(UTC)},
    )
    if claimed == 0:
        raise ValueError("Transfer is no longer open for execution")


async def _reject_competing_transfers(tx: Prisma, transfer: TransferRequest) -> None:
    """Close other open requests whose source can no longer transfer this resource."""
    await tx.transferrequest.update_many(
        where={
            "id": {"not": transfer.id},
            "resourceType": transfer.resourceType,
            "resourceId": transfer.resourceId,
            "status": {"in": _OPEN_STATUSES},
        },
        data={"status": "REJECTED"},
    )


async def _reject_stale_transfer(transfer_id: str) -> NoReturn:
    """Reject after rollback without overwriting a concurrent terminal state."""
    rejected = await prisma.transferrequest.update_many(
        where={"id": transfer_id, "status": {"not_in": _TERMINAL_STATUSES}},
        data={"status": "REJECTED"},
    )
    if rejected == 0:
        raise ValueError("Transfer is no longer open for execution")
    raise ValueError(
        "Resource was deleted or no longer belongs to the source "
        "organization; the transfer has been rejected"
    )


async def _validate_resource_ownership(
    resource_type: str, resource_id: str, org_id: str, db: Prisma | None = None
) -> None:
    """Verify the resource exists and belongs to the given org."""
    db = db if db is not None else prisma
    if resource_type == "AgentGraph":
        graph = await db.agentgraph.find_first(
            where={"id": resource_id, "isActive": True}
        )
        if graph is None:
            raise NotFoundError(f"AgentGraph '{resource_id}' not found")
        if graph.organizationId != org_id:
            raise ValueError("AgentGraph does not belong to the source organization")

    elif resource_type == "StoreListing":
        listing = await db.storelisting.find_unique(where={"id": resource_id})
        if listing is None or listing.isDeleted:
            raise NotFoundError(f"StoreListing '{resource_id}' not found")
        if listing.owningOrgId != org_id:
            raise ValueError("StoreListing does not belong to the source organization")


async def _move_resource(
    resource_type: str,
    resource_id: str,
    source_org_id: str,
    target_org_id: str,
    db: Prisma | None = None,
) -> int:
    """Move the resource from the source to the target organization.

    Returns the number of rows moved; 0 means the source org no longer owns it.
    """
    db = db if db is not None else prisma
    if resource_type == "AgentGraph":
        # Move ALL versions, not just the active one, and land the graph at
        # org-home (teamId=None) — a stale source-org teamId would fail every
        # clause of the target org's visibility_filter, making the graph
        # invisible to all target-org members.
        return await db.agentgraph.update_many(
            where={"id": resource_id, "organizationId": source_org_id},
            data={"organizationId": target_org_id, "teamId": None},
        )

    elif resource_type == "StoreListing":
        return await db.storelisting.update_many(
            where={"id": resource_id, "owningOrgId": source_org_id},
            data={"owningOrgId": target_org_id},
        )

    return 0


async def _create_audit_logs(
    transfer, actor_user_id: str, db: Prisma | None = None
) -> None:
    """Create audit log entries for both source and target organizations."""
    db = db if db is not None else prisma
    # Json columns need SafeJson; a bare dict is rejected by prisma (#15267).
    after_json = SafeJson(
        {
            "resourceType": transfer.resourceType,
            "resourceId": transfer.resourceId,
            "sourceOrganizationId": transfer.sourceOrganizationId,
            "targetOrganizationId": transfer.targetOrganizationId,
        }
    )
    common = {
        "actorUserId": actor_user_id,
        "entityType": "TransferRequest",
        "entityId": transfer.id,
        "action": "TRANSFER_EXECUTED",
        "afterJson": after_json,
        "correlationId": transfer.id,
    }

    for organization_id in (
        transfer.sourceOrganizationId,
        transfer.targetOrganizationId,
    ):
        await db.auditlog.create(
            data={
                **common,
                "organizationId": organization_id,
                "beforeJson": SafeJson(
                    {"organizationId": transfer.sourceOrganizationId}
                ),
            }
        )
