"""Database operations for resource transfer management."""

import logging
from datetime import datetime, timezone

from backend.data.db import prisma
from backend.util.exceptions import NotFoundError

from .model import TransferResponse

logger = logging.getLogger(__name__)

_VALID_RESOURCE_TYPES = {"AgentGraph", "StoreListing"}


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

    if tr.status in ("COMPLETED", "REJECTED"):
        raise ValueError(f"Cannot approve a transfer with status '{tr.status}'")

    update_data: dict = {}

    if org_id == tr.sourceOrganizationId:
        if tr.sourceApprovedByUserId is not None:
            raise ValueError("Source organization has already approved this transfer")
        update_data["sourceApprovedByUserId"] = user_id
        if tr.targetApprovedByUserId is not None:
            # Both sides approved — ready for execution (NOT completed yet)
            update_data["status"] = "TARGET_APPROVED"
        else:
            update_data["status"] = "SOURCE_APPROVED"

    elif org_id == tr.targetOrganizationId:
        if tr.targetApprovedByUserId is not None:
            raise ValueError("Target organization has already approved this transfer")
        update_data["targetApprovedByUserId"] = user_id
        if tr.sourceApprovedByUserId is not None:
            # Both sides approved — ready for execution (NOT completed yet)
            update_data["status"] = "SOURCE_APPROVED"
        else:
            update_data["status"] = "TARGET_APPROVED"

    else:
        raise ValueError("Your active organization is not a party to this transfer")

    updated = await prisma.transferrequest.update(
        where={"id": transfer_id},
        data=update_data,
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

    if tr.status in ("COMPLETED", "REJECTED"):
        raise ValueError(f"Cannot reject a transfer with status '{tr.status}'")

    if org_id not in (tr.sourceOrganizationId, tr.targetOrganizationId):
        raise ValueError("Your active organization is not a party to this transfer")

    updated = await prisma.transferrequest.update(
        where={"id": transfer_id},
        data={"status": "REJECTED"},
    )
    return TransferResponse.from_db(updated)


async def execute_transfer(
    transfer_id: str,
    user_id: str,
) -> TransferResponse:
    """Execute an approved transfer -- move the resource to the target org.

    Requires both source and target approvals. Updates the resource's
    organization ownership and creates AuditLog entries for both orgs.
    """
    tr = await prisma.transferrequest.find_unique(where={"id": transfer_id})
    if tr is None:
        raise NotFoundError(f"Transfer request {transfer_id} not found")

    if tr.sourceApprovedByUserId is None or tr.targetApprovedByUserId is None:
        raise ValueError(
            "Transfer requires approval from both source and target organizations"
        )

    if tr.status == "COMPLETED":
        raise ValueError("Transfer has already been executed")

    if tr.status == "REJECTED":
        raise ValueError("Cannot execute a rejected transfer")

    await _move_resource(
        resource_type=tr.resourceType,
        resource_id=tr.resourceId,
        target_org_id=tr.targetOrganizationId,
    )

    now = datetime.now(timezone.utc)
    updated = await prisma.transferrequest.update(
        where={"id": transfer_id},
        data={"status": "COMPLETED", "completedAt": now},
    )

    await _create_audit_logs(
        transfer=updated,
        actor_user_id=user_id,
    )

    return TransferResponse.from_db(updated)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _validate_resource_ownership(
    resource_type: str, resource_id: str, org_id: str
) -> None:
    """Verify the resource exists and belongs to the given org."""
    if resource_type == "AgentGraph":
        graph = await prisma.agentgraph.find_first(
            where={"id": resource_id, "isActive": True}
        )
        if graph is None:
            raise NotFoundError(f"AgentGraph '{resource_id}' not found")
        if graph.organizationId != org_id:
            raise ValueError("AgentGraph does not belong to the source organization")

    elif resource_type == "StoreListing":
        listing = await prisma.storelisting.find_unique(where={"id": resource_id})
        if listing is None or listing.isDeleted:
            raise NotFoundError(f"StoreListing '{resource_id}' not found")
        if listing.owningOrgId != org_id:
            raise ValueError("StoreListing does not belong to the source organization")


async def _move_resource(
    resource_type: str,
    resource_id: str,
    target_org_id: str,
) -> None:
    """Move the resource to the target organization."""
    if resource_type == "AgentGraph":
        await prisma.agentgraph.update_many(
            where={"id": resource_id, "isActive": True},
            data={"organizationId": target_org_id},
        )

    elif resource_type == "StoreListing":
        await prisma.storelisting.update(
            where={"id": resource_id},
            data={"owningOrgId": target_org_id},
        )


async def _create_audit_logs(transfer, actor_user_id: str) -> None:
    """Create audit log entries for both source and target organizations."""
    common = {
        "actorUserId": actor_user_id,
        "entityType": "TransferRequest",
        "entityId": transfer.id,
        "action": "TRANSFER_EXECUTED",
        "afterJson": {
            "resourceType": transfer.resourceType,
            "resourceId": transfer.resourceId,
            "sourceOrganizationId": transfer.sourceOrganizationId,
            "targetOrganizationId": transfer.targetOrganizationId,
        },
        "correlationId": transfer.id,
    }

    await prisma.auditlog.create(
        data={
            **common,
            "organizationId": transfer.sourceOrganizationId,
            "beforeJson": {"organizationId": transfer.sourceOrganizationId},
        }
    )

    await prisma.auditlog.create(
        data={
            **common,
            "organizationId": transfer.targetOrganizationId,
            "beforeJson": {"organizationId": transfer.sourceOrganizationId},
        }
    )
