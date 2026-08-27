"""Database operations for resource transfer management."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import cast

from prisma import Json, Prisma
from prisma import types as prisma_types
from prisma.models import TransferRequest

from backend.blocks.agent import AgentExecutorBlock
from backend.data.db import TRANSACTION_TIMEOUT, execute_raw_with_schema, prisma
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import NotFoundError

from .model import TransferResponse

logger = logging.getLogger(__name__)

_VALID_RESOURCE_TYPES = {"AgentGraph", "StoreListing"}


async def _count_incoming_graph_references(db: Prisma, graph_id: str) -> int:
    return await db.agentnode.count(
        where={
            "agentBlockId": AgentExecutorBlock().id,
            "constantInput": cast(
                prisma_types.JsonFilter,
                {
                    "path": ["graph_id"],
                    "equals": Json(graph_id),
                },
            ),
        }
    )


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

    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        await _lock_live_users(tx, user_id)
        await _lock_live_organizations(tx, source_org_id, target_org_id)
        await _assert_live_transfer_admin(tx, user_id, source_org_id)
        await _lock_transfer_resource(tx, resource_type, resource_id)
        await _validate_resource_ownership(
            resource_type, resource_id, source_org_id, client=tx
        )
        existing = await tx.transferrequest.find_first(
            where={
                "resourceType": resource_type,
                "resourceId": resource_id,
                "status": {"in": ["PENDING", "SOURCE_APPROVED", "TARGET_APPROVED"]},
            }
        )
        if existing is not None:
            raise ValueError("This resource already has an open transfer request")

        transfer = await tx.transferrequest.create(
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
        return TransferResponse.from_db(transfer)


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
    async with _locked_transfer(transfer_id) as (tx, tr):
        if tr.status in ("COMPLETED", "REJECTED"):
            raise ValueError(f"Cannot approve a transfer with status '{tr.status}'")

        update_data: dict[str, str] = {}
        if org_id == tr.sourceOrganizationId:
            if tr.sourceApprovedByUserId is not None:
                raise ValueError(
                    "Source organization has already approved this transfer"
                )
            update_data = {
                "sourceApprovedByUserId": user_id,
                "status": (
                    "TARGET_APPROVED"
                    if tr.targetApprovedByUserId is not None
                    else "SOURCE_APPROVED"
                ),
            }
        elif org_id == tr.targetOrganizationId:
            if tr.targetApprovedByUserId is not None:
                raise ValueError(
                    "Target organization has already approved this transfer"
                )
            update_data = {
                "targetApprovedByUserId": user_id,
                "status": (
                    "SOURCE_APPROVED"
                    if tr.sourceApprovedByUserId is not None
                    else "TARGET_APPROVED"
                ),
            }
        else:
            raise ValueError("Your active organization is not a party to this transfer")

        await _lock_live_users(tx, user_id)
        await _lock_live_organizations(tx, org_id)
        await _assert_live_transfer_admin(tx, user_id, org_id)
        updated = await tx.transferrequest.update(
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
    async with _locked_transfer(transfer_id) as (tx, tr):
        if tr.status in ("COMPLETED", "REJECTED"):
            raise ValueError(f"Cannot reject a transfer with status '{tr.status}'")

        if org_id not in (tr.sourceOrganizationId, tr.targetOrganizationId):
            raise ValueError("Your active organization is not a party to this transfer")

        await _lock_live_users(tx, user_id)
        await _lock_live_organizations(tx, org_id)
        await _assert_live_transfer_admin(tx, user_id, org_id)
        updated = await tx.transferrequest.update(
            where={"id": transfer_id},
            data={"status": "REJECTED"},
        )
        return TransferResponse.from_db(updated)


async def execute_transfer(
    transfer_id: str,
    user_id: str,
    org_id: str,
) -> TransferResponse:
    """Execute an approved transfer -- move the resource to the target org.

    Requires both source and target approvals, and the caller's active org
    must be a party to the transfer — TRANSFER_RESOURCES is granted to every
    personal-org owner, so without this check any authenticated user could
    execute an approved transfer between two unrelated orgs.
    """
    async with _locked_transfer(transfer_id) as (tx, tr):
        if org_id not in (tr.sourceOrganizationId, tr.targetOrganizationId):
            raise ValueError(
                "Your organization is not a party to this transfer request"
            )

        if tr.sourceApprovedByUserId is None or tr.targetApprovedByUserId is None:
            raise ValueError(
                "Transfer requires approval from both source and target organizations"
            )
        if tr.status == "COMPLETED":
            raise ValueError("Transfer has already been executed")
        if tr.status == "REJECTED":
            raise ValueError("Cannot execute a rejected transfer")

        source_approver = tr.sourceApprovedByUserId
        target_approver = tr.targetApprovedByUserId
        assert source_approver is not None and target_approver is not None
        graph_id = await _transfer_graph_id(tx, tr.resourceType, tr.resourceId)
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended('agent-graph:' || $1, 0))",
            graph_id,
        )
        await _lock_live_users(tx, user_id, source_approver, target_approver)
        await _lock_live_organizations(
            tx,
            tr.sourceOrganizationId,
            tr.targetOrganizationId,
        )
        await _assert_live_transfer_admin(tx, user_id, org_id)
        await _assert_live_transfer_admin(tx, source_approver, tr.sourceOrganizationId)
        await _assert_live_transfer_admin(tx, target_approver, tr.targetOrganizationId)
        await _validate_resource_ownership(
            tr.resourceType,
            tr.resourceId,
            tr.sourceOrganizationId,
            client=tx,
        )
        await _move_resource(
            resource_type=tr.resourceType,
            resource_id=tr.resourceId,
            source_org_id=tr.sourceOrganizationId,
            target_org_id=tr.targetOrganizationId,
            target_owner_user_id=target_approver,
            client=tx,
        )

        updated = await tx.transferrequest.update(
            where={"id": transfer_id},
            data={"status": "COMPLETED", "completedAt": datetime.now(timezone.utc)},
        )
        assert updated is not None
        await _create_audit_logs(updated, user_id, client=tx)
        return TransferResponse.from_db(updated)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _locked_transfer(
    transfer_id: str,
) -> AsyncIterator[tuple[Prisma, TransferRequest]]:
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        await execute_raw_with_schema(
            'UPDATE {schema_prefix}"TransferRequest" '
            'SET "updatedAt" = "updatedAt" WHERE "id" = $1',
            transfer_id,
            client=tx,
        )
        transfer = await tx.transferrequest.find_unique(where={"id": transfer_id})
        if transfer is None:
            raise NotFoundError(f"Transfer request {transfer_id} not found")
        yield tx, transfer


async def _lock_live_organizations(
    tx: Prisma,
    *organization_ids: str,
) -> None:
    for org_id in sorted(set(organization_ids)):
        locked = await execute_raw_with_schema(
            'UPDATE {schema_prefix}"Organization" '
            'SET "updatedAt" = "updatedAt" '
            'WHERE "id" = $1 AND "deletedAt" IS NULL',
            org_id,
            client=tx,
        )
        if locked != 1:
            raise NotFoundError(f"Organization '{org_id}' not found")


async def _assert_live_transfer_admin(
    tx: Prisma,
    user_id: str,
    organization_id: str,
) -> None:
    member = await tx.orgmember.find_first(
        where={
            "userId": user_id,
            "orgId": organization_id,
            "status": "ACTIVE",
            "OR": [{"isOwner": True}, {"isAdmin": True}],
            "Org": {"is": {"deletedAt": None}},
        }
    )
    if member is None:
        raise ValueError("Transfer administrator is no longer active")


async def _lock_live_users(tx: Prisma, *user_ids: str) -> None:
    for user_id in sorted(set(user_ids)):
        locked = await execute_raw_with_schema(
            'UPDATE {schema_prefix}"User" '
            'SET "updatedAt" = "updatedAt" WHERE "id" = $1',
            user_id,
            client=tx,
        )
        if locked != 1:
            raise ValueError("Transfer administrator is no longer active")


async def _lock_transfer_resource(
    tx: Prisma,
    resource_type: str,
    resource_id: str,
) -> None:
    await tx.query_raw(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        f"transfer:{resource_type}:{resource_id}",
    )


async def _transfer_graph_id(tx: Prisma, resource_type: str, resource_id: str) -> str:
    if resource_type == "AgentGraph":
        return resource_id
    if resource_type == "StoreListing":
        listing = await tx.storelisting.find_unique(where={"id": resource_id})
        if listing is None or listing.isDeleted:
            raise NotFoundError(f"StoreListing '{resource_id}' not found")
        return listing.agentGraphId
    raise ValueError(f"Unsupported transfer resource type '{resource_type}'")


async def _validate_resource_ownership(
    resource_type: str,
    resource_id: str,
    org_id: str,
    client: Prisma | None = None,
) -> None:
    """Verify the resource exists and belongs to the given org."""
    db = client or prisma
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
        graph = await db.agentgraph.find_first(
            where={"id": listing.agentGraphId, "isActive": True}
        )
        if graph is None or graph.organizationId != org_id:
            raise ValueError(
                "StoreListing's agent does not belong to the source organization"
            )


async def _move_resource(
    resource_type: str,
    resource_id: str,
    source_org_id: str,
    target_org_id: str,
    target_owner_user_id: str,
    client: Prisma | None = None,
) -> None:
    """Move the resource to the target organization."""
    db = client or prisma
    if resource_type == "AgentGraph":
        await _move_agent_graph(
            resource_id,
            source_org_id,
            target_org_id,
            target_owner_user_id,
            db,
        )

    elif resource_type == "StoreListing":
        listing = await db.storelisting.find_unique(where={"id": resource_id})
        if listing is None:
            raise NotFoundError(f"StoreListing '{resource_id}' not found")
        await _move_agent_graph(
            listing.agentGraphId,
            source_org_id,
            target_org_id,
            target_owner_user_id,
            db,
        )


async def _move_agent_graph(
    graph_id: str,
    source_org_id: str,
    target_org_id: str,
    target_owner_user_id: str,
    db: Prisma,
) -> None:
    await db.execute_raw(
        "SELECT pg_advisory_xact_lock(hashtextextended('agent-graph:' || $1, 0))",
        graph_id,
    )
    graph_versions = await db.agentgraph.find_many(where={"id": graph_id})
    if not graph_versions or any(
        graph.organizationId != source_org_id for graph in graph_versions
    ):
        raise ValueError(
            "Every AgentGraph version must belong to the source organization"
        )
    active_graph = next((graph for graph in graph_versions if graph.isActive), None)
    if active_graph is None:
        raise NotFoundError(f"AgentGraph '{graph_id}' not found")
    if await db.agentnode.count(
        where={
            "agentGraphId": graph_id,
            "agentBlockId": AgentExecutorBlock().id,
        }
    ):
        raise ValueError("Composed agents cannot be transferred between organizations")
    if await _count_incoming_graph_references(db, graph_id):
        raise ValueError(
            "Agents referenced by another composed agent cannot be transferred "
            "between organizations"
        )

    workflow_references = await db.expertworkflow.count(
        where={
            "Expert": {"is": {"isArchived": False}},
            "LibraryAgent": {
                "is": {
                    "agentGraphId": graph_id,
                    "organizationId": source_org_id,
                }
            },
        }
    )
    if workflow_references:
        raise ValueError(
            "Cannot transfer an agent while it is installed on an active expert"
        )
    presets = await db.agentpreset.count(
        where={
            "agentGraphId": graph_id,
            "organizationId": source_org_id,
            "isDeleted": False,
        }
    )
    node_webhooks = await db.agentnode.count(
        where={"agentGraphId": graph_id, "webhookId": {"not": None}}
    )
    schedules = await get_scheduler_client().get_execution_schedules(
        graph_id=graph_id,
        kind="graph",
        include_paused=True,
    )
    if presets or node_webhooks or schedules:
        raise ValueError(
            "Remove this agent's schedules, presets, and webhooks before transfer"
        )

    listing = await db.storelisting.find_first(where={"agentGraphId": graph_id})
    if listing is not None:
        profile = await db.profile.find_unique(where={"userId": target_owner_user_id})
        if profile is None:
            raise ValueError(
                "The target approver needs a marketplace profile before this "
                "listed agent can be transferred"
            )
        versions = await db.storelistingversion.find_many(
            where={"storeListingId": listing.id}
        )
        if any(
            version.organizationId not in (None, source_org_id) for version in versions
        ):
            raise ValueError(
                "Every StoreListingVersion must belong to the source organization"
            )

    await db.agentgraphgrant.delete_many(
        where={
            "agentGraphId": graph_id,
            "organizationId": source_org_id,
        }
    )
    await db.libraryagent.update_many(
        where={
            "agentGraphId": graph_id,
            "organizationId": source_org_id,
            "isDeleted": False,
        },
        data={
            "isDeleted": True,
            "isArchived": True,
            "folderId": None,
        },
    )
    moved = await db.agentgraph.update_many(
        where={"id": graph_id, "organizationId": source_org_id},
        data={
            "userId": target_owner_user_id,
            "organizationId": target_org_id,
            "teamId": None,
            "visibility": "ORG",
        },
    )
    if moved != len(graph_versions):
        raise ValueError("AgentGraph no longer belongs to the source organization")

    if listing is not None:
        moved_listing = await db.storelisting.update_many(
            where={"id": listing.id, "owningOrgId": source_org_id},
            data={
                "owningUserId": target_owner_user_id,
                "owningOrgId": target_org_id,
            },
        )
        if moved_listing != 1:
            raise ValueError(
                "StoreListing no longer belongs to the source organization"
            )
        await db.storelistingversion.update_many(
            where={"storeListingId": listing.id},
            data={"organizationId": target_org_id, "teamId": None},
        )

    target_library_agent = await db.libraryagent.find_first(
        where={
            "userId": target_owner_user_id,
            "agentGraphId": graph_id,
            "agentGraphVersion": active_graph.version,
            "organizationId": target_org_id,
            "teamId": None,
        }
    )
    library_data = {
        "isCreatedByUser": True,
        "useGraphIsActiveVersion": True,
        "isDeleted": False,
        "isArchived": False,
        "isHidden": False,
        "folderId": None,
        "visibility": "ORG",
    }
    if target_library_agent is None:
        await db.libraryagent.create(
            data={
                "userId": target_owner_user_id,
                "agentGraphId": graph_id,
                "agentGraphVersion": active_graph.version,
                "organizationId": target_org_id,
                "teamId": None,
                **library_data,
            }
        )
    else:
        await db.libraryagent.update(
            where={"id": target_library_agent.id},
            data=library_data,
        )


async def _create_audit_logs(
    transfer: TransferRequest,
    actor_user_id: str,
    client: Prisma | None = None,
) -> None:
    """Create audit log entries for both source and target organizations."""
    db = client or prisma
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

    await db.auditlog.create(
        data={
            **common,
            "organizationId": transfer.sourceOrganizationId,
            "beforeJson": {"organizationId": transfer.sourceOrganizationId},
        }
    )

    await db.auditlog.create(
        data={
            **common,
            "organizationId": transfer.targetOrganizationId,
            "beforeJson": {"organizationId": transfer.sourceOrganizationId},
        }
    )
