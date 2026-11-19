"""DB-backed tests for execute_transfer (#15267, #15271).

These run against the real database on purpose: the original audit-log bug
was invisible to tests that mock ``prisma.auditlog.create``.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from prisma.models import AgentGraph, AuditLog, StoreListing, TransferRequest

from backend.api.features.transfers import db as transfers_db


async def _approved_transfer(world: dict, target_org: str) -> str:
    tr = await transfers_db.create_transfer(
        source_org_id=world["org_a"],
        target_org_id=target_org,
        resource_type=world["resource_type"],
        resource_id=world["resource_id"],
        user_id=world["source_user"],
    )
    await transfers_db.approve_transfer(tr.id, world["source_user"], world["org_a"])
    await transfers_db.approve_transfer(tr.id, world["target_user"], target_org)
    return tr.id


async def _resource_orgs(world: dict) -> set[str | None]:
    if world["resource_type"] == "StoreListing":
        listing = await StoreListing.prisma().find_unique_or_raise(
            where={"id": world["resource_id"]}
        )
        return {listing.owningOrgId}
    rows = await AgentGraph.prisma().find_many(where={"id": world["graph_id"]})
    return {row.organizationId for row in rows}


async def _status(transfer_id: str) -> str:
    tr = await TransferRequest.prisma().find_unique_or_raise(where={"id": transfer_id})
    return tr.status


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_moves_resource_and_writes_audit_rows(transfer_world):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])

    result = await transfers_db.execute_transfer(
        t1, world["source_user"], world["org_a"]
    )

    assert result.status == "COMPLETED"
    assert await _resource_orgs(world) == {world["org_b"]}
    if world["resource_type"] == "AgentGraph":
        versions = await AgentGraph.prisma().find_many(where={"id": world["graph_id"]})
        assert len(versions) == 2
        assert all(
            v.teamId is None and v.userId == world["source_user"] for v in versions
        )
    else:
        listing = await StoreListing.prisma().find_unique_or_raise(
            where={"id": world["resource_id"]}
        )
        assert listing.owningUserId == world["source_user"]
        versions = await AgentGraph.prisma().find_many(where={"id": world["graph_id"]})
        assert all(v.organizationId == world["org_a"] for v in versions)
    audit = await AuditLog.prisma().find_many(where={"entityId": t1})
    assert len(audit) == 2
    assert {row.organizationId for row in audit} == {world["org_a"], world["org_b"]}
    for row in audit:
        assert row.action == "TRANSFER_EXECUTED"
        assert row.afterJson == {
            "resourceType": world["resource_type"],
            "resourceId": world["resource_id"],
            "sourceOrganizationId": world["org_a"],
            "targetOrganizationId": world["org_b"],
        }
        assert row.beforeJson == {"organizationId": world["org_a"]}


@pytest.mark.asyncio(loop_scope="session")
async def test_audit_failure_rolls_back_the_move(transfer_world):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])
    t2 = await _approved_transfer(world, world["org_c"])
    audit_actions = type(transfers_db.prisma.auditlog)
    create_audit = audit_actions.create
    inserted = []

    async def fail_second_audit(self, **kwargs):
        if inserted:
            raise RuntimeError("second audit down")
        row = await create_audit(self, **kwargs)
        inserted.append(row.id)
        return row

    with (
        patch.object(audit_actions, "create", fail_second_audit),
        pytest.raises(RuntimeError, match="second audit down"),
    ):
        await transfers_db.execute_transfer(t1, world["source_user"], world["org_a"])

    assert len(inserted) == 1
    assert await _resource_orgs(world) == {world["org_a"]}
    assert await _status(t1) == "SOURCE_APPROVED"
    assert await _status(t2) == "SOURCE_APPROVED"
    assert await AuditLog.prisma().count(where={"entityId": t1}) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_stale_transfer_cannot_move_resource_away_from_new_owner(transfer_world):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])
    t2 = await _approved_transfer(world, world["org_c"])

    await transfers_db.execute_transfer(t1, world["source_user"], world["org_a"])

    # Executing T1 closes the competing T2...
    assert await _status(t2) == "REJECTED"
    with pytest.raises(ValueError):
        await transfers_db.execute_transfer(t2, world["source_user"], world["org_a"])
    assert await _resource_orgs(world) == {world["org_b"]}

    # ...and even if T2 were still open, execute re-checks ownership.
    await TransferRequest.prisma().update(
        where={"id": t2}, data={"status": "SOURCE_APPROVED"}
    )
    with pytest.raises(ValueError, match="no longer belongs"):
        await transfers_db.execute_transfer(t2, world["source_user"], world["org_a"])
    assert await _resource_orgs(world) == {world["org_b"]}
    assert await _status(t2) == "REJECTED"


@pytest.mark.asyncio(loop_scope="session")
async def test_deleted_resource_rejects_the_transfer(transfer_world):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])
    if world["resource_type"] == "AgentGraph":
        await AgentGraph.prisma().update_many(
            where={"id": world["graph_id"]}, data={"isActive": False}
        )
    else:
        await StoreListing.prisma().update(
            where={"id": world["resource_id"]}, data={"isDeleted": True}
        )

    with pytest.raises(ValueError):
        await transfers_db.execute_transfer(t1, world["source_user"], world["org_a"])

    assert await _resource_orgs(world) == {world["org_a"]}
    assert await _status(t1) == "REJECTED"


@pytest.mark.asyncio(loop_scope="session")
async def test_cancelled_transfer_cannot_be_executed(transfer_world):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])
    await TransferRequest.prisma().update(
        where={"id": t1}, data={"status": "CANCELLED"}
    )

    with pytest.raises(ValueError):
        await transfers_db.execute_transfer(t1, world["source_user"], world["org_a"])

    assert await _resource_orgs(world) == {world["org_a"]}
    assert await _status(t1) == "CANCELLED"


@pytest.mark.asyncio(loop_scope="session")
async def test_reject_racing_execute_cannot_undo_a_completed_transfer(
    transfer_world,
):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])
    # The reject read the transfer while it was still open...
    stale = await TransferRequest.prisma().find_unique_or_raise(where={"id": t1})

    # ...then the execute committed before the reject wrote.
    await transfers_db.execute_transfer(t1, world["source_user"], world["org_a"])

    # The actions object uses __slots__, so patch its class for this call.
    with patch.object(
        type(transfers_db.prisma.transferrequest),
        "find_unique",
        AsyncMock(return_value=stale),
    ), pytest.raises(ValueError):
        await transfers_db.reject_transfer(t1, world["target_user"], world["org_b"])

    assert await _status(t1) == "COMPLETED"
    assert await _resource_orgs(world) == {world["org_b"]}


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_competing_transfers_move_resource_once(transfer_world):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])
    t2 = await _approved_transfer(world, world["org_c"])

    results = await asyncio.gather(
        transfers_db.execute_transfer(t1, world["source_user"], world["org_a"]),
        transfers_db.execute_transfer(t2, world["source_user"], world["org_a"]),
        return_exceptions=True,
    )

    completed = [r for r in results if not isinstance(r, BaseException)]
    assert len(completed) == 1, results
    # The loser gets a clean ValueError (400), not a DB/deadlock error.
    assert all(isinstance(r, ValueError) for r in results if r not in completed)
    winner_org = completed[0].target_organization_id
    assert await _resource_orgs(world) == {winner_org}
    statuses = sorted([await _status(t1), await _status(t2)])
    assert statuses == ["COMPLETED", "REJECTED"]


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_double_execute_moves_once(transfer_world):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])

    results = await asyncio.gather(
        *(
            transfers_db.execute_transfer(t1, world["source_user"], world["org_a"])
            for _ in range(2)
        ),
        return_exceptions=True,
    )

    assert sum(not isinstance(r, BaseException) for r in results) == 1, results
    assert all(
        isinstance(r, (ValueError, transfers_db.TransferResponse)) for r in results
    ), results
    assert await _status(t1) == "COMPLETED"
    assert await AuditLog.prisma().count(where={"entityId": t1}) == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_claim_rechecks_status_after_stale_read(transfer_world):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])
    stale = await TransferRequest.prisma().find_unique_or_raise(where={"id": t1})
    await transfers_db.reject_transfer(t1, world["target_user"], world["org_b"])

    with patch.object(
        type(transfers_db.prisma.transferrequest),
        "find_unique",
        AsyncMock(return_value=stale),
    ), pytest.raises(ValueError, match="no longer open"):
        await transfers_db.execute_transfer(t1, world["source_user"], world["org_a"])

    assert await _resource_orgs(world) == {world["org_a"]}
    assert await _status(t1) == "REJECTED"
    assert await AuditLog.prisma().count(where={"entityId": t1}) == 0
