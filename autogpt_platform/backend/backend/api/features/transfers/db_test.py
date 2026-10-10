"""DB-backed tests for execute_transfer (#15267, #15271).

These run against the real database on purpose: the original audit-log bug
was invisible to tests that mock ``prisma.auditlog.create``.
"""

import asyncio
from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.models import AgentGraph, AuditLog, Organization, TransferRequest, User

from backend.api.features.transfers import db as transfers_db


@pytest_asyncio.fixture(loop_scope="session")
async def transfer_world(server):
    """Three orgs (A, B, C), two users, and a graph owned by org A."""
    source_user = str(uuid4())
    target_user = str(uuid4())
    for user_id in (source_user, target_user):
        await User.prisma().create(
            data={"id": user_id, "email": f"transfer-{user_id}@example.com"}
        )
    orgs = []
    for name in ("a", "b", "c"):
        org = await Organization.prisma().create(
            data={"name": f"Transfer {name}", "slug": f"transfer-{name}-{uuid4()}"}
        )
        orgs.append(org.id)
    org_a, org_b, org_c = orgs
    graph_id = str(uuid4())
    for version in (1, 2):
        await AgentGraph.prisma().create(
            data={
                "id": graph_id,
                "version": version,
                "userId": source_user,
                "organizationId": org_a,
                "isActive": version == 2,
            }
        )

    yield {
        "source_user": source_user,
        "target_user": target_user,
        "org_a": org_a,
        "org_b": org_b,
        "org_c": org_c,
        "graph_id": graph_id,
    }

    await AuditLog.prisma().delete_many(where={"organizationId": {"in": orgs}})
    await TransferRequest.prisma().delete_many(where={"resourceId": graph_id})
    await AgentGraph.prisma().delete_many(where={"id": graph_id})
    await Organization.prisma().delete_many(where={"id": {"in": orgs}})
    await User.prisma().delete_many(where={"id": {"in": [source_user, target_user]}})


async def _approved_transfer(world: dict, target_org: str) -> str:
    tr = await transfers_db.create_transfer(
        source_org_id=world["org_a"],
        target_org_id=target_org,
        resource_type="AgentGraph",
        resource_id=world["graph_id"],
        user_id=world["source_user"],
    )
    await transfers_db.approve_transfer(tr.id, world["source_user"], world["org_a"])
    await transfers_db.approve_transfer(tr.id, world["target_user"], target_org)
    return tr.id


async def _graph_orgs(graph_id: str) -> set[str | None]:
    rows = await AgentGraph.prisma().find_many(where={"id": graph_id})
    return {row.organizationId for row in rows}


async def _status(transfer_id: str) -> str:
    tr = await TransferRequest.prisma().find_unique_or_raise(where={"id": transfer_id})
    return tr.status


@pytest.mark.asyncio(loop_scope="session")
async def test_execute_moves_graph_and_writes_audit_rows(transfer_world):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])

    result = await transfers_db.execute_transfer(
        t1, world["source_user"], world["org_a"]
    )

    assert result.status == "COMPLETED"
    assert await _graph_orgs(world["graph_id"]) == {world["org_b"]}
    audit = await AuditLog.prisma().find_many(where={"entityId": t1})
    assert {row.organizationId for row in audit} == {world["org_a"], world["org_b"]}
    for row in audit:
        assert row.action == "TRANSFER_EXECUTED"
        assert row.afterJson == {
            "resourceType": "AgentGraph",
            "resourceId": world["graph_id"],
            "sourceOrganizationId": world["org_a"],
            "targetOrganizationId": world["org_b"],
        }
        assert row.beforeJson == {"organizationId": world["org_a"]}


@pytest.mark.asyncio(loop_scope="session")
async def test_audit_failure_rolls_back_the_move(transfer_world):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])

    with patch.object(
        transfers_db, "_create_audit_logs", side_effect=RuntimeError("audit down")
    ):
        with pytest.raises(RuntimeError, match="audit down"):
            await transfers_db.execute_transfer(
                t1, world["source_user"], world["org_a"]
            )

    assert await _graph_orgs(world["graph_id"]) == {world["org_a"]}
    assert await _status(t1) != "COMPLETED"
    assert await AuditLog.prisma().count(where={"entityId": t1}) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_stale_transfer_cannot_move_graph_away_from_new_owner(transfer_world):
    world = transfer_world
    t1 = await _approved_transfer(world, world["org_b"])
    t2 = await _approved_transfer(world, world["org_c"])

    await transfers_db.execute_transfer(t1, world["source_user"], world["org_a"])

    # Executing T1 closes the competing T2...
    assert await _status(t2) == "REJECTED"
    with pytest.raises(ValueError):
        await transfers_db.execute_transfer(t2, world["source_user"], world["org_a"])
    assert await _graph_orgs(world["graph_id"]) == {world["org_b"]}

    # ...and even if T2 were still open, execute re-checks ownership.
    await TransferRequest.prisma().update(
        where={"id": t2}, data={"status": "SOURCE_APPROVED"}
    )
    with pytest.raises(ValueError, match="no longer belongs"):
        await transfers_db.execute_transfer(t2, world["source_user"], world["org_a"])
    assert await _graph_orgs(world["graph_id"]) == {world["org_b"]}
    assert await _status(t2) == "REJECTED"


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_competing_transfers_move_graph_once(transfer_world):
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
    assert await _graph_orgs(world["graph_id"]) == {winner_org}
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
