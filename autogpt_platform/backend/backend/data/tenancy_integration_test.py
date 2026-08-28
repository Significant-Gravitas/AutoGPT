import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import prisma.errors as prisma_errors
import pytest
from prisma import Json

from backend.api.features.library import db as library_db
from backend.api.features.orgs.db import create_org
from backend.api.features.transfers.db import _count_incoming_graph_references
from backend.blocks.agent import AgentExecutorBlock
from backend.data.db import prisma
from backend.data.graph import Graph, create_graph
from backend.data.tenancy import (
    LIVE_TRANSACTION_LEASE_CLASS_LIMIT,
    acquire_live_resource_lease,
    agent_graph_attachment_barriers,
    connect_live_transaction_lease_database,
    disconnect_live_transaction_lease_database,
    is_live_transaction_lease_database_connected,
    live_agent_graph_access_barrier,
    live_resource_access_barrier,
    release_live_resource_lease,
)
from backend.usecases.sample import create_test_graph
from backend.util.exceptions import GraphNotAccessibleError


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="session")
async def test_cross_team_grant_lookup_and_fork_stay_in_the_same_org(server):
    owner_id = f"grant-fork-owner-{uuid4()}"
    member_id = f"grant-fork-member-{uuid4()}"
    await prisma.user.create(data={"id": owner_id, "email": f"{owner_id}@example.com"})
    await prisma.user.create(
        data={"id": member_id, "email": f"{member_id}@example.com"}
    )
    org = await create_org("Grant fork integration", f"grant-fork-{uuid4()}", owner_id)
    source_team = await prisma.team.find_first(
        where={"orgId": org.id, "isDefault": True}
    )
    assert source_team is not None
    target_team = await prisma.team.create(
        data={
            "name": f"Grant target {uuid4()}",
            "orgId": org.id,
            "createdByUserId": owner_id,
        }
    )
    await prisma.orgmember.create(
        data={"orgId": org.id, "userId": member_id, "status": "ACTIVE"}
    )
    await prisma.teammember.create(
        data={"teamId": target_team.id, "userId": member_id, "status": "ACTIVE"}
    )

    source = create_test_graph()
    source.id = str(uuid4())
    source.name = "Cross-team shared agent"
    source_graph = await create_graph(
        source,
        owner_id,
        organization_id=org.id,
        team_id=source_team.id,
    )

    def close_background_task(coroutine, **_kwargs):
        coroutine.close()

    with (
        patch.object(
            library_db, "_fetch_schedule_info", new=AsyncMock(return_value={})
        ),
        patch.object(
            library_db, "spawn_background_task", side_effect=close_background_task
        ),
        patch.object(library_db, "schedule_library_agent_embedding"),
    ):
        source_library = (
            await library_db.create_library_agent(
                source_graph,
                owner_id,
                organization_id=org.id,
                team_id=source_team.id,
            )
        )[0]
        await prisma.agentgraphgrant.create(
            data={
                "agentGraphId": source_graph.id,
                "agentGraphVersion": source_graph.version,
                "principalType": "TEAM",
                "principalId": target_team.id,
                "capability": "EXECUTE",
                "credentialMode": "CONSUMER",
                "organizationId": org.id,
                "createdByUserId": owner_id,
            }
        )

        resolved = await library_db.get_library_agent_by_graph_id(
            member_id,
            source_graph.id,
            source_graph.version,
            organization_id=org.id,
            team_id_restriction=target_team.id,
            include_granted=True,
        )
        assert resolved is not None
        assert resolved.id == source_library.id

        forked = await library_db.fork_library_agent(
            source_library.id,
            member_id,
            organization_id=org.id,
            team_id=target_team.id,
        )

    forked_graph = await prisma.agentgraph.find_first(
        where={"id": forked.graph_id, "version": forked.graph_version}
    )
    assert forked_graph is not None
    assert forked_graph.userId == member_id
    assert forked_graph.organizationId == org.id
    assert forked_graph.teamId == target_team.id
    assert forked.organization_id == org.id
    assert forked.team_id == target_team.id


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="session")
async def test_dedicated_resource_lease_pool_preserves_queries_and_revocation_lock(
    server,
):
    owner_id = f"lease-pool-owner-{uuid4()}"
    member_id = f"lease-pool-member-{uuid4()}"
    slug = f"lease-pool-{uuid4()}"
    await prisma.user.create(data={"id": owner_id, "email": f"{owner_id}@example.com"})
    await prisma.user.create(
        data={"id": member_id, "email": f"{member_id}@example.com"}
    )
    org = await create_org("Lease pool", slug, owner_id)
    team = await prisma.team.find_first(where={"orgId": org.id, "isDefault": True})
    assert team is not None
    await prisma.orgmember.create(
        data={"orgId": org.id, "userId": member_id, "status": "ACTIVE"}
    )
    await prisma.teammember.create(
        data={"teamId": team.id, "userId": member_id, "status": "ACTIVE"}
    )

    lease_client_was_connected = is_live_transaction_lease_database_connected()
    await connect_live_transaction_lease_database()
    try:
        lease_ids = await asyncio.gather(
            *(
                acquire_live_resource_lease(member_id, org.id, team.id, "execute")
                for _ in range(LIVE_TRANSACTION_LEASE_CLASS_LIMIT)
            )
        )
        assert all(lease_id is not None for lease_id in lease_ids)
        try:
            assert await prisma.query_raw("SELECT 1 AS value") == [{"value": 1}]
            revoke_started = asyncio.Event()

            async def revoke_membership():
                revoke_started.set()
                return await prisma.teammember.update(
                    where={
                        "teamId_userId": {
                            "teamId": team.id,
                            "userId": member_id,
                        }
                    },
                    data={"status": "SUSPENDED"},
                )

            revoke_task = asyncio.create_task(revoke_membership())
            await asyncio.wait_for(revoke_started.wait(), timeout=1)
            await asyncio.sleep(0.1)
            assert not revoke_task.done()
        finally:
            released = await asyncio.gather(
                *(
                    release_live_resource_lease(lease_id)
                    for lease_id in lease_ids
                    if lease_id is not None
                )
            )
            assert released and all(released)

        await asyncio.wait_for(revoke_task, timeout=5)
        assert (
            await acquire_live_resource_lease(member_id, org.id, team.id, "execute")
            is None
        )
    finally:
        if not lease_client_was_connected:
            await disconnect_live_transaction_lease_database()


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="session")
async def test_scope_changes_wait_for_inflight_resource_writes(server):
    owner_id = f"tenancy-owner-{uuid4()}"
    member_id = f"tenancy-member-{uuid4()}"
    slug = f"tenancy-race-{uuid4()}"
    await prisma.user.create(data={"id": owner_id, "email": f"{owner_id}@example.com"})
    await prisma.user.create(
        data={"id": member_id, "email": f"{member_id}@example.com"}
    )
    org = await create_org("Tenancy race", slug, owner_id)
    team = await prisma.team.find_first(where={"orgId": org.id, "isDefault": True})
    assert team is not None
    await prisma.orgmember.create(
        data={"orgId": org.id, "userId": member_id, "status": "ACTIVE"}
    )
    await prisma.teammember.create(
        data={"teamId": team.id, "userId": member_id, "status": "ACTIVE"}
    )

    barrier_entered = asyncio.Event()
    release_write = asyncio.Event()

    async def write_inside_barrier():
        async with live_resource_access_barrier(
            member_id, org.id, team.id, "create"
        ) as allowed:
            assert allowed
            barrier_entered.set()
            await release_write.wait()
            return await prisma.libraryfolder.create(
                data={
                    "userId": member_id,
                    "name": f"race-{uuid4()}",
                    "organizationId": org.id,
                    "teamId": team.id,
                    "visibility": "TEAM",
                }
            )

    write_task = asyncio.create_task(write_inside_barrier())
    await asyncio.wait_for(barrier_entered.wait(), timeout=5)
    revoke_task = asyncio.create_task(
        prisma.teammember.update(
            where={"teamId_userId": {"teamId": team.id, "userId": member_id}},
            data={"status": "SUSPENDED"},
        )
    )
    await asyncio.sleep(0.1)
    assert not revoke_task.done()

    release_write.set()
    folder = await asyncio.wait_for(write_task, timeout=20)
    await asyncio.wait_for(revoke_task, timeout=20)
    assert folder.organizationId == org.id
    assert folder.teamId == team.id

    async with live_resource_access_barrier(
        member_id, org.id, team.id, "create"
    ) as allowed:
        assert not allowed
    with pytest.raises(prisma_errors.PrismaError, match="active workspace membership"):
        await prisma.libraryfolder.create(
            data={
                "userId": member_id,
                "name": f"rejected-{uuid4()}",
                "organizationId": org.id,
                "teamId": team.id,
                "visibility": "TEAM",
            }
        )

    await prisma.teammember.update(
        where={"teamId_userId": {"teamId": team.id, "userId": member_id}},
        data={"status": "ACTIVE"},
    )
    graph_id = str(uuid4())
    await prisma.agentgraph.create(
        data={
            "id": graph_id,
            "version": 1,
            "name": "Graph lock race",
            "userId": member_id,
            "organizationId": org.id,
            "teamId": team.id,
            "visibility": "TEAM",
        }
    )
    graph_barrier_entered = asyncio.Event()
    release_graph_write = asyncio.Event()

    async def write_graph_inside_barrier():
        async with live_agent_graph_access_barrier(
            member_id, org.id, team.id, "create", graph_id, 1
        ) as allowed:
            assert allowed
            graph_barrier_entered.set()
            await release_graph_write.wait()
            return await prisma.agentgraph.update(
                where={"graphVersionId": {"id": graph_id, "version": 1}},
                data={"description": "updated before transfer"},
            )

    async def take_transfer_lock():
        async with prisma.tx() as tx:
            await tx.execute_raw(
                "SELECT pg_advisory_xact_lock("
                "hashtextextended('agent-graph:' || $1, 0))",
                graph_id,
            )

    graph_write_task = asyncio.create_task(write_graph_inside_barrier())
    await asyncio.wait_for(graph_barrier_entered.wait(), timeout=5)
    transfer_task = asyncio.create_task(take_transfer_lock())
    await asyncio.sleep(0.1)
    assert not transfer_task.done()

    release_graph_write.set()
    graph = await asyncio.wait_for(graph_write_task, timeout=20)
    await asyncio.wait_for(transfer_task, timeout=20)
    assert graph.description == "updated before transfer"

    execution = await prisma.agentgraphexecution.create(
        data={
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "userId": member_id,
            "organizationId": org.id,
            "teamId": team.id,
            "visibility": "TEAM",
            "executionStatus": "INCOMPLETE",
        }
    )
    await prisma.teammember.update(
        where={"teamId_userId": {"teamId": team.id, "userId": member_id}},
        data={"status": "SUSPENDED"},
    )
    terminated = await prisma.agentgraphexecution.update(
        where={"id": execution.id}, data={"executionStatus": "TERMINATED"}
    )
    assert terminated.executionStatus == "TERMINATED"
    with pytest.raises(prisma_errors.PrismaError, match="active workspace membership"):
        await prisma.agentgraphexecution.create(
            data={
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
                "userId": member_id,
                "organizationId": org.id,
                "teamId": team.id,
                "visibility": "TEAM",
                "executionStatus": "INCOMPLETE",
            }
        )


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="session")
async def test_resource_validation_locks_allow_writes_and_recheck_revocation(server):
    owner_id = f"validation-owner-{uuid4()}"
    member_id = f"validation-member-{uuid4()}"
    await prisma.user.create(data={"id": owner_id, "email": f"{owner_id}@example.com"})
    await prisma.user.create(
        data={"id": member_id, "email": f"{member_id}@example.com"}
    )
    org = await create_org(
        "Validation lock race", f"validation-lock-{uuid4()}", owner_id
    )
    team = await prisma.team.find_first(where={"orgId": org.id, "isDefault": True})
    assert team is not None
    await prisma.orgmember.create(
        data={"orgId": org.id, "userId": member_id, "status": "ACTIVE"}
    )
    await prisma.teammember.create(
        data={"teamId": team.id, "userId": member_id, "status": "ACTIVE"}
    )

    first_created = asyncio.Event()
    release_first = asyncio.Event()

    async def hold_first_write():
        async with prisma.tx() as tx:
            folder = await tx.libraryfolder.create(
                data={
                    "userId": member_id,
                    "name": f"first-{uuid4()}",
                    "organizationId": org.id,
                    "teamId": team.id,
                    "visibility": "TEAM",
                }
            )
            first_created.set()
            await release_first.wait()
            return folder

    first_write = asyncio.create_task(hold_first_write())
    await asyncio.wait_for(first_created.wait(), timeout=5)
    second_folder = await asyncio.wait_for(
        prisma.libraryfolder.create(
            data={
                "userId": member_id,
                "name": f"second-{uuid4()}",
                "organizationId": org.id,
                "teamId": team.id,
                "visibility": "TEAM",
            }
        ),
        timeout=5,
    )
    release_first.set()
    first_folder = await asyncio.wait_for(first_write, timeout=20)
    assert first_folder.organizationId == second_folder.organizationId == org.id

    graph_id = str(uuid4())
    await prisma.agentgraph.create(
        data={
            "id": graph_id,
            "version": 1,
            "name": "Concurrent execution validation",
            "userId": member_id,
            "organizationId": org.id,
            "teamId": team.id,
            "visibility": "TEAM",
        }
    )
    execution_created = asyncio.Event()
    release_execution = asyncio.Event()

    async def hold_first_execution():
        async with prisma.tx() as tx:
            execution = await tx.agentgraphexecution.create(
                data={
                    "agentGraphId": graph_id,
                    "agentGraphVersion": 1,
                    "userId": member_id,
                    "organizationId": org.id,
                    "teamId": team.id,
                    "visibility": "TEAM",
                    "executionStatus": "INCOMPLETE",
                }
            )
            execution_created.set()
            await release_execution.wait()
            return execution

    first_execution = asyncio.create_task(hold_first_execution())
    await asyncio.wait_for(execution_created.wait(), timeout=5)
    second_execution = await asyncio.wait_for(
        prisma.agentgraphexecution.create(
            data={
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
                "userId": member_id,
                "organizationId": org.id,
                "teamId": team.id,
                "visibility": "TEAM",
                "executionStatus": "INCOMPLETE",
            }
        ),
        timeout=5,
    )
    release_execution.set()
    held_execution = await asyncio.wait_for(first_execution, timeout=20)
    assert held_execution.agentGraphId == second_execution.agentGraphId == graph_id

    revoke_updated = asyncio.Event()
    release_revoke = asyncio.Event()

    async def hold_revocation():
        async with prisma.tx() as tx:
            await tx.teammember.update(
                where={"teamId_userId": {"teamId": team.id, "userId": member_id}},
                data={"status": "SUSPENDED"},
            )
            revoke_updated.set()
            await release_revoke.wait()

    revocation = asyncio.create_task(hold_revocation())
    await asyncio.wait_for(revoke_updated.wait(), timeout=5)
    rejected_write = asyncio.create_task(
        prisma.libraryfolder.create(
            data={
                "userId": member_id,
                "name": f"revoked-{uuid4()}",
                "organizationId": org.id,
                "teamId": team.id,
                "visibility": "TEAM",
            }
        )
    )
    await asyncio.sleep(0.1)
    assert not rejected_write.done()
    release_revoke.set()
    await asyncio.wait_for(revocation, timeout=20)
    with pytest.raises(prisma_errors.PrismaError, match="active workspace membership"):
        await asyncio.wait_for(rejected_write, timeout=20)


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="session")
async def test_subgraph_id_cannot_cross_owner_or_workspace(server):
    victim_id = f"graph-victim-{uuid4()}"
    attacker_id = f"graph-attacker-{uuid4()}"
    await prisma.user.create(
        data={"id": victim_id, "email": f"{victim_id}@example.com"}
    )
    await prisma.user.create(
        data={"id": attacker_id, "email": f"{attacker_id}@example.com"}
    )
    victim_org = await create_org(
        "Victim graph org", f"victim-graph-{uuid4()}", victim_id
    )
    attacker_org = await create_org(
        "Attacker graph org", f"attacker-graph-{uuid4()}", attacker_id
    )
    victim_team = await prisma.team.find_first(
        where={"orgId": victim_org.id, "isDefault": True}
    )
    attacker_team = await prisma.team.find_first(
        where={"orgId": attacker_org.id, "isDefault": True}
    )
    assert victim_team is not None and attacker_team is not None

    victim_graph_id = str(uuid4())
    await prisma.agentgraph.create(
        data={
            "id": victim_graph_id,
            "version": 1,
            "name": "Victim graph",
            "userId": victim_id,
            "organizationId": victim_org.id,
            "teamId": victim_team.id,
            "visibility": "TEAM",
        }
    )
    attacker_root_id = str(uuid4())
    payload = Graph(
        id=attacker_root_id,
        version=1,
        name="Attacker root",
        description="",
        nodes=[],
        links=[],
        sub_graphs=[
            Graph(
                id=victim_graph_id,
                version=1,
                name="Injected subgraph",
                description="",
                nodes=[],
                links=[],
            )
        ],
    )

    with pytest.raises(GraphNotAccessibleError):
        await create_graph(
            payload,
            attacker_id,
            organization_id=attacker_org.id,
            team_id=attacker_team.id,
        )

    assert await prisma.agentgraph.count(where={"id": victim_graph_id}) == 1
    assert await prisma.agentgraph.count(where={"id": attacker_root_id}) == 0


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="session")
async def test_parent_creation_serializes_with_child_transfer_reference_check(server):
    owner_id = f"composition-owner-{uuid4()}"
    await prisma.user.create(data={"id": owner_id, "email": f"{owner_id}@example.com"})
    org = await create_org("Composition race", f"composition-race-{uuid4()}", owner_id)
    team = await prisma.team.find_first(where={"orgId": org.id, "isDefault": True})
    assert team is not None

    child_id = str(uuid4())
    parent_id = str(uuid4())
    await prisma.agentgraph.create(
        data={
            "id": child_id,
            "version": 1,
            "name": "Child",
            "userId": owner_id,
            "organizationId": org.id,
            "teamId": team.id,
            "visibility": "TEAM",
        }
    )

    writer_entered = asyncio.Event()
    release_writer = asyncio.Event()

    async def create_parent_reference() -> None:
        async with agent_graph_attachment_barriers([parent_id, child_id]):
            writer_entered.set()
            await release_writer.wait()
            await prisma.agentgraph.create(
                data={
                    "id": parent_id,
                    "version": 1,
                    "name": "Parent",
                    "userId": owner_id,
                    "organizationId": org.id,
                    "teamId": team.id,
                    "visibility": "TEAM",
                }
            )
            await prisma.agentnode.create(
                data={
                    "agentBlockId": AgentExecutorBlock().id,
                    "agentGraphId": parent_id,
                    "agentGraphVersion": 1,
                    "constantInput": Json({"graph_id": child_id}),
                }
            )

    async def transfer_reference_check() -> int:
        async with prisma.tx() as tx:
            await tx.execute_raw(
                "SELECT pg_advisory_xact_lock("
                "hashtextextended('agent-graph:' || $1, 0))",
                child_id,
            )
            return await _count_incoming_graph_references(tx, child_id)

    writer = asyncio.create_task(create_parent_reference())
    await asyncio.wait_for(writer_entered.wait(), timeout=5)
    transfer = asyncio.create_task(transfer_reference_check())
    await asyncio.sleep(0.1)
    assert not transfer.done()

    release_writer.set()
    await asyncio.wait_for(writer, timeout=20)
    assert await asyncio.wait_for(transfer, timeout=20) == 1
