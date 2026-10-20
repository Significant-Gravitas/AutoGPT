"""DB-backed: a shared run's outputs equal the owner's (#15652).

Static Agent Output blocks (value typed in the builder, no inbound link) are
starting nodes: ``create_graph_execution`` gives them ``Input`` rows and no
``executionData``. The share query didn't load ``Input``, so they vanished
from the public page.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

import prisma.models
import pytest
from prisma import Json
from prisma.enums import AgentExecutionStatus

from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.data.execution import (
    create_graph_execution,
    get_graph_execution,
    get_graph_execution_by_share_token,
    update_graph_execution_share_status,
)
from backend.data.includes import MAX_NODE_EXECUTIONS_FETCH

logger = logging.getLogger(__name__)
OUTPUT_BLOCK_ID = AgentOutputBlock().id
MODERATED = "Failed due to content moderation"


async def _setup() -> tuple[str, str, dict[str, str]]:
    user_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())
    await prisma.models.User.prisma().create(
        data={
            "id": user_id,
            "email": f"share-outputs-{user_id}@example.com",
            "name": "Share Outputs Test",
        }
    )
    await prisma.models.AgentGraph.prisma().create(
        data={
            "id": graph_id,
            "version": 1,
            "name": "share-outputs-test",
            "description": "share-outputs-test",
            "userId": user_id,
            "isActive": True,
        }
    )
    nodes = {}
    for key in ("static_note", "static_number", "linked_note", "moderated"):
        node = await prisma.models.AgentNode.prisma().create(
            data={
                "agentBlockId": OUTPUT_BLOCK_ID,
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
            }
        )
        nodes[key] = node.id
    return user_id, graph_id, nodes


async def _cleanup(user_id: str, graph_id: str) -> None:
    try:
        await prisma.models.AgentGraphExecution.prisma().delete_many(
            where={"userId": user_id}
        )
        await prisma.models.AgentGraph.prisma().delete_many(where={"id": graph_id})
        await prisma.models.User.prisma().delete_many(where={"id": user_id})
    except Exception as exc:
        logger.warning("cleanup for %s failed: %s", user_id, exc)


@pytest.mark.asyncio(loop_scope="session")
async def test_shared_outputs_include_static_outputs_and_match_owner(server):
    user_id, graph_id, nodes = await _setup()
    try:
        # Static outputs: starting nodes, Input rows only.
        execution = await create_graph_execution(
            graph_id=graph_id,
            graph_version=1,
            starting_nodes_input=[
                (nodes["static_note"], {"name": "static_note", "value": "typed"}),
                (nodes["static_number"], {"name": "static_number", "value": 42}),
            ],
            inputs={},
            user_id=user_id,
        )
        # Linked output: executionData written when it was enqueued.
        await prisma.models.AgentNodeExecution.prisma().create(
            data={
                "agentNodeId": nodes["linked_note"],
                "agentGraphExecutionId": execution.id,
                "executionStatus": AgentExecutionStatus.COMPLETED,
                "queuedTime": datetime.now(timezone.utc),
                "executionData": Json({"name": "linked_note", "value": "linked"}),
            }
        )
        # Moderated output: original executionData stays on the row.
        await prisma.models.AgentNodeExecution.prisma().create(
            data={
                "agentNodeId": nodes["moderated"],
                "agentGraphExecutionId": execution.id,
                "executionStatus": AgentExecutionStatus.FAILED,
                "queuedTime": datetime.now(timezone.utc),
                "executionData": Json({"name": "secret", "value": "original"}),
                "stats": Json(
                    {
                        "error": MODERATED,
                        "cleared_inputs": {
                            "name": [MODERATED],
                            "value": [MODERATED],
                        },
                        "cleared_outputs": {},
                    }
                ),
            }
        )
        token = str(uuid.uuid4())
        await update_graph_execution_share_status(
            execution_id=execution.id,
            user_id=user_id,
            is_shared=True,
            share_token=token,
            shared_at=datetime.now(timezone.utc),
        )

        owner = await get_graph_execution(
            user_id, execution.id, include_node_executions=True
        )
        shared = await get_graph_execution_by_share_token(token)

        assert owner is not None and shared is not None
        assert dict(shared.outputs) == dict(owner.outputs)
        assert shared.outputs["static_note"] == ["typed"]
        assert shared.outputs["static_number"] == [42]
        assert shared.outputs["linked_note"] == ["linked"]
        assert "secret" not in shared.outputs
        assert "original" not in shared.model_dump_json()
    finally:
        await _cleanup(user_id, graph_id)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "newer_count", [MAX_NODE_EXECUTIONS_FETCH - 1, MAX_NODE_EXECUTIONS_FETCH]
)
async def test_shared_outputs_match_owner_at_node_execution_limit(server, newer_count):
    user_id, graph_id, nodes = await _setup()
    try:
        execution = await create_graph_execution(
            graph_id=graph_id,
            graph_version=1,
            starting_nodes_input=[
                (nodes["static_note"], {"name": "static_note", "value": "typed"})
            ],
            inputs={},
            user_id=user_id,
        )
        input_node = await prisma.models.AgentNode.prisma().create(
            data={
                "agentBlockId": AgentInputBlock().id,
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
            }
        )
        now = datetime.now(timezone.utc)
        await prisma.models.AgentNodeExecution.prisma().create_many(
            data=[
                {
                    "agentNodeId": input_node.id,
                    "agentGraphExecutionId": execution.id,
                    "executionStatus": AgentExecutionStatus.COMPLETED,
                    "queuedTime": now + timedelta(seconds=i + 1),
                    "executionData": Json({"name": "input", "value": i}),
                }
                for i in range(newer_count)
            ]
        )
        token = str(uuid.uuid4())
        await update_graph_execution_share_status(
            execution_id=execution.id,
            user_id=user_id,
            is_shared=True,
            share_token=token,
            shared_at=now,
        )

        owner = await get_graph_execution(
            user_id, execution.id, include_node_executions=True
        )
        shared = await get_graph_execution_by_share_token(token)

        assert owner is not None and shared is not None
        assert dict(owner.outputs) == (
            {"static_note": ["typed"]}
            if newer_count < MAX_NODE_EXECUTIONS_FETCH
            else {}
        )
        assert dict(shared.outputs) == dict(owner.outputs)
    finally:
        await _cleanup(user_id, graph_id)
