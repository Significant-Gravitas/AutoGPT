"""Integration tests for pending-review batching + expert attribution.

Covers get_pending_reviews_for_user's enrichment: expert attribution from
the graph execution (or, for CoPilot run_block reviews, from the chat
session), the requesting agent's display name, and the library agent id
used for run deep links.
"""

import logging
from uuid import uuid4

import pytest
from prisma.enums import ReviewStatus
from prisma.errors import UniqueViolationError
from prisma.models import (
    AgentBlock,
    AgentGraph,
    AgentGraphExecution,
    AgentNode,
    AgentNodeExecution,
    ChatSession,
    Expert,
    LibraryAgent,
    PendingHumanReview,
    User,
)

from backend.copilot.constants import COPILOT_NODE_PREFIX, COPILOT_SESSION_PREFIX
from backend.copilot.db import create_chat_session
from backend.data.human_review import get_pending_reviews_for_user
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _create_user(user_id: str) -> None:
    try:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"review-enrich-{user_id}@example.com",
                "name": "Review Enrichment Test",
            }
        )
    except UniqueViolationError:
        pass


async def _create_graph(graph_id: str, user_id: str, name: str) -> None:
    await AgentGraph.prisma().create(
        data={
            "id": graph_id,
            "version": 1,
            "name": name,
            "description": "test",
            "userId": user_id,
            "isActive": True,
        }
    )


async def _cleanup(user_id: str, graph_ids: list[str], session_ids: list[str]) -> None:
    try:
        await PendingHumanReview.prisma().delete_many(where={"userId": user_id})
        await AgentGraphExecution.prisma().delete_many(where={"userId": user_id})
        if session_ids:
            await ChatSession.prisma().delete_many(where={"id": {"in": session_ids}})
        await LibraryAgent.prisma().delete_many(where={"userId": user_id})
        for graph_id in graph_ids:
            await AgentGraph.prisma().delete_many(where={"id": graph_id})
        await Expert.prisma().delete_many(where={"ownerUserId": user_id})
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as exc:
        logger.warning("cleanup for %s failed: %s", user_id, exc)


async def test_pending_reviews_are_enriched(server: SpinTestServer):
    user_id = f"review-enrich-{uuid4()}"
    graph_id = str(uuid4())
    real_exec_id = str(uuid4())
    graph_ids = [graph_id]
    session_ids: list[str] = []

    await _create_user(user_id)
    try:
        await _create_graph(graph_id, user_id, "Lead Finder")

        ana = await Expert.prisma().create(
            data={
                "ownerUserId": user_id,
                "name": "Ana",
                "avatarUrl": "https://example.com/ana.png",
                "role": "Researcher",
                "identity": "You are Ana, a research expert.",
            }
        )

        lib_agent = await LibraryAgent.prisma().create(
            data={
                "userId": user_id,
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
                "name": "Lead Finder",
                "isCreatedByUser": True,
            }
        )

        await AgentGraphExecution.prisma().create(
            data={
                "id": real_exec_id,
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
                "userId": user_id,
                "expertId": ana.id,
            }
        )

        session = await create_chat_session(
            f"sess-{uuid4().hex[:8]}", user_id, expert_id=ana.id
        )
        session_ids.append(session.session_id)
        copilot_exec_id = f"{COPILOT_SESSION_PREFIX}{session.session_id}"

        await PendingHumanReview.prisma().create(
            data={
                "nodeExecId": f"node-exec-{uuid4()}",
                "userId": user_id,
                "graphExecId": real_exec_id,
                "graphId": graph_id,
                "graphVersion": 1,
                "payload": SafeJson({"foo": "bar"}),
                "editable": True,
                "status": ReviewStatus.WAITING,
            }
        )
        await PendingHumanReview.prisma().create(
            data={
                "nodeExecId": f"{COPILOT_NODE_PREFIX}some-block:{uuid4().hex[:8]}",
                "userId": user_id,
                "graphExecId": copilot_exec_id,
                "graphId": copilot_exec_id,
                "graphVersion": 1,
                "payload": SafeJson({"foo": "bar"}),
                "editable": True,
                "status": ReviewStatus.WAITING,
            }
        )

        reviews = await get_pending_reviews_for_user(user_id, 1, 25)
        by_exec = {r.graph_exec_id: r for r in reviews}

        real = by_exec[real_exec_id]
        assert real.expert_id == ana.id
        assert real.expert_name == "Ana"
        assert real.expert_avatar_url == "https://example.com/ana.png"
        assert real.agent_name == "Lead Finder"
        assert real.library_agent_id == lib_agent.id
        assert real.session_id is None

        copilot = by_exec[copilot_exec_id]
        assert copilot.session_id == session.session_id
        assert copilot.expert_id == ana.id
        assert copilot.expert_name == "Ana"
    finally:
        await _cleanup(user_id, graph_ids, session_ids)


async def test_plain_run_review_has_null_expert(server: SpinTestServer):
    user_id = f"review-enrich-plain-{uuid4()}"
    graph_id = str(uuid4())
    plain_exec_id = str(uuid4())
    graph_ids = [graph_id]

    await _create_user(user_id)
    try:
        await _create_graph(graph_id, user_id, "Plain Agent")

        await LibraryAgent.prisma().create(
            data={
                "userId": user_id,
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
                "name": "Plain Agent",
                "isCreatedByUser": True,
            }
        )

        await AgentGraphExecution.prisma().create(
            data={
                "id": plain_exec_id,
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
                "userId": user_id,
            }
        )

        await PendingHumanReview.prisma().create(
            data={
                "nodeExecId": f"node-exec-{uuid4()}",
                "userId": user_id,
                "graphExecId": plain_exec_id,
                "graphId": graph_id,
                "graphVersion": 1,
                "payload": SafeJson({"foo": "bar"}),
                "editable": True,
                "status": ReviewStatus.WAITING,
            }
        )

        reviews = await get_pending_reviews_for_user(user_id, 1, 25)
        plain = next(r for r in reviews if r.graph_exec_id == plain_exec_id)
        assert plain.expert_id is None
        assert plain.agent_name is not None
    finally:
        await _cleanup(user_id, graph_ids, [])


async def test_enrichment_never_resolves_foreign_node_executions(
    server: SpinTestServer,
):
    """A review pointing at a node execution owned by another user must not
    resolve that user's node id — the lookup is user-scoped, so node_id
    falls back to the raw exec id."""
    user_id = f"review-enrich-scope-{uuid4()}"
    other_user_id = f"review-enrich-scope-other-{uuid4()}"
    other_graph_id = str(uuid4())
    other_exec_id = str(uuid4())

    await _create_user(user_id)
    await _create_user(other_user_id)
    try:
        await _create_graph(other_graph_id, other_user_id, "Their Agent")

        block = await AgentBlock.prisma().find_first()
        assert block is not None, "test DB should have seeded blocks"
        node = await AgentNode.prisma().create(
            data={
                "agentBlockId": block.id,
                "agentGraphId": other_graph_id,
                "agentGraphVersion": 1,
            }
        )
        await AgentGraphExecution.prisma().create(
            data={
                "id": other_exec_id,
                "agentGraphId": other_graph_id,
                "agentGraphVersion": 1,
                "userId": other_user_id,
            }
        )
        foreign_node_exec = await AgentNodeExecution.prisma().create(
            data={
                "agentGraphExecutionId": other_exec_id,
                "agentNodeId": node.id,
            }
        )

        await PendingHumanReview.prisma().create(
            data={
                "nodeExecId": foreign_node_exec.id,
                "userId": user_id,
                "graphExecId": other_exec_id,
                "graphId": other_graph_id,
                "graphVersion": 1,
                "payload": SafeJson({"foo": "bar"}),
                "editable": True,
                "status": ReviewStatus.WAITING,
            }
        )

        reviews = await get_pending_reviews_for_user(user_id, 1, 25)
        crossed = next(r for r in reviews if r.node_exec_id == foreign_node_exec.id)
        assert crossed.node_id == foreign_node_exec.id  # fallback, not node.id
        assert crossed.expert_id is None
        assert crossed.agent_name is None
    finally:
        await _cleanup(user_id, [], [])
        await _cleanup(other_user_id, [other_graph_id], [])
