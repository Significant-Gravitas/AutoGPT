from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.store import db
from backend.util.exceptions import NotFoundError


def _submission() -> MagicMock:
    graph = MagicMock(
        id="graph-1",
        version=3,
        userId="creator-1",
        organizationId="org-1",
        teamId="team-1",
        Nodes=[],
    )
    return MagicMock(AgentGraph=graph)


@asynccontextmanager
async def _attachment_barrier(_graph_ids):
    yield


@pytest.mark.asyncio
async def test_review_rechecks_creator_live_team_scope(mocker) -> None:
    submission = _submission()
    client = MagicMock(find_unique=AsyncMock(return_value=submission))
    mocker.patch("prisma.models.StoreListingVersion.prisma", return_value=client)
    mocker.patch.object(db, "get_sub_graphs", AsyncMock(return_value=[]))
    mocker.patch.object(db, "agent_graph_attachment_barriers", _attachment_barrier)

    @asynccontextmanager
    async def denied(*args):
        assert args == (
            "creator-1",
            "org-1",
            "team-1",
            "create",
            "graph-1",
            3,
        )
        yield False

    mocker.patch.object(db, "live_agent_graph_access_barrier", denied)

    with pytest.raises(NotFoundError):
        async with db._stable_review_graphs("version-1"):
            pytest.fail("revoked creator scope must not reach approval")


@pytest.mark.asyncio
async def test_review_rejects_inaccessible_subgraph_before_approval(mocker) -> None:
    submission = _submission()
    submission.AgentGraph.Nodes = [
        MagicMock(
            AgentBlock=MagicMock(id=db._AGENT_EXECUTOR_BLOCK_ID),
            constantInput={"graph_id": "foreign-graph", "graph_version": 1},
        )
    ]
    client = MagicMock(find_unique=AsyncMock(return_value=submission))
    mocker.patch("prisma.models.StoreListingVersion.prisma", return_value=client)
    mocker.patch.object(db, "get_sub_graphs", AsyncMock(return_value=[]))
    attachment_barriers = mocker.patch.object(db, "agent_graph_attachment_barriers")
    live_barrier = mocker.patch.object(db, "live_agent_graph_access_barrier")

    with pytest.raises(NotFoundError, match="inaccessible subgraph"):
        async with db._stable_review_graphs("version-1"):
            pytest.fail("mixed-tenant composition must not reach approval")

    attachment_barriers.assert_not_called()
    live_barrier.assert_not_called()


@pytest.mark.asyncio
async def test_review_yields_stable_graphs_with_live_scope(mocker) -> None:
    submission = _submission()
    child = MagicMock(
        id="graph-2",
        version=5,
        userId="creator-1",
        organizationId="org-1",
        teamId="team-1",
        Nodes=[],
    )
    submission.AgentGraph.Nodes = [
        MagicMock(
            AgentBlock=MagicMock(id=db._AGENT_EXECUTOR_BLOCK_ID),
            constantInput={"graph_id": "graph-2", "graph_version": 5},
        )
    ]
    client = MagicMock(find_unique=AsyncMock(return_value=submission))
    mocker.patch("prisma.models.StoreListingVersion.prisma", return_value=client)
    mocker.patch.object(db, "get_sub_graphs", AsyncMock(return_value=[child]))
    mocker.patch.object(db, "agent_graph_attachment_barriers", _attachment_barrier)

    barrier_calls: list[tuple] = []

    @asynccontextmanager
    async def allowed(*args):
        barrier_calls.append(args)
        yield True

    mocker.patch.object(db, "live_agent_graph_access_barrier", allowed)

    async with db._stable_review_graphs("version-1") as (locked, subgraphs):
        assert locked is submission
        assert subgraphs == [child]

    assert client.find_unique.await_count == 2
    assert barrier_calls == [
        ("creator-1", "org-1", "team-1", "create", "graph-1", 3),
        ("creator-1", "org-1", "team-1", "create", "graph-2", 5),
    ]
