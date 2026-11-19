"""Library and Home reads must not wait on a slow scheduler.

Real database, real read paths; only the scheduler RPC client is replaced.
Before the bounded read, a scheduler that never answered held every one of
these reads for the 300s RPC timeout (BUILDER-8RR/8SK, 504s on
``/library/agents/{id}`` and ``/home``).
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

import pytest
from pytest_mock import MockerFixture

from backend.api.features.home import service as home_service
from backend.api.features.library import _get_many, _schedule_info
from backend.api.features.library import db as library_db
from backend.api.features.library.model import ScheduleInfo
from backend.api.model import CreateGraph
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.code_executor import ExecuteCodeBlock
from backend.blocks.llm import AITextGeneratorBlock
from backend.data import graph as graph_db
from backend.data.user import get_or_create_user
from backend.executor.scheduler import GraphExecutionJobInfo
from backend.usecases.sample import create_test_graph
from backend.util.test import SpinTestServer

_READ_TIMEOUT_S = 0.5
# Generous enough for a slow CI database, far below the 60s the stub hangs.
_PROMPT_S = 5.0

SchedulerMode = Literal["healthy", "hangs", "errors"]


class _SchedulerStub:
    """Stands in for ``SchedulerClient`` at the RPC boundary."""

    def __init__(self, mode: SchedulerMode, schedules: list[GraphExecutionJobInfo]):
        self.mode = mode
        self.schedules = schedules

    async def get_graph_execution_schedules(
        self, graph_id: str | None = None, user_id: str | None = None
    ) -> list[GraphExecutionJobInfo]:
        await self._misbehave()
        return [
            s
            for s in self.schedules
            if s.user_id == user_id and graph_id in (None, s.graph_id)
        ]

    async def get_execution_schedules(
        self, user_id: str | None = None
    ) -> list[GraphExecutionJobInfo]:
        await self._misbehave()
        return [s for s in self.schedules if s.user_id == user_id]

    async def _misbehave(self) -> None:
        if self.mode == "hangs":
            await asyncio.sleep(60)
        if self.mode == "errors":
            raise ConnectionError("scheduler unavailable")


async def _timed(coro):
    started = time.monotonic()
    result = await asyncio.wait_for(coro, timeout=30)
    return result, time.monotonic() - started


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("mode", ["healthy", "hangs", "errors"])
async def test_library_and_home_reads_do_not_wait_on_the_scheduler(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch, mode: SchedulerMode
):
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"schedule-read-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Schedule Read",
        }
    )
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), user.id
    )
    agent = await library_db.get_library_agent_by_graph_id(user.id, graph.id)
    assert agent is not None

    next_run = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    schedule = GraphExecutionJobInfo(
        id=f"schedule-{uuid.uuid4().hex[:8]}",
        name="Hourly",
        next_run_time=next_run,
        user_id=user.id,
        graph_id=graph.id,
        graph_version=graph.version,
        cron="0 * * * *",
        input_data={},
    )
    scheduler = _SchedulerStub(mode, [schedule])
    for module in (_schedule_info, home_service):
        monkeypatch.setattr(module, "get_scheduler_client", lambda: scheduler)
        monkeypatch.setattr(
            module.settings.config, "schedule_read_timeout_seconds", _READ_TIMEOUT_S
        )
    scheduled = mode == "healthy"

    single, elapsed = await _timed(library_db.get_library_agent(agent.id, user.id))
    assert elapsed < _PROMPT_S
    assert (single.id, single.graph_id) == (agent.id, graph.id)
    assert single.is_scheduled is scheduled
    assert single.next_scheduled_run == (next_run if scheduled else None)
    # Without an answer the schedule is unknown, which the page must not show
    # as "not scheduled".
    assert single.schedule_info_unavailable is not scheduled

    listing, elapsed = await _timed(
        library_db.list_library_agents(user_id=user.id, page_size=50)
    )
    assert elapsed < _PROMPT_S
    [listed] = [a for a in listing.agents if a.id == agent.id]
    assert listed.is_scheduled is scheduled
    assert listed.schedule_info_unavailable is not scheduled

    batch, elapsed = await _timed(
        _get_many.get_library_agents_by_ids([agent.id, "missing-id"], user.id)
    )
    assert elapsed < _PROMPT_S
    # The batch answers exactly what the single read does, for every card.
    assert [a.model_dump() for a in batch] == [single.model_dump()]

    home, elapsed = await _timed(home_service.build_home_dashboard(user_id=user.id))
    assert elapsed < _PROMPT_S
    upcoming_ids = [task.id for task in home.upcoming_tasks]
    assert upcoming_ids == ([schedule.id] if scheduled else [])
    assert home.schedules_unavailable is not scheduled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, schedules, expected",
    [
        ("healthy", True, ScheduleInfo(next_runs={"graph-1": "2030-01-01T09:00:00"})),
        # A scheduler that answers with no schedules: known to be none.
        ("healthy", False, ScheduleInfo()),
        ("hangs", True, ScheduleInfo(available=False)),
        ("errors", True, ScheduleInfo(available=False)),
    ],
)
async def test_schedule_info_is_unknown_only_when_the_scheduler_did_not_answer(
    monkeypatch: pytest.MonkeyPatch,
    mode: SchedulerMode,
    schedules: bool,
    expected: ScheduleInfo,
):
    schedule = GraphExecutionJobInfo(
        id="schedule-1",
        name="Daily",
        next_run_time="2030-01-01T09:00:00",
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        cron="0 9 * * *",
        input_data={},
    )
    scheduler = _SchedulerStub(mode, [schedule] if schedules else [])
    monkeypatch.setattr(_schedule_info, "get_scheduler_client", lambda: scheduler)
    monkeypatch.setattr(
        _schedule_info.settings.config,
        "schedule_read_timeout_seconds",
        _READ_TIMEOUT_S,
    )

    info = await asyncio.wait_for(
        _schedule_info._fetch_schedule_info("user-1"), timeout=30
    )

    assert info == expected


@pytest.mark.asyncio(loop_scope="session")
async def test_batch_read_leaves_out_other_users_agents(server: SpinTestServer):
    owner, stranger = [
        await get_or_create_user(
            {
                "sub": str(uuid.uuid4()),
                "email": f"batch-{uuid.uuid4().hex[:8]}@example.com",
                "name": "Batch Read",
            }
        )
        for _ in range(2)
    ]
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), owner.id
    )
    agent = await library_db.get_library_agent_by_graph_id(owner.id, graph.id)
    assert agent is not None

    assert await _get_many.get_library_agents_by_ids([agent.id], stranger.id) == []
    assert [
        a.id for a in await _get_many.get_library_agents_by_ids([agent.id], owner.id)
    ] == [agent.id]


@pytest.mark.asyncio(loop_scope="session")
async def test_batch_read_converts_and_walks_a_sub_agent_its_agents_share_once(
    server: SpinTestServer, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
):
    user_id, agent_ids = await _agents_running_one_sub_agent(
        graph_db.Node(block_id=ExecuteCodeBlock().id)
    )
    node_from_db = mocker.spy(graph_db.NodeModel, "from_db")
    lookups = _count_block_lookups(monkeypatch)

    batch = await _get_many.get_library_agents_by_ids(agent_ids, user_id)

    # Each agent's own node and the sub-agent's node are converted once.
    assert node_from_db.call_count == 3 + 1
    batch_lookups = lookups.pop(ExecuteCodeBlock().id, 0)
    single = await library_db.get_library_agent(agent_ids[0], user_id)
    assert batch_lookups == lookups.pop(ExecuteCodeBlock().id, 0) > 0
    singles = [single] + [
        await library_db.get_library_agent(agent_id, user_id)
        for agent_id in agent_ids[1:]
    ]
    assert [a.model_dump() for a in batch] == [a.model_dump() for a in singles]
    for agent in batch:
        assert agent.credentials_input_schema is not None
        assert "e2b_api_key_credentials" in agent.credentials_input_schema["properties"]


@pytest.mark.asyncio(loop_scope="session")
async def test_batch_read_walks_a_failing_sub_agent_once(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    user_id, agent_ids = await _agents_running_one_sub_agent(
        graph_db.Node(
            block_id=AITextGeneratorBlock().id,
            input_default={"prompt": "Hi", "model": "no-such-model"},
        )
    )
    lookups = _count_block_lookups(monkeypatch)

    with pytest.raises(ValueError, match="no-such-model"):
        await library_db.get_library_agent(agent_ids[0], user_id)
    single_lookups = lookups.pop(AITextGeneratorBlock().id, 0)
    assert single_lookups > 0

    assert await _get_many.get_library_agents_by_ids(agent_ids, user_id) == []
    assert lookups.pop(AITextGeneratorBlock().id, 0) == single_lookups


def _count_block_lookups(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Counts each `Node.block` lookup by block id; walking a graph's nodes
    looks up each node's block."""
    lookups: dict[str, int] = {}
    block = graph_db.Node.block

    def counting(node: graph_db.Node):
        lookups[node.block_id] = lookups.get(node.block_id, 0) + 1
        return block.fget(node)

    monkeypatch.setattr(graph_db.Node, "block", property(counting))
    return lookups


async def _agents_running_one_sub_agent(
    sub_agent_node: graph_db.Node,
) -> tuple[str, list[str]]:
    """A new user's id, and the ids of three of their library agents that are
    each one node running the same sub-agent, itself the one node given."""
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"batch-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Batch Read",
        }
    )
    sub_agent = await graph_db.create_graph(
        graph_db.Graph(name="Sub-agent", description="", nodes=[sub_agent_node]),
        user.id,
    )
    runs_sub_agent = {
        "user_id": user.id,
        "graph_id": sub_agent.id,
        "graph_version": sub_agent.version,
        "inputs": {},
        "input_schema": {},
        "output_schema": {},
    }
    agent_ids: list[str] = []
    for i in range(3):
        graph = await graph_db.create_graph(
            graph_db.Graph(
                name=f"Caller {i}",
                description="",
                nodes=[
                    graph_db.Node(
                        block_id=AgentExecutorBlock().id, input_default=runs_sub_agent
                    )
                ],
            ),
            user.id,
        )
        [agent] = await library_db.create_library_agent(
            graph, user.id, create_library_agents_for_sub_graphs=False
        )
        agent_ids.append(agent.id)
    return user.id, agent_ids
