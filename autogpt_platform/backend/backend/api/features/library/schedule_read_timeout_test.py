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

from backend.api.features.home import service as home_service
from backend.api.features.library import _get_many, _schedule_info
from backend.api.features.library import db as library_db
from backend.api.model import CreateGraph
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

    listing, elapsed = await _timed(
        library_db.list_library_agents(user_id=user.id, page_size=50)
    )
    assert elapsed < _PROMPT_S
    [listed] = [a for a in listing.agents if a.id == agent.id]
    assert listed.is_scheduled is scheduled

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
