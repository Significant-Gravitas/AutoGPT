"""A schedule create keyed by the caller is applied at most once.

A create whose answer is lost (the scheduler applied it, then the caller timed
out with ``scheduler_unconfirmed``) is retried by the user. Without a key,
every retry minted a new id and added another recurring schedule. Real
scheduler service, real jobstore.
"""

import asyncio
import uuid

import pytest

from backend.api.model import CreateGraph
from backend.data import db
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import ConflictError, NotAuthorizedError
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_create_with_the_same_key_makes_one_schedule(server: SpinTestServer):
    await db.connect()
    owner = await create_test_user()
    other = await create_test_user(alt_user=True)
    graph = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=create_test_graph()), user_id=owner.id
    )
    other_graph = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=create_test_graph()), user_id=owner.id
    )
    scheduler = get_scheduler_client()

    async def create(
        key: str,
        user_id: str = owner.id,
        name: str = "Morning",
        graph_id: str = graph.id,
        cron: str = "0 9 * * *",
        timezone: str = "UTC",
        inputs: dict | None = None,
    ):
        return await scheduler.add_execution_schedule(
            graph_id=graph_id,
            user_id=user_id,
            graph_version=graph.version,
            cron=cron,
            input_data=inputs or {"input": "data"},
            input_credentials={},
            name=name,
            user_timezone=timezone,
            schedule_id=key,
        )

    key = str(uuid.uuid4())
    racing_key = str(uuid.uuid4())
    try:
        first = await create(key)
        retried = await create(key)
        # Two copies of one submission in flight at once.
        raced = await asyncio.gather(create(racing_key), create(racing_key))

        assert first.id == retried.id == key
        assert {s.id for s in raced} == {racing_key}
        stored = await scheduler.get_execution_schedules(graph.id, owner.id)
        assert sorted(s.id for s in stored) == sorted([key, racing_key])

        # A key can't be used to read or take over someone else's schedule.
        with pytest.raises(NotAuthorizedError):
            await create(key, user_id=other.id, name="Hijack")
        # Nor can it pass off one agent's schedule as another agent's create,
        # or answer a different submission with the schedule it doesn't match.
        with pytest.raises(ConflictError):
            await create(key, graph_id=other_graph.id)
        for changed in (
            {"cron": "30 18 * * *"},
            {"timezone": "Europe/Paris"},
            {"inputs": {"input": "other"}},
            {"name": "Evening"},
        ):
            with pytest.raises(ConflictError):
                await create(key, **changed)
        [kept] = [
            s
            for s in await scheduler.get_execution_schedules(graph.id, owner.id)
            if s.id == key
        ]
        assert (kept.user_id, kept.name, kept.cron) == (
            owner.id,
            "Morning",
            "0 9 * * *",
        )
    finally:
        for schedule_id in (key, racing_key):
            try:
                await scheduler.delete_schedule(schedule_id, user_id=owner.id)
            except Exception:
                pass
