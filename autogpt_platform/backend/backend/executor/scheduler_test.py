import uuid
from datetime import datetime, timedelta, timezone

import pytest

from backend.api.model import CreateGraph
from backend.data import db
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.clients import get_scheduler_client
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_schedule(server: SpinTestServer):
    await db.connect()
    test_user = await create_test_user()
    test_graph = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=create_test_graph()),
        user_id=test_user.id,
    )

    scheduler = get_scheduler_client()
    schedules = await scheduler.get_execution_schedules(test_graph.id, test_user.id)
    assert len(schedules) == 0

    schedule = await scheduler.add_execution_schedule(
        graph_id=test_graph.id,
        user_id=test_user.id,
        graph_version=1,
        cron="0 0 * * *",
        input_data={"input": "data"},
        input_credentials={},
    )
    assert schedule

    schedules = await scheduler.get_execution_schedules(test_graph.id, test_user.id)
    assert len(schedules) == 1
    assert schedules[0].cron == "0 0 * * *"

    await scheduler.delete_schedule(schedule.id, user_id=test_user.id)
    schedules = await scheduler.get_execution_schedules(
        test_graph.id, user_id=test_user.id
    )
    assert len(schedules) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_copilot_turn_schedule_one_shot(server: SpinTestServer):
    await db.connect()
    test_user = await create_test_user(alt_user=True)
    session_id = f"session-{uuid.uuid4()}"

    scheduler = get_scheduler_client()
    # Schedule should not yet exist for this fresh session.
    existing = await scheduler.get_execution_schedules(
        session_id=session_id, user_id=test_user.id
    )
    assert existing == []

    run_at = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    schedule = await scheduler.add_copilot_turn_schedule(
        user_id=test_user.id,
        session_id=session_id,
        message="check on the long-running task",
        run_at=run_at,
        user_timezone="UTC",
    )
    assert schedule.kind == "copilot_turn"
    assert schedule.session_id == session_id
    assert schedule.run_at is not None
    assert schedule.cron is None

    # Polymorphic listing returns the copilot-turn schedule.
    listed = await scheduler.get_execution_schedules(
        session_id=session_id, user_id=test_user.id
    )
    assert len(listed) == 1
    assert listed[0].kind == "copilot_turn"
    assert listed[0].id == schedule.id

    # Graph-only filter excludes copilot-turn schedules.
    graph_only = await scheduler.get_graph_execution_schedules(user_id=test_user.id)
    assert all(s.kind == "graph" for s in graph_only)
    assert schedule.id not in {s.id for s in graph_only}

    # Cleanup — delete_schedule is polymorphic.
    await scheduler.delete_schedule(schedule.id, user_id=test_user.id)
    remaining = await scheduler.get_execution_schedules(
        session_id=session_id, user_id=test_user.id
    )
    assert remaining == []


@pytest.mark.asyncio(loop_scope="session")
async def test_copilot_turn_schedule_requires_cron_xor_run_at(server: SpinTestServer):
    await db.connect()
    test_user = await create_test_user(alt_user=True)
    scheduler = get_scheduler_client()
    session_id = f"session-{uuid.uuid4()}"

    with pytest.raises(Exception) as exc:
        await scheduler.add_copilot_turn_schedule(
            user_id=test_user.id,
            session_id=session_id,
            message="x",
            user_timezone="UTC",
        )
    # ValueError from _build_trigger propagates as a RemoteError
    # through the AppService transport; just verify the call rejected.
    assert exc.value is not None
