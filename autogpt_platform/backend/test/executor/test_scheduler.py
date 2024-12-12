import pytest

from backend.data import db
from backend.executor import ExecutionScheduler
from backend.server.model import CreateGraph
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.service import get_service_client
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(scope="session")
async def test_agent_schedule(server: SpinTestServer):
    await db.connect()
    test_user = await create_test_user()
    test_graph = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=create_test_graph()),
        user_id=test_user.id,
    )

    scheduler = get_service_client(ExecutionScheduler)
    schedules = scheduler.get_execution_schedules(test_graph.id, test_user.id)
    assert len(schedules) == 0

    schedule = scheduler.add_execution_schedule(
        graph_id=test_graph.id,
        user_id=test_user.id,
        graph_version=1,
        cron="0 0 * * *",
        input_data={"input": "data"},
    )
    assert schedule

    schedules = scheduler.get_execution_schedules(test_graph.id, test_user.id)
    assert len(schedules) == 1
    assert schedules[0].cron == "0 0 * * *"

    scheduler.delete_schedule(schedule.id, user_id=test_user.id)
    schedules = scheduler.get_execution_schedules(test_graph.id, user_id=test_user.id)
    assert len(schedules) == 0
