import pytest

from autogpt_server.data import db, graph
from autogpt_server.executor import ExecutionScheduler
from autogpt_server.usecases.sample import create_test_graph
from autogpt_server.util.service import get_service_client
from autogpt_server.util.test import SpinTestServer


@pytest.mark.asyncio(scope="session")
async def test_agent_schedule():
    await db.connect()
    test_graph = await graph.create_graph(create_test_graph())

    async with SpinTestServer():
        scheduler = get_service_client(ExecutionScheduler)

        schedules = scheduler.get_execution_schedules(test_graph.id)
        assert len(schedules) == 0

        schedule_id = scheduler.add_execution_schedule(
            graph_id=test_graph.id,
            graph_version=1,
            cron="0 0 * * *",
            input_data={"input": "data"},
        )
        assert schedule_id

        schedules = scheduler.get_execution_schedules(test_graph.id)
        assert len(schedules) == 1
        assert schedules[schedule_id] == "0 0 * * *"

        scheduler.update_schedule(schedule_id, is_enabled=False)
        schedules = scheduler.get_execution_schedules(test_graph.id)
        assert len(schedules) == 0
