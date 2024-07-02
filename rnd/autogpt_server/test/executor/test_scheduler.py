import pytest

import test_manager
from autogpt_server.executor.scheduler import ExecutionScheduler
from autogpt_server.util.service import PyroNameServer, get_service_client


@pytest.mark.asyncio(scope="session")
async def test_agent_schedule():
    await test_manager.db.connect()
    test_graph = await test_manager.create_test_graph()

    with PyroNameServer():
        with ExecutionScheduler():
            scheduler = get_service_client(ExecutionScheduler)

            schedules = scheduler.get_execution_schedules(test_graph.id)
            assert len(schedules) == 0

            schedule_id = scheduler.add_execution_schedule(
                test_graph.id,
                "0 0 * * *",
                {"input": "data"}
            )
            assert schedule_id

            schedules = scheduler.get_execution_schedules(test_graph.id)
            assert len(schedules) == 1
            assert schedules[schedule_id] == "0 0 * * *"

            scheduler.update_schedule(schedule_id, is_enabled=False)
            schedules = scheduler.get_execution_schedules(test_graph.id)
            assert len(schedules) == 0
