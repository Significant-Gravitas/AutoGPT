import pytest

import test_manager  # type: ignore
from autogpt_server.executor.scheduler import ExecutionScheduler
from autogpt_server.util.service import PyroNameServer, get_service_client
from autogpt_server.server import AgentServer


@pytest.mark.asyncio(scope="session")
async def test_agent_schedule():
    await test_manager.db.connect()
    test_graph = await test_manager.create_test_graph()

    with PyroNameServer():
        with AgentServer():
            with ExecutionScheduler():
                scheduler = get_service_client(ExecutionScheduler)

                schedules = scheduler.get_execution_schedules(test_graph.graph_id)
                assert len(schedules) == 0

                schedule_id = scheduler.add_execution_schedule(
                    graph_id=test_graph.graph_id,
                    version=1,
                    cron="0 0 * * *",
                    input_data={"input": "data"},
                )
                assert schedule_id

                schedules = scheduler.get_execution_schedules(test_graph.graph_id)
                assert len(schedules) == 1
                assert schedules[schedule_id] == "0 0 * * *"

                scheduler.update_schedule(schedule_id, is_enabled=False)
                schedules = scheduler.get_execution_schedules(test_graph.graph_id)
                assert len(schedules) == 0
