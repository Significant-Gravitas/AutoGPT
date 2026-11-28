"""Recovery after a schedule create ends before its index write."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from backend.executor import scheduler as scheduler_module
from backend.executor.schedule_index_test_helpers import _scheduler_with_index
from backend.executor.scheduler import Jobstores


def test_replayed_create_restores_the_missing_schedule_index_row() -> None:
    service = _scheduler_with_index()
    service.scheduler = BackgroundScheduler(
        jobstores={Jobstores.EXECUTION.value: MemoryJobStore()}
    )
    service.scheduler.start(paused=True)
    schedule_id = str(uuid4())
    request: dict[str, Any] = {
        "schedule_id": schedule_id,
        "user_id": "user-1",
        "graph_id": "graph-1",
        "graph_version": 1,
        "cron": "0 9 * * *",
        "input_data": {},
        "input_credentials": {},
        "name": "Morning run",
        "user_timezone": "UTC",
    }

    try:
        with (
            asyncio.Runner(loop_factory=asyncio.new_event_loop) as runner,
            patch.object(scheduler_module, "run_async", side_effect=runner.run),
            patch.object(
                scheduler_module.execution_utils,
                "validate_and_construct_node_execution_input",
                new_callable=AsyncMock,
            ) as validate,
        ):
            with (
                patch.object(
                    service, "_upsert_schedule_index_row", side_effect=InterruptedError
                ),
                pytest.raises(InterruptedError),
            ):
                service.add_graph_execution_schedule(**request)

            assert (
                service.get_graph_execution_schedules(
                    user_id="user-1", graph_id="graph-1"
                )
                == []
            )
            replay = service.add_graph_execution_schedule(**request)

        validate.assert_awaited_once()
        assert replay.id == schedule_id
        assert [job.id for job in service.scheduler.get_jobs()] == [schedule_id]
        assert service._schedule_index_ready
        listed = service.get_graph_execution_schedules(
            user_id="user-1", graph_id="graph-1"
        )
        assert [job.id for job in listed] == [schedule_id]
    finally:
        service.scheduler.shutdown()
