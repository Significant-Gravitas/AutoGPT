"""Background subtask handoff and continuation regressions."""

import asyncio
import logging
import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import background_execution
from backend.copilot.background_execution import background_timeout
from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata
from backend.copilot.tools.run_sub_session import RunSubSessionTool
from backend.copilot.tree import TreeRefusal, TurnEnvelope


def test_background_delivery_is_explicitly_opt_in():
    parameters = RunSubSessionTool().parameters
    assert parameters["properties"]["background"]["default"] is False
    assert "background" not in parameters["required"]


@pytest.mark.asyncio
async def test_worker_enforces_the_saved_deadline_even_during_silent_work():
    entry = CoPilotExecutionEntry(
        session_id="child",
        user_id="owner",
        turn_id="turn",
        message="work",
        message_metadata={
            "background_task_id": "task",
            "background_task_stage": "child",
        },
        envelope=TurnEnvelope(
            tree_id="tree",
            depth=1,
            deadline_at=datetime.now(UTC) + timedelta(milliseconds=25),
        ),
    )
    cancelled = asyncio.Event()

    async def silent_work(*_args):
        try:
            await asyncio.sleep(0.3)
            pytest.fail("The worker ran beyond the saved deadline")
        finally:
            cancelled.set()

    with (
        patch(
            "backend.copilot.executor.processor.claim_background_execution",
            AsyncMock(return_value=True),
        ),
        patch.object(CoPilotProcessor, "_run_turn", silent_work),
        pytest.raises(TimeoutError),
    ):
        await CoPilotProcessor()._execute_async(
            entry,
            threading.Event(),
            MagicMock(),
            CoPilotLogMetadata(logging.getLogger("test")),
        )
    assert cancelled.is_set()


def test_background_work_fails_closed_without_a_deadline_and_foreground_is_unchanged():
    entry = CoPilotExecutionEntry(session_id="session", user_id="owner", message="work")
    assert background_timeout(entry) is None
    entry.message_metadata = {"background_task_id": "task"}
    with pytest.raises(ValueError, match="deadline"):
        background_timeout(entry)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TreeRefusal("Budget used up."), ValueError()])
async def test_admission_failure_closes_the_stream_even_if_finalization_fails(
    monkeypatch: pytest.MonkeyPatch, error: Exception
):
    entry = CoPilotExecutionEntry(
        session_id="child",
        user_id="owner",
        turn_id="turn",
        message="work",
        message_metadata={
            "background_task_id": "task",
            "background_task_stage": "child",
        },
        envelope=TurnEnvelope(
            tree_id="tree",
            depth=1,
            deadline_at=datetime.now(UTC) + timedelta(minutes=1),
        ),
    )
    monkeypatch.setattr(
        background_execution,
        "background_task_db",
        lambda: MagicMock(claim_background_turn=AsyncMock(return_value=True)),
    )
    monkeypatch.setattr(
        background_execution, "readmit_turn", AsyncMock(side_effect=error)
    )
    finish = AsyncMock(side_effect=RuntimeError("Database unavailable"))
    monkeypatch.setattr(background_execution, "finish_background_execution", finish)
    completed = AsyncMock()
    monkeypatch.setattr(
        background_execution.stream_registry, "mark_session_completed", completed
    )

    with pytest.raises(type(error)) as raised:
        await background_execution.claim_background_execution(entry)

    assert raised.value is error
    message = str(error) or type(error).__name__
    finish.assert_awaited_once_with(entry, message)
    completed.assert_awaited_once_with("child", error_message=message, turn_id="turn")
