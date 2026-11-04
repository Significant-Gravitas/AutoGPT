"""The three tools a worker keeps its task's record with."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.tasks import contract as c
from backend.api.features.tasks.contract_db import ContractUpdate
from backend.api.features.tasks.models import (
    Actor,
    ContractState,
    Task,
    TaskPriority,
    TaskStatus,
)

from ._test_data import make_session
from .models import ErrorResponse, TaskContractToolResponse
from .tasks_contract_tools import AddEvidenceTool, AskUserTool, UpdateTaskTool

_USER = "user-contract"
_NOW = datetime(2026, 11, 4, tzinfo=UTC)
_TASK = Task(
    id="task-1",
    number=1,
    key="T-1",
    title="Launch pricing page",
    status=TaskStatus.IN_PROGRESS,
    priority=TaskPriority.NONE,
    created_at=_NOW,
    updated_at=_NOW,
)


def _update(contract: c.Contract, **extra) -> ContractUpdate:
    state = c.derive_state(_TASK.status, contract, None)
    return ContractUpdate(
        task=_TASK.model_copy(update={"state": state}),
        contract=contract,
        version=2,
        patches=[],
        **extra,
    )


@pytest.fixture
def store(monkeypatch):
    store = MagicMock()
    module = "backend.copilot.tools.tasks_contract_tools"
    monkeypatch.setattr(f"{module}.tasks_store", lambda: store)
    monkeypatch.setattr(f"{module}.resolve_task", AsyncMock(return_value=_TASK))
    monkeypatch.setattr(f"{module}.caller_actor", AsyncMock(return_value=Actor.otto()))
    monkeypatch.setattr(f"{module}.publish_contract_update", AsyncMock())
    monkeypatch.setattr(
        "backend.copilot.tools.tasks_helpers.is_feature_enabled",
        AsyncMock(return_value=True),
    )
    return store


async def _run(tool, **kwargs):
    return await tool._execute(user_id=_USER, session=make_session(_USER), **kwargs)


@pytest.mark.asyncio
async def test_update_task_sends_only_what_changed(store):
    contract = c.Contract(goal="Launch pricing page")
    store.contract_update_task = AsyncMock(return_value=_update(contract))

    result = await _run(
        UpdateTaskTool(), goal="Launch pricing page", plan=None, done_when=["Live"]
    )

    assert isinstance(result, TaskContractToolResponse)
    assert store.contract_update_task.await_args.args[2] == {
        "goal": "Launch pricing page",
        "done_when": ["Live"],
    }


@pytest.mark.asyncio
async def test_an_empty_update_is_refused(store):
    store.contract_update_task = AsyncMock()

    result = await _run(UpdateTaskTool())

    assert isinstance(result, ErrorResponse)
    store.contract_update_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_refused_patch_comes_back_as_the_reason(store):
    store.contract_update_task = AsyncMock(
        side_effect=ValueError("No step with id s9. Known ids: s1.")
    )

    result = await _run(UpdateTaskTool(), steps=[{"id": "s9", "status": "done"}])

    assert isinstance(result, ErrorResponse)
    assert "Known ids: s1" in result.message


@pytest.mark.asyncio
async def test_a_blocker_tells_the_worker_to_stop(store):
    contract = c.Contract(goal="x", blocker=c.make_blocker(None, "No access"))
    store.contract_update_task = AsyncMock(return_value=_update(contract))

    result = await _run(UpdateTaskTool(), blocker={"what_stopped_it": "No access"})

    assert isinstance(result, TaskContractToolResponse)
    assert result.state == ContractState.STUCK.value
    assert result.message == "Blocker recorded. End your turn."


@pytest.mark.asyncio
async def test_add_evidence_passes_raw_content_through(store):
    contract, item, _ = c.add_evidence(c.Contract(goal="x"), "file", "Page", "HTML")
    store.contract_add_evidence = AsyncMock(
        return_value=_update(contract, evidence=item)
    )

    result = await _run(
        AddEvidenceTool(),
        kind="file",
        title="Page",
        summary="HTML",
        content="<html></html>",
        criterion_ids="d1",
    )

    assert isinstance(result, TaskContractToolResponse)
    assert result.message == "Evidence e1 saved (v2)."
    kwargs = store.contract_add_evidence.await_args.kwargs
    assert kwargs["content"] == "<html></html>"
    assert kwargs["criterion_ids"] == ["d1"]


@pytest.mark.asyncio
async def test_ask_user_opens_one_question(store):
    contract, _, _ = c.open_ask(c.Contract(goal="x"), "Approve the page?")
    store.contract_ask = AsyncMock(return_value=_update(contract))

    result = await _run(
        AskUserTool(),
        question="Approve the page?",
        options=["Ship it", "Change copy"],
        expires_at="2026-11-05T09:00:00+00:00",
    )

    assert isinstance(result, TaskContractToolResponse)
    assert result.state == ContractState.NEEDS_YOU.value
    kwargs = store.contract_ask.await_args.kwargs
    assert kwargs["options"] == ["Ship it", "Change copy"]
    assert kwargs["expires_at"] == datetime(2026, 11, 5, 9, tzinfo=UTC)
