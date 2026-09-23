"""Every route that starts expert work reaches the same spend-approval seam
(SECRT-2599): with the expert at her threshold, the run is created, parked
unpublished, and nothing else happens. Chat's ``run_block`` path is covered in
``copilot/tools/run_block_test.py``; ``delegate_to_expert`` and scheduled
copilot turns open a session scoped to the expert (their own tests pin that)
and then spend through these same seams."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.executor.utils_test import (
    _mock_add_graph_execution_create_path,
    _mock_expert_personal_tenancy,
    _mock_spend_gate,
    _spend_needed,
)


@pytest.fixture
def parked(mocker):
    """Seam A armed at the threshold; yields (park, queue, create)."""
    mock_edb, _ = _mock_add_graph_execution_create_path(mocker)
    _mock_expert_personal_tenancy(mocker)
    _, park, queue = _mock_spend_gate(mocker, _spend_needed())
    return park, queue, mock_edb.create_graph_execution


def _assert_parked(park, queue, create, *, expert_id: str = "expert-1") -> None:
    park.assert_awaited_once()
    assert park.await_args.kwargs["graph_exec_id"] == "exec-id"
    assert create.await_args.kwargs["expert_id"] == expert_id
    queue.publish_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_schedule_fire_is_parked(mocker, parked) -> None:
    from backend.executor import scheduler

    park, queue, create = parked
    mocker.patch.object(
        scheduler,
        "get_database_manager_async_client",
        return_value=MagicMock(increment_onboarding_runs=AsyncMock()),
    )
    args = scheduler.GraphExecutionJobArgs(
        user_id="owner",
        graph_id="g",
        graph_version=1,
        cron="0 9 * * *",
        input_data={},
        organization_id="org",
        team_id="team",
        expert_id="expert-1",
        schedule_id="sched-1",
    )

    await scheduler._execute_graph(**args.model_dump())

    _assert_parked(park, queue, create)


@pytest.mark.asyncio
async def test_webhook_trigger_is_parked(mocker, parked) -> None:
    from backend.api.features.integrations import router as ingress

    park, queue, create = parked
    mocker.patch.object(
        ingress,
        "experts_db",
        return_value=MagicMock(
            resolve_private_expert_tenancy=AsyncMock(return_value=("org", "team"))
        ),
    )
    trigger_node = MagicMock(id="node-1")
    trigger_node.block.is_triggered_by_event_type.return_value = True
    mocker.patch.object(
        ingress,
        "get_graph",
        AsyncMock(return_value=MagicMock(webhook_input_node=trigger_node)),
    )
    preset = MagicMock(
        id="preset-1",
        user_id="owner",
        is_active=True,
        expert_id="expert-1",
        graph_id="g",
        graph_version=1,
        inputs={},
        credentials={},
    )
    webhook = MagicMock(id="wh-1", user_id="owner")

    await ingress._execute_webhook_preset_trigger(preset, webhook, "wh-1", "push", {})

    _assert_parked(park, queue, create)


@pytest.mark.asyncio
async def test_preset_api_run_is_parked(mocker, parked) -> None:
    from backend.api.features.library.routes import presets as presets_routes

    park, queue, create = parked
    preset = MagicMock(
        graph_id="g",
        graph_version=1,
        inputs={},
        credentials={},
        expert_id="expert-1",
        organization_id="org",
        team_id="team",
    )
    mocker.patch.object(presets_routes.db, "get_preset", AsyncMock(return_value=preset))

    result = await presets_routes.execute_preset(
        preset_id="preset-1",
        user_id="owner",
        ctx=MagicMock(org_id="org", team_id="team"),
        inputs={},
        credential_inputs={},
    )

    assert result.status.value == "REVIEW"
    _assert_parked(park, queue, create)


@pytest.mark.asyncio
async def test_run_agent_from_expert_chat_is_parked_and_says_so(mocker, parked) -> None:
    from backend.copilot.tools import run_agent as run_agent_mod
    from backend.copilot.tools._test_data import make_session
    from backend.copilot.tools.models import ExecutionStartedResponse

    park, queue, create = parked
    library_agent = MagicMock(id="lib-1", graph_id="g")
    library_agent.name = "Weekly report"
    mocker.patch.object(
        run_agent_mod,
        "get_or_create_library_agent",
        AsyncMock(return_value=library_agent),
    )
    mocker.patch.object(run_agent_mod, "emit_tool_display_name")
    mocker.patch.object(run_agent_mod, "track_agent_run_success")
    mocker.patch.object(run_agent_mod, "_safe_link_to_chat_share", AsyncMock())
    session = make_session(user_id="owner", expert_id="expert-1")

    response = await run_agent_mod.RunAgentTool()._run_agent(
        user_id="owner",
        session=session,
        graph=MagicMock(id="g"),
        graph_credentials={},
        inputs={},
        dry_run=False,
    )

    assert isinstance(response, ExecutionStartedResponse)
    assert response.status == "REVIEW"
    assert "approval" in response.message
    _assert_parked(park, queue, create)
    assert session.successful_agent_runs.get("g", 0) == 0
