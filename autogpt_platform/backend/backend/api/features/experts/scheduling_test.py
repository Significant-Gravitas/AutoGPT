from types import SimpleNamespace
from unittest.mock import AsyncMock

import prisma.models
import pytest
from prisma.enums import ResourceVisibility

from backend.api.features.experts import scheduling
from backend.util.exceptions import ExpertRunPausedError


@pytest.mark.asyncio
async def test_reattach_rehomes_presets_to_current_personal_tenancy(mocker) -> None:
    resolve_tenancy = mocker.patch(
        "backend.api.features.experts.experts_db.resolve_private_expert_tenancy",
        new=AsyncMock(return_value=("current-personal-org", "current-personal-team")),
    )
    preset_client = mocker.MagicMock()
    preset_client.update_many = AsyncMock(return_value=1)
    workflow_client = mocker.MagicMock()
    workflow_client.find_many = AsyncMock(return_value=[])
    mocker.patch.object(
        scheduling.prisma.models.AgentPreset,
        "prisma",
        return_value=preset_client,
    )
    mocker.patch.object(
        scheduling.prisma.models.ExpertWorkflow,
        "prisma",
        return_value=workflow_client,
    )
    scheduler_client = mocker.MagicMock()
    scheduler_client.get_execution_schedules = AsyncMock(return_value=[])
    scheduler_client.resume_schedule = AsyncMock()
    mocker.patch.object(
        scheduling, "get_scheduler_client", return_value=scheduler_client
    )

    await scheduling.reattach_expert_triggers("owner", "expert-1")

    resolve_tenancy.assert_awaited_once_with("owner", "expert-1")
    preset_client.update_many.assert_awaited_once_with(
        where={
            "expertId": "expert-1",
            "userId": "owner",
            "isDeleted": False,
            "deactivatedByExpertArchive": True,
        },
        data={
            "isActive": True,
            "deactivatedByExpertArchive": False,
            "organizationId": "current-personal-org",
            "teamId": "current-personal-team",
        },
    )


@pytest.mark.asyncio
async def test_reattach_fails_before_preset_update_when_expert_is_unavailable(
    mocker,
) -> None:
    resolve_tenancy = mocker.patch(
        "backend.api.features.experts.experts_db.resolve_private_expert_tenancy",
        new=AsyncMock(side_effect=ValueError("expert unavailable")),
    )
    preset_client = mocker.MagicMock()
    preset_client.update_many = AsyncMock()
    mocker.patch.object(
        scheduling.prisma.models.AgentPreset,
        "prisma",
        return_value=preset_client,
    )

    with pytest.raises(ValueError, match="expert unavailable"):
        await scheduling.reattach_expert_triggers("attacker", "victim-expert")

    resolve_tenancy.assert_awaited_once_with("attacker", "victim-expert")
    preset_client.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_pause_only_mutates_private_expert(mocker) -> None:
    expert_client = mocker.MagicMock()
    expert_client.update_many = AsyncMock(return_value=0)
    pause_event_client = mocker.MagicMock()
    pause_event_client.create = AsyncMock()
    mocker.patch.object(prisma.models.Expert, "prisma", return_value=expert_client)
    mocker.patch.object(
        prisma.models.ExpertPauseEvent,
        "prisma",
        return_value=pause_event_client,
    )

    assert not await scheduling.pause_expert_schedules("owner", "shared-expert", "test")

    where = expert_client.update_many.call_args.kwargs["where"]
    assert where["ownerUserId"] == "owner"
    assert where["visibility"] == ResourceVisibility.PRIVATE
    # Archived rows are refused: the rest of the API 404s them, so a pause
    # must not silently mutate one (archive_expert pauses BEFORE archiving).
    assert where["isArchived"] is False
    pause_event_client.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_only_mutates_private_expert(mocker) -> None:
    expert_client = mocker.MagicMock()
    expert_client.update_many = AsyncMock(return_value=0)
    reset_spend = mocker.patch.object(scheduling, "reset_weekly_spend", new=AsyncMock())
    mocker.patch.object(prisma.models.Expert, "prisma", return_value=expert_client)

    assert not await scheduling.resume_expert_schedules("owner", "shared-expert")

    where = expert_client.update_many.call_args.kwargs["where"]
    assert where["ownerUserId"] == "owner"
    assert where["visibility"] == ResourceVisibility.PRIVATE
    # An archived expert must not be resumable: without this filter the
    # resume route would un-pause its schedules and THEN report 404.
    assert where["isArchived"] is False
    reset_spend.assert_not_awaited()


@pytest.mark.asyncio
async def test_budget_gate_fails_closed_for_non_private_expert(mocker) -> None:
    expert_client = mocker.MagicMock()
    expert_client.find_first = AsyncMock(return_value=None)
    spend = mocker.patch.object(scheduling, "get_weekly_spend", new=AsyncMock())
    mocker.patch.object(prisma.models.Expert, "prisma", return_value=expert_client)

    with pytest.raises(ExpertRunPausedError, match="unavailable"):
        await scheduling.enforce_expert_run_budget("owner", "shared-expert")

    where = expert_client.find_first.call_args.kwargs["where"]
    assert where["ownerUserId"] == "owner"
    assert where["visibility"] == ResourceVisibility.PRIVATE
    spend.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_workflow_schedule_fills_credentials_from_the_allow_list(
    mocker,
) -> None:
    """The scheduler validates credential inputs, so an empty map fails every
    graph that needs one; the expert's reachable credentials must be passed."""
    from backend.data.model import CredentialsMetaInput

    meta = CredentialsMetaInput(
        id="cred-notion", provider="notion", type="api_key", title="Notion"
    )
    mocker.patch.object(
        scheduling, "_load_workflow_graph", new=AsyncMock(return_value=_graph([]))
    )
    mocker.patch.object(
        scheduling,
        "_resolve_workflow_credentials",
        new=AsyncMock(return_value={"notion_credentials": meta}),
    )
    scheduler_client = mocker.MagicMock()
    scheduler_client.add_execution_schedule = AsyncMock(
        return_value=SimpleNamespace(id="sched-1")
    )
    mocker.patch.object(
        scheduling, "get_scheduler_client", return_value=scheduler_client
    )
    workflow_client = mocker.MagicMock()
    workflow_client.update_many = AsyncMock(return_value=1)
    mocker.patch.object(
        scheduling.prisma.models.ExpertWorkflow,
        "prisma",
        return_value=workflow_client,
    )

    created = await scheduling.create_workflow_schedule(
        workflow_row_id="wf-1",
        expert_id="expert-1",
        user_id="owner",
        cron="0 9 * * 1",
        graph_id="g1",
        graph_version=1,
        name="SEO Audit",
        user_timezone="UTC",
    )

    assert created is True
    assert scheduler_client.add_execution_schedule.await_args.kwargs[
        "input_credentials"
    ] == {"notion_credentials": meta}


@pytest.mark.asyncio
async def test_pending_schedules_are_retried_per_workflow(mocker) -> None:
    workflow_client = mocker.MagicMock()
    workflow_client.find_many = AsyncMock(
        return_value=[
            SimpleNamespace(
                id="wf-1",
                scheduleCron="0 9 * * 1",
                scheduleId=None,
                LibraryAgent=SimpleNamespace(agentGraphId="g1", agentGraphVersion=2),
                StoreListingVersion=SimpleNamespace(name="SEO Audit"),
            ),
            # No cadence: nothing to create.
            SimpleNamespace(
                id="wf-2",
                scheduleCron=None,
                scheduleId=None,
                LibraryAgent=SimpleNamespace(agentGraphId="g2", agentGraphVersion=1),
                StoreListingVersion=None,
            ),
        ]
    )
    mocker.patch.object(
        scheduling.prisma.models.ExpertWorkflow,
        "prisma",
        return_value=workflow_client,
    )
    mocker.patch.object(
        scheduling,
        "get_user_by_id",
        new=AsyncMock(return_value=SimpleNamespace(timezone="Europe/Madrid")),
    )
    create = mocker.patch.object(
        scheduling, "create_workflow_schedule", new=AsyncMock(return_value=True)
    )

    created = await scheduling.create_pending_workflow_schedules("owner", "expert-1")

    assert created == 1
    # Scoped to the owner, so a caller that skips its own ownership check
    # reaches nothing rather than another user's expert.
    assert workflow_client.find_many.await_args.kwargs["where"] == {
        "expertId": "expert-1",
        "scheduleId": None,
        "Expert": {"is": {"ownerUserId": "owner"}},
    }
    create.assert_awaited_once_with(
        workflow_row_id="wf-1",
        expert_id="expert-1",
        user_id="owner",
        cron="0 9 * * 1",
        graph_id="g1",
        graph_version=2,
        name="SEO Audit",
        user_timezone="Europe/Madrid",
    )


def _graph(required: list[str], titles: dict[str, str] | None = None):
    return SimpleNamespace(
        input_schema={
            "properties": {
                name: {"title": (titles or {}).get(name)} for name in required
            },
            "required": required,
        }
    )


def test_unsatisfied_required_inputs_prefers_the_field_title() -> None:
    graph = _graph(["recipient"], {"recipient": "Email Address"})
    assert scheduling.unsatisfied_required_inputs(graph) == ["Email Address"]


def test_unsatisfied_required_inputs_falls_back_to_the_field_name() -> None:
    assert scheduling.unsatisfied_required_inputs(_graph(["recipient"])) == [
        "recipient"
    ]


def test_a_graph_whose_inputs_all_have_defaults_has_nothing_unsatisfied() -> None:
    assert scheduling.unsatisfied_required_inputs(_graph([])) == []


@pytest.mark.asyncio
async def test_no_schedule_is_created_while_an_input_is_unsatisfied(mocker) -> None:
    scheduler_client = mocker.MagicMock()
    scheduler_client.add_execution_schedule = AsyncMock()
    mocker.patch.object(
        scheduling, "get_scheduler_client", return_value=scheduler_client
    )
    workflow_client = mocker.MagicMock()
    workflow_client.update_many = AsyncMock()
    mocker.patch.object(
        scheduling.prisma.models.ExpertWorkflow, "prisma", return_value=workflow_client
    )
    mocker.patch.object(
        scheduling,
        "_load_workflow_graph",
        new=AsyncMock(return_value=_graph(["recipient"], {"recipient": "Email"})),
    )
    # Mocked so the guard is the only thing that can stop the schedule:
    # the real resolver raises on a stub graph, which would pass this test
    # with the guard removed.
    mocker.patch.object(
        scheduling, "_resolve_workflow_credentials", new=AsyncMock(return_value={})
    )

    created = await scheduling.create_workflow_schedule(
        workflow_row_id="wf-1",
        expert_id="expert-1",
        user_id="owner",
        cron="40 7 * * *",
        graph_id="g1",
        graph_version=1,
        name="Personal Newsletter",
        user_timezone="UTC",
    )

    assert created is False
    scheduler_client.add_execution_schedule.assert_not_awaited()
    workflow_client.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_fully_defaulted_graph_still_gets_its_schedule(mocker) -> None:
    scheduler_client = mocker.MagicMock()
    scheduler_client.add_execution_schedule = AsyncMock(
        return_value=SimpleNamespace(id="sched-1")
    )
    mocker.patch.object(
        scheduling, "get_scheduler_client", return_value=scheduler_client
    )
    workflow_client = mocker.MagicMock()
    workflow_client.update_many = AsyncMock(return_value=1)
    mocker.patch.object(
        scheduling.prisma.models.ExpertWorkflow, "prisma", return_value=workflow_client
    )
    mocker.patch.object(
        scheduling, "_load_workflow_graph", new=AsyncMock(return_value=_graph([]))
    )
    mocker.patch.object(
        scheduling, "_resolve_workflow_credentials", new=AsyncMock(return_value={})
    )

    created = await scheduling.create_workflow_schedule(
        workflow_row_id="wf-1",
        expert_id="expert-1",
        user_id="owner",
        cron="40 7 * * *",
        graph_id="g1",
        graph_version=1,
        name="Lead Finder",
        user_timezone="UTC",
    )

    assert created is True
    scheduler_client.add_execution_schedule.assert_awaited_once()
    workflow_client.update_many.assert_awaited_once_with(
        where={"id": "wf-1", "scheduleId": None}, data={"scheduleId": "sched-1"}
    )
