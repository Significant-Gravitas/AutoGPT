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
