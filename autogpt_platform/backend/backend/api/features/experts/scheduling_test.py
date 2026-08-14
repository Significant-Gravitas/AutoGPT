from unittest.mock import AsyncMock

import pytest

from backend.api.features.experts import scheduling


@pytest.mark.asyncio
async def test_reattach_rehomes_presets_to_current_personal_tenancy(mocker) -> None:
    resolve_tenancy = mocker.patch(
        "backend.api.features.experts.experts_db.resolve_expert_personal_tenancy",
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
        "backend.api.features.experts.experts_db.resolve_expert_personal_tenancy",
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
