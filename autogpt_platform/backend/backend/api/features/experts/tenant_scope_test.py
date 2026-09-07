from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import ResourceVisibility

from backend.api.features.experts import credentials, experts_db
from backend.util.exceptions import ExpertNotFoundError


@pytest.mark.asyncio
async def test_workspace_roster_keeps_private_experts_owned_by_the_requester(mocker):
    lookup = AsyncMock(return_value=[])
    mocker.patch.object(
        experts_db.prisma.models.Expert,
        "prisma",
        return_value=MagicMock(find_many=lookup),
    )

    await experts_db.list_experts(
        "actor",
        organization_id="org-1",
        team_id_restriction="team-1",
        team_ids=["team-1"],
        with_metrics=False,
    )

    where = lookup.await_args.kwargs["where"]
    assert where["ownerUserId"] == "actor"
    assert where["visibility"] == ResourceVisibility.PRIVATE


@pytest.mark.asyncio
async def test_credential_management_rejects_nonprivate_experts(mocker):
    async def find_private_expert(*, where, include):
        if where.get("visibility") == ResourceVisibility.PRIVATE:
            return None
        return MagicMock(id="expert-shared")

    mocker.patch.object(
        credentials.prisma.models.Expert,
        "prisma",
        return_value=MagicMock(find_first=AsyncMock(side_effect=find_private_expert)),
    )

    with pytest.raises(ExpertNotFoundError):
        await credentials._owned_expert("actor", "expert-shared")
