from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from .db import (
    get_shared_memory_hold_buffer,
    get_shared_memory_org_access,
    list_shared_memory_team_access,
)


@pytest.mark.asyncio
async def test_shared_memory_org_access_returns_prisma_free_role() -> None:
    with patch("backend.api.features.orgs.db.prisma") as mock_prisma:
        mock_prisma.orgmember.find_first = AsyncMock(
            return_value=SimpleNamespace(isAdmin=False, isOwner=True)
        )

        access = await get_shared_memory_org_access("org-1", "user-1")

    assert access is not None
    assert access.is_admin is True
    mock_prisma.orgmember.find_first.assert_awaited_once_with(
        where={
            "orgId": "org-1",
            "userId": "user-1",
            "status": "ACTIVE",
            "Org": {"deletedAt": None},
        }
    )


@pytest.mark.asyncio
async def test_shared_memory_team_access_excludes_missing_relations() -> None:
    with patch("backend.api.features.orgs.db.prisma") as mock_prisma:
        mock_prisma.teammember.find_many = AsyncMock(
            return_value=[
                SimpleNamespace(
                    teamId="team-1",
                    isAdmin=True,
                    Team=SimpleNamespace(name="Platform"),
                ),
                SimpleNamespace(teamId="deleted", isAdmin=False, Team=None),
            ]
        )

        access = await list_shared_memory_team_access("org-1", "user-1")

    assert [item.model_dump() for item in access] == [
        {"team_id": "team-1", "name": "Platform", "is_admin": True}
    ]
    mock_prisma.teammember.find_many.assert_awaited_once_with(
        where={
            "userId": "user-1",
            "status": "ACTIVE",
            "Team": {
                "is": {
                    "orgId": "org-1",
                    "archivedAt": None,
                    "Org": {
                        "is": {
                            "deletedAt": None,
                            "Members": {
                                "some": {"userId": "user-1", "status": "ACTIVE"}
                            },
                        }
                    },
                }
            },
        },
        include={"Team": True},
    )


@pytest.mark.asyncio
async def test_shared_memory_team_access_requires_live_org() -> None:
    with patch("backend.api.features.orgs.db.prisma") as mock_prisma:
        mock_prisma.teammember.find_many = AsyncMock(return_value=[])

        assert await list_shared_memory_team_access("org-1", "user-1") == []

    team_filter = mock_prisma.teammember.find_many.call_args.kwargs["where"]["Team"][
        "is"
    ]
    assert team_filter["Org"]["is"]["deletedAt"] is None


@pytest.mark.asyncio
async def test_shared_memory_team_access_requires_active_org_membership() -> None:
    with patch("backend.api.features.orgs.db.prisma") as mock_prisma:
        mock_prisma.teammember.find_many = AsyncMock(return_value=[])

        assert await list_shared_memory_team_access("org-1", "user-1") == []

    org_filter = mock_prisma.teammember.find_many.call_args.kwargs["where"]["Team"][
        "is"
    ]["Org"]["is"]
    assert org_filter["Members"]["some"] == {
        "userId": "user-1",
        "status": "ACTIVE",
    }


@pytest.mark.asyncio
async def test_shared_memory_hold_buffer_parses_persisted_json() -> None:
    with patch("backend.api.features.orgs.db.prisma") as mock_prisma:
        mock_prisma.organization.find_unique = AsyncMock(
            return_value=SimpleNamespace(
                deletedAt=None,
                settings='{"memory": {"holdBuffer": false}}',
            )
        )

        enabled = await get_shared_memory_hold_buffer("org-1")

    assert enabled is False


@pytest.mark.asyncio
async def test_shared_memory_hold_buffer_fails_closed_for_missing_org() -> None:
    with patch("backend.api.features.orgs.db.prisma") as mock_prisma:
        mock_prisma.organization.find_unique = AsyncMock(return_value=None)

        enabled = await get_shared_memory_hold_buffer("org-1")

    assert enabled is True
