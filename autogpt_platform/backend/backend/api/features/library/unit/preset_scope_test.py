from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.library import db


@pytest.mark.asyncio
async def test_list_presets_is_owner_and_exact_team_scoped() -> None:
    client = MagicMock(
        find_many=AsyncMock(return_value=[]),
        count=AsyncMock(return_value=0),
    )
    with patch("prisma.models.AgentPreset.prisma", return_value=client):
        await db.list_presets(
            user_id="user-1",
            page=1,
            page_size=10,
            organization_id="org-1",
            team_id="team-1",
            enforce_team_scope=True,
        )

    assert client.find_many.await_args.kwargs["where"] == {
        "userId": "user-1",
        "isDeleted": False,
        "organizationId": "org-1",
        "teamId": "team-1",
    }


@pytest.mark.asyncio
async def test_get_preset_is_owner_and_org_home_scoped() -> None:
    row = SimpleNamespace(isDeleted=False)
    client = MagicMock(find_first=AsyncMock(return_value=row))
    response = MagicMock()
    with (
        patch("prisma.models.AgentPreset.prisma", return_value=client),
        patch.object(
            db.library_model.LibraryAgentPreset, "from_db", return_value=response
        ),
    ):
        result = await db.get_preset("user-1", "preset-1", "org-1", None, True)

    assert result is response
    assert client.find_first.await_args.kwargs["where"] == {
        "id": "preset-1",
        "userId": "user-1",
        "organizationId": "org-1",
        "teamId": None,
    }


@pytest.mark.asyncio
async def test_update_preset_reasserts_loaded_tenancy_in_mutation() -> None:
    current = MagicMock(
        name="Preset",
        expert_id=None,
        organization_id="org-1",
        team_id="team-1",
    )
    client = MagicMock(update=AsyncMock(return_value=MagicMock()))

    @asynccontextmanager
    async def fake_transaction():
        yield MagicMock()

    with (
        patch.object(db, "get_preset", new=AsyncMock(return_value=current)),
        patch.object(db, "transaction", fake_transaction),
        patch("prisma.models.AgentPreset.prisma", return_value=client),
        patch.object(
            db.library_model.LibraryAgentPreset,
            "from_db",
            return_value=MagicMock(),
        ),
    ):
        await db.update_preset(
            "user-1",
            "preset-1",
            name="Updated",
            organization_id="org-1",
            team_id="team-1",
            enforce_team_scope=True,
        )

    assert client.update.await_args.kwargs["where"] == {
        "id": "preset-1",
        "userId": "user-1",
        "organizationId": "org-1",
        "teamId": "team-1",
        "isDeleted": False,
    }


@pytest.mark.asyncio
async def test_delete_preset_reasserts_loaded_tenancy_in_mutation() -> None:
    current = MagicMock(organization_id="org-1", team_id="team-1")
    client = MagicMock(update_many=AsyncMock(return_value=1))
    with (
        patch.object(db, "get_preset", new=AsyncMock(return_value=current)),
        patch("prisma.models.AgentPreset.prisma", return_value=client),
    ):
        await db.delete_preset(
            "user-1",
            "preset-1",
            "org-1",
            "team-1",
            True,
        )

    assert client.update_many.await_args.kwargs["where"] == {
        "id": "preset-1",
        "userId": "user-1",
        "organizationId": "org-1",
        "teamId": "team-1",
        "isDeleted": False,
    }
