from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.library import db, model


@pytest.mark.asyncio
async def test_library_detail_loads_its_own_installation_without_store_version_lookup(
    mocker,
):
    row = MagicMock(agentGraphId="graph", agentGraphVersion=2)
    client = MagicMock(find_first=AsyncMock(return_value=row))
    mocker.patch("prisma.models.LibraryAgent.prisma", return_value=client)
    store_client = MagicMock(find_many=AsyncMock(return_value=[]))
    mocker.patch("prisma.models.StoreListingVersion.prisma", return_value=store_client)
    mocker.patch.object(
        db, "_fetch_marketplace_details", AsyncMock(return_value=(None, None))
    )
    mocker.patch.object(db, "_fetch_schedule_info", AsyncMock(return_value={}))
    mocker.patch.object(db.graph_db, "get_sub_graphs", AsyncMock(return_value=[]))
    expected = MagicMock()
    mocker.patch.object(model.LibraryAgent, "from_db", return_value=expected)

    result = await db._get_library_agent_locked(
        "installation", "caller", organization_id="org", team_id_restriction="team"
    )

    assert result is expected
    store_client.find_many.assert_not_awaited()
    assert client.find_first.await_args.kwargs["where"] == {
        "id": "installation",
        "userId": "caller",
        "isDeleted": False,
        "organizationId": "org",
        "teamId": "team",
    }
