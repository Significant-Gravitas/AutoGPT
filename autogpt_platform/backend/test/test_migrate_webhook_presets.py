"""
Unit tests for migrate_webhook_presets_to_new_version.
Mocks prisma to avoid needing a running database.
"""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_prisma():
    with patch("prisma.models.AgentPreset.prisma") as mock:
        mock_client = AsyncMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.mark.asyncio
async def test_migrate_updates_matching_presets(mock_prisma):
    from backend.api.features.library.db import migrate_webhook_presets_to_new_version

    mock_prisma.update_many = AsyncMock(return_value=3)

    count = await migrate_webhook_presets_to_new_version(
        user_id="user-123", graph_id="graph-abc", new_version=5
    )

    assert count == 3
    mock_prisma.update_many.assert_called_once_with(
        where={
            "userId": "user-123",
            "agentGraphId": "graph-abc",
            "agentGraphVersion": {"not": 5},
            "webhookId": {"not": None},
            "isDeleted": False,
        },
        data={"agentGraphVersion": 5},
    )


@pytest.mark.asyncio
async def test_migrate_returns_zero_when_no_matches(mock_prisma):
    from backend.api.features.library.db import migrate_webhook_presets_to_new_version

    mock_prisma.update_many = AsyncMock(return_value=0)

    count = await migrate_webhook_presets_to_new_version(
        user_id="user-123", graph_id="graph-abc", new_version=1
    )

    assert count == 0


@pytest.mark.asyncio
async def test_migrate_filters_correctly(mock_prisma):
    from backend.api.features.library.db import migrate_webhook_presets_to_new_version

    mock_prisma.update_many = AsyncMock(return_value=1)

    await migrate_webhook_presets_to_new_version(
        user_id="user-456", graph_id="graph-xyz", new_version=10
    )

    where = mock_prisma.update_many.call_args.kwargs["where"]
    assert where["webhookId"] == {"not": None}
    assert where["isDeleted"] is False
    assert where["userId"] == "user-456"
    assert where["agentGraphId"] == "graph-xyz"
    assert where["agentGraphVersion"] == {"not": 10}
