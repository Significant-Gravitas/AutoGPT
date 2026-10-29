"""Visibility tests for API key listing."""

from unittest.mock import AsyncMock

import pytest
from prisma.models import APIKey as PrismaAPIKey

from backend.data.auth.api_key import list_user_api_keys


@pytest.fixture
def mock_key_client(mocker):
    client = AsyncMock()
    client.find_many.return_value = []
    mocker.patch.object(PrismaAPIKey, "prisma", return_value=client)
    return client


@pytest.mark.asyncio
async def test_list_keys_without_org_is_own_only(mock_key_client):
    await list_user_api_keys("u-1")

    where = mock_key_client.find_many.call_args.kwargs["where"]
    assert where == {"userId": "u-1"}


@pytest.mark.asyncio
async def test_list_keys_org_mode_includes_org_owned_keys(mock_key_client):
    """Own keys (incl. untagged pre-backfill rows) + the org's ORG-owned
    keys — org-wide and those pinned to the caller's teams. Personal keys
    of OTHER members must not appear."""
    await list_user_api_keys("u-1", organization_id="org-1", team_ids=["team-a"])

    where = mock_key_client.find_many.call_args.kwargs["where"]
    assert where == {
        "OR": [
            {
                "userId": "u-1",
                "OR": [{"organizationId": "org-1"}, {"organizationId": None}],
            },
            {
                "organizationId": "org-1",
                "ownerType": "ORG",
                "OR": [{"teamId": None}, {"teamId": {"in": ["team-a"]}}],
            },
        ]
    }
