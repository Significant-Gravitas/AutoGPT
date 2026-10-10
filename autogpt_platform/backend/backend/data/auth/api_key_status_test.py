"""Regression tests for #15296: a REVOKED API key is terminal, so suspend and
permission edits must not touch it."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import APIKey as PrismaAPIKey

from backend.data.auth.api_key import suspend_api_key, update_api_key_permissions


def _row(status: APIKeyStatus):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SimpleNamespace(
        id="key-1",
        name="k",
        head="agpt_abc",
        tail="wxyz",
        status=status,
        permissions=[APIKeyPermission.EXECUTE_GRAPH],
        createdAt=now,
        lastUsedAt=None,
        revokedAt=now if status == APIKeyStatus.REVOKED else None,
        description=None,
        userId="u-1",
        organizationId=None,
        ownerType=None,
        teamId=None,
        teamIdRestriction=None,
    )


@pytest.fixture
def key_client(mocker):
    client = AsyncMock()
    mocker.patch.object(PrismaAPIKey, "prisma", return_value=client)
    return client


@pytest.mark.asyncio
async def test_suspend_revoked_key_is_rejected(key_client):
    key_client.find_unique.return_value = _row(APIKeyStatus.REVOKED)

    with pytest.raises(ValueError, match="revoked"):
        await suspend_api_key("key-1", "u-1")
    key_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_permissions_on_revoked_key_is_rejected(key_client):
    key_client.find_unique.return_value = _row(APIKeyStatus.REVOKED)

    with pytest.raises(ValueError, match="revoked"):
        await update_api_key_permissions("key-1", "u-1", [APIKeyPermission.READ_GRAPH])
    key_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_suspend_active_key_still_works(key_client):
    key_client.find_unique.return_value = _row(APIKeyStatus.ACTIVE)
    key_client.update.return_value = _row(APIKeyStatus.SUSPENDED)

    info = await suspend_api_key("key-1", "u-1")

    assert info.status == APIKeyStatus.SUSPENDED
    key_client.update.assert_awaited_once()
