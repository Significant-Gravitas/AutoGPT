"""Regression tests for #15296: a REVOKED API key is terminal, so suspend and
permission edits must not touch it."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import APIKey as PrismaAPIKey

from backend.data.auth.api_key import (
    revoke_api_key,
    suspend_api_key,
    update_api_key_permissions,
)
from backend.util.exceptions import NotFoundError


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


def _write_where(key_client) -> dict:
    return key_client.update_many.await_args.kwargs["where"]


@pytest.mark.asyncio
async def test_suspend_revoked_key_is_rejected(key_client):
    key_client.find_unique.return_value = _row(APIKeyStatus.REVOKED)
    key_client.update_many.return_value = 0

    with pytest.raises(ValueError, match="revoked"):
        await suspend_api_key("key-1", "u-1")
    key_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_permissions_on_revoked_key_is_rejected(key_client):
    key_client.find_unique.return_value = _row(APIKeyStatus.REVOKED)
    key_client.update_many.return_value = 0

    with pytest.raises(ValueError, match="revoked"):
        await update_api_key_permissions("key-1", "u-1", [APIKeyPermission.READ_GRAPH])
    key_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_writes_are_conditional_on_not_revoked(key_client):
    """The REVOKED check lives in the UPDATE's WHERE clause, scoped to the
    owner and org, so a revoke committed after the read can't be overwritten."""
    key_client.find_unique.side_effect = [
        _row(APIKeyStatus.ACTIVE),
        _row(APIKeyStatus.SUSPENDED),
    ]
    key_client.update_many.return_value = 1

    info = await suspend_api_key("key-1", "u-1", organization_id=None)

    assert info.status == APIKeyStatus.SUSPENDED
    assert _write_where(key_client) == {
        "id": "key-1",
        "userId": "u-1",
        "status": {"not": APIKeyStatus.REVOKED},
    }
    key_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_conditional_write_includes_org_scope(key_client):
    row = _row(APIKeyStatus.ACTIVE)
    row.organizationId = "org-1"
    key_client.find_unique.return_value = row
    key_client.update_many.return_value = 1

    await update_api_key_permissions(
        "key-1", "u-1", [APIKeyPermission.READ_GRAPH], organization_id="org-1"
    )

    assert _write_where(key_client)["organizationId"] == "org-1"


@pytest.mark.asyncio
async def test_revoke_lost_race_to_concurrent_revoke_keeps_original_timestamp(
    key_client,
):
    """Re-revoking is idempotent and must not move revokedAt."""
    revoked = _row(APIKeyStatus.REVOKED)
    key_client.find_unique.side_effect = [_row(APIKeyStatus.ACTIVE), revoked]
    key_client.update_many.return_value = 0

    info = await revoke_api_key("key-1", "u-1")

    assert info.status == APIKeyStatus.REVOKED
    assert info.revoked_at == revoked.revokedAt


@pytest.mark.asyncio
async def test_key_deleted_mid_write_is_not_found(key_client):
    key_client.find_unique.side_effect = [_row(APIKeyStatus.ACTIVE), None]
    key_client.update_many.return_value = 0

    with pytest.raises(NotFoundError):
        await suspend_api_key("key-1", "u-1")
