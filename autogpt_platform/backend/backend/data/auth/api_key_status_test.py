"""Status-transition regressions for API keys (issue #15296)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import APIKey as PrismaAPIKey

from backend.data.auth.api_key import (
    revoke_api_key,
    suspend_api_key,
    update_api_key_permissions,
)
from backend.util.exceptions import ConflictError

KEY_ID = "key-1"
USER_ID = "user-1"
REVOKED_AT = datetime(2025, 6, 2, tzinfo=UTC)


def _key_row(*, status: APIKeyStatus, revoked_at=None):
    row = MagicMock()
    row.id = KEY_ID
    row.userId = USER_ID
    row.name = "test-key"
    row.head = "agpt_head"
    row.tail = "tail"
    row.status = status
    row.permissions = [APIKeyPermission.IDENTITY]
    row.createdAt = datetime(2025, 6, 1, tzinfo=UTC)
    row.lastUsedAt = None
    row.revokedAt = revoked_at
    row.description = None
    row.organizationId = None
    row.ownerType = None
    row.teamIdRestriction = None
    return row


def _updated(count: int) -> MagicMock:
    result = MagicMock()
    result.count = count
    return result


@pytest.fixture
def mock_key_client(mocker):
    client = AsyncMock()
    mocker.patch.object(PrismaAPIKey, "prisma", return_value=client)
    return client


@pytest.mark.asyncio
async def test_suspend_revoked_key_is_rejected_and_state_is_unchanged(
    mock_key_client,
):
    revoked = _key_row(status=APIKeyStatus.REVOKED, revoked_at=REVOKED_AT)
    mock_key_client.find_unique.return_value = revoked
    mock_key_client.update_many.return_value = _updated(0)

    with pytest.raises(ConflictError):
        await suspend_api_key(KEY_ID, USER_ID)

    # ACTIVE in the predicate is what prevents the REVOKED row being overwritten.
    where = mock_key_client.update_many.call_args.kwargs["where"]
    assert where == {"id": KEY_ID, "status": APIKeyStatus.ACTIVE}
    assert revoked.status == APIKeyStatus.REVOKED
    assert revoked.revokedAt == REVOKED_AT


@pytest.mark.asyncio
async def test_suspend_active_key_still_works(mock_key_client):
    active = _key_row(status=APIKeyStatus.ACTIVE)
    suspended = _key_row(status=APIKeyStatus.SUSPENDED)
    mock_key_client.find_unique.side_effect = [active, suspended]
    mock_key_client.update_many.return_value = _updated(1)

    info = await suspend_api_key(KEY_ID, USER_ID)

    assert info.status == APIKeyStatus.SUSPENDED
    where = mock_key_client.update_many.call_args.kwargs["where"]
    assert where == {"id": KEY_ID, "status": APIKeyStatus.ACTIVE}


@pytest.mark.asyncio
async def test_revoke_revoked_key_keeps_original_revoked_at(mock_key_client):
    revoked = _key_row(status=APIKeyStatus.REVOKED, revoked_at=REVOKED_AT)
    mock_key_client.find_unique.return_value = revoked

    info = await revoke_api_key(KEY_ID, USER_ID)

    assert info.status == APIKeyStatus.REVOKED
    assert info.revoked_at == REVOKED_AT
    mock_key_client.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoke_active_key_still_works(mock_key_client):
    active = _key_row(status=APIKeyStatus.ACTIVE)
    revoked = _key_row(status=APIKeyStatus.REVOKED, revoked_at=REVOKED_AT)
    mock_key_client.find_unique.side_effect = [active, revoked]
    mock_key_client.update_many.return_value = _updated(1)

    info = await revoke_api_key(KEY_ID, USER_ID)

    assert info.status == APIKeyStatus.REVOKED
    assert info.revoked_at == REVOKED_AT
    where = mock_key_client.update_many.call_args.kwargs["where"]
    assert where == {"id": KEY_ID, "status": {"not_in": [APIKeyStatus.REVOKED]}}


@pytest.mark.asyncio
async def test_update_permissions_on_suspended_key_still_works(mock_key_client):
    suspended = _key_row(status=APIKeyStatus.SUSPENDED)
    updated = _key_row(status=APIKeyStatus.SUSPENDED)
    updated.permissions = [APIKeyPermission.READ_GRAPH]
    mock_key_client.find_unique.side_effect = [suspended, updated]
    mock_key_client.update_many.return_value = _updated(1)

    info = await update_api_key_permissions(
        KEY_ID, USER_ID, [APIKeyPermission.READ_GRAPH]
    )

    assert info.scopes == [APIKeyPermission.READ_GRAPH]
    where = mock_key_client.update_many.call_args.kwargs["where"]
    assert where == {"id": KEY_ID, "status": {"not_in": [APIKeyStatus.REVOKED]}}


@pytest.mark.asyncio
async def test_update_permissions_on_revoked_key_is_rejected(mock_key_client):
    revoked = _key_row(status=APIKeyStatus.REVOKED, revoked_at=REVOKED_AT)
    mock_key_client.find_unique.return_value = revoked

    with pytest.raises(ConflictError):
        await update_api_key_permissions(KEY_ID, USER_ID, [APIKeyPermission.READ_GRAPH])

    assert revoked.permissions == [APIKeyPermission.IDENTITY]
    mock_key_client.update_many.assert_not_awaited()
