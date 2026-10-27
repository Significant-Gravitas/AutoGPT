"""A validated API key records when it was used, without a write per request."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import APIKey as PrismaAPIKey

from backend.data.auth.api_key import LAST_USED_RESOLUTION, keysmith, validate_api_key

_KEY = keysmith.generate_key()


@pytest.fixture
def key_client(mocker):
    client = AsyncMock()
    mocker.patch.object(PrismaAPIKey, "prisma", return_value=client)
    return client


@pytest.mark.asyncio
async def test_first_use_is_recorded(key_client):
    key_client.find_many.return_value = [_row(last_used_at=None)]

    info = await validate_api_key(_KEY.key)

    assert info is not None
    key_client.update.assert_awaited_once()
    call = key_client.update.await_args.kwargs
    assert call["where"] == {"id": "key-1"}
    assert _is_recent(call["data"]["lastUsedAt"])
    assert info.last_used_at == call["data"]["lastUsedAt"]


@pytest.mark.asyncio
async def test_use_within_the_resolution_is_not_written_again(key_client):
    recent = datetime.now(timezone.utc) - LAST_USED_RESOLUTION / 2
    key_client.find_many.return_value = [_row(last_used_at=recent)]

    info = await validate_api_key(_KEY.key)

    assert info is not None
    key_client.update.assert_not_awaited()
    assert info.last_used_at == recent


@pytest.mark.asyncio
async def test_use_after_the_resolution_is_recorded(key_client):
    stale = datetime.now(timezone.utc) - LAST_USED_RESOLUTION - timedelta(seconds=1)
    key_client.find_many.return_value = [_row(last_used_at=stale)]

    info = await validate_api_key(_KEY.key)

    assert info is not None
    key_client.update.assert_awaited_once()
    assert _is_recent(key_client.update.await_args.kwargs["data"]["lastUsedAt"])


@pytest.mark.asyncio
async def test_a_failed_write_still_accepts_the_key(key_client):
    key_client.find_many.return_value = [_row(last_used_at=None)]
    key_client.update.side_effect = RuntimeError("db unavailable")

    info = await validate_api_key(_KEY.key)

    assert info is not None
    assert info.id == "key-1"
    assert info.last_used_at is None


@pytest.mark.asyncio
async def test_a_wrong_key_records_nothing(key_client):
    key_client.find_many.return_value = [_row(last_used_at=None)]
    wrong = _KEY.key[:-4] + ("aaaa" if not _KEY.key.endswith("aaaa") else "bbbb")

    assert await validate_api_key(wrong) is None
    key_client.update.assert_not_awaited()


def _row(last_used_at: datetime | None) -> MagicMock:
    row = MagicMock(spec=PrismaAPIKey)
    row.id = "key-1"
    row.name = "ci"
    row.head = _KEY.head
    row.tail = _KEY.tail
    row.hash = _KEY.hash
    row.salt = _KEY.salt
    row.status = APIKeyStatus.ACTIVE
    row.permissions = [APIKeyPermission.IDENTITY]
    row.createdAt = datetime.now(timezone.utc) - timedelta(days=30)
    row.lastUsedAt = last_used_at
    row.revokedAt = None
    row.description = None
    row.userId = "user-1"
    row.organizationId = None
    row.ownerType = None
    row.teamIdRestriction = None
    return row


def _is_recent(value: datetime) -> bool:
    return datetime.now(timezone.utc) - value < timedelta(seconds=30)
