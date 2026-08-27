import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import APIKeyPermission

from backend.data.auth import oauth


def _transaction_factory():
    @asynccontextmanager
    async def transaction():
        yield MagicMock()

    return transaction


def _racing_delegate(record):
    delegate = MagicMock()
    arrivals = 0
    ready = asyncio.Event()
    consumed = False
    lock = asyncio.Lock()

    async def find_unique(*args, **kwargs):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            ready.set()
        await ready.wait()
        return record

    async def update_many(*args, **kwargs):
        nonlocal consumed
        async with lock:
            if consumed:
                return 0
            consumed = True
            return 1

    delegate.find_unique = AsyncMock(side_effect=find_unique)
    delegate.update_many = AsyncMock(side_effect=update_many)
    return delegate


@pytest.mark.asyncio
async def test_authorization_code_concurrent_exchange_has_one_winner(mocker):
    code = MagicMock(
        applicationId="app-1",
        userId="user-1",
        scopes=[APIKeyPermission.READ_GRAPH],
        redirectUri="https://client.example/callback",
        usedAt=None,
        expiresAt=datetime.now(timezone.utc) + timedelta(minutes=5),
        codeChallenge=None,
    )
    delegate = _racing_delegate(code)
    model = mocker.patch.object(oauth, "PrismaOAuthAuthorizationCode")
    model.prisma.return_value = delegate
    mocker.patch.object(oauth, "transaction", _transaction_factory())

    results = await asyncio.gather(
        *(
            oauth.consume_authorization_code(
                "code-1", "app-1", "https://client.example/callback"
            )
            for _ in range(2)
        ),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, oauth.InvalidGrantError) for result in results) == 1


@pytest.mark.asyncio
async def test_refresh_token_concurrent_rotation_mints_one_family(mocker):
    token = MagicMock(
        applicationId="app-1",
        userId="user-1",
        scopes=[APIKeyPermission.READ_GRAPH],
        revokedAt=None,
        expiresAt=datetime.now(timezone.utc) + timedelta(days=1),
    )
    delegate = _racing_delegate(token)
    model = mocker.patch.object(oauth, "PrismaOAuthRefreshToken")
    model.prisma.return_value = delegate
    mocker.patch.object(oauth, "transaction", _transaction_factory())
    create_access = mocker.patch.object(
        oauth, "create_access_token", AsyncMock(return_value=MagicMock())
    )
    create_refresh = mocker.patch.object(
        oauth, "create_refresh_token", AsyncMock(return_value=MagicMock())
    )

    results = await asyncio.gather(
        *(oauth.refresh_tokens("refresh-1", "app-1") for _ in range(2)),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, oauth.InvalidGrantError) for result in results) == 1
    create_access.assert_awaited_once()
    create_refresh.assert_awaited_once()
