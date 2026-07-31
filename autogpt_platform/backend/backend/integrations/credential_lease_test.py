import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.integrations.credential_lease import CredentialLease
from backend.integrations.creds_manager_test import _provider_runtime_credentials


@pytest.mark.asyncio
async def test_checkpoint_uses_existing_lock_without_reacquiring():
    credentials = _provider_runtime_credentials()
    updated = credentials.model_copy(update={"title": "refreshed"})
    lock = AsyncMock()
    lock.locked.return_value = True
    lock.owned.return_value = True
    lock.timeout = 60
    checkpoint = AsyncMock()
    lease = CredentialLease(credentials, lock, checkpoint)

    await lease.checkpoint(updated)

    checkpoint.assert_awaited_once_with(updated, lock)
    assert lease.credentials is updated


@pytest.mark.asyncio
async def test_checkpoint_fails_closed_after_lock_ownership_is_lost():
    credentials = _provider_runtime_credentials()
    lock = AsyncMock()
    lock.locked.return_value = True
    lock.owned.return_value = False
    checkpoint = AsyncMock()
    lease = CredentialLease(credentials, lock, checkpoint)

    with pytest.raises(RuntimeError, match="lease"):
        await lease.checkpoint(credentials)

    checkpoint.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_stops_heartbeat_before_unlocking():
    credentials = _provider_runtime_credentials()
    lock = AsyncMock()
    lock.locked.return_value = True
    lock.owned.return_value = True
    lock.timeout = 0.03
    lock.extend.return_value = True
    lease = CredentialLease(credentials, lock, AsyncMock())

    lease.start_heartbeat()
    await asyncio.sleep(0.02)
    await lease.release()

    lock.extend.assert_awaited()
    lock.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_for_failure_reports_heartbeat_loss_immediately():
    credentials = _provider_runtime_credentials()
    lock = AsyncMock()
    lock.locked.return_value = True
    lock.owned.return_value = True
    lock.timeout = 0.03
    lock.extend.return_value = False
    lease = CredentialLease(credentials, lock, AsyncMock())

    lease.start_heartbeat()

    with pytest.raises(RuntimeError, match="heartbeat"):
        await asyncio.wait_for(lease.wait_for_failure(), timeout=1)
    with pytest.raises(RuntimeError, match="heartbeat"):
        await lease.validate()

    await lease.release()


@pytest.mark.asyncio
async def test_delete_uses_owned_lock_and_prevents_later_checkpoint():
    credentials = _provider_runtime_credentials()
    lock = AsyncMock()
    lock.locked.return_value = True
    lock.owned.return_value = True
    checkpoint = AsyncMock()
    delete = AsyncMock()
    lease = CredentialLease(credentials, lock, checkpoint, delete)

    await lease.delete()
    await lease.delete()

    delete.assert_awaited_once_with(credentials, lock)
    with pytest.raises(RuntimeError, match="deleted"):
        await lease.checkpoint(credentials)
    checkpoint.assert_not_awaited()
