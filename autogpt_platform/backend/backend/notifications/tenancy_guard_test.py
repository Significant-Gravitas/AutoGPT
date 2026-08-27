import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.db_accessors import LiveResourceAccessRevoked, LiveResourceLeaseGuard


@pytest.mark.asyncio
async def test_inactive_guard_never_starts_the_protected_action() -> None:
    client = MagicMock()
    client.is_live_resource_lease_active = AsyncMock(return_value=False)
    guard = LiveResourceLeaseGuard(client, "lease-1")
    started = False

    async def action() -> None:
        nonlocal started
        started = True

    with pytest.raises(LiveResourceAccessRevoked):
        await guard.run(action())

    assert started is False


@pytest.mark.asyncio
async def test_guard_cancels_action_when_lease_is_lost() -> None:
    client = MagicMock()
    client.is_live_resource_lease_active = AsyncMock(side_effect=[True, False])
    guard = LiveResourceLeaseGuard(client, "lease-1")
    cancelled = asyncio.Event()

    async def action() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with pytest.raises(LiveResourceAccessRevoked):
        await guard.run(action())

    assert cancelled.is_set()
