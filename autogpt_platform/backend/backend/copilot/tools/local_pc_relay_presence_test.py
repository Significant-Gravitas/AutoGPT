from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.tools.local_pc_relay import RedisShimRelay
from backend.copilot.tools.local_pc_relay_presence import (
    MAX_DISCOVERY_PRESENCES,
    clear_presence,
    owner_presences,
    register_presence,
)
from backend.copilot.tools.local_pc_relay_protocol import owner_presence_index_key
from backend.copilot.tools.local_pc_relay_test_support import FakeRedis

pytestmark = pytest.mark.asyncio


class _NoScanRedis(FakeRedis):
    async def scan_iter(self, *, match: str):
        raise AssertionError(f"owner lookup must not scan Redis: {match}")
        yield ""


async def _collect_owner_presences(
    redis: FakeRedis,
    user_id: str,
    client_id: str | None,
    *,
    connection_kind: str | None = None,
    limit: int | None = None,
):
    return [
        presence
        async for presence in owner_presences(
            redis,
            user_id,
            client_id,
            connection_kind=connection_kind,
            limit=limit,
        )
    ]


async def test_owner_presences_uses_scoped_index() -> None:
    redis = _NoScanRedis()
    first, _ = await register_presence(
        redis,
        "session-1",
        hello={},
        user_id="user-1",
        client_id="client-1",
    )
    await register_presence(
        redis,
        "session-2",
        hello={},
        user_id="user-1",
        client_id="client-2",
    )
    await register_presence(
        redis,
        "session-3",
        hello={},
        user_id="user-2",
        client_id="client-1",
    )

    client_presences = await _collect_owner_presences(redis, "user-1", "client-1")
    user_presences = await _collect_owner_presences(redis, "user-1", None)

    assert client_presences == [first]
    assert {presence.session_id for presence in user_presences} == {
        "session-1",
        "session-2",
    }


async def test_discovery_is_bounded_without_truncating_revocation_index() -> None:
    redis = FakeRedis()
    total = MAX_DISCOVERY_PRESENCES + 5
    for number in range(total):
        await register_presence(
            redis,
            f"session-{number}",
            hello={"connection_kind": "machine"},
            user_id="user-1",
            client_id="client-1",
        )

    index_key = owner_presence_index_key("user-1", "client-1")
    indexed = redis.sorted_sets[index_key]
    all_presences = await _collect_owner_presences(redis, "user-1", "client-1")
    discovery_presences = await _collect_owner_presences(
        redis,
        "user-1",
        "client-1",
        connection_kind="machine",
        limit=MAX_DISCOVERY_PRESENCES,
    )

    assert len(indexed) == total
    assert len(all_presences) == total
    assert len(discovery_presences) == MAX_DISCOVERY_PRESENCES
    assert {presence.session_id for presence in discovery_presences} == {
        f"session-{number}" for number in range(total - MAX_DISCOVERY_PRESENCES, total)
    }

    relay = RedisShimRelay(redis)
    with patch.object(relay, "_publish_control", AsyncMock()) as publish:
        assert (
            await relay.revoke_owner("user-1", "client-1", reason="user_revoked")
            == total
        )
    assert publish.await_count == total


async def test_clear_presence_removes_owner_indexes() -> None:
    redis = FakeRedis()
    presence, _ = await register_presence(
        redis,
        "session-1",
        hello={},
        user_id="user-1",
        client_id="client-1",
    )

    await clear_presence(redis, presence)

    assert await _collect_owner_presences(redis, "user-1", None) == []
    assert await _collect_owner_presences(redis, "user-1", "client-1") == []


async def test_replacement_survives_old_connection_clear_race() -> None:
    class _ReplaceDuringClearRedis(FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self.after_clear: Callable[[], Awaitable[None]] | None = None

        async def eval(
            self, script: str, number_of_keys: int, *keys_and_args: str
        ) -> int:
            result = await super().eval(script, number_of_keys, *keys_and_args)
            if "DEL" in script and result == 1 and self.after_clear is not None:
                after_clear, self.after_clear = self.after_clear, None
                await after_clear()
            return result

    redis = _ReplaceDuringClearRedis()
    old_presence, _ = await register_presence(
        redis,
        "session-1",
        hello={"connection_kind": "machine"},
        user_id="user-1",
        client_id="client-1",
        connection_id="old-connection",
    )

    async def register_replacement() -> None:
        await register_presence(
            redis,
            "session-1",
            hello={"connection_kind": "machine"},
            user_id="user-1",
            client_id="client-1",
            connection_id="new-connection",
        )

    redis.after_clear = register_replacement
    await clear_presence(redis, old_presence)

    presences = await _collect_owner_presences(
        redis,
        "user-1",
        "client-1",
        connection_kind="machine",
    )
    assert [presence.connection_id for presence in presences] == ["new-connection"]
