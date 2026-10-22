"""The owner-email day counter on a real Redis: what the test double assumes."""

import uuid

import pytest

from backend.data import redis_client
from backend.data.redis_client import get_redis_async
from backend.notifications import owner_email
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="needs Redis",
)


async def test_the_day_count_stops_at_the_limit_and_expires():
    counter = f"owner_email:daily:test-{uuid.uuid4()}:2026-10-21"
    redis = await get_redis_async()
    try:
        taken = [await owner_email._take_daily_send(counter, 2) for _ in range(3)]
        count = int(await redis.get(counter))
        ttl = await redis.ttl(counter)
        await owner_email._give_back(counter)
        after_give_back = await owner_email._take_daily_send(counter, 2)
    finally:
        await redis.delete(counter)

    assert taken == [True, True, False]
    assert count == 2
    assert 0 < ttl <= owner_email._COUNTER_TTL_SECONDS
    assert after_give_back
