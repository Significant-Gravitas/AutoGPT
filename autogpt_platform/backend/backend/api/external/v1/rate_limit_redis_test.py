import asyncio

import pytest

from backend.api.external import rate_limit
from backend.data import redis_client
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="needs Redis",
)


async def test_concurrent_routes_share_an_atomic_budget(
    v1_client, v1_key_rows, v1_clock, monkeypatch
):
    # A loaded runner can take longer than the production 0.25 s per Redis call,
    # and a timed-out check fails open, which this atomicity test would miscount.
    monkeypatch.setattr(rate_limit, "RATE_LIMIT_REDIS_TIMEOUT_SECONDS", 5)
    redis = await redis_client.get_redis_async()
    window = rate_limit._window_start(v1_clock.now.return_value, 60)
    key = f"external_api:rl:{v1_key_rows[0].user_id}:read:{window}"
    gate = asyncio.Barrier(20)

    async def request(index):
        await gate.wait()
        return await v1_client.get(
            "/v1/me", headers={"Authorization": f"Bearer agpt_xt_test-{index % 2}"}
        )

    try:
        await redis.set(key, 290, ex=60)
        responses = await asyncio.wait_for(
            asyncio.gather(*(request(i) for i in range(20))), timeout=10
        )
        assert sum(r.status_code == 200 for r in responses) == 10
        assert sum(r.status_code == 429 for r in responses) == 10
        assert sorted(
            int(r.headers["x-ratelimit-remaining"])
            for r in responses
            if r.status_code == 200
        ) == list(range(10))
        assert int(await redis.get(key)) == 310
        assert 0 < await redis.ttl(key) <= 60
    finally:
        await redis.delete(key)
