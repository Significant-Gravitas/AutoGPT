"""The request quota: counted per box and per user, failing closed."""

import pytest

from swap_proxy.quota import RequestQuota


class Counter:
    def __init__(self, fail: bool = False):
        self.counts: dict[str, int] = {}
        self.ttls: dict[str, int] = {}
        self.fail = fail

    async def incr(self, name):
        if self.fail:
            raise ConnectionError("redis down")
        self.counts[name] = self.counts.get(name, 0) + 1
        return self.counts[name]

    async def expire(self, name, time):
        self.ttls[name] = time


NOW = 7200.0 + 600  # 600 s into a window of an hour


def quota(redis, per_box=2, per_user=3, window=3600) -> RequestQuota:
    return RequestQuota(redis, per_box=per_box, per_user=per_user, window=window)


async def test_a_box_is_refused_past_its_limit_and_told_when_it_resets():
    q = quota(Counter())
    assert (await q.take("session:s-1", "user-a", now=NOW)).allowed
    assert (await q.take("session:s-1", "user-a", now=NOW)).allowed
    verdict = await q.take("session:s-1", "user-a", now=NOW)
    assert not verdict.allowed
    assert (verdict.reason, verdict.scope, verdict.limit) == (
        "quota-exceeded",
        "box",
        2,
    )
    assert verdict.retry_after == 3000
    assert "resets in 3000 s" in verdict.message()
    assert "Do not retry in a loop" in verdict.message()


async def test_a_user_is_limited_across_their_boxes():
    q = quota(Counter())
    for box in ("session:s-1", "session:s-2", "expert:e-1"):
        assert (await q.take(box, "user-a", now=NOW)).allowed
    verdict = await q.take("session:s-3", "user-a", now=NOW)
    assert (verdict.reason, verdict.scope) == ("quota-exceeded", "user")
    # Another user's boxes are not touched.
    assert (await q.take("session:s-9", "user-b", now=NOW)).allowed


async def test_the_window_is_fixed_and_the_count_expires_with_it():
    redis = Counter()
    q = quota(redis, per_box=1)
    assert (await q.take("session:s-1", "user-a", now=NOW)).allowed
    assert not (await q.take("session:s-1", "user-a", now=NOW)).allowed
    assert (await q.take("session:s-1", "user-a", now=NOW + 3600)).allowed
    assert set(redis.ttls.values()) == {3600}


async def test_a_count_that_cannot_be_taken_refuses():
    verdict = await quota(Counter(fail=True)).take("session:s-1", "user-a", now=NOW)
    assert verdict.reason == "quota-unavailable"
    assert "no credential was attached" in verdict.message()


@pytest.mark.parametrize("per_box, per_user", [(0, 0)])
async def test_zero_turns_a_limit_off(per_box, per_user):
    redis = Counter(fail=True)  # never asked
    q = quota(redis, per_box=per_box, per_user=per_user)
    assert (await q.take("session:s-1", "user-a", now=NOW)).allowed
