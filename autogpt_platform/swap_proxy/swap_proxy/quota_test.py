"""The request quota: counted per box and per user, failing closed."""

import fakeredis
import pytest

from swap_proxy.quota import RequestQuota


class Counter:
    """A Redis that runs the real ``_TAKE_SCRIPT`` (fakeredis evaluates Lua),
    recording the scripts it is sent; *fail* makes every call fail."""

    def __init__(self, fail: bool = False):
        self.redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        self.fail = fail
        self.scripts: list[str] = []

    async def eval(self, script, numkeys, *keys_and_args):
        if self.fail:
            raise ConnectionError("redis down")
        self.scripts.append(script)
        return await self.redis.eval(script, numkeys, *keys_and_args)

    async def counts(self) -> dict[str, int]:
        keys = await self.redis.keys("*")
        return {str(k): int(str(await self.redis.get(k))) for k in keys}

    async def ttls(self) -> dict[str, int]:
        keys = await self.redis.keys("*")
        return {str(k): await self.redis.ttl(k) for k in keys}


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
    # Set by the script on first increment, and only then.
    assert set((await redis.ttls()).values()) == {3600}


async def test_a_count_that_cannot_be_taken_refuses():
    verdict = await quota(Counter(fail=True)).take("session:s-1", "user-a", now=NOW)
    assert verdict.reason == "quota-unavailable"
    assert "no credential was attached" in verdict.message()


@pytest.mark.parametrize(
    "per_box, per_user, window",
    [(-1, 3, 3600), (2, -1, 3600), (2, 3, 0), (2, 3, -60)],
)
def test_a_value_that_would_quietly_weaken_the_quota_is_refused(
    per_box, per_user, window
):
    with pytest.raises(ValueError):
        quota(Counter(), per_box=per_box, per_user=per_user, window=window)


@pytest.mark.parametrize("per_box, per_user", [(0, 0)])
async def test_zero_turns_a_limit_off(per_box, per_user):
    redis = Counter(fail=True)  # never asked
    q = quota(redis, per_box=per_box, per_user=per_user)
    assert (await q.take("session:s-1", "user-a", now=NOW)).allowed


async def test_a_request_refused_on_the_user_quota_spends_nothing_on_the_box():
    redis = Counter()
    q = quota(redis, per_box=5, per_user=1)
    assert (await q.take("session:s-1", "user-a", now=NOW)).allowed
    before = await redis.counts()
    verdict = await q.take("session:s-2", "user-a", now=NOW)
    assert (verdict.reason, verdict.scope) == ("quota-exceeded", "user")
    # Neither counter moved: not the refused user's, not the new box's.
    after = await redis.counts()
    assert after == before
    assert not any("session:s-2" in key for key in after)


async def test_both_scopes_are_counted_in_one_script_on_one_cluster_slot():
    redis = Counter()
    await quota(redis).take("session:s-1", "user-a", now=NOW)
    assert len(redis.scripts) == 1
    keys = list(await redis.counts())
    assert len(keys) == 2
    # One hash tag, so Redis Cluster runs the script on one slot.
    assert all("{user-a}" in key for key in keys)


async def test_the_script_runs_as_lua_and_refuses_at_the_limit_not_past_it():
    """The real script, no Python stand-in: with a limit of 2 the third
    request is refused and the counters stop at 2."""
    redis = Counter()
    q = quota(redis, per_box=2, per_user=0)
    results = [
        (await q.take("session:s-1", "user-a", now=NOW)).allowed for _ in range(3)
    ]
    assert results == [True, True, False]
    assert list((await redis.counts()).values()) == [2]
