import json
import time

from backend.voice_relay import locks
from backend.voice_relay.config import (
    CALL_RECORD_TTL_S,
    LIVE_CALL_STALE_S,
    USER_LOCK_TTL_S,
)

USER = "user-1"


async def test_the_first_call_takes_the_users_lock_with_a_ttl(redis):
    assert await locks.take_user_lock(redis, USER, "call-a") is None
    assert await locks.live_call_of(redis, USER) == "call-a"
    assert 0 < await redis.ttl(locks.user_lock_key(USER)) <= USER_LOCK_TTL_S


async def test_a_second_call_is_told_which_call_holds_the_lock(redis):
    await locks.take_user_lock(redis, USER, "call-a")
    assert await locks.take_user_lock(redis, USER, "call-b") == "call-a"
    assert await locks.live_call_of(redis, USER) == "call-a"


async def test_only_the_call_that_holds_the_lock_can_release_it(redis):
    await locks.take_user_lock(redis, USER, "call-a")
    assert not await locks.release_user_lock(redis, USER, "call-b")
    assert await locks.live_call_of(redis, USER) == "call-a"
    assert await locks.release_user_lock(redis, USER, "call-a")
    assert await locks.live_call_of(redis, USER) is None


async def test_a_call_whose_lock_expired_cannot_free_its_successors(redis):
    await locks.take_user_lock(redis, USER, "call-a")
    await redis.delete(locks.user_lock_key(USER))
    assert await locks.take_user_lock(redis, USER, "call-b") is None
    assert not await locks.release_user_lock(redis, USER, "call-a")
    assert await locks.live_call_of(redis, USER) == "call-b"


async def test_users_do_not_share_a_lock(redis):
    await locks.take_user_lock(redis, USER, "call-a")
    assert await locks.take_user_lock(redis, "user-2", "call-b") is None


async def test_live_calls_are_counted_across_the_platform(redis):
    now = time.time()
    await locks.register_live_call(redis, "call-a", now)
    await locks.register_live_call(redis, "call-b", now)
    # A refresh does not count a call twice.
    await locks.register_live_call(redis, "call-a", now + 30)
    assert await locks.live_call_count(redis, now + 30) == 2
    await locks.unregister_live_call(redis, "call-b")
    assert await locks.live_call_count(redis, now + 30) == 1


async def test_a_call_whose_pod_died_stops_counting_once_stale(redis):
    now = time.time()
    await locks.register_live_call(redis, "dead-pod-call", now)
    await locks.register_live_call(redis, "live-call", now)
    later = now + LIVE_CALL_STALE_S + 1
    await locks.register_live_call(redis, "live-call", later)
    assert await locks.live_call_count(redis, later) == 1
    # The next registration sweeps the stale entry away for good.
    assert await redis.zscore(locks.LIVE_CALLS_KEY, "dead-pod-call") is None


async def test_an_end_request_is_seen_by_the_pod_that_owns_the_call(redis):
    assert not await locks.end_requested(redis, "call-a")
    await locks.request_end(redis, "call-a")
    assert await locks.end_requested(redis, "call-a")
    assert not await locks.end_requested(redis, "call-b")


async def test_the_call_record_keeps_strings_and_json_for_a_day(redis):
    await locks.write_call_record(
        redis,
        "call-a",
        {"status": "ended", "turns": 3, "transcript": [{"who": "otto"}], "error": None},
    )
    record = await locks.read_call_record(redis, "call-a")
    assert record["status"] == "ended"
    assert json.loads(record["turns"]) == 3
    assert json.loads(record["transcript"]) == [{"who": "otto"}]
    assert "error" not in record
    assert 0 < await redis.ttl(locks.call_record_key("call-a")) <= CALL_RECORD_TTL_S


async def test_a_store_keeps_a_call_going_when_redis_is_down():
    class DownRedis:
        def __getattr__(self, name):
            async def fail(*args, **kwargs):
                raise ConnectionError("redis is down")

            if name == "pipeline":
                raise ConnectionError("redis is down")
            return fail

    store = locks.CallStore(DownRedis(), call_id="call-a", user_id=USER)  # type: ignore[arg-type]
    await store.record(status="live")
    await store.refresh_live(time.time())
    await store.release()
    assert await store.end_requested() is False
