import asyncio
import json
import time
from datetime import UTC, datetime

import pytest
from prisma.enums import BrainDumpInputMode, BrainDumpStatus

from backend.api.features.onboarding_dump.models import FinalizeResponse
from backend.voice_relay import locks
from backend.voice_relay.config import (
    CALL_RECORD_TTL_S,
    LIVE_CALL_STALE_S,
    POSTCALL_CLAIM_TTL_S,
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


async def test_nothing_spent_today_reads_as_zero(redis):
    assert await locks.daily_spend_usd(redis, datetime.now(UTC)) == 0.0


def test_the_daily_budget_waits_for_the_utc_day_to_turn():
    now = datetime(2026, 11, 9, 23, 59, 30, tzinfo=UTC)
    assert locks.seconds_until_tomorrow(now) == 30
    assert locks.daily_spend_key(now) == "voice:spend:2026-11-09"


async def test_a_call_takes_a_place_under_the_platforms_ceiling(redis):
    now = time.time()
    assert await locks.acquire_live_call(redis, "call-a", now, capacity=2)
    assert await locks.acquire_live_call(redis, "call-b", now, capacity=2)
    assert not await locks.acquire_live_call(redis, "call-c", now, capacity=2)
    # Refused with nothing written.
    assert await redis.zscore(locks.LIVE_CALLS_KEY, "call-c") is None
    assert await locks.live_call_count(redis, now) == 2


async def test_a_call_already_counted_keeps_its_place(redis):
    now = time.time()
    assert await locks.acquire_live_call(redis, "call-a", now, capacity=1)
    assert await locks.acquire_live_call(redis, "call-a", now + 10, capacity=1)
    assert await locks.live_call_count(redis, now + 10) == 1


async def test_entries_a_dead_pod_left_hold_no_place(redis):
    now = time.time()
    await locks.register_live_call(redis, "dead-pod-call", now - LIVE_CALL_STALE_S - 1)
    assert await locks.acquire_live_call(redis, "call-a", now, capacity=1)
    assert await redis.zscore(locks.LIVE_CALLS_KEY, "dead-pod-call") is None


async def test_a_post_call_job_is_claimed_by_one_holder_and_given_up_by_it(redis):
    assert await locks.claim_postcall(redis, "call-a", "pod-a")
    assert not await locks.claim_postcall(redis, "call-a", "pod-b")
    assert (
        0
        < await redis.ttl(locks.postcall_claim_key("call-a"))
        <= (POSTCALL_CLAIM_TTL_S)
    )
    # Only the holder can give it up.
    await locks.release_postcall(redis, "call-a", "pod-b")
    assert not await locks.claim_postcall(redis, "call-a", "pod-b")
    await locks.release_postcall(redis, "call-a", "pod-a")
    assert await locks.claim_postcall(redis, "call-a", "pod-b")


async def test_calls_stay_pending_until_their_job_clears_them(redis):
    await locks.add_postcall_pending(redis, "call-b")
    await locks.add_postcall_pending(redis, "call-a")
    await locks.add_postcall_pending(redis, "call-a")
    assert await locks.postcall_pending(redis) == ["call-a", "call-b"]
    await locks.remove_postcall_pending(redis, "call-a")
    assert await locks.postcall_pending(redis) == ["call-b"]


async def test_a_call_is_live_while_its_pod_refreshes_it(redis):
    now = time.time()
    await locks.register_live_call(redis, "call-a", now)
    assert await locks.is_live(redis, "call-a", now)
    assert not await locks.is_live(redis, "call-a", now + LIVE_CALL_STALE_S + 1)
    assert not await locks.is_live(redis, "call-b", now)


async def test_a_call_record_reads_back_typed(redis):
    await locks.write_call_record(
        redis,
        "call-a",
        {
            "uid": USER,
            "status": "failed",
            "turns": [{"index": 0, "at": 0, "text": "Hi"}, {"index": 1, "at": 640}],
            "hand_off": {
                "status": "failed",
                "input_mode": "typed",
                "error_code": "quality_check_unavailable",
            },
            "transcript_of_record": "I run a bakery.",
            "postcall_attempts": 2,
        },
    )

    record = await locks.read_call(redis, "call-a")

    assert record is not None
    assert (record.uid, record.status) == (USER, "failed")
    assert [mark.at for mark in record.turns] == [0, 640]
    assert record.hand_off is not None
    assert record.hand_off.error_code == "quality_check_unavailable"
    assert record.transcript_of_record == "I run a bakery."
    assert record.postcall_attempts == 2
    assert not record.accepted
    assert not record.took_interview
    assert await locks.read_call(redis, "call-b") is None


async def test_a_call_that_took_the_interview_says_so_on_its_record(redis):
    await locks.write_call_record(
        redis,
        "call-a",
        {
            "uid": USER,
            "status": "handed_off",
            "hand_off": {"status": "transcribed", "input_mode": "typed"},
            "interview": "taken",
        },
    )

    record = await locks.read_call(redis, "call-a")

    assert record is not None
    assert record.accepted
    assert record.took_interview


async def test_a_record_with_a_field_that_cannot_be_read_keeps_its_owner(redis):
    await locks.write_call_record(
        redis, "call-a", {"uid": USER, "status": "on-the-moon", "turns": "{oops"}
    )

    record = await locks.read_call(redis, "call-a")

    assert record is not None
    assert record.uid == USER
    assert record.status is None


async def test_a_field_that_cannot_be_read_drops_only_itself(redis):
    await locks.write_call_record(
        redis,
        "call-a",
        {
            "uid": USER,
            "status": "transcribed",
            "transcript_of_record": "I run a bakery.",
            "hand_off": {"status": "on-the-moon"},
            "postcall_attempts": "two",
        },
    )

    record = await locks.read_call(redis, "call-a")

    assert record is not None
    assert (record.uid, record.status) == (USER, "transcribed")
    assert record.transcript_of_record == "I run a bakery."
    assert record.hand_off is None
    assert record.postcall_attempts == 0


async def test_a_stored_transcript_is_never_replaced_by_an_empty_one(redis):
    await locks.write_call_record(
        redis, "call-a", {"uid": USER, "audio_path": "gcs://b/x.wav"}
    )
    first = await locks.store_transcript(
        redis, "call-a", "I run a bakery.", uid=USER, from_upload=True
    )

    again = await locks.store_transcript(
        redis, "call-a", "", uid=USER, from_upload=False
    )

    assert first == again == "I run a bakery."
    record = await locks.read_call(redis, "call-a")
    assert record is not None
    assert record.transcript_of_record == "I run a bakery."
    assert record.transcribed_from_upload
    assert record.audio_path is None


async def test_a_stored_transcript_moves_the_call_to_transcribed_with_its_time(redis):
    await locks.write_call_record(redis, "call-a", {"uid": USER, "status": "ended"})

    await locks.store_transcript(
        redis, "call-a", "I run a bakery.", uid=USER, from_upload=False, now=1234.5
    )

    record = await locks.read_call(redis, "call-a")
    assert record is not None
    assert (record.status, record.transcribed_at) == ("transcribed", 1234.5)


async def test_a_transcript_never_lands_on_a_call_whose_hand_off_answered(redis):
    await locks.write_call_record(
        redis,
        "call-a",
        {
            "uid": USER,
            "status": "handed_off",
            "transcript_of_record": "I run a bakery.",
            "hand_off": {"status": "transcribed", "input_mode": "typed"},
        },
    )

    stored = await locks.store_transcript(
        redis, "call-a", "Something else.", uid=USER, from_upload=False
    )

    assert stored == "I run a bakery."
    record = await locks.read_call(redis, "call-a")
    assert record is not None
    assert (record.status, record.transcript_of_record) == (
        "handed_off",
        "I run a bakery.",
    )


async def test_the_relays_own_answer_is_written_on_the_record(redis):
    await locks.record_answer(
        redis,
        "call-a",
        uid=USER,
        status=locks.FAILED,
        answer=FinalizeResponse(
            status=BrainDumpStatus.failed,
            input_mode=BrainDumpInputMode.typed,
            error_code="transcription_unavailable",
        ),
    )

    record = await locks.read_call(redis, "call-a")
    assert record is not None and record.hand_off is not None
    assert (record.uid, record.status) == (USER, "failed")
    assert record.hand_off.error_code == "transcription_unavailable"


async def test_a_transcript_made_again_from_the_whole_recording_replaces_it(redis):
    await locks.write_call_record(redis, "call-a", {"uid": USER})
    await locks.store_transcript(redis, "call-a", "I run a", uid=USER, from_upload=True)

    await locks.store_transcript(
        redis, "call-a", "I run a bakery.", uid=USER, from_upload=False
    )

    record = await locks.read_call(redis, "call-a")
    assert record is not None
    assert record.transcript_of_record == "I run a bakery."
    assert not record.transcribed_from_upload


async def test_of_two_interview_claims_on_a_call_exactly_one_stands(redis):
    await locks.write_call_record(redis, "call-a", {"uid": USER, "status": "ended"})

    standing = await asyncio.gather(
        locks.claim_interview(redis, "call-a", locks.INTERVIEW_TAKEN),
        locks.claim_interview(redis, "call-a", locks.INTERVIEW_SUPERSEDED),
    )

    assert standing[0] == standing[1]
    assert await locks.interview_of(redis, "call-a") == standing[0]
    # Claiming again changes nothing.
    other = (
        locks.INTERVIEW_SUPERSEDED
        if standing[0] == locks.INTERVIEW_TAKEN
        else locks.INTERVIEW_TAKEN
    )
    assert await locks.claim_interview(redis, "call-a", other) == standing[0]


async def test_a_call_with_no_record_takes_no_interview_claim(redis):
    assert (
        await locks.claim_interview(redis, "call-a", locks.INTERVIEW_SUPERSEDED) is None
    )
    assert not await redis.exists(locks.call_record_key("call-a"))


@pytest.mark.parametrize("status", [None, "connecting", "live"])
async def test_a_call_not_yet_ended_takes_no_interview_claim(redis, status):
    """A typed message admitted before the call started does not outrank it."""
    await locks.write_call_record(redis, "call-a", {"uid": USER, "status": status})

    assert (
        await locks.claim_interview(redis, "call-a", locks.INTERVIEW_SUPERSEDED) is None
    )
    assert await locks.interview_of(redis, "call-a") is None

    await locks.write_call_record(redis, "call-a", {"status": "ended"})
    assert await locks.claim_interview(redis, "call-a", locks.INTERVIEW_SUPERSEDED) == (
        locks.INTERVIEW_SUPERSEDED
    )


async def test_every_write_of_a_call_names_the_call_before_it(redis):
    store = locks.CallStore(redis, call_id="call-b", user_id=USER)
    store.prev = "call-a"

    await store.advance(locks.LIVE, moves_from=(None, locks.CONNECTING))
    record = await locks.read_call(redis, "call-b")
    assert record is not None and (record.uid, record.prev) == (USER, "call-a")

    await redis.hdel(locks.call_record_key("call-b"), "prev")
    await store.record(turns=[])
    record = await locks.read_call(redis, "call-b")
    assert record is not None and record.prev == "call-a"


async def test_a_call_ends_on_its_record_only_while_it_still_runs(redis):
    await locks.write_call_record(
        redis,
        "call-a",
        {"uid": USER, "status": "transcribing", "end_reason": "crashed"},
    )

    moved = await locks.advance_call(
        redis,
        "call-a",
        locks.ENDED,
        moves_from=(None, *locks.RUNNING),
        guarded={"end_reason": "finished"},
        fields={"elapsed_s": 42.0},
    )

    assert not moved
    record = await locks.read_call_record(redis, "call-a")
    assert record["status"] == "transcribing"
    assert record["end_reason"] == "crashed"
    assert json.loads(record["elapsed_s"]) == 42.0


async def test_a_running_call_moves_to_ended_with_its_reason(redis):
    await locks.write_call_record(redis, "call-a", {"uid": USER, "status": "live"})

    moved = await locks.advance_call(
        redis,
        "call-a",
        locks.ENDED,
        moves_from=(None, *locks.RUNNING),
        guarded={"end_reason": "finished"},
    )

    assert moved
    record = await locks.read_call_record(redis, "call-a")
    assert (record["status"], record["end_reason"]) == ("ended", "finished")
    assert 0 < await redis.ttl(locks.call_record_key("call-a")) <= CALL_RECORD_TTL_S


async def test_every_write_a_call_makes_names_its_owner(redis):
    store = locks.CallStore(redis, call_id="call-a", user_id=USER)

    await store.record(turns=[])
    await redis.hdel(locks.call_record_key("call-a"), "uid")
    await store.advance(locks.LIVE, moves_from=(None,), connect_s=0.4)

    record = await locks.read_call_record(redis, "call-a")
    assert record["uid"] == USER
    assert record["status"] == "live"


async def test_only_its_holder_can_renew_a_post_call_claim(redis):
    await locks.claim_postcall(redis, "call-a", "pod-a", ttl_s=30)

    assert not await locks.hold_postcall(redis, "call-a", "pod-b")
    assert await locks.hold_postcall(redis, "call-a", "pod-a")
    assert await redis.ttl(locks.postcall_claim_key("call-a")) > 30
    await redis.delete(locks.postcall_claim_key("call-a"))
    assert not await locks.hold_postcall(redis, "call-a", "pod-a")


async def test_a_running_call_holds_its_post_call_claim_with_each_refresh(redis):
    store = locks.CallStore(redis, call_id="call-a", user_id=USER, holder="pod-a")
    assert await store.claim_postcall()
    await redis.expire(locks.postcall_claim_key("call-a"), 5)

    await store.refresh_live(time.time())

    assert await redis.ttl(locks.postcall_claim_key("call-a")) > 5
    assert store.claimed


# ---------- whose lock holds the person back ----------


async def test_a_call_whose_pod_refreshes_it_holds_the_person(redis):
    now = time.time()
    await locks.take_user_lock(redis, USER, "call-a", now)
    await locks.register_live_call(redis, "call-a", now)
    await locks.write_call_record(redis, "call-a", {"uid": USER, "status": "live"})

    assert await locks.live_call_holding(redis, USER, now) == "call-a"


async def test_a_call_still_starting_holds_the_person(redis):
    now = time.time()
    await locks.take_user_lock(redis, USER, "call-a", now)

    assert await locks.live_call_holding(redis, USER, now) == "call-a"


@pytest.mark.parametrize("what", ["pod-died", "ended-unreleased", "dropped-as-stale"])
async def test_a_lock_whose_call_no_longer_runs_holds_nobody_back(redis, what):
    now = time.time()
    await locks.take_user_lock(redis, USER, "call-a", now)
    if what == "pod-died":
        await locks.register_live_call(redis, "call-a", now - LIVE_CALL_STALE_S - 1)
        await locks.write_call_record(redis, "call-a", {"uid": USER, "status": "live"})
    elif what == "ended-unreleased":
        await locks.register_live_call(redis, "call-a", now)
        await locks.write_call_record(redis, "call-a", {"uid": USER, "status": "ended"})
    else:
        await locks.write_call_record(redis, "call-a", {"uid": USER, "status": "live"})

    assert await locks.live_call_holding(redis, USER, now) is None
    # Whose the call is still reads from the lock.
    assert await locks.live_call_of(redis, USER) == "call-a"


async def test_a_new_call_takes_the_lock_over_from_one_whose_pod_died(redis):
    now = time.time()
    await locks.take_user_lock(redis, USER, "call-a", now)
    await locks.register_live_call(redis, "call-a", now - LIVE_CALL_STALE_S - 1)
    await locks.write_call_record(redis, "call-a", {"uid": USER, "status": "live"})

    assert await locks.take_user_lock(redis, USER, "call-b", now) is None
    assert await locks.live_call_of(redis, USER) == "call-b"
    assert 0 < await redis.ttl(locks.user_lock_key(USER)) <= USER_LOCK_TTL_S


async def test_the_transcript_is_stored_in_the_write_that_forgets_the_recording(
    redis,
):
    await locks.write_call_record(redis, "call-a", {"audio_path": "gcs://b/x.wav"})

    await locks.forget_call_fields(
        redis, "call-a", {"transcript_of_record": "Hello."}, "audio_path"
    )

    record = await locks.read_call_record(redis, "call-a")
    assert record == {"transcript_of_record": "Hello."}


async def test_a_registered_room_names_the_person_it_is_for(redis):
    await locks.register_room(redis, "a" * 32, "user-1", 100.0)
    await redis.hset(locks.ROOMS_KEY, "c" * 32, "{not json")

    assert await locks.room_owner(redis, "a" * 32) == "user-1"
    assert await locks.room_owner(redis, "b" * 32) is None
    assert await locks.room_owner(redis, "c" * 32) is None
