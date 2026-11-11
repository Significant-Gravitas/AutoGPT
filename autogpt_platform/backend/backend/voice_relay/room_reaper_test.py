"""The calls' LiveKit rooms, against a fake of LiveKit's room service and a
fake Redis: how a room is closed, by the call that is done with it and by
the sweep, and what the sweep leaves alone."""

import asyncio
import time

import pytest
from livekit import api
from prometheus_client import REGISTRY

from backend.voice_relay import locks, room_reaper, rooms
from backend.voice_relay.config import (
    CAPS,
    ROOM_END_GRACE_S,
    ROOM_FORGET_AFTER_REVOKED_S,
    ROOM_PLAYOUT_S,
    ROOM_REVOCATION_S,
    ROOM_REVOKE_AGAIN_AFTER_S,
    ROOM_REVOKE_AHEAD_S,
    ROOM_START_GRACE_S,
    ROOM_SWEEP_S,
    ROOM_TOKEN_LEEWAY_S,
    ROOM_TOKEN_TTL_S,
    ROOM_UNREGISTERED_GRACE_S,
)
from backend.voice_relay.conftest import (
    LIVEKIT,
    FakeMedia,
    FakeRoomService,
    FakeSocket,
    livekit_edge,
)
from backend.voice_relay.conftest import room_reaper as reaper_on
from backend.voice_relay.conftest import running_relay, until
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.room_reaper import RoomReaper
from backend.voice_relay.session_test import USER, Call, _session, open_call

CALL = "a" * 32
OTHER_CALL = "b" * 32
# The person, as they join CALL's room.
PERSON = rooms.person_identity(USER, CALL)
NOW = 1_800_000_000.0
# A close's cutoff: every token of the person's issued until a moment past
# the close.
CUTOFF = int(NOW) + ROOM_REVOKE_AHEAD_S


class Clock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _closed(why: str, outcome: str = "closed") -> float:
    return (
        REGISTRY.get_sample_value(
            "voice_interview_livekit_rooms_closed_total",
            {"why": why, "outcome": outcome},
        )
        or 0.0
    )


def _sweeps(outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "voice_interview_livekit_sweeps_total", {"outcome": outcome}
        )
        or 0.0
    )


def _gauge(name: str, labels: dict | None = None) -> float | None:
    return REGISTRY.get_sample_value(name, labels or {})


async def _entry(redis, call_id: str = CALL) -> locks.RoomEntry | None:
    return (await locks.registered_rooms(redis)).get(call_id)


# --- closing a room ----------------------------------------------------------


async def test_a_close_removes_the_person_with_their_tokens_revoked_past_now_then_deletes_the_room(
    redis,
):
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON, "otto")
    await locks.register_room(redis, CALL, USER, NOW - 100)
    before = _closed("call_end")

    assert await reaper_on(redis, service, clock=Clock()).close(CALL, USER, "call_end")

    assert service.kinds() == ["remove", "delete"]
    [(_, remove), (_, delete)] = service.requests
    assert isinstance(remove, api.RoomParticipantIdentity)
    assert (remove.room, remove.identity) == (rooms.room_name(CALL), PERSON)
    # A little ahead of now, so a token issued this same second is caught.
    assert remove.revoke_token_ts == CUTOFF > NOW
    assert isinstance(delete, api.DeleteRoomRequest)
    assert delete.room == rooms.room_name(CALL)
    assert service.rooms == {}
    entry = await _entry(redis)
    assert entry is not None and entry.closed_at == NOW
    assert _closed("call_end") - before == 1


async def test_a_room_or_a_person_already_gone_counts_as_closed(redis):
    service = FakeRoomService()
    await locks.register_room(redis, CALL, USER, NOW - 100)

    assert await reaper_on(redis, service, clock=Clock()).close(CALL, USER, "orphaned")

    assert service.kinds() == ["remove", "delete"]
    entry = await _entry(redis)
    assert entry is not None and entry.closed_at == NOW


@pytest.mark.parametrize(
    "fails",
    ["the deletion raises", "the deletion hangs", "the removal raises"],
)
async def test_a_close_that_fails_is_counted_and_leaves_the_entry_to_the_sweep(
    redis, monkeypatch, fails
):
    monkeypatch.setattr(room_reaper, "ROOM_LEAVE_TIMEOUT_S", 0.1)
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON)
    if fails == "the deletion raises":
        service.fail["delete"] = api.TwirpError("unavailable", "busy", status=503)
    elif fails == "the deletion hangs":
        service.hang.add("delete")
    else:
        service.fail["remove"] = api.TwirpError(
            "invalid_argument", "revoke_token_ts out of range", status=400
        )
    await locks.register_room(redis, CALL, USER, NOW - 100)
    reaper = reaper_on(redis, service, clock=Clock())
    before = _closed("call_end", "failed")

    assert not await reaper.close(CALL, USER, "call_end")

    entry = await _entry(redis)
    assert entry is not None and entry.closed_at is None
    assert _closed("call_end", "failed") - before == 1
    assert reaper.closing == 0
    # The room is deleted whatever became of the removal.
    assert service.kinds() == ["remove", "delete"]


async def test_the_room_service_is_made_from_the_project_once_and_closed_with_the_reaper(
    redis, monkeypatch
):
    made: list[tuple] = []
    service = FakeRoomService()
    clients: list["FakeApi"] = []

    class FakeApi:
        def __init__(self, url, key, secret) -> None:
            made.append((url, key, secret))
            clients.append(self)
            self.room = service
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(api, "LiveKitAPI", FakeApi)

    async def get_redis():
        return redis

    reaper = RoomReaper(LIVEKIT, redis=get_redis)

    await reaper.close(CALL, USER, "call_end")
    await reaper.close(OTHER_CALL, USER, "call_end")
    await reaper.aclose()

    assert made == [(LIVEKIT.url, LIVEKIT.api_key, LIVEKIT.api_secret)]
    assert clients[0].closed
    assert service.kinds() == ["remove", "delete", "remove", "delete"]


async def test_closing_a_reaper_that_made_no_room_service_does_nothing(redis):
    async def get_redis():
        return redis

    await RoomReaper(LIVEKIT, redis=get_redis).aclose()


# --- the sweep ---------------------------------------------------------------


async def test_the_sweep_leaves_a_room_alone_while_its_call_may_still_start(redis):
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON)
    await locks.register_room(redis, CALL, USER, NOW - 74)
    clock = Clock()
    reaper = reaper_on(redis, service, clock=clock)
    before = _closed("never_started")

    assert await reaper.sweep()
    assert service.kinds() == ["list"]

    # Past the start's grace, the call never started: its token is dead.
    clock.now += 2
    assert await reaper.sweep()

    assert service.kinds() == ["list", "list", "remove", "delete"]
    assert _closed("never_started") - before == 1
    entry = await _entry(redis)
    assert entry is not None and entry.closed_at == clock.now


async def test_the_sweep_leaves_a_room_alone_while_its_call_runs_and_closes_it_once_its_pod_stops_refreshing(
    redis,
):
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON, "otto")
    await locks.register_room(redis, CALL, USER, NOW - 290)
    await locks.write_call_record(redis, CALL, {"status": "live", "uid": USER})
    await locks.register_live_call(redis, CALL, NOW - 10)
    clock = Clock()
    reaper = reaper_on(redis, service, clock=clock)
    before = _closed("orphaned")

    assert await reaper.sweep()
    assert service.kinds() == ["list"]

    # The call's pod died: its place in the live count went stale.
    clock.now += locks.LIVE_CALL_STALE_S
    assert await reaper.sweep()

    assert service.kinds() == ["list", "list", "remove", "delete"]
    assert _closed("orphaned") - before == 1


async def test_the_sweep_waits_out_the_end_of_a_call_then_closes_a_room_its_close_missed(
    redis,
):
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON)
    await locks.register_room(redis, CALL, USER, NOW - 200)
    await locks.write_call_record(
        redis, CALL, {"status": "transcribing", "ended_at": NOW - 29, "uid": USER}
    )
    clock = Clock()
    reaper = reaper_on(redis, service, clock=clock)

    assert await reaper.sweep()
    assert service.kinds() == ["list"]

    clock.now += 2
    assert await reaper.sweep()

    assert service.kinds() == ["list", "list", "remove", "delete"]
    assert service.rooms == {}


async def test_the_sweep_closes_a_room_whose_call_record_cannot_say_when_it_ended(
    redis,
):
    service = FakeRoomService()
    await locks.register_room(redis, CALL, USER, NOW - 200)
    await locks.write_call_record(redis, CALL, {"status": "ended", "uid": USER})

    assert await reaper_on(redis, service, clock=Clock()).sweep()

    assert service.kinds() == ["list", "remove", "delete"]


async def test_a_closed_room_listed_again_is_closed_again(redis):
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON)
    await locks.register_room(redis, CALL, USER, NOW - 100)
    await locks.mark_room_closed(redis, CALL, USER, NOW - 50)
    before = _closed("reopened")

    assert await reaper_on(redis, service, clock=Clock()).sweep()

    assert service.kinds() == ["list", "remove", "delete"]
    assert service.rooms == {}
    assert _closed("reopened") - before == 1


async def test_a_closed_rooms_person_is_revoked_again_before_the_first_revocation_lapses(
    redis,
):
    service = FakeRoomService()
    await locks.register_room(redis, CALL, USER, NOW - 700)
    await locks.mark_room_closed(
        redis, CALL, USER, NOW - ROOM_REVOKE_AGAIN_AFTER_S + 10
    )
    clock = Clock()
    reaper = reaper_on(redis, service, clock=clock)
    before = _closed("revoke_again")

    assert await reaper.sweep()
    assert service.kinds() == ["list"]

    clock.now += 20
    assert await reaper.sweep()

    # The person again, with a fresh cutoff; the room is not listed, so it
    # is not deleted again.
    assert service.kinds() == ["list", "list", "remove"]
    _, remove = service.requests[-1]
    assert isinstance(remove, api.RoomParticipantIdentity)
    assert remove.identity == PERSON
    assert remove.revoke_token_ts == int(clock.now) + ROOM_REVOKE_AHEAD_S
    entry = await _entry(redis)
    assert entry is not None and entry.revoked_again_at == clock.now
    assert _closed("revoke_again") - before == 1
    assert clock.now - entry.closed_at < ROOM_REVOCATION_S - 2 * ROOM_SWEEP_S

    # Revoked again once only.
    clock.now += ROOM_SWEEP_S
    assert await reaper.sweep()
    assert service.kinds().count("remove") == 1


async def test_a_second_revocation_that_fails_is_tried_again_and_holds_the_heartbeat_back(
    redis,
):
    service = FakeRoomService()
    await locks.register_room(redis, CALL, USER, NOW - 900)
    await locks.mark_room_closed(redis, CALL, USER, NOW - ROOM_REVOKE_AGAIN_AFTER_S)
    service.fail["remove"] = api.TwirpError("unavailable", "busy", status=503)
    clock = Clock()
    reaper = reaper_on(redis, service, clock=clock)
    before = _closed("revoke_again", "failed")

    assert not await reaper.sweep()

    entry = await _entry(redis)
    assert entry is not None and entry.revoked_again_at is None
    assert _closed("revoke_again", "failed") - before == 1
    assert not await locks.rooms_swept(redis)

    del service.fail["remove"]
    clock.now += ROOM_SWEEP_S
    assert await reaper.sweep()
    entry = await _entry(redis)
    assert entry is not None and entry.revoked_again_at == clock.now


async def test_a_heartbeat_that_cannot_be_written_fails_the_sweep(redis, monkeypatch):
    async def unwritable(redis, now):
        raise ConnectionError("redis went away")

    monkeypatch.setattr(locks, "mark_rooms_swept", unwritable)
    before = _sweeps("failed")

    assert not await reaper_on(redis, FakeRoomService(), clock=Clock()).sweep()

    assert _sweeps("failed") - before == 1


async def test_an_entry_is_forgotten_only_once_its_second_revocation_and_the_leeway_ran_out(
    redis,
):
    service = FakeRoomService()
    await locks.register_room(redis, CALL, USER, NOW - 2_000)
    await locks.mark_room_closed(redis, CALL, USER, NOW - 1_000)
    await locks.mark_room_revoked_again(
        redis, CALL, USER, closed_at=NOW - 1_000, revoked_at=NOW - 600
    )
    clock = Clock()
    reaper = reaper_on(redis, service, clock=clock)

    # Past the second revocation, but not the leeway after it.
    assert await reaper.sweep()
    assert await _entry(redis) is not None

    # Past both, but open again: closed again, not forgotten.
    clock.now += ROOM_TOKEN_LEEWAY_S + 1
    service.open(rooms.room_name(CALL), PERSON)
    assert await reaper.sweep()
    assert service.rooms == {}
    entry = await _entry(redis)
    assert entry is not None and entry.closed_at == clock.now
    # The new close has its own second revocation to come.
    assert entry.revoked_again_at is None

    clock.now += ROOM_REVOKE_AGAIN_AFTER_S
    assert await reaper.sweep()
    clock.now += ROOM_FORGET_AFTER_REVOKED_S
    assert await reaper.sweep()
    assert await _entry(redis) is not None
    clock.now += 1
    assert await reaper.sweep()
    assert await _entry(redis) is None


def test_no_token_for_a_closed_room_outlives_its_entry():
    # The second revocation comes some sweeps before the first lapses, and
    # the entry outlasts it by the leeway LiveKit gives an expired token.
    assert ROOM_REVOKE_AGAIN_AFTER_S + 2 * ROOM_SWEEP_S < ROOM_REVOCATION_S
    assert ROOM_TOKEN_TTL_S <= ROOM_REVOCATION_S
    assert ROOM_FORGET_AFTER_REVOKED_S >= ROOM_REVOCATION_S + ROOM_TOKEN_LEEWAY_S
    assert 0 < ROOM_REVOKE_AHEAD_S < ROOM_TOKEN_LEEWAY_S


async def test_a_close_stamp_that_outlived_its_entry_still_revokes_the_person_it_names(
    redis,
):
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON)
    # Another relay forgot the entry while this one's close was stamping it.
    await locks.register_room(redis, CALL, USER, NOW - 900)
    await locks.forget_room(redis, CALL)
    await locks.mark_room_closed(redis, CALL, USER, NOW - 30)

    assert await reaper_on(redis, service, clock=Clock()).sweep()

    assert service.kinds() == ["list", "remove", "delete"]
    _, remove = service.requests[1]
    assert isinstance(remove, api.RoomParticipantIdentity)
    assert remove.identity == PERSON and remove.revoke_token_ts == CUTOFF


async def test_on_a_shared_project_a_listed_room_with_no_entry_is_counted_as_another_and_never_touched(
    redis,
):
    service = FakeRoomService()
    stranger = rooms.room_name("c" * 32)
    service.open(stranger, "user:someone", created=int(NOW) - 5_000)
    service.open("standup", "user:a", "user:b")

    assert await reaper_on(redis, service, clock=Clock()).sweep()

    assert service.kinds() == ["list"]
    assert set(service.rooms) == {stranger, "standup"}
    assert _gauge("voice_interview_livekit_rooms", {"owner": "other"}) == 1
    assert _gauge("voice_interview_livekit_rooms", {"owner": "ours"}) == 0
    assert _gauge("voice_interview_livekit_oldest_room_seconds") == 0


# --- rooms with no entry, where the project is this deployment's alone ---


STRAY_CALL = "c" * 32
STRAY_PERSON = rooms.person_identity("someone", STRAY_CALL)
OLD = int(NOW) - int(ROOM_UNREGISTERED_GRACE_S) - 5


def test_a_room_with_no_entry_is_given_longer_than_any_calls_room_can_live():
    longest_room = (
        ROOM_START_GRACE_S
        + CAPS.max_call_s
        + ROOM_PLAYOUT_S
        + ROOM_END_GRACE_S
        + 2 * ROOM_SWEEP_S
    )
    assert ROOM_UNREGISTERED_GRACE_S > longest_room


async def test_on_an_exclusive_project_a_room_with_no_entry_is_left_alone_while_young(
    redis,
):
    service = FakeRoomService()
    young = int(NOW) - int(ROOM_UNREGISTERED_GRACE_S) + 5
    service.open(rooms.room_name(STRAY_CALL), STRAY_PERSON, created=young)

    assert await reaper_on(redis, service, clock=Clock(), exclusive=True).sweep()

    assert service.kinds() == ["list"]
    assert await locks.registered_rooms(redis) == {}


async def test_on_an_exclusive_project_an_old_room_with_no_entry_is_adopted_and_closed_revoking_its_person(
    redis,
):
    service = FakeRoomService()
    service.open(rooms.room_name(STRAY_CALL), "otto", STRAY_PERSON, created=OLD)
    clock = Clock()
    reaper = reaper_on(redis, service, clock=clock, exclusive=True)
    before = _closed("unregistered")

    assert await reaper.sweep()

    assert service.kinds() == ["list", "participants", "remove", "delete"]
    _, remove = service.requests[2]
    assert isinstance(remove, api.RoomParticipantIdentity)
    assert remove.identity == STRAY_PERSON and remove.revoke_token_ts == CUTOFF
    assert service.rooms == {}
    entry = await _entry(redis, STRAY_CALL)
    assert entry is not None
    assert (entry.uid, entry.minted_at, entry.closed_at) == ("someone", OLD, NOW)
    assert _closed("unregistered") - before == 1

    # From then on it is a closed room like any other: revoked again in time.
    clock.now += ROOM_REVOKE_AGAIN_AFTER_S
    assert await reaper.sweep()
    _, again = service.requests[-1]
    assert isinstance(again, api.RoomParticipantIdentity)
    assert again.identity == STRAY_PERSON


@pytest.mark.parametrize(
    "inside",
    [(), ("otto",), (rooms.person_identity("someone", OTHER_CALL),)],
    ids=["nobody", "only-otto", "another-calls-identity"],
)
async def test_on_an_exclusive_project_a_room_with_no_entry_and_no_person_is_deleted(
    redis, inside
):
    service = FakeRoomService()
    service.open(rooms.room_name(STRAY_CALL), *inside, created=OLD)

    assert await reaper_on(redis, service, clock=Clock(), exclusive=True).sweep()

    assert service.kinds() == ["list", "participants", "delete"]
    entry = await _entry(redis, STRAY_CALL)
    assert entry is not None and (entry.uid, entry.closed_at) == ("", NOW)


async def test_on_an_exclusive_project_a_room_whose_people_cannot_be_listed_waits_for_the_next_sweep(
    redis,
):
    service = FakeRoomService()
    service.open(rooms.room_name(STRAY_CALL), STRAY_PERSON, created=OLD)
    service.fail["participants"] = api.TwirpError("unavailable", "busy", status=503)
    clock = Clock()
    reaper = reaper_on(redis, service, clock=clock, exclusive=True)

    assert not await reaper.sweep()

    assert service.kinds() == ["list", "participants"]
    assert await locks.registered_rooms(redis) == {}
    assert not await locks.rooms_swept(redis)

    del service.fail["participants"]
    clock.now += ROOM_SWEEP_S
    assert await reaper.sweep()
    assert service.rooms == {}


async def test_on_an_exclusive_project_a_room_gone_before_its_people_were_listed_needs_nothing(
    redis,
):
    service = FakeRoomService()
    service.open(rooms.room_name(STRAY_CALL), STRAY_PERSON, created=OLD)
    service.fail["participants"] = api.TwirpError("not_found", "no room", status=404)

    assert await reaper_on(redis, service, clock=Clock(), exclusive=True).sweep()

    assert service.kinds() == ["list", "participants"]
    assert await locks.registered_rooms(redis) == {}


async def test_on_an_exclusive_project_an_adopted_room_whose_close_failed_is_closed_by_the_next_sweep(
    redis,
):
    service = FakeRoomService()
    service.open(rooms.room_name(STRAY_CALL), STRAY_PERSON, created=OLD)
    service.fail["delete"] = api.TwirpError("unavailable", "busy", status=503)
    clock = Clock()
    reaper = reaper_on(redis, service, clock=clock, exclusive=True)

    assert not await reaper.sweep()
    entry = await _entry(redis, STRAY_CALL)
    assert entry is not None and entry.uid == "someone" and entry.closed_at is None

    del service.fail["delete"]
    clock.now += ROOM_SWEEP_S
    assert await reaper.sweep()

    assert service.rooms == {}
    assert service.kinds()[-2:] == ["remove", "delete"]
    entry = await _entry(redis, STRAY_CALL)
    assert entry is not None and entry.closed_at == clock.now


async def test_the_gauges_follow_the_projects_rooms(redis):
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON, "otto", created=int(NOW) - 90)
    service.open(rooms.room_name(OTHER_CALL), "user:other", created=int(NOW) - 30)
    service.open(rooms.room_name("c" * 32), "user:someone", created=int(NOW) - 900)
    service.open("standup", "user:a", "user:b", "user:c")
    await locks.register_room(redis, CALL, USER, NOW - 10)
    await locks.register_room(redis, OTHER_CALL, "other", NOW - 10)

    assert await reaper_on(redis, service, clock=Clock()).sweep()

    assert _gauge("voice_interview_livekit_rooms", {"owner": "ours"}) == 2
    assert _gauge("voice_interview_livekit_rooms", {"owner": "other"}) == 1
    # Everyone connected to the project, which is what LiveKit bills.
    assert _gauge("voice_interview_livekit_participants") == 7
    assert _gauge("voice_interview_livekit_oldest_room_seconds") == 90


async def test_two_relays_sweeping_at_once_close_a_room_without_error(redis):
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON)
    await locks.register_room(redis, CALL, USER, NOW - 200)
    failed_before = _closed("never_started", "failed")

    swept = await asyncio.gather(
        reaper_on(redis, service, clock=Clock()).sweep(),
        reaper_on(redis, service, clock=Clock()).sweep(),
    )

    assert swept == [True, True]
    assert service.rooms == {}
    assert _closed("never_started", "failed") == failed_before
    entry = await _entry(redis)
    assert entry is not None and entry.closed_at == NOW


async def test_the_heartbeat_is_written_by_a_sweep_that_ran_and_not_by_one_that_could_not_list(
    redis,
):
    service = FakeRoomService()
    service.fail["list"] = api.TwirpError("unavailable", "down", status=503)
    await locks.register_room(redis, CALL, USER, NOW - 200)
    reaper = reaper_on(redis, service, clock=Clock())
    failed_before = _sweeps("failed")

    assert not await reaper.sweep()
    assert not await locks.rooms_swept(redis)
    # A sweep that could not list closes nothing.
    assert service.kinds() == ["list"]
    assert _sweeps("failed") - failed_before == 1

    del service.fail["list"]
    ok_before = _sweeps("ok")
    assert await reaper.sweep()

    assert await locks.rooms_swept(redis)
    assert 0 < await redis.ttl(locks.ROOMS_SWEPT_KEY) <= 120
    assert _sweeps("ok") - ok_before == 1


async def test_a_sweep_that_cannot_reach_redis_closes_nothing_and_writes_no_heartbeat():
    service = FakeRoomService()

    async def no_redis():
        raise ConnectionError("redis is down")

    reaper = RoomReaper(LIVEKIT, redis=no_redis, service=service)

    assert not await reaper.sweep()
    assert service.requests == []


async def test_a_room_the_sweep_cannot_settle_leaves_the_rest_of_the_sweep_running_with_no_heartbeat(
    redis, monkeypatch
):
    service = FakeRoomService()
    await locks.register_room(redis, CALL, USER, NOW - 200)
    await locks.register_room(redis, OTHER_CALL, USER, NOW - 200)
    real = room_reaper._unused

    async def unused(redis, entry, now):
        if entry.call_id == CALL:
            raise ConnectionError("its record could not be read")
        return await real(redis, entry, now)

    monkeypatch.setattr(room_reaper, "_unused", unused)

    assert not await reaper_on(redis, service, clock=Clock()).sweep()

    [_, (_, remove), _] = service.requests
    assert isinstance(remove, api.RoomParticipantIdentity)
    assert remove.room == rooms.room_name(OTHER_CALL)
    assert not await locks.rooms_swept(redis)


async def test_while_closes_keep_failing_no_heartbeat_is_written_and_no_call_starts(
    redis,
):
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON)
    service.fail["remove"] = api.TwirpError(
        "invalid_argument", "revoke_token_ts out of range", status=400
    )
    await locks.register_room(redis, CALL, USER, NOW - 200)
    clock = Clock()
    reaper = reaper_on(redis, service, clock=clock)

    assert not await reaper.sweep()

    # The room went even so; the close is not stamped, and no heartbeat says
    # rooms are being closed.
    assert service.kinds() == ["list", "remove", "delete"]
    assert service.rooms == {}
    entry = await _entry(redis)
    assert entry is not None and entry.closed_at is None
    assert not await locks.rooms_swept(redis)

    del service.fail["remove"]
    clock.now += ROOM_SWEEP_S
    assert await reaper.sweep()
    entry = await _entry(redis)
    assert entry is not None and entry.closed_at == clock.now
    assert await locks.rooms_swept(redis)


async def test_the_relay_sweeps_again_and_again(redis):
    service = FakeRoomService()
    sweeping = asyncio.create_task(
        reaper_on(redis, service, clock=Clock(), sweep_s=0.01).sweep_forever()
    )

    await until(lambda: service.kinds().count("list") >= 3)
    sweeping.cancel()
    await asyncio.gather(sweeping, return_exceptions=True)


# --- the registry ------------------------------------------------------------


async def test_the_registry_skips_what_it_cannot_read_and_keeps_a_stray_close_with_its_person(
    redis,
):
    await redis.hset(locks.ROOMS_KEY, CALL, "{not json")
    await locks.mark_room_closed(redis, CALL, USER, NOW - 20)
    await locks.mark_room_closed(redis, OTHER_CALL, USER, NOW - 10)
    await redis.hset(locks.ROOMS_KEY, "c" * 32 + ":closed", "never")
    await redis.hset(locks.ROOMS_KEY, "d" * 32 + ":revoked", "{}")

    entries = await locks.registered_rooms(redis)

    assert list(entries) == [OTHER_CALL]
    stray = entries[OTHER_CALL]
    assert (stray.uid, stray.minted_at, stray.closed_at) == (USER, NOW - 10, NOW - 10)


async def test_a_second_revocation_stamped_for_an_earlier_close_reads_as_none(redis):
    await locks.register_room(redis, CALL, USER, NOW - 900)
    await locks.mark_room_closed(redis, CALL, USER, NOW - 100)
    await locks.mark_room_revoked_again(
        redis, CALL, USER, closed_at=NOW - 800, revoked_at=NOW - 300
    )

    entry = await _entry(redis)

    assert entry is not None
    assert (entry.closed_at, entry.revoked_again_at) == (NOW - 100, None)


# --- a call's room, closed by the call ----------------------------------------


async def _room_call(redis, service: FakeRoomService, media: FakeMedia, **options):
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)])
    call = await open_call(
        redis,
        vendor,
        media=media,
        reaper=reaper_on(redis, service, **options),
    )
    call_id = call.session.call_id
    await locks.register_room(redis, call_id, USER, time.time())
    service.open(rooms.room_name(call_id), rooms.person_identity(USER, call_id), "otto")
    await call.ws.wait_for("turn_complete")
    return call


async def _hang_up(call: Call) -> dict:
    call.ws.send_json({"type": "hangup"})
    return await call.ended()


async def test_a_calls_end_closes_its_room_after_otto_played_out_and_the_relay_left(
    redis,
):
    service = FakeRoomService()
    media = FakeMedia()
    call = await _room_call(redis, service, media, clock=Clock())
    source = media.sources[0]
    source.playing = asyncio.Event()
    seen: list[tuple[str, bool]] = []
    service.watch = lambda kind: seen.append((kind, media.rooms[0].left))

    call.ws.send_json({"type": "hangup"})
    await call.ws.wait_for("ended")
    await asyncio.sleep(0.05)
    assert seen == []
    source.playing.set()
    await asyncio.wait_for(call.task, timeout=5)

    # The relay left first, then the person went, then the room.
    assert seen == [("remove", True), ("delete", True)]
    [(_, remove), _] = service.requests
    assert isinstance(remove, api.RoomParticipantIdentity)
    assert remove.identity == rooms.person_identity(USER, call.session.call_id)
    assert remove.revoke_token_ts == CUTOFF
    entry = await _entry(redis, call.session.call_id)
    assert entry is not None and entry.closed_at == NOW


async def test_a_call_whose_room_the_relay_could_not_join_closes_it_as_it_ends(
    redis,
):
    service = FakeRoomService()
    media = FakeMedia(refuse=ConnectionError("the plan's allowance is used up"))
    session = _session(
        redis,
        FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)]),
        FakeSocket(),
        [],
        edge=livekit_edge(media),
        reaper=reaper_on(redis, service),
    )

    await asyncio.wait_for(session.run(), timeout=5)

    assert session.end_reason == "room_error"
    assert service.kinds() == ["remove", "delete"]


@pytest.mark.parametrize("fails", ["raises", "hangs"])
async def test_a_room_whose_close_fails_still_lets_the_call_end_normally(
    redis, monkeypatch, fails
):
    monkeypatch.setattr(room_reaper, "ROOM_LEAVE_TIMEOUT_S", 0.1)
    service = FakeRoomService()
    if fails == "raises":
        service.fail["remove"] = ConnectionError("LiveKit is unreachable")
    else:
        service.hang.add("remove")
    call = await _room_call(redis, service, FakeMedia())

    assert (await _hang_up(call))["reason"] == "user_hangup"
    await asyncio.wait_for(call.task, timeout=5)

    entry = await _entry(redis, call.session.call_id)
    assert entry is not None and entry.closed_at is None


# --- a start the browser gave up on --------------------------------------------


async def test_the_sweep_closes_at_once_a_room_whose_start_the_person_gave_up_on(
    redis,
):
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON)
    # Well inside the start's grace: the call could still start, but its end
    # was asked for.
    await locks.register_room(redis, CALL, USER, NOW - 5)
    await locks.request_end(redis, CALL)
    before = _closed("never_started")

    assert await reaper_on(redis, service, clock=Clock()).sweep()

    assert service.kinds() == ["list", "remove", "delete"]
    assert _closed("never_started") - before == 1
    entry = await _entry(redis)
    assert entry is not None and entry.closed_at == NOW


async def test_the_sweep_leaves_a_started_call_alone_whose_end_was_asked_for(
    redis,
):
    """Its relay ends it within a second, and closes its room."""
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON, "otto")
    await locks.register_room(redis, CALL, USER, NOW - 5)
    await locks.write_call_record(redis, CALL, {"uid": USER, "status": "live"})
    await locks.register_live_call(redis, CALL, NOW)
    await locks.request_end(redis, CALL)

    assert await reaper_on(redis, service, clock=Clock()).sweep()

    assert service.kinds() == ["list"]


# --- the relay ---------------------------------------------------------------


async def test_a_relay_with_a_project_sweeps_the_rooms_from_its_start(redis):
    service = FakeRoomService()

    async with running_relay(redis, FakeVendor([]), reaper=reaper_on(redis, service)):
        await until(lambda: "list" in service.kinds())

    assert await locks.rooms_swept(redis)


async def test_a_draining_relay_waits_for_a_room_close_under_way(redis):
    service = FakeRoomService()
    service.hang.add("delete")
    reaper = reaper_on(redis, service)

    async with running_relay(redis, FakeVendor([]), reaper=reaper) as running:
        await until(lambda: "list" in service.kinds())
        assert not running.relay.busy
        closing = asyncio.create_task(reaper.close(CALL, USER, "call_end"))
        await until(lambda: "delete" in service.kinds())

        assert running.relay.busy

        closing.cancel()
        await asyncio.gather(closing, return_exceptions=True)
        assert not running.relay.busy


class BrokenReaper:
    """A room closer that raises where a close should only ever report."""

    def __init__(self) -> None:
        self.closes: list[str] = []

    async def close(self, call_id: str, user_id: str, why: str) -> bool:
        self.closes.append(why)
        raise RuntimeError("the reaper broke")


async def test_a_room_closer_that_raises_still_lets_the_call_run_and_end(redis):
    reaper = BrokenReaper()
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)])
    call = await open_call(redis, vendor, reaper=reaper)
    await call.ws.wait_for("turn_complete")

    assert (await _hang_up(call))["reason"] == "user_hangup"
    assert reaper.closes == ["call_end"]
