"""The rooms of calls on the LiveKit leg, against a fake of LiveKit's room
service and a fake Redis: how a room is closed, by the call that is done with
it and by the sweep, and what the sweep leaves alone."""

import asyncio
import time

import pytest
from livekit import api
from prometheus_client import REGISTRY

from backend.voice_relay import locks, room_reaper, rooms
from backend.voice_relay.conftest import (
    LIVEKIT,
    FakeMedia,
    FakeRoomService,
    livekit_edge,
)
from backend.voice_relay.conftest import room_reaper as reaper_on
from backend.voice_relay.conftest import running_relay, until
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.room_reaper import RoomReaper
from backend.voice_relay.session_test import USER, Call, open_call

CALL = "a" * 32
OTHER_CALL = "b" * 32
PERSON = f"user:{USER}"
NOW = 1_800_000_000.0


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


async def test_a_close_removes_the_person_with_their_tokens_revoked_until_now_then_deletes_the_room(
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
    assert remove.revoke_token_ts == int(NOW)
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


@pytest.mark.parametrize("fails", ["raises", "hangs"])
async def test_a_close_that_fails_is_counted_and_leaves_the_entry_to_the_sweep(
    redis, monkeypatch, fails
):
    monkeypatch.setattr(room_reaper, "ROOM_LEAVE_TIMEOUT_S", 0.1)
    service = FakeRoomService()
    service.open(rooms.room_name(CALL), PERSON)
    if fails == "raises":
        service.fail["delete"] = api.TwirpError("unavailable", "busy", status=503)
    else:
        service.hang.add("delete")
    await locks.register_room(redis, CALL, USER, NOW - 100)
    reaper = reaper_on(redis, service, clock=Clock())
    before = _closed("call_end", "failed")

    assert not await reaper.close(CALL, USER, "call_end")

    entry = await _entry(redis)
    assert entry is not None and entry.closed_at is None
    assert _closed("call_end", "failed") - before == 1
    assert reaper.closing == 0


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
    await locks.mark_room_closed(redis, CALL, NOW - 50)
    before = _closed("reopened")

    assert await reaper_on(redis, service, clock=Clock()).sweep()

    assert service.kinds() == ["list", "remove", "delete"]
    assert service.rooms == {}
    assert _closed("reopened") - before == 1


async def test_an_entry_is_forgotten_only_once_closed_longer_than_a_token_lives_and_unlisted(
    redis,
):
    service = FakeRoomService()
    await locks.register_room(redis, CALL, USER, NOW - 700)
    await locks.mark_room_closed(redis, CALL, NOW - 590)
    clock = Clock()
    reaper = reaper_on(redis, service, clock=clock)

    await reaper.sweep()
    assert await _entry(redis) is not None

    # Past every token's life, but open again: closed again, not forgotten.
    clock.now += 20
    service.open(rooms.room_name(CALL), PERSON)
    await reaper.sweep()
    assert service.rooms == {}
    assert await _entry(redis) is not None

    clock.now += 700
    await reaper.sweep()
    assert await _entry(redis) is None


async def test_a_listed_room_with_no_entry_is_counted_as_another_and_never_touched(
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


async def test_a_room_the_sweep_cannot_settle_leaves_the_rest_of_the_sweep_running(
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

    assert await reaper_on(redis, service, clock=Clock()).sweep()

    [_, (_, remove), _] = service.requests
    assert isinstance(remove, api.RoomParticipantIdentity)
    assert remove.room == rooms.room_name(OTHER_CALL)


async def test_the_relay_sweeps_again_and_again(redis):
    service = FakeRoomService()
    sweeping = asyncio.create_task(
        reaper_on(redis, service, clock=Clock(), sweep_s=0.01).sweep_forever()
    )

    await until(lambda: service.kinds().count("list") >= 3)
    sweeping.cancel()
    await asyncio.gather(sweeping, return_exceptions=True)


# --- the registry ------------------------------------------------------------


async def test_the_registry_skips_an_unreadable_entry_and_keeps_a_stray_close_to_forget(
    redis,
):
    await redis.hset(locks.ROOMS_KEY, CALL, "{not json")
    await locks.mark_room_closed(redis, OTHER_CALL, NOW - 10)
    await redis.hset(locks.ROOMS_KEY, "c" * 32 + ":closed", "never")

    entries = await locks.registered_rooms(redis)

    assert list(entries) == [OTHER_CALL]
    stray = entries[OTHER_CALL]
    assert (stray.uid, stray.closed_at) == ("", NOW - 10)


# --- a call's room, closed by the call ----------------------------------------


async def _room_call(redis, service: FakeRoomService, media: FakeMedia, **options):
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)])
    call = await open_call(
        redis,
        vendor,
        leg="livekit",
        token_leg="livekit",
        edge=livekit_edge(media),
        reaper=reaper_on(redis, service, **options),
    )
    call_id = call.session.call_id
    await locks.register_room(redis, call_id, USER, time.time())
    service.open(rooms.room_name(call_id), PERSON, "otto")
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
    assert remove.identity == PERSON and remove.revoke_token_ts == int(NOW)
    entry = await _entry(redis, call.session.call_id)
    assert entry is not None and entry.closed_at == NOW


async def test_a_call_whose_browser_could_not_join_its_room_closes_the_room_at_once(
    redis,
):
    service = FakeRoomService()
    before = _closed("downgrade")
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)])

    call = await open_call(
        redis,
        vendor,
        leg="websocket",
        token_leg="livekit",
        reaper=reaper_on(redis, service),
    )

    await until(lambda: service.kinds() == ["remove", "delete"])
    assert not call.session.ended.is_set()
    assert _closed("downgrade") - before == 1
    assert (await _hang_up(call))["reason"] == "user_hangup"
    # The call's end has nothing left to close.
    assert service.kinds() == ["remove", "delete"]


async def test_a_relay_that_could_not_join_the_calls_room_closes_it_at_once(redis):
    service = FakeRoomService()
    media = FakeMedia(refuse=ConnectionError("the plan's allowance is used up"))
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)])

    call = await open_call(
        redis,
        vendor,
        leg="livekit",
        token_leg="livekit",
        edge=livekit_edge(media),
        reaper=reaper_on(redis, service),
    )

    assert call.ws.sent[0] == {"type": "leg", "leg": "websocket"}
    await until(lambda: service.kinds() == ["remove", "delete"])
    assert not call.session.ended.is_set()
    await _hang_up(call)
    assert service.kinds() == ["remove", "delete"]


async def test_a_call_whose_downgrade_close_failed_closes_its_room_again_as_it_ends(
    redis,
):
    service = FakeRoomService()
    service.fail["delete"] = api.TwirpError("unavailable", "busy", status=503)
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)])
    call = await open_call(
        redis,
        vendor,
        leg="websocket",
        token_leg="livekit",
        reaper=reaper_on(redis, service),
    )
    await until(lambda: service.kinds() == ["remove", "delete"])

    del service.fail["delete"]
    await _hang_up(call)

    assert service.kinds() == ["remove", "delete", "remove", "delete"]


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


async def test_a_call_on_the_socket_from_its_start_closes_no_room(redis):
    service = FakeRoomService()
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.1)])
    call = await open_call(redis, vendor, reaper=reaper_on(redis, service))
    await call.ws.wait_for("turn_complete")

    await _hang_up(call)

    assert service.requests == []


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
    call = await open_call(
        redis, vendor, leg="websocket", token_leg="livekit", reaper=reaper
    )
    await call.ws.wait_for("turn_complete")

    assert (await _hang_up(call))["reason"] == "user_hangup"
    assert reaper.closes == ["downgrade", "call_end"]
