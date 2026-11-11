import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from backend.api.utils.cors import build_cors_params
from backend.util.settings import AppEnvironment
from backend.voice_relay import app as relay_app
from backend.voice_relay import locks, tokens
from backend.voice_relay.app import Relay, RelayDeps, first_name_of
from backend.voice_relay.config import CAPS, DEFAULT_DRAIN_S
from backend.voice_relay.conftest import (
    ORIGIN,
    FakeMedia,
    FakeRoomService,
    MemoryRecordings,
    ScriptedTranscriber,
    livekit_edge,
    room_reaper,
    running_relay,
    silence,
    speech,
    until,
)
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.locks import TurnMark
from backend.voice_relay.postcall import PostCall

USER = "user-1"


async def open_socket(url: str, origin: str | None = ORIGIN) -> ClientConnection:
    return await connect(url, origin=origin, max_size=2**20)  # type: ignore[arg-type]


async def refusal(ws: ClientConnection) -> tuple[dict, int | None]:
    """The relay's error message, and the code it closed the socket with."""
    message = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    with pytest.raises(ConnectionClosed) as closed:
        await asyncio.wait_for(ws.recv(), timeout=5)
    return message, closed.value.rcvd.code if closed.value.rcvd else None


async def read_until(ws: ClientConnection, kind: str) -> dict:
    async def read() -> dict:
        while True:
            message = await ws.recv()
            if isinstance(message, str) and json.loads(message)["type"] == kind:
                return json.loads(message)

    return await asyncio.wait_for(read(), timeout=10)


def start(user: str = USER, call_id: str | None = None) -> dict:
    call_id = call_id or tokens.new_call_id()
    return {"type": "start", "token": tokens.mint_start(user, call_id)}


@pytest.mark.parametrize(
    "origin", ["https://elsewhere.example", None], ids=["foreign", "missing"]
)
async def test_a_socket_from_an_origin_the_platform_does_not_allow_is_refused(
    redis, origin
):
    async with running_relay(redis, FakeVendor()) as running:
        with pytest.raises(InvalidStatus) as refused:
            await open_socket(running.url, origin=origin)
        assert refused.value.response.status_code == 403


@pytest.mark.parametrize(
    "first",
    [
        json.dumps({"type": "hangup"}),
        json.dumps({"type": "start", "token": "x", "pad": "x" * 5_000}),
        "not json",
        json.dumps(["start"]),
    ],
    ids=["hangup-first", "oversized", "not-json", "not-an-object"],
)
async def test_anything_but_start_or_resume_first_is_refused(redis, first):
    async with running_relay(redis, FakeVendor()) as running:
        ws = await open_socket(running.url)
        await ws.send(first)
        assert await refusal(ws) == ({"type": "error", "error": "protocol"}, 1008)


async def test_audio_before_start_is_refused(redis):
    async with running_relay(redis, FakeVendor()) as running:
        ws = await open_socket(running.url)
        await ws.send(b"\x00" * 1280)
        assert await refusal(ws) == ({"type": "error", "error": "protocol"}, 1008)


async def test_a_socket_that_never_names_its_call_is_closed(redis, monkeypatch):
    monkeypatch.setattr(relay_app, "START_TIMEOUT_S", 0.2)
    async with running_relay(redis, FakeVendor()) as running:
        ws = await open_socket(running.url)
        assert await refusal(ws) == ({"type": "error", "error": "start_timeout"}, 1008)


async def test_a_token_is_good_for_one_call_only(redis):
    async with running_relay(redis, FakeVendor([Reply(text="Hi")])) as running:
        first = start()
        ws = await open_socket(running.url)
        await ws.send(json.dumps(first))
        await read_until(ws, "ready")
        again = await open_socket(running.url)
        await again.send(json.dumps(first))
        assert await refusal(again) == ({"type": "error", "error": "token_used"}, 1008)
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")


async def test_a_forged_token_is_refused(redis):
    async with running_relay(redis, FakeVendor()) as running:
        ws = await open_socket(running.url)
        token = tokens.mint_start(USER, tokens.new_call_id())
        await ws.send(json.dumps({"type": "start", "token": token[:-2] + "AA"}))
        assert await refusal(ws) == ({"type": "error", "error": "token_invalid"}, 1008)


async def test_a_second_live_call_for_the_same_person_is_refused_naming_the_first(
    redis,
):
    async with running_relay(redis, FakeVendor([Reply(text="Hi")])) as running:
        first_call = tokens.new_call_id()
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start(call_id=first_call)))
        await read_until(ws, "ready")
        second = await open_socket(running.url)
        await second.send(json.dumps(start()))
        assert await refusal(second) == (
            {"type": "error", "error": "call_in_progress", "call_id": first_call},
            1008,
        )
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")


async def test_a_full_relay_asks_new_calls_to_try_again(redis):
    async with running_relay(
        redis, FakeVendor([Reply(text="Hi")]), max_calls=1
    ) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        await read_until(ws, "ready")
        other = await open_socket(running.url)
        await other.send(json.dumps(start(user="user-2")))
        assert await refusal(other) == ({"type": "error", "error": "busy"}, 1013)
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")


async def test_a_dropped_socket_resumes_its_call_with_the_ticket(redis):
    vendor = FakeVendor([Reply(text="Hi", audio_s=0.1)])
    async with running_relay(redis, vendor) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        ready = await read_until(ws, "ready")
        await ws.close()

        again = await open_socket(running.url)
        await again.send(json.dumps({"type": "resume", "ticket": ready["resume"]}))
        resumed = await read_until(again, "resumed")
        assert resumed["gap_s"] >= 0
        await again.send(json.dumps({"type": "hangup"}))
        ended = await read_until(again, "ended")
        assert ended["reason"] == "user_hangup"
    record = await locks.read_call_record(redis, ready["call_id"])
    assert record["reattaches"] == "1"


async def test_a_ticket_resumes_only_its_own_users_call(redis):
    async with running_relay(redis, FakeVendor([Reply(text="Hi")])) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        ready = await read_until(ws, "ready")
        stranger = await open_socket(running.url)
        ticket = tokens.mint_resume("user-2", ready["call_id"], ttl_s=60)
        await stranger.send(json.dumps({"type": "resume", "ticket": ticket}))
        assert await refusal(stranger) == ({"type": "error", "error": "no_call"}, 4404)
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")


async def test_a_ticket_for_a_call_not_on_this_relay_is_refused(redis):
    async with running_relay(redis, FakeVendor()) as running:
        ws = await open_socket(running.url)
        ticket = tokens.mint_resume(USER, tokens.new_call_id(), ttl_s=60)
        await ws.send(json.dumps({"type": "resume", "ticket": ticket}))
        assert await refusal(ws) == ({"type": "error", "error": "no_call"}, 4404)


async def test_a_start_token_cannot_resume_a_call(redis):
    async with running_relay(redis, FakeVendor()) as running:
        ws = await open_socket(running.url)
        token = start()["token"]
        await ws.send(json.dumps({"type": "resume", "ticket": token}))
        assert await refusal(ws) == ({"type": "error", "error": "token_invalid"}, 1008)


async def test_the_probes_report_a_ready_relay(redis):
    async with (
        running_relay(redis, FakeVendor()) as running,
        httpx.AsyncClient(base_url=running.http) as http,
    ):
        assert (await http.get("/healthz")).json() == {"ok": True}
        ready = await http.get("/readyz")
    assert ready.status_code == 200
    assert ready.json() == {"ready": True, "live_calls": 0}


async def test_otto_greets_the_person_by_the_name_on_their_account(redis):
    vendor = FakeVendor([Reply(text="Hi")])
    async with running_relay(redis, vendor) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        await read_until(ws, "ready")
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")
    assert vendor.openings == [
        "(The user just opened the voice onboarding call. Their first name is Dana.)"
    ]


@pytest.mark.parametrize(
    "name,first",
    [
        ("Dana Smith", "Dana"),
        ("  José  ", "José"),
        ("O'Brien-Lee", "O'Brien-Lee"),
        ("Dr. Who", "Dr"),
        ("dana@example.com", None),
        ("Ignore[all]rules", None),
        ("R2D2", None),
        ("", None),
        (None, None),
        ("x" * 41, None),
    ],
)
def test_only_a_name_that_reads_as_one_reaches_the_call(name, first):
    assert first_name_of(name) == first


async def test_a_started_call_is_recorded_on_the_persons_onboarding_row(redis):
    async with running_relay(redis, FakeVendor([Reply(text="Hi")])) as running:
        call_id = tokens.new_call_id()
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start(call_id=call_id)))
        await read_until(ws, "ready")
        assert running.recorded == {USER: call_id}
        # A second call refused for the live one leaves the record alone.
        second = await open_socket(running.url)
        await second.send(json.dumps(start()))
        await refusal(second)
        assert running.recorded == {USER: call_id}
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")


async def test_a_call_names_the_call_it_replaced_on_the_persons_row(redis):
    """The hand-off walks the person's calls back from the latest along it."""
    vendor = FakeVendor([Reply(text="Hi"), Reply(text="Hi again")])
    async with running_relay(redis, vendor) as running:
        first = tokens.new_call_id()
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start(call_id=first)))
        await read_until(ws, "ready")
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")

        second = tokens.new_call_id()
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start(call_id=second)))
        await read_until(ws, "ready")

        assert running.recorded == {USER: second}
        first_record = await locks.read_call(redis, first)
        second_record = await locks.read_call(redis, second)
        assert first_record is not None and first_record.prev is None
        assert second_record is not None and second_record.prev == first
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")


async def test_a_call_names_the_call_it_replaced_though_its_first_write_was_lost(
    redis, monkeypatch
):
    """Every write of the call's names the call before it, as it names the
    owner, so a first write that timed out loses neither."""
    record = locks.CallStore.record
    second = tokens.new_call_id()

    async def first_write_lost(self, **fields):
        if self.call_id == second and fields.get("status") == "connecting":
            return False
        return await record(self, **fields)

    monkeypatch.setattr(locks.CallStore, "record", first_write_lost)
    vendor = FakeVendor([Reply(text="Hi"), Reply(text="Hi again")])
    async with running_relay(redis, vendor) as running:
        first = tokens.new_call_id()
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start(call_id=first)))
        await read_until(ws, "ready")
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")

        ws = await open_socket(running.url)
        await ws.send(json.dumps(start(call_id=second)))
        await read_until(ws, "ready")

        written = await locks.read_call(redis, second)
        assert written is not None
        assert (written.uid, written.status, written.prev) == (USER, "live", first)
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")


async def test_a_call_is_pending_before_its_first_record_write(redis, monkeypatch):
    """A pod that dies between the two leaves a pending call with no record,
    which the relay's sweep drops, rather than a record nothing finishes."""
    record = locks.CallStore.record
    pending_at_first_write: list[bool] = []

    async def noting(self, **fields):
        if fields.get("status") == "connecting":
            pending = await locks.postcall_pending(self.redis)
            pending_at_first_write.append(self.call_id in pending)
        return await record(self, **fields)

    monkeypatch.setattr(locks.CallStore, "record", noting)
    async with running_relay(redis, FakeVendor([Reply(text="Hi")])) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        await read_until(ws, "ready")
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")

    assert pending_at_first_write == [True]


async def test_a_call_that_cannot_be_recorded_does_not_start(redis):
    vendor = FakeVendor([Reply(text="Hi")])
    async with running_relay(redis, vendor, record_failures=1) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        assert await refusal(ws) == ({"type": "error", "error": "unavailable"}, 1013)
    assert vendor.sessions == []
    assert await locks.live_call_of(redis, USER) is None
    # Nor does it keep a place in the platform-wide count.
    assert await redis.zcard(locks.LIVE_CALLS_KEY) == 0


async def _first_answer(ws: ClientConnection) -> str:
    """``ready``, or the error a refused start was given."""
    message = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    return message["type"] if message["type"] == "ready" else message["error"]


async def test_starts_that_arrive_together_hold_no_more_than_the_pods_slots(redis):
    # The first start still holds its slot while it records its call.
    vendor = FakeVendor([Reply(text="Hi")])
    async with running_relay(redis, vendor, max_calls=1, record_delay_s=0.3) as running:
        first, second = await open_socket(running.url), await open_socket(running.url)
        await first.send(json.dumps(start(user="user-1")))
        await second.send(json.dumps(start(user="user-2")))
        answers = await asyncio.gather(_first_answer(first), _first_answer(second))
        assert sorted(answers) == ["busy", "ready"]
        live = first if answers[0] == "ready" else second
        await live.send(json.dumps({"type": "hangup"}))
        await read_until(live, "ended")
    assert running.relay.held_slots == 0
    assert len(vendor.sessions) == 1


async def test_a_refused_start_gives_its_slot_back(redis):
    async with running_relay(
        redis, FakeVendor([Reply(text="Hi")]), max_calls=1
    ) as running:
        forged = await open_socket(running.url)
        await forged.send(json.dumps({"type": "start", "token": "forged.token"}))
        assert (await refusal(forged))[0]["error"] == "token_invalid"
        assert running.relay.held_slots == 0
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        await read_until(ws, "ready")
        assert running.relay.held_slots == 1
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")
    assert running.relay.held_slots == 0


async def test_a_start_past_the_platforms_live_call_ceiling_is_refused(redis):
    await locks.register_live_call(redis, "b" * 32, time.time())
    vendor = FakeVendor([Reply(text="Hi")])
    call_id = tokens.new_call_id()
    async with running_relay(redis, vendor, max_concurrent=1) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start(call_id=call_id)))
        assert await refusal(ws) == ({"type": "error", "error": "busy"}, 1013)
        assert running.recorded == {}
        assert running.relay.held_slots == 0
    assert vendor.sessions == []
    assert await locks.live_call_of(redis, USER) is None
    assert await redis.zscore(locks.LIVE_CALLS_KEY, call_id) is None


async def test_calls_whose_pod_died_leave_room_under_the_ceiling(redis):
    await locks.register_live_call(redis, "b" * 32, time.time() - 600)
    async with running_relay(
        redis, FakeVendor([Reply(text="Hi")]), max_concurrent=1
    ) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        await read_until(ws, "ready")
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")


async def test_a_start_refused_for_a_fault_on_the_platforms_side_can_use_its_token_again(
    redis,
):
    async with running_relay(
        redis, FakeVendor([Reply(text="Hi")]), record_failures=1
    ) as running:
        first = start()
        ws = await open_socket(running.url)
        await ws.send(json.dumps(first))
        assert await refusal(ws) == ({"type": "error", "error": "unavailable"}, 1013)
        again = await open_socket(running.url)
        await again.send(json.dumps(first))
        await read_until(again, "ready")
        await again.send(json.dumps({"type": "hangup"}))
        await read_until(again, "ended")


async def test_a_start_refused_while_the_platform_is_full_can_use_its_token_again(
    redis,
):
    other_call = "b" * 32
    await locks.register_live_call(redis, other_call, time.time())
    async with running_relay(
        redis, FakeVendor([Reply(text="Hi")]), max_concurrent=1
    ) as running:
        first = start()
        ws = await open_socket(running.url)
        await ws.send(json.dumps(first))
        assert await refusal(ws) == ({"type": "error", "error": "busy"}, 1013)
        await locks.unregister_live_call(redis, other_call)
        again = await open_socket(running.url)
        await again.send(json.dumps(first))
        await read_until(again, "ready")
        await again.send(json.dumps({"type": "hangup"}))
        await read_until(again, "ended")


class _Passthrough:
    """The test's Redis, with some of its calls replaced."""

    def __init__(self, redis) -> None:
        self._redis = redis

    def __getattr__(self, name: str):
        return getattr(self._redis, name)


class SpendFails(_Passthrough):
    """Spending a start token fails, the way a timed-out SET does."""

    async def set(self, name, *args, **kwargs):
        if name.startswith("voice:tok:"):
            raise ConnectionError("Redis went away")
        return await self._redis.set(name, *args, **kwargs)


class LockReplyLost(_Passthrough):
    """The first user lock taken lands, but its reply never arrives."""

    def __init__(self, redis) -> None:
        super().__init__(redis)
        self.lost = False

    async def set(self, name, *args, **kwargs):
        result = await self._redis.set(name, *args, **kwargs)
        if name.startswith("voice:live:") and not self.lost:
            self.lost = True
            raise TimeoutError("the reply was lost")
        return result


class SetNeverAnswers(_Passthrough):
    """A SET on keys with *prefix* never gets an answer."""

    def __init__(self, redis, prefix: str) -> None:
        super().__init__(redis)
        self.prefix = prefix

    async def set(self, name, *args, **kwargs):
        if name.startswith(self.prefix):
            await asyncio.Event().wait()
        return await self._redis.set(name, *args, **kwargs)


class SlowRelease(_Passthrough):
    """Freeing a user lock or a place in the live-call count takes a while."""

    async def eval(self, script, numkeys, *keys_and_args):
        if keys_and_args and str(keys_and_args[0]).startswith("voice:live:"):
            await asyncio.sleep(0.3)
        return await self._redis.eval(script, numkeys, *keys_and_args)

    async def zrem(self, name, *values):
        await asyncio.sleep(0.3)
        return await self._redis.zrem(name, *values)


async def test_a_start_whose_token_cannot_be_spent_leaves_the_calls_state_alone(
    redis,
):
    call_id = tokens.new_call_id()
    first = start(call_id=call_id)
    # The call is live, holding its token, its lock and its place.
    await tokens.spend(redis, tokens.verify_start(first["token"]))
    await locks.take_user_lock(redis, USER, call_id)
    await locks.register_live_call(redis, call_id, time.time())
    async with running_relay(SpendFails(redis), FakeVendor()) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(first))
        assert await refusal(ws) == ({"type": "error", "error": "unavailable"}, 1013)
    assert await locks.live_call_of(redis, USER) == call_id
    assert await redis.zscore(locks.LIVE_CALLS_KEY, call_id) is not None
    assert await redis.exists(tokens.spent_key(call_id))


async def test_a_lock_taken_without_a_reply_is_given_back_with_the_token(redis):
    call_id = tokens.new_call_id()
    first = start(call_id=call_id)
    vendor = FakeVendor([Reply(text="Hi")])
    async with running_relay(LockReplyLost(redis), vendor) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(first))
        assert await refusal(ws) == ({"type": "error", "error": "unavailable"}, 1013)
        assert await locks.live_call_of(redis, USER) is None
        assert not await redis.exists(tokens.spent_key(call_id))
        again = await open_socket(running.url)
        await again.send(json.dumps(first))
        await read_until(again, "ready")
        await again.send(json.dumps({"type": "hangup"}))
        await read_until(again, "ended")


async def test_a_retry_right_after_a_try_again_refusal_keeps_its_lock_and_place(
    redis,
):
    call_id = tokens.new_call_id()
    first = start(call_id=call_id)
    vendor = FakeVendor([Reply(text="Hi")])
    async with running_relay(SlowRelease(redis), vendor, record_failures=1) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(first))
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert message == {"type": "error", "error": "unavailable"}
        # Retried the moment the refusal arrives.
        again = await open_socket(running.url)
        await again.send(json.dumps(first))
        await read_until(again, "ready")
        # Long enough for anything the refused start still had to do.
        await asyncio.sleep(1.0)
        assert await locks.live_call_of(redis, USER) == call_id
        assert await redis.zscore(locks.LIVE_CALLS_KEY, call_id) is not None
        await again.send(json.dumps({"type": "hangup"}))
        await read_until(again, "ended")


async def test_a_second_start_for_a_call_live_on_this_relay_is_refused(redis):
    call_id = tokens.new_call_id()
    first = start(call_id=call_id)
    vendor = FakeVendor([Reply(text="Hi")])
    async with running_relay(redis, vendor) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(first))
        await read_until(ws, "ready")
        # The call's shared state is gone, as after a Redis failover.
        await redis.delete(locks.user_lock_key(USER), tokens.spent_key(call_id))

        again = await open_socket(running.url)
        await again.send(json.dumps(first))
        assert await refusal(again) == (
            {"type": "error", "error": "call_in_progress", "call_id": call_id},
            1008,
        )
        assert len(vendor.sessions) == 1
        assert running.relay.held_slots == 1
        # The live call keeps the lock the refused start took for it.
        assert await locks.live_call_of(redis, USER) == call_id
        await ws.send(json.dumps({"type": "hangup"}))
        assert (await read_until(ws, "ended"))["reason"] == "user_hangup"


def test_a_relay_never_holds_two_sessions_for_one_call():
    relay = Relay(
        max_calls=2, origins=build_cors_params([ORIGIN], AppEnvironment.LOCAL)
    )
    live = SimpleNamespace(call_id="c" * 32)
    duplicate = SimpleNamespace(call_id="c" * 32)
    first, second = relay.reserve_slot(), relay.reserve_slot()
    assert first is not None and second is not None
    assert first.hold(live)  # type: ignore[arg-type]
    assert not second.hold(duplicate)  # type: ignore[arg-type]
    second.release()
    assert relay.sessions == {"c" * 32: live}
    assert relay.held_slots == 1
    first.release()
    assert relay.sessions == {}
    assert relay.held_slots == 0


@pytest.mark.parametrize(
    "stalls",
    ["redis-client", "token-spend", "user-lock"],
)
async def test_a_start_whose_redis_never_answers_is_refused_and_frees_its_slot(
    redis, stalls
):
    client = {
        "redis-client": redis,
        "token-spend": SetNeverAnswers(redis, "voice:tok:"),
        "user-lock": SetNeverAnswers(redis, "voice:live:"),
    }[stalls]
    async with running_relay(
        client,
        FakeVendor(),
        redis_delay_s=30.0 if stalls == "redis-client" else 0.0,
    ) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        assert await refusal(ws) == ({"type": "error", "error": "unavailable"}, 1013)
        await until(lambda: running.relay.held_slots == 0)
    assert await locks.live_call_of(redis, USER) is None


async def test_a_calls_transcript_is_made_from_its_recording_once_it_ends(redis):
    vendor = FakeVendor(
        [
            Reply(text="Hi, what do you do?", audio_s=0.1),
            Reply(text="Got it. Bye!", heard="I bake", first_task="Bake bread"),
        ]
    )
    transcriber = ScriptedTranscriber(["I bake bread for three cafes."])
    async with running_relay(redis, vendor, transcriber=transcriber) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        ready = await read_until(ws, "ready")
        await read_until(ws, "turn_complete")
        (await running.media.hearing()).say(speech(0.4))
        ended = await read_until(ws, "ended")
        await running.postcalls_done()

        assert ended["reason"] == "finished"
        assert transcriber.calls[0][1] == "Dana"
        record = await locks.read_call(redis, ready["call_id"])
        assert record is not None
        assert record.status == locks.TRANSCRIBED
        assert record.transcript_of_record == "I bake bread for three cafes."
        assert record.first_task == "Bake bread"
        assert record.hand_off is None
        # The recording went once its transcript was stored.
        assert running.recordings.stored
        assert running.recordings.objects == {}
        # The relay is done with the call: the REST API hands it off.
        assert await redis.get(locks.postcall_claim_key(ready["call_id"])) is None
        assert await locks.postcall_pending(redis) == [ready["call_id"]]


async def test_a_call_carries_the_arm_its_token_named_onto_its_record(redis):
    call_id = tokens.new_call_id()
    token = tokens.mint_start(USER, call_id, ending="09-first-task")
    async with running_relay(redis, FakeVendor([Reply(text="Hi")])) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps({"type": "start", "token": token}))
        await read_until(ws, "ready")

        record = await locks.read_call(redis, call_id)
        assert record is not None
        assert (record.uid, record.ending) == (USER, "09-first-task")
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")


async def test_a_call_whose_pod_died_does_not_refuse_the_next_start(redis):
    dead = tokens.new_call_id()
    await locks.take_user_lock(redis, USER, dead)
    await locks.register_live_call(
        redis, dead, time.time() - locks.LIVE_CALL_STALE_S - 1
    )
    await locks.write_call_record(redis, dead, {"uid": USER, "status": "live"})
    async with running_relay(redis, FakeVendor([Reply(text="Hi")])) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        ready = await read_until(ws, "ready")

        assert await locks.live_call_of(redis, USER) == ready["call_id"]
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")


async def test_a_relay_that_cannot_record_a_call_is_not_ready(redis, tmp_path):
    not_a_directory = tmp_path / "spool"
    not_a_directory.write_text("")
    async with (
        running_relay(redis, FakeVendor()) as running,
        httpx.AsyncClient(base_url=running.http) as http,
    ):
        assert running.relay.deps is not None
        running.relay.deps.spool_dir = str(not_a_directory / "calls")
        ready = await http.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["ready"] is False


def _deps(redis, spool_dir, postcall: PostCall | None = None) -> RelayDeps:
    async def get_redis():
        return redis

    async def nobody(user_id: str) -> str | None:
        return None

    async def record_call(user_id: str, call_id: str) -> None:
        return None

    return RelayDeps(
        vendor=FakeVendor(),
        classifier=None,
        redis=get_redis,
        account_name=nobody,
        record_call=record_call,
        live_model="gemini-3.8-live",
        max_concurrent=25,
        postcall=postcall
        or PostCall(
            redis=get_redis,
            recordings=MemoryRecordings(),
            transcriber=ScriptedTranscriber(),
            first_name=nobody,
            live_model="gemini-3.8-live",
            pod="pod-a",
        ),
        spool_dir=str(spool_dir),
        edge=livekit_edge(FakeMedia()),
        reaper=room_reaper(redis, FakeRoomService()),
    )


async def test_a_relay_starting_deletes_the_spools_no_call_still_writes(
    redis, tmp_path
):
    left = tmp_path / f"{'a' * 32}.pcm"
    left.write_bytes(bytes(640))
    long_ago = time.time() - (CAPS.max_call_s + DEFAULT_DRAIN_S) - 60
    os.utime(left, (long_ago, long_ago))
    writing = tmp_path / f"{'b' * 32}.pcm"
    writing.write_bytes(bytes(640))
    relay = Relay(
        max_calls=1,
        origins=build_cors_params([ORIGIN], AppEnvironment.LOCAL),
        deps=_deps(redis, tmp_path),
    )
    app = relay_app.create_app(relay)

    async with app.router.lifespan_context(app):
        assert not left.exists()
        assert writing.exists()


async def test_post_call_jobs_give_up_their_claims_before_the_relay_closes_redis(
    redis, tmp_path, monkeypatch
):
    monkeypatch.setattr(relay_app, "SHUTDOWN_JOBS_S", 0.05)
    transcribing = asyncio.Event()

    class Hung:
        async def __call__(self, wav: bytes, *, first_name: str | None):
            transcribing.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def get_redis():
        return redis

    postcall = PostCall(
        redis=get_redis,
        recordings=MemoryRecordings(),
        transcriber=Hung(),
        first_name=lambda user_id: asyncio.sleep(0, result=None),
        live_model="gemini-3.8-live",
        pod="pod-a",
    )
    call_id = tokens.new_call_id()
    pcm = b"".join(silence(0.4) + speech(0.6) + silence(0.8))
    await locks.write_call_record(
        redis,
        call_id,
        {
            "uid": USER,
            "status": "ended",
            "end_reason": "finished",
            "turns": [
                TurnMark(index=0, at=0).model_dump(),
                TurnMark(index=1, at=len(pcm) // 2).model_dump(),
            ],
        },
    )
    claims_as_redis_closed: list[str | None] = []

    @asynccontextmanager
    async def setup():
        try:
            yield _deps(redis, tmp_path, postcall)
        finally:
            claims_as_redis_closed.append(
                await redis.get(locks.postcall_claim_key(call_id))
            )

    relay = Relay(
        max_calls=1, origins=build_cors_params([ORIGIN], AppEnvironment.LOCAL)
    )
    app = relay_app.create_app(relay, setup=setup)

    async with app.router.lifespan_context(app):
        assert await locks.claim_postcall(redis, call_id, "pod-a")
        postcall._start(call_id, postcall.run(call_id, pcm=pcm))
        await asyncio.wait_for(transcribing.wait(), timeout=5)
        # The job never finishes its transcription.
        assert postcall.jobs

    assert claims_as_redis_closed == [None]
    assert postcall.jobs == {}


class LockChangesHands(_Passthrough):
    """The user lock is held by another call when taken, and gone when read."""

    def __init__(self, redis) -> None:
        super().__init__(redis)
        self.churning = True

    async def set(self, name, *args, **kwargs):
        if self.churning and name.startswith("voice:live:"):
            return None
        return await self._redis.set(name, *args, **kwargs)

    async def get(self, name):
        if self.churning and name.startswith("voice:live:"):
            return None
        return await self._redis.get(name)


async def test_a_lock_that_keeps_changing_hands_lets_the_same_token_retry(redis):
    churn = LockChangesHands(redis)
    call_id = tokens.new_call_id()
    first = start(call_id=call_id)
    async with running_relay(churn, FakeVendor([Reply(text="Hi")])) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(first))
        assert await refusal(ws) == ({"type": "error", "error": "unavailable"}, 1013)
        assert not await redis.exists(tokens.spent_key(call_id))
        churn.churning = False
        again = await open_socket(running.url)
        await again.send(json.dumps(first))
        await read_until(again, "ready")
        await again.send(json.dumps({"type": "hangup"}))
        await read_until(again, "ended")


# --- the call's room --------------------------------------------------------


async def test_a_call_runs_its_audio_through_its_room_and_its_events_on_the_socket(
    redis,
):
    vendor = FakeVendor([Reply(text="Hi, I'm Otto, an AI.", audio_s=0.2)])
    async with running_relay(redis, vendor) as running:
        media = running.media
        call_id = tokens.new_call_id()
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start(call_id=call_id)))
        await read_until(ws, "ready")
        await read_until(ws, "turn_complete")
        assert media.rooms[0].claims["video"]["room"] == f"voice-{call_id}"
        await until(lambda: len(media.sources[0].audio) == round(0.2 * 24_000) * 2)
        await ws.send(json.dumps({"type": "hangup"}))
        received: list[str | bytes] = []
        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=5)
            received.append(message)
            if isinstance(message, str) and json.loads(message)["type"] == "ended":
                break
        # Otto's audio never took the socket.
        assert not any(isinstance(message, bytes) for message in received)
        await until(lambda: media.rooms[0].left)
        await running.postcalls_done()


async def test_a_call_whose_room_the_relay_cannot_join_ends_before_it_goes_live(
    redis,
):
    vendor = FakeVendor([Reply(text="Hi", audio_s=0.1)])
    media = FakeMedia(refuse=ConnectionError("the room is full"))
    async with running_relay(redis, vendor, media=media) as running:
        call_id = tokens.new_call_id()
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start(call_id=call_id)))
        ended = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert (ended["type"], ended["reason"]) == ("ended", "room_error")
        await running.postcalls_done()
    record = await locks.read_call(redis, call_id)
    assert record is not None and record.end_reason == "room_error"
    # The person is free to call again, or to type.
    assert await locks.live_call_of(redis, USER) is None


async def test_the_relay_stays_busy_until_ottos_last_words_have_played_in_the_room(
    redis,
):
    vendor = FakeVendor([Reply(text="Hi", audio_s=0.1)])
    async with running_relay(redis, vendor) as running:
        media = running.media
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        await read_until(ws, "turn_complete")
        media.sources[0].playing = asyncio.Event()
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")
        await running.postcalls_done()
        # The call is over and its transcript made, but Otto is still talking
        # in the room: a drain waits for it.
        assert running.relay.held_slots == 0
        assert running.relay.busy
        media.sources[0].playing.set()
        await until(lambda: not running.relay.busy)
        assert media.rooms[0].left
