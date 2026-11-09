import asyncio
import json

import httpx
import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from backend.voice_relay import app as relay_app
from backend.voice_relay import locks, tokens
from backend.voice_relay.app import first_name_of
from backend.voice_relay.conftest import ORIGIN, running_relay
from backend.voice_relay.fake_vendor import FakeVendor, Reply

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
    return {"type": "start", "token": tokens.mint_start(user, call_id, "websocket")}


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
    async with running_relay(redis, FakeVendor([Reply("Hi")])) as running:
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
        token = tokens.mint_start(USER, tokens.new_call_id(), "websocket")
        await ws.send(json.dumps({"type": "start", "token": token[:-2] + "AA"}))
        assert await refusal(ws) == ({"type": "error", "error": "token_invalid"}, 1008)


async def test_a_second_live_call_for_the_same_person_is_refused_naming_the_first(
    redis,
):
    async with running_relay(redis, FakeVendor([Reply("Hi")])) as running:
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
    async with running_relay(redis, FakeVendor([Reply("Hi")]), max_calls=1) as running:
        ws = await open_socket(running.url)
        await ws.send(json.dumps(start()))
        await read_until(ws, "ready")
        other = await open_socket(running.url)
        await other.send(json.dumps(start(user="user-2")))
        assert await refusal(other) == ({"type": "error", "error": "busy"}, 1013)
        await ws.send(json.dumps({"type": "hangup"}))
        await read_until(ws, "ended")


async def test_a_dropped_socket_resumes_its_call_with_the_ticket(redis):
    vendor = FakeVendor([Reply("Hi", audio_s=0.1)])
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
    async with running_relay(redis, FakeVendor([Reply("Hi")])) as running:
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
    vendor = FakeVendor([Reply("Hi")])
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
