import asyncio
import json
import signal

import httpx
import pytest
import uvicorn
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from backend.api.utils.cors import build_cors_params
from backend.util.settings import AppEnvironment
from backend.voice_relay import locks, tokens
from backend.voice_relay.app import Relay, create_app
from backend.voice_relay.conftest import (
    ORIGIN,
    ScriptedTranscriber,
    running_relay,
    speech,
)
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.serve import DrainingServer


async def start_call(url: str, user: str = "user-1") -> tuple[ClientConnection, str]:
    ws = await connect(url, origin=ORIGIN, max_size=2**20)  # type: ignore[arg-type]
    call_id = tokens.new_call_id()
    token = tokens.mint_start(user, call_id, "websocket")
    await ws.send(json.dumps({"type": "start", "token": token, "leg": "websocket"}))
    return ws, call_id


async def read_until(ws: ClientConnection, kind: str, timeout: float = 10.0) -> dict:
    async def read() -> dict:
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                continue
            event = json.loads(message)
            if event["type"] == kind:
                return event

    return await asyncio.wait_for(read(), timeout=timeout)


async def test_sigterm_lets_a_live_call_finish_then_the_server_exits(redis):
    vendor = FakeVendor(
        [
            Reply(text="Hi", audio_s=0.2),
            Reply(text="Bye!", heard="I bake bread", first_task="Bake"),
        ]
    )
    async with running_relay(redis, vendor) as running:
        ws, call_id = await start_call(running.url)
        await read_until(ws, "turn_complete")

        running.server.handle_exit(signal.SIGTERM, None)

        async with httpx.AsyncClient(base_url=running.http) as http:
            ready = await http.get("/readyz")
        assert ready.status_code == 503
        assert ready.json() == {"ready": False, "live_calls": 1}

        late, _ = await start_call(running.url, user="user-2")
        assert json.loads(await late.recv()) == {"type": "error", "error": "draining"}
        with pytest.raises(ConnectionClosed) as closed:
            await late.recv()
        assert closed.value.rcvd is not None and closed.value.rcvd.code == 1013

        for frame in speech(0.3):
            await ws.send(frame)
        ended = await read_until(ws, "ended")
        assert ended["reason"] == "finished"
        # The server exits on its own once its last call has ended.
        await asyncio.wait_for(running.serving, timeout=10)
    record = await locks.read_call_record(redis, call_id)
    assert record["end_reason"] == "finished"


async def test_a_call_still_live_at_the_drain_limit_ends_as_server_restart(redis):
    vendor = FakeVendor([Reply(text="Hi", audio_s=0.1)])
    async with running_relay(redis, vendor, drain_max_s=0.3) as running:
        ws, call_id = await start_call(running.url)
        await read_until(ws, "ready")
        running.server.handle_exit(signal.SIGTERM, None)
        ended = await read_until(ws, "ended")
        assert ended["reason"] == "server_restart"
        with pytest.raises(ConnectionClosed) as closed:
            await ws.recv()
        assert closed.value.rcvd is not None and closed.value.rcvd.code == 1012
        await asyncio.wait_for(running.serving, timeout=10)
    record = await locks.read_call_record(redis, call_id)
    assert record["end_reason"] == "server_restart"
    assert await locks.live_call_of(redis, "user-1") is None


async def test_a_drain_with_no_live_calls_exits_at_once():
    relay = Relay(
        max_calls=1, origins=build_cors_params([ORIGIN], AppEnvironment.LOCAL)
    )
    relay.loop = asyncio.get_running_loop()
    server = DrainingServer(
        uvicorn.Config(create_app(relay)), relay=relay, drain_max_s=30.0
    )
    server.handle_exit(signal.SIGTERM, None)
    assert relay.draining
    assert not server.should_exit
    await asyncio.sleep(0)
    assert server.drain_task is not None
    # With no calls left, the drain ends at once.
    await asyncio.wait_for(server.drain_task, timeout=5)
    assert server.should_exit


async def test_a_second_sigterm_exits_at_once():
    relay = Relay(
        max_calls=1, origins=build_cors_params([ORIGIN], AppEnvironment.LOCAL)
    )
    relay.loop = asyncio.get_running_loop()
    server = DrainingServer(
        uvicorn.Config(create_app(relay)), relay=relay, drain_max_s=300.0
    )
    relay.sessions["call"] = object()  # type: ignore[assignment]
    server.handle_exit(signal.SIGTERM, None)
    await asyncio.sleep(0)
    assert relay.draining and not server.should_exit
    server.handle_exit(signal.SIGTERM, None)
    assert server.should_exit
    assert server.drain_task is not None
    server.drain_task.cancel()


def test_sigint_exits_without_draining():
    relay = Relay(
        max_calls=1, origins=build_cors_params([ORIGIN], AppEnvironment.LOCAL)
    )
    server = DrainingServer(
        uvicorn.Config(create_app(relay)), relay=relay, drain_max_s=300.0
    )
    server.handle_exit(signal.SIGINT, None)
    assert server.should_exit
    assert not relay.draining


async def test_sigterm_waits_for_a_calls_post_call_job_before_exiting(redis):
    class HeldTranscriber(ScriptedTranscriber):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()

        async def __call__(self, wav: bytes, *, first_name: str | None):
            await self.release.wait()
            return await super().__call__(wav, first_name=first_name)

    transcriber = HeldTranscriber()
    vendor = FakeVendor(
        [
            Reply(text="Hi", audio_s=0.1),
            Reply(text="Bye!", heard="I bake bread", first_task="Bake"),
        ]
    )
    async with running_relay(redis, vendor, transcriber=transcriber) as running:
        ws, call_id = await start_call(running.url)
        await read_until(ws, "turn_complete")
        for frame in speech(0.3):
            await ws.send(frame)
        await read_until(ws, "ended")

        running.server.handle_exit(signal.SIGTERM, None)
        await asyncio.sleep(0.5)
        # The call is over but its words are not handed off yet.
        assert not running.serving.done()
        assert running.hand_offs.calls == []

        transcriber.release.set()
        await asyncio.wait_for(running.serving, timeout=10)
        assert [record.call_id for record in running.hand_offs.calls] == [call_id]
