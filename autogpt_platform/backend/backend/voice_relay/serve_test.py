import asyncio
import json
import signal
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import sentry_sdk
import uvicorn
from sentry_sdk.transport import Transport
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from backend.api.utils.cors import build_cors_params
from backend.util.settings import AppEnvironment
from backend.voice_relay import locks, tokens
from backend.voice_relay.app import Relay, create_app
from backend.voice_relay.conftest import ORIGIN, running_relay, speech, until
from backend.voice_relay.fake_vendor import FakeVendor, Reply
from backend.voice_relay.serve import (
    DrainingServer,
    keep_frame_locals_out_of_sentry,
    strip_frame_locals,
)


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
    # A call still holds its slot, so the drain waits.
    assert relay.reserve_slot() is not None
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


SAID = "what the person said on the call"


def test_an_error_report_from_the_relay_keeps_no_frame_locals():
    def frame(module: str) -> dict:
        return {"module": module, "function": "f", "lineno": 1, "vars": {"t": SAID}}

    event = {
        "exception": {
            "values": [
                {
                    "stacktrace": {
                        "frames": [
                            frame("backend.voice_relay.session"),
                            frame("json.encoder"),
                        ]
                    }
                }
            ]
        },
        "threads": {"values": [{"stacktrace": {"frames": [frame("asyncio")]}}]},
        "stacktrace": {"frames": [frame("logging")]},
    }
    cleaned = strip_frame_locals(event, {})
    assert SAID not in json.dumps(cleaned)
    session_frame = cleaned["exception"]["values"][0]["stacktrace"]["frames"][0]
    assert session_frame == {
        "module": "backend.voice_relay.session",
        "function": "f",
        "lineno": 1,
    }


class _CapturedEvents(Transport):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def capture_envelope(self, envelope) -> None:
        for item in envelope.items:
            if item.headers.get("type") == "event" and item.payload.json:
                self.events.append(item.payload.json)


def _fail_holding(caption: str) -> None:
    raise RuntimeError("the call failed")


def test_the_relay_process_sends_sentry_no_frame_locals():
    transport = _CapturedEvents()
    client = sentry_sdk.Client(
        dsn="https://key@sentry.invalid/1",
        transport=transport,
        include_local_variables=True,
        default_integrations=False,
    )

    def report() -> dict:
        with sentry_sdk.isolation_scope() as scope:
            scope.set_client(client)
            try:
                _fail_holding(SAID)
            except RuntimeError:
                sentry_sdk.capture_exception()
        client.flush()
        return transport.events.pop()

    processors = sentry_sdk.get_global_scope()._event_processors
    assert strip_frame_locals not in processors
    assert SAID in json.dumps(report())
    keep_frame_locals_out_of_sentry()
    try:
        assert SAID not in json.dumps(report())
    finally:
        processors.remove(strip_frame_locals)


async def test_a_call_resumes_on_its_relay_while_the_relay_drains(redis):
    vendor = FakeVendor([Reply(text="Hi", audio_s=0.1)])
    async with running_relay(redis, vendor) as running:
        ws, call_id = await start_call(running.url)
        ready = await read_until(ws, "ready")
        running.server.handle_exit(signal.SIGTERM, None)
        await ws.close()

        again = await connect(
            running.url, origin=ORIGIN, max_size=2**20  # type: ignore[arg-type]
        )
        await again.send(json.dumps({"type": "resume", "ticket": ready["resume"]}))
        assert (await read_until(again, "resumed"))["gap_s"] >= 0
        await again.send(json.dumps({"type": "hangup"}))
        assert (await read_until(again, "ended"))["reason"] == "user_hangup"
        await asyncio.wait_for(running.serving, timeout=10)
    record = await locks.read_call_record(redis, call_id)
    assert record["reattaches"] == "1"


async def test_a_call_starting_as_the_drain_begins_runs_and_the_drain_waits_for_it(
    redis,
):
    vendor = FakeVendor([Reply(text="Hi", audio_s=0.1)])
    # The start holds its slot while it records the call, and the drain, not
    # the short settle after it, is what waits for the call.
    async with running_relay(
        redis, vendor, record_delay_s=0.5, settle_s=0.2
    ) as running:
        ws, call_id = await start_call(running.url)
        await until(lambda: running.relay.held_slots == 1)
        assert running.relay.sessions == {}
        running.server.handle_exit(signal.SIGTERM, None)

        await read_until(ws, "ready")
        await asyncio.sleep(0.3)
        assert not running.server.should_exit
        await ws.send(json.dumps({"type": "hangup"}))
        assert (await read_until(ws, "ended"))["reason"] == "user_hangup"
        await asyncio.wait_for(running.serving, timeout=10)
    assert running.recorded == {"user-1": call_id}


def test_the_relay_process_does_not_load_the_executor():
    # A fresh interpreter: this one has loaded the executor through other tests.
    probe = (
        "import json, sys\n"
        "import backend.voice_relay.serve\n"
        "loaded = sorted(m for m in sys.modules"
        " if m.split('.')[:2] == ['backend', 'executor'])\n"
        "print('EXECUTOR_MODULES=' + json.dumps(loaded))\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert run.returncode == 0, run.stderr[-2000:]
    reported = next(
        line for line in run.stdout.splitlines() if line.startswith("EXECUTOR_MODULES=")
    )
    assert json.loads(reported.split("=", 1)[1]) == []
