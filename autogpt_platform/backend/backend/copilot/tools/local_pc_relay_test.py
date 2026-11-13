from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import aclosing

import pytest

from backend.copilot.tools import local_pc_machine
from backend.copilot.tools import local_pc_relay_routing as relay_routing
from backend.copilot.tools.local_pc_relay import RedisShimRelay
from backend.copilot.tools.local_pc_relay_backlog import (
    append_request,
    append_response,
    append_stream_envelope,
)
from backend.copilot.tools.local_pc_relay_protocol import (
    response_stream_key,
    status_stream_key,
    stream_key,
)
from backend.copilot.tools.local_pc_relay_test_support import FakeRedis, FakeWebSocket
from backend.copilot.tools.local_pc_relay_transport import RedisRelayTransport
from backend.copilot.tools.local_pc_shim import (
    LocalPCShim,
    ShimConnectionManager,
    ShimHello,
)


def _hello() -> ShimHello:
    return ShimHello(
        machine_id="machine-1",
        platform="linux",
        arch="x86_64",
        allowed_root="/workspace",
        capabilities=["shell", "files"],
        protocol_version="1.0",
    )


@pytest.mark.asyncio
async def test_relay_rejects_uncorrelated_shim_response() -> None:
    redis = FakeRedis()
    manager = ShimConnectionManager(relay=RedisShimRelay(redis))
    websocket = FakeWebSocket()
    serve = asyncio.create_task(
        manager.serve_websocket(
            "session-1",
            websocket,
            _hello(),
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )
    await manager._relay.wait_for_presence("session-1", timeout=1)
    await websocket.inbound.put(
        json.dumps(
            {
                "type": "COMMAND_RESULT",
                "id": "unknown-request",
                "ts": 0,
                "payload": {
                    "stdout": "",
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="no outstanding request"):
        await asyncio.wait_for(serve, 1)


@pytest.mark.asyncio
async def test_two_process_managers_relay_rpc_and_hello() -> None:
    redis = FakeRedis()
    server_manager = ShimConnectionManager(relay=RedisShimRelay(redis))
    worker_manager = ShimConnectionManager(relay=RedisShimRelay(redis))
    websocket = FakeWebSocket()
    serve = asyncio.create_task(
        server_manager.serve_websocket(
            "session-1",
            websocket,
            _hello(),
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )

    shim = await LocalPCShim.for_session(
        "session-1", manager=worker_manager, connect_timeout=1
    )
    command_task = asyncio.create_task(shim.commands.run("echo relayed"))
    request = json.loads(await asyncio.wait_for(websocket.outbound.get(), 1))
    assert request["type"] == "EXECUTE_COMMAND"
    await websocket.inbound.put(
        json.dumps(
            {
                "type": "COMMAND_RESULT",
                "id": request["id"],
                "ts": 0,
                "payload": {
                    "stdout": "relayed\n",
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                },
            }
        )
    )
    result = await asyncio.wait_for(command_task, 1)
    assert result.stdout == "relayed\n"
    assert shim.machine_id == "machine-1"

    await websocket.close()
    await asyncio.wait_for(serve, 1)
    await shim.kill()


@pytest.mark.asyncio
async def test_worker_clears_cached_relay_session_after_remote_disconnect() -> None:
    redis = FakeRedis()
    server = ShimConnectionManager(relay=RedisShimRelay(redis))
    worker = ShimConnectionManager(relay=RedisShimRelay(redis))
    websocket = FakeWebSocket()
    serve = asyncio.create_task(
        server.serve_websocket(
            "session-1",
            websocket,
            _hello(),
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )
    shim = await LocalPCShim.for_session("session-1", manager=worker, connect_timeout=1)

    assert "session-1" in worker._relay_transports
    assert "session-1" in worker._relay_connection_ids
    assert worker.get_hello("session-1") is not None

    await websocket.close()
    await asyncio.wait_for(serve, 1)
    await asyncio.wait_for(shim.wait_closed(), 2)

    assert "session-1" not in worker._relay_transports
    assert "session-1" not in worker._relay_connection_ids
    assert worker.get_hello("session-1") is None
    await shim.kill()


@pytest.mark.asyncio
async def test_two_workers_receive_only_their_correlated_rpc_response() -> None:
    redis = FakeRedis()
    server = ShimConnectionManager(relay=RedisShimRelay(redis))
    first_worker = ShimConnectionManager(relay=RedisShimRelay(redis))
    second_worker = ShimConnectionManager(relay=RedisShimRelay(redis))
    websocket = FakeWebSocket()
    serve = asyncio.create_task(
        server.serve_websocket(
            "session-1",
            websocket,
            _hello(),
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )
    first_shim = await LocalPCShim.for_session(
        "session-1", manager=first_worker, connect_timeout=1
    )
    second_shim = await LocalPCShim.for_session(
        "session-1", manager=second_worker, connect_timeout=1
    )

    first_task = asyncio.create_task(first_shim.commands.run("echo first"))
    second_task = asyncio.create_task(second_shim.commands.run("echo second"))
    requests = [
        json.loads(await asyncio.wait_for(websocket.outbound.get(), 1))
        for _ in range(2)
    ]
    by_command = {request["payload"]["command"]: request for request in requests}
    second_request = by_command["echo second"]
    await websocket.inbound.put(
        json.dumps(
            {
                "type": "COMMAND_RESULT",
                "id": second_request["id"],
                "ts": 0,
                "payload": {
                    "stdout": "second\n",
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                },
            }
        )
    )
    assert (await asyncio.wait_for(second_task, 1)).stdout == "second\n"
    assert not first_task.done()

    first_request = by_command["echo first"]
    await websocket.inbound.put(
        json.dumps(
            {
                "type": "COMMAND_RESULT",
                "id": first_request["id"],
                "ts": 0,
                "payload": {
                    "stdout": "first\n",
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                },
            }
        )
    )
    assert (await asyncio.wait_for(first_task, 1)).stdout == "first\n"

    await websocket.close()
    await asyncio.wait_for(serve, 1)
    await asyncio.gather(first_shim.kill(), second_shim.kill())


@pytest.mark.asyncio
async def test_replacement_and_cross_process_revocation_close_owner_socket() -> None:
    redis = FakeRedis()
    first_manager = ShimConnectionManager(relay=RedisShimRelay(redis))
    second_manager = ShimConnectionManager(relay=RedisShimRelay(redis))
    revoker = ShimConnectionManager(relay=RedisShimRelay(redis))
    first = FakeWebSocket()
    second = FakeWebSocket()
    first_task = asyncio.create_task(
        first_manager.serve_websocket(
            "session-1",
            first,
            _hello(),
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )
    await first_manager._relay.wait_for_presence("session-1", timeout=1)
    second_task = asyncio.create_task(
        second_manager.serve_websocket(
            "session-1",
            second,
            _hello(),
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )
    revoked = json.loads(await asyncio.wait_for(first.outbound.get(), 2))
    assert revoked["payload"]["reason"] == "another_shim_connected"
    await asyncio.wait_for(first.closed.wait(), 1)
    assert first.close_code == 4427

    assert (
        await revoker.revoke_user_shims(
            "user-1", "autogpt-local-executor", reason="user_revoked"
        )
        == 1
    )
    revoked = json.loads(await asyncio.wait_for(second.outbound.get(), 2))
    assert revoked["payload"]["reason"] == "user_revoked"
    await asyncio.wait_for(second.closed.wait(), 1)
    assert second.close_code == 4428
    await asyncio.gather(first_task, second_task)


@pytest.mark.asyncio
async def test_worker_manager_replaces_cached_transport_after_reconnect() -> None:
    redis = FakeRedis()
    server = ShimConnectionManager(relay=RedisShimRelay(redis))
    worker = ShimConnectionManager(relay=RedisShimRelay(redis))
    first = FakeWebSocket()
    second = FakeWebSocket()
    first_task = asyncio.create_task(
        server.serve_websocket(
            "session-1",
            first,
            _hello(),
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )
    shim = await LocalPCShim.for_session("session-1", manager=worker, connect_timeout=1)
    original_transport = shim._ws

    second_task = asyncio.create_task(
        server.serve_websocket(
            "session-1",
            second,
            _hello(),
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )
    await asyncio.wait_for(first.closed.wait(), 2)
    await worker.wait_for("session-1", timeout=1)

    assert shim._ws is not original_transport
    command_task = asyncio.create_task(shim.commands.run("echo reconnected"))
    request = json.loads(await asyncio.wait_for(second.outbound.get(), 1))
    await second.inbound.put(
        json.dumps(
            {
                "type": "COMMAND_RESULT",
                "id": request["id"],
                "ts": 0,
                "payload": {
                    "stdout": "reconnected\n",
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                },
            }
        )
    )
    assert (await asyncio.wait_for(command_task, 1)).stdout == "reconnected\n"
    await second.close()
    await asyncio.gather(first_task, second_task)
    await shim.kill()


@pytest.mark.asyncio
async def test_a_request_over_the_relay_cap_fails_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request the relay can't take fails fast for its own worker. Ending
    the connection instead would fail every other request on it."""
    monkeypatch.setattr(relay_routing, "_MAX_PENDING_REQUESTS", 1)
    redis = FakeRedis()
    server = ShimConnectionManager(relay=RedisShimRelay(redis))
    first_worker = ShimConnectionManager(relay=RedisShimRelay(redis))
    second_worker = ShimConnectionManager(relay=RedisShimRelay(redis))
    websocket = FakeWebSocket()
    serve = asyncio.create_task(
        server.serve_websocket(
            "session-1",
            websocket,
            _hello(),
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )
    first_shim = await LocalPCShim.for_session(
        "session-1", manager=first_worker, connect_timeout=1
    )
    second_shim = await LocalPCShim.for_session(
        "session-1", manager=second_worker, connect_timeout=1
    )
    first_task = asyncio.create_task(first_shim.commands.run("echo first"))
    first_request = json.loads(await asyncio.wait_for(websocket.outbound.get(), 1))

    with pytest.raises(RuntimeError, match="concurrency cap"):
        await asyncio.wait_for(second_shim.commands.run("echo second"), 5)

    assert websocket.outbound.empty()
    assert not serve.done()
    await websocket.inbound.put(
        json.dumps(
            {
                "type": "COMMAND_RESULT",
                "id": first_request["id"],
                "ts": 0,
                "payload": {
                    "stdout": "first\n",
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                },
            }
        )
    )
    assert (await asyncio.wait_for(first_task, 1)).stdout == "first\n"

    await websocket.close()
    await asyncio.wait_for(serve, 1)
    await asyncio.gather(first_shim.kill(), second_shim.kill())


@pytest.mark.asyncio
async def test_a_request_read_just_before_a_replacement_is_not_sent() -> None:
    """The pump checks its connection is current before each send, not only
    before each read, so a request it read just before the connection was
    replaced never reaches the superseded socket."""
    redis = FakeRedis()
    relay = RedisShimRelay(redis)
    websocket = FakeWebSocket()
    reading = asyncio.Event()
    real_xread = redis.xread

    async def xread(*, streams: dict[str, str], block: int, count: int):
        if any(key.endswith(":requests") for key in streams):
            reading.set()
        return await real_xread(streams=streams, block=block, count=count)

    redis.xread = xread
    serve = asyncio.create_task(
        relay.serve_websocket(
            "session-1",
            websocket,
            hello=_hello().model_dump(mode="json"),
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )
    presence = await relay.wait_for_presence("session-1", timeout=1)
    await asyncio.wait_for(reading.wait(), 1)
    await relay._register_presence(
        "session-1", hello={}, user_id="user-1", client_id="autogpt-local-executor"
    )
    await append_request(
        redis,
        stream_key("session-1", presence.connection_id, "requests"),
        json.dumps({"type": "FILE_READ", "id": "request-1", "ts": 0, "payload": {}}),
        reply_id="a" * 32,
    )

    await asyncio.wait_for(serve, 2)

    sent = [
        json.loads(websocket.outbound.get_nowait())["type"]
        for _ in range(websocket.outbound.qsize())
    ]
    assert sent == ["SESSION_REVOKED"]
    assert websocket.close_code == 4427


def _status_frame(message_id: str, capacity: int, *, pad: int = 0) -> str:
    """A STATUS frame shaped as PROTOCOL.md documents it, ``pad`` bytes longer."""
    frame = {
        "type": "STATUS",
        "id": message_id,
        "ts": 0,
        "version": "2.0",
        "pending_capacity": capacity,
        "payload": {"in_flight": 4 - capacity, "max_concurrent": 4},
    }
    if pad:
        frame["pad"] = "x" * pad
    return json.dumps(frame)


async def _eventually(check: Callable[[], bool], *, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.005)


async def _answer_next_command(websocket: FakeWebSocket, stdout: str) -> None:
    request = json.loads(await asyncio.wait_for(websocket.outbound.get(), 2))
    assert request["type"] == "EXECUTE_COMMAND"
    await websocket.inbound.put(
        json.dumps(
            {
                "type": "COMMAND_RESULT",
                "id": request["id"],
                "ts": 0,
                "payload": {
                    "stdout": stdout,
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": False,
                },
            }
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "asked_before", [True, False], ids=["after-a-request", "before-any-request"]
)
async def test_a_worker_with_nothing_outstanding_hears_the_shims_status(
    asked_before: bool,
) -> None:
    """A worker whose last reading was 0 holds its next request back until a
    response or a STATUS reports a free slot, with nothing outstanding while
    it waits. The connection's STATUS reaches it all the same: first the full
    signal, then the one that lets the held request go."""
    redis = FakeRedis()
    server = ShimConnectionManager(relay=RedisShimRelay(redis))
    worker = ShimConnectionManager(relay=RedisShimRelay(redis))
    websocket = FakeWebSocket()
    serve = asyncio.create_task(
        server.serve_websocket(
            "session-1",
            websocket,
            _hello(),
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )
    shim = await LocalPCShim.for_session("session-1", manager=worker, connect_timeout=1)
    if asked_before:
        earlier = asyncio.create_task(shim.commands.run("echo earlier"))
        await _answer_next_command(websocket, "earlier\n")
        assert (await asyncio.wait_for(earlier, 1)).stdout == "earlier\n"

    await websocket.inbound.put(_status_frame("status-full", 0))
    await _eventually(lambda: shim.pending_capacity == 0)
    held = asyncio.create_task(shim.commands.run("echo held"))
    await asyncio.sleep(0.2)
    assert websocket.outbound.empty()

    await websocket.inbound.put(_status_frame("status-free", 1))
    await _answer_next_command(websocket, "held\n")
    assert (await asyncio.wait_for(held, 1)).stdout == "held\n"

    await websocket.close()
    await asyncio.wait_for(serve, 1)
    await shim.kill()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_first", [True, False], ids=["status-first", "response-first"]
)
async def test_a_transport_reads_status_and_responses_in_the_order_they_were_stored(
    status_first: bool,
) -> None:
    """STATUS has a stream of its own, so one read can return a STATUS and a
    response together. Each says how many slots the shim has free, so the
    later one must be read last. The response is this transport's own and is
    acknowledged; the STATUS stays for the connection's other transports."""
    redis = FakeRedis()
    relay = RedisShimRelay(redis)
    presence, _ = await relay._register_presence(
        "session-1", hello={}, user_id="user-1", client_id="autogpt-local-executor"
    )
    reply_id = "a" * 32
    reply_key = response_stream_key("session-1", presence.connection_id, reply_id)
    status_key = status_stream_key("session-1", presence.connection_id)
    response = json.dumps(
        {"type": "FILE_CONTENTS", "id": "request-1", "ts": 0, "payload": {}}
    )

    async def store_status() -> None:
        await append_stream_envelope(
            redis, status_key, _status_frame("status-1", 1), maxlen=1
        )

    async def store_response() -> None:
        await append_response(redis, reply_key, response)

    for store in (
        (store_status, store_response)
        if status_first
        else (store_response, store_status)
    ):
        await store()

    read: list[str] = []
    transport = RedisRelayTransport(relay, presence, reply_id)
    async with asyncio.timeout(2), aclosing(transport.iter_text()) as frames:
        async for raw in frames:
            read.append(json.loads(raw)["type"])
            if len(read) == 2:
                break

    assert read == (
        ["STATUS", "FILE_CONTENTS"] if status_first else ["FILE_CONTENTS", "STATUS"]
    )
    assert redis.streams[reply_key] == []
    assert len(redis.streams[status_key]) == 1


def _status_bytes_held(redis: FakeRedis) -> int:
    """The bytes of every STATUS frame in any of the fake's streams."""
    return sum(
        len(fields["envelope"].encode())
        for entries in redis.streams.values()
        for _, fields in entries
        if json.loads(fields["envelope"]).get("type") == "STATUS"
    )


async def _answer_directory_listings(websocket: FakeWebSocket) -> None:
    """The shim's side of the folder browser: every folder is empty."""
    while True:
        request = json.loads(await websocket.outbound.get())
        if request["type"] == "DIRECTORY_LIST_REQUEST":
            await websocket.inbound.put(
                json.dumps(
                    {
                        "type": "DIRECTORY_LIST_RESPONSE",
                        "id": request["id"],
                        "ts": 0,
                        "payload": {"entries": []},
                    }
                )
            )


@pytest.mark.asyncio
async def test_status_after_many_answered_machine_requests_is_held_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Local PC routes' machine requests each open a reply stream, read
    their answer and stop reading. STATUS frames the shim sends after twenty
    of them must hold one frame's bytes in Redis, not a copy per answered
    request, and a worker on the connection still reads the latest one."""
    redis = FakeRedis()
    relay = RedisShimRelay(redis)
    monkeypatch.setattr(local_pc_machine, "get_local_pc_relay", lambda: relay)
    websocket = FakeWebSocket()
    serve = asyncio.create_task(
        relay.serve_websocket(
            "machine-scope",
            websocket,
            hello={"connection_kind": "machine", "machine_id": "machine-1"},
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
    )
    presence = await relay.wait_for_presence("machine-scope", timeout=1)
    shim_side = asyncio.create_task(_answer_directory_listings(websocket))

    async def list_folder() -> None:
        answer = await local_pc_machine.machine_rpc(
            presence, "DIRECTORY_LIST_REQUEST", {"path": "/"}, timeout=2
        )
        assert answer["type"] == "DIRECTORY_LIST_RESPONSE"

    try:
        for _ in range(20):
            await list_folder()
        frames = [_status_frame(f"status-{n}", 2, pad=8192) for n in range(3)]
        for raw in frames:
            await websocket.inbound.put(raw)
        # The relay handles the shim's frames in order, so once a request
        # sent after them is answered, the frames have been handled.
        await list_folder()

        assert _status_bytes_held(redis) == len(frames[-1].encode())

        worker = await relay.open_transport(presence)
        async with asyncio.timeout(2), aclosing(worker.iter_text()) as read:
            first = json.loads(await anext(read))
        assert first["id"] == "status-2"
    finally:
        shim_side.cancel()
        await asyncio.gather(shim_side, return_exceptions=True)
        await websocket.close()
        await asyncio.wait_for(serve, 1)
