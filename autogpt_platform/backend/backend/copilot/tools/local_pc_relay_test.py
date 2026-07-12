from __future__ import annotations

import asyncio
import json

import pytest

from backend.copilot.tools.local_pc_relay import RedisShimRelay
from backend.copilot.tools.local_pc_relay_test_support import FakeRedis, FakeWebSocket
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
