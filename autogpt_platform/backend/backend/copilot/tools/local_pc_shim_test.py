"""Tests for the LocalPCShim adapter — duck-type contract + cross-OS kwargs."""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from .local_pc_relay_protocol import RelayPresence
from .local_pc_shim import (
    _IDEMPOTENT_OPS,
    COMMAND_TIMEOUT_SECONDS,
    MAX_FILE_SIZE_BYTES,
    ComputerUseDriver,
    LocalPCShim,
    OpUnconfirmedError,
    ShimComputerUseError,
    ShimConnectionManager,
    ShimFileTooLargeError,
    ShimHello,
    ShimOverloadedError,
    ShimProtocolError,
    ShimRecordingError,
    WriteUnconfirmedError,
    _CommandsProxy,
    _ComputerProxy,
    _FilesProxy,
    _RecordingProxy,
    _RpcAttemptFailed,
)


@pytest.mark.asyncio
async def test_correlated_response_requires_exact_response_type_and_id() -> None:
    shim = LocalPCShim.__new__(LocalPCShim)
    future = asyncio.get_running_loop().create_future()
    shim._pending = {"request-1": (frozenset({"FILE_CONTENTS"}), future)}
    shim._streaming = {}
    shim._recording_steps = {}
    shim._pending_capacity = None

    shim._process_raw_message(
        json.dumps(
            {
                "type": "ACK",
                "id": "request-1",
                "payload": {"ok": True},
            }
        )
    )

    with pytest.raises(ShimProtocolError):
        await future

    other_future = asyncio.get_running_loop().create_future()
    shim._pending = {"request-2": (frozenset({"FILE_CONTENTS"}), other_future)}
    shim._process_raw_message(
        json.dumps(
            {
                "type": "FILE_CONTENTS",
                "id": "wrong-id",
                "payload": {"content": "secret"},
            }
        )
    )
    assert not other_future.done()
    other_future.cancel()


@pytest.mark.asyncio
async def test_cancelled_rpc_removes_pending_future() -> None:
    sent = asyncio.Event()

    class _SendOnlyTransport:
        async def send_text(self, _data: str) -> None:
            sent.set()

        async def iter_text(self):
            if False:
                yield ""

        async def close(self) -> None:
            return None

    shim = LocalPCShim.__new__(LocalPCShim)
    shim._ws = _SendOnlyTransport()
    shim._pending = {}
    request = asyncio.create_task(
        shim._send_and_wait("FILE_READ", {"path": "inside.txt"}, timeout=30)
    )
    await sent.wait()
    assert len(shim._pending) == 1

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert shim._pending == {}


def test_recording_fetch_is_idempotent_but_apply_review_is_not() -> None:
    assert "RECORDING_FETCH" in _IDEMPOTENT_OPS
    assert "APPLY_RECORDING_REVIEW" not in _IDEMPOTENT_OPS


def _make_shim_with_rpc(rpc_return: dict) -> LocalPCShim:
    """Build a LocalPCShim with _rpc stubbed to return a canned payload."""
    shim = LocalPCShim.__new__(LocalPCShim)
    shim.sandbox_id = "test-session"
    shim.allowed_root = "/Users/test/workspace"
    shim.machine_id = "machine-uuid"
    shim.platform = "darwin"
    shim.arch = "arm64"
    shim.capabilities = ["shell", "files"]
    shim._rpc = AsyncMock(return_value=rpc_return)
    shim.files = _FilesProxy(shim)
    shim.commands = _CommandsProxy(shim)
    shim.computer = _ComputerProxy(shim)
    shim.computer_use_driver = None
    shim.computer_use_tools = []
    shim._connection_generation = 1
    return shim


class TestShimHello:
    def test_from_payload_extracts_all_fields(self):
        hello = ShimHello.from_payload(
            {
                "machine_id": "abc-123",
                "platform": "windows",
                "arch": "x86_64",
                "shim_version": "0.1.0",
                "allowed_root": "C:\\workspace",
                "capabilities": ["shell", "files", "computer_use"],
                "screen_resolution": [1920, 1080],
                "local_llm_models": ["llama3.2:3b"],
                "hardware_devices": [{"type": "serial", "port": "COM3"}],
            }
        )
        assert hello.machine_id == "abc-123"
        assert hello.platform == "windows"
        assert hello.arch == "x86_64"
        assert hello.allowed_root == "C:\\workspace"
        assert hello.capabilities == ["shell", "files", "computer_use"]
        assert hello.screen_resolution == (1920, 1080)

    def test_from_payload_rejects_missing_identity_fields(self):
        with pytest.raises(ValueError):
            ShimHello.from_payload({})

    def test_from_payload_rejects_malformed_screen_resolution(self):
        hello = ShimHello.from_payload(
            {
                "machine_id": "abc-123",
                "platform": "windows",
                "arch": "x86_64",
                "allowed_root": "C:\\workspace",
                "screen_resolution": [1920],
            }
        )
        assert hello.screen_resolution is None

    def test_from_payload_reads_the_driver_and_sorts_its_tools(self):
        hello = ShimHello.from_payload(
            {
                "machine_id": "abc-123",
                "platform": "linux",
                "arch": "x86_64",
                "allowed_root": "/workspace",
                "capabilities": ["files", "computer_use"],
                "computer_use_driver": {
                    "name": "cua-driver",
                    "version": "0.38.0",
                    "runtime": "direct",
                },
                "computer_use_tools": ["type_text", "click", "click"],
            }
        )

        assert hello.computer_use_driver == ComputerUseDriver(
            name="cua-driver", version="0.38.0", runtime="direct"
        )
        assert hello.computer_use_driver.label == "cua-driver 0.38.0"
        assert hello.computer_use_tools == ["click", "type_text"]

    @pytest.mark.parametrize(
        "field",
        [
            {"computer_use_tools": ["Click"]},
            {"computer_use_tools": ["click; rm"]},
            {"computer_use_tools": ["x"] * 129},
            {"computer_use_driver": "cua-driver"},
            {"computer_use_driver": {"name": "cua-driver", "version": "0.38.0"}},
            {
                "computer_use_driver": {
                    "name": "cua\ndriver",
                    "version": "0.38.0",
                    "runtime": "direct",
                }
            },
        ],
    )
    def test_from_payload_rejects_a_malformed_computer_use_scope(self, field):
        with pytest.raises(ValueError):
            ShimHello.from_payload(
                {
                    "machine_id": "abc-123",
                    "platform": "linux",
                    "arch": "x86_64",
                    "allowed_root": "/workspace",
                    **field,
                }
            )


class TestShimConnectionManager:
    @pytest.mark.asyncio
    async def test_close_missing_session_never_allocates(self) -> None:
        manager = ShimConnectionManager(relay=self.NoPresenceRelay())
        manager.wait_for = AsyncMock()
        await manager.close_existing_session("missing")
        manager.wait_for.assert_not_awaited()
        assert not manager._shims
        assert not manager._waiters

    @pytest.mark.asyncio
    async def test_close_existing_session_closes_registered_transport(self) -> None:
        manager = ShimConnectionManager(relay=self.NoPresenceRelay())
        websocket = MagicMock()
        websocket.close = AsyncMock()
        manager.register("session-1", websocket, ShimHello())

        await manager.close_existing_session("session-1")

        websocket.close.assert_awaited_once()
        assert manager.get("session-1") is None
        assert manager.get_hello("session-1") is None
        assert not manager._shims

    class NoPresenceRelay:
        def __init__(self):
            self.revocations: list[tuple[str, str | None, str]] = []

        async def get_presence(self, session_id: str):
            return None

        async def wait_for_presence(self, session_id: str, *, timeout: float):
            await asyncio.sleep(timeout)
            raise TimeoutError

        async def open_transport(self, presence):
            raise AssertionError("no presence")

        async def serve_websocket(self, *args, **kwargs):
            raise AssertionError("not used")

        async def revoke_owner(
            self, user_id: str, client_id: str | None, *, reason: str
        ):
            self.revocations.append((user_id, client_id, reason))
            return 0

    def test_register_stores_hello_alongside_ws(self):
        manager = ShimConnectionManager()
        ws = MagicMock()
        hello = ShimHello(machine_id="m1", platform="linux", allowed_root="/home/u/ws")
        manager.register("session-1", ws, hello)
        assert manager.get("session-1") is ws
        assert manager.get_hello("session-1") is hello

    def test_unregister_clears_hello(self):
        manager = ShimConnectionManager()
        manager.register("s1", MagicMock(), ShimHello(machine_id="m1"))
        manager.unregister("s1")
        assert manager.get("s1") is None
        assert manager.get_hello("s1") is None

    @pytest.mark.asyncio
    async def test_for_session_reuses_single_receive_loop_owner(self):
        release = asyncio.Event()

        class BlockingWebSocket:
            async def iter_text(self):
                await release.wait()
                if False:
                    yield ""

            async def close(self):
                release.set()

        manager = ShimConnectionManager()
        websocket = BlockingWebSocket()
        manager.register("s1", websocket, ShimHello(machine_id="m1"))

        first = manager.get_or_create_shim("s1")
        second = await LocalPCShim.for_session("s1", manager=manager)

        assert second is first
        release.set()
        await first.wait_closed()
        assert manager.get("s1") is None

    @pytest.mark.asyncio
    async def test_wait_for_observes_concurrent_direct_registration(self):
        manager = ShimConnectionManager(relay=self.NoPresenceRelay())
        waiter = asyncio.create_task(manager.wait_for("direct", timeout=30))
        await asyncio.sleep(0.01)
        websocket = MagicMock()
        manager.register("direct", websocket, ShimHello(machine_id="m1"))

        assert await waiter is websocket

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing_operation", ["send", "close"])
    async def test_revocation_reports_direct_transport_failure(
        self, failing_operation: str
    ):
        relay = self.NoPresenceRelay()
        manager = ShimConnectionManager(relay=relay)
        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        websocket.close = AsyncMock()
        operation = (
            websocket.send_text if failing_operation == "send" else websocket.close
        )
        operation.side_effect = ConnectionError(f"{failing_operation} failed")
        manager.register(
            "direct-session",
            websocket,
            user_id="user-1",
            client_id="autogpt-local-executor",
        )

        with pytest.raises(RuntimeError, match="session revocation"):
            await manager.revoke_user_shims(
                "user-1", "autogpt-local-executor", reason="user_revoked"
            )

        websocket.send_text.assert_awaited_once()
        websocket.close.assert_awaited_once_with(code=4428, reason="Token revoked")
        assert relay.revocations == [
            ("user-1", "autogpt-local-executor", "user_revoked")
        ]
        if failing_operation == "close":
            assert manager.get("direct-session") is websocket
        else:
            assert manager.get("direct-session") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("hanging_operation", ["send", "close"])
    async def test_revocation_times_out_hanging_direct_transport(
        self, hanging_operation: str, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "backend.copilot.tools.local_pc_shim._REVOCATION_OPERATION_TIMEOUT_SECONDS",
            0.01,
        )
        relay = self.NoPresenceRelay()
        manager = ShimConnectionManager(relay=relay)
        never_finishes = asyncio.Event()

        async def hang(*_args: object, **_kwargs: object) -> None:
            await never_finishes.wait()

        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        websocket.close = AsyncMock()
        healthy_websocket = MagicMock()
        healthy_websocket.send_text = AsyncMock()
        healthy_websocket.close = AsyncMock()
        operation = (
            websocket.send_text if hanging_operation == "send" else websocket.close
        )
        operation.side_effect = hang
        manager.register(
            "direct-session",
            websocket,
            user_id="user-1",
            client_id="autogpt-local-executor",
        )
        manager.register(
            "healthy-session",
            healthy_websocket,
            user_id="user-1",
            client_id="autogpt-local-executor",
        )

        with pytest.raises(RuntimeError, match="session revocation"):
            await asyncio.wait_for(
                manager.revoke_user_shims(
                    "user-1", "autogpt-local-executor", reason="user_revoked"
                ),
                timeout=0.2,
            )

        websocket.send_text.assert_awaited_once()
        websocket.close.assert_awaited_once_with(code=4428, reason="Token revoked")
        healthy_websocket.send_text.assert_awaited_once()
        healthy_websocket.close.assert_awaited_once_with(
            code=4428, reason="Token revoked"
        )
        assert relay.revocations == [
            ("user-1", "autogpt-local-executor", "user_revoked")
        ]
        assert manager.get("healthy-session") is None

    @pytest.mark.asyncio
    async def test_revocation_times_out_hanging_relay(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "backend.copilot.tools.local_pc_shim._REVOCATION_OPERATION_TIMEOUT_SECONDS",
            0.01,
        )
        relay_started = asyncio.Event()
        never_finishes = asyncio.Event()

        class HangingRelay(self.NoPresenceRelay):
            async def revoke_owner(
                self, user_id: str, client_id: str | None, *, reason: str
            ) -> int:
                relay_started.set()
                await never_finishes.wait()
                return 0

        manager = ShimConnectionManager(relay=HangingRelay())

        with pytest.raises(RuntimeError, match="session revocation"):
            await asyncio.wait_for(
                manager.revoke_user_shims(
                    "user-1", "autogpt-local-executor", reason="user_revoked"
                ),
                timeout=0.2,
            )

        assert relay_started.is_set()

    @pytest.mark.asyncio
    async def test_wait_for_uses_one_timeout_budget(self):
        manager = ShimConnectionManager(relay=self.NoPresenceRelay())
        started = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError, match="did not connect within"):
            await manager.wait_for("missing", timeout=0.03)
        elapsed = asyncio.get_running_loop().time() - started

        # 10x the budget: room for a stalled runner, and still far short of a
        # wait that ignored the budget.
        assert elapsed < 0.3

    @pytest.mark.asyncio
    async def test_cached_presence_lookup_cannot_miss_direct_registration(self):
        lookup_started = asyncio.Event()
        release_lookup = asyncio.Event()

        class RacingRelay(TestShimConnectionManager.NoPresenceRelay):
            async def get_presence(self, session_id: str):
                lookup_started.set()
                await release_lookup.wait()
                return None

        manager = ShimConnectionManager(relay=RacingRelay())
        stale_transport = MagicMock()
        stale_transport.close = AsyncMock()
        manager._relay_transports["direct"] = stale_transport
        manager._relay_connection_ids["direct"] = "old-connection"
        waiter = asyncio.create_task(manager.wait_for("direct", timeout=30))
        await lookup_started.wait()

        direct = MagicMock()
        manager.register("direct", direct, ShimHello(machine_id="new"))
        release_lookup.set()

        assert await asyncio.wait_for(waiter, 30) is direct
        stale_transport.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cached_presence_lookup_uses_original_timeout_budget(self):
        class SlowCachedRelay(TestShimConnectionManager.NoPresenceRelay):
            async def get_presence(self, session_id: str):
                await asyncio.sleep(1)
                return None

        manager = ShimConnectionManager(relay=SlowCachedRelay())
        manager._relay_transports["stale"] = MagicMock()
        manager._relay_connection_ids["stale"] = "old-connection"
        started = asyncio.get_running_loop().time()

        with pytest.raises(TimeoutError):
            await manager.wait_for("stale", timeout=0.03)

        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 0.3

    @pytest.mark.asyncio
    async def test_transport_open_uses_original_timeout_budget(self):
        class SlowOpenRelay(TestShimConnectionManager.NoPresenceRelay):
            async def wait_for_presence(self, session_id: str, *, timeout: float):
                return RelayPresence(
                    session_id=session_id,
                    connection_id="connection-1",
                    user_id="user-1",
                    client_id="client-1",
                    expires_at=9999999999,
                )

            async def open_transport(self, presence):
                await asyncio.sleep(1)
                raise AssertionError("open should be cancelled by the timeout")

        manager = ShimConnectionManager(relay=SlowOpenRelay())
        started = asyncio.get_running_loop().time()

        with pytest.raises(TimeoutError):
            await manager.wait_for("slow-open", timeout=0.03)

        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 0.3

    @pytest.mark.asyncio
    async def test_duplicate_session_revokes_old_connection_without_losing_new(self):
        class ManagedWebSocket:
            def __init__(self):
                self.closed = asyncio.Event()
                self.sent: list[str] = []
                self.close_code: int | None = None

            async def iter_text(self):
                await self.closed.wait()
                if False:
                    yield ""

            async def send_text(self, message: str):
                self.sent.append(message)

            async def close(self, *, code: int = 1000, reason: str = ""):
                self.close_code = code
                self.closed.set()

        manager = ShimConnectionManager()
        old_websocket = ManagedWebSocket()
        new_websocket = ManagedWebSocket()
        manager.register("s1", old_websocket, ShimHello(machine_id="old"))
        shim = manager.get_or_create_shim("s1")
        old_route = asyncio.create_task(shim.wait_closed())
        await asyncio.sleep(0)

        manager.register("s1", new_websocket, ShimHello(machine_id="new"))
        await asyncio.wait_for(old_route, timeout=1)

        assert old_websocket.close_code == 4427
        revocation = json.loads(old_websocket.sent[0])
        assert revocation["type"] == "SESSION_REVOKED"
        assert revocation["payload"]["reason"] == "another_shim_connected"
        assert manager.get("s1") is new_websocket
        assert manager.get_or_create_shim("s1") is shim

        await new_websocket.close()
        await shim.wait_closed()


class TestDisconnectCleanup:
    @pytest.mark.asyncio
    async def test_disconnect_fails_pending_futures_without_timeout(self):
        shim = LocalPCShim.__new__(LocalPCShim)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict] = loop.create_future()
        shim._pending = {"wire-1": (frozenset({"ACK"}), future)}
        shim._streaming = {}
        shim._recording_steps = {}

        shim._fail_in_flight(ConnectionError("connection lost"))

        with pytest.raises(ConnectionError, match="connection lost"):
            await future
        assert shim._pending == {}

    def test_disconnect_unblocks_stream_and_recording_consumers(self):
        shim = LocalPCShim.__new__(LocalPCShim)
        stream_queue: asyncio.Queue[dict] = asyncio.Queue()
        recording_queue: asyncio.Queue = asyncio.Queue()
        shim._pending = {}
        shim._streaming = {"stream-1": stream_queue}
        shim._recording_steps = {"rec-1": recording_queue}

        shim._fail_in_flight(ConnectionError("connection lost"))

        assert stream_queue.get_nowait()["payload"]["code"] == "CONNECTION_LOST"
        assert recording_queue.get_nowait() is None
        assert shim._recording_steps == {}


class TestFilesReadFormatBytes:
    """The platform calls sandbox.files.read(path, format="bytes") today.
    Without this contract LocalPCShim crashes the first time it's used."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_format_bytes_returns_decoded_bytes(self):
        raw = b"\x00\x01\x02hello"
        encoded = base64.b64encode(raw).decode("ascii")
        shim = _make_shim_with_rpc({"payload": {"content": encoded}})
        result = await shim.files.read("/Users/test/workspace/file.bin", format="bytes")
        assert isinstance(result, bytes)
        assert result == raw

    @pytest.mark.asyncio(loop_scope="session")
    async def test_format_bytes_sends_base64_encoding_on_wire(self):
        raw = b"abc"
        encoded = base64.b64encode(raw).decode("ascii")
        shim = _make_shim_with_rpc({"payload": {"content": encoded}})
        await shim.files.read("/Users/test/workspace/f.bin", format="bytes")
        shim._rpc.assert_awaited_once_with(
            "FILE_READ", {"path": "/Users/test/workspace/f.bin", "encoding": "base64"}
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_format_text_returns_str(self):
        shim = _make_shim_with_rpc({"payload": {"content": "hello\n"}})
        result = await shim.files.read("/Users/test/workspace/f.txt")
        assert isinstance(result, str)
        assert result == "hello\n"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_read_error_raises_oserror(self):
        shim = _make_shim_with_rpc({"type": "ERROR", "payload": {"message": "denied"}})
        with pytest.raises(OSError, match="denied"):
            await shim.files.read("/etc/passwd")


def _contents(data: bytes) -> dict:
    return {
        "type": "FILE_CONTENTS",
        "payload": {"content": base64.b64encode(data).decode("ascii")},
    }


_TOO_LARGE = {
    "type": "ERROR",
    "payload": {"code": "FILE_TOO_LARGE", "message": "over the limit"},
}


class TestFilesReadCapped:
    """``read_capped`` reads no more than the caller's limit plus one byte,
    the way ``read_sandbox_file`` reads an E2B file."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_cap_under_the_file_limit_is_asked_for_as_a_range(self):
        shim = _make_shim_with_rpc(_contents(b"x" * 11))

        data = await shim.files.read_capped("/Users/test/workspace/big.png", 10)

        assert data == b"x" * 11
        shim._rpc.assert_awaited_once_with(
            "FILE_READ",
            {
                "path": "/Users/test/workspace/big.png",
                "encoding": "base64",
                "offset": 0,
                "length": 11,
            },
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_cap_over_the_file_limit_reads_whole_under_the_companions_limit(
        self,
    ):
        shim = _make_shim_with_rpc(_contents(b"abc"))

        data = await shim.files.read_capped(
            "/Users/test/workspace/f.txt", MAX_FILE_SIZE_BYTES
        )

        assert data == b"abc"
        shim._rpc.assert_awaited_once_with(
            "FILE_READ", {"path": "/Users/test/workspace/f.txt", "encoding": "base64"}
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_companion_with_a_lower_limit_still_reads_a_small_file(self):
        shim = _make_shim_with_rpc({})
        shim._rpc = AsyncMock(side_effect=[_TOO_LARGE, _contents(b"small")])

        data = await shim.files.read_capped("/Users/test/workspace/f.png", 1000)

        assert data == b"small"
        assert [call.args[1] for call in shim._rpc.await_args_list] == [
            {
                "path": "/Users/test/workspace/f.png",
                "encoding": "base64",
                "offset": 0,
                "length": 1001,
            },
            {"path": "/Users/test/workspace/f.png", "encoding": "base64"},
        ]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_file_over_the_companions_limit_is_refused(self):
        shim = _make_shim_with_rpc(_TOO_LARGE)

        with pytest.raises(ShimFileTooLargeError):
            await shim.files.read_capped(
                "/Users/test/workspace/huge.bin", MAX_FILE_SIZE_BYTES
            )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_other_errors_are_not_retried(self):
        shim = _make_shim_with_rpc(
            {"type": "ERROR", "payload": {"code": "PATH_NOT_FOUND", "message": "gone"}}
        )

        with pytest.raises(OSError) as raised:
            await shim.files.read_capped("/Users/test/workspace/gone.txt", 10)

        assert not isinstance(raised.value, ShimFileTooLargeError)
        shim._rpc.assert_awaited_once()


class TestFilesWriteBytes:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_bytes_content_is_base64_encoded(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.files.write("/Users/test/workspace/f.bin", b"\x00\xff")
        call_payload = shim._rpc.await_args.args[1]
        assert call_payload["encoding"] == "base64"
        assert base64.b64decode(call_payload["content"]) == b"\x00\xff"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_str_content_passes_through_as_utf8(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.files.write("/Users/test/workspace/f.txt", "hello")
        call_payload = shim._rpc.await_args.args[1]
        assert call_payload["encoding"] == "utf-8"
        assert call_payload["content"] == "hello"


class TestCommandsRunShellSelector:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_default_shell_is_auto(self):
        shim = _make_shim_with_rpc(
            {"payload": {"stdout": "ok", "stderr": "", "exit_code": 0}}
        )
        await shim.commands.run("ls")
        payload = shim._rpc.await_args.args[1]
        assert payload["command"] == "ls"
        assert payload["shell"] == "auto"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_explicit_shell_passed_through(self):
        shim = _make_shim_with_rpc(
            {"payload": {"stdout": "", "stderr": "", "exit_code": 0}}
        )
        await shim.commands.run("Get-Process", shell="pwsh")
        payload = shim._rpc.await_args.args[1]
        assert payload["shell"] == "pwsh"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_argv_form_skips_shell_field(self):
        shim = _make_shim_with_rpc(
            {"payload": {"stdout": "", "stderr": "", "exit_code": 0}}
        )
        await shim.commands.run(argv=["python", "-c", "print(1)"])
        payload = shim._rpc.await_args.args[1]
        assert payload["argv"] == ["python", "-c", "print(1)"]
        assert "shell" not in payload
        assert "command" not in payload

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_command_or_argv_raises(self):
        shim = _make_shim_with_rpc({"payload": {}})
        with pytest.raises(ValueError):
            await shim.commands.run()


class TestCommandsRunE2BKwargs:
    """The platform passes E2B-style kwargs (`envs=`, `timeout=`) — adapter
    must translate them transparently or platform code will break."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_envs_kwarg_translates_to_wire_env(self):
        shim = _make_shim_with_rpc(
            {"payload": {"stdout": "", "stderr": "", "exit_code": 0}}
        )
        await shim.commands.run("printenv", envs={"FOO": "bar"})
        payload = shim._rpc.await_args.args[1]
        assert payload["env"] == {"FOO": "bar"}
        assert "envs" not in payload

    @pytest.mark.asyncio(loop_scope="session")
    async def test_timeout_kwarg_translates_to_wire_timeout_seconds(self):
        shim = _make_shim_with_rpc(
            {"payload": {"stdout": "", "stderr": "", "exit_code": 0}}
        )
        await shim.commands.run("sleep 1", timeout=5)
        payload = shim._rpc.await_args.args[1]
        assert payload["timeout_seconds"] == 5
        assert "timeout" not in payload

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize(
        ("requested", "deadline"),
        [
            (5, 5),
            (None, COMMAND_TIMEOUT_SECONDS),
            (120, COMMAND_TIMEOUT_SECONDS),
        ],
    )
    async def test_rpc_outlasts_the_companions_command_deadline(
        self, requested: int | None, deadline: int
    ):
        """The companion stops a command at the smaller of the requested
        timeout and the HELLO_ACK cap, then still has to report it. An RPC that
        gave up at that moment would turn every timed-out command into an
        unconfirmed side effect."""
        shim = _make_shim_with_rpc(
            {"payload": {"stdout": "", "stderr": "", "exit_code": 0}}
        )
        await shim.commands.run("sleep 1", timeout=requested)
        rpc_timeout = shim._rpc.await_args.kwargs["timeout"]
        assert deadline < rpc_timeout <= deadline + 30


class TestLocalPCShimHelloAttributes:
    """LocalPCShim surfaces HELLO data as attributes for cross-OS dispatch."""

    def test_attributes_default_when_no_hello(self):
        shim = LocalPCShim.__new__(LocalPCShim)
        hello = ShimHello()
        shim.sandbox_id = "s1"
        shim.machine_id = hello.machine_id
        shim.platform = hello.platform
        shim.arch = hello.arch
        shim.allowed_root = hello.allowed_root
        shim.capabilities = hello.capabilities
        assert shim.allowed_root == ""
        assert shim.capabilities == []

    def test_attributes_set_from_hello(self):
        shim = LocalPCShim.__new__(LocalPCShim)
        hello = ShimHello(
            machine_id="m", platform="windows", arch="x86_64", allowed_root="C:\\ws"
        )
        shim.sandbox_id = "s1"
        shim.machine_id = hello.machine_id
        shim.platform = hello.platform
        shim.arch = hello.arch
        shim.allowed_root = hello.allowed_root
        assert shim.platform == "windows"
        assert shim.arch == "x86_64"
        assert shim.allowed_root == "C:\\ws"


class TestFilesStat:
    """FILE_STAT replaces shell `readlink -f` / `stat` / `test -e` for cross-OS."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_stat_returns_payload(self):
        shim = _make_shim_with_rpc(
            {
                "payload": {
                    "exists": True,
                    "is_file": True,
                    "size_bytes": 42,
                    "mode": "0644",
                }
            }
        )
        result = await shim.files.stat("/Users/test/workspace/f.txt")
        assert result["exists"] is True
        assert result["size_bytes"] == 42

    @pytest.mark.asyncio(loop_scope="session")
    async def test_stat_default_follow_symlinks_true(self):
        shim = _make_shim_with_rpc({"payload": {"exists": False}})
        await shim.files.stat("/Users/test/workspace/f.txt")
        payload = shim._rpc.await_args.args[1]
        assert payload["follow_symlinks"] is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_stat_error_raises_oserror(self):
        shim = _make_shim_with_rpc({"type": "ERROR", "payload": {"message": "denied"}})
        with pytest.raises(OSError, match="denied"):
            await shim.files.stat("/etc/passwd")


# ---------------------------------------------------------------------------
# _ComputerProxy — wire-op contract for computer-use surface
# ---------------------------------------------------------------------------


class TestRecordingProxy:
    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (TimeoutError("RPC timed out (retry also failed)"), "COMMAND_TIMEOUT"),
            (OpUnconfirmedError("STOP_RECORDING", "id-1"), "OP_UNCONFIRMED"),
            (ShimOverloadedError("no capacity"), "SHIM_OVERLOADED"),
            (ShimProtocolError("unexpected reply"), "INTERNAL_ERROR"),
        ],
        ids=["timeout", "disconnected", "overloaded", "malformed"],
    )
    async def test_a_transport_failure_comes_back_as_a_recording_error(
        self, error: Exception, code: str
    ):
        """The recording routes and tools catch ShimRecordingError only, so
        these escaped them as a 500 with no code to act on."""
        shim = _make_shim_with_rpc({})
        shim._rpc = AsyncMock(side_effect=error)
        shim.recording_channels = ["floor"]
        shim.recording_routes = ["extract_then_cloud"]
        recording = _RecordingProxy(shim)
        calls = {
            "REQUEST_RECORDING_CONSENT": recording.request_consent,
            "START_RECORDING": lambda: recording.start(
                mode="copilot",
                interpretation_route="extract_then_cloud",
                channels=["floor"],
                consent_token="token",
            ),
            "STOP_RECORDING": lambda: recording.stop("rec-1"),
            "APPLY_RECORDING_REVIEW": lambda: recording.apply_review(
                "rec-1", removed_step_seqs=[], redacted_step_seqs=[]
            ),
            "RECORDING_FETCH": lambda: recording.fetch("rec-1"),
        }

        for op, call in calls.items():
            with pytest.raises(ShimRecordingError) as caught:
                await call()
            assert caught.value.code == code
            assert caught.value.details == {"op": op}


class TestComputerProxy:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_tools_are_listed_once_per_connection(self):
        catalog = {"driver": {"name": "cua-driver"}, "instructions": "x", "tools": []}
        shim = _make_shim_with_rpc(
            {"type": "COMPUTER_TOOLS_RESPONSE", "payload": catalog}
        )

        assert await shim.computer.tools() == catalog
        assert await shim.computer.tools() == catalog
        assert shim._rpc.await_count == 1
        assert shim._rpc.await_args.args == ("COMPUTER_TOOLS_REQUEST", {})

        # A reconnect may bring another driver.
        shim._connection_generation += 1
        await shim.computer.tools()
        assert shim._rpc.await_count == 2

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_call_carries_the_tool_arguments_and_deadline(self):
        result = {"tool": "click", "is_error": False, "content": []}
        shim = _make_shim_with_rpc({"type": "COMPUTER_TOOL_RESULT", "payload": result})

        assert (
            await shim.computer.call(
                "click", {"element_token": "s1:2"}, timeout_seconds=30
            )
            == result
        )
        op, payload = shim._rpc.await_args.args
        assert op == "COMPUTER_TOOL_CALL"
        assert payload == {
            "tool": "click",
            "arguments": {"element_token": "s1:2"},
            "timeout_seconds": 30,
        }
        # The shim's own deadline fires first.
        assert shim._rpc.await_args.kwargs["timeout"] == 45

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_call_without_a_deadline_waits_for_the_shims_default(self):
        shim = _make_shim_with_rpc({"type": "COMPUTER_TOOL_RESULT", "payload": {}})

        await shim.computer.call("list_windows", {})

        assert "timeout_seconds" not in shim._rpc.await_args.args[1]
        assert shim._rpc.await_args.kwargs["timeout"] == 135

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_shim_error_raises_with_its_code_and_details(self):
        shim = _make_shim_with_rpc(
            {
                "type": "ERROR",
                "payload": {
                    "code": "FEATURE_NOT_SUPPORTED",
                    "message": "kill_app is not available on this machine.",
                    "details": {"tool": "kill_app"},
                },
            }
        )

        with pytest.raises(ShimComputerUseError) as exc_info:
            await shim.computer.call("kill_app", {"pid": 1})

        assert exc_info.value.code == "FEATURE_NOT_SUPPORTED"
        assert exc_info.value.details == {"tool": "kill_app"}
        assert "local_pc_computer_tools" in str(exc_info.value)

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (TimeoutError("RPC timed out (retry also failed)"), "COMMAND_TIMEOUT"),
            (OpUnconfirmedError("COMPUTER_TOOLS_REQUEST", "id-1"), "OP_UNCONFIRMED"),
            (ShimOverloadedError("no capacity"), "SHIM_OVERLOADED"),
            (ShimProtocolError("unexpected reply"), "INTERNAL_ERROR"),
        ],
        ids=["timeout", "disconnected", "overloaded", "malformed"],
    )
    async def test_a_transport_failure_comes_back_as_a_computer_use_error(
        self, error: Exception, code: str
    ):
        """The handlers and the held-call replay catch ShimComputerUseError
        only, so these escaped them with no code or guidance for Otto."""
        shim = _make_shim_with_rpc({})
        shim._rpc = AsyncMock(side_effect=error)

        with pytest.raises(ShimComputerUseError) as listing:
            await shim.computer.tools()
        with pytest.raises(ShimComputerUseError) as calling:
            await shim.computer.call("click", {"element_token": "s1:2"})

        assert listing.value.code == code
        assert listing.value.details == {"op": "COMPUTER_TOOLS_REQUEST"}
        assert calling.value.details["tool"] == "click"
        assert calling.value.code == code

    @pytest.mark.asyncio(loop_scope="session")
    async def test_an_unconfirmed_call_is_never_retried(self):
        shim = _make_shim_with_send_and_wait(
            [_RpcAttemptFailed("id-call-1", "COMPUTER_TOOL_CALL", timed_out=True)]
        )

        with pytest.raises(ShimComputerUseError) as exc_info:
            await shim.computer.call("click", {"element_token": "s1:2"})

        assert exc_info.value.details == {"tool": "click", "op": "COMPUTER_TOOL_CALL"}
        assert shim._send_and_wait.await_count == 1


# ---------------------------------------------------------------------------
# In-flight semantics on disconnect — per PROTOCOL.md "In-flight semantics
# on disconnect". Idempotent ops auto-retry once; non-idempotent ops raise
# WriteUnconfirmedError / OpUnconfirmedError immediately.
# ---------------------------------------------------------------------------


def _make_shim_with_send_and_wait(side_effect) -> LocalPCShim:
    """Build a LocalPCShim with _send_and_wait stubbed so we can exercise
    the retry-vs-raise branches in _rpc directly."""
    shim = LocalPCShim.__new__(LocalPCShim)
    shim.sandbox_id = "test-session"
    shim.allowed_root = "/Users/test/workspace"
    shim.machine_id = "machine-uuid"
    shim.platform = "darwin"
    shim.arch = "arm64"
    shim.capabilities = ["shell", "files"]
    shim.computer_use_driver = None
    shim.computer_use_tools = []
    shim._connection_generation = 1
    shim._manager = None
    shim._send_and_wait = AsyncMock(side_effect=side_effect)
    shim._await_reconnect_for_retry = AsyncMock(return_value=None)
    shim.files = _FilesProxy(shim)
    shim.commands = _CommandsProxy(shim)
    shim.computer = _ComputerProxy(shim)
    return shim


class TestIdempotencyTable:
    """The set of idempotent ops mirrors PROTOCOL.md. A drift here means the
    spec and the platform adapter have parted ways — explicit guard."""

    def test_read_ops_are_idempotent(self):
        for op in ("FILE_READ", "FILE_STAT", "FILE_LIST", "COMPUTER_TOOLS_REQUEST"):
            assert op in _IDEMPOTENT_OPS, f"{op} should be idempotent"

    def test_side_effecting_ops_are_not_idempotent(self):
        for op in (
            "EXECUTE_COMMAND",
            "FILE_WRITE",
            "FILE_DELETE",
            "FILE_MOVE",
            "COMPUTER_TOOL_CALL",
        ):
            assert op not in _IDEMPOTENT_OPS, f"{op} must not be idempotent"


class TestRpcAutoRetryForIdempotentOps:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_computer_read_retry_does_not_cross_reconnect(self):
        op = "COMPUTER_TOOLS_REQUEST"
        shim = _make_shim_with_send_and_wait(
            [_RpcAttemptFailed("id-1", op, timed_out=False)]
        )
        shim.capabilities = ["computer_use"]
        shim.computer_use_tools = ["click"]
        guard = shim.capture_connection_guard()

        async def reconnect():
            shim._connection_generation += 1
            shim.machine_id = "other-machine"

        shim._await_reconnect_for_retry = AsyncMock(side_effect=reconnect)

        with pytest.raises(ShimComputerUseError) as exc_info:
            await shim._rpc(op, {}, connection_guard=guard)

        assert exc_info.value.code == "COMPUTER_USE_CONNECTION_CHANGED"
        assert shim._send_and_wait.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_idempotent_op_retries_once_on_timeout_then_succeeds(self):
        ok = {"type": "FILE_CONTENTS", "payload": {"content": "hello"}}
        shim = _make_shim_with_send_and_wait(
            [
                _RpcAttemptFailed("id-1", "FILE_READ", timed_out=True),
                ok,
            ]
        )
        result = await shim._rpc("FILE_READ", {"path": "/x"})
        assert result is ok
        assert shim._send_and_wait.await_count == 2
        shim._await_reconnect_for_retry.assert_awaited_once()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_idempotent_op_retries_once_on_disconnect_then_succeeds(self):
        ok = {"type": "FILE_LIST_RESPONSE", "payload": {"entries": []}}
        shim = _make_shim_with_send_and_wait(
            [
                _RpcAttemptFailed("id-1", "FILE_LIST", timed_out=False),
                ok,
            ]
        )
        result = await shim._rpc("FILE_LIST", {"path": "/x"})
        assert result is ok
        assert shim._send_and_wait.await_count == 2

    @pytest.mark.asyncio(loop_scope="session")
    async def test_idempotent_op_retry_failure_raises_timeout_error(self):
        shim = _make_shim_with_send_and_wait(
            [
                _RpcAttemptFailed("id-1", "FILE_READ", timed_out=True),
                _RpcAttemptFailed("id-2", "FILE_READ", timed_out=True),
            ]
        )
        with pytest.raises(TimeoutError, match="retry also failed"):
            await shim._rpc("FILE_READ", {"path": "/x"})
        assert shim._send_and_wait.await_count == 2

    @pytest.mark.asyncio(loop_scope="session")
    async def test_idempotent_op_retry_disconnect_raises_op_unconfirmed(self):
        shim = _make_shim_with_send_and_wait(
            [
                _RpcAttemptFailed("id-1", "FILE_READ", timed_out=False),
                _RpcAttemptFailed("id-2", "FILE_READ", timed_out=False),
            ]
        )
        with pytest.raises(OpUnconfirmedError) as exc_info:
            await shim._rpc("FILE_READ", {"path": "/x"})
        assert exc_info.value.op == "FILE_READ"
        assert shim._send_and_wait.await_count == 2

    @pytest.mark.asyncio(loop_scope="session")
    async def test_retry_hidden_from_caller_on_success(self):
        """The caller sees one return value, not the intermediate failure."""
        ok = {"type": "COMPUTER_TOOLS_RESPONSE", "payload": {"tools": []}}
        shim = _make_shim_with_send_and_wait(
            [
                _RpcAttemptFailed("id-1", "COMPUTER_TOOLS_REQUEST", timed_out=False),
                ok,
            ]
        )
        result = await shim._rpc("COMPUTER_TOOLS_REQUEST", {})
        assert result is ok


class TestRpcNoRetryForNonIdempotentOps:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_file_write_timeout_raises_write_unconfirmed(self):
        shim = _make_shim_with_send_and_wait(
            [_RpcAttemptFailed("id-write-1", "FILE_WRITE", timed_out=True)]
        )
        with pytest.raises(WriteUnconfirmedError) as exc_info:
            await shim._rpc("FILE_WRITE", {"path": "/x", "content": "hi"})
        assert exc_info.value.code == "WRITE_UNCONFIRMED"
        assert exc_info.value.op == "FILE_WRITE"
        assert exc_info.value.wire_id == "id-write-1"
        # MUST NOT retry — non-idempotent.
        assert shim._send_and_wait.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_file_write_disconnect_raises_write_unconfirmed(self):
        shim = _make_shim_with_send_and_wait(
            [_RpcAttemptFailed("id-write-2", "FILE_WRITE", timed_out=False)]
        )
        with pytest.raises(WriteUnconfirmedError) as exc_info:
            await shim._rpc("FILE_WRITE", {"path": "/x", "content": "hi"})
        assert exc_info.value.wire_id == "id-write-2"
        assert shim._send_and_wait.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_execute_command_timeout_raises_op_unconfirmed(self):
        shim = _make_shim_with_send_and_wait(
            [_RpcAttemptFailed("id-exec-1", "EXECUTE_COMMAND", timed_out=True)]
        )
        with pytest.raises(OpUnconfirmedError) as exc_info:
            await shim._rpc("EXECUTE_COMMAND", {"command": "rm -rf /"})
        # Generic op-unconfirmed, NOT WriteUnconfirmedError.
        assert not isinstance(exc_info.value, WriteUnconfirmedError)
        assert exc_info.value.code == "OP_UNCONFIRMED"
        assert exc_info.value.op == "EXECUTE_COMMAND"
        assert shim._send_and_wait.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_file_delete_disconnect_does_not_retry(self):
        shim = _make_shim_with_send_and_wait(
            [_RpcAttemptFailed("id-del-1", "FILE_DELETE", timed_out=False)]
        )
        with pytest.raises(OpUnconfirmedError):
            await shim._rpc("FILE_DELETE", {"path": "/x"})
        assert shim._send_and_wait.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_computer_tool_call_timeout_does_not_retry(self):
        shim = _make_shim_with_send_and_wait(
            [_RpcAttemptFailed("id-call-1", "COMPUTER_TOOL_CALL", timed_out=True)]
        )
        with pytest.raises(OpUnconfirmedError) as exc_info:
            await shim._rpc("COMPUTER_TOOL_CALL", {"tool": "click", "arguments": {}})
        assert exc_info.value.op == "COMPUTER_TOOL_CALL"
        assert shim._send_and_wait.await_count == 1


class TestFilesWriteUnconfirmedSurface:
    """FilesProxy.write maps WriteUnconfirmedError into OSError with the
    translator's actionable English so the LLM never sees a raw exception."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_write_unconfirmed_becomes_os_error_with_recovery_hint(self):
        shim = LocalPCShim.__new__(LocalPCShim)
        shim.sandbox_id = "s"
        shim.allowed_root = "/Users/test/workspace"
        shim.platform = "darwin"
        shim.arch = "arm64"
        shim.machine_id = "m"
        shim.capabilities = ["files"]
        shim._rpc = AsyncMock(
            side_effect=WriteUnconfirmedError(
                wire_id="id-w",
                message="[LocalPC] FILE_WRITE (id=id-w) dropped",
            )
        )
        shim.files = _FilesProxy(shim)
        with pytest.raises(OSError) as exc_info:
            await shim.files.write("/Users/test/workspace/x.txt", "hi")
        msg = exc_info.value.args[0]
        # Translator surface — not the raw Python exception text.
        assert "FILE_WRITE" in msg
        assert "FILE_STAT" in msg


class TestOpUnconfirmedErrorShape:
    def test_op_unconfirmed_carries_op_and_wire_id(self):
        err = OpUnconfirmedError(op="FILE_DELETE", wire_id="abc")
        assert err.op == "FILE_DELETE"
        assert err.wire_id == "abc"
        assert err.code == "OP_UNCONFIRMED"
        # Subclass of RuntimeError so callers can catch broadly.
        assert isinstance(err, RuntimeError)

    def test_write_unconfirmed_specializes_op_and_code(self):
        err = WriteUnconfirmedError(wire_id="xyz")
        assert err.op == "FILE_WRITE"
        assert err.code == "WRITE_UNCONFIRMED"
        assert isinstance(err, OpUnconfirmedError)


# ---------------------------------------------------------------------------
# Backpressure consumer — pending_capacity tracking + STATUS frame
# consumption + 30s overload timeout (matches SHIM_OVERLOADED).
# ---------------------------------------------------------------------------


def _make_backpressure_shim() -> LocalPCShim:
    """Build a LocalPCShim with init'd backpressure state but stubbed I/O."""
    shim = LocalPCShim.__new__(LocalPCShim)
    shim.sandbox_id = "bp-session"
    shim.allowed_root = "/Users/test/workspace"
    shim.machine_id = "m"
    shim.platform = "darwin"
    shim.arch = "arm64"
    shim.capabilities = ["files"]
    shim.computer_use_driver = None
    shim.computer_use_tools = []
    shim._manager = None
    shim._pending = {}
    shim._pending_capacity = None
    shim._capacity_available = asyncio.Event()
    shim._capacity_available.set()
    return shim


class TestPendingCapacityTracking:
    def test_capacity_defaults_to_none(self):
        shim = _make_backpressure_shim()
        assert shim.pending_capacity is None

    def test_envelope_capacity_updates_property(self):
        shim = _make_backpressure_shim()
        shim._handle_envelope_capacity(
            {"type": "FILE_CONTENTS", "id": "x", "pending_capacity": 3, "payload": {}}
        )
        assert shim.pending_capacity == 3

    def test_payload_capacity_updates_property(self):
        shim = _make_backpressure_shim()
        shim._handle_envelope_capacity(
            {"type": "ACK", "payload": {"ok": True, "pending_capacity": 2}}
        )
        assert shim.pending_capacity == 2

    def test_capacity_zero_clears_event(self):
        shim = _make_backpressure_shim()
        assert shim._capacity_available.is_set()
        shim._update_pending_capacity(0, source="test")
        assert shim.pending_capacity == 0
        assert not shim._capacity_available.is_set()

    def test_capacity_above_zero_sets_event(self):
        shim = _make_backpressure_shim()
        shim._update_pending_capacity(0, source="test")
        assert not shim._capacity_available.is_set()
        shim._update_pending_capacity(4, source="test")
        assert shim._capacity_available.is_set()
        assert shim.pending_capacity == 4

    def test_non_int_capacity_is_ignored(self):
        shim = _make_backpressure_shim()
        shim._update_pending_capacity(5, source="seed")
        shim._update_pending_capacity("oops", source="bad")
        assert shim.pending_capacity == 5  # unchanged

    def test_negative_capacity_is_ignored(self):
        shim = _make_backpressure_shim()
        shim._update_pending_capacity(3, source="seed")
        shim._update_pending_capacity(-1, source="bad")
        assert shim.pending_capacity == 3

    def test_none_capacity_does_not_clear_property(self):
        shim = _make_backpressure_shim()
        shim._update_pending_capacity(2, source="seed")
        shim._update_pending_capacity(None, source="absent")
        assert shim.pending_capacity == 2


class TestStatusFrameConsumption:
    def test_status_with_pending_capacity_updates(self):
        shim = _make_backpressure_shim()
        shim._handle_status_frame(
            {
                "type": "STATUS",
                "payload": {
                    "in_flight": 2,
                    "max_concurrent": 4,
                    "queue_depth": 0,
                    "audit_log_bytes": 1024,
                    "uptime_seconds": 60,
                    "pending_capacity": 2,
                },
            }
        )
        assert shim.pending_capacity == 2

    def test_status_without_pending_capacity_derives_from_in_flight(self):
        shim = _make_backpressure_shim()
        shim._handle_status_frame(
            {
                "type": "STATUS",
                "payload": {
                    "in_flight": 4,
                    "max_concurrent": 4,
                    "queue_depth": 1,
                },
            }
        )
        assert shim.pending_capacity == 0
        assert not shim._capacity_available.is_set()

    def test_status_derived_capacity_clamped_to_zero(self):
        shim = _make_backpressure_shim()
        shim._handle_status_frame(
            {
                "type": "STATUS",
                "payload": {"in_flight": 10, "max_concurrent": 4},
            }
        )
        # Bad shim arithmetic shouldn't go negative.
        assert shim.pending_capacity == 0

    def test_status_with_malformed_payload_does_not_crash(self):
        shim = _make_backpressure_shim()
        # All of these should be no-ops, not exceptions.
        shim._handle_status_frame({"type": "STATUS"})
        shim._handle_status_frame({"type": "STATUS", "payload": None})
        shim._handle_status_frame({"type": "STATUS", "payload": "garbage"})
        shim._handle_status_frame(
            {"type": "STATUS", "payload": {"in_flight": "not-an-int"}}
        )
        assert shim.pending_capacity is None  # never updated

    def test_status_with_partial_fields_still_logs(self, caplog):
        shim = _make_backpressure_shim()
        import logging

        with caplog.at_level(
            logging.DEBUG, logger="backend.copilot.tools.local_pc_shim"
        ):
            shim._handle_status_frame(
                {
                    "type": "STATUS",
                    "payload": {
                        "in_flight": 1,
                        "max_concurrent": 4,
                        "uptime_seconds": 7,
                    },
                }
            )
        # DEBUG snapshot should be emitted with the partial fields visible.
        assert any("STATUS frame for" in r.message for r in caplog.records)


class TestBackpressureBlocking:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_rpc_blocks_when_capacity_zero(self):
        shim = _make_backpressure_shim()
        shim._send_and_wait = AsyncMock(
            return_value={"type": "FILE_CONTENTS", "payload": {"content": "ok"}}
        )
        shim._update_pending_capacity(0, source="seed")

        # Kick off the rpc — it should hang on the capacity event.
        task = asyncio.create_task(shim._rpc("FILE_READ", {"path": "/x"}))
        await asyncio.sleep(0.05)
        assert not task.done()
        # send_and_wait MUST NOT have fired yet.
        assert shim._send_and_wait.await_count == 0

        # Now clear capacity. The blocked _rpc should wake and complete.
        shim._update_pending_capacity(3, source="release")
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result["payload"]["content"] == "ok"
        assert shim._send_and_wait.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_rpc_unblocks_when_status_frame_releases_capacity(self):
        shim = _make_backpressure_shim()
        shim._send_and_wait = AsyncMock(
            return_value={"type": "ACK", "payload": {"ok": True}}
        )
        shim._update_pending_capacity(0, source="seed")

        task = asyncio.create_task(shim._rpc("FILE_STAT", {"path": "/x"}))
        await asyncio.sleep(0.05)
        assert not task.done()

        # STATUS frame announcing headroom should release the gate.
        shim._handle_status_frame(
            {
                "type": "STATUS",
                "payload": {"in_flight": 1, "max_concurrent": 4},
            }
        )
        await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_a_replacement_connection_starts_with_open_capacity(self):
        """The old connection's "no headroom" says nothing about the new one.
        Keeping it would hold every request until the new one's first STATUS
        frame, and fail them with SHIM_OVERLOADED if none came within 30 s."""
        shim = _make_backpressure_shim()
        shim._connection_generation = 1
        shim._streaming = {}
        shim._recording_steps = {}
        stop = asyncio.Event()

        class Transport:
            async def send_text(self, data: str) -> None:
                return None

            async def iter_text(self):
                await stop.wait()
                if False:
                    yield ""

            async def close(self) -> None:
                return None

        shim._ws = Transport()
        shim._recv_task = asyncio.create_task(shim._recv_loop(shim._ws))
        shim._update_pending_capacity(0, source="seed")

        previous_reader = shim._replace_connection(Transport(), ShimHello())
        try:
            await shim._await_capacity("FILE_READ", timeout=0.05)
        finally:
            stop.set()
            await asyncio.gather(
                *(task for task in (previous_reader, shim._recv_task) if task),
                return_exceptions=True,
            )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_rpc_raises_shim_overloaded_after_timeout(self):
        shim = _make_backpressure_shim()
        shim._send_and_wait = AsyncMock()  # should never fire
        shim._update_pending_capacity(0, source="seed")

        with pytest.raises(ShimOverloadedError) as exc_info:
            # Patch the timeout to something tiny to keep the test fast.
            await shim._await_capacity("FILE_READ", timeout=0.05)
        assert exc_info.value.code == "SHIM_OVERLOADED"
        assert "pending_capacity=0" in str(exc_info.value)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_rpc_does_not_block_when_capacity_unknown(self):
        """capacity=None (pre-STATUS / shim doesn't advertise) must not stall."""
        shim = _make_backpressure_shim()
        shim._send_and_wait = AsyncMock(
            return_value={"type": "ACK", "payload": {"ok": True}}
        )
        assert shim.pending_capacity is None
        await shim._rpc("FILE_READ", {"path": "/x"})
        assert shim._send_and_wait.await_count == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_rpc_does_not_block_when_capacity_positive(self):
        shim = _make_backpressure_shim()
        shim._send_and_wait = AsyncMock(
            return_value={"type": "ACK", "payload": {"ok": True}}
        )
        shim._update_pending_capacity(2, source="seed")
        await shim._rpc("FILE_READ", {"path": "/x"})
        assert shim._send_and_wait.await_count == 1
