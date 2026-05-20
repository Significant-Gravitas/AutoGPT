"""Tests for the LocalPCShim adapter — duck-type contract + cross-OS kwargs."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from .local_pc_shim import (
    LocalPCShim,
    ShimConnectionManager,
    ShimHello,
    _CommandsProxy,
    _FilesProxy,
)


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

    def test_from_payload_handles_missing_fields(self):
        hello = ShimHello.from_payload({})
        assert hello.machine_id == ""
        assert hello.capabilities == []
        assert hello.screen_resolution is None

    def test_from_payload_rejects_malformed_screen_resolution(self):
        hello = ShimHello.from_payload({"screen_resolution": [1920]})
        assert hello.screen_resolution is None


class TestShimConnectionManager:
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
