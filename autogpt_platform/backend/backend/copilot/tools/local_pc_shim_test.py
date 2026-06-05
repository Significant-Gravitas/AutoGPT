"""Tests for the LocalPCShim adapter — duck-type contract + cross-OS kwargs."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from .local_pc_shim import (
    _IDEMPOTENT_OPS,
    LocalPCShim,
    OpUnconfirmedError,
    ShimComputerUseError,
    ShimConnectionManager,
    ShimHello,
    WriteUnconfirmedError,
    _CommandsProxy,
    _ComputerProxy,
    _FilesProxy,
    _RpcAttemptFailed,
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
    shim.computer = _ComputerProxy(shim)
    shim.computer_use_features = []
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


# ---------------------------------------------------------------------------
# _ComputerProxy — wire-op contract for computer-use surface
# ---------------------------------------------------------------------------


class TestComputerScreenshot:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_screenshot_default_payload(self):
        shim = _make_shim_with_rpc(
            {"payload": {"image_base64": "abc", "width": 800, "height": 600}}
        )
        result = await shim.computer.screenshot()
        op, payload = shim._rpc.await_args.args
        assert op == "SCREENSHOT_REQUEST"
        assert payload["monitor"] == 0
        assert payload["format"] == "png"
        assert payload["include_cursor"] is False
        assert payload["quality"] == 75
        assert "region" not in payload
        assert "window_id" not in payload
        assert result["width"] == 800

    @pytest.mark.asyncio(loop_scope="session")
    async def test_screenshot_region_forwards_as_list(self):
        shim = _make_shim_with_rpc({"payload": {"image_base64": "x"}})
        await shim.computer.screenshot(region=(10, 20, 100, 200), include_cursor=True)
        payload = shim._rpc.await_args.args[1]
        assert payload["region"] == [10, 20, 100, 200]
        assert payload["include_cursor"] is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_screenshot_window_id_forwards(self):
        shim = _make_shim_with_rpc({"payload": {"image_base64": "x"}})
        await shim.computer.screenshot(window_id="win_abc")
        payload = shim._rpc.await_args.args[1]
        assert payload["window_id"] == "win_abc"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_screenshot_region_and_window_id_mutually_exclusive(self):
        shim = _make_shim_with_rpc({"payload": {}})
        with pytest.raises(ValueError):
            await shim.computer.screenshot(region=(0, 0, 1, 1), window_id="win_x")

    @pytest.mark.asyncio(loop_scope="session")
    async def test_screenshot_error_raises_typed_error(self):
        shim = _make_shim_with_rpc(
            {
                "type": "ERROR",
                "payload": {
                    "code": "PERMISSION_PENDING",
                    "message": "screen recording denied",
                    "details": {"permission": "screen_recording"},
                },
            }
        )
        with pytest.raises(ShimComputerUseError) as exc_info:
            await shim.computer.screenshot()
        assert exc_info.value.code == "PERMISSION_PENDING"
        assert exc_info.value.details.get("permission") == "screen_recording"


class TestComputerInputActions:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_left_click_uses_left_click_action(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.computer.click([100, 200])
        op, payload = shim._rpc.await_args.args
        assert op == "INPUT_ACTION"
        assert payload["action"] == "left_click"
        assert payload["coordinate"] == [100, 200]
        assert payload["button"] == "left"
        assert "modifiers" not in payload

    @pytest.mark.asyncio(loop_scope="session")
    async def test_right_click_maps_to_right_click_action(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.computer.click([5, 6], button="right")
        payload = shim._rpc.await_args.args[1]
        assert payload["action"] == "right_click"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_modifiers_forwarded(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.computer.click([10, 10], modifiers=["shift", "ctrl"])
        payload = shim._rpc.await_args.args[1]
        assert payload["modifiers"] == ["shift", "ctrl"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_scroll_uses_scroll_direction_field(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.computer.scroll([500, 500], direction="up", scroll_amount=4)
        payload = shim._rpc.await_args.args[1]
        assert payload["action"] == "scroll"
        assert payload["scroll_direction"] == "up"
        assert payload["scroll_amount"] == 4

    @pytest.mark.asyncio(loop_scope="session")
    async def test_type_default_explicit_paste_and_preserve(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.computer.type("hello world")
        payload = shim._rpc.await_args.args[1]
        assert payload["action"] == "type"
        assert payload["text"] == "hello world"
        # The wire schema treats absent and false the same; the proxy
        # always forwards the explicit value so the shim's per-OS default
        # never silently overrides what the platform asked for.
        assert payload["paste"] is False
        assert payload["preserve_clipboard"] is False

    @pytest.mark.asyncio(loop_scope="session")
    async def test_type_paste_true_forwards(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.computer.type("x" * 500, paste=True, preserve_clipboard=True)
        payload = shim._rpc.await_args.args[1]
        assert payload["paste"] is True
        assert payload["preserve_clipboard"] is True

    @pytest.mark.asyncio(loop_scope="session")
    async def test_key_forwards_key_field(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.computer.key("ctrl+s")
        payload = shim._rpc.await_args.args[1]
        assert payload["action"] == "key"
        assert payload["key"] == "ctrl+s"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_drag_serialises_path_as_list_of_lists(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.computer.drag([(0, 0), (50, 50), (100, 100)], duration_ms=500)
        payload = shim._rpc.await_args.args[1]
        assert payload["action"] == "drag"
        assert payload["path"] == [[0, 0], [50, 50], [100, 100]]
        assert payload["duration_ms"] == 500

    @pytest.mark.asyncio(loop_scope="session")
    async def test_wait_forwards_duration_ms(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.computer.wait(1200)
        payload = shim._rpc.await_args.args[1]
        assert payload["action"] == "wait"
        assert payload["duration_ms"] == 1200

    @pytest.mark.asyncio(loop_scope="session")
    async def test_input_out_of_bounds_carries_details(self):
        shim = _make_shim_with_rpc(
            {
                "type": "ERROR",
                "payload": {
                    "code": "INPUT_OUT_OF_BOUNDS",
                    "message": "outside displays",
                    "details": {
                        "requested_coordinate": [99999, 99999],
                        "displays": [
                            {"index": 0, "origin": [0, 0], "size": [1920, 1080]}
                        ],
                    },
                },
            }
        )
        with pytest.raises(ShimComputerUseError) as exc_info:
            await shim.computer.click([99999, 99999])
        assert exc_info.value.code == "INPUT_OUT_OF_BOUNDS"
        assert exc_info.value.details["requested_coordinate"] == [99999, 99999]


class TestComputerCursorAndDisplay:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_cursor_position_returns_payload(self):
        shim = _make_shim_with_rpc({"payload": {"x": 12, "y": 34, "monitor": 0}})
        result = await shim.computer.cursor_position()
        op, _ = shim._rpc.await_args.args
        assert op == "CURSOR_POSITION_REQUEST"
        assert result == {"x": 12, "y": 34, "monitor": 0}

    @pytest.mark.asyncio(loop_scope="session")
    async def test_display_info_returns_payload(self):
        shim = _make_shim_with_rpc(
            {"payload": {"monitors": [{"index": 0, "logical_size": [1920, 1080]}]}}
        )
        result = await shim.computer.display_info()
        assert len(result["monitors"]) == 1


class TestComputerWindowsAndApps:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_list_windows_returns_array(self):
        shim = _make_shim_with_rpc(
            {
                "payload": {
                    "windows": [
                        {"window_id": "win_1", "title": "Safari"},
                        {"window_id": "win_2", "title": "Mail"},
                    ],
                    "truncated": False,
                }
            }
        )
        result = await shim.computer.list_windows()
        assert len(result) == 2
        assert result[0]["window_id"] == "win_1"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_focus_window_sends_raise_field(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.computer.focus_window("win_1", raise_=False)
        op, payload = shim._rpc.await_args.args
        assert op == "WINDOW_FOCUS"
        assert payload["window_id"] == "win_1"
        assert payload["raise"] is False

    @pytest.mark.asyncio(loop_scope="session")
    async def test_window_stale_raises_typed_error(self):
        shim = _make_shim_with_rpc(
            {
                "type": "ERROR",
                "payload": {
                    "code": "WINDOW_STALE",
                    "message": "stale",
                    "details": {"window_id": "win_x"},
                },
            }
        )
        with pytest.raises(ShimComputerUseError) as exc_info:
            await shim.computer.focus_window("win_x")
        assert exc_info.value.code == "WINDOW_STALE"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_list_apps_returns_array(self):
        shim = _make_shim_with_rpc(
            {"payload": {"apps": [{"pid": 1, "name": "Safari"}]}}
        )
        result = await shim.computer.list_apps()
        assert result[0]["pid"] == 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_launch_app_requires_bundle_or_path(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True, "pid": 99}})
        with pytest.raises(ValueError):
            await shim.computer.launch_app()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_launch_app_with_bundle_id(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True, "pid": 99}})
        result = await shim.computer.launch_app(bundle_id="com.apple.Safari")
        payload = shim._rpc.await_args.args[1]
        assert payload["bundle_id"] == "com.apple.Safari"
        assert payload["activate"] is True
        assert result["pid"] == 99


class TestComputerClipboard:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_clipboard_read_returns_content(self):
        shim = _make_shim_with_rpc(
            {"payload": {"format": "text", "content": "hello", "size_bytes": 5}}
        )
        result = await shim.computer.clipboard_read()
        op, payload = shim._rpc.await_args.args
        assert op == "CLIPBOARD_READ"
        assert payload["format"] == "text"
        assert result == "hello"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_clipboard_write_forwards_content(self):
        shim = _make_shim_with_rpc({"payload": {"ok": True}})
        await shim.computer.clipboard_write("secret")
        op, payload = shim._rpc.await_args.args
        assert op == "CLIPBOARD_WRITE"
        assert payload["content"] == "secret"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_clipboard_concealed_raises_typed_error(self):
        shim = _make_shim_with_rpc(
            {
                "type": "ERROR",
                "payload": {
                    "code": "CLIPBOARD_CONCEALED",
                    "message": "concealed",
                    "details": {"reason": "writeback_only"},
                },
            }
        )
        with pytest.raises(ShimComputerUseError) as exc_info:
            await shim.computer.clipboard_read()
        assert exc_info.value.code == "CLIPBOARD_CONCEALED"


class TestComputerPermissions:
    @pytest.mark.asyncio(loop_scope="session")
    async def test_permissions_check_default_set(self):
        shim = _make_shim_with_rpc(
            {
                "payload": {
                    "permissions": {
                        "screen_recording": "granted",
                        "accessibility": "denied",
                        "input_monitoring": "unknown",
                    }
                }
            }
        )
        result = await shim.computer.permissions_check()
        payload = shim._rpc.await_args.args[1]
        assert payload["permissions"] == [
            "screen_recording",
            "accessibility",
            "input_monitoring",
        ]
        assert result["accessibility"] == "denied"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_permissions_check_custom_set(self):
        shim = _make_shim_with_rpc({"payload": {"permissions": {}}})
        await shim.computer.permissions_check(["accessibility"])
        payload = shim._rpc.await_args.args[1]
        assert payload["permissions"] == ["accessibility"]


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
    shim.computer_use_features = []
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
        for op in (
            "FILE_READ",
            "FILE_STAT",
            "FILE_LIST",
            "CURSOR_POSITION_REQUEST",
            "DISPLAY_INFO_REQUEST",
            "WINDOW_LIST_REQUEST",
            "APP_LIST_REQUEST",
            "CLIPBOARD_READ",
            "PERMISSIONS_CHECK_REQUEST",
            "SCREENSHOT_REQUEST",
        ):
            assert op in _IDEMPOTENT_OPS, f"{op} should be idempotent"

    def test_side_effecting_ops_are_not_idempotent(self):
        for op in (
            "EXECUTE_COMMAND",
            "FILE_WRITE",
            "FILE_DELETE",
            "FILE_MOVE",
            "WINDOW_FOCUS",
            "APP_LAUNCH",
            "CLIPBOARD_WRITE",
            "INPUT_ACTION",
        ):
            assert op not in _IDEMPOTENT_OPS, f"{op} must not be idempotent"


class TestRpcAutoRetryForIdempotentOps:
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
        ok = {"type": "SCREENSHOT_RESPONSE", "payload": {"image_base64": "x"}}
        shim = _make_shim_with_send_and_wait(
            [
                _RpcAttemptFailed("id-1", "SCREENSHOT_REQUEST", timed_out=False),
                ok,
            ]
        )
        result = await shim._rpc("SCREENSHOT_REQUEST", {})
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
    async def test_input_action_timeout_does_not_retry(self):
        shim = _make_shim_with_send_and_wait(
            [_RpcAttemptFailed("id-input-1", "INPUT_ACTION", timed_out=True)]
        )
        with pytest.raises(OpUnconfirmedError) as exc_info:
            await shim._rpc("INPUT_ACTION", {"action": "left_click"})
        assert exc_info.value.op == "INPUT_ACTION"
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
