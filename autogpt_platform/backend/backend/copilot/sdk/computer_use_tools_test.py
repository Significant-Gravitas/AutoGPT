"""Handler-level coverage for Local PC computer-use MCP tools."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import backend.copilot.sdk.computer_use_tools as tools
from backend.copilot.tools.local_pc_shim import LocalPCShim, ShimComputerUseError


def _shim(*, capabilities: tuple[str, ...] = ("computer_use",)) -> LocalPCShim:
    shim = LocalPCShim.__new__(LocalPCShim)
    shim.sandbox_id = "session-1"
    shim.platform = "darwin"
    shim.machine_id = "machine-1"
    shim.allowed_root = "/Users/test/workspace"
    shim.capabilities = list(capabilities)
    shim.capability_set = frozenset(capabilities)
    shim.computer_use_features_coarse = ["screenshot", "input"]
    shim.computer_use_features = ["screenshot.capture", "input.click"]
    shim._connection_generation = 1
    shim.computer = MagicMock()
    for method in (
        "screenshot",
        "click",
        "type",
        "key",
        "scroll",
        "cursor_position",
        "list_windows",
        "focus_window",
        "list_apps",
        "launch_app",
        "clipboard_read",
        "clipboard_write",
        "permissions_check",
    ):
        setattr(shim.computer, method, AsyncMock())
    return shim


def _install(monkeypatch: pytest.MonkeyPatch, shim: LocalPCShim | None) -> None:
    monkeypatch.setattr(tools, "get_current_sandbox", lambda: shim)
    monkeypatch.setattr(
        tools,
        "get_execution_context",
        lambda: ("user-1", SimpleNamespace(session_id="session-1")),
    )
    monkeypatch.setattr(tools, "is_computer_use_approved", AsyncMock(return_value=True))


def _payload(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


class TestGating:
    @pytest.mark.asyncio
    async def test_no_executor_fails_closed(self, monkeypatch: pytest.MonkeyPatch):
        _install(monkeypatch, None)

        result = await tools._h_click({"coordinate": [1, 2]})

        assert result["isError"] is True
        assert _payload(result)["code"] == "NO_LOCAL_PC_EXECUTOR"

    @pytest.mark.asyncio
    async def test_missing_capability_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _install(monkeypatch, _shim(capabilities=("files",)))

        result = await tools._h_screenshot({})

        assert result["isError"] is True
        assert _payload(result)["code"] == "CAPABILITY_NOT_GRANTED"

    @pytest.mark.asyncio
    async def test_revoked_session_consent_blocks_each_operation(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        _install(monkeypatch, shim)
        consent = AsyncMock(return_value=False)
        monkeypatch.setattr(tools, "is_computer_use_approved", consent)

        result = await tools._h_click({"coordinate": [1, 2]})

        assert result["isError"] is True
        assert _payload(result)["code"] == "COMPUTER_USE_CONSENT_REQUIRED"
        consent.assert_awaited_once_with(
            "session-1",
            "user-1",
            machine_id="machine-1",
            features_coarse=("screenshot", "input"),
            features=("screenshot.capture", "input.click"),
        )
        shim.computer.click.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reconnect_during_consent_check_blocks_dispatch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        _install(monkeypatch, shim)
        consent_started = asyncio.Event()
        release_consent = asyncio.Event()

        async def delayed_approval(*_args, **_kwargs):
            consent_started.set()
            await release_consent.wait()
            return True

        monkeypatch.setattr(tools, "is_computer_use_approved", delayed_approval)
        task = asyncio.create_task(tools._h_click({"coordinate": [1, 2]}))
        await consent_started.wait()
        shim._connection_generation += 1
        shim.machine_id = "machine-2"
        release_consent.set()

        result = await task

        assert result["isError"] is True
        assert _payload(result)["code"] == "COMPUTER_USE_CONNECTION_CHANGED"
        shim.computer.click.assert_not_awaited()


class TestObservationHandlers:
    @pytest.mark.asyncio
    async def test_screenshot_returns_image_and_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        shim.computer.screenshot.return_value = {
            "image_base64": "aW1hZ2U=",
            "mime_type": "image/png",
            "width": 800,
            "height": 600,
        }
        _install(monkeypatch, shim)

        result = await tools._h_screenshot({"monitor": 1, "include_cursor": True})

        assert result["isError"] is False
        assert json.loads(result["content"][0]["text"])["width"] == 800
        assert result["content"][1] == {
            "type": "image",
            "data": "aW1hZ2U=",
            "mimeType": "image/png",
        }
        shim.computer.screenshot.assert_awaited_once_with(
            monitor=1,
            region=None,
            window_id=None,
            format="jpeg",
            include_cursor=True,
            quality=75,
            _guard=shim.capture_connection_guard(),
        )

    @pytest.mark.asyncio
    async def test_cursor_windows_and_apps_are_serialized(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        shim.computer.cursor_position.return_value = {"coordinate": [10, 20]}
        shim.computer.list_windows.return_value = [{"window_id": "win-1"}]
        shim.computer.list_apps.return_value = [{"pid": 7, "name": "Editor"}]
        _install(monkeypatch, shim)

        cursor = _payload(await tools._h_cursor_position({}))
        windows = _payload(await tools._h_list_windows({"include_minimized": True}))
        apps = _payload(await tools._h_list_apps({"include_background": True}))

        assert cursor["coordinate"] == [10, 20]
        assert windows["windows"][0]["window_id"] == "win-1"
        assert apps["apps"][0]["pid"] == 7
        shim.computer.list_windows.assert_awaited_once_with(
            app_bundle_id=None,
            include_minimized=True,
            include_offscreen=False,
            _guard=shim.capture_connection_guard(),
        )

    @pytest.mark.asyncio
    async def test_permissions_check_forwards_requested_subset(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        shim.computer.permissions_check.return_value = {"accessibility": "granted"}
        _install(monkeypatch, shim)

        result = await tools._h_permissions_check({"permissions": ["accessibility"]})

        assert _payload(result) == {"accessibility": "granted"}
        shim.computer.permissions_check.assert_awaited_once_with(
            ["accessibility"], _guard=shim.capture_connection_guard()
        )


class TestInputHandlers:
    @pytest.mark.asyncio
    async def test_click_rejects_malformed_coordinate(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        _install(monkeypatch, shim)

        result = await tools._h_click({"coordinate": [1]})

        assert _payload(result)["code"] == "INVALID_ARGUMENT"
        shim.computer.click.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_click_surfaces_structured_shim_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        shim.computer.click.side_effect = ShimComputerUseError(
            "INPUT_OUT_OF_BOUNDS", "outside display", {"coordinate": [9, 9]}
        )
        _install(monkeypatch, shim)

        result = await tools._h_click({"coordinate": [9, 9]})

        assert result["isError"] is True
        assert _payload(result)["code"] == "INPUT_OUT_OF_BOUNDS"

    @pytest.mark.asyncio
    async def test_type_key_scroll_and_focus_forward_arguments(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        _install(monkeypatch, shim)

        await tools._h_type(
            {"text": "hello", "paste": True, "preserve_clipboard": True}
        )
        await tools._h_key({"key": "ctrl+s"})
        await tools._h_scroll(
            {"coordinate": [4, 5], "direction": "up", "scroll_amount": 3}
        )
        await tools._h_focus_window({"window_id": "win-1", "raise": False})

        shim.computer.type.assert_awaited_once_with(
            "hello",
            paste=True,
            preserve_clipboard=True,
            _guard=shim.capture_connection_guard(),
        )
        shim.computer.key.assert_awaited_once_with(
            "ctrl+s", _guard=shim.capture_connection_guard()
        )
        shim.computer.scroll.assert_awaited_once_with(
            [4, 5],
            direction="up",
            scroll_amount=3,
            modifiers=None,
            _guard=shim.capture_connection_guard(),
        )
        shim.computer.focus_window.assert_awaited_once_with(
            "win-1", raise_=False, _guard=shim.capture_connection_guard()
        )

    @pytest.mark.asyncio
    async def test_key_requires_nonempty_value(self, monkeypatch: pytest.MonkeyPatch):
        shim = _shim()
        _install(monkeypatch, shim)

        result = await tools._h_key({})

        assert _payload(result)["code"] == "INVALID_ARGUMENT"
        shim.computer.key.assert_not_awaited()


class TestLaunchAndClipboard:
    @pytest.mark.asyncio
    async def test_launch_rejects_executable_outside_workspace(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        _install(monkeypatch, shim)

        result = await tools._h_launch_app({"executable_path": "/usr/bin/python"})

        assert _payload(result)["code"] == "PATH_OUTSIDE_ALLOWED_ROOT"
        shim.computer.launch_app.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_launch_resolves_workspace_executable(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        shim = _shim()
        shim.computer.launch_app.return_value = {"pid": 42}
        _install(monkeypatch, shim)

        result = await tools._h_launch_app(
            {"executable_path": "bin/tool", "args": ["--safe"]}
        )

        assert _payload(result)["pid"] == 42
        shim.computer.launch_app.assert_awaited_once_with(
            bundle_id=None,
            executable_path="/Users/test/workspace/bin/tool",
            args=["--safe"],
            activate=True,
            _guard=shim.capture_connection_guard(),
        )

    @pytest.mark.asyncio
    async def test_clipboard_read_and_write(self, monkeypatch: pytest.MonkeyPatch):
        shim = _shim()
        shim.computer.clipboard_read.return_value = "hello"
        _install(monkeypatch, shim)

        read_result = await tools._h_clipboard_read({"format": "text"})
        write_result = await tools._h_clipboard_write(
            {"content": "world", "format": "text"}
        )

        assert _payload(read_result)["content"] == "hello"
        assert _payload(write_result) == {"ok": True}
        shim.computer.clipboard_write.assert_awaited_once_with(
            "world", format="text", _guard=shim.capture_connection_guard()
        )
