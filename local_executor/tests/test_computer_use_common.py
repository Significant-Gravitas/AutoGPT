"""OS-agnostic tests for the computer-use backend layer."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autogpt_local_executor.computer_use import (
    ClipboardConcealedError,
    FeatureNotSupportedError,
    InputOutOfBoundsError,
    NullBackend,
    PermissionPendingError,
    WindowStaleError,
)
from autogpt_local_executor.computer_use.backend import (
    ClipboardReadResult,
    ScreenshotResult,
)
from autogpt_local_executor.computer_use.backends._common import (
    in_any_rect,
    normalize_displays_for_error,
)
from autogpt_local_executor.computer_use.clipboard import (
    ClipboardWritebackRegistry,
)
from autogpt_local_executor.computer_use.window_registry import (
    WindowFingerprint,
    WindowRegistry,
)
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import ComputerUseHandler
from autogpt_local_executor.protocol import (
    AppLaunchMessage,
    AppLaunchPayload,
    ClipboardReadMessage,
    ClipboardWriteMessage,
    ClipboardWritePayload,
    CursorPositionRequestMessage,
    DisplayInfoRequestMessage,
    DisplayMonitor,
    ErrorCode,
    ErrorMessage,
    InputActionMessage,
    InputActionPayload,
    PermissionsCheckRequestMessage,
    PermissionsCheckRequestPayload,
    ScreenshotRequestMessage,
    ScreenshotRequestPayload,
    WindowFocusMessage,
    WindowFocusPayload,
    WindowListRequestMessage,
    new_id,
    now_ts,
)

# ── Test helpers ───────────────────────────────────────────────────────


def make_config(tmp_path: Path, **overrides) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        enable_computer_use=True,
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        **overrides,
    )


class FakeBackend:
    """In-memory backend stub used to drive the handler dispatcher."""

    def __init__(self):
        self.coarse_features = MagicMock(return_value=["screenshot", "input"])
        self.features = MagicMock(return_value=["screenshot.region"])
        self.screenshot = MagicMock()
        self.input_action = MagicMock()
        self.cursor_position = MagicMock()
        self.display_info = MagicMock()
        self.window_list = MagicMock()
        self.window_focus = MagicMock()
        self.app_list = MagicMock()
        self.app_launch = MagicMock()
        self.clipboard_read = MagicMock()
        self.clipboard_write = MagicMock()
        self.permissions_check = MagicMock()
        self.on_hello = MagicMock()


def attach_fake(handler: ComputerUseHandler) -> FakeBackend:
    fb = FakeBackend()
    handler._backend = fb  # type: ignore[assignment]
    return fb


# ── WindowRegistry / fingerprint ───────────────────────────────────────


def test_window_registry_mints_unique_uuids_with_prefix() -> None:
    reg = WindowRegistry()
    fp = WindowFingerprint(pid=1, class_name="X", creation_timestamp=None)
    wid1 = reg.mint("handle1", fp)
    wid2 = reg.mint("handle2", fp)
    assert wid1 != wid2
    assert wid1.startswith("win_")
    assert wid2.startswith("win_")


def test_window_registry_clear_wipes_all() -> None:
    reg = WindowRegistry()
    fp = WindowFingerprint(pid=1, class_name="X", creation_timestamp=None)
    reg.mint("h1", fp)
    reg.mint("h2", fp)
    assert len(reg) == 2
    reg.clear()
    assert len(reg) == 0


def test_window_registry_verify_raises_on_dead_window() -> None:
    """Q2 contract: a verify against a window the OS no longer has must
    raise WINDOW_STALE — never silently re-bind."""
    reg = WindowRegistry()
    fp = WindowFingerprint(pid=12, class_name="C", creation_timestamp=None)
    wid = reg.mint("native", fp)

    def _gone(_native):
        return None  # OS reports the window as dead

    with pytest.raises(WindowStaleError) as ei:
        reg.verify(wid, _gone)
    assert wid in ei.value.details["window_id"]
    assert ei.value.details["hint"] == "re-list windows"
    # And after a stale verify the entry should be dropped, so a
    # follow-up lookup also fails.
    with pytest.raises(WindowStaleError):
        reg.verify(wid, _gone)


def test_window_registry_verify_raises_on_fingerprint_mismatch() -> None:
    """The OS reassigning the handle to a different process/class is the
    common silent-corruption case Q2 specifically guards against."""
    reg = WindowRegistry()
    fp = WindowFingerprint(pid=12, class_name="A", creation_timestamp=None)
    wid = reg.mint("native", fp)

    def _different(_native):
        return WindowFingerprint(pid=13, class_name="A", creation_timestamp=None)

    with pytest.raises(WindowStaleError):
        reg.verify(wid, _different)


def test_window_registry_unknown_id() -> None:
    reg = WindowRegistry()
    with pytest.raises(WindowStaleError):
        reg.verify("win_does-not-exist", lambda _: None)


# ── Clipboard writeback registry (Q3) ──────────────────────────────────


def test_writeback_returns_record_when_in_window() -> None:
    reg = ClipboardWritebackRegistry()
    reg.record_write("hello", sequence=10)
    rec = reg.check(live_content="hello", live_sequence=10)
    assert rec is not None
    assert rec.sha256


def test_writeback_returns_none_after_window_expires() -> None:
    reg = ClipboardWritebackRegistry(window_seconds=0.01)
    reg.record_write("hello", sequence=10)
    time.sleep(0.05)
    assert reg.check(live_content="hello", live_sequence=10) is None


def test_writeback_returns_none_on_content_mismatch() -> None:
    """User overwrites clipboard with a different value → snooping
    refused."""
    reg = ClipboardWritebackRegistry()
    reg.record_write("ours", sequence=1)
    assert reg.check(live_content="theirs", live_sequence=2) is None


def test_writeback_returns_none_on_sequence_mismatch() -> None:
    reg = ClipboardWritebackRegistry()
    reg.record_write("hello", sequence=1)
    # changeCount advanced — a third party wrote then wrote our exact
    # value back (extremely unlikely but the contract is sequence wins).
    assert reg.check(live_content="hello", live_sequence=2) is None


def test_writeback_clear() -> None:
    reg = ClipboardWritebackRegistry()
    reg.record_write("hello")
    reg.clear()
    assert reg.snapshot() is None


# ── Bounds helpers (Q1) ────────────────────────────────────────────────


def test_in_any_rect_single_display() -> None:
    d = [{"index": 0, "origin": (0, 0), "size": (1920, 1080)}]
    assert in_any_rect((0, 0), d)
    assert in_any_rect((1919, 1079), d)
    assert not in_any_rect((1920, 0), d)
    assert not in_any_rect((-1, 0), d)
    assert not in_any_rect((10000, 10000), d)


def test_in_any_rect_multi_display_stitch() -> None:
    """Two stitched displays: 1920+2560 wide. Boundary pixel between
    them must not be double-counted."""
    d = [
        {"index": 0, "origin": (0, 0), "size": (1920, 1080)},
        {"index": 1, "origin": (1920, 0), "size": (2560, 1440)},
    ]
    assert in_any_rect((0, 0), d)
    assert in_any_rect((1920, 0), d)  # left edge of monitor 1
    assert in_any_rect((4479, 1439), d)
    assert not in_any_rect((4480, 0), d)


def test_normalize_displays_for_error_shape() -> None:
    raw = [{"index": 0, "origin": (10, 20), "size": (1920, 1080)}]
    out = normalize_displays_for_error(raw)
    assert out[0]["origin"] == [10, 20]
    assert out[0]["size"] == [1920, 1080]


# ── Structured errors carry the right wire `details` ───────────────────


def test_input_out_of_bounds_error_payload() -> None:
    displays = [{"index": 0, "origin": [0, 0], "size": [1920, 1080]}]
    exc = InputOutOfBoundsError((10000, 10000), displays)
    assert exc.code == ErrorCode.INPUT_OUT_OF_BOUNDS
    assert exc.details["requested_coordinate"] == [10000, 10000]
    assert exc.details["displays"] == displays


def test_window_stale_error_payload() -> None:
    exc = WindowStaleError("win_xyz")
    assert exc.code == ErrorCode.WINDOW_STALE
    assert exc.details["window_id"] == "win_xyz"
    assert exc.details["hint"] == "re-list windows"


def test_clipboard_concealed_error_writeback_only() -> None:
    exc = ClipboardConcealedError("writeback_only")
    assert exc.code == ErrorCode.CLIPBOARD_CONCEALED
    assert exc.details["reason"] == "writeback_only"


def test_clipboard_concealed_error_concealed_type() -> None:
    exc = ClipboardConcealedError("concealed_type", marker="org.nspasteboard.ConcealedType")
    assert exc.details["marker"] == "org.nspasteboard.ConcealedType"


def test_permission_pending_error_payload() -> None:
    exc = PermissionPendingError("accessibility", "darwin", hint="open Settings")
    assert exc.code == ErrorCode.PERMISSION_PENDING
    assert exc.details["permission"] == "accessibility"
    assert exc.details["platform"] == "darwin"
    assert exc.details["hint"] == "open Settings"


def test_feature_not_supported_error_payload() -> None:
    exc = FeatureNotSupportedError("input", reason="Wayland session")
    assert exc.code == ErrorCode.FEATURE_NOT_SUPPORTED
    assert exc.details["feature"] == "input"
    assert exc.details["reason"] == "Wayland session"


# ── NullBackend & get_backend ───────────────────────────────────────────


def test_null_backend_advertises_nothing(tmp_path: Path) -> None:
    nb = NullBackend(make_config(tmp_path))
    assert nb.coarse_features() == []
    assert nb.features() == []


def test_null_backend_refuses_screenshot(tmp_path: Path) -> None:
    nb = NullBackend(make_config(tmp_path))
    with pytest.raises(FeatureNotSupportedError):
        nb.screenshot()


def test_null_backend_returns_not_applicable_for_perms(tmp_path: Path) -> None:
    nb = NullBackend(make_config(tmp_path))
    out = nb.permissions_check(["accessibility", "screen_recording"])
    assert out == {"accessibility": "not_applicable", "screen_recording": "not_applicable"}


# ── Dispatcher: BackendError → wire ERROR ──────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_window_stale_to_wire(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path))
    fb = attach_fake(handler)
    fb.window_focus.side_effect = WindowStaleError("win_xyz")
    msg = WindowFocusMessage(
        id=new_id(), ts=now_ts(), payload=WindowFocusPayload(window_id="win_xyz")
    )
    resp = await handler.handle(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.WINDOW_STALE
    assert resp.payload.details["window_id"] == "win_xyz"
    assert resp.payload.details["hint"] == "re-list windows"


@pytest.mark.asyncio
async def test_dispatch_input_out_of_bounds_echoes_display_rects(
    tmp_path: Path,
) -> None:
    """Q1: INPUT_ACTION with coord outside display bounds gives a
    structured error echoing the valid display rects."""
    handler = ComputerUseHandler(make_config(tmp_path))
    fb = attach_fake(handler)
    displays = [{"index": 0, "origin": [0, 0], "size": [1920, 1080]}]
    fb.input_action.side_effect = InputOutOfBoundsError((10000, 10000), displays)
    msg = InputActionMessage(
        id=new_id(),
        ts=now_ts(),
        payload=InputActionPayload(action="left_click", coordinate=(10000, 10000)),
    )
    resp = await handler.handle(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.INPUT_OUT_OF_BOUNDS
    assert resp.payload.details["requested_coordinate"] == [10000, 10000]
    assert resp.payload.details["displays"] == displays


@pytest.mark.asyncio
async def test_dispatch_permission_pending_passthrough(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path))
    fb = attach_fake(handler)
    fb.input_action.side_effect = PermissionPendingError(
        "accessibility", "darwin", hint="Open Settings"
    )
    msg = InputActionMessage(
        id=new_id(),
        ts=now_ts(),
        payload=InputActionPayload(action="left_click", coordinate=(10, 10)),
    )
    resp = await handler.handle(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.PERMISSION_PENDING
    assert resp.payload.details["permission"] == "accessibility"


@pytest.mark.asyncio
async def test_dispatch_clipboard_concealed_passthrough(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path))
    fb = attach_fake(handler)
    fb.clipboard_read.side_effect = ClipboardConcealedError(
        "concealed_type", marker="org.nspasteboard.ConcealedType"
    )
    msg = ClipboardReadMessage(id=new_id(), ts=now_ts())
    resp = await handler.handle(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.CLIPBOARD_CONCEALED
    assert resp.payload.details["reason"] == "concealed_type"


@pytest.mark.asyncio
async def test_dispatch_feature_not_supported_passthrough(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path))
    fb = attach_fake(handler)
    fb.window_list.side_effect = FeatureNotSupportedError("window.list", reason="Wayland session")
    msg = WindowListRequestMessage(id=new_id(), ts=now_ts())
    resp = await handler.handle(msg)
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.FEATURE_NOT_SUPPORTED
    assert resp.payload.details["reason"] == "Wayland session"


# ── Dispatcher: happy paths for new ops ────────────────────────────────


@pytest.mark.asyncio
async def test_cursor_position_response_shape(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path))
    fb = attach_fake(handler)
    fb.cursor_position.return_value = (100, 200, 0)
    msg = CursorPositionRequestMessage(id=new_id(), ts=now_ts())
    resp = await handler.handle(msg)
    assert resp.payload.x == 100
    assert resp.payload.y == 200
    assert resp.payload.monitor == 0


@pytest.mark.asyncio
async def test_display_info_response_shape(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path))
    fb = attach_fake(handler)
    fb.display_info.return_value = [
        DisplayMonitor(
            index=0,
            primary=True,
            physical_size=(3024, 1964),
            logical_size=(1512, 982),
            scale=2.0,
            origin=(0, 0),
        )
    ]
    msg = DisplayInfoRequestMessage(id=new_id(), ts=now_ts())
    resp = await handler.handle(msg)
    assert len(resp.payload.monitors) == 1
    assert resp.payload.monitors[0].scale == 2.0


@pytest.mark.asyncio
async def test_app_launch_uses_pid_from_backend(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path))
    fb = attach_fake(handler)
    fb.app_launch.return_value = 12345
    msg = AppLaunchMessage(
        id=new_id(), ts=now_ts(), payload=AppLaunchPayload(executable_path="/bin/echo")
    )
    resp = await handler.handle(msg)
    assert resp.payload.ok is True
    fb.app_launch.assert_called_once()


@pytest.mark.asyncio
async def test_clipboard_write_round_trip(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path, enable_clipboard=True))
    fb = attach_fake(handler)
    msg = ClipboardWriteMessage(
        id=new_id(), ts=now_ts(), payload=ClipboardWritePayload(content="hello")
    )
    resp = await handler.handle(msg)
    assert resp.payload.ok is True
    fb.clipboard_write.assert_called_once_with(format="text", content="hello")


@pytest.mark.asyncio
async def test_clipboard_read_rejects_oversized_result(tmp_path: Path) -> None:
    handler = ComputerUseHandler(
        make_config(tmp_path, enable_clipboard=True, max_file_size_bytes=4)
    )
    fb = attach_fake(handler)
    fb.clipboard_read.return_value = ClipboardReadResult(
        format="text", content="hello", size_bytes=5
    )

    response = await handler.handle(ClipboardReadMessage(id=new_id(), ts=now_ts()))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.FILE_TOO_LARGE


@pytest.mark.asyncio
async def test_clipboard_write_rejects_oversized_utf8_before_backend(tmp_path: Path) -> None:
    handler = ComputerUseHandler(
        make_config(tmp_path, enable_clipboard=True, max_file_size_bytes=4)
    )
    fb = attach_fake(handler)
    message = ClipboardWriteMessage(
        id=new_id(),
        ts=now_ts(),
        payload=ClipboardWritePayload(content="ééé"),
    )

    response = await handler.handle(message)

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.FILE_TOO_LARGE
    fb.clipboard_write.assert_not_called()


@pytest.mark.asyncio
async def test_permissions_check_response(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path))
    fb = attach_fake(handler)
    fb.permissions_check.return_value = {
        "accessibility": "denied",
        "screen_recording": "granted",
    }
    msg = PermissionsCheckRequestMessage(
        id=new_id(),
        ts=now_ts(),
        payload=PermissionsCheckRequestPayload(permissions=["accessibility", "screen_recording"]),
    )
    resp = await handler.handle(msg)
    assert resp.payload.permissions["accessibility"] == "denied"


# ── Screenshot meta echo (Q1) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_screenshot_response_carries_origin_and_display_id(
    tmp_path: Path,
) -> None:
    handler = ComputerUseHandler(make_config(tmp_path))
    fb = attach_fake(handler)
    fb.screenshot.return_value = ScreenshotResult(
        image_bytes=b"img",
        mime_type="image/jpeg",
        width=800,
        height=500,
        monitor=1,
        region=(100, 200, 900, 700),
        display_scale=2.0,
        logical_size=(2560, 1440),
        origin=(100, 200),
        display_id=1,
    )
    msg = ScreenshotRequestMessage(
        id=new_id(),
        ts=now_ts(),
        payload=ScreenshotRequestPayload(region=(100, 200, 900, 700), monitor=1),
    )
    resp = await handler.handle(msg)
    assert resp.payload.meta.origin == (100, 200)
    assert resp.payload.meta.display_id == 1
    assert resp.payload.region == (100, 200, 900, 700)


@pytest.mark.asyncio
async def test_screenshot_rejects_oversized_bytes_before_base64(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path, max_file_size_bytes=3))
    fb = attach_fake(handler)
    fb.screenshot.return_value = ScreenshotResult(
        image_bytes=b"four",
        mime_type="image/png",
        width=1,
        height=1,
        monitor=0,
        region=None,
        display_scale=1.0,
        logical_size=(1, 1),
        origin=(0, 0),
        display_id=0,
    )

    response = await handler.handle(
        ScreenshotRequestMessage(
            id=new_id(),
            ts=now_ts(),
            payload=ScreenshotRequestPayload(),
        )
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.FILE_TOO_LARGE


@pytest.mark.asyncio
async def test_screenshot_rate_limit_rejects_excess_request(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path, max_screenshots_per_minute=1))
    backend = attach_fake(handler)
    backend.screenshot.return_value = ScreenshotResult(
        image_bytes=b"img",
        mime_type="image/jpeg",
        width=1,
        height=1,
        monitor=0,
        region=None,
        display_scale=1.0,
        logical_size=(1, 1),
        origin=(0, 0),
        display_id=0,
    )
    request = ScreenshotRequestMessage(
        id=new_id(),
        ts=now_ts(),
        payload=ScreenshotRequestPayload(),
    )

    first = await handler.handle(request)
    second = await handler.handle(request.model_copy(update={"id": new_id()}))

    assert first.type == "SCREENSHOT_RESPONSE"
    assert isinstance(second, ErrorMessage)
    assert second.payload.code == ErrorCode.SHIM_OVERLOADED
    backend.screenshot.assert_called_once()
