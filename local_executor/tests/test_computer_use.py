"""Tests for ComputerUseHandler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import ComputerUseHandler
from autogpt_local_executor.protocol import (
    ErrorCode,
    ErrorMessage,
    ScreenshotRequestMessage,
    ScreenshotRequestPayload,
    ScreenshotResponseMessage,
    new_id,
    now_ts,
)


def make_config(tmp_path: Path, enable: bool = True) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        enable_computer_use=enable,
        platform_url="http://localhost:9999",
        machine_id="test-machine",
    )


def _screenshot_msg(**kw) -> ScreenshotRequestMessage:
    return ScreenshotRequestMessage(
        id=new_id(),
        ts=now_ts(),
        payload=ScreenshotRequestPayload(**kw),
    )


@pytest.mark.asyncio
async def test_screenshot_disabled(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path, enable=False))
    resp = await handler.handle(_screenshot_msg())
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.CAPABILITY_NOT_GRANTED


@pytest.mark.asyncio
async def test_screenshot_no_backend_deps(tmp_path: Path, monkeypatch) -> None:
    """When mss/Pillow aren't available the backend raises
    FeatureNotSupportedError → wire FEATURE_NOT_SUPPORTED."""
    from autogpt_local_executor.computer_use import errors as cu_errors

    handler = ComputerUseHandler(make_config(tmp_path, enable=True))

    def _raise(*_, **__):
        raise cu_errors.FeatureNotSupportedError("screenshot", reason="mss missing")

    monkeypatch.setattr(handler.backend, "screenshot", _raise)
    # Also silence the legacy back-compat shim that honors a monkeypatched _pyautogui.
    import autogpt_local_executor.handlers as h_mod

    monkeypatch.setattr(h_mod, "_pyautogui", None)
    resp = await handler.handle(_screenshot_msg())
    assert isinstance(resp, ErrorMessage)
    assert resp.payload.code == ErrorCode.FEATURE_NOT_SUPPORTED


@pytest.mark.asyncio
async def test_screenshot_returns_image_base64(tmp_path: Path) -> None:
    handler = ComputerUseHandler(make_config(tmp_path, enable=True))
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("Pillow not available")

    fake_img = Image.new("RGB", (100, 50), color=(255, 0, 0))
    mock_pg = MagicMock()
    mock_pg.screenshot.return_value = fake_img

    import autogpt_local_executor.handlers as h_mod

    orig_pg, orig_img = h_mod._pyautogui, h_mod._Image
    h_mod._pyautogui = mock_pg
    h_mod._Image = Image
    try:
        resp = await handler.handle(_screenshot_msg(quality=75))
    finally:
        h_mod._pyautogui = orig_pg
        h_mod._Image = orig_img

    assert isinstance(resp, ScreenshotResponseMessage)
    assert resp.payload.mime_type == "image/jpeg"
    assert len(resp.payload.image_base64) > 0
    assert resp.payload.width == 100
    assert resp.payload.height == 50
