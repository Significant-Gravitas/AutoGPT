"""`monitor: -1` captures every monitor stitched together (PROTOCOL.md)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from autogpt_local_executor.computer_use.backends import linux, macos, windows
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.protocol import DisplayMonitor

VIRTUAL_DESKTOP = {"left": 0, "top": 0, "width": 3000, "height": 1080}
DISPLAYS = [
    DisplayMonitor(
        index=0,
        primary=True,
        physical_size=(1920, 1080),
        logical_size=(1920, 1080),
        scale=1.0,
        origin=(0, 0),
    ),
    DisplayMonitor(
        index=1,
        primary=False,
        physical_size=(1080, 1080),
        logical_size=(1080, 1080),
        scale=1.0,
        origin=(1920, 0),
    ),
]


class _FakeScreenCapture:
    """mss.mss(): monitors[0] is the stitched virtual desktop."""

    monitors = [
        VIRTUAL_DESKTOP,
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 1920, "top": 0, "width": 1080, "height": 1080},
    ]

    def __init__(self) -> None:
        self.grabbed: list[dict[str, int]] = []

    def __enter__(self) -> _FakeScreenCapture:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def grab(self, rect: dict[str, int]) -> SimpleNamespace:
        self.grabbed.append(dict(rect))
        return SimpleNamespace(size=(rect["width"], rect["height"]), bgra=b"")


@pytest.mark.parametrize(
    "backend_module, backend_class",
    [
        (linux, linux.LinuxBackend),
        (macos, macos.MacOSBackend),
        (windows, windows.WindowsBackend),
    ],
    ids=["linux", "macos", "windows"],
)
def test_monitor_minus_one_captures_every_monitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_module, backend_class
) -> None:
    capture = _FakeScreenCapture()
    monkeypatch.setitem(sys.modules, "mss", types.SimpleNamespace(mss=lambda: capture))
    pil = types.ModuleType("PIL")
    pil.Image = SimpleNamespace(  # type: ignore[attr-defined]
        frombytes=lambda _mode, size, *_args: SimpleNamespace(width=size[0], height=size[1])
    )
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setattr(backend_module, "image_to_base64", lambda *_a, **_k: (b"", "image/png"))
    backend = backend_class(
        ShimConfig(
            allowed_root=tmp_path,
            audit_log_path=tmp_path / "audit.log",
            enable_computer_use=True,
        )
    )
    monkeypatch.setattr(backend, "display_info", lambda: DISPLAYS)
    if backend_class is linux.LinuxBackend:
        monkeypatch.setattr(backend, "_wayland", False)

    result = backend.screenshot(monitor=-1, include_cursor=False)

    assert capture.grabbed == [VIRTUAL_DESKTOP]
    assert (result.monitor, result.display_id) == (-1, -1)
    assert result.origin == (0, 0)
    assert result.logical_size == (3000, 1080)
    assert result.display_scale == 1.0
