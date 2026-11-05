"""Helpers for tests that run the fake cua-driver (fake_cua_driver.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from autogpt_local_executor.computer_use import catalog
from autogpt_local_executor.computer_use.driver import CuaDriverBridge
from autogpt_local_executor.computer_use.mcp_client import StdioMcpClient

FAKE = Path(__file__).with_name("fake_cua_driver.py")

_FAKE_SETTINGS = (
    "FAKE_CUA_VERSION",
    "FAKE_CUA_EXTRA_TOOLS",
    "FAKE_CUA_PERMISSIONS",
    "FAKE_CUA_SCREEN",
    "FAKE_CUA_CLIPBOARD_TYPES",
    "FAKE_CUA_IMAGE_BYTES",
    "FAKE_CUA_EXIT_AT_START",
    "FAKE_CUA_RPC_ERROR",
)


def fake_client(argv: list[str], **kwargs: Any) -> StdioMcpClient:
    """Run the fake with this interpreter: a .py file isn't executable on
    every OS. argv[0] is the "binary", the fake itself."""
    return StdioMcpClient([sys.executable, *argv], **kwargs)


def fake_bridge(
    *, clipboard: bool = False, platform: str = "linux", **kwargs: Any
) -> CuaDriverBridge:
    return CuaDriverBridge(
        tools=catalog.offered_tools(enable_clipboard=clipboard),
        binary=FAKE,
        platform=platform,
        client_factory=fake_client,
        **kwargs,
    )


@pytest.fixture
def fake_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Where the fake driver logs what it saw; its settings start clean."""
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_CUA_LOG", str(log))
    monkeypatch.setenv("FAKE_CUA_CLIPBOARD", str(tmp_path / "clipboard.txt"))
    for name in _FAKE_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    return log


def records(log: Path, event: str | None = None) -> list[dict]:
    if not log.exists():
        return []
    lines = [json.loads(line) for line in log.read_text().splitlines() if line]
    return [r for r in lines if event is None or r.get("event") == event]
