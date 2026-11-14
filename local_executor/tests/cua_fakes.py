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
    "FAKE_CUA_PERMISSIONS_SLOW_SECONDS",
    "FAKE_CUA_APPS_FILE",
    "FAKE_CUA_LINUX_LAUNCH",
    "FAKE_CUA_APPS_GONE_AT_LAUNCH",
    "FAKE_CUA_SCREEN",
    "FAKE_CUA_CLIPBOARD_TYPES",
    "FAKE_CUA_IMAGE_BYTES",
    "FAKE_CUA_EXIT_AT_START",
    "FAKE_CUA_RPC_ERROR",
    "FAKE_CUA_RPC_ERROR_CODE",
    "FAKE_CUA_NESTED_LINE",
    "FAKE_CUA_NEXT_CURSOR",
    "FAKE_CUA_SLOW_SECONDS",
    "FAKE_CUA_INITIALIZE",
    "FAKE_CUA_TOOLS_LIST",
    "FAKE_CUA_SCREEN_ERROR",
    "FAKE_CUA_SCREEN_EXIT",
    "FAKE_CUA_STALL_AT_INITIALIZE",
    "FAKE_CUA_IGNORE_TERM",
)


def fake_client(argv: list[str], **kwargs: Any) -> StdioMcpClient:
    """Run the fake with this interpreter: a .py file isn't executable on
    every OS. argv[0] is the "binary", the fake itself."""
    return StdioMcpClient([sys.executable, *argv], **kwargs)


def fake_bridge(
    *,
    clipboard: bool = False,
    platform: str = "linux",
    binary: str | Path = FAKE,
    **kwargs: Any,
) -> CuaDriverBridge:
    return CuaDriverBridge(
        tools=catalog.offered_tools(enable_clipboard=clipboard),
        binary=binary,
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
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in lines if event is None or r.get("event") == event]
