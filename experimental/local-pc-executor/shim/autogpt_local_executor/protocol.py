"""
Protocol constants and message builders.
See docs/PROTOCOL.md for the full spec.
"""

from __future__ import annotations

import platform
import time
import uuid
from typing import Any

import pyautogui  # noqa: used for screen resolution detection below
# If pyautogui not installed, screen_resolution advertised as None


class MessageType:
    # Handshake
    HELLO = "HELLO"
    HELLO_ACK = "HELLO_ACK"
    # Shell
    EXECUTE_COMMAND = "EXECUTE_COMMAND"
    COMMAND_RESULT = "COMMAND_RESULT"
    # Files
    FILE_READ = "FILE_READ"
    FILE_CONTENTS = "FILE_CONTENTS"
    FILE_WRITE = "FILE_WRITE"
    # Computer use
    SCREENSHOT_REQUEST = "SCREENSHOT_REQUEST"
    SCREENSHOT_RESPONSE = "SCREENSHOT_RESPONSE"
    INPUT_ACTION = "INPUT_ACTION"
    # Generic
    ACK = "ACK"
    ERROR = "ERROR"
    PING = "PING"
    PONG = "PONG"


def build_hello(config: Any) -> dict:
    """Build the HELLO handshake message from ShimConfig."""
    capabilities = []
    if config.enable_shell:
        capabilities.append("shell")
    capabilities.append("files")  # always advertised
    if config.enable_computer_use:
        capabilities.append("computer_use")
    if config.enable_local_llm:
        capabilities.append("local_llm")
    if config.enable_hardware:
        capabilities.append("hardware_serial")

    screen_resolution = None
    if config.enable_computer_use:
        try:
            import pyautogui
            screen_resolution = list(pyautogui.size())
        except Exception:
            pass

    return {
        "type": MessageType.HELLO,
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "payload": {
            "shim_version": "0.0.1-experimental",
            "machine_id": config.machine_id,
            "platform": platform.system().lower(),
            "arch": platform.machine(),
            "screen_resolution": screen_resolution,
            "capabilities": capabilities,
            "allowed_root": str(config.allowed_root),
            "local_llm_models": [],  # TODO: query ollama /api/tags
            "hardware_devices": [],  # TODO: enumerate serial/USB
        },
    }


def parse_message(raw: str) -> dict:
    """Parse a raw WebSocket message string into a message dict."""
    import json
    msg = json.loads(raw)
    if "type" not in msg or "id" not in msg:
        raise ValueError(f"Invalid message — missing type or id: {raw[:200]}")
    return msg
