"""
Capability handlers — one class per capability type.

Each handler receives a parsed protocol message dict and returns a response dict
ready to be JSON-serialized and sent back over the WebSocket.

All handlers enforce their own security checks (path jail, rate limits, etc.)
before touching the local system.
"""

from __future__ import annotations

import asyncio
import base64
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import ShimConfig
from .protocol import MessageType


class FileHandler:
    """
    Handles FILE_READ and FILE_WRITE messages.

    Security:
        - All paths are resolved and checked against config.allowed_root
        - Symlinks are resolved before the check (no symlink escape)
        - File size limited to config.max_file_size_bytes
        - Binary files returned as base64; text as UTF-8

    Protocol:
        In:  FILE_READ  { path, encoding, offset, length }
             FILE_WRITE { path, content, encoding, create_parents }
        Out: FILE_CONTENTS { content, encoding, size_bytes, truncated }
             ACK            { ok }
             ERROR          { code, message }
    """

    def __init__(self, config: ShimConfig) -> None:
        self.config = config

    def _jail(self, path_str: str) -> Path:
        """
        Resolve path and verify it's inside allowed_root.
        Raises ValueError if the path escapes the jail.
        """
        allowed = self.config.allowed_root.resolve()
        target = Path(path_str).resolve()
        if not str(target).startswith(str(allowed)):
            raise ValueError(
                f"PATH_OUTSIDE_ALLOWED_ROOT: {target} is outside {allowed}"
            )
        return target

    async def handle(self, msg: dict) -> dict:
        msg_type = msg["type"]
        if msg_type == MessageType.FILE_READ:
            return await self._handle_read(msg)
        elif msg_type == MessageType.FILE_WRITE:
            return await self._handle_write(msg)
        raise ValueError(f"FileHandler got unexpected type: {msg_type}")

    async def _handle_read(self, msg: dict) -> dict:
        payload = msg["payload"]
        try:
            path = self._jail(payload["path"])
        except ValueError as e:
            return _error(msg["id"], "PATH_OUTSIDE_ALLOWED_ROOT", str(e))

        if not path.exists():
            return _error(msg["id"], "FILE_NOT_FOUND", f"{path} does not exist")

        size = path.stat().st_size
        if size > self.config.max_file_size_bytes:
            return _error(
                msg["id"], "FILE_TOO_LARGE",
                f"{size} bytes exceeds limit {self.config.max_file_size_bytes}"
            )

        encoding = payload.get("encoding", "utf-8")
        offset = payload.get("offset", 0)
        length = payload.get("length")  # None = whole file

        raw = path.read_bytes()
        chunk = raw[offset : offset + length] if length else raw[offset:]
        truncated = length is not None and len(raw) - offset > length

        if encoding == "base64":
            content = base64.b64encode(chunk).decode("ascii")
        else:
            content = chunk.decode("utf-8", errors="replace")

        return {
            "type": MessageType.FILE_CONTENTS,
            "id": msg["id"],
            "ts": time.time(),
            "payload": {
                "content": content,
                "encoding": encoding,
                "size_bytes": len(chunk),
                "truncated": truncated,
            },
        }

    async def _handle_write(self, msg: dict) -> dict:
        payload = msg["payload"]
        try:
            path = self._jail(payload["path"])
        except ValueError as e:
            return _error(msg["id"], "PATH_OUTSIDE_ALLOWED_ROOT", str(e))

        if payload.get("create_parents", False):
            path.parent.mkdir(parents=True, exist_ok=True)

        encoding = payload.get("encoding", "utf-8")
        content = payload["content"]

        if encoding == "base64":
            raw = base64.b64decode(content)
        else:
            raw = content.encode("utf-8")

        if len(raw) > self.config.max_file_size_bytes:
            return _error(msg["id"], "FILE_TOO_LARGE", f"Content exceeds size limit")

        path.write_bytes(raw)
        return _ack(msg["id"])


class CommandHandler:
    """
    Handles EXECUTE_COMMAND messages.

    Security:
        - cwd is jailed to allowed_root (same logic as FileHandler)
        - Timeout enforced via asyncio.wait_for
        - Environment is sanitized: only whitelisted vars passed through,
          plus any vars specified in the message payload
        - stdout/stderr captured and returned; never streamed to avoid partial results

    Protocol:
        In:  EXECUTE_COMMAND { command, cwd, timeout_seconds, env }
        Out: COMMAND_RESULT  { stdout, stderr, exit_code, timed_out, duration_seconds }
             ERROR           { code, message }

    Future:
        - Allow/deny list for command patterns (e.g. block `rm -rf /`)
        - Optional bwrap sandboxing wrapper (Linux only)
        - Streaming stdout/stderr via COMMAND_OUTPUT_CHUNK messages
    """

    SAFE_ENV_VARS = {"PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TMPDIR"}

    def __init__(self, config: ShimConfig) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_commands)

    async def handle(self, msg: dict) -> dict:
        payload = msg["payload"]
        command = payload["command"]
        cwd_str = payload.get("cwd") or str(self.config.allowed_root)
        timeout = payload.get("timeout_seconds", self.config.command_timeout_seconds)
        extra_env = payload.get("env", {})

        # Jail the working directory
        try:
            cwd = _jail_path(cwd_str, self.config.allowed_root)
        except ValueError as e:
            return _error(msg["id"], "PATH_OUTSIDE_ALLOWED_ROOT", str(e))

        import os
        safe_env = {k: v for k, v in os.environ.items() if k in self.SAFE_ENV_VARS}
        safe_env.update(extra_env)

        async with self._semaphore:
            start = time.monotonic()
            timed_out = False
            try:
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_shell(
                        command,
                        cwd=str(cwd),
                        env=safe_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    ),
                    timeout=timeout,
                )
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                exit_code = proc.returncode
            except asyncio.TimeoutError:
                timed_out = True
                stdout_b, stderr_b, exit_code = b"", b"", -1
                try:
                    proc.kill()
                except Exception:
                    pass
            duration = time.monotonic() - start

        return {
            "type": MessageType.COMMAND_RESULT,
            "id": msg["id"],
            "ts": time.time(),
            "payload": {
                "stdout": stdout_b.decode("utf-8", errors="replace"),
                "stderr": stderr_b.decode("utf-8", errors="replace"),
                "exit_code": exit_code,
                "timed_out": timed_out,
                "duration_seconds": round(duration, 3),
            },
        }


class ComputerUseHandler:
    """
    Handles SCREENSHOT_REQUEST and INPUT_ACTION messages.

    Requires pyautogui (cross-platform) or xdotool (Linux X11).

    How it fits into Claude's computer use loop:
        1. Platform receives Claude's `computer_20251124` tool call requesting `screenshot`
        2. Platform sends SCREENSHOT_REQUEST to shim
        3. Shim captures screen → JPEG → base64 → SCREENSHOT_RESPONSE
        4. Platform passes image back to Claude as tool_result with image content
        5. Claude analyzes pixel coordinates and issues an action
        6. Platform sends INPUT_ACTION to shim
        7. Shim executes via pyautogui → INPUT_ACK
        8. Repeat until Claude stops issuing tool calls

    Security:
        - Only active when config.enable_computer_use = True
        - Platform must have granted `local_executor:computer_use` scope
        - TODO: add per-session confirmation prompt ("Claude is requesting screen access")

    Supported actions:
        screenshot, left_click, right_click, double_click,
        mouse_move, type, key, scroll

    Future:
        - Multi-monitor support (monitor index in payload)
        - Window-targeted screenshots (by window title / PID)
        - Sensitive region masking (redact areas before sending to Claude)
        - Record/replay for testing

    Dependencies:
        pip install pyautogui Pillow
        # Linux also needs: python3-tk python3-dev scrot
    """

    def __init__(self, config: ShimConfig) -> None:
        self.config = config

    async def handle(self, msg: dict) -> dict:
        if not self.config.enable_computer_use:
            return _error(
                msg["id"],
                "CAPABILITY_NOT_GRANTED",
                "Computer use is not enabled on this shim. "
                "Set enable_computer_use=true in shim config.",
            )

        msg_type = msg["type"]
        if msg_type == MessageType.SCREENSHOT_REQUEST:
            return await self._handle_screenshot(msg)
        elif msg_type == MessageType.INPUT_ACTION:
            return await self._handle_input(msg)
        raise ValueError(f"ComputerUseHandler got unexpected type: {msg_type}")

    async def _handle_screenshot(self, msg: dict) -> dict:
        """
        Capture the screen and return as base64-encoded JPEG.

        Uses pyautogui.screenshot() → PIL Image → JPEG bytes → base64.
        Runs in a thread pool to avoid blocking the event loop.
        """
        payload = msg["payload"]
        quality = payload.get("quality", 75)

        # TODO: implement using pyautogui
        # import pyautogui
        # from PIL import Image
        # import io
        #
        # def _capture():
        #     img = pyautogui.screenshot()
        #     buf = io.BytesIO()
        #     img.convert("RGB").save(buf, format="JPEG", quality=quality)
        #     return buf.getvalue(), img.width, img.height
        #
        # loop = asyncio.get_event_loop()
        # img_bytes, w, h = await loop.run_in_executor(None, _capture)
        # img_b64 = base64.b64encode(img_bytes).decode("ascii")

        raise NotImplementedError(
            "Screenshot capture not yet implemented. "
            "Install pyautogui and Pillow, then implement _capture()."
        )

    async def _handle_input(self, msg: dict) -> dict:
        """
        Execute a mouse/keyboard action via pyautogui.

        Action types: left_click, right_click, double_click,
                      mouse_move, type, key, scroll
        """
        payload = msg["payload"]
        action = payload["action"]
        coordinate = payload.get("coordinate")  # [x, y]
        text = payload.get("text")
        key = payload.get("key")

        # TODO: implement using pyautogui
        # import pyautogui
        # pyautogui.FAILSAFE = True  # move mouse to corner to abort
        #
        # if action == "left_click" and coordinate:
        #     pyautogui.click(coordinate[0], coordinate[1])
        # elif action == "right_click" and coordinate:
        #     pyautogui.rightClick(coordinate[0], coordinate[1])
        # elif action == "double_click" and coordinate:
        #     pyautogui.doubleClick(coordinate[0], coordinate[1])
        # elif action == "mouse_move" and coordinate:
        #     pyautogui.moveTo(coordinate[0], coordinate[1])
        # elif action == "type" and text:
        #     pyautogui.write(text, interval=0.02)
        # elif action == "key" and key:
        #     pyautogui.hotkey(*key.split("+"))
        # elif action == "scroll" and coordinate:
        #     direction = payload.get("direction", "down")
        #     clicks = payload.get("clicks", 3)
        #     pyautogui.scroll(clicks if direction == "up" else -clicks,
        #                      x=coordinate[0], y=coordinate[1])

        raise NotImplementedError(
            "Input injection not yet implemented. "
            "Install pyautogui, then implement the action dispatcher."
        )

        return _ack(msg["id"])


# --- helpers ---

def _jail_path(path_str: str, allowed_root: Path) -> Path:
    allowed = allowed_root.resolve()
    target = Path(path_str).resolve()
    if not str(target).startswith(str(allowed)):
        raise ValueError(f"{target} is outside {allowed}")
    return target


def _ack(msg_id: str) -> dict:
    return {
        "type": MessageType.ACK,
        "id": msg_id,
        "ts": time.time(),
        "payload": {"ok": True},
    }


def _error(msg_id: str, code: str, message: str, fatal: bool = False) -> dict:
    return {
        "type": MessageType.ERROR,
        "id": msg_id,
        "ts": time.time(),
        "payload": {"code": code, "message": message, "fatal": fatal},
    }
